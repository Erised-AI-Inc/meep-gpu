#!/usr/bin/env python3
"""Native MPS byte gate for the FOLDED COMPLEX fused magnetic kernel: ``step_B`` -> ``update_H``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans FOUR driver
passes — ``step_B``, ``fill_symmetry_bc_B``, ``zero_metal_B`` and ``update_H`` — so a
per-sub-step comparison could not see it at all: the whole claim is about the SEAM
between them, and about a mirror fill carried INSIDE one dispatch whose source cell a
different thread computes, under COMPLEX storage where the parity is a complex product
rather than a sign. The comparison is per COMPLETE DRIVER STEP over a stated budget,
against an array-path oracle running the identical live pass list, with the first
divergent step reported rather than a final pass/fail.

THE TEN THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that was
   never launched is byte-identical to the oracle BY CONSTRUCTION, because the oracle
   is the array path this walk would otherwise take. So every case asserts the exact
   LAUNCH COUNT (``launches_per_run`` x steps) and a nonzero ``runs``, and every case
   asserts every compared array MOVED from its seeded value.
2. **A hollow pass.** Seventeen shader mutations and TWO host-binding mutations are
   ARMED. Each is scored CAUGHT or NULL CONFIRMED — never merely "permitted" — and
   leg ``disarm`` reruns the identical case with the shipped bytes and no host patch
   and requires zero, so a mutation reported as caught cannot be a harness that
   diverges anyway.
3. **A DEAD-BRANCH MUTATION.** A needle can rewrite a real line the scored case never
   reaches, because the line sits under a guard that case does not enter; it then
   reports UNCAUGHT while measuring nothing. Every mutation here declares the CASE it
   is armed on, and ``main`` asserts that case's specialisation carries the feature
   the needle depends on — a fold on the right axis, both parities, a live wall, a
   phased axis — before a single edit is applied. The two mutation cases exist because
   no single folded-complex grid can carry both a fold on every axis and a phased one:
   a folded axis may not be phased (``folded_complex._reachable_phase_flags``).
4. **A LINE NOTHING CAN SEE.** This family's one novel line is the imaged ghost's
   coefficient pair, read at stored index 0 rather than reused from the source thread
   at index 2. On a FOLDED axis the mirror plane carries no absorber
   (``stepping._require_consistent_pml``: high face only), so at the shared fixture's
   2-cell layer ``kps[0] == kps[2]`` EXACTLY and the mutation that reuses the wrong
   entry is UNOBSERVABLE. Leg ``moved_coefficient`` censuses the coefficients,
   PREDICTS observability from them, runs the mutation on a case where the prediction
   is False and one where it is True, and requires ``caught == observable`` on both.
5. **A construction claimed rather than measured.** All THREE pieces of this kernel
   are LIFTED from certified emitters' own output rather than retyped. Leg
   ``transcription`` re-runs those emitters and requires (a) the lifted curl head to
   be a verbatim prefix of ``folded_complex.folded_bloch_curl_source``'s body, (b)
   this family's parameterised constitutive statements to reproduce
   ``complex_fields.bloch_constitutive_source('H')`` statement for statement, and (c)
   the ghost write to be ``folded_complex.folded_mirror_fill_complex_source``'s own
   line under the stated rename.
6. **An unmeasured platform assumption.** This family is ONE dispatch for all three
   components because the eleven non-pointer arguments are packed into a
   ``constant Params&``. Leg ``binding_ceiling`` compiles the 35-binding
   separate-scalar signature and requires the FAILURE, then compiles AND LAUNCHES the
   packed one with THIS FAMILY'S OWN SIX-float2 record and requires every struct field
   to read back correctly — the padding question the wider record raises is answered
   by launching it, not by arithmetic.
7. **A refusal that is really an omission — AND A CARRY THAT IS REALLY A REFUSAL.**
   Leg ``refusal`` builds a folded PERIODIC grid, where ``fill_folded_far_ghosts_B`` is
   live inside the seam, and requires the composer to agree the pass is LIVE, the
   predicate to COVER it, ``REPLACES`` to NAME it and the plan to report the Yee
   table's far axes and ``stepping._far_reflect_rows``' own rows — the FLIP of what
   this leg asserted before the far carry. Beside it, a run with a declared MAGNETIC
   source, an UNFOLDED complex grid, a REAL folded grid and three stub grids whose
   reflect row is illegal must each be refused BY NAME.
8. **A vacuous subnormal precondition.** Every case asserts the reference stayed
   subnormal-free, which is only a precondition if the census can fire. Leg
   ``value_classes`` runs a SUBNORMAL_BAND seed and requires the census to be NONZERO,
   plus a uniform seed and a +-0 lattice that must both stay bit-identical.
9. **A COLLAPSE INHERITED RATHER THAN MEASURED.** The real board's far carry folds a
   composed parity into ONE compile-time sign and states the result is order-
   independent. Under complex storage the parity is a ``float2`` and the application is
   a full complex multiply, so leg ``parity_chain_order`` re-establishes the question
   over an exhaustive signed-zero/subnormal table: re-ordering a two-parity chain moves
   8 of 128 uint32 words and folding it into one word moves up to 23 of 128, at EVERY
   sign combination. Two source mutations arm exactly those two collapses.
10. **A policy claimed rather than resolved.** Leg ``policy`` records
   ``subnormal.mps_policy_report`` and MEASURES the other policy: a requested ``keep``
   must be refused BY NAME on this executor. Metal has ONE attainable float32
   subnormal policy, not two, and that is a measurement this gate carries rather than
   a gap it leaves.

LEGS
  0  policy            the resolved policy, and ``keep`` refused by name
  1  expansion         the probe binds the arm; missing / ambiguous / wrong-backend
                       probes must each REFUSE
  2  binding_ceiling   35 separate bindings must FAIL; 27 pointers + one packed
                       Params& must COMPILE, LAUNCH and read back every field — now
                       NINE float2 members and three reflect rows, 104 bytes, and
                       still 28 bindings
  3  transcription     all four pieces are the certified emitters' own text (the FAR
                       fill line joined the near one with the far carry)
  4  parity_chain_order  the composed parity is a CHAIN; its ORDER moves bytes and so
                       does folding it into one word, so neither collapse the real
                       board takes is available here
  5  product           complete steps, per-step byte compare, launch counters,
                       movement, subnormal census, signed-zero seeding floor
  6  separate_control  the ALREADY CERTIFIED folded complex products stepping the
                       same seam as separate dispatches with the fills and the wall
                       clear on the host between them: three-way byte agreement plus
                       the dispatch and host-pass counts the fusion removes
  7  refusal           THE FLIP (a folded PERIODIC axis is now CARRIED, not refused)
                       plus the boundaries that remain: sources, the two sibling
                       scopes, and three stub grids whose reflect row is illegal
  8  value_classes     uniform and +-0 identical; the band seed FIRES the census
  9  moved_coefficient predicted observability vs measured catch, on two cases
 10  mutation          thirty armed defects, each CAUGHT or NULL CONFIRMED
 11  disarm            the same harness, shipped bytes, must not diverge
 12  planted_defect    the RELEASE VERDICT itself must FLIP against a planted defect

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_fields, folded_complex,
    folded_complex_fused_magnetic_pair as family, launch as metal_launch, shaders,
    subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: E402
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402
from meep_gpu.metal_kernels.symmetry import MIRROR_CODES  # noqa: E402
from meep_gpu.triton_kernels.symmetry import (  # noqa: E402
    CODE_MIRROR_PERIODIC, CODE_PERIODIC, folded_axis_kinds,
)

#: The budget every case runs. Twelve, matching every other fused pair's and the
#: whole-step arbiter's, and for their reason: the classes this gate exists for
#: COMPOUND. ``fu_B`` and ``f_w_H`` are state carried between steps, and a ghost plane
#: imaged one row over is a defect that needs several steps to reach the low bits of
#: the interior. The comparison is per COMPLETE STEP.
STEPS = 12

#: THE BASE SEED. Every case's actual seed is a DIGEST of the case's identity added to
#: this, never ``hash()``: Python salts ``hash()`` of a string with PYTHONHASHSEED, so
#: a hash-seeded gate draws a different fixture every process and a failing case cannot
#: be replayed.
SEED = 5_100_000


def seed_for(*parts: str) -> int:
    """A stable per-case seed from a sha256 digest of the case's identity."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big")


#: Which array-path function each live pass is. The two FILL passes are in this table
#: and are not in the unfolded complex pair's, which is the whole difference between an
#: unfolded seam and a folded one.
ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
    "update_P": lambda f, p: stepping.update_P(f, p),
}

#: Every stored volume a folded complete step can touch. The ``fu_*`` PML auxiliaries
#: and the ``f_w_*`` constitutive workspaces are STATE: a kernel right for one launch
#: and wrong forever after diverges only once they accumulate. The D/E half is in this
#: list even though this family does not touch it, because a fused pair that corrupted
#: B would reach E through ``step_D`` on the very next step.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The value classes leg's seeds, and what each is for.
VALUE_CLASSES: Tuple[str, ...] = ("gaussian", "uniform", "signed_zero_lattice",
                                  "subnormal_band")

#: name -> (``metal_composition_matrix.folded`` keywords, explicit PML faces or None).
#:
#: EVERY ROW IS complex_storage=True. THE MIRROR_PERIODIC ROWS ARE NEW AS OF THE FAR
#: CARRY: until it landed, a folded PERIODIC axis was refused BY NAME because
#: ``fill_folded_far_ghosts_B`` (driver.py:3287) sits inside this seam, and leg
#: ``refusal`` was where the refusal was measured. The kernel now carries that pass,
#: so those rows are WALKED here and the refusal leg keeps only the boundaries that
#: are still boundaries. What the rows carry between them:
#:
#:   PARITY         even and odd, and MIXED on a two-axis fold. The parity is a RUNTIME
#:                  complex word on this family (unlike the real fold, where it is a
#:                  source specialisation), so two rows differing only in parity
#:                  compile to the SAME kernel and differ only in the Params record —
#:                  which is exactly why both must be walked;
#:   WALL           ``zero_metal_B`` only emits a line when some axis is metallic and
#:                  NOT mirrored, so the rows without a ``z`` wall would report the
#:                  wall mutations as structurally absent;
#:   FOLD TERMINATION MIRROR_METALLIC (the top plane is owned and stepped, no far
#:                  ghost) and MIRROR_PERIODIC (the stored array carries the not-owned
#:                  slot past ``big_corner`` and the far fill images it). A mutation
#:                  caught on one is not caught on the other, so both are carried and
#:                  the two mutation cases below are one of each;
#:   PARITY CHAINS  on the NEAR half a component is a ghost destination on its OWN axis
#:                  alone, so there is no ordering question there. The FAR half is that
#:                  set's exact complement — up to TWO far planes per component — so a
#:                  MIRROR_PERIODIC row with two folded axes carries a two-parity chain
#:                  and ``xyz_periodic_odd_3d`` carries a three-parity one, where ONE
#:                  source thread owns SEVEN ghost cells for one component. Those are
#:                  the rows leg ``parity_chain_order`` and the chain mutations exist
#:                  for, because under complex storage the chain's ORDER and its
#:                  GROUPING both move bytes;
#:   FULL-COUNT PARITY ``extent`` decides whether ``_far_reflect_rows``' row is
#:                  ``stored - 2`` (even count) or ``stored - 3`` (odd). At the default
#:                  2.0 the WRONG fixed ``n - 2`` happens to be right, so every
#:                  periodic row is carried at BOTH parities;
#:   BLOCH PHASE    ``k = 0`` is the REDUCTION (no rotation is emitted at all);
#:                  ``k = 0.3`` is a generic interior point where both planes are live;
#:                  ``k = 0.5`` is the BRILLOUIN EDGE where the phase is exactly
#:                  ``-1+0j`` and a plane-wise collapse would be invisible on random
#:                  data. A FOLDED axis may never be phased, so every phased row phases
#:                  a plain periodic axis beside the fold;
#:   DIMENSIONALITY the 2-D row is the corpus's most common folded shape, and the 3-D
#:                  rows are where a ghost plane is 132 words rather than 11;
#:   ABSORBER DEPTH ``xy_deep_pml_3d`` is the ONLY row on which this family's moved
#:                  NEAR coefficient index is observable at all (leg
#:                  ``moved_coefficient`` measures exactly that), because a folded axis
#:                  carries no absorber at the mirror plane and the shared matrix's
#:                  2-cell layer never reaches stored cell 2. Its PML is built here
#:                  rather than by the matrix for that one reason.
CASES: Tuple[Tuple[str, Dict[str, Any], Optional[Any]], ...] = (
    ("y_metallic_even_3d",
     dict(complex_storage=True, boundaries={"y": "metallic"}, depth=1.2), None),
    ("y_metallic_odd_3d",
     dict(complex_storage=True, phase=-1, boundaries={"y": "metallic"}, depth=1.2),
     None),
    ("y_metallic_even_2d",
     dict(complex_storage=True, boundaries={"y": "metallic"}), None),
    ("y_metallic_wall_z_3d",
     dict(complex_storage=True, boundaries={"y": "metallic", "z": "metallic"},
          depth=1.2), None),
    ("xy_mixed_wall_z_3d",
     dict(complex_storage=True, axis="XY", phase=(1, -1),
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=1.2), None),
    ("xy_mixed_odd_3d",
     dict(complex_storage=True, axis="XY", phase=(-1, 1),
          boundaries={"x": "metallic", "y": "metallic"}, depth=1.2), None),
    ("xyz_all_folded_3d",
     dict(complex_storage=True, axis="XYZ", phase=(1, -1, 1),
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=2.0), None),
    ("y_metallic_kxz_3d",
     dict(complex_storage=True, boundaries={"y": "metallic"}, depth=1.2,
          k_point=(0.3, 0.0, 0.17)), None),
    ("y_metallic_edge_kx_3d",
     dict(complex_storage=True, boundaries={"y": "metallic"}, depth=1.2,
          k_point=(0.5, 0.0, 0.0)), None),
    ("y_metallic_odd_kx_3d",
     dict(complex_storage=True, phase=-1, boundaries={"y": "metallic"}, depth=1.2,
          k_point=(0.3, 0.0, 0.0)), None),
    ("xy_deep_pml_3d",
     dict(complex_storage=True, axis="XY", phase=(1, -1), extent=0.8,
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=1.2), ((0, 4), (0, 4), (2, 2))),
    # THE TWO ROWS THIS PRODUCT ACTUALLY SERVES BOTH CARRY AN OFF-DIAGONAL chi1inv
    # ROW, and until these two cases landed the gate had never built one. That is
    # not a cosmetic gap: `examples:solve-cw.py` and
    # `tests:TestArrayMetadata.test_array_metadata` are the entire corpus admission
    # measured in
    # `results/fusion_matrix_metal_2026-08-20_foldedcomplex/fusion_matrix.json`, so a
    # gate that certified only diagonal fixtures would have certified a shape the
    # product is never asked for.
    #
    # WHAT AN OFF-DIAGONAL ROW DOES AND DOES NOT REACH, and this is the property the
    # two cases MEASURE rather than assume: an off-diagonal chi1inv is an ELECTRIC
    # material, read by `update_E` through a transverse Yee average
    # (stepping.py:1219-1254). `update_H` reads B and mu with the integer coefficient
    # pair and NEVER an inverse epsilon — which is exactly why this family binds 27
    # pointers rather than 30, having dropped the certified complex constitutive's
    # three inverse-epsilon volumes. If that were wrong, these are the rows on which
    # it would show, and the walk runs the whole step including `update_E` on the
    # array path.
    #
    # THEY ARE ALSO WHERE THE TWO BACKENDS' PREDICATES DISAGREE. The TRITON census
    # routes these same corpus rows to a `folded complex off-diagonal` arm on BOTH
    # step_B and update_H (launch.py:2684-2695) and its plain folded-complex
    # constitutive REFUSES them by name; Metal has no off-diagonal magnetic arm and
    # admits. Metal's admission is the one being certified here, on the fixture that
    # would break it.
    ("y_metallic_offdiag_3d",
     dict(complex_storage=True, boundaries={"y": "metallic"}, depth=1.2,
          rows={"Ex": ("Ey",), "Ey": ("Ex",)}), None),
    ("xy_mixed_offdiag_wall_z_3d",
     dict(complex_storage=True, axis="XY", phase=(1, -1),
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
          depth=1.2, rows={"Ex": ("Ez",), "Ez": ("Ex",)}), None),
    # --- the folded PERIODIC rows: fill_folded_far_ghosts_B is LIVE on every one,
    # and until the far carry landed every one of them was refused by name.
    ("y_periodic_even_2d", dict(complex_storage=True), None),
    ("y_periodic_even_3d", dict(complex_storage=True, depth=1.2), None),
    ("y_periodic_odd_3d",
     dict(complex_storage=True, extent=2.1, depth=1.2), None),
    ("y_periodic_odd_parity_2d",
     dict(complex_storage=True, phase=-1, extent=2.1), None),
    ("y_periodic_wall_z_3d",
     dict(complex_storage=True, boundaries={"z": "metallic"}, depth=1.2), None),
    # TWO folded periodic axes: a component is a far destination on BOTH of the axes
    # that are not its own, so the corner where both fire carries a TWO-PARITY CHAIN
    # and a cell that is a near destination on one and a far one on another carries
    # the near parity INSIDE the far one. This is the first row where the chain's
    # order is observable at all.
    ("xy_periodic_mixed_wall_z_3d",
     dict(complex_storage=True, axis="XY", phase=(1, -1),
          boundaries={"z": "metallic"}, depth=1.2), None),
    ("xy_periodic_mixed_odd_wall_z_3d",
     dict(complex_storage=True, axis="XY", phase=(1, -1), extent=2.1,
          boundaries={"z": "metallic"}, depth=1.2), None),
    # A fold of EACH termination on the same grid: x images a far ghost and y does
    # not, so the two code paths are exercised against each other in one kernel.
    ("x_periodic_y_metallic_3d",
     dict(complex_storage=True, axis="XY", phase=(-1, 1),
          boundaries={"y": "metallic"}, depth=1.2), None),
    # THREE folded periodic axes: ONE source thread owns SEVEN ghost cells for one
    # component (two single far planes, three pairs, one triple), which is the
    # largest composition carried_destinations can produce on this family and the
    # only row carrying a THREE-parity chain.
    ("xyz_periodic_odd_3d",
     dict(complex_storage=True, axis="XYZ", phase=(-1, 1, -1), extent=2.1,
          depth=2.1), None),
    # A folded PERIODIC row with the off-diagonal chi1inv the two SERVED corpus rows
    # carry. Neither served row folds periodically, but the two features are
    # independent and a product that admitted their conjunction untested would be
    # admitting a shape nothing had built.
    ("y_periodic_offdiag_3d",
     dict(complex_storage=True, depth=1.2,
          rows={"Ex": ("Ey",), "Ey": ("Ex",)}), None),
    # The periodic twin of xy_deep_pml_3d: a 4-cell high-face layer on a 6-cell
    # folded axis is the first that reaches stored cell 2, which is what makes the
    # COMPOSITE near/far ghost's coefficient index observable.
    ("xy_periodic_deep_pml_3d",
     dict(complex_storage=True, axis="XY", phase=(1, -1), extent=0.8,
          boundaries={"z": "metallic"}, depth=1.2), ((0, 4), (0, 4), (2, 2))),
)

#: The case the FOLD, GHOST, WALL and CONSTITUTIVE mutations are armed on. Two folded
#: axes with MIXED parities and a live ``z`` wall, so every line those needles reach is
#: present: two ghost carries with different parity words, the wall clear on the owned
#: cell, and the full three-row diagonal ownership mask. It carries NO phased axis, and
#: it cannot: a folded axis may not be phased and ``z`` here is walled.
MUTATION_CASE = "xy_mixed_wall_z_3d"

#: The case the BLOCH PHASE mutations are armed on. One folded axis (``y``) and TWO
#: phased periodic axes (``x`` and ``z``), so both rotation blocks are emitted and the
#: fold's ghost carry is still present. A phase mutation armed on
#: :data:`MUTATION_CASE` would rewrite nothing — that is the dead-branch trap, and the
#: split is what avoids it.
PHASE_MUTATION_CASE = "y_metallic_kxz_3d"

#: The case the FAR CARRY and PARITY CHAIN mutations are armed on. TWO folded PERIODIC
#: axes with MIXED parities, an ODD full count and a live ``z`` wall, so every line the
#: far carry can emit is present at once: a near carry, two far carries with different
#: parity words, a near/far composite chain, a FAR/FAR composite chain (component 2,
#: whose own axis is unfolded), the top-plane curl mask, a reflect row that is NOT
#: ``stored - 2``, and the wall clear on the owned cell. :data:`MUTATION_CASE` folds
#: MIRROR_METALLIC and emits NONE of them, which is exactly why the split exists.
FAR_MUTATION_CASE = "xy_periodic_mixed_odd_wall_z_3d"

#: The axis ``far_parity_dropped``'s needle rewrites, and the axis
#: ``reflect_row_is_the_fixed_n_minus_two``'s does. Named as data because the two
#: preconditions above must ask about the SAME axis the needle edits: both mutations
#: scored 12 launches at 0 differing words while a precondition quantified over all
#: three (``results/metal_folded_complex_far_carry_2026-08-21/``, legs 57 and 59).
PARITY_DROP_AXIS = 0
REFLECT_MUTATION_AXIS = 1

#: The case leg ``moved_coefficient`` uses for its OBSERVABLE half. Same fold, same
#: parities, same wall — only the absorber reaches stored cell 2.
DEEP_CASE = "xy_deep_pml_3d"

#: The row leg ``refusal`` builds for the far fill: a folded PERIODIC axis, where
#: ``fill_folded_far_ghosts_B`` is live inside the seam. It was the REFUSAL fixture
#: until the far carry landed and is now the fixture that measures the FLIP; the name
#: is kept so a reader diffing this gate against its predecessor can see that the same
#: grid changed verdict rather than that a different grid appeared.
REFUSAL_FAR_CASE: Dict[str, Any] = dict(complex_storage=True, depth=1.2)

#: Bound once by leg ``expansion`` and read by every later leg. Deliberately NOT
#: defaulted: a leg that ran before the probe would bind a guess.
EXPANSION: Optional[str] = None
PROBE_ARTIFACT: Optional[str] = None


def _carryable_magnetic_source(fields: Any) -> Any:
    """A REAL magnetic ``VolumeSource`` on this grid that publishes its deposit index.

    SEARCHED, NOT SPELLED. Which component and which offset a folded grid will accept
    depends on the mirror planes it carries: a source centred on an even plane and
    driving a component that plane makes ODD is refused by ``sources`` itself, by name,
    because the parity condition would constrain it against itself. Hard-coding one
    placement would therefore turn a fixture change into a gate that refuses to build
    rather than one that measures. This walks the three magnetic components and a few
    half-cell offsets and returns the first placement the ENGINE accepts and that
    actually deposits, so the leg beside it is comparing against a real deposit.
    """
    grid = fields.grid
    spacing = float(grid.dx)
    for component in ("Hy", "Hz", "Hx"):
        for step in range(6):
            centre = (0.0, 0.5 * step * spacing, 0.0)
            try:
                source = VolumeSource(
                    grid=grid, component=component, center=centre,
                    size=(0.0, 0.0, 0.0),
                    envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                    amplitude=1.0)
            except ValueError:
                continue          # a parity this plane refuses; not this placement
            if getattr(source, "_point_ix", None) is None:
                continue
            if not source._n_source_points:
                continue          # deposits nothing; it could not discriminate
            return source
    raise SystemExit(
        "no magnetic VolumeSource could be placed on this fixture, so the admission "
        "half of the source leg would measure the absence of a source rather than the "
        "predicate; fix the placement rather than dropping the clause")


class _MagneticSource:
    """The smallest thing the source clause can read: a declared field type."""

    field_type = "B"


class _ElectricSource:
    """Its complement. An electric source is injected in the OTHER seam."""

    field_type = "D"


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose.

    A complex64 array's word view is its two planes interleaved, which is exactly what
    the kernel writes, so this is the same comparison the real gate makes with no
    complex-specific relaxation anywhere.
    """
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def complex_from_planes(real: Any, imag: Any) -> Any:
    """Build a complex64 array from two float32 planes THROUGH THE WORD VIEW.

    ``real + 1j*imag`` DESTROYS THE SIGN OF ZEROS in ``imag``: ``1j`` is a complex128
    scalar, so the product is a full complex multiply whose imaginary part comes out
    ``-0.0`` regardless of what ``imag`` held. That defect made the first three
    expansion probes classify NumPy as matching NO arm. Every complex construction in
    this file goes through the word view, for that reason.
    """
    out = np.empty(np.shape(real), dtype=np.complex64)
    view = out.view(np.float32)
    view[..., 0::2] = np.asarray(real, dtype=np.float32)
    view[..., 1::2] = np.asarray(imag, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

def _planes(rng: Any, shape: Sequence[int], value_class: str
            ) -> Tuple[np.ndarray, np.ndarray]:
    """One (real, imaginary) plane pair in ONE value class."""
    if value_class == "gaussian":
        real = (rng.standard_normal(shape) * 0.37).astype(np.float32)
        imag = (rng.standard_normal(shape) * 0.29).astype(np.float32)
        # SIGNED ZEROS SEEDED INTO BOTH PLANES. They are the class the complex
        # helpers' literal zero cross terms exist for: folding `z_im * 0.0f` to a
        # literal misses 12/128 words on the exhaustive table and 0/16,384 on random
        # data, so random seeding alone would arm those mutations at nothing.
        real.reshape(-1)[::17] = np.float32(-0.0)
        real.reshape(-1)[7::23] = np.float32(0.0)
        imag.reshape(-1)[3::19] = np.float32(-0.0)
        imag.reshape(-1)[11::29] = np.float32(0.0)
        return real, imag
    if value_class == "uniform":
        return (rng.uniform(-1.0, 1.0, size=shape).astype(np.float32),
                rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
    if value_class == "signed_zero_lattice":
        # +0.0 and -0.0 on a checkerboard, in BOTH planes. The two words differ
        # (0x00000000 vs 0x80000000) and a byte gate SEES the difference, so this is
        # what exercises `flag ? float2(0,0) : v`, every subtraction's sign convention
        # and the parity product against an exact zero.
        real = np.where(rng.integers(0, 2, size=shape) == 0, 0.0, -0.0)
        imag = np.where(rng.integers(0, 2, size=shape) == 0, 0.0, -0.0)
        return real.astype(np.float32), imag.astype(np.float32)
    if value_class == "subnormal_band":
        return ((rng.standard_normal(shape) * 1e-38).astype(np.float32),
                (rng.standard_normal(shape) * 1e-38).astype(np.float32))
    raise ValueError(f"unknown value class {value_class!r}")


def build(keywords: Mapping[str, Any], seed: int,
          pml_faces: Optional[Any] = None,
          value_class: str = "gaussian") -> Tuple[Any, Any]:
    """One seeded COMPLEX folded engine. Called twice (or three times) per case.

    The GRID and the MATERIAL come from the shared composition matrix, which is the
    same builder the folded complex family's own gate uses, so the fixture this
    certifies is the fixture that family was certified on. Only the field state is
    reseeded here, and identically on every side — the epsilon volume must stay
    bit-equal or the comparison measures the material rather than the kernel.

    ``pml_faces`` REPLACES the matrix's absorber with an explicitly requested one.
    Exactly one case uses it and the reason is measured rather than aesthetic: the
    matrix gives a folded axis a 2-cell high-face layer, and on such a layer
    ``kps[0] == kps[2]`` to the bit, which makes this family's moved coefficient index
    unobservable.
    """
    fields, pml = matrix.folded(**dict(keywords))
    if pml_faces is not None:
        pml = PML(grid=fields.grid, thickness=pml_faces)
    rng = np.random.default_rng(seed)
    shape = tuple(fields.grid.shape)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        real, imag = _planes(rng, shape, value_class)
        array[...] = complex_from_planes(real, imag)
    return fields, pml


def state_of(fields: Any) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Any, right: Any) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    """The composer's OWN live set, never a second model of what a step is.

    ``None`` is a refusal rather than an empty tuple: a gate that stepped nothing would
    compare a no-op with a no-op and pass.
    """
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    return tuple(live)


def array_step(fields: Any, pml: Any, live: Sequence[str]) -> None:
    for name in live:
        ARRAY_PATH[name](fields, pml)


def metal_step(fields: Any, pml: Any, dispatch: Mapping[str, Any],
               owned: Sequence[str], residency: Residency,
               live: Sequence[str]) -> None:
    """One complete step, on the device where a plan owns the pass.

    A pass no plan owns runs on the array path and is bracketed with an explicit
    ``sync_out`` / ``sync_in``. That bracket is the residency clause made operational:
    without it the next device launch would read the mirror's stale bytes, which is
    smooth, plausible and wrong.

    ``owned`` and ``dispatch`` are SEPARATE because this plan owns FOUR passes and
    dispatches at one of them. Deriving the skip set from the dispatch keys would
    silently leave ``fill_symmetry_bc_B`` and ``zero_metal_B`` running on the host on
    top of the copies the kernel already carried — which, for the fill, is IDEMPOTENT
    and would hide a dropped carry.
    """
    skip = set(owned)
    for name in live:
        if name in skip:
            plan = dispatch.get(name)
            if plan is not None:
                plan.run()
            continue
        residency.sync_out()
        ARRAY_PATH[name](fields, pml)
        residency.sync_in()


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(keywords: Mapping[str, Any], seed: int, steps: int,
             pml_faces: Optional[Any] = None,
             functions: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             value_class: str = "gaussian",
             require_every_volume_moved: bool = True,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    reference, reference_pml = build(keywords, seed, pml_faces, value_class)
    actual, actual_pml = build(keywords, seed, pml_faces, value_class)

    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    plan = family.plan_metal_folded_complex_fused_magnetic_pair(
        actual, actual_pml, sources=(), residency=residency, functions=functions)
    if plan is None:
        reasons = family.metal_folded_complex_fused_magnetic_pair_coverage(
            actual, actual_pml, (), residency).reasons
        return {"passed": False,
                "reason": "the folded complex fused magnetic pair was refused",
                "refusals": list(reasons)}
    if patch is not None:
        patch(plan, actual_pml, residency)

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    before = frozen(actual)
    # THE SIGNED-ZERO FLOOR. A census of zero would mean the seeding never built the
    # class the literal zero cross terms exist for, and every zero-term mutation would
    # be armed at nothing. Not required on the band seed, which has no zeros by
    # construction.
    zeros = subnormal.signed_zero_census(
        np.concatenate([words(value).view(np.float32) for value in before.values()]))
    residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(actual, actual_pml, {family.SLOT: plan},
                   plan.replaces_sub_steps, residency, live)
        residency.sync_out()
        difference = compare(reference, actual)
        census = sum(subnormal.census(value)
                     for value in state_of(reference).values())
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": int(census)})
        if difference or census:
            break

    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    clean = all(row["reference_subnormals"] == 0 for row in per_step)
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step))
    seeded_zeros = (int(zeros.get("negative_zero", 0))
                    + int(zeros.get("positive_zero", 0)))
    movement_ok = (not still) if require_every_volume_moved else True
    return {
        "passed": bool(identical and clean and launches_ok and movement_ok),
        "bit_identical": identical,
        "subnormal_free": clean,
        "seeded_signed_zeros": dict(zeros),
        "seeded_signed_zero_count": seeded_zeros,
        "value_class": value_class,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "reference_subnormals": per_step[-1]["reference_subnormals"],
        "arrays_compared": len(before),
        "arrays_that_never_moved": still,
        "moved_words": int(sum(moved.values())),
        "runs": plan.runs, "launches": plan.launches,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
        "boundary_codes": list(plan.codes), "phased": list(plan.phased),
        "phase_values": [list(value) for value in plan.phase_values],
        "phases": list(plan.phases),
        "parity_values": [list(value) for value in plan.parity_values],
        "far_parity_values": [list(value) for value in plan.far_parity_values],
        "reflect_rows": list(plan.reflect),
        "zero_metal": list(plan.zero_metal),
        "carried_ghost_axes": [list(axes) for axes in plan.carried_axes],
        "carried_far_ghost_axes": [list(axes) for axes in plan.far_carried_axes],
        "expansion": plan.expansion,
        "has_offdiagonal_epsilon": bool(
            getattr(actual, "has_offdiagonal_epsilon", False)),
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(keywords: Mapping[str, Any], seed: int, steps: int,
                         pml_faces: Optional[Any] = None) -> Dict[str, Any]:
    """The SEPARATE certified folded complex products beside the fused one, same state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three engines
    from one seed: the array path, the two ALREADY CERTIFIED folded complex Metal
    products stepping the same seam as separate dispatches with the mirror fill and the
    wall clear left on the HOST between them, and the fused product. All three must
    agree word for word at every complete step, and the DISPATCH COUNTS and the
    surviving in-seam host passes are recorded on both sides — which is the only place
    the difference between the compositions shows up at all, since a correct fusion is
    byte-neutral by construction.

    The separate side is not a strawman: those two plans are exactly what ``plan_step``
    composes today for this configuration.
    """
    reference, reference_pml = build(keywords, seed, pml_faces)
    separate, separate_pml = build(keywords, seed, pml_faces)
    fused, fused_pml = build(keywords, seed, pml_faces)

    separate_residency = Residency()
    curl = folded_complex.plan_folded_complex_pml_curl(
        separate, separate_pml, "step_B", separate_residency)
    magnetic = folded_complex.plan_folded_complex_constitutive(
        separate, separate_pml, "H", separate_residency)
    fused_residency = Residency()
    plan = family.plan_metal_folded_complex_fused_magnetic_pair(
        fused, fused_pml, sources=(), residency=fused_residency)
    if curl is None or magnetic is None or plan is None:
        return {"passed": False, "reason": "a declared product was refused",
                "curl": curl is not None, "constitutive": magnetic is not None,
                "fused": plan is not None}

    live = live_passes(fused, fused_pml)
    separate_residency.sync_in()
    fused_residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(separate, separate_pml, {"step_B": curl, "update_H": magnetic},
                   ("step_B", "update_H"), separate_residency, live)
        metal_step(fused, fused_pml, {family.SLOT: plan},
                   plan.replaces_sub_steps, fused_residency, live)
        separate_residency.sync_out()
        fused_residency.sync_out()
        per_step.append({
            "step": step,
            "separate_vs_array": sum(compare(reference, separate).values()),
            "fused_vs_array": sum(compare(reference, fused).values()),
            "fused_vs_separate": sum(compare(separate, fused).values()),
        })
        if any(value for key, value in per_step[-1].items() if key != "step"):
            break

    # The host passes each composition leaves on the array path INSIDE the seam. The
    # fused side carries the mirror fill and the wall clear in the kernel; the separate
    # side does not, so on every folded run it pays a host round trip the fused side
    # does not.
    seam = ("fill_B", "zero_metal_B", "fill_folded_far_ghosts_B")
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    return {
        "passed": bool(identical and plan.launches and curl.launches
                       and magnetic.launches),
        "per_step": per_step,
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": (curl.launches_per_run
                                         + magnetic.launches_per_run),
        "fused_dispatches_per_step": plan.launches_per_run,
        "separate_launches": curl.launches + magnetic.launches,
        "fused_launches": plan.launches,
        "seam_host_passes_for_separate": [n for n in live if n in seam],
        "seam_host_passes_for_fused": [],
        "live_passes": list(live),
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _specialisation(fields: Any, pml: Any) -> Dict[str, Any]:
    """The (codes, Bloch flags, walls, parities) the shipped plan compiles from."""
    residency = Residency()
    plan = family.plan_metal_folded_complex_fused_magnetic_pair(
        fields, pml, sources=(), residency=residency)
    assert plan is not None, (
        "the shipped plan was refused for a case the gate arms mutations on; every "
        "needle below would then be applied to a source nothing launches")
    return {"codes": plan.codes, "phased": plan.phased, "walls": plan.zero_metal,
            "phases": plan.phases, "expansion": plan.expansion,
            "carried_axes": plan.carried_axes}


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``count`` occurrences and REFUSE a no-op edit.

    A mutation that changed nothing would launch the shipped kernel and report the
    defect as uncaught, which is the one failure mode a mutation leg cannot see from
    its own result.
    """
    if old not in source:
        raise AssertionError(f"mutation needle is absent from the source: {old!r}")
    mutated = source.replace(old, new, count)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


def fold_edits(base: str) -> Dict[str, str]:
    """Fifteen source defects reachable on :data:`MUTATION_CASE`.

    THE MUTATION CASE'S GEOMETRY, which is what makes each of these reachable: x and y
    folded with parities (+1, -1), z a live metallic wall, no phased axis. Component 0
    (Bx) images its ghost on x with parity word ``mp0``; component 1 (By) images on y
    with ``mp1``; component 2 (Bz) images nothing — z is not folded — and is the one
    the wall clear reaches. That is the B DIAGONAL, and it is why no row here touches a
    composite corner: this family cannot have one.
    """
    return {
        # THE SEAM ITSELF: take the curl instead of the recurrence's output.
        "seam_takes_pre_recurrence_curl":
            needle(base, "float2 o0_src = v0;", "float2 o0_src = curl0;"),
        # THE FILL, DROPPED. The ghost cell then keeps last step's displacement and
        # last step's H — what a fused pair that ignored `fill_symmetry_bc_B` would
        # produce on every folded grid.
        "near_fill_dropped":
            needle(base, "        if (i == 2) {", "        if (false) {"),
        # THE PARITY, DROPPED. `c_mul(mp1, v1)` is component 1's ghost write on the ODD
        # plane and is the only such call in the source; a plain copy is what a reader
        # who believed "phase +1 is a copy" would write for both planes.
        "fold_parity_dropped":
            needle(base, "f1[g1_n_i] = c_mul(mp1, v1);", "f1[g1_n_i] = v1;"),
        # THE PARITY, WRONG AXIS'S WORD. Two folded axes with different parities is
        # exactly the configuration where this is a plane of wrong signs rather than a
        # no-op.
        "fold_parity_takes_the_other_axis_word":
            needle(base, "f1[g1_n_i] = c_mul(mp1, v1);", "f1[g1_n_i] = c_mul(mp0, v1);"),
        # THE SOURCE ROW. MEEP's `little_owned_corner0` puts the image at stored cell
        # 2, never cell 1 (stepping._mirror_source, MIRROR_SOURCE_INDEX).
        "ghost_images_the_wrong_source_row":
            needle(base, "int g1_n_i = ii - 2 * nzi;", "int g1_n_i = ii - 1 * nzi;"),
        # THE OWNERSHIP CARVE-OUT, DROPPED. The destination thread then forms `v` from
        # a word the source thread writes and stores its own displacement over the
        # imaged ghost — the race the restructure exists to remove.
        "destination_thread_keeps_its_displacement":
            needle(base, "    if (!(i == 0)) {", "    if (!(false)) {"),
        # THE CELL-0 MASK, on the B DIAGONAL (`at_x` for component 0, not the D side's
        # off-diagonal pair).
        "ownership_mask_dropped":
            needle(base, "    curl0 = at_x ? float2(0.0f, 0.0f) : curl0;\n", ""),
        # THE WALL CLEAR on the owned cell, and it is a COMPLEX zero.
        "zero_metal_dropped":
            needle(base, "    v2 = at_z ? float2(0.0f, 0.0f) : v2;\n", ""),
        # THE WALL CLEAR, REAL PLANE ONLY. The array path assigns a complex zero to
        # BOTH planes (S:1896, :1902); clearing only the real one is the defect a
        # reader porting the real twin's `0.0f` would write.
        "zero_metal_clears_only_the_real_plane":
            needle(base, "v2 = at_z ? float2(0.0f, 0.0f) : v2;",
                   "v2 = at_z ? float2(0.0f, v2.y) : v2;"),
        # shaders.py rule 2: flattening these parens is a different float32 number.
        "curl_parens_flattened":
            needle(base, "float2 t0 = ((c_y - c) + (b - b_z));",
                   "float2 t0 = (c_y - c + b - b_z);"),
        # THE DIRECTION. `step_B` shifts UP and `step_D` shifts DOWN
        # (SUB_STEPS['step_B']['backward'] == 0); this is the D curl in the B seam.
        "curl_shift_is_backward":
            needle(base, "int si = i + 1, sj = j + 1, sk = k + 1;",
                   "int si = i - 1, sj = j - 1, sk = k - 1;"),
        "fu_store_dropped":
            needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
                   "    u1[ii] = n1; u2[ii] = n2;"),
        "constitutive_accumulations_reversed":
            needle(base,
                   "        o0_acc = o0_acc + c_mul_coefficient_left(kp_0, o0_src);\n"
                   "        o0_acc = o0_acc - c_mul_coefficient_left(km_0, o0_prev);",
                   "        o0_acc = o0_acc - c_mul_coefficient_left(km_0, o0_src);\n"
                   "        o0_acc = o0_acc + c_mul_coefficient_left(kp_0, o0_prev);"),
        # THE GHOST'S OWN CONSTITUTIVE UPDATE, in two separable pieces. `update_H`
        # reads the ghost cell the fill just wrote, so both are live work.
        "ghost_fw_store_dropped":
            needle(base, "            w0[g0_n_i] = g0_n_src;",
                   "            // stale ghost f_w_Hx"),
        "ghost_constitutive_dropped":
            needle(base, "            h0[g0_n_i] = g0_n_acc;\n", ""),
    }


def far_edits(base: str) -> Dict[str, str]:
    """The FAR CARRY's own defects, reachable ONLY on :data:`FAR_MUTATION_CASE`.

    THE DEAD-BRANCH GUARD IS STRUCTURAL HERE. Every needle below sits on a line the
    kernel emits only for a folded PERIODIC axis, and :data:`MUTATION_CASE` folds two
    MIRROR_METALLIC axes and so emits none of them. ``needle`` raises if the text is
    absent, so a case that stopped carrying a far ghost fails loudly rather than
    scoring a mutation against nothing.

    The case folds ``x`` and ``y`` PERIODIC with MIXED parities at an ODD full count
    and walls ``z``, which is what puts every one of these lines in the source at once:
    a near carry, two far carries with DIFFERENT parity words, a near/far composite, a
    far/far composite (component 2, whose own axis is not folded), the top-plane curl
    mask, a reflect row that is NOT ``stored - 2``, and the wall clear on the owned
    cell.
    """
    return {
        # THE FAR FILL, DROPPED. The top stored slot then keeps whatever `step_B` left
        # there — which the top-plane mask made a complex zero — instead of the image
        # `connect_the_chunks` builds. This is the state of this family BEFORE the
        # carry, with the refusal removed: the exact silent-wrong-answer the clause
        # existed to prevent.
        "far_fill_dropped":
            needle(base, "        if (j == reflect_y) {", "        if (false) {"),
        # THE FAR PARITY, DROPPED. `mirror_parity` is `phase * (1 - 2*iyee)`, so a
        # shift-1 far destination takes MINUS the plane's phase; a plain copy is what a
        # reader who carried the near fill's rule to the far one would write.
        #
        # IT IS AIMED AT COMPONENT 1'S FAR AXIS, NOT COMPONENT 0'S, AND THE CHOICE IS
        # FORCED. This case phases (+1, -1), so the far words are `fp0 = -phase_x = -1`
        # and `fp1 = -phase_y = +1` (measured: far_parity_words gives
        # ((-1,0), (+1,0), (0,0))). `fp1` IS THE IDENTITY, so dropping it is arithmetic
        # no-op and the same needle on `f0[g0_j_i]` reported 12 launches and 0 differing
        # words — a mutation scored against nothing. `_far_axis_word_is_not_unity`
        # asserts the axis this needle uses stays non-identity.
        "far_parity_dropped":
            needle(base, "f1[g1_i_i] = c_mul(fp0, v1);", "f1[g1_i_i] = v1;"),
        # THE FAR PARITY, TAKING THE NEAR WORD. The two differ by a sign at every
        # parity (mirror_parity_coefficients returns +phase and -phase), so this is a
        # plane of wrong signs rather than a rounding question.
        "far_parity_takes_the_near_word":
            needle(base, "f0[g0_j_i] = c_mul(fp1, v0);", "f0[g0_j_i] = c_mul(mp1, v0);"),
        # THE REFLECT ROW, REPLACED BY THE FIXED `n - 2`. `_far_reflect_rows`' own
        # docstring says in capitals that this is a whole cell wrong at an ODD full
        # count, and this case carries one (extent=2.1).
        # The substituted constant is `nyi - 2` because that is the FIXED row the
        # docstring names and the row the precondition compares against. It was
        # `nyi - 3` and that is exactly `_far_reflect_rows`' own answer on this grid
        # (shape 13, reflect row 10), so the edit rewrote the plane to itself and
        # scored 12 launches at 0 differing words.
        "reflect_row_is_the_fixed_n_minus_two":
            needle(base, "        if (j == reflect_y) {",
                   "        if (j == nyi - 2) {"),
        # THE DESTINATION PLANE. The far ghost is the LAST stored slot; writing the one
        # below it images into an owned, stepped cell.
        "far_destination_is_one_row_low":
            needle(base, "int g0_j_i = ii + ((nyi - 1) - reflect_y) * nzi;",
                   "int g0_j_i = ii + ((nyi - 2) - reflect_y) * nzi;"),
        # THE OWNERSHIP CARVE-OUT for the far plane. Without it the top-plane thread
        # stores its own (masked) displacement over the image the source thread wrote —
        # the race the restructure exists to remove, and a real one: the two writes are
        # unordered.
        "far_destination_thread_keeps_its_displacement":
            needle(base, "    if (!(i == 0 || last_y)) {",
                   "    if (!(i == 0)) {"),
        # THE PARITY CHAIN's two defects — FOLDED INTO ONE WORD and REVERSED — are
        # armed in :data:`NULL_EDITS`, not here, and the reason is MEASURED rather than
        # chosen. See that table's entries: a mirror parity is exactly +-1 + 0j on
        # every axis of every case (`far_parity_words` / `parity_words` return
        # ((-1,0),(+1,0),(0,0)) on this one), and +-1 composes exactly, so neither the
        # grouping nor the order can move a NORMAL operand's bits. Both were armed here
        # as CAUGHT-required and both reported 12 launches at 0 differing words
        # (results/metal_folded_complex_far_carry_2026-08-21/, legs 62 and 63); a
        # re-run on `signed_zero_lattice` reported the same
        # (…_2026-08-21c/, same legs). Leg `parity_chain_order` is where the residual
        # LIVES and it is measured directly there, on an engineered table.
        # THE FAR/FAR COMPOSITE, DROPPED. Component 2's own axis is not folded here, so
        # it is a far destination on BOTH x and y and the corner where both fire is the
        # only cell in the kernel carrying two FAR parities. Nothing else reaches it.
        "far_far_composite_dropped":
            needle(base, "        if (i == reflect_x && j == reflect_y) {\n",
                   "        if (false && i == reflect_x && j == reflect_y) {\n"),
        # THE TOP-PLANE CURL MASK. `folded_top_plane_mask` emits it only on a folded
        # PERIODIC axis; it is what makes the top slot a not-owned allocation slot
        # rather than a stepped cell, and it feeds `fu` at that cell.
        "top_plane_curl_mask_dropped":
            needle(base, "    curl0 = last_y ? float2(0.0f, 0.0f) : curl0;\n", ""),
        # THE FAR GHOST'S COEFFICIENT PAIR. A far destination shares the indexed axis
        # with its source thread and therefore takes that thread's own pair; reloading
        # at index 0 would apply the mirror plane's absorber profile to the top plane.
        "far_ghost_reloads_the_near_coefficient":
            needle(base,
                   "            g0_j_acc = g0_j_acc + c_mul_coefficient_left(kp_0, "
                   "g0_j_src);",
                   "            g0_j_acc = g0_j_acc + c_mul_coefficient_left(kp0[0], "
                   "g0_j_src);"),
    }


def phase_edits(base: str) -> Dict[str, str]:
    """Two source defects reachable ONLY on :data:`PHASE_MUTATION_CASE`.

    A folded axis may never be phased (``folded_complex._reachable_phase_flags``), so
    the two-fold mutation case above emits NO rotation block at all and both of these
    would rewrite absent text. That is precisely the dead-branch trap, and the split
    into two mutation cases is what avoids buying it: ``needle`` raises here if the
    line is not present, so a case that stopped carrying a phased axis fails loudly.
    """
    return {
        # THE WRAP PREDICATE. The phase applies on the WRAPPED LANE ONLY; applying it
        # everywhere is a rotation of the whole volume.
        "bloch_phase_applied_on_every_lane":
            needle(base, "b_x = wx ? c_mul(b_x, px) : b_x;",
                   "b_x = c_mul(b_x, px);"),
        # THE OPERAND ORDER on the CURL side. `complex_fields` spells the wrap rotation
        # FIELD LEFT (S:1862); the coefficient-left form is a different rounding
        # wherever both planes are live, which is what the generic k = 0.3 x-phase in
        # this case delivers.
        "bloch_phase_operand_order_swapped":
            needle(base, "b_x = wx ? c_mul(b_x, px) : b_x;",
                   "b_x = wx ? c_mul(px, b_x) : b_x;"),
    }


#: THE MUTATIONS THAT MUST *NOT* CATCH, with the measurement that says so. Scored NULL
#: CONFIRMED rather than skipped: a defect nobody arms is a claim, and one that fires
#: unexpectedly is a finding either way.
#:
#: A NULL IS ONLY A NULL IF THE LINE IS LIVE, and here that is measured rather than
#: argued: both needles sit on lines inside ``if (j == 2)`` / ``if (i == 2)``, and
#: OTHER needles on the SAME two lines — ``fold_parity_dropped`` and
#: ``fold_parity_takes_the_other_axis_word`` on the first, ``ghost_constitutive_dropped``
#: reading the second's register on the other — are required to be CAUGHT on the same
#: case. A null whose siblings catch on the same statement cannot be the dead-branch
#: failure; :data:`SIBLING_WITNESS` names the pairing so the artifact carries it.
SIBLING_WITNESS: Dict[str, Tuple[str, ...]] = {
    "near_fill_operand_order_swapped": ("fold_parity_dropped",
                                        "fold_parity_takes_the_other_axis_word"),
    "ghost_reload_replaced_by_the_register": ("ghost_constitutive_dropped",
                                              "ghost_fw_store_dropped"),
    "parity_chain_folded_into_one_word": ("far_fill_dropped",
                                          "far_far_composite_dropped"),
    "parity_chain_order_reversed": ("far_fill_dropped",
                                    "far_far_composite_dropped"),
}

#: ``name -> (case, old, new, why)``. The CASE is per entry because the two chain
#: nulls sit on lines only :data:`FAR_MUTATION_CASE` emits; running them on
#: :data:`MUTATION_CASE` would be the dead-branch failure ``needle`` refuses.
NULL_EDITS: Dict[str, Tuple[str, str, str, str]] = {
    "near_fill_operand_order_swapped": (
        MUTATION_CASE,
        "f1[g1_n_i] = c_mul(mp1, v1);", "f1[g1_n_i] = c_mul(v1, mp1);",
        "folded_complex.folded_mirror_fill_complex_source records the swap as an "
        "EQUIVALENCE measured over a 512-word engineered table: with the parity's "
        "imaginary word an exact +0.0 the two fma addends are the same exact zero and "
        "float addition is sign-commutative. Armed anyway, because an equivalence that "
        "stopped holding is a finding."),
    "ghost_reload_replaced_by_the_register": (
        MUTATION_CASE,
        "float2 g0_n_b = f0[g0_n_i];", "float2 g0_n_b = c_mul(mp0, v0);",
        "the reload returns the word the line above stored, by the same thread at the "
        "same address, so recomputing it is the same bits. Armed to show the reload is "
        "not load-bearing arithmetic — only a faithful re-read of what the array "
        "path's fill leaves for update_H."),
    # THE TWO CHAIN DEFECTS. They were armed as CAUGHT-required and are NULL by
    # measurement, not by preference — and the distinction matters, because the ORDER
    # is still transcribed rather than chosen (family.parity_chain) and leg
    # `parity_chain_order` still scores the collapse as observable.
    "parity_chain_folded_into_one_word": (
        FAR_MUTATION_CASE,
        "            float2 g0_jn_v = c_mul(mp0, v0);\n"
        "            f0[g0_jn_i] = c_mul(fp1, g0_jn_v);",
        "            f0[g0_jn_i] = c_mul(mp0, v0);",
        "a mirror parity is exactly +-1 + 0j on every axis (mirror_parity is "
        "phase * (1 - 2*iyee) with phase in {+1,-1}), and +-1 composes EXACTLY in "
        "float32, so collapsing the chain cannot move a normal operand's bits. Leg "
        "parity_chain_order measures where the residual does live — up to 23 of 128 "
        "uint32 words on an engineered signed-zero/subnormal table — and this needle "
        "on a device fixture reports 12 launches at 0 differing words on the default "
        "value class AND on signed_zero_lattice "
        "(results/metal_folded_complex_far_carry_2026-08-21/ and _2026-08-21c/, "
        "leg 62). Armed anyway: a collapse that started diverging on ordinary field "
        "values would mean the parity words had stopped being +-1."),
    "parity_chain_order_reversed": (
        FAR_MUTATION_CASE,
        "            float2 g0_jn_v = c_mul(mp0, v0);\n"
        "            f0[g0_jn_i] = c_mul(fp1, g0_jn_v);",
        "            float2 g0_jn_v = c_mul(fp1, v0);\n"
        "            f0[g0_jn_i] = c_mul(mp0, g0_jn_v);",
        "the same +-1 exactness as the fold: the array path applies the near pass "
        "before the far one (driver.py:3285 then :3287) and family.parity_chain "
        "TRANSCRIBES that order, but two exact +-1 multiplies commute bitwise on "
        "normal operands. Leg parity_chain_order measures the residual at 8 of 128 "
        "words on the engineered table; this needle reports 0 differing words over 12 "
        "launches on the device fixture (same artifacts, leg 63)."),
}


def compile_edits(edits: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    return {name: {shaders.CONTRACT_OFF: compile_source(
        source).folded_complex_fused_magnetic_pair_step}
        for name, source in edits.items()}


#: THE ONE MUTATION WHOSE CATCH IS CONDITIONAL, and the condition is measured. Reusing
#: the source thread's coefficient pair for the imaged ghost is exactly the shortcut
#: :mod:`.folded_fused_pair` is entitled to and this family is not. It is observable
#: only where the two entries differ; leg ``moved_coefficient`` predicts that from the
#: coefficient vectors and requires the prediction to hold.
MOVED_COEFFICIENT_EDIT = ("float g0_n_kp = kp0[0], g0_n_km = km0[0];",
                          "float g0_n_kp = kp_0, g0_n_km = km_0;")

#: Where each binding group starts in the plan's argument tuple. Spelled once, so the
#: two host mutations and the kernel signature cannot drift apart.
CURL_COEFFICIENT_SLOTS = tuple(range(15, 21))
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(21, 27))


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_h_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind the HALF-INTEGER constitutive coefficients to the H half.

    ``update_H`` takes ``kps_a``/``kms_a`` and ``update_E`` takes ``kps_a_h``
    (stepping.py:948 vs :1015). The kernel takes six pointers and never asks which
    lattice they came from, so this is a silent half-cell error in the absorber profile
    — converged, smooth and wrong — and no shader mutation can reach it.
    """
    _rebind(plan, CONSTITUTIVE_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}_h",
                              getattr(pml, f"{stem}_{axis}_h"), constant=True)
             for axis in "xyz" for stem in ("kps", "kms")])


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the B curl the INTEGER split-field coefficients.

    ``step_B`` reads half-integer positions and ``step_D`` integer ones (SUB_STEPS'
    ``suffix``). Getting it backwards is a half-cell error, not a crash.
    """
    _rebind(plan, CURL_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}",
                              getattr(pml, f"{stem}_{axis}"), constant=True)
             for axis in "xyz" for stem in ("kms", "sinv")])


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "h_half_takes_the_half_integer_lattice": swap_h_lattice,
    "curl_takes_the_integer_lattice": swap_curl_lattice,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_policy() -> Dict[str, Any]:
    """The resolved float32 subnormal policy, and ``keep`` REFUSED BY NAME.

    Metal has ONE attainable policy. That is not a gap in this gate: it is a
    measurement the executor's own arm carries (``subnormal.ATTAINABLE``), and a gate
    that quietly ran under whatever the environment happened to hold would be
    certifying bytes under an unknown precondition. So the report is recorded, the
    admitted flag must be true for the policy this process actually resolved, and the
    OTHER policy must produce a named refusal rather than a silent downgrade.
    """
    report = subnormal.mps_policy_report()
    keep = subnormal.mps_policy_report(subnormal.KEEP)
    keep_named = [reason for reason in keep["reasons"]
                  if "cannot honour it" in reason and "Metal flushes" in reason]
    return {
        "passed": bool(report["admitted"] and report["resolved"] == subnormal.FLUSH
                       and not keep["admitted"] and keep_named),
        "report": report,
        "attainable": list(subnormal.ATTAINABLE),
        "keep_admitted": keep["admitted"],
        "keep_named_refusals": keep_named,
        "note": "both float32 subnormal policies are ASKED here; this executor has "
                "one, and the other's refusal is the measurement",
    }


def leg_expansion() -> Dict[str, Any]:
    """The complex-multiply arm is BOUND BY A PROBE ARTIFACT, never defaulted.

    This family carries no probe of its own: its arithmetic is spliced from
    :mod:`.folded_complex`'s and :mod:`.complex_fields`' emitters, so it binds through
    the folded-complex probe and inherits its refusals. A second probe would be a
    second answer to one question.

    Three things are required:

    1. an artifact must be CONFIGURED and must classify this host;
    2. a MISSING probe must refuse — never a default arm;
    3. an AMBIGUOUS probe (every pattern ``AMBIGUOUS_BOTH``) must ALSO refuse, because
       a reference that matches both arms licenses neither.
    """
    global EXPANSION, PROBE_ARTIFACT
    configured = os.environ.get(folded_complex.PROBE_PATH_ENVIRONMENT)
    record = folded_complex.load_expansion_probe()
    arm = folded_complex.expansion_from_probe(record)
    EXPANSION, PROBE_ARTIFACT = arm, configured

    missing_refused = folded_complex.expansion_from_probe(None) is None
    # EVERY pattern of the family's SUPERSET list, ambiguous. Read off
    # `PARITY_PROBE_PATTERNS` rather than `complex_fields.PROBE_PATTERNS`: this family
    # requires a sixth orientation, and a set of names that missed it would leave the
    # sixth pattern absent — which refuses for a DIFFERENT reason and would report an
    # ambiguity check that never ran.
    ambiguous = {"backend": folded_complex.PROBE_BACKEND,
                 "patterns": {name: complex_fields.AMBIGUOUS_BOTH
                              for name in folded_complex.PARITY_PROBE_PATTERNS}}
    assert folded_complex.PARITY_PROBE_PATTERN in folded_complex.PARITY_PROBE_PATTERNS
    ambiguous_refused = folded_complex.expansion_from_probe(ambiguous) is None
    wrong_backend = dict(record or {}, backend="not-a-backend")
    backend_refused = folded_complex.expansion_from_probe(wrong_backend) is None
    digest = (hashlib.sha256(Path(configured).read_bytes()).hexdigest()
              if configured and Path(configured).exists() else None)
    return {
        "passed": bool(arm and configured and missing_refused
                       and ambiguous_refused and backend_refused),
        "expansion": arm,
        "probe_artifact": configured,
        "probe_sha256": digest,
        "probe_backend": (record or {}).get("backend"),
        "missing_probe_refused": missing_refused,
        "ambiguous_probe_refused": ambiguous_refused,
        "wrong_backend_probe_refused": backend_refused,
        "parity_pattern": folded_complex.PARITY_PROBE_PATTERN,
    }


def leg_binding_ceiling() -> Dict[str, Any]:
    """35 separate bindings must FAIL; the packed 28 must COMPILE, LAUNCH and read.

    This is what makes "one dispatch for all three components" a measurement rather
    than a preference. The REFUTED source is :mod:`.complex_fused_magnetic_pair`'s —
    the fold adds no POINTER, so the two families' signatures are the same shape — and
    it is imported rather than re-spelled, then re-measured here on this tree.

    THE PACKED HALF IS THIS FAMILY'S OWN, and it has to be: the record carries NINE
    ``float2`` members where the unfolded pair carries three — the far carry added the
    three FAR parity words and three reflect rows, taking the struct from 72 bytes to
    104 — so the padding question is a different question and is answered by LAUNCHING
    the struct and reading every field back rather than by arithmetic. THE BINDING
    COUNT IS UNCHANGED AT 28, which is the property the far carry had to preserve:
    complex storage already spent this platform's headroom on ``float2`` volumes
    (complex_fields.py:38), so six more buffers would have been 34 against a ceiling
    of 31 and the carry would have been unbuildable.
    """
    import torch  # noqa: PLC0415

    row: Dict[str, Any] = {
        "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
        "packed_bindings": family.PACKED_BINDINGS,
        "params_itemsize": family.PARAMS_ITEMSIZE, "ceiling": 31}
    try:
        compile_source(family.refuted_separate_scalar_source())
        row.update(separate_compiled=True, separate_error="", passed=False,
                   note="the separate-scalar signature COMPILED; this family's shape "
                        "rests on a ceiling this host does not have")
        return row
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        row["separate_compiled"] = False
        row["separate_error"] = message.splitlines()[0] if message else ""
        row["separate_refused_for_the_right_reason"] = (
            "out of bounds" in message and "buffer" in message)

    # The packed signature, compiled AND launched, with every field read back. 27
    # pointers plus one packed Params& at buffer 27 is the shipped shape.
    pointers = "\n".join(f"    device float*       b{n:<6}[[buffer({n})]],"
                         for n in range(family.PACKED_BINDINGS - 1))
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params {",
        "    float2 px; float2 py; float2 pz;",
        "    float2 m0; float2 m1; float2 m2;",
        "    float2 d0; float2 d1; float2 d2;",
        "    uint nx; uint ny; uint nz; uint n_elem; float dtdx;",
        "    int rx; int ry; int rz;",
        "};", "",
        "kernel void packed_probe(", pointers,
        f"    constant Params&    prm     [[buffer({family.PACKED_BINDINGS - 1})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        "    b0[idx] = float(prm.nx);", "    b1[idx] = float(prm.ny);",
        "    b2[idx] = float(prm.nz);", "    b3[idx] = float(prm.n_elem);",
        "    b4[idx] = prm.dtdx;",
        "    b5[idx] = prm.px.x;", "    b6[idx] = prm.px.y;",
        "    b7[idx] = prm.pz.x;", "    b8[idx] = prm.m0.x;",
        "    b9[idx] = prm.m1.x;", "    b10[idx] = prm.m2.y;",
        "    b11[idx] = prm.d0.x;", "    b12[idx] = prm.d1.x;",
        "    b13[idx] = prm.d2.y;",
        "    b14[idx] = float(prm.rx);", "    b15[idx] = float(prm.ry);",
        "    b16[idx] = float(prm.rz);",
        f"    b17[idx] = b{family.PACKED_BINDINGS - 2}[idx] + 1.0f;", "}", ""))
    try:
        function = compile_source(source).packed_probe
    except Exception as exc:  # noqa: BLE001
        row.update(packed_compiled=False, packed_error=str(exc).splitlines()[0],
                   passed=False)
        return row
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
               for _ in range(family.PACKED_BINDINGS - 1)]
    buffers[-1] = torch.full((count,), 7.0, dtype=torch.float32, device="mps")
    shape, dtdx = (3, 4, 5), 0.125
    phases = ((0.25, -0.5), (0.75, 0.125), (-0.375, 0.625))
    parities = ((1.0, 0.0), (-1.0, 0.0), (0.0, 0.5))
    # DISTINCT WORDS FROM THE NEAR TRIPLE, deliberately: a struct whose far members
    # aliased the near ones would read back correctly while the kernel multiplied by
    # the wrong parity, which is the exact defect `far_parity_takes_the_near_word`
    # arms in the source.
    far_parities = ((0.5, -0.25), (-0.75, 0.0), (0.0, -0.875))
    reflect = (9, -1, 4)
    # The record comes from the PLAN BUILDER's own packer, so the field the kernel
    # reads is the field the plan writes rather than a hand-built twin.
    params = family._params_tensor(shape, dtdx, phases, parities, far_parities,
                                   reflect, "mps")
    function(*buffers, params)
    torch.mps.synchronize()
    read = [buffers[index].cpu().numpy() for index in range(18)]
    expected = (float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]), float(np.float32(dtdx)),
                phases[0][0], phases[0][1], phases[2][0],
                parities[0][0], parities[1][0], parities[2][1],
                far_parities[0][0], far_parities[1][0], far_parities[2][1],
                float(reflect[0]), float(reflect[1]), float(reflect[2]), 8.0)
    fields_ok = all(bool(np.all(read[index] == expected[index]))
                    for index in range(18))
    row.update(packed_launched=True,
               packed_fields_read_back=[float(value[0]) for value in read],
               packed_fields_expected=[float(value) for value in expected],
               packed_fields_correct=fields_ok,
               passed=bool(row["separate_refused_for_the_right_reason"] and fields_ok))
    return row


def leg_transcription(codes: Sequence[int], phased: Sequence[int],
                      expansion: str) -> Dict[str, Any]:
    """All THREE pieces must be the certified emitters' OWN TEXT, not a similar one.

    A construction is a hypothesis until something compares the strings. Three
    comparisons, and the third is the one no sibling family has:

    * the lifted curl head must be a VERBATIM PREFIX of
      ``folded_complex.folded_bloch_curl_source(codes, backward=False, phased)``'s body
      — so the ghost gather, the Bloch rotation, the curl grouping and the B-diagonal
      cell-0 mask in the fused kernel are character-for-character the certified folded
      complex curl's;
    * this family's parameterised constitutive statements, rendered with the certified
      spellings, must equal ``complex_fields.bloch_constitutive_source('H')``'s own
      statements;
    * the GHOST WRITE must be
      ``folded_complex.folded_mirror_fill_complex_source``'s own near-pass line under
      the stated rename. That is the piece which is neither curl nor constitutive, and
      without this check it would be the one hand-typed complex multiply in the file.
    """
    rows: Dict[str, Any] = {"modes": {}}
    ok = True
    folded = [axis for axis, code in enumerate(codes) if int(code) in MIRROR_CODES]
    # The FAR pass exists only on a folded PERIODIC axis; asked on EVERY axis anyway,
    # because the emitter's line is a property of the axis and not of this case's
    # termination, and a check that ran on nothing would report as a pass.
    for mode in shaders.CONTRACT_MODES:
        head = family.certified_curl_head(codes, phased, expansion, mode)
        certified = folded_complex.folded_bloch_curl_source(
            codes, family.BACKWARD, phased, expansion, mode)
        body = certified.split("uint idx [[thread_position_in_grid]])\n{\n", 1)[1]
        head_ok = body.startswith(head) and bool(head.strip())
        transcription = family.constitutive_transcription(expansion, mode)
        statements = family.certified_curl_statements(codes, phased, expansion, mode)
        recurrence_ok = all(
            all(line in body for line in triple)
            for triple in statements["recurrence"])
        fills = {axis: family.near_fill_transcription(axis, expansion, mode)
                 for axis in folded}
        fill_ok = bool(fills) and all(row["identical"] for row in fills.values())
        far_fills = {axis: family.far_fill_transcription(axis, expansion, mode)
                     for axis in range(3)}
        far_ok = all(row["identical"] for row in far_fills.values())
        rows["modes"][mode] = {
            "curl_head_is_a_verbatim_prefix": head_ok,
            "curl_head_lines": len(head.splitlines()),
            "recurrence_statements_are_lifted": recurrence_ok,
            "constitutive_identical": transcription["identical"],
            "constitutive_statements_per_component":
                [len(block) for block in transcription["certified"]],
            "near_fill_identical": fill_ok,
            "near_fill_rows": {str(axis): row["rows"]
                               for axis, row in fills.items()},
            "far_fill_identical": far_ok,
            "far_fill_rows": {str(axis): row["rows"]
                              for axis, row in far_fills.items()},
        }
        ok = (ok and head_ok and recurrence_ok and transcription["identical"]
              and fill_ok and far_ok)
    rows["folded_axes"] = folded
    rows["passed"] = bool(ok)
    return rows


def leg_parity_chain_order() -> Dict[str, Any]:
    """The composed parity is a CHAIN, its ORDER is normative, and both are measured.

    THE ONE PLACE THIS FAMILY DIVERGES FROM THE REAL BOARD'S FAR CARRY.
    :func:`.folded_fused_magnetic_pair.carried_destinations` folds a composed parity
    into ONE compile-time sign and states that the result is "independent of the order
    the axes are applied in". Under COMPLEX storage the parity is a ``float2`` word and
    the application is a full complex multiply, so BOTH of those collapses have to be
    re-established or refused. This leg refuses them, over an exhaustive
    signed-zero/subnormal table rather than random data:

    1. the array path's ``phase * complex64_array`` (``stepping._write_mirror_ghost``
       :1451, ``_fill_folded_far_ghosts``:1529) must be the FULL complex product and
       NOT a plane-wise real scaling — everything else rests on which one it is;
    2. re-ordering a two-parity chain must MOVE WORDS, or the ordering would be free
       and the emitter's transcription of the driver's pass order would be decoration;
    3. folding a chain into a single word must MOVE WORDS, or this family could have
       imported the real board's closed form.

    Then the closed form itself is checked, not just the arithmetic: for every folded
    quadruple this family builds, :func:`family.parity_chain` must put the NEAR
    application first and the FAR ones in ASCENDING axis order, which is the order
    ``driver.step`` runs the two passes (:3285 then :3287) and the order
    ``stepping._fill_folded_far_ghosts`` loops its axes in (:1518, :1528).
    """
    import itertools as _itertools

    table = np.array(
        [complex(float(np.float32(re)), float(np.float32(im)))
         for re in (0.0, -0.0, 1.0, -1.0, 1e-42, -1e-42, 3.5, -2.25)
         for im in (0.0, -0.0, 1.0, -1.0, 1e-42, -1e-42, 3.5, -2.25)],
        dtype=np.complex64)
    total = int(words(table).size)

    reference: List[Dict[str, Any]] = []
    for phase in (1, -1):
        got = phase * table
        full = np.empty_like(table)
        coefficient = np.complex64(phase)
        full.real = (np.float32(coefficient.real) * table.real
                     - np.float32(coefficient.imag) * table.imag)
        full.imag = (np.float32(coefficient.real) * table.imag
                     + np.float32(coefficient.imag) * table.real)
        scaled = np.empty_like(table)
        scaled.real = np.float32(phase) * table.real
        scaled.imag = np.float32(phase) * table.imag
        reference.append({"phase": phase,
                          "vs_full_complex_product": differing(got, full),
                          "vs_real_scaling": differing(got, scaled)})
    is_full = all(row["vs_full_complex_product"] == 0 for row in reference)
    not_scaling = all(row["vs_real_scaling"] > 0 for row in reference)

    pairs: List[Dict[str, Any]] = []
    for first, second in _itertools.product((1, -1), repeat=2):
        sequential = second * (first * table)
        pairs.append({
            "first": first, "second": second,
            "chain_vs_reversed_chain": differing(sequential, first * (second * table)),
            "chain_vs_folded_product": differing(sequential,
                                                 (first * second) * table)})
    triples: List[Dict[str, Any]] = []
    for first, second, third in _itertools.product((1, -1), repeat=3):
        sequential = third * (second * (first * table))
        triples.append({
            "parities": [first, second, third],
            "chain_vs_folded_product": differing(
                sequential, (first * second * third) * table)})
    order_observable = any(row["chain_vs_reversed_chain"] for row in pairs)
    folding_observable = (all(row["chain_vs_folded_product"] for row in pairs)
                          and all(row["chain_vs_folded_product"] for row in triples))

    # THE CLOSED FORM: near first, then far ascending, on every quadruple.
    codes_set = (CODE_PERIODIC, symmetry_metallic(), MIRROR_CODES[0], MIRROR_CODES[1])
    order_rows: List[Dict[str, Any]] = []
    order_ok = True
    chains_seen = {1: 0, 2: 0, 3: 0}
    for quad in _itertools.product(codes_set, repeat=3):
        if not any(int(code) in MIRROR_CODES for code in quad):
            continue
        for target in range(3):
            near = family.near_fill_axes(quad, target)
            far = family.far_fill_axes(quad, target)
            for subset, carries_near in family.carried_destinations(near, far):
                chain = family.parity_chain(near, subset, carries_near)
                kinds = [kind for _, kind in chain]
                axes = [axis for axis, kind in chain if kind == "far"]
                good = (kinds.count("near") <= 1
                        and (not carries_near or kinds[0] == "near")
                        and axes == sorted(axes)
                        and len(chain) == len(subset) + int(carries_near))
                chains_seen[len(chain)] = chains_seen.get(len(chain), 0) + 1
                order_ok = order_ok and good
                if not good:
                    order_rows.append({"codes": list(quad), "target": target,
                                       "chain": [list(entry) for entry in chain]})
    return {
        "passed": bool(is_full and not_scaling and order_observable
                       and folding_observable and order_ok
                       and chains_seen.get(2) and chains_seen.get(3)),
        "uint32_words_per_comparison": total,
        "array_path_multiply": reference,
        "array_path_is_the_full_complex_product": is_full,
        "array_path_is_not_a_real_scaling": not_scaling,
        "two_axis": pairs,
        "three_axis": triples,
        "reordering_the_chain_moves_words": order_observable,
        "folding_the_chain_moves_words": folding_observable,
        "max_words_moved_by_reordering": max(row["chain_vs_reversed_chain"]
                                             for row in pairs),
        "max_words_moved_by_folding": max(
            [row["chain_vs_folded_product"] for row in pairs]
            + [row["chain_vs_folded_product"] for row in triples]),
        "chain_lengths_enumerated": {str(k): v for k, v in sorted(chains_seen.items())},
        "chain_order_violations": order_rows,
        "chain_order_is_near_then_far_ascending": order_ok,
    }


def symmetry_metallic() -> int:
    """``CODE_METALLIC``, imported where it is used rather than at module scope."""
    from meep_gpu.triton_kernels.symmetry import CODE_METALLIC  # noqa: PLC0415

    return CODE_METALLIC


class _StubGrid:
    """The smallest grid :func:`family._far_carry_reasons` can read, and it LIES.

    Not an engine grid: the clauses under test are arithmetic on ``shape``,
    ``stored_cells``, ``owned_cells`` and ``shape_full``, and building a real ``Grid``
    that violates them is not possible — ``Grid`` derives the stored extent from the
    fold, so a bad reflect row cannot be requested through it. The refusals still have
    to be measured, because on this family a violated row is an OUT-OF-RANGE WRITE
    issued by a thread that owns neither cell, so a stub that answers the four
    questions is the honest fixture. Lifted from
    :mod:`gate_metal_folded_fused_magnetic_pair`, whose far carry asks the identical
    geometric questions.
    """

    cylindrical = False

    def __init__(self, shape: Sequence[int], full: Sequence[int],
                 mirrored: Sequence[bool]) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.shape_full = tuple(int(n) for n in full)
        self._mirrored = tuple(bool(v) for v in mirrored)

    def is_mirrored(self, axis: int) -> bool:
        return self._mirrored[axis]

    def is_metallic(self, axis: int) -> bool:
        return False

    def is_axis(self, axis: int) -> bool:
        return False

    def has_symmetry(self) -> bool:
        return any(self._mirrored)

    def mirror_phase(self, axis: int) -> Optional[int]:
        return 1 if self._mirrored[axis] else None

    def stored_cells(self, axis: int) -> int:
        return self.shape[axis]

    def owned_cells(self, axis: int) -> int:
        # One less on every mirrored axis, which is what makes _stored_past_owned
        # True there — the definition of a folded PERIODIC axis.
        return self.shape[axis] - (1 if self._mirrored[axis] else 0)


def leg_refusal(cases: Mapping[str, Tuple[Dict[str, Any], Optional[Any]]]
                ) -> Dict[str, Any]:
    """THE FLIP, and the boundaries that remain boundaries.

    Until the far carry this leg asserted the OPPOSITE on its first fixture: it built
    a folded PERIODIC grid, where ``fill_folded_far_ghosts_B`` is live inside the seam
    (driver.py:3287), and required the predicate to REFUSE it. The refusal is gone
    because the kernel now images that plane itself, so the same fixture measures the
    flip:

    * the composer must still say the pass is LIVE on that grid — otherwise the carry
      would be about a pass that never runs and this half of the leg would be
      decoration;
    * the predicate must now COVER it and the plan must be BUILT;
    * ``REPLACES`` must NAME the pass, so the walk's ``owned`` set skips it on the host
      and a dropped carry cannot hide behind an idempotent host re-run;
    * the plan must report a far ghost on the components the Yee table says, and a
      reflect row that is ``stepping._far_reflect_rows``' own answer.

    THE REFUSALS THAT REMAIN are measured beside it, because a product that stopped
    refusing everything would pass this leg by accident. An ELECTRIC source is not in
    this seam, which is the asymmetry this whole family rests on; an UNDECLARED source
    set is a refusal rather than an assumed empty one; an UNFOLDED complex grid belongs
    to :mod:`.complex_fused_magnetic_pair` and a REAL folded grid to
    :mod:`.folded_fused_magnetic_pair`; and the reflect-row clauses the carry rests on
    refuse, on stubs, a grid that cannot support the ownership move.

    THE MAGNETIC CLAUSE IS NOW TWO CLAUSES, 2026-08-30. This family gained a
    ``launch.FUSED_PAIR_ARMS`` row, so ``_install_fused_pair`` brackets its launch and
    ``CARRIES_DEPOSIT_REPAIR`` went True in the same edit; before that row existed
    nothing bracketed anything here and an in-seam magnetic source was barred outright,
    which is what this leg used to require. What replaces it is BOTH directions of the
    flag, because either alone is satisfied by a predicate that is wrong in the other:

    * a REAL magnetic deposit, one that publishes the index it writes, must be
      ADMITTED -- otherwise the flag is inert and the row bought nothing;
    * a magnetic source that publishes NO index must still be refused BY NAME -- a
      closure cannot carry what a source will not name, and admitting it would put a
      fused launch on a deposit nothing can restore.
    """
    far_fields, far_pml = build(REFUSAL_FAR_CASE, seed_for("refusal", "far"))
    residency = Residency()
    far = family.metal_folded_complex_fused_magnetic_pair_coverage(
        far_fields, far_pml, (), residency)
    far_plan = family.plan_metal_folded_complex_fused_magnetic_pair(
        far_fields, far_pml, sources=(), residency=residency)
    live = live_passes(far_fields, far_pml)
    far_codes, _ = folded_axis_kinds(far_fields.grid, far_pml)
    far_rows = stepping._far_reflect_rows(far_fields.grid)
    expected_far = [list(family.far_fill_axes(far_codes, target))
                    for target in range(3)]
    carried_ok = bool(
        far.covered and far_plan is not None
        and "fill_folded_far_ghosts_B" in live
        and "fill_folded_far_ghosts_B" in family.REPLACES
        and [list(axes) for axes in far_plan.far_carried_axes] == expected_far
        and any(expected_far)
        and list(far_plan.reflect) == [None if r is None else int(r)
                                       for r in far_rows])

    # The reflect-row clauses, on stubs that violate one requirement each.
    periodic_codes = (CODE_PERIODIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC)
    stubs: Dict[str, Any] = {}
    # (a) reflect row past the top plane: n_full - stored + 2 == stored - 1.
    stubs["reflect_row_is_the_plane_it_writes"] = {
        "reasons": family._far_carry_reasons(
            _StubGrid((8, 8, 8), (8, 13, 8), (False, True, False)), periodic_codes),
        "wanted": "outside [0,"}
    # (b) reflect row 0, the plane the NEAR fill writes.
    stubs["reflect_row_is_the_near_fills_destination"] = {
        "reasons": family._far_carry_reasons(
            _StubGrid((8, 8, 8), (8, 6, 8), (False, True, False)), periodic_codes),
        "wanted": "reflect row is 0"}
    # (c) the code and stepping._stored_past_owned disagree: the kernel would mask the
    #     top plane and image the far ghost on a different axis set than the array
    #     path. Declared PERIODIC, grid says the axis is not folded at all.
    stubs["code_and_stored_past_owned_disagree"] = {
        "reasons": family._far_carry_reasons(
            _StubGrid((8, 8, 8), (8, 14, 8), (False, False, False)), periodic_codes),
        "wanted": "disagree"}
    for entry in stubs.values():
        entry["named"] = [reason for reason in entry["reasons"]
                          if entry["wanted"] in reason]

    keywords, faces = cases[MUTATION_CASE]
    fields, pml = build(keywords, seed_for("refusal", "sources"), faces)
    # THE ADMISSION, since 2026-08-30. This family gained a `launch.FUSED_PAIR_ARMS`
    # row, so `_install_fused_pair` now brackets its launch with
    # `LeadingRepairPlan`/`TrailingRepairPlan` and `CARRIES_DEPOSIT_REPAIR` went True
    # in the same edit. A REAL magnetic deposit -- one that publishes the index it
    # writes -- is therefore COVERED, because the repair saves it before the launch and
    # restores it after the post-injection fills have imaged it.
    carried_deposit = family.metal_folded_complex_fused_magnetic_pair_coverage(
        fields, pml, (_carryable_magnetic_source(fields),), Residency())
    # THE REFUSAL THAT SURVIVES IT, and why the leg asks both. `_MagneticSource`
    # declares a field type and nothing else, so it names no cell for the repair to
    # save; a closure cannot carry what a source will not name, and that is refused BY
    # NAME rather than admitted on the strength of a flag. Asking only the admission
    # would pass on a predicate that admitted everything; asking only the refusal would
    # pass on one that admitted nothing and left the flag inert.
    magnetic = family.metal_folded_complex_fused_magnetic_pair_coverage(
        fields, pml, (_MagneticSource(),), Residency())
    magnetic_named = [reason for reason in magnetic.reasons
                      if "does not publish the index it writes" in reason]
    electric = family.metal_folded_complex_fused_magnetic_pair_coverage(
        fields, pml, (_ElectricSource(),), Residency())
    undeclared = family.metal_folded_complex_fused_magnetic_pair_coverage(
        fields, pml, None, Residency())
    undeclared_named = [reason for reason in undeclared.reasons
                        if "was not declared" in reason]

    # THE TWO SIBLING SCOPES. An unfolded complex grid and a real folded one.
    unfolded_fields, unfolded_pml = matrix.cart(complex_storage=True)
    unfolded = family.metal_folded_complex_fused_magnetic_pair_coverage(
        unfolded_fields, unfolded_pml, (), Residency())
    real_fields, real_pml = matrix.folded(boundaries={"y": "metallic"}, depth=1.2)
    real = family.metal_folded_complex_fused_magnetic_pair_coverage(
        real_fields, real_pml, (), Residency())

    return {
        "passed": bool(carried_ok
                       and all(entry["named"] for entry in stubs.values())
                       and carried_deposit.covered
                       and not magnetic.covered and magnetic_named
                       and electric.covered
                       and not undeclared.covered and undeclared_named
                       and not unfolded.covered and not real.covered
                       and family.plan_metal_folded_complex_fused_magnetic_pair(
                           fields, pml, None, Residency()) is None),
        "far_fill_covered": far.covered,
        "far_fill_plan_built": far_plan is not None,
        "far_fill_refusals_if_any": list(far.reasons),
        "far_pass_is_live": "fill_folded_far_ghosts_B" in live,
        "far_pass_is_in_replaces": "fill_folded_far_ghosts_B" in family.REPLACES,
        "replaces": list(family.REPLACES),
        "carried_far_ghost_axes": ([list(a) for a in far_plan.far_carried_axes]
                                   if far_plan is not None else None),
        "carried_far_ghost_axes_expected": expected_far,
        "reflect_rows": (list(far_plan.reflect) if far_plan is not None else None),
        "reflect_rows_expected": [None if r is None else int(r) for r in far_rows],
        "far_boundary_codes": list(far_codes) if far_codes is not None else None,
        "stub_grid_refusals": {name: {"named": entry["named"],
                                      "all": entry["reasons"]}
                               for name, entry in stubs.items()},
        "real_deposit_admitted": carried_deposit.covered,
        "real_deposit_refusals_if_any": list(carried_deposit.reasons),
        "magnetic_source_covered": magnetic.covered,
        "magnetic_named_refusals": magnetic_named,
        "electric_source_covered": electric.covered,
        "undeclared_sources_covered": undeclared.covered,
        "undeclared_named_refusals": undeclared_named,
        "unfolded_complex_covered": unfolded.covered,
        "unfolded_complex_reasons": list(unfolded.reasons),
        "real_folded_covered": real.covered,
        "real_folded_reasons": list(real.reasons),
        "live_passes": list(live),
    }


def coefficient_census(pml: Any, codes: Sequence[int]) -> Dict[str, Any]:
    """Does the moved index CHANGE the coefficient, per folded axis?

    The imaged ghost sits at stored index 0 of the folded axis and its source at
    ``NEAR_SOURCE_INDEX``. This family reads ``kps``/``kms`` at 0; reusing the source's
    entry is only a DIFFERENT NUMBER where the two entries differ, and on a folded axis
    they usually do not — the mirror plane carries no absorber
    (``stepping._require_consistent_pml`` admits the high face only), so the layer has
    to be deep enough to reach stored cell 2 before the distinction exists.
    """
    out: Dict[str, Any] = {}
    for axis, name in enumerate("xyz"):
        if int(codes[axis]) not in MIRROR_CODES:
            continue
        entry: Dict[str, Any] = {}
        for stem in ("kps", "kms"):
            vector = np.asarray(getattr(pml, f"{stem}_{name}")).ravel()
            entry[stem] = [float(vector[0]),
                           float(vector[family.NEAR_SOURCE_INDEX])]
        entry["differs"] = bool(entry["kps"][0] != entry["kps"][1]
                                or entry["kms"][0] != entry["kms"][1])
        out[name] = entry
    return out


def leg_moved_coefficient(cases: Mapping[str, Tuple[Dict[str, Any], Optional[Any]]],
                          steps: int) -> Dict[str, Any]:
    """PREDICT observability from the coefficients, then measure the catch.

    THE LEG THIS FAMILY EXISTS TO HAVE. Its one line that
    :mod:`.folded_fused_pair` does not carry is the ghost's own coefficient pair, and a
    mutation leg that simply armed the reuse would have reported it UNCAUGHT on every
    ordinary fixture and left a reader to guess whether the line is wrong or the
    fixture is blind. So the prediction is made first, from the absorber vectors, and
    the requirement is agreement: caught exactly where the two coefficient entries
    differ, not caught where they are the same word.

    The mutation edits COMPONENT 0's ghost, so the prediction is read off the x axis;
    the census reports every folded axis so a reader can see the whole picture.
    """
    edits: Dict[str, str] = {}
    predictions: Dict[str, Any] = {}
    for label in (MUTATION_CASE, DEEP_CASE):
        keywords, faces = cases[label]
        fields, pml = build(keywords, seed_for("moved", label, "build"), faces)
        spec = _specialisation(fields, pml)
        census = coefficient_census(pml, spec["codes"])
        source = family.folded_complex_fused_magnetic_pair_source(
            spec["codes"], spec["phased"], spec["walls"], spec["expansion"])
        edits[label] = needle(source, *MOVED_COEFFICIENT_EDIT)
        predictions[label] = {
            "census": census,
            "shape": [int(n) for n in fields.grid.shape],
            "pml_faces": [list(face) for face in pml.thickness_by_face],
            "observable": bool(census.get("x", {}).get("differs", False)),
        }
    compiled = compile_edits(edits)
    rows: Dict[str, Any] = {}
    ok = True
    for label, functions in compiled.items():
        keywords, faces = cases[label]
        result = run_case(keywords, seed_for("moved", label, "run"), steps, faces,
                          functions=functions)
        caught = not result.get("bit_identical", False)
        agree = bool(caught) == bool(predictions[label]["observable"])
        ok = ok and agree and bool(result.get("launches"))
        rows[label] = {**predictions[label], "caught": caught,
                       "prediction_holds": agree,
                       "first_divergence": result.get("first_divergence"),
                       "differing_words": result.get("differing_words"),
                       "differing_arrays": result.get("differing_arrays"),
                       "launches": result.get("launches")}
    return {"passed": bool(ok), "cases": rows,
            "note": "the imaged ghost reads kps/kms at stored index 0; reusing the "
                    "source thread's entry at index 2 is a different number only "
                    "where the folded axis's absorber reaches stored cell 2"}


def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['leg']}] {row['label']}: "
        f"passed={row.get('passed')} diff={row.get('differing_words', '-')} "
        f"launches={row.get('launches', '-')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")

    matrix.prepare_environment()
    cases = {name: (keywords, faces) for name, keywords, faces in CASES}

    # THE EXPANSION ARM FIRST. Every later leg emits a source, and a source emitted
    # before the arm is bound would carry a guessed complex multiply.
    expansion_row = leg_expansion()
    if not expansion_row["passed"]:
        log("the expansion arm is unbound; the gate cannot certify")
    expansion = EXPANSION or "FMA_V1"

    # THE TWO MUTATION CASES' SPECIALISATIONS, CHECKED BEFORE ANY NEEDLE IS APPLIED.
    # This is the dead-branch guard: a needle whose line sits under a feature the case
    # does not carry would report UNCAUGHT while measuring nothing.
    fold_keywords, fold_faces = cases[MUTATION_CASE]
    fold_fields, fold_pml = build(fold_keywords, seed_for("arm", MUTATION_CASE),
                                  fold_faces)
    fold_spec = _specialisation(fold_fields, fold_pml)
    assert any(fold_spec["walls"]), (
        f"{MUTATION_CASE} has no walled axis; the two wall mutations would be vacuous")
    folded_axes = [axis for axis, code in enumerate(fold_spec["codes"])
                   if int(code) in MIRROR_CODES]
    assert len(folded_axes) >= 2, (
        f"{MUTATION_CASE} folds fewer than two axes; the wrong-axis parity mutation "
        f"would have no second word to take")
    assert set(fold_spec["phases"][axis] for axis in folded_axes) == {1, -1}, (
        f"{MUTATION_CASE} does not carry BOTH parities: {fold_spec['phases']}")
    assert not any(fold_spec["phased"]), (
        f"{MUTATION_CASE} carries a phased axis; the phase mutations belong on "
        f"{PHASE_MUTATION_CASE} and this assertion is what keeps the split honest")

    phase_keywords, phase_faces = cases[PHASE_MUTATION_CASE]
    phase_fields, phase_pml = build(phase_keywords,
                                    seed_for("arm", PHASE_MUTATION_CASE), phase_faces)
    phase_spec = _specialisation(phase_fields, phase_pml)
    assert sum(phase_spec["phased"]) >= 2, (
        f"{PHASE_MUTATION_CASE} phases fewer than two axes: {phase_spec['phased']}; "
        f"the rotation mutations would rewrite absent text")
    assert any(int(code) in MIRROR_CODES for code in phase_spec["codes"]), (
        f"{PHASE_MUTATION_CASE} is not folded; it would not be this family at all")

    # THE FAR MUTATION CASE, checked the same way. Its needles all sit on lines the
    # kernel emits ONLY for a folded PERIODIC axis, so the fold split is the dead-branch
    # guard: two MIRROR_PERIODIC axes with MIXED parities at an ODD full count, and a
    # component whose OWN axis is unfolded so that a FAR/FAR corner exists at all.
    far_keywords, far_faces = cases[FAR_MUTATION_CASE]
    far_fields, far_pml = build(far_keywords, seed_for("arm", FAR_MUTATION_CASE),
                                far_faces)
    far_spec = _specialisation(far_fields, far_pml)
    far_periodic = [axis for axis, code in enumerate(far_spec["codes"])
                    if int(code) == CODE_MIRROR_PERIODIC]
    assert len(far_periodic) >= 2, (
        f"{FAR_MUTATION_CASE} folds fewer than two PERIODIC axes "
        f"({far_spec['codes']}); the parity-chain mutations would have no chain")
    assert set(far_spec["phases"][axis] for axis in far_periodic) == {1, -1}, (
        f"{FAR_MUTATION_CASE} does not carry BOTH parities: {far_spec['phases']}")
    assert any(far_spec["walls"]), (
        f"{FAR_MUTATION_CASE} has no walled axis; the far ghost's inheritance of the "
        f"wall clear would never be exercised")
    # THE REFLECT ROW, ON THE AXIS THE NEEDLE ACTUALLY REWRITES. An `any` over the
    # three axes passed here while `reflect_y` was itself the substituted constant,
    # and the mutation scored 12 launches at 0 differing words. A precondition that
    # quantifies over axes the needle does not touch measures a different claim from
    # the one the leg makes, so this names the axis.
    far_reflect = stepping._far_reflect_rows(far_fields.grid)
    row_y = far_reflect[REFLECT_MUTATION_AXIS]
    fixed = int(far_fields.grid.shape[REFLECT_MUTATION_AXIS]) - 2
    assert row_y is not None and int(row_y) != fixed, (
        f"{FAR_MUTATION_CASE} has reflect row {row_y} on axis "
        f"{REFLECT_MUTATION_AXIS} of shape {tuple(far_fields.grid.shape)}, which IS "
        f"the fixed n - 2 = {fixed} the reflect_row_is_the_fixed_n_minus_two needle "
        f"substitutes; the mutation would rewrite the plane to itself")

    # THE FAR PARITY WORD ON THE AXIS `far_parity_dropped` REWRITES. `mirror_parity`
    # is `phase * (1 - 2*iyee)`, so the FAR word is MINUS the plane's phase and an ODD
    # plane hands the far pass an IDENTITY. Dropping an identity is not a defect, and
    # asserting the plane carries both parities does NOT establish this — the needle
    # is on one axis and it is the far word, not the phase, that has to be non-unity.
    far_words = family.far_parity_words(far_spec["codes"], far_spec["phases"])
    dropped = far_words[PARITY_DROP_AXIS]
    assert (float(dropped[0]), float(dropped[1])) != (1.0, 0.0), (
        f"{FAR_MUTATION_CASE} far parity words are {far_words}; the word on axis "
        f"{PARITY_DROP_AXIS} that far_parity_dropped removes is the identity, so the "
        f"mutation is arithmetically a no-op and would score against nothing")
    assert any(len(family.far_fill_axes(far_spec["codes"], target)) == 2
               for target in range(3)), (
        f"{FAR_MUTATION_CASE} gives no component two far axes; the FAR/FAR composite "
        f"and its two-parity chain would be absent")

    fold_base = family.folded_complex_fused_magnetic_pair_source(
        fold_spec["codes"], fold_spec["phased"], fold_spec["walls"], expansion)
    phase_base = family.folded_complex_fused_magnetic_pair_source(
        phase_spec["codes"], phase_spec["phased"], phase_spec["walls"], expansion)
    far_base = family.folded_complex_fused_magnetic_pair_source(
        far_spec["codes"], far_spec["phased"], far_spec["walls"], expansion)
    fold_mutants = compile_edits(fold_edits(fold_base))
    phase_mutants = compile_edits(phase_edits(phase_base))
    far_mutants = compile_edits(far_edits(far_base))
    null_bases = {MUTATION_CASE: fold_base, FAR_MUTATION_CASE: far_base}
    null_mutants = compile_edits(
        {name: needle(null_bases[case], old, new)
         for name, (case, old, new, _why) in NULL_EDITS.items()})

    controls = ("y_metallic_wall_z_3d", "xy_mixed_wall_z_3d",
                FAR_MUTATION_CASE)
    total = (1 + 1 + 1 + 1 + 1 + len(CASES) + 1 + len(controls) + 1 + 3 + 1
             + len(fold_mutants) + len(phase_mutants) + len(far_mutants)
             + len(null_mutants) + len(HOST_MUTATIONS) + 1 + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        index += 1
        rows.append({"index": index, "total": total, "leg": "policy",
                     "label": "flush_resolved_and_keep_refused_by_name",
                     **leg_policy()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "expansion",
                     "label": "the_probe_binds_the_arm", **expansion_row})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "binding_ceiling",
                     "label": "twenty_seven_pointers_plus_one_packed_struct",
                     **leg_binding_ceiling()})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "transcription",
                     "label": "all_three_pieces_are_the_certified_text",
                     **leg_transcription(fold_spec["codes"], fold_spec["phased"],
                                         expansion)})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "parity_chain_order",
                     "label": "the_chain_is_ordered_and_neither_collapse_is_free",
                     **leg_parity_chain_order()})
        emit(handle, rows[-1])

        for name, keywords, faces in CASES:
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": name,
                   "steps": args.steps,
                   **run_case(keywords, seed_for("product", name), args.steps, faces)}
            rows.append(row)
            emit(handle, row)

        # THE OFF-DIAGONAL FLOOR. Two of the product cases exist to carry an
        # off-diagonal chi1inv row — the shape the only two corpus rows this product
        # serves actually have. A `rows=` keyword that stopped installing one would
        # make those cases SILENT DUPLICATES of the diagonal rows beside them and
        # this gate would report eleven configurations while measuring nine. The
        # floor reads the flag off the rows that already ran rather than rebuilding.
        index += 1
        by_case = {row["label"]: row for row in rows if row["leg"] == "product"}
        offdiagonal = sorted(name for name, row in by_case.items()
                             if row.get("has_offdiagonal_epsilon"))
        # THREE since the far carry: `y_periodic_offdiag_3d` is the periodic twin
        # added so the off-diagonal row and a folded PERIODIC axis are exercised
        # together rather than one at a time.
        expected = ["xy_mixed_offdiag_wall_z_3d", "y_metallic_offdiag_3d",
                    "y_periodic_offdiag_3d"]
        rows.append({"index": index, "total": total, "leg": "value_classes",
                     "label": "the_off_diagonal_cases_really_are_off_diagonal",
                     "cases_with_an_offdiagonal_row": offdiagonal,
                     "expected": expected,
                     "corpus_rows_this_product_serves": [
                         "examples:solve-cw.py",
                         "tests:TestArrayMetadata.test_array_metadata"],
                     "passed": offdiagonal == expected,
                     "note": "both corpus rows carry an off-diagonal chi1inv row; "
                             "update_H reads B and mu only, which is why this family "
                             "drops the certified complex constitutive's three "
                             "inverse-epsilon volumes and binds 27 pointers"})
        emit(handle, rows[-1])

        for name in controls:
            index += 1
            keywords, faces = cases[name]
            row = {"index": index, "total": total, "leg": "separate_control",
                   "label": name, "steps": args.steps,
                   **run_separate_control(keywords, seed_for("separate", name),
                                          args.steps, faces)}
            rows.append(row)
            emit(handle, row)

        index += 1
        rows.append({"index": index, "total": total, "leg": "refusal",
                     "label": "far_fill_sources_and_the_two_sibling_scopes",
                     **leg_refusal(cases)})
        emit(handle, rows[-1])

        # VALUE CLASSES. The first two must be bit-identical; the third must FIRE the
        # subnormal census, which is what stops the precondition being vacuous. Its
        # `passed` is therefore the census being NONZERO, and identity is NOT claimed.
        for value_class in ("uniform", "signed_zero_lattice"):
            index += 1
            row = {"index": index, "total": total, "leg": "value_classes",
                   "label": value_class, "steps": args.steps,
                   **run_case(fold_keywords, seed_for("value", value_class),
                              args.steps, fold_faces, value_class=value_class,
                              require_every_volume_moved=(
                                  value_class != "signed_zero_lattice"))}
            rows.append(row)
            emit(handle, row)
        band = run_case(fold_keywords, seed_for("value", "subnormal_band"),
                        args.steps, fold_faces, value_class="subnormal_band")
        index += 1
        rows.append({"index": index, "total": total, "leg": "value_classes",
                     "label": "subnormal_band_precondition_fires",
                     "steps": args.steps,
                     "reference_subnormals": band.get("reference_subnormals"),
                     "bit_identical": band.get("bit_identical"),
                     "passed": bool(band.get("reference_subnormals")),
                     "note": "the precondition must FIRE on a band seed; identity is "
                             "NOT claimed here and the census is the measurement"})
        emit(handle, rows[-1])

        index += 1
        rows.append({"index": index, "total": total, "leg": "moved_coefficient",
                     "label": "predicted_observability_vs_measured_catch",
                     **leg_moved_coefficient(cases, args.steps)})
        emit(handle, rows[-1])

        for label, mutants, case_name in (
                ("fold", fold_mutants, MUTATION_CASE),
                ("phase", phase_mutants, PHASE_MUTATION_CASE),
                ("far", far_mutants, FAR_MUTATION_CASE)):
            keywords, faces = cases[case_name]
            for name, functions in mutants.items():
                index += 1
                result = run_case(keywords, seed_for("mutation", name), args.steps,
                                  faces, functions=functions)
                caught = not result.get("bit_identical", False)
                row = {"index": index, "total": total, "leg": "mutation",
                       "label": name, "group": label, "case": case_name,
                       "caught": caught,
                       "passed": bool(caught and result.get("launches")),
                       "first_divergence": result.get("first_divergence"),
                       "differing_words": result.get("differing_words"),
                       "differing_arrays": result.get("differing_arrays"),
                       "launches": result.get("launches")}
                rows.append(row)
                emit(handle, row)

        for name, functions in null_mutants.items():
            index += 1
            null_case = NULL_EDITS[name][0]
            null_keywords, null_faces = cases[null_case]
            result = run_case(null_keywords, seed_for("null", name), args.steps,
                              null_faces, functions=functions)
            diverged = not result.get("bit_identical", False)
            # THE DEAD-BRANCH GUARD FOR A NULL. A needle that changed a line the
            # scored case never reaches reports "no divergence" while measuring
            # nothing. Every null here names sibling needles on the SAME statement
            # that are required to be CAUGHT, and this reads their verdicts off the
            # rows already collected rather than asserting the line is live.
            caught_by = {row["label"]: row.get("caught")
                         for row in rows if row["leg"] == "mutation"}
            witnesses = SIBLING_WITNESS[name]
            missing = [w for w in witnesses if w not in caught_by]
            if missing:
                raise AssertionError(
                    f"{name}'s sibling witnesses {missing} are not among the "
                    f"mutations this gate ran; a name that matches nothing disables "
                    f"the check it feeds")
            witnesses_caught = all(bool(caught_by[w]) for w in witnesses)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "group": "null", "case": null_case,
                   "expected": "NULL", "diverged": diverged,
                   "verdict": "DIVERGED" if diverged else "NULL CONFIRMED",
                   "passed": bool(not diverged and result.get("launches")
                                  and witnesses_caught),
                   "why": NULL_EDITS[name][3],
                   "sibling_witnesses": list(SIBLING_WITNESS[name]),
                   "sibling_witnesses_caught": witnesses_caught,
                   "differing_words": result.get("differing_words"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        for name, patch in HOST_MUTATIONS.items():
            index += 1
            result = run_case(fold_keywords, seed_for("host", name), args.steps,
                              fold_faces, patch=patch)
            caught = not result.get("bit_identical", False)
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "case": MUTATION_CASE, "host_defect": True,
                   "caught": caught,
                   "passed": bool(caught and result.get("launches")),
                   "first_divergence": result.get("first_divergence"),
                   "differing_words": result.get("differing_words"),
                   "differing_arrays": result.get("differing_arrays"),
                   "launches": result.get("launches")}
            rows.append(row)
            emit(handle, row)

        # THE DISARM CHECK. The identical harness, the identical case, the SHIPPED
        # bytes and no host patch. A nonzero here would mean every "caught" above is a
        # harness that diverges on its own.
        index += 1
        result = run_case(fold_keywords, seed_for("disarm"), args.steps, fold_faces)
        row = {"index": index, "total": total, "leg": "disarm",
               "label": "shipped_bytes_on_the_mutation_case",
               "differing_words": result.get("differing_words"),
               "launches": result.get("launches"),
               "arrays_that_never_moved": result.get("arrays_that_never_moved"),
               "passed": bool(result.get("passed"))}
        rows.append(row)
        emit(handle, row)

        # THE VERDICT ITSELF, SHOWN TO FLIP. Every leg above can pass while the
        # release rule that reads them is wrong. This row recomputes the verdict over
        # the rows collected so far WITH ONE PLANTED FAILURE and requires it to be
        # FAIL — so the arbiter is measured, not assumed.
        index += 1
        planted = rows + [{"index": -1, "total": total, "leg": "planted",
                           "label": "a_deliberately_failing_row", "passed": False}]
        verdict_without = "PASS" if all(r["passed"] for r in rows) else "FAIL"
        verdict_with = "PASS" if all(r["passed"] for r in planted) else "FAIL"
        row = {"index": index, "total": total, "leg": "planted_defect",
               "label": "the_release_verdict_flips",
               "verdict_without_the_plant": verdict_without,
               "verdict_with_the_plant": verdict_with,
               "passed": bool(verdict_with == "FAIL")}
        rows.append(row)
        emit(handle, row)

    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "counts": {"product": len(CASES), "separate_controls": len(controls),
                   "fold_mutations": len(fold_mutants),
                   "phase_mutations": len(phase_mutants),
                   "null_mutations": len(null_mutants),
                   "host_mutations": len(HOST_MUTATIONS)},
        "mutation_case": MUTATION_CASE,
        "phase_mutation_case": PHASE_MUTATION_CASE,
        "deep_absorber_case": DEEP_CASE,
        "base_seed": SEED,
        "seed_rule": "SEED + sha256(case identity)[:4]; never hash()",
        "mutation_specialisation": {
            "fold": {key: list(value) if isinstance(value, tuple) else value
                     for key, value in fold_spec.items()},
            "phase": {key: list(value) if isinstance(value, tuple) else value
                      for key, value in phase_spec.items()},
        },
        "expansion": expansion,
        "expansion_probe": PROBE_ARTIFACT,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "subnormal_policy_report": subnormal.mps_policy_report(),
        "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    log(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
        f"artifact {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
