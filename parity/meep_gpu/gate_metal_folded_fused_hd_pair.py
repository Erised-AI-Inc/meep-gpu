#!/usr/bin/env python3
"""Native MPS byte gate for the FOLDED Metal H->D weld: folded ``update_H`` into
folded ``step_D``.

THE SECOND PRODUCT ON THE FOURTH SEAM, and the largest cell on this backend: the
``(folded -> folded)`` arm pair is **78 corpus rows** against the plain product's
49. ``meep_gpu/metal_kernels/folded_fused_hd_pair.py`` computes the folded
``update_H`` constitutive into launch-local SCRATCH, takes its own cell's magnetic
field from registers, RECOMPUTES every one of the folded curl's six foreign taps from
pre-launch state, steps ``D``/``fu_D`` in place, and rotates the ``H``/``f_w_H``
bindings afterwards.

THE CLAIM IS PER COMPLETE DRIVER STEP -- ``FdtdDriver.step``'s own consult order,
with the MIRROR FILLS, the wall clear and the FAR GHOST PASSES exactly where the
driver runs them -- over a stated budget, as uint32 WORDS over every stored volume
the engine allocates (never ``allclose``: ``-0.0 == 0.0`` lies), against FOUR
reference engines from one seed:

  1. the ARRAY PATH -- ``stepping``'s passes under the driver's own loop;
  2. the CERTIFIED FOLDED SINGLES -- ``symmetry.plan_folded_constitutive(..., "H")``
     at ``update_H`` and ``symmetry.plan_folded_pml_curl(..., "step_D")`` at
     ``step_D``, two dispatches on the seam;
  3. the COMPOSITION THE COMPOSER INSTALLS TODAY -- ``plan_step(fuse=True)``, which
     on every row this product reaches puts the released ``folded_fused_magnetic_pair``
     across ``step_B``/``update_H``. This is the reference the arbitration finding
     rests on, and its slot table is recorded per case;
  4. the same slots DISPATCHED UNFUSED -- ``plan_step(fuse=False)``.

=============================================================================
WHAT IS FOLD-SPECIFIC HERE, AND WHY EACH LEG EXISTS
=============================================================================

**The masks are INVISIBLE at whole-step granularity.** ``symmetry.py:55-60`` states
it and ``gate_metal_symmetry`` measured it: the driver's fill passes overwrite exactly
the planes the cell-0 mask and the top-plane mask protect, so a kernel carrying
NEITHER mask is bytewise-identical after a complete step. A whole-step gate therefore
CERTIFIES A MASK-LESS KERNEL AS CORRECT. Leg ``sub_step`` compares ``D``/``fu_D``
IMMEDIATELY after the fused launch -- before ``fill_D``, ``zero_metal_D`` and the far
D pass -- against the two certified singles at the same point, and the two mask
mutations are required to fire THERE. Their whole-step verdict is recorded as a
PREDICTED NULL with its citation, never scored as an uncaught defect.

**The codes are the one thing this family can get wrong silently.** The plain product
builds its triple as ``1 if kind == "metallic" else 0`` over ``_boundary_kinds``,
which on a folded grid reports ``"mirror"`` and maps to 0 = PERIODIC: the ghost
becomes a WRAP to the far plane and the cell-0 mask is not emitted. Neither
``folded_curl_source`` nor the compiler can catch it (0 and 1 are valid codes). Leg
``codes_source`` builds the plan with exactly that expression and requires divergence
on every folded fixture, with the shipped ``folded_axis_kinds`` codes as the control
that must not diverge.

**The product must be PHASE-BLIND.** This kernel performs no fill and its ghost is a
literal zero, so the mirror parity never reaches it; parity lives in the fill kernels,
outside this seam. Leg ``phase_blind`` asserts the emitted source for a ``+1`` plane
and a ``-1`` plane is byte-identical by sha256 AND drives the odd-plane fixture to
byte identity against the array path with the odd fill live.

**Two ghost cases the source settles and this gate measures.** On a folded axis the
backward ghost at index -1 is the METALLIC literal ``0.0f``, not the array path's
``parity * H[2]`` -- legal because that value is DEAD under a mask
(``symmetry.py:30-43``). The mutation ``ghost_zero_replaced_by_the_mirror_source``
serves ``h_cell(2, j, k)`` there and is a PREDICTED NULL; it is paired with the same
edit PLUS ``drop_cell0_mask``, which MUST fire, so the null is EARNED rather than
assumed. And stored cell 0 of a folded axis IS read, by the thread at index 1, as a
real ``update_H`` output over a ``B`` the driver's near fill wrote before the seam --
leg ``purity`` counts how many foreign taps land there and how many of THOSE carry a
value ``update_H`` moved, because a near-zero count would mean the race legs license
nothing on that fixture.

=============================================================================
THE FIVE THINGS THIS GATE REFUSES TO LET PASS SILENTLY
=============================================================================

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: an unlaunched
   plan is byte-identical to the oracle BY CONSTRUCTION, because the oracle is the
   array path. Every case asserts the exact launch count from TWO independent
   witnesses -- the plans' own counters and a wrapper around every compiled function
   -- plus ``dispatched["update_H"] == steps`` and ``absorbed["step_D"] == steps``,
   and requires every compared array to have MOVED from its seed.
2. **A hollow pass.** Every armed defect must be CAUGHT, and leg ``disarm`` reruns the
   identical harness with the shipped bytes and requires zero divergence, so a
   mutation reported caught cannot be a harness that diverges anyway. Leg
   ``byte_neutral`` is the complement: the one armed edit required NOT to diverge.
3. **A vacuous claim.** On MPS the float32 subnormal flush is native and has no lever,
   so byte identity is claimed subject to a CHECKED subnormal-free precondition, taken
   on the oracle's own state BEFORE each comparison; a banded step is stepped, named
   and NOT compared.
4. **An unmeasured platform assumption.** Leg ``binding_ceiling`` compiles the three
   refuted signatures (35 separate scalars, 34 unshared ``kms``, 32 one-more-pointer)
   and requires each to FAIL, BISECTS the ceiling on this host (30 pointers + one
   packed struct compiles, 31 + struct does not), compiles the shipped 31 on EVERY
   fixture's own folded codes, and LAUNCHES the packed probe reading back every
   ``Params`` field. Nothing here is counted from a signature.
5. **A refusal that is really an omission.** Leg ``refusal`` names each one and drives
   the two admissions that are not this seam's business, including the INVERTED fold
   clause in both directions: an unfolded grid must be refused naming ``fused_hd_pair``,
   and ``fused_hd_pair`` must refuse the folded fixture naming this cell.

LEGS
  host
  driver_order     REPLACES is the driver's two adjacent consults; the only statement
                   between them is the electric withdraw loop; all six fold passes sit
                   OUTSIDE the span -- read off the tree, never spelled
  transcription    the emitted source differs from ``symmetry.folded_curl_source`` +
                   ``shaders.constitutive_source('H')`` in EXACTLY the declared edits;
                   the top-plane and cell-0 mask lines are character-identical to the
                   folded emitter's; on (P,P,P) the source is the plain product's plus
                   one dead ``last_*`` line; every mutation needle resolves once
  fixture_shape    the drivers this gate builds carry the SAME grid shape, boundary
                   codes and PML thickness as ``metal_composition_matrix.folded``'s
                   own, including the Z-fold rows no standing gate exercises
  purity           the fold-specific tap ledger (below)
  phase_blind      the source is byte-identical at both parities, by sha256
  refusal          each refusal by name, in both directions on the fold clause
  compile
  binding_ceiling  three refuted signatures FAIL, the ceiling is bisected, the shipped
                   31 COMPILES on every fixture's codes, the packed probe LAUNCHES
  mutants_compile  every armed defect and the byte-neutral control COMPILE
  device
  product          the four-reference identity on nine folded fixtures, complete driver
                   steps, movement floor, launch counters, subnormal census
  codes_source     the _boundary_kinds triple must DIVERGE on every folded fixture
  sub_step         D/fu_D immediately after the fused launch vs the two singles
  race             H and f_w_H bound IN PLACE must diverge, on every fixture
  rotation         the post-launch rotation dropped must diverge
  arbitration      the composer, asked, with this product REGISTERED: zero installed,
                   and folded_fused_magnetic_pair still installed on every fixture
  sync             the product force-installed with flux_in_box driven mid-run
  launch_structure launches per step at the seam and over the whole step
  lift             every corpus row the standing census puts in this cell, re-lifted in
                   its own interpreter and driven
  byte_neutral     the one armed edit required NOT to diverge
  mutation         every armed shader and host defect, each of which MUST diverge;
                   predicted nulls recorded with their citation, never dropped
  disarm           the identical harness, shipped bytes, must not diverge

THE GENERIC MACHINERY IS IMPORTED FROM ``gate_metal_fused_hd_pair`` RATHER THAN
COPIED, and that is a correctness decision rather than a size one. The driver walk,
the capture/restore/compare mechanism, the shim that answers the engine's own dispatch
seam, the two launch counters, the rotation settling, the orphaned-mirror guard, the
subnormal and intermediate censuses and the two host-defect launchers are the SAME
mechanism this product needs, and they carry defects already measured and repaired
(the 2026-09-05 ``cavity_arrayslice`` orphaned mirror, the doubled function counter,
the stale ``id(plan)`` PML lookup). A second copy would be a second place for each of
those to come back. What this file owns is everything FOLDED: the fixtures, the folded
arrangements, the fold-specific legs and the fold-specific defects.

Rule 7: one flushed line per case; every row appended and fsynced as it lands; the
lift leg writes one JSON per corpus row as it lands and a progress log the child
appends to per step.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# THE POLICY, SET BEFORE ANY meep_gpu MODULE IS REACHED. `flush` is the only value the
# MPS executor can honour; a resolved `keep` refuses every predicate by name.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
# BY NAME, never by parents[N]: a moved harness resolving a wrong root measures nothing.
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import h_to_d_seam  # noqa: E402
import metal_composition_matrix as matrix  # noqa: E402

# THE SIBLING GATE AS A LIBRARY. See the module docstring: the walk, the shim, the two
# counters and the two host-defect launchers are the same mechanism, and one copy of
# them is one place for the defects already found in them to stay fixed.
import gate_metal_fused_hd_pair as _shared  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import SYNC_PASS_OWNERS, SYNC_PATH_SLOTS, SYNC_UPDATE_H_PASS  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    folded_fused_hd_pair as family, fused_hd_pair as plain,
    launch as metal_launch, shaders, subnormal, symmetry, templates,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.symmetry import (  # noqa: E402
    CODE_METALLIC as M, CODE_MIRROR_METALLIC as MM, CODE_MIRROR_PERIODIC as MP,
    CODE_PERIODIC as P,
)

# The generic machinery, bound to short names so the folded legs read as their own.
ABSORBED = _shared.ABSORBED
Arrangement = _shared.Arrangement
AliasedLaunch = _shared.AliasedLaunch
RotationSkipped = _shared.RotationSkipped
Shim = _shared.Shim
SyncedPlan = _shared.SyncedPlan
capture = _shared.capture
compare_snapshots = _shared.compare_snapshots
declaring = _shared.declaring
differing = _shared.differing
drive = _shared.drive
install = _shared.install
needle = _shared.needle
pin_array_path = _shared.pin_array_path
rebind_static = _shared.rebind_static
restore = _shared.restore
stored_volumes = _shared.stored_volumes
synced = _shared.synced
words = _shared.words

#: ``measure_predicate_coverage`` digests this package into every lifted row's
#: ``subject_manifest_sha256``. Declared, as every battery declares it.
SUBJECT_PACKAGE = "metal_kernels"

#: The budget every synthetic case runs. SIXTY, the sibling H->D gate's, and for its
#: reason: the state this weld carries between steps -- the rotated ``H``/``f_w_H``
#: pair, the in-place ``fu_D``, and on a fold the ghost planes the fills rewrite --
#: compounds across steps. The comparison is per COMPLETE STEP.
STEPS = 60

#: The steps at which the sync leg calls the flux accessor. Two, so the hazard is
#: measured to persist and is not a one-step transient.
SYNC_STEPS: Tuple[int, ...] = (3, 7)

#: HOW MANY CLEAN COMPLETE STEPS A LIFTED CORPUS ROW MUST REACH TO COUNT. The
#: synthetic fixture's amplitude is this gate's to choose and is placed clear of the
#: denormal band for the whole budget; a lifted row's state is the ROW's, starts at the
#: engine's zeros and enters the band on its own schedule. It is a FLOOR, not a target:
#: every row carries its own step count and the words it compared.
LIFT_CLEAN_STEP_FLOOR = 8

#: THE SEED IS SCALED BY 2^80, and the exponent is the sibling gate's MEASUREMENT
#: reused rather than re-derived: the solver is linear in the field state and every
#: coefficient it multiplies by is field-independent, so scaling every stored volume by
#: 2^n shifts each float32 EXPONENT by n and leaves every MANTISSA and every rounding
#: decision untouched. 2^80 clears a sixty-step budget on the sibling's fixture with
#: twelve decimal orders of headroom under float32's finite range at the top.
#:
#: THE FOLD MAKES IT MORE NECESSARY, NOT LESS: ``matrix.folded``'s thickness rule puts
#: the absorber on the FAR face of a folded axis only (``(0, 2)``), which is exactly
#: where the tiniest magnitudes live, and the fills read and write those planes.
SEED_SCALE_BITS = 80

#: name -> the folded fixture's keywords. EVERY LEVER THE FOLD HAS, AT BOTH VALUES,
#: sized so a ghost plane is 192 words rather than 16.
#:
#:   TERMINATION   MIRROR_METALLIC vs MIRROR_PERIODIC is this family's single point of
#:                 failure: the far ghost pass and the top-plane mask exist on one and
#:                 not the other, so a mutation caught on one is not caught on the
#:                 other;
#:   PHASE         the fills' parity is a SOURCE specialisation elsewhere in this
#:                 family; this product must be BLIND to it, which is a claim only an
#:                 odd-plane fixture can measure;
#:   WALL          ``zero_metal_B`` only emits work for an axis that is metallic and
#:                 NOT mirrored, so a matrix without a live non-folded wall would
#:                 report the wall-scoped questions as structurally absent -- and 28 of
#:                 the cell's 78 rows carry one;
#:   FOLDED AXES   one exercises no composition; two make a corner unowned on both
#:                 planes; three is the maximum the engine builds;
#:   FULL-COUNT    ``_far_reflect_rows`` is ``n_full - stored + 2``, which is
#:   PARITY        ``stored - 2`` at an even full count and ``stored - 3`` at an odd
#:                 one, so ``extent=2.1`` is where the wrong formula is a whole cell
#:                 wrong;
#:   DIMENSION     73 of the cell's 78 rows are 2-D and 5 are 3-D, and a Z fold is
#:                 exercised by NO standing gate at all (leg ``fixture_shape``).
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("xy_metallic_even_2d",
     dict(axis="XY", boundaries={"x": "metallic", "y": "metallic"})),
    ("y_metallic_x_wall_even_2d",
     dict(axis="Y", boundaries={"x": "metallic", "y": "metallic"})),
    ("y_periodic_even_2d", dict(axis="Y")),
    ("y_periodic_odd_count_3d", dict(axis="Y", extent=2.1, depth=1.2)),
    ("y_periodic_odd_phase_3d", dict(axis="Y", phase=-1, depth=1.2)),
    ("xy_periodic_mixed_3d", dict(axis="XY", phase=(1, -1), depth=1.2)),
    ("yz_metallic_x_wall_3d",
     dict(axis="YZ", depth=1.2,
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"})),
    ("yz_periodic_odd_count_3d", dict(axis="YZ", extent=2.1, depth=1.2)),
    ("xyz_metallic_3d",
     dict(axis="XYZ", depth=2.0,
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"})),
)

#: The case the shader mutations are armed on: two folded PERIODIC axes with MIXED
#: parities, in 3-D. Every line the shipped kernel can emit is present -- two
#: top-plane mask lines, two folded ghost ternaries, the full six-row cell-0 mask --
#: and the doubly-unowned corner exists. A mutation leg run on a single-fold metallic
#: row would report the top-plane defects as uncaught for the uninteresting reason
#: that no line is emitted there.
MUTATION_CASE = "xy_periodic_mixed_3d"

#: THE SECOND MUTATION CASE, and it is not a duplicate. ``zero_metal_B`` emits work
#: only for an axis that is metallic and NOT mirrored, so the wall-scoped questions --
#: and the whole ``(M, MM, *)`` partition, 28 of the cell's 78 rows -- are reachable
#: only here. It is also the ONLY specialisation on which a MIRROR_METALLIC top plane
#: is stepped rather than masked, which is what makes
#: ``top_plane_mask_on_a_metallic_fold`` (deleting a real cell) a live defect.
WALL_MUTATION_CASE = "y_metallic_x_wall_even_2d"

#: The fixtures leg ``sub_step`` runs on: one of each termination, because the two
#: masks live on different terminations and the leg exists for the masks.
SUB_STEP_CASES: Tuple[str, ...] = (MUTATION_CASE, WALL_MUTATION_CASE,
                                   "y_periodic_even_2d")

#: The fixture geometry, ``metal_composition_matrix.folded``'s own constants, so the
#: grid this certifies is the grid the folded family was certified on. Leg
#: ``fixture_shape`` MEASURES that equality rather than asserting it.
RESOLUTION = 10.0
COURANT = 0.35
CROSS = 1.6

#: Per-component permittivity. ``metal_composition_matrix._epsilon``'s own values, so
#: the material is the folded family's too.
EPSILON: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}

#: The reference engines, in the order the record reports them.
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused",
                          "weld", "weld_seam_only")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "fixture_shape", "purity",
             "phase_blind", "refusal"),
    "compile": ("binding_ceiling", "mutants_compile"),
    "device": ("product", "codes_source", "sub_step", "race", "rotation",
               "arbitration", "sync", "launch_structure", "lift", "byte_neutral",
               "mutation", "disarm"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The census this gate lifts its corpus rows from, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows the predicate must refuse.
CENSUS = "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = "h_to_d_seam_2026-09-04"
CELL_ARMS: Tuple[str, str] = ("folded", "folded")

#: The runner's "cannot certify on this host" code: a partial run (not every leg
#: requested) exits with it so ``release.released`` is False and nothing can mint it.
EXIT_INCOMPLETE = 75


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The folded fixture: a REAL driver on the composition matrix's own geometry
# ---------------------------------------------------------------------------

def _geometry(keywords: Mapping[str, Any]) -> Dict[str, Any]:
    """``metal_composition_matrix.folded``'s OWN size and thickness rules, re-derived.

    Re-derived rather than called, because that builder returns ``(fields, pml)`` and
    every claim in this file is about a COMPLETE DRIVER STEP -- the withdraw loop, the
    injection slot, both mirror fills, both wall clears, both far passes and the sync
    channel are ``FdtdDriver.step``'s, and a hand-written walk over the live pass list
    would be a second model of what a step is. Leg ``fixture_shape`` then MEASURES
    that this re-derivation and ``matrix.folded`` build the same grid, so "the fixture
    the folded family was certified on" is a comparison and not a claim.
    """
    axes = tuple(keywords.get("axis", "Y"))
    phase = keywords.get("phase", 1)
    phases = (phase,) * len(axes) if isinstance(phase, int) else tuple(phase)
    if len(phases) != len(axes):
        raise ValueError(f"{len(axes)} folded axes but {len(phases)} phases")
    extent = float(keywords.get("extent", 2.0))
    depth = float(keywords.get("depth", 0.0))
    size = [CROSS, CROSS, depth]
    for name in axes:
        size["XYZ".index(name)] = extent
    return {"axes": axes, "phases": phases, "cell_size": tuple(size),
            "dimensions": 2 if not depth else 3,
            "boundaries": keywords.get("boundaries"),
            "folded_indices": {"XYZ".index(name) for name in axes}}


def _pml_thickness(shape: Sequence[int], folded: Sequence[int]) -> Dict[str, Any]:
    """``matrix.folded``'s thickness rule: ``(0, 2)`` on a folded axis, ``(2, 2)``
    elsewhere, and nothing at all on an axis too thin to hold a layer.

    THE ASYMMETRY IS THE POINT: a mirror plane is not an absorber, so the LOW face of
    a folded axis holds no layer at all and the deepest absorber cells sit on the FAR
    face -- which is the plane the far ghost pass writes and the top-plane mask
    protects.
    """
    out: Dict[str, Any] = {}
    for index, name in enumerate("xyz"):
        if int(shape[index]) < 6:
            out[name] = 0
        elif index in folded:
            out[name] = (0, 2)
        else:
            out[name] = (2, 2)
    return out


def build_driver(keywords: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS) -> Any:
    """One seeded folded ``FdtdDriver`` on the composition matrix's own geometry."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    spec = _geometry(keywords)
    driver = FdtdDriver(cell_size=spec["cell_size"], resolution=RESOLUTION,
                        courant=COURANT, force_complex_fields=False,
                        symmetry=tuple(Mirror(name, int(value))
                                       for name, value in zip(spec["axes"],
                                                              spec["phases"])),
                        boundaries=spec["boundaries"],
                        dimensions=spec["dimensions"])
    driver.setup_pml(_pml_thickness(driver.grid.shape, spec["folded_indices"]))
    shape = tuple(driver.grid.shape)
    driver.fields.set_epsilon_volumes(
        {name: np.full(shape, np.float32(value), np.float32)
         for name, value in EPSILON.items()},
        {name: np.full(shape, np.float32(1.0 / value), np.float32)
         for name, value in EPSILON.items()})
    for source in sources:
        driver.add_source(dict(source))
    rng = np.random.default_rng(seed)
    scale = np.float32(2.0) ** int(scale_bits)
    for array in stored_volumes(driver.fields).values():
        array[...] = (0.37 * rng.standard_normal(array.shape)).astype(array.dtype)
        array *= scale
    driver.invalidate_fast_path()
    pin_array_path(driver)
    return driver


def codes_of(driver: Any) -> Tuple[int, ...]:
    """The FOUR-CODE folded quadruple this driver compiles to.

    From ``folded_axis_kinds`` and from nowhere else -- see the family module. A
    triple built from ``_boundary_kinds`` alone loses the MIRROR_METALLIC /
    MIRROR_PERIODIC split, which decides the top-plane mask; leg ``codes_source``
    drives exactly that mistake and requires it to diverge.
    """
    codes, reasons = symmetry.folded_axis_kinds(driver.grid, driver.pml)
    if codes is None:
        raise RuntimeError(f"folded_axis_kinds refused this fixture: {list(reasons)}")
    return tuple(int(code) for code in codes)


def boundary_kinds_codes(driver: Any) -> Tuple[int, ...]:
    """THE ARMED STRUCTURAL DEFECT: the PLAIN product's own codes expression.

    ``fused_hd_pair.plan_metal_fused_hd_pair`` builds ``1 if kind == "metallic" else
    0`` over ``stepping._boundary_kinds``. On a folded grid that resolver reports
    ``"mirror"`` (stepping.py:2191-2196), which this expression maps to 0 = PERIODIC:
    the ghost becomes a WRAP to the far plane AND the cell-0 mask is not emitted. It
    is a smooth wrong answer on every folded axis rather than a crash, and
    ``folded_curl_source`` cannot catch it either since 0 and 1 are valid codes there.
    """
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    return tuple(1 if kind == "metallic" else 0
                 for kind in _boundary_kinds(driver.grid, driver.pml))


CODE_NAMES = {P: "P", M: "M", MM: "MM", MP: "MP"}


def codes_label(codes: Sequence[int]) -> str:
    return "_".join(CODE_NAMES.get(int(code), f"?{code}") for code in codes)


# ---------------------------------------------------------------------------
# The arrangements — folded
# ---------------------------------------------------------------------------

def count_functions(plan: Any) -> List[_shared.CountingFunction]:
    """Replace every compiled function a plan holds with a counting wrapper.

    THE SIBLING GATE'S HELPER CANNOT BE REUSED HERE, AND THE REASON IS THE FOLD. It
    assumes ``plan._functions`` is ``{mode: function}``, which is true of every plan
    an UNFOLDED step carries. A folded step also carries
    :class:`.symmetry.MirrorGhostFillPlan`, whose table is ``{mode: {axis entry:
    function}}`` (symmetry.py:1493) because one launch per pass per folded axis is
    what a mirror fill is -- so wrapping the outer values replaces a DICT with a
    counter and the next launch dies on ``'CountingFunction' object is not
    subscriptable``. Measured here on the first device run: every arrangement's step
    raised at step 1 and every case reported ``steps_compared: 0``, which reads as a
    weld defect and is a harness one.

    Both shapes are handled, and an unrecognised one is left ALONE rather than
    wrapped: a plan this cannot count contributes zero to the independent counter,
    which ``run_product`` scores as a counter disagreement -- a visible failure rather
    than a silent one.
    """
    owner = declaring(plan)
    table = getattr(owner, "_functions", None)
    if not isinstance(table, dict):
        return []
    counters: List[_shared.CountingFunction] = []
    wrapped: Dict[str, Any] = {}
    for mode, entry in table.items():
        if isinstance(entry, dict):
            inner: Dict[str, Any] = {}
            for key, function in entry.items():
                counter = _shared.CountingFunction(function)
                counters.append(counter)
                inner[key] = counter
            wrapped[mode] = inner
        elif callable(entry):
            counter = _shared.CountingFunction(entry)
            counters.append(counter)
            wrapped[mode] = counter
        else:
            return []
    owner._functions = wrapped  # noqa: SLF001 - the sibling gates' mutation seam
    return counters


class FoldedArrangement:
    """One engine: a shim (or the array path), its residency, and its bookkeeping.

    ``Arrangement``'s shape exactly -- ``drive`` reads ``shim``, ``residency``,
    ``settle``, ``launches``, ``selected``, ``reasons`` and ``on_step_done`` and
    nothing else -- with ONE difference: the independent launch counter is installed
    through this file's :func:`count_functions`, which understands the mirror fill's
    nested function table. The rotation bookkeeping is the sibling's, imported, since
    that is where the 2026-09-05 orphaned-mirror defect was found and fixed.
    """

    __slots__ = ("name", "shim", "residency", "counters", "rotating", "originals",
                 "selected", "reasons", "on_step_done")

    def __init__(self, name: str, shim: Optional[Shim],
                 residency: Optional[Residency],
                 selected: Optional[Mapping[str, str]] = None,
                 reasons: Optional[Mapping[str, Any]] = None) -> None:
        self.name = name
        self.shim = shim
        self.residency = residency
        self.counters: List[_shared.CountingFunction] = []
        if shim is not None:
            # ONCE PER DECLARING OWNER, never once per slot: a fused pair occupies
            # BOTH slots of its seam, so walking slots would wrap that owner's table
            # twice and the independent counter would report exactly twice the
            # launches the plans report.
            seen: List[int] = []
            for plan in shim.plans.values():
                if plan is ABSORBED:
                    continue
                owner = declaring(plan)
                if id(owner) in seen:
                    continue
                seen.append(id(owner))
                self.counters.extend(count_functions(plan))
        self.rotating = _shared.rotating_owners(shim)
        self.originals = ({} if shim is None
                          else _shared.rotating_originals(shim.fields, self.rotating))
        self.selected = dict(selected or {})
        self.reasons = dict(reasons or {})
        self.on_step_done: Optional[Callable[[], None]] = None

    def settle(self, fields: Any) -> int:
        return sum(_shared.settle_rotation(plan, fields, self.originals)
                   for plan in self.rotating)

    def launches(self) -> Dict[str, int]:
        return {"plans": 0 if self.shim is None else self.shim.launches,
                "functions": sum(counter.calls for counter in self.counters)}


def arrangement_singles(driver: Any) -> "FoldedArrangement":
    """Reference 2: the two CERTIFIED FOLDED singles at the seam's two slots.

    ``symmetry.plan_folded_constitutive`` is the certified ``ConstitutivePlan`` built
    through the certified builder -- the fold adds NO kernel on ``update_H`` -- and
    ``symmetry.plan_folded_pml_curl`` is the folded curl. Together they are exactly
    what ``plan_step`` puts on these two slots when neither neighbouring pair claims
    them, so this reference is a composition rather than a strawman.
    """
    residency = Residency()
    fields, pml = driver.fields, driver.pml
    constitutive = symmetry.plan_folded_constitutive(fields, pml, "H", residency)
    curl = symmetry.plan_folded_pml_curl(fields, pml, "step_D", residency)
    if constitutive is None or curl is None:
        raise RuntimeError(
            "a certified folded single was refused on a fixture this gate expects it "
            f"to admit (constitutive={constitutive is not None}, "
            f"curl={curl is not None})")
    shim = Shim(fields, {"update_H": synced(constitutive, residency),
                         "step_D": synced(curl, residency)})
    return FoldedArrangement("singles", shim, residency,
                             selected={"update_H": "folded", "step_D": "folded"})


def _composed(driver: Any, fuse: bool,
              residency: Optional[Residency] = None) -> Tuple[Any, Residency]:
    """``plan_step`` on this driver, on ``residency`` or on a fresh one.

    ONE ARRANGEMENT IS ONE RESIDENCY. The registry's whole purpose is that every plan
    touching a volume binds the SAME device tensor: ``step_B`` writes ``Bx`` where
    ``update_H`` reads it, the mirror fill writes it again, and this weld writes ``D``
    in place where ``update_E`` reads it. Two registries in one arrangement would
    launch against two device buffers for one host array and the copies back would
    overwrite each other.
    """
    residency = Residency() if residency is None else residency
    plan = metal_launch.plan_step(driver.fields, driver.pml, residency=residency,
                                  sources=tuple(getattr(driver, "_sources", ())),
                                  fuse=fuse)
    return plan, residency


def arrangement_composition(driver: Any, fuse: bool) -> "FoldedArrangement":
    """References 3 (``fuse=True``) and 4 (unfused), built by the SHIPPED composer.

    Reaching for the family builders here would measure the kernels and skip the
    composition, and the composition is what the arbitration finding is about.
    """
    plan, residency = _composed(driver, fuse)
    plans = {slot: synced(plan.plans[slot], residency)
             for slot in metal_launch.STEP_ORDER if slot in plan.plans}
    return FoldedArrangement(
        "composition_today" if fuse else "unfused",
        Shim(driver.fields, plans), residency, selected=plan.selected,
        reasons={key: list(value) for key, value in plan.reasons.items()
                 if key.startswith("fused_pair")})


def build_weld(driver: Any, residency: Residency,
               functions: Optional[Mapping[str, Any]] = None,
               sources: Any = (),
               codes: Optional[Sequence[int]] = None) -> Any:
    plan = family.plan_metal_folded_fused_hd_pair(
        driver.fields, driver.pml, sources=sources, residency=residency,
        functions=functions, codes=codes)
    if plan is None:
        reasons = family.metal_folded_fused_hd_pair_coverage(
            driver.fields, driver.pml, sources, residency).reasons
        raise RuntimeError("the folded fused H/D pair was refused: "
                           + "; ".join(reasons))
    return plan


def arrangement_weld(driver: Any, functions: Optional[Mapping[str, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     sync_hazard: bool = False, rest_unfused: bool = True,
                     codes: Optional[Sequence[int]] = None,
                     name: str = "weld") -> "FoldedArrangement":
    """The SUBJECT: the folded weld force-installed at ``update_H``, ``step_D``
    absorbed.

    FORCE-INSTALLED, because the composer refuses this product by name (no absorb row,
    ``INSTALLABLE = False``) and this gate must measure it anyway. Every other slot --
    ``step_B``, both mirror fills, both wall clears, both far passes and ``update_E``
    -- carries whatever ``plan_step(fuse=False)`` selects for it, so the ONLY
    difference between this arrangement and reference 4 is the seam.

    ``launcher`` is the HOST mutation seam, ``functions`` the SHADER one and ``codes``
    the STRUCTURAL one.
    """
    residency = Residency()
    plan = build_weld(driver, residency, functions=functions, sources=(), codes=codes)
    occupant: Any = (plan if launcher is None
                     else launcher(plan, residency, driver.pml))
    occupant = SyncedPlan(occupant, residency)
    plans: Dict[str, Any] = {}
    if rest_unfused:
        # ON THIS ARRANGEMENT'S OWN RESIDENCY -- see :func:`_composed`.
        base, _base = _composed(driver, fuse=False, residency=residency)
        for slot in metal_launch.STEP_ORDER:
            if slot in base.plans and slot not in family.REPLACES:
                plans[slot] = synced(base.plans[slot], residency)
    plans["update_H"] = occupant
    plans["step_D"] = ABSORBED
    shim = Shim(driver.fields, plans, sync_hazard=sync_hazard)
    return FoldedArrangement(
        name, shim, residency,
        selected={"update_H": "folded fused H/D pair (forced)",
                  "step_D": "folded fused H/D pair (forced)"})


def all_arrangements(driver: Any, functions: Optional[Mapping[str, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     codes: Optional[Sequence[int]] = None,
                     ) -> Dict[str, "FoldedArrangement"]:
    """The array path, the three references and the subject TWICE, on one driver.

    ``weld_seam_only`` is the same launch with every other slot LEFT ON THE ARRAY PATH
    -- the ``singles`` composition with the two halves welded. It exists because
    ``weld`` SHARES its non-seam plans with reference 4, so a defect in one of those
    reads as a weld defect; isolating the seam is the only way such a row can be
    attributed, and attribution is the difference between a finding and a red.
    """
    return {
        "array": FoldedArrangement("array", None, None),
        "singles": arrangement_singles(driver),
        "composition_today": arrangement_composition(driver, fuse=True),
        "unfused": arrangement_composition(driver, fuse=False),
        "weld": arrangement_weld(driver, functions=functions, launcher=launcher,
                                 codes=codes),
        "weld_seam_only": arrangement_weld(driver, functions=functions,
                                           launcher=launcher, codes=codes,
                                           rest_unfused=False, name="weld_seam_only"),
    }


def divergence(result: Mapping[str, Any]) -> Dict[str, Any]:
    """WHERE and BY HOW MUCH each arrangement parted company, and WHICH did.

    ``drive`` keeps this per arrangement, which is the only place it can live: a
    defect that moves the weld and leaves the three references alone is a finding
    about this product, and one that moves a reference too is a finding about the
    harness or the fixture. A mutation leg reporting only "not identical" cannot tell
    them apart, so every leg here reports the attribution beside the verdict.
    """
    legs = result.get("arrangements") or {}
    diverged = sorted(name for name, leg in legs.items() if not leg.get("identical"))
    weld = legs.get("weld") or {}
    return {
        "arrangements_that_diverged": diverged,
        "weld_first_divergence": weld.get("first_divergence"),
        "weld_differing_words": weld.get("differing_words_final"),
        "weld_differing_volumes": weld.get("differing_volumes_final"),
        "only_the_weld_diverged": diverged in ([], ["weld", "weld_seam_only"],
                                               ["weld"], ["weld_seam_only"]),
        "first_divergence_per_arrangement": {
            name: leg.get("first_divergence") for name, leg in legs.items()},
    }


def run_product(driver: Any, steps: int,
                progress: Optional[Callable[[str], None]] = None,
                movement_floor: str = "all",
                functions: Optional[Mapping[str, Any]] = None,
                launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                codes: Optional[Sequence[int]] = None,
                stop_on_divergence: bool = True,
                require_full_budget: bool = True,
                clean_floor: int = 0) -> Dict[str, Any]:
    """The four-reference identity on one folded driver, per complete step."""
    arrangements = all_arrangements(driver, functions=functions, launcher=launcher,
                                    codes=codes)
    result = drive(driver, arrangements, steps, progress=progress,
                   secondary="singles", stop_on_divergence=stop_on_divergence)
    legs = result["arrangements"]
    identical = all(legs[name]["identical"] for name in legs)
    compared = int(result["steps_compared"])
    budget_ok = (compared == steps if require_full_budget
                 else compared >= max(clean_floor, 1))
    launched = all(legs[name]["launches"]["plans"] > 0
                   and legs[name]["launches"]["plans"]
                   == legs[name]["launches"]["functions"]
                   for name in legs)
    weld_ok = all(
        legs[name]["dispatched"].get("update_H", 0) == result["steps_stepped"]
        and legs[name]["absorbed"].get("step_D", 0) == result["steps_stepped"]
        for name in ("weld", "weld_seam_only"))
    # THE FLOOR, AND WHICH ONE DEPENDS ON WHERE THE STATE CAME FROM. ``all`` is the
    # SYNTHETIC fixture's: every stored volume is seeded, so every one must move or the
    # comparison is a no-op agreeing with a no-op. ``seam`` is the LIFTED ROW's, where
    # a component the row's polarization never excites is identically zero for the
    # whole run -- a 2-D TM row moves Ez, Hx and Hy and nothing else.
    seam_outputs = tuple(name for name in
                         [f"{stem}{axis}" for stem in ("D", "fu_D", "H", "f_w_H")
                          for axis in "xyz"]
                         if name in result["volumes_compared"])
    seam_moved = {name: int(result["moved_from_seed"].get(name, 0))
                  for name in seam_outputs}
    if movement_floor == "all":
        floor = not result["arrays_that_never_moved"]
    else:
        floor = bool(seam_outputs) and sum(seam_moved.values()) > 0
    settled = bool(result["every_arrangement_gave_the_engine_its_volumes_back"])
    census_covered = result["the_intermediate_census_modelled_every_helper_the_step_ran"]
    result.update({
        "passed": bool(identical and launched and weld_ok and floor and budget_ok
                       and settled and result["step_error"] is None
                       and census_covered is not False
                       and (result["precondition_clean"] or not require_full_budget)),
        "bit_identical": identical,
        "budget_met": budget_ok,
        "require_full_budget": require_full_budget,
        "clean_floor": clean_floor,
        "every_arrangement_launched_and_the_two_counters_agree": launched,
        "weld_launched_once_per_step_and_absorbed_step_D": weld_ok,
        "movement_floor": movement_floor,
        "movement_floor_met": floor,
        "seam_output_words_moved": seam_moved,
        "seam_output_volumes_that_never_moved": sorted(
            name for name, n in seam_moved.items() if not n),
        "weld_agrees_with_the_certified_folded_singles":
            legs["weld"]["identical_vs_secondary"],
        "the_seam_alone_agrees_with_the_array_path": legs["weld_seam_only"]["identical"],
        "references_that_disagree_with_the_array_path": sorted(
            name for name in ("singles", "composition_today", "unfused")
            if not legs[name]["identical"]),
        "composition_today_selected": legs["composition_today"]["selected"],
    })
    return result


# ---------------------------------------------------------------------------
# The armed defects — the plain product's set plus the fold's own
# ---------------------------------------------------------------------------

def _tap_call(coordinates: str, target: int) -> str:
    return f"h_cell({coordinates}, {plain.H_CELL_TAIL_ARGS}).a{target}"


_A_Y = f"    float a_y = vy ? {_tap_call('i, sj, k', 0)} : 0.0f;\n"
_ACCUMULATION_0 = ("        a0 = a0 + kp_0 * src0;\n"
                   "        a0 = a0 - km_0 * prev0;\n")


def shader_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str]]:
    """Every armed source defect as ``name -> (source, why it must fire)``.

    Every needle is anchored on text only the mutated line carries and is verified to
    resolve EXACTLY ONCE before it is applied -- an absent or doubled needle raises
    here rather than arming nothing.

    THE SET IS THE PLAIN PRODUCT'S PLUS THE FOLD'S OWN, and the split is deliberate:
    the shared ones are shared because the two welds lift the same two certified
    bodies, so a defect in the lift is a defect in both, and re-arming them here is
    what keeps the folded emission from drifting away from the plain one silently.
    """
    base = family.folded_fused_hd_pair_source(codes)
    out: Dict[str, Tuple[str, str]] = {}

    def arm(name: str, source: str, why: str) -> None:
        out[name] = (source, why)

    # --- the seam's own defects: the foreign tap ----------------------------
    arm("foreign_tap_reads_stale_H",
        needle(base, _A_Y, "    float a_y = vy ? hi0[oy] : 0.0f;\n"),
        "the tap reads the PRE-LAUNCH H at the neighbour instead of recomputing "
        "update_H there; fires wherever update_H moves that cell (the purity ledger)")
    arm("foreign_tap_reads_B",
        needle(base, _A_Y, "    float a_y = vy ? b0[oy] : 0.0f;\n"),
        "the tap reads the flux density where the curl needs the stepped H")
    arm("foreign_tap_reads_the_scratch_output",
        needle(base, _A_Y, "    float a_y = vy ? ho0[oy] : 0.0f;\n"),
        "THE PLANTED RACE: the tap reads another thread's store, which holds either "
        "the neighbour's new H or two-launches-old scratch")
    arm("halo_recompute_dropped",
        needle(base, _A_Y, "    float a_y = 0.0f;\n"),
        "one shifted magnetic load replaced by the ghost value everywhere")
    arm("halo_moved_to_the_forward_neighbour",
        needle(base, "    int si = i - 1, sj = j - 1, sk = k - 1;\n",
               "    int si = i - 1, sj = j + 1, sk = k - 1;\n"),
        "step_D is the BACKWARD curl; one axis differenced forward is step_B's "
        "stencil, and on a fold it also moves which plane the ghost lands on")
    arm("own_cell_reads_stale_H_not_the_register",
        needle(base, "    float a   = own.a0;\n", "    float a   = hi0[ii];\n"),
        "the curl takes the pre-launch H at its own cell: the seam undone")
    # --- the constitutive half, inside h_cell (8-space indent) --------------
    arm("constitutive_accumulations_regrouped",
        needle(base, _ACCUMULATION_0,
               "        a0 = a0 + (kp_0 * src0 - km_0 * prev0);\n"),
        "((f + kps*src) - kms*prev) regrouped to f + (kps*src - kms*prev) is a "
        "different float32 number")
    arm("constitutive_accumulations_reversed",
        needle(base, _ACCUMULATION_0,
               "        a0 = a0 - km_0 * prev0;\n        a0 = a0 + kp_0 * src0;\n"),
        "the two accumulations in the other order")
    arm("prev_reads_the_value_the_store_writes",
        needle(base, "        float prev0 = wi0[ii];\n",
               "        float prev0 = b0[ii];\n"),
        "the split-field history read AFTER the store that overwrote it with B")
    arm("kps_kms_swapped",
        needle(base, "        float kp_0 = kp0[i], km_0 = kmx[i];\n",
               "        float kp_0 = kmx[i], km_0 = kp0[i];\n"),
        "the two absorber coefficients exchanged on one component")
    arm("h_store_dropped",
        needle(base, "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n",
               "    ho1[ii] = own.a1; ho2[ii] = own.a2;\n"),
        "one magnetic component never reaches the scratch; the next step reads stale")
    arm("fw_store_dropped",
        needle(base,
               "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n",
               "    wo1[ii] = own.src1; wo2[ii] = own.src2;\n"),
        "one split-field history never advances")
    # --- the curl half ------------------------------------------------------
    arm("curl_parens_flattened",
        needle(base, "dtdx * ((c_y - c) + (b - b_z))", "dtdx * (c_y - c + b - b_z)"),
        "shaders.py rule 2: reassociation is a different float32 number")
    arm("recurrence_axis_pair_swapped",
        needle(base, "    float n0 = ((p0 * km_y) - curl0) * si_y;\n",
               "    float n0 = ((p0 * km_z) - curl0) * si_z;\n"),
        "vec.hpp's cycle_direction: target 0 takes (y, z)")
    arm("fu_store_dropped",
        needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;\n",
               "    u1[ii] = n1; u2[ii] = n2;\n"),
        "the split-field auxiliary never advances on one component")
    arm("flux_store_dropped",
        needle(base, "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n",
               "    f1[ii] = v1; f2[ii] = v2;\n"),
        "one displacement component never advances")
    arm("contraction_pragma_removed",
        needle(base, shaders.contraction_pragma(shaders.CONTRACT_OFF) + "\n", "\n"),
        "without the pragma the toolchain may contract a multiply-add into an fma")
    # --- THE FOLD'S OWN -----------------------------------------------------
    # `_reduced_codes` maps both mirror codes to METALLIC; mapping them to PERIODIC
    # instead is the OTHER plausible reduction, and it is exactly the codes defect
    # expressed in the emitter: the ghost wraps to the far plane instead of serving
    # zero AND the cell-0 mask vanishes (`templates.ownership_mask` emits nothing for
    # a periodic axis).
    arm("mirror_reduced_to_periodic",
        _folded_source_with_reduced_codes(codes, shaders.PERIODIC),
        "both mirror codes reduced to PERIODIC rather than METALLIC: the backward "
        "ghost wraps to the far plane instead of serving an exact 0.0f, and the "
        "cell-0 mask is not emitted at all (symmetry._reduced_codes:456-480)")
    return out


def _folded_source_with_reduced_codes(codes: Sequence[int], reduced: int) -> str:
    """The folded source with every MIRROR code reduced the WRONG way.

    Emitted through the certified emitters at the wrong reduction rather than by
    string surgery, so the mutant is a kernel somebody could plausibly have written
    rather than a syntactic scar.
    """
    codes = tuple(int(code) for code in codes)
    wrong = tuple(reduced if code in symmetry.MIRROR_CODES
                  else (shaders.METALLIC if code == M else shaders.PERIODIC)
                  for code in codes)
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    source = templates.substitute(symmetry._FOLDED_CURL_TEMPLATE, {  # noqa: SLF001
        "__CONTRACT__": templates.contraction_pragma(shaders.CONTRACT_OFF),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", wrong[0], backward),
        "__GHOST_Y__": templates.ghost("y", wrong[1], backward),
        "__GHOST_Z__": templates.ghost("z", wrong[2], backward),
        "__MASK__": templates.ownership_mask(wrong, backward),
        "__TOP_MASK__": symmetry.folded_top_plane_mask(codes, backward),
    })
    return _weld_a_folded_curl_source(source)


def _weld_a_folded_curl_source(curl: str) -> str:
    """Splice an ARBITRARY folded curl source into the fused template.

    The mutation seam for every defect that lives in the CURL EMITTER rather than in
    a line of the emitted text: the mutant is built by asking the emitter for a
    different (and wrong) specialisation and welding THAT, so the mutated kernel is
    the one a wrong emitter call would produce.
    """
    tail = curl.split(plain.DECODE_END, 1)[1]
    assert tail.endswith("}\n")
    tail = tail[: -len("}\n")]
    offsets = plain.offset_coordinates(tail)
    for old, new in plain.OWN_LOAD_EDITS:
        tail = _shared.needle(tail, old, new)
    for var, target in plain.HALO_TAPS:
        prefix = f"    float {var} = "
        matches = [line + "\n" for line in tail.splitlines()
                   if line.startswith(prefix)]
        assert len(matches) == 1, (var, len(matches))
        old = matches[0]
        head, rest = old.split(" ? ", 1)
        expected = f"g{target}["
        assert rest.startswith(expected) and rest.endswith(" : 0.0f;\n"), rest
        offset = rest[len(expected):].split("]", 1)[0]
        call = _tap_call(", ".join(offsets[offset]), target)
        tail = _shared.needle(tail, old, f"{head} ? {call} : 0.0f;\n")
    prologue = (curl.split(plain._BODY_ANCHOR, 1)[1]  # noqa: SLF001
                .split(plain.DECODE_END, 1)[0] + plain.DECODE_END)
    return shaders.substitute(plain._TEMPLATE, {  # noqa: SLF001
        "__CONTRACT__": shaders.contraction_pragma(shaders.CONTRACT_OFF),
        "__H_CELL__": plain.h_cell_function(shaders.CONTRACT_OFF),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": plain.H_CELL_TAIL_ARGS,
        "__CURL__": tail,
    })


def mask_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str]]:
    """The two MASK defects, which are invisible at whole-step granularity.

    THEY ARE NOT ORDINARY MUTATIONS AND SCORING THEM AS SUCH WOULD BE WRONG. The
    driver's fill passes overwrite exactly the planes both masks protect, so a kernel
    carrying NEITHER is bytewise-identical after a complete step (symmetry.py:55-60;
    ``gate_metal_symmetry`` measured ``drop_top_plane_mask`` at 384-1,080 words at
    SUB-STEP granularity and 0 after a complete step). Leg ``sub_step`` is where they
    must fire; the whole-step verdict is RECORDED as a predicted null with that
    citation rather than scored.
    """
    base = family.folded_fused_hd_pair_source(codes)
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    out: Dict[str, Tuple[str, str]] = {}
    top = symmetry.folded_top_plane_mask(codes, backward)
    if "curl" in top:
        out["drop_top_plane_mask"] = (
            needle(base, top + "\n", ""),
            "the far ghost slot of a folded PERIODIC axis is stepped instead of "
            "masked; the fill pass rewrites it, so this is visible ONLY at sub-step "
            "granularity (stepping.py:1934-1944)")
    cell_zero = symmetry.folded_cell_zero_mask(codes, backward)
    if "curl" in cell_zero:
        out["drop_cell0_mask"] = (
            needle(base, cell_zero + "\n", ""),
            "the non-owned cell-0 plane keeps a curl the array path masks; the near "
            "fill rewrites it, so this too is a sub-step measurement")
    return out


def fold_specific_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str]]:
    """The defects only a fold can carry, scored at WHOLE-STEP granularity."""
    base = family.folded_fused_hd_pair_source(codes)
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    out: Dict[str, Tuple[str, str]] = {}
    # A top-plane mask on a MIRROR_METALLIC axis DELETES A STEPPED CELL. There the
    # stored array stops at MEEP's big_corner and the top plane is OWNED
    # (symmetry.py:533-536), so the mask is not a redundancy, it is a wrong answer --
    # and nothing rewrites that plane afterwards, which is why this one IS visible at
    # whole-step granularity while dropping the mask on a periodic fold is not.
    metallic_folds = [axis for axis, code in enumerate(codes) if int(code) == MM]
    if metallic_folds:
        forced = tuple(MP if int(code) == MM else code for code in codes)
        lines = symmetry.folded_top_plane_mask(forced, backward)
        anchor = symmetry.folded_top_plane_mask(codes, backward)
        out["top_plane_mask_on_a_metallic_fold"] = (
            needle(base, anchor + "\n", lines + "\n"),
            "the top-plane mask emitted on a MIRROR_METALLIC axis, whose top plane is "
            "OWNED and STEPPED: the mask deletes a real cell and no fill rewrites it")
    return out


def predicted_null_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str, str]]:
    """The edits predicted NOT to fire, each PAIRED with one that must.

    A predicted null that is merely reported is an untested claim. Each entry carries
    an ``earned_by`` mutation -- the SAME edit plus the mask whose presence is the
    reason the value is dead -- so "this value is dead under the mask" is measured in
    both directions on the same fixture rather than cited.
    """
    base = family.folded_fused_hd_pair_source(codes)
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    out: Dict[str, Tuple[str, str, str]] = {}
    folded = [axis for axis, code in enumerate(codes)
              if int(code) in symmetry.MIRROR_CODES]
    if 1 not in folded:
        return out
    # THE ARRAY PATH'S OWN GHOST VALUE, served where the kernel serves an exact zero:
    # `_shift_down` gives `parity * field[2]` (stepping.py:1870-1873), which at parity
    # +1 is h_cell(i, 2, k). The kernel's literal 0.0f is a DIFFERENT VALUE and the
    # substitution is legal only because the value is DEAD under the cell-0 mask
    # (symmetry.py:30-43). Predicted null.
    mirrored_ghost = needle(
        base, _A_Y,
        f"    float a_y = vy ? {_tap_call('i, sj, k', 0)} : "
        f"{_tap_call('i, 2, k', 0)};\n")
    cell_zero = symmetry.folded_cell_zero_mask(codes, backward)
    out["ghost_zero_replaced_by_the_mirror_source"] = (
        mirrored_ghost,
        "the array path's own folded ghost value (parity * H[2] at parity +1) served "
        "where the kernel serves an exact 0.0f. PREDICTED NULL: the only consumer of "
        "that ghost is a plane the cell-0 mask zeroes (symmetry.py:30-43), so the two "
        "different values produce the same stored bytes",
        needle(mirrored_ghost, cell_zero + "\n", "") if "curl" in cell_zero else "")
    return out


#: Where each coefficient group starts in the plan's STATIC argument tuple. The plain
#: product's, imported: this weld's static tuple is that one exactly.
CURL_COEFFICIENT_STATIC_SLOTS = _shared.CURL_COEFFICIENT_STATIC_SLOTS
CONSTITUTIVE_COEFFICIENT_STATIC_SLOTS = _shared.CONSTITUTIVE_COEFFICIENT_STATIC_SLOTS

HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any, Residency, Any], Any], str]] = {
    "H_and_f_w_H_written_in_place": (
        lambda plan, residency, pml: AliasedLaunch(plan, family.ROTATED_NAMES),
        "THE RACE NULL CONTROL: every rotating volume bound in place, so the scratch "
        "IS the live buffer and a foreign recompute at a neighbour whose thread "
        "already stored reads the value that thread wrote"),
    "f_w_H_written_in_place": (
        lambda plan, residency, pml: AliasedLaunch(
            plan, ("f_w_Hx", "f_w_Hy", "f_w_Hz")),
        "the split-field history stored in place: on this side f_w_H_new[ii] == B[ii] "
        "exactly, so a racing neighbour reads B where it needs B_prev"),
    "rotation_skipped": (
        lambda plan, residency, pml: RotationSkipped(plan),
        "the launcher's post-launch rotation dropped: the engine keeps naming the "
        "pre-launch buffers and every later pass reads a stale H"),
    "curl_takes_the_half_integer_lattice": (
        _shared.curl_takes_the_half_integer_lattice,
        "the SHARED kms and the curl's sinv bound from the half-integer set: a "
        "half-cell error in the absorber profile, which no shader mutation can reach"),
    "constitutive_takes_the_half_integer_lattice": (
        _shared.constitutive_takes_the_half_integer_lattice,
        "kps bound from update_E's lattice"),
}


# ---------------------------------------------------------------------------
# Host legs
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """REPLACES is the driver's two ADJACENT consults, and the SIX FOLD PASSES ARE
    OUTSIDE IT.

    The first half is the sibling gate's and is read off ``driver.py`` by the seam
    module's own locator, which RAISES unless the only statement between the two
    consults is the electric withdraw loop. The second half is this product's own and
    is the reason it carries no fill: both mirror fills, both wall clears and both far
    ghost passes must be absent from the span AND present in the driver's live pass
    list on a folded row, which is read from the composer's own ``live_sub_steps``
    rather than from a table here.
    """
    fact = h_to_d_seam.driver_seam_fact()
    span_ok = (tuple(family.REPLACES) == tuple(h_to_d_seam.HALVES)
               == tuple(withdraw_hoist.SEAM_SPAN))
    driver = build_driver(dict(CASES)[MUTATION_CASE], 4242)
    live = tuple(metal_launch.live_sub_steps(driver.fields, driver.pml, ()) or ())
    fold_passes = ("fill_B", "zero_metal_B", "fill_folded_far_ghosts_B",
                   "fill_D", "zero_metal_D", "fill_folded_far_ghosts_D")
    live_fold_passes = tuple(name for name in fold_passes if name in live)
    inside = tuple(name for name in fold_passes if name in family.REPLACES)
    sync_owner = SYNC_PASS_OWNERS.get(SYNC_UPDATE_H_PASS)
    outside = tuple(name for name in family.REPLACES if name not in SYNC_PATH_SLOTS)
    return {
        "passed": bool(span_ok and not inside and len(live_fold_passes) >= 4
                       and sync_owner == "update_H" and outside == ("step_D",)
                       and family.SEAM == withdraw_hoist.SEAM),
        "driver_seam": fact,
        "replaces": list(family.REPLACES),
        "seam_halves": list(h_to_d_seam.HALVES),
        "withdraw_span": list(withdraw_hoist.SEAM_SPAN),
        "live_passes_on_a_folded_row": list(live),
        "fold_passes_live": list(live_fold_passes),
        "fold_passes_inside_the_span": list(inside),
        "sync_consult": SYNC_UPDATE_H_PASS,
        "sync_owner_slot": sync_owner,
        "span_slots_outside_the_sync_path": list(outside),
        "note": ("the six fold passes are live on this row and NONE is inside the "
                 "span: three close before the update_H consult (driver.py:3305-3310) "
                 "and three open after the step_D consult (:3325-3330), which is why "
                 "this weld carries no fill and must not"),
    }


def _tail_after_decode(source: str) -> str:
    tail = source.split(plain.DECODE_END, 1)[1]
    return tail[: -len("}\n")] if tail.endswith("}\n") else tail


def leg_transcription(codes: Sequence[int]) -> Dict[str, Any]:
    """The emission is the CERTIFIED FOLDED emitters' own bytes, differing exactly
    where declared."""
    source = family.folded_fused_hd_pair_source(codes)
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    certified_curl = _tail_after_decode(
        symmetry.folded_curl_source(codes, backward))
    welded_curl = family.folded_welded_curl_tail(codes)

    # 1. The curl differs in EXACTLY the nine redirected magnetic loads: three own
    #    cells and six halo taps. Every other line is character-identical.
    certified_lines = certified_curl.splitlines()
    welded_lines = welded_curl.splitlines()
    same_length = len(certified_lines) == len(welded_lines)
    changed = ([index for index, (a, b) in enumerate(zip(certified_lines, welded_lines))
                if a != b] if same_length else [])
    redirected_ok = same_length and len(changed) == 9

    # 2. The two FOLD-SPECIFIC blocks survive the lift character for character. That
    #    is the property that makes "only the magnetic loads move" true rather than
    #    approximately true.
    top = symmetry.folded_top_plane_mask(codes, backward)
    cell_zero = symmetry.folded_cell_zero_mask(codes, backward)
    masks_intact = (top in welded_curl) and (cell_zero in welded_curl)

    # 3. No magnetic pointer survives, and the ghost ternary still serves an exact
    #    literal zero past every face.
    no_g = not any(f"g{target}[" in welded_curl for target in range(3))
    ghosts = sum(1 for line in welded_lines if " : 0.0f;" in line and "h_cell(" in line)

    # 4. The constitutive tail is the certified body with exactly the declared edits.
    constitutive = plain.certified_constitutive_tail()
    declared = tuple(edit["line"] for edit in plain.CONSTITUTIVE_LIFT_EDITS)

    # 5. ON AN UNFOLDED TRIPLE THE EMISSION REDUCES TO THE PLAIN PRODUCT'S. That is
    #    the strongest single statement about the fold's device-code delta, and it is
    #    a DIFF rather than a claim: comments plus one dead `last_*` declaration.
    unfolded = family.folded_fused_hd_pair_source((P, P, P))
    plain_source = plain.fused_hd_pair_source((0, 0, 0))
    import difflib  # noqa: PLC0415

    diff = [line for line in difflib.unified_diff(
        plain_source.splitlines(), unfolded.splitlines(), lineterm="", n=0)]
    added = [line[1:] for line in diff
             if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:] for line in diff
               if line.startswith("-") and not line.startswith("---")]
    code_added = [line for line in added
                  if line.strip() and not line.strip().startswith("//")]
    code_removed = [line for line in removed
                    if line.strip() and not line.strip().startswith("//")]
    reduces = (len(code_added) == 1 and "bool last_x" in code_added[0]
               and not code_removed)

    # 6. The mutation needles all resolve, and the text is ASCII.
    needles_ok = True
    needle_error = ""
    try:
        armed = shader_mutations(codes)
        armed.update(mask_mutations(codes))
        armed.update(fold_specific_mutations(codes))
        for name, (_source, _why) in armed.items():
            if _source == family.folded_fused_hd_pair_source(codes):
                needles_ok = False
                needle_error = f"{name} changed nothing"
    except Exception as exc:  # noqa: BLE001 - an unarmable needle IS the finding
        needles_ok = False
        needle_error = f"{type(exc).__name__}: {exc}"
    ascii_ok = all(ord(ch) < 128 for ch in source)

    return {
        "passed": bool(redirected_ok and masks_intact and no_g and reduces
                       and needles_ok and ascii_ok and ghosts == 6
                       and all(edit in str(plain.CONSTITUTIVE_LIFT_EDITS)
                               for edit in declared)
                       and "wo0[ii]" in source and "ho0[ii]" in source
                       and "w0[ii]" not in constitutive),
        "codes": list(int(c) for c in codes),
        "codes_label": codes_label(codes),
        "curl_lines": len(certified_lines),
        "curl_lines_changed_by_the_lift": len(changed),
        "curl_lines_changed_expected": 9,
        "the_two_fold_blocks_survive_character_for_character": masks_intact,
        "top_plane_mask": top,
        "cell_zero_mask_lines": len(cell_zero.splitlines()),
        "no_magnetic_pointer_survives": no_g,
        "halo_taps_still_served_an_exact_literal_zero": ghosts,
        "unfolded_reduces_to_the_plain_product": reduces,
        "unfolded_diff_code_lines_added": code_added,
        "unfolded_diff_code_lines_removed": code_removed,
        "unfolded_diff_comment_lines": len(added) + len(removed) - len(code_added)
        - len(code_removed),
        "declared_constitutive_lift_edits": len(plain.CONSTITUTIVE_LIFT_EDITS),
        "every_mutation_needle_resolves": needles_ok,
        "needle_error": needle_error,
        "ascii": ascii_ok,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def _thickness(pml: Any) -> List[List[float]]:
    """One PML's per-axis, per-side layer depth, NORMALISED to a pair per axis.

    ``PML.thickness`` may hold a scalar where both sides agree and a pair where they
    do not, so a comparison of the raw attribute would report two spellings of the
    same absorber as different -- and would silently pass a genuine difference between
    a scalar and a matching pair.
    """
    raw = getattr(pml, "thickness", None)
    if raw is None:
        return []
    if not isinstance(raw, (tuple, list)):
        raw = (raw,) * 3
    out: List[List[float]] = []
    for entry in raw:
        pair = list(entry) if isinstance(entry, (tuple, list)) else [entry, entry]
        out.append([float(value) for value in pair])
    return out


def leg_fixture_shape() -> Dict[str, Any]:
    """The drivers this gate builds ARE ``metal_composition_matrix.folded``'s grids.

    Both halves of this leg are measurements the design could not make by reading.
    ``matrix.folded`` returns ``(fields, pml)`` and this gate needs an ``FdtdDriver``,
    so the geometry is re-derived here; that re-derivation is compared, per case,
    against the matrix builder's own output -- grid shape, resolved boundary codes and
    PML thickness per face. And ``matrix.folded(axis="YZ"|"XYZ", depth>0)`` is
    exercised by NO STANDING GATE, so whether it builds the intended 3-D grid with a Z
    fold is measured here rather than assumed: the leg requires every named axis to
    resolve to a MIRROR code and every 3-D case to have three nontrivial extents.
    """
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 11, scale_bits=0)
        fields, pml = matrix.folded(**dict(keywords))
        spec = _geometry(keywords)
        driver_codes = codes_of(driver)
        matrix_codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        shape_ok = tuple(driver.grid.shape) == tuple(fields.grid.shape)
        codes_ok = tuple(driver_codes) == tuple(int(c) for c in (matrix_codes or ()))
        thick_ok = (_thickness(driver.pml) == _thickness(pml))
        folded_ok = all(int(driver_codes["XYZ".index(a)]) in symmetry.MIRROR_CODES
                        for a in spec["axes"])
        dims_ok = (spec["dimensions"] == 2) or int(driver.grid.shape[2]) > 1
        rows[name] = {
            "passed": bool(shape_ok and codes_ok and thick_ok and folded_ok
                           and dims_ok),
            "axes": "".join(spec["axes"]), "phases": list(spec["phases"]),
            "driver_shape": list(driver.grid.shape),
            "matrix_shape": list(fields.grid.shape),
            "driver_codes": list(driver_codes),
            "driver_codes_label": codes_label(driver_codes),
            "matrix_codes": list(int(c) for c in (matrix_codes or ())),
            "driver_pml_thickness": _thickness(driver.pml),
            "matrix_pml_thickness": _thickness(pml),
            "every_named_axis_resolved_to_a_mirror_code": folded_ok,
            "dimensionality_as_requested": dims_ok,
            "cells": int(np.prod(driver.grid.shape)),
            "ghost_plane_words": int(np.prod([n for index, n
                                              in enumerate(driver.grid.shape)
                                              if index != 1])),
        }
        log(f"  fixture_shape {name}: shape={rows[name]['driver_shape']} "
            f"codes={rows[name]['driver_codes_label']} "
            f"pml={rows[name]['driver_pml_thickness']} "
            f"passed={rows[name]['passed']}")
    return {"passed": all(row["passed"] for row in rows.values()),
            "cases": rows,
            "z_folds_exercised": sorted(name for name, keywords in CASES
                                        if "Z" in str(keywords.get("axis", "Y"))),
            "note": ("matrix.folded(axis='YZ'|'XYZ', depth>0) is exercised by no "
                     "standing gate; this leg drives it and compares the grid it "
                     "builds against the driver this gate builds")}


def purity_ledger(driver: Any) -> Dict[str, Any]:
    """Of the folded curl's valid foreign taps, how many are RACED -- and how many
    land on the ghost plane.

    THREE COUNTS, NOT ONE, AND THE SECOND TWO ARE THE FOLD'S.

    (a) Every foreign tap reads a neighbour whose stepped value an in-place weld would
        have overwritten at a schedule-decided moment, so the tap is RACED on only
        where the neighbour's new H differs from its old one. Near zero would mean the
        race legs and the stale-H mutation license nothing on this fixture.

    (b) THE GHOST PLANE. On a folded axis stored cell 0 is a REAL stored cell read by
        the thread at index 1, whose H is ``update_H``'s output over a B the driver's
        NEAR fill wrote. Its recompute is only PROVED by taps whose recomputed value
        differs from what an in-place arrangement could have read -- so this counts the
        taps landing there AND how many of those carry an H that moved.

    (c) THE LITERAL-ZERO GHOST. The taps that resolve past the face are not recomputes
        at all: the ternary's guard is false and no call is evaluated. Reported apart
        so the two cannot be confused.

    Measured on the ARRAY PATH: a property of the physics, not of the kernel.
    """
    fields, pml = driver.fields, driver.pml
    codes = codes_of(driver)
    before = {name: np.array(getattr(fields, name), copy=True)
              for name in ("Hx", "Hy", "Hz")}
    stepping.update_H(fields, pml)
    moved = {name: words(before[name]) != words(getattr(fields, name))
             for name in before}
    shape = tuple(fields.grid.shape)
    per_tap: Dict[str, Dict[str, int]] = {}
    total_valid = total_raced = total_ghost = total_ghost_moved = total_zero = 0
    for var, target in plain.HALO_TAPS:
        axis = "xyz".index(var[-1])
        component = ("Hx", "Hy", "Hz")[target]
        flags = moved[component].reshape(shape)
        # The tap reads the BACKWARD neighbour along `axis`. On a PERIODIC axis it
        # wraps; on a METALLIC or MIRROR axis the guard is false at index 0 and the
        # emitter serves an exact 0.0f -- not a read, and not a recompute.
        shifted = np.roll(flags, 1, axis=axis)
        valid = np.ones(shape, dtype=bool)
        zeroed = 0
        if int(codes[axis]) != P:
            index = [slice(None)] * 3
            index[axis] = 0
            valid[tuple(index)] = False
            zeroed = int(np.prod([n for i, n in enumerate(shape) if i != axis]))
        # The taps whose BACKWARD neighbour is stored cell 0 of a FOLDED axis: the
        # threads at index 1 along that axis.
        ghost = np.zeros(shape, dtype=bool)
        if int(codes[axis]) in symmetry.MIRROR_CODES and shape[axis] > 1:
            index = [slice(None)] * 3
            index[axis] = 1
            ghost[tuple(index)] = True
        raced = int(np.count_nonzero(shifted & valid))
        ghost_taps = int(np.count_nonzero(ghost & valid))
        ghost_moved = int(np.count_nonzero(ghost & valid & shifted))
        count = int(np.count_nonzero(valid))
        per_tap[var] = {"valid_taps": count, "taps_on_a_moved_cell": raced,
                        "taps_on_a_folded_ghost_plane": ghost_taps,
                        "ghost_plane_taps_whose_H_moved": ghost_moved,
                        "taps_served_the_literal_zero": zeroed}
        total_valid += count
        total_raced += raced
        total_ghost += ghost_taps
        total_ghost_moved += ghost_moved
        total_zero += zeroed
    return {
        "codes": list(codes), "codes_label": codes_label(codes),
        "cells": int(np.prod(shape)),
        "cells_moved_by_update_H": {name: int(np.count_nonzero(flag))
                                    for name, flag in moved.items()},
        "per_tap": per_tap,
        "valid_foreign_taps": total_valid,
        "foreign_taps_whose_recompute_differs_from_the_in_place_read": total_raced,
        "foreign_taps_on_a_folded_ghost_plane": total_ghost,
        "folded_ghost_plane_taps_whose_H_moved": total_ghost_moved,
        "taps_served_the_literal_zero_and_never_evaluated": total_zero,
        "fraction_raced": (round(total_raced / total_valid, 6) if total_valid
                           else 0.0),
    }


def leg_purity() -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        row = purity_ledger(build_driver(keywords, 5150))
        rows[name] = row
        raced_here = row["foreign_taps_whose_recompute_differs_from_the_in_place_read"]
        log(f"  purity {name}: {raced_here} raced / {row['valid_foreign_taps']} "
            f"valid, {row['folded_ghost_plane_taps_whose_H_moved']} of "
            f"{row['foreign_taps_on_a_folded_ghost_plane']} ghost-plane taps moved, "
            f"{row['taps_served_the_literal_zero_and_never_evaluated']} served the "
            f"literal zero")
    # A FLOOR ON THE LEDGER, not merely a report: a fixture where nothing is raced
    # would make the race legs vacuous, and a fixture where no ghost-plane tap carries
    # a moved H would make the ghost-recompute claim unmeasured.
    raced = all(row["foreign_taps_whose_recompute_differs_from_the_in_place_read"] > 0
                for row in rows.values())
    ghosts = all(row["folded_ghost_plane_taps_whose_H_moved"] > 0
                 for row in rows.values())
    return {"passed": bool(raced and ghosts), "cases": rows,
            "every_fixture_races_at_least_one_tap": raced,
            "every_fixture_recomputes_a_moved_ghost_plane_cell": ghosts}


def leg_phase_blind() -> Dict[str, Any]:
    """The product is BLIND to the mirror parity, and that is measured twice.

    (a) BY SHA256 ON THE SOURCE. The parity is a compile-time source specialisation
        everywhere else in this family (``mirror_ghost_fill_source``), so "this kernel
        does not carry it" is falsifiable: the emitted text for a ``+1`` plane and a
        ``-1`` plane must be the SAME BYTES. The codes are what the kernel
        specialises on, and ``folded_axis_kinds`` resolves the same quadruple at both
        parities.
    (b) BY THE DRIVER. The odd-plane fixture is stepped through the whole budget with
        the ODD fill live -- ``fill_symmetry_bc_B`` writing ``-B[2]`` into cell 0 --
        and must be byte-identical to the array path, which is the claim that the
        parity reaching the kernel only through B is enough.
    """
    even = dict(CASES)["y_periodic_odd_phase_3d"] | {"phase": 1}
    odd = dict(CASES)["y_periodic_odd_phase_3d"]
    even_driver = build_driver(even, 77, scale_bits=0)
    odd_driver = build_driver(odd, 77, scale_bits=0)
    even_codes, odd_codes = codes_of(even_driver), codes_of(odd_driver)
    even_source = family.folded_fused_hd_pair_source(even_codes)
    odd_source = family.folded_fused_hd_pair_source(odd_codes)
    even_sha = hashlib.sha256(even_source.encode()).hexdigest()
    odd_sha = hashlib.sha256(odd_source.encode()).hexdigest()
    phases = [int(odd_driver.grid.mirror_phase(axis) or 0) for axis in range(3)]
    # THE NON-VACUITY: the two fixtures must actually differ somewhere, or the sha
    # equality says nothing. The FILL kernel is where the parity lives, so its source
    # must differ at the two parities on the same grid.
    fill_even = symmetry.mirror_ghost_fill_source  # the emitter that DOES carry it
    fill_differs: Optional[bool] = None
    with contextlib.suppress(Exception):
        entries_even = symmetry.ghost_fill_axis_entries(even_driver.grid, "B", "near")
        entries_odd = symmetry.ghost_fill_axis_entries(odd_driver.grid, "B", "near")
        if entries_even and entries_odd:
            fill_differs = (fill_even(entries_even[0]["shifts"],
                                      int(entries_even[0]["phase"]))
                            != fill_even(entries_odd[0]["shifts"],
                                         int(entries_odd[0]["phase"])))
    return {
        "passed": bool(even_sha == odd_sha and phases[1] == -1
                       and fill_differs is not False),
        "even_phase_source_sha256": even_sha,
        "odd_phase_source_sha256": odd_sha,
        "identical": even_sha == odd_sha,
        "even_codes": list(even_codes), "odd_codes": list(odd_codes),
        "odd_fixture_phases": phases,
        "the_fill_emitter_does_carry_the_parity": fill_differs,
        "note": ("the fill kernel's source DOES differ at the two parities on the "
                 "same grid, which is what makes this kernel's byte equality a "
                 "measurement rather than a coincidence of the harness"),
    }


def leg_refusal() -> Dict[str, Any]:
    """Each refusal by NAME, and the two admissions that are not this seam's business.

    THE FOLD CLAUSE IS DRIVEN IN BOTH DIRECTIONS, because disjointness is a property
    of a PAIR of predicates and neither one alone can carry it: this product must
    refuse an unfolded grid naming ``fused_hd_pair``, and ``fused_hd_pair`` must
    refuse this fixture naming the folded cell.
    """
    def source(fields: Any, component: str, integrated: bool) -> Any:
        """A point source IN THE STORED HALF of every folded axis.

        The sibling gate's helper places its source at ``y = -0.1``, which under a Y
        mirror is in the half the fold DISCARDS -- and ``sources._validate_symmetry``
        refuses that outright rather than returning a silently empty run. Every
        coordinate here is positive, which is the stored half of any plane this
        matrix builds.
        """
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        return VolumeSource(grid=fields.grid, component=component,
                            center=(0.15, 0.1, 0.05), size=(0.0, 0.0, 0.0),
                            envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                      is_integrated=integrated))

    cover = family.metal_folded_fused_hd_pair_coverage
    folded_fields, folded_pml = matrix.folded(axis="Y", depth=1.2)
    unfolded_fields, unfolded_pml = matrix.cart()
    no_pml_fields, no_pml_pml = matrix.folded_no_pml(axis="Y")
    complex_fields, complex_pml = matrix.folded(axis="Y", depth=1.2,
                                                complex_storage=True)
    cyl_fields, cyl_pml = matrix.cylindrical(m=0, complex_storage=False)

    admitted = cover(folded_fields, folded_pml, (), Residency())
    undeclared = cover(folded_fields, folded_pml, None, Residency())
    integrated = cover(folded_fields, folded_pml,
                       (source(folded_fields, "Ez", True),), Residency())
    plain_electric = cover(folded_fields, folded_pml,
                           (source(folded_fields, "Ez", False),), Residency())
    magnetic_source = cover(folded_fields, folded_pml,
                            (source(folded_fields, "Hy", True),), Residency())
    unfolded = cover(unfolded_fields, unfolded_pml, (), Residency())
    inactive = cover(no_pml_fields, no_pml_pml, (), Residency())
    complex_storage = cover(complex_fields, complex_pml, (), Residency())
    cylindrical = cover(cyl_fields, cyl_pml, (), Residency())
    # THE OTHER DIRECTION OF THE INVERTED CLAUSE. Disjointness is a property of a PAIR
    # of predicates and neither one alone can carry it: the plain product must refuse
    # THIS fixture, naming the fold.
    plain_on_folded = plain.metal_fused_hd_pair_coverage(
        folded_fields, folded_pml, (), Residency())
    with _shared.policy("keep"):
        keep = cover(folded_fields, folded_pml, (), Residency())
        keep_report = subnormal.mps_policy_report()

    def named(verdict: Any, *needles: str) -> List[str]:
        return [reason for reason in verdict.reasons
                if all(text in reason for text in needles)]

    checks = {
        "admitted_on_the_cell": admitted.covered,
        "undeclared_refused_by_name": bool(not undeclared.covered
                                           and named(undeclared, "was not declared")),
        "integrated_electric_withdraw_refused_by_name": bool(
            not integrated.covered
            and named(integrated, "standing integrated",
                      "HOISTS_THE_WITHDRAW = False")),
        "plain_electric_source_not_refused": plain_electric.covered,
        "magnetic_source_not_refused": magnetic_source.covered,
        "unfolded_grid_refused_naming_the_plain_product": bool(
            not unfolded.covered and named(unfolded, "fused_hd_pair")),
        "the_plain_product_refuses_this_fixture_naming_the_fold": bool(
            not plain_on_folded.covered and named(plain_on_folded, "folded")),
        "inactive_absorber_refused_by_the_constitutive_half_first": bool(
            not inactive.covered and inactive.reasons
            and inactive.reasons[0].startswith("folded constitutive half:")),
        "complex_storage_refused_naming_the_folded_complex_family": bool(
            not complex_storage.covered
            and named(complex_storage, "complex")),
        "cylindrical_refused": bool(not cylindrical.covered
                                    and named(cylindrical, "cylindrical")),
        "keep_policy_refused_by_name": bool(
            not keep.covered and named(keep, "subnormal policy is 'keep'")
            and not keep_report["admitted"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reasons": {
            "undeclared": list(undeclared.reasons),
            "integrated_electric": list(integrated.reasons),
            "unfolded": list(unfolded.reasons),
            "the_plain_product_on_this_fixture": list(plain_on_folded.reasons),
            "inactive_absorber": list(inactive.reasons)[:4],
            "complex_storage": list(complex_storage.reasons)[:4],
            "cylindrical": list(cylindrical.reasons)[:4],
            "keep_policy": list(keep.reasons),
        },
        "keep_policy_report": keep_report,
    }


# ---------------------------------------------------------------------------
# Compile legs
# ---------------------------------------------------------------------------

def leg_binding_ceiling() -> Dict[str, Any]:
    """The three refuted signatures FAIL, the ceiling is BISECTED, the shipped 31
    compiles on every fixture's own folded codes, and the packed probe LAUNCHES.

    NOTHING HERE IS COUNTED FROM A SIGNATURE. The design's central structural claim is
    that the fold adds no kernel argument on this backend, and the only way that can be
    wrong is a compile: so the ceiling is found by bisection on this host (30 pointers
    plus one packed struct must compile, 31 plus the struct must not), the shipped
    emission is compiled on the codes of EVERY fixture rather than one, and the packed
    record is bound by the plan builder's own packer and read back field by field.
    """
    import torch  # noqa: PLC0415

    row: Dict[str, Any] = {
        "ceiling_constant": MAX_BUFFER_BINDINGS,
        "shipped_bindings_counted_off_the_folded_source":
            family.shipped_signature_bindings(),
        "packed_bindings": family.PACKED_BINDINGS,
        "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
        "unshared_kms_bindings": family.UNSHARED_KMS_BINDINGS,
        "one_more_pointer_bindings": family.ONE_MORE_POINTER_BINDINGS,
    }
    refuted: Dict[str, Any] = {}
    for name, source in (("separate_scalar", family.refuted_separate_scalar_source()),
                         ("unshared_kms", family.refuted_unshared_kms_source()),
                         ("one_more_pointer", family.refuted_one_more_pointer_source())):
        try:
            compile_source(source)
            refuted[name] = {"compiled": True, "refused_for_the_right_reason": False}
        except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
            message = str(exc)
            refuted[name] = {
                "compiled": False,
                "error": (message.splitlines() or [""])[0][:220],
                "refused_for_the_right_reason": ("out of bounds" in message
                                                 and "buffer" in message)}
        log(f"  binding_ceiling refuted/{name}: compiled="
            f"{refuted[name]['compiled']}")
    row["refuted"] = refuted

    # THE BISECTION. A ceiling read from a constant is a claim; found by compiling one
    # signature on each side of it, it is a measurement of this host.
    def probe_source(pointers: int) -> str:
        lines = "\n".join(f"    device float*       b{n:<6}[[buffer({n})]],"
                          for n in range(pointers))
        return "\n".join((
            "#include <metal_stdlib>", "using namespace metal;", "",
            "struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };",
            "", "kernel void packed_probe(", lines,
            f"    constant Params&    prm     [[buffer({pointers})]],",
            "    uint idx [[thread_position_in_grid]])", "{",
            "    if (idx >= prm.n_elem) { return; }",
            "    b0[idx] = float(prm.nx);", "    b1[idx] = float(prm.ny);",
            "    b2[idx] = float(prm.nz);", "    b3[idx] = float(prm.n_elem);",
            "    b4[idx] = prm.dtdx;",
            f"    b5[idx] = b{pointers - 1}[idx] + 1.0f;", "}", ""))

    bisection: Dict[str, Any] = {}
    function = None
    for pointers in (MAX_BUFFER_BINDINGS - 1, MAX_BUFFER_BINDINGS):
        try:
            compiled = compile_source(probe_source(pointers)).packed_probe
            bisection[f"{pointers}_pointers_plus_params"] = {
                "bindings": pointers + 1, "compiled": True}
            if pointers == MAX_BUFFER_BINDINGS - 1:
                function = compiled
        except Exception as exc:  # noqa: BLE001
            bisection[f"{pointers}_pointers_plus_params"] = {
                "bindings": pointers + 1, "compiled": False,
                "error": (str(exc).splitlines() or [""])[0][:220]}
    row["bisection"] = bisection
    ceiling_ok = (bisection[f"{MAX_BUFFER_BINDINGS - 1}_pointers_plus_params"]["compiled"]
                  and not bisection[f"{MAX_BUFFER_BINDINGS}_pointers_plus_params"]
                  ["compiled"])
    row["the_ceiling_is_an_equality_on_this_host"] = ceiling_ok

    # THE SHIPPED SIGNATURE, ON EVERY FIXTURE'S OWN CODES.
    shipped: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 3, scale_bits=0)
        codes = codes_of(driver)
        source = family.folded_fused_hd_pair_source(codes)
        signature = source.split("kernel void fused_hd_pair_step(", 1)[1]
        signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
        try:
            compile_source(source)
            compiled, error = True, ""
        except Exception as exc:  # noqa: BLE001
            compiled, error = False, (str(exc).splitlines() or [""])[0][:220]
        shipped[name] = {"codes": list(codes), "codes_label": codes_label(codes),
                         "bindings": signature.count("[[buffer("),
                         "compiled": compiled, "error": error}
        log(f"  binding_ceiling shipped/{name}: {codes_label(codes)} "
            f"bindings={shipped[name]['bindings']} compiled={compiled}")
    row["shipped_per_fixture"] = shipped
    shipped_ok = all(entry["compiled"] and entry["bindings"] == family.PACKED_BINDINGS
                     for entry in shipped.values())
    row["every_fixture_compiles_at_the_ceiling"] = shipped_ok

    # AND THE PACKED RECORD IS LAUNCHED, bound by the plan builder's OWN packer.
    fields_ok = False
    if function is not None:
        count = 8
        buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
                   for _ in range(MAX_BUFFER_BINDINGS - 1)]
        buffers[-1] = torch.full((count,), 7.0, dtype=torch.float32, device="mps")
        shape, dtdx = (3, 4, 5), 0.125
        params = plain._params_tensor(shape, dtdx, "mps")  # noqa: SLF001
        function(*buffers, params)
        torch.mps.synchronize()
        read = [buffers[index].cpu().numpy() for index in range(6)]
        expected = (float(shape[0]), float(shape[1]), float(shape[2]),
                    float(shape[0] * shape[1] * shape[2]),
                    float(np.float32(dtdx)), 8.0)
        fields_ok = all(bool(np.all(read[index] == expected[index]))
                        for index in range(6))
        row["packed_fields_read_back"] = [float(value[0]) for value in read]
        row["packed_fields_expected"] = [float(value) for value in expected]
    row["packed_launched_and_every_field_read_back"] = fields_ok
    row["passed"] = bool(
        all(not entry["compiled"] and entry["refused_for_the_right_reason"]
            for entry in refuted.values())
        and ceiling_ok and shipped_ok and fields_ok
        and row["shipped_bindings_counted_off_the_folded_source"]
        == family.PACKED_BINDINGS == MAX_BUFFER_BINDINGS)
    return row


def leg_mutants_compile(codes: Sequence[int],
                        wall_codes: Sequence[int]) -> Dict[str, Any]:
    """Every armed defect and the byte-neutral control COMPILE.

    A mutant that did not compile would be reported as caught by whatever error the
    launch raised, which is a hollow pass with a stack trace.
    """
    rows: Dict[str, Any] = {}
    armed: List[Tuple[str, str]] = []
    for name, (source, _why) in shader_mutations(codes).items():
        armed.append((f"{name}", source))
    for name, (source, _why) in mask_mutations(codes).items():
        armed.append((f"mask/{name}", source))
    for name, (source, _why) in fold_specific_mutations(wall_codes).items():
        armed.append((f"wall/{name}", source))
    for name, (source, _why, earned) in predicted_null_mutations(codes).items():
        armed.append((f"null/{name}", source))
        if earned:
            armed.append((f"null/{name}+drop_cell0_mask", earned))
    armed.append(("byte_neutral", byte_neutral_source(codes)))
    for name, source in armed:
        try:
            compile_source(source)
            rows[name] = {"compiled": True}
        except Exception as exc:  # noqa: BLE001
            rows[name] = {"compiled": False,
                          "error": (str(exc).splitlines() or [""])[0][:220]}
    return {"passed": all(row["compiled"] for row in rows.values()),
            "mutants": rows, "count": len(rows)}


def byte_neutral_source(codes: Sequence[int]) -> str:
    """THE ONE ARMED EDIT REQUIRED NOT TO DIVERGE.

    The curl's own-cell magnetic reads come from the registers ``h_cell`` just
    produced; replacing them with reloads of the SCRATCH this thread just stored is a
    read of a value this thread itself wrote at a point where no other thread can have
    intervened, so it must be byte-identical. It is the control that says the mutation
    leg is discriminating rather than merely sensitive: a harness that diverged on any
    edit at all would fail here.
    """
    base = family.folded_fused_hd_pair_source(codes)
    for index, register in enumerate(("a", "b", "c")):
        base = needle(base, f"    float {register}   = own.a{index};\n",
                      f"    float {register}   = ho{index}[ii];\n")
    return base


# ---------------------------------------------------------------------------
# Device legs
# ---------------------------------------------------------------------------

def leg_product(steps: int) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for index, (name, keywords) in enumerate(CASES, start=1):
        started = time.time()
        driver = build_driver(keywords, 90000 + index)
        codes = codes_of(driver)
        result = run_product(driver, steps)
        result.update({"case": name, "codes": list(codes),
                       "codes_label": codes_label(codes),
                       "shape": list(driver.grid.shape)})
        rows[name] = result
        log(f"  product {index}/{len(CASES)} {name} ({codes_label(codes)}, "
            f"{list(driver.grid.shape)}): passed={result['passed']} "
            f"identical={result['bit_identical']} "
            f"steps={result['steps_compared']}/{steps} "
            f"words={result.get('words_compared')} ({time.time() - started:.1f} s)")
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows}


def leg_codes_source(steps: int) -> Dict[str, Any]:
    """THE FOLD-SPECIFIC STRUCTURAL MUTATION: the plain product's codes expression.

    Built through the plan's own ``codes`` seam, so what is driven is the kernel a
    module that took its triple from ``_boundary_kinds`` would have compiled. Required
    to DIVERGE on every folded fixture, with the shipped ``folded_axis_kinds`` codes as
    the control that must not.
    """
    rows: Dict[str, Any] = {}
    for index, (name, keywords) in enumerate(CASES, start=1):
        driver = build_driver(keywords, 41000 + index)
        shipped = codes_of(driver)
        wrong = boundary_kinds_codes(driver)
        if tuple(wrong) == tuple(shipped):
            rows[name] = {"passed": False, "shipped_codes": list(shipped),
                          "boundary_kinds_codes": list(wrong),
                          "note": "the two code sources agree here; the mutation is "
                                  "not armed on this fixture"}
            continue
        result = run_product(driver, min(steps, 8), codes=wrong,
                             stop_on_divergence=True)
        caught = not result["bit_identical"]
        rows[name] = {
            "passed": bool(caught),
            "caught": caught,
            "shipped_codes": list(shipped), "shipped_label": codes_label(shipped),
            "boundary_kinds_codes": list(wrong),
            "steps_compared": result.get("steps_compared"),
            **divergence(result),
        }
        log(f"  codes_source {index}/{len(CASES)} {name}: shipped="
            f"{codes_label(shipped)} wrong={list(wrong)} caught={caught}")
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows,
            "note": ("stepping._boundary_kinds reports 'mirror' on a folded axis, "
                     "which `1 if kind == \"metallic\" else 0` maps to 0 = PERIODIC: "
                     "the backward ghost wraps to the far plane and the cell-0 mask "
                     "is not emitted")}


def _sub_step_state(driver: Any, plan: Any, residency: Residency) -> Dict[str, Any]:
    """``D`` and ``fu_D`` immediately after one launch, before any fill runs."""
    residency.sync_in()
    plan.run()
    residency.sync_out()
    return {name: np.array(getattr(driver.fields, name), copy=True)
            for name in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")}


def leg_sub_step() -> Dict[str, Any]:
    """THE MASKS, MEASURED WHERE THEY ARE VISIBLE.

    ``D`` and ``fu_D`` are compared IMMEDIATELY after the fused launch -- before
    ``fill_D``, ``zero_metal_D`` and ``fill_folded_far_ghosts_D`` -- against the two
    certified folded singles run to the same point on an identically seeded engine.
    The shipped kernel must agree word for word; both mask mutations must DIVERGE
    here. Their whole-step verdict is recorded by the mutation leg as a predicted null
    with the citation, never scored as an uncaught defect.
    """
    rows: Dict[str, Any] = {}
    for name in SUB_STEP_CASES:
        keywords = dict(CASES)[name]
        codes = codes_of(build_driver(keywords, 1, scale_bits=0))

        def one(functions: Optional[Mapping[str, Any]] = None,
                seed: int = 6060) -> Dict[str, Any]:
            driver = build_driver(keywords, seed)
            residency = Residency()
            plan = build_weld(driver, residency, functions=functions)
            return _sub_step_state(driver, plan, residency)

        def singles(seed: int = 6060) -> Dict[str, Any]:
            driver = build_driver(keywords, seed)
            residency = Residency()
            constitutive = symmetry.plan_folded_constitutive(
                driver.fields, driver.pml, "H", residency)
            curl = symmetry.plan_folded_pml_curl(
                driver.fields, driver.pml, "step_D", residency)
            residency.sync_in()
            constitutive.run()
            curl.run()
            residency.sync_out()
            return {volume: np.array(getattr(driver.fields, volume), copy=True)
                    for volume in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")}

        reference = singles()
        shipped = one()
        base = {volume: differing(reference[volume], shipped[volume])
                for volume in reference}
        entry: Dict[str, Any] = {
            "codes": list(codes), "codes_label": codes_label(codes),
            "shipped_differing_words": int(sum(base.values())),
            "shipped_per_volume": {k: int(v) for k, v in base.items() if v},
            "mask_mutations": {},
        }
        for mutant, (source, why) in mask_mutations(codes).items():
            functions = {shaders.CONTRACT_OFF:
                         compile_source(source).fused_hd_pair_step}
            state = one(functions=functions)
            moved = {volume: differing(reference[volume], state[volume])
                     for volume in reference}
            total = int(sum(moved.values()))
            entry["mask_mutations"][mutant] = {
                "caught_at_sub_step": total > 0,
                "differing_words": total,
                "per_volume": {k: int(v) for k, v in moved.items() if v},
                "why": why}
            log(f"  sub_step {name}/{mutant}: {total} differing words at sub-step")
        entry["passed"] = bool(
            entry["shipped_differing_words"] == 0
            and entry["mask_mutations"]
            and all(row["caught_at_sub_step"]
                    for row in entry["mask_mutations"].values()))
        rows[name] = entry
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows,
            "note": ("the masks are invisible after a complete step -- the driver's "
                     "fill passes overwrite exactly the planes they protect "
                     "(symmetry.py:55-60) -- so a whole-step gate would certify a "
                     "mask-less kernel as correct")}


def leg_host_defect(steps: int, name: str,
                    launcher: Callable[[Any, Residency, Any], Any],
                    why: str, cases: Sequence[str]) -> Dict[str, Any]:
    """One HOST defect, driven on the named fixtures; each must DIVERGE."""
    rows: Dict[str, Any] = {}
    for case in cases:
        driver = build_driver(dict(CASES)[case], 31000 + len(case))
        result = run_product(driver, min(steps, 6), launcher=launcher,
                             stop_on_divergence=True)
        caught = not result["bit_identical"]
        report = divergence(result)
        rows[case] = {"passed": bool(caught), "caught": caught, **report}
        log(f"  {name} {case}: caught={caught} at step "
            f"{report['weld_first_divergence']} "
            f"({report['weld_differing_words']} words)")
    return {"passed": all(row["passed"] for row in rows.values()),
            "defect": name, "why": why, "cases": rows}


def leg_arbitration() -> Dict[str, Any]:
    """The composer, asked, with this product REGISTERED.

    WHAT THIS PINS IS AN OUTCOME AND NOT A MECHANISM: on every folded fixture the
    released ``folded_fused_magnetic_pair`` holds ``step_B`` and ``update_H``, this
    product installs NOWHERE, and the composer names its refusal. The registration is
    what makes the question non-trivial -- an unregistered row cannot be refused --
    and the family module registers on import, so importing it (which this gate does)
    is the registration.
    """
    rows: Dict[str, Any] = {}
    registered = [(spec.family, spec.label, spec.wired)
                  for spec in _registered_on("update_H")]
    for name, keywords in CASES:
        driver = build_driver(keywords, 8080, scale_bits=0)
        plan, _residency = _composed(driver, fuse=True)
        selected = dict(plan.selected)
        refusal = list(plan.reasons.get(f"fused_pair_{family.FAMILY}", ()))
        installed = any(family.FAMILY in str(value) for value in selected.values())
        incumbent = selected.get("update_H")
        rows[name] = {
            # THE INCUMBENT'S LABEL IS MEASURED, NOT SPELLED FROM A DESIGN NOTE. The
            # released B->H product registers as `folded fused B/H pair` on this
            # backend (measured 2026-09-07 through the composer on all nine fixtures);
            # a design note put it at `fused pair B (folded)`, which is the D->E
            # sibling's convention and would have made this leg red for the wrong
            # reason. What the leg pins is the OUTCOME: some released folded B->H pair
            # holds update_H on every fixture and this product holds nothing.
            "passed": bool(not installed and incumbent
                           and "folded" in str(incumbent)
                           and family.FAMILY not in str(incumbent)),
            "selected": selected,
            "this_product_installed": installed,
            "update_H_incumbent": incumbent,
            "composer_reason_for_this_product": refusal,
            "launches_over_the_four_slots": sum(
                1 for slot in ("step_B", "update_H", "step_D", "update_E")
                if slot in plan.plans
                and getattr(plan.plans[slot], "performs_device_work", True)),
        }
        log(f"  arbitration {name}: update_H -> {incumbent!r}, "
            f"this product installed={installed}")
    return {
        "passed": all(row["passed"] for row in rows.values()),
        "cases": rows,
        "registered_arms_on_update_H": registered,
        "this_product_is_registered": any(f == family.FAMILY for f, _l, _w
                                          in registered),
        "this_product_is_wired": any(w for f, _l, w in registered
                                     if f == family.FAMILY),
        "installable_flag": family.INSTALLABLE,
        "absorb_row": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
        "note": ("the row is registered UNWIRED and, since the 2026-09-07 wiring, "
                 "holds a FUSED_PAIR_ARMS row -- so the seam loop ASKS it and then "
                 "refuses it on INSTALLABLE = False before it reaches the gate or "
                 "the predicate, and with that flag out of the way on the released "
                 "B->H pair, which keeps update_H on every fixture"),
    }


def _registered_on(slot: str) -> Tuple[Any, ...]:
    from meep_gpu.metal_kernels import arms  # noqa: PLC0415

    return arms.registered(slot)


def leg_sync(steps: int) -> Dict[str, Any]:
    """The sync channel: a weld spanning ``step_D`` must DECLINE it, and answering it
    must diverge.

    ``synchronize_magnetic_fields`` (driver.py:4350-4392) repeats ``step_B``,
    ``fill_B``, the far pass and ``update_H`` and consults ``update_H_synchronize``.
    A product spanning into ``step_D`` that answered would run the D curl inside the
    magnetic half-step -- a pass the half-step does not perform -- so ``fastpath``'s
    containment rule refuses it by SPAN. The rule is derived, not this file's opinion;
    this leg drives both arms of it.

    NO METAL ADAPTER HAD EVER BEEN ASKED ``update_H_synchronize`` BEFORE THIS LEG.
    """
    driver = build_driver(dict(CASES)[MUTATION_CASE], 7171)
    hazard = arrangement_weld(driver, sync_hazard=True, name="weld_hazard")
    declining = arrangement_weld(driver, sync_hazard=False, name="weld_declining")
    arrangements = {"array": FoldedArrangement("array", None, None),
                    "weld_declining": declining, "weld_hazard": hazard}

    def hook(step: int, name: str, engine: Any) -> Any:
        # THE ENGINE'S OWN ACCESSOR, which is what reaches
        # `synchronize_magnetic_fields` and therefore the by-name consult. A hook that
        # merely read an array would never touch the sync channel at all, and both
        # arms would come back with zero consults -- a leg reporting a green control
        # against an unarmed hazard.
        if step in SYNC_STEPS:
            return {"step": step, "flux_z": float(engine.flux_in_box(2))}
        return None

    result = drive(driver, arrangements, max(SYNC_STEPS) + 1, per_step_hook=hook,
                   stop_on_divergence=False)
    legs = result["arrangements"]
    first = legs["weld_hazard"]["first_divergence"]
    hazard_rows = legs["weld_hazard"]["per_step"]
    diverged_in_D = bool(first is not None and any(
        any(name.startswith(("D", "fu_D")) for name in row["differing_volumes"])
        for row in hazard_rows if row["differing_words"]))
    log(f"  sync declining: identical={legs['weld_declining']['identical']} "
        f"refusals={legs['weld_declining']['sync_refusals']}")
    log(f"  sync hazard: answered={legs['weld_hazard']['sync_answered']} "
        f"first_divergence={first} in_D={diverged_in_D}")
    return {
        "passed": bool(legs["weld_declining"]["identical"]
                       and legs["weld_declining"]["sync_refusals"] == len(SYNC_STEPS)
                       and legs["weld_hazard"]["sync_answered"] == len(SYNC_STEPS)
                       and first is not None and diverged_in_D
                       and result["step_error"] is None),
        "sync_steps": list(SYNC_STEPS),
        "declining_identical": legs["weld_declining"]["identical"],
        "declining_sync_refusals": legs["weld_declining"]["sync_refusals"],
        "hazard_sync_answered": legs["weld_hazard"]["sync_answered"],
        "hazard_first_divergence": first,
        "hazard_diverges_in_D_or_fu_D": diverged_in_D,
        "hazard_differing_volumes_at_first_divergence": next(
            (row["differing_volumes"] for row in hazard_rows
             if row["differing_words"]), {}),
        "flux_values": {name: legs[name]["hook"] for name in legs},
        "array_flux_values": result["reference_hook"],
        "sync_path_slots": sorted(SYNC_PATH_SLOTS),
        "span_slots_outside_it": [n for n in family.REPLACES
                                  if n not in SYNC_PATH_SLOTS],
        "steps_compared": result["steps_compared"],
        "step_error": result["step_error"],
        "citation": (
            "a product installed at update_H that also advances D corrupts D on every "
            "flux_in_box and field_energy_in_box call unless it declines the "
            "update_H_synchronize consult; NO METAL ADAPTER HAD EVER BEEN ASKED THAT "
            "CONSULT before this leg"),
    }


def leg_launch_structure(steps: int) -> Dict[str, Any]:
    """Launches per step at the seam and over the whole step, TWO counters each.

    Reported, not argued. Over the ``step_B - update_H - step_D - update_E`` path
    launches are ``4 - (installed pairs)``: the composition installed today runs the
    released B->H pair plus a D->E product where one admits, and this weld runs one
    pair plus two singles. The honest number is that the weld does NOT reduce the
    step's launches against what the composer builds -- which is the arbitration
    ruling reproduced as a measurement.
    """
    driver = build_driver(dict(CASES)[MUTATION_CASE], 1212)
    arrangements = all_arrangements(driver)
    result = drive(driver, arrangements, min(steps, 4), stop_on_divergence=False)
    rows: Dict[str, Any] = {}
    for name, arrangement in arrangements.items():
        counts = arrangement.launches()
        leg = result["arrangements"].get(name, {})
        rows[name] = {
            "launches_by_the_plans": counts["plans"],
            "launches_by_the_function_wrappers": counts["functions"],
            "counters_agree": counts["plans"] == counts["functions"],
            "dispatched": leg.get("dispatched"),
            "absorbed": leg.get("absorbed"),
            "selected": leg.get("selected"),
        }
    steps_stepped = int(result["steps_stepped"]) or 1
    return {
        "passed": all(row["counters_agree"] for row in rows.values()
                      if row["launches_by_the_plans"]),
        "steps": steps_stepped,
        "arrangements": rows,
        "seam_launches_per_step": {
            "weld": 1,
            "certified_folded_singles": 2,
        },
        "whole_step_launches_per_step": {
            name: round(row["launches_by_the_plans"] / steps_stepped, 3)
            for name, row in rows.items() if row["launches_by_the_plans"]},
        "note": ("the weld saves one launch AT THE SEAM against the two certified "
                 "folded singles and saves nothing against the composition the "
                 "composer installs, which already spans update_H with the released "
                 "B->H pair. Launch counts are not time and no timing exists for "
                 "this shape"),
    }


def leg_mutation(steps: int, codes: Sequence[int],
                 wall_codes: Sequence[int]) -> Dict[str, Any]:
    """Every armed defect, each of which MUST diverge; predicted nulls recorded."""
    budget = min(steps, 6)
    rows: Dict[str, Any] = {}

    def drive_mutant(case: str, source: str, seed: int) -> Dict[str, Any]:
        driver = build_driver(dict(CASES)[case], seed)
        functions = {shaders.CONTRACT_OFF: compile_source(source).fused_hd_pair_step}
        return run_product(driver, budget, functions=functions,
                           stop_on_divergence=True)

    for index, (name, (source, why)) in enumerate(shader_mutations(codes).items(),
                                                  start=1):
        result = drive_mutant(MUTATION_CASE, source, 20000 + index)
        caught = not result["bit_identical"]
        report = divergence(result)
        rows[name] = {"passed": bool(caught), "caught": caught,
                      "scope": MUTATION_CASE, "why": why, **report}
        log(f"  mutation {name}: caught={caught} at step "
            f"{report['weld_first_divergence']} "
            f"({report['weld_differing_words']} words)")

    for index, (name, (source, why)) in enumerate(
            fold_specific_mutations(wall_codes).items(), start=1):
        result = drive_mutant(WALL_MUTATION_CASE, source, 25000 + index)
        caught = not result["bit_identical"]
        report = divergence(result)
        rows[name] = {"passed": bool(caught), "caught": caught,
                      "scope": WALL_MUTATION_CASE, "why": why, **report}
        log(f"  mutation {name}: caught={caught} at step "
            f"{report['weld_first_divergence']} "
            f"({report['weld_differing_words']} words, {WALL_MUTATION_CASE})")

    for index, (name, (launcher, why)) in enumerate(HOST_MUTATIONS.items(), start=1):
        driver = build_driver(dict(CASES)[MUTATION_CASE], 26000 + index)
        result = run_product(driver, budget, launcher=launcher,
                             stop_on_divergence=True)
        caught = not result["bit_identical"]
        report = divergence(result)
        rows[name] = {"passed": bool(caught), "caught": caught, "scope": "host",
                      "why": why, **report}
        log(f"  mutation host/{name}: caught={caught} at step "
            f"{report['weld_first_divergence']} "
            f"({report['weld_differing_words']} words)")

    # THE TWO MASK MUTATIONS AT WHOLE-STEP GRANULARITY, AND A MEASURED REFUTATION.
    #
    # THEY WERE EXPECTED TO BE NULLS HERE. ``symmetry.py:55-60`` states -- and
    # ``gate_metal_symmetry:55-57`` measured, at 384-1,080 words at sub-step and 0
    # after a complete step -- that the driver's fill passes overwrite exactly the
    # planes both masks protect, so a kernel carrying NEITHER is bytewise-identical
    # after a complete step. This gate's build plan carried that forward and told this
    # leg to record them as predicted nulls.
    #
    # MEASURED HERE 2026-09-07, ON THIS SEAM, THEY BOTH FIRE -- and the attribution
    # says exactly why the earlier measurement and this one are both right.
    # ``drop_cell0_mask`` moves 540 words and ``drop_top_plane_mask`` 264, and EVERY
    # ONE OF THEM IS IN ``fu_D``: zero words move in ``D``. The masks zero ``curlN``,
    # which feeds BOTH the split-field auxiliary ``nN`` and the displacement ``vN``,
    # and ``stepping._fill_symmetry_ghost_cells`` and ``._zero_metal`` walk
    # ``D_CURL_TERMS`` / ``D_COMPONENTS`` and NEVER ``fu_*`` -- so the fills restore
    # the displacement at those planes and nothing restores the auxiliary.
    #
    # So the masks ARE invisible in D after a complete step, exactly as recorded, and
    # they are VISIBLE in fu_D, which the earlier measurement's comparison did not
    # carry. They are therefore SCORED here rather than excused, and the attribution
    # is asserted rather than reported: each must move fu_D and must NOT move D. That
    # is a strictly stronger check than either "it fires" or "it is null".
    mask_rows: Dict[str, Any] = {}
    for index, (name, (source, why)) in enumerate(mask_mutations(codes).items(),
                                                  start=1):
        result = drive_mutant(MUTATION_CASE, source, 27000 + index)
        report = divergence(result)
        volumes = report.get("weld_differing_volumes") or {}
        in_d = sum(int(n) for volume, n in volumes.items()
                   if volume in ("Dx", "Dy", "Dz"))
        in_fu = sum(int(n) for volume, n in volumes.items()
                    if volume in ("fu_Dx", "fu_Dy", "fu_Dz"))
        fired = not result["bit_identical"]
        mask_rows[name] = {
            "passed": bool(fired and in_fu > 0 and in_d == 0),
            "caught": fired, "scope": MUTATION_CASE, "why": why,
            "words_in_fu_D": in_fu, "words_in_D": in_d,
            "attribution": (
                "the fills restore D at the masked plane and never touch fu_*, so the "
                "mask is invisible in D after a complete step and visible in fu_D"),
            **report}
        rows[f"mask/{name}"] = mask_rows[name]
        log(f"  mutation mask/{name}: caught={fired} at step "
            f"{report['weld_first_divergence']} "
            f"({in_fu} words in fu_D, {in_d} in D)")

    predicted: Dict[str, Any] = {
        "the_masks_were_predicted_null_at_whole_step_and_are_not": {
            "predicted": False,
            "prediction_held": False,
            "measured": {name: {"words_in_fu_D": row["words_in_fu_D"],
                                "words_in_D": row["words_in_D"]}
                         for name, row in mask_rows.items()},
            "citation_that_was_carried_forward":
                "symmetry.py:55-60 and gate_metal_symmetry:55-57 -- the driver's fill "
                "passes overwrite exactly the planes the two masks protect",
            "why_both_are_right":
                "the masks zero curlN, which feeds the split-field auxiliary nN as "
                "well as the displacement vN; stepping._fill_symmetry_ghost_cells and "
                "._zero_metal walk D_CURL_TERMS / D_COMPONENTS and never fu_*, so the "
                "displacement is restored at those planes and the auxiliary is not. "
                "Invisible in D, visible in fu_D -- and this gate's comparison carries "
                "every stored volume, so it sees the half the earlier one did not",
            "consequence":
                "both mask mutations are SCORED at whole step here, with the "
                "attribution asserted (fu_D must move, D must not), and leg sub_step "
                "keeps the finer measurement where D moves too"},
    }

    # AND THE EARNED NULL: the mirror-source ghost, plus the same edit with the mask
    # dropped, which MUST fire. A null that is only reported is an untested claim.
    for index, (name, (source, why, earned)) in enumerate(
            predicted_null_mutations(codes).items(), start=1):
        result = drive_mutant(MUTATION_CASE, source, 28000 + index)
        fired = not result["bit_identical"]
        entry: Dict[str, Any] = {"fired_at_whole_step": fired, "predicted": False,
                                 "prediction_held": not fired, "why": why}
        if earned:
            control = drive_mutant(MUTATION_CASE, earned, 29000 + index)
            entry["paired_control_with_the_mask_dropped_fires"] = (
                not control["bit_identical"])
            entry["paired_control_first_divergence"] = divergence(
                control)["weld_first_divergence"]
            entry["null_is_earned"] = bool(not fired
                                           and not control["bit_identical"])
        predicted[name] = entry
        log(f"  predicted-null {name}: fired={fired} "
            f"earned={entry.get('null_is_earned')}")

    # THE PREDICTIONS ARE NOT ALL EXPECTED TO HOLD ANY MORE -- one of them is a
    # RECORDED REFUTATION and is scored above instead. What must still hold is that
    # every entry still CLASSED as a null held, and that each held null is EARNED by a
    # paired control that fires; a null nobody could break is an untested claim.
    nulls_ok = all(entry["prediction_held"] for entry in predicted.values()
                   if entry.get("scored_by") != "the mutation table above"
                   and "measured" not in entry)
    earned_ok = all(entry.get("null_is_earned", True)
                    for entry in predicted.values())
    return {
        "passed": bool(all(row["passed"] for row in rows.values())
                       and nulls_ok and earned_ok),
        "armed": len(rows), "caught": sum(1 for row in rows.values() if row["caught"]),
        "mutations": rows,
        "predicted_null": predicted,
        "every_prediction_held": nulls_ok,
        "every_null_is_earned_by_a_paired_control": earned_ok,
        "mutation_case": MUTATION_CASE, "wall_mutation_case": WALL_MUTATION_CASE,
    }


def leg_byte_neutral(steps: int, codes: Sequence[int]) -> Dict[str, Any]:
    """The one armed edit required NOT to diverge."""
    driver = build_driver(dict(CASES)[MUTATION_CASE], 33333)
    source = byte_neutral_source(codes)
    functions = {shaders.CONTRACT_OFF: compile_source(source).fused_hd_pair_step}
    result = run_product(driver, min(steps, 8), functions=functions)
    return {"passed": bool(result["bit_identical"]),
            "identical": result["bit_identical"],
            **divergence(result),
            "edit": "the three own-cell register reads replaced by reloads of the "
                    "scratch this thread just stored",
            "why_it_must_not_fire": "a thread reading a value it itself wrote, at a "
                                    "point no other thread can have intervened"}


def leg_disarm(steps: int, codes: Sequence[int]) -> Dict[str, Any]:
    """The identical harness, SHIPPED BYTES, must not diverge.

    Without it a mutation reported as caught could be a harness that diverges anyway.
    The shipped source is compiled and handed through the SAME ``functions`` seam the
    mutants use, so the only difference between this row and a mutation row is the
    bytes.
    """
    driver = build_driver(dict(CASES)[MUTATION_CASE], 20000)
    source = family.folded_fused_hd_pair_source(codes)
    functions = {shaders.CONTRACT_OFF: compile_source(source).fused_hd_pair_step}
    result = run_product(driver, min(steps, 6), functions=functions)
    return {"passed": bool(result["passed"] and result["bit_identical"]),
            "identical": result["bit_identical"],
            "steps_compared": result.get("steps_compared"),
            **divergence(result),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}


# ---------------------------------------------------------------------------
# The battery hook — one lifted corpus row
# ---------------------------------------------------------------------------

def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate this battery at all, or ``[]``."""
    return list(_shared.runtime_reasons())


def _child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_FOLDED_HD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_FOLDED_HD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:  # noqa: ARG001
    """The census driver's battery hook: the four-reference identity on ONE lifted row."""
    steps = int(os.environ.get("MEEP_GPU_FOLDED_HD_GATE_STEPS", 12))
    max_cells = os.environ.get("MEEP_GPU_FOLDED_HD_GATE_MAX_CELLS")
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps,
        "n_sources": len(sources),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(
                         withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources]}
    pin_array_path(driver)
    verdict = family.metal_folded_fused_hd_pair_coverage(
        driver.fields, driver.pml, sources, Residency())
    block["predicate_admits"] = bool(verdict.covered)
    block["predicate_reasons"] = list(verdict.reasons)
    with contextlib.suppress(Exception):
        codes, _ = symmetry.folded_axis_kinds(driver.grid, driver.pml)
        block["codes"] = list(int(c) for c in (codes or ()))
        block["codes_label"] = codes_label(codes or ())
    composed, _residency = _composed(driver, fuse=True)
    block["composer_selected"] = dict(composed.selected)
    block["composer_refuses_this_product"] = list(
        composed.reasons.get(f"fused_pair_{family.FAMILY}", ()))
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    block["grid_shape"] = list(driver.grid.shape)
    if not verdict.covered:
        block["driven"] = False
        block["why_not_driven"] = "the predicate refused this row"
        return {"metal_folded_fused_hd_pair_gate": block}
    if max_cells and cells > int(max_cells):
        block["driven"] = False
        block["why_not_driven"] = (
            f"{cells} cells exceeds MEEP_GPU_FOLDED_HD_GATE_MAX_CELLS={max_cells}; "
            f"refused rather than run partially")
        return {"metal_folded_fused_hd_pair_gate": block}
    determinism = _shared.waveform_determinism(driver, steps)
    block["waveform_determinism"] = determinism
    if not determinism["deterministic"]:
        block["driven"] = False
        block["undefined_by_construction"] = True
        block["why_not_driven"] = determinism["reason"]
        _child_progress(f"REFUSED as undefined: {determinism['reason'][:160]}")
        return {"metal_folded_fused_hd_pair_gate": block}
    block["driven"] = True
    block.update(run_product(driver, steps, progress=_child_progress,
                             movement_floor="seam", require_full_budget=False,
                             clean_floor=1))
    _child_progress(f"passed={block['passed']} words={block.get('words_compared')}")
    return {"metal_folded_fused_hd_pair_gate": block}


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing census puts in THIS product's cell, and the seam facts.

    DERIVED from the census's own ``plan_step.selected`` -- the arms the composer chose
    for each row -- never from a list here. The seam record supplies each row's
    interpreter and module and its ``withdraw_in_seam`` flag, which names the rows the
    predicate must refuse.
    """
    census = results / CENSUS
    seam = results / SEAM_RECORD
    if not census.is_dir() or not seam.is_dir():
        raise SystemExit(f"the lift leg needs {census} and {seam}")
    seam_rows = {row["label"]: row for row in _load_jsonl(seam / "h_to_d_seam.jsonl")}
    rows: List[dict] = []
    for leg in ("examples", "tests", "tests_param_matched"):
        path = census / f"{leg}.jsonl"
        if not path.is_file():
            continue
        for record in _load_jsonl(path):
            selected = ((record.get("plan_step") or {}).get("selected") or {})
            if (selected.get("update_H"), selected.get("step_D")) != CELL_ARMS:
                continue
            label = f"{record.get('leg', leg)}:{record['row']}"
            seam_row = seam_rows.get(label, {})
            rows.append({
                "label": label, "leg": record.get("leg", leg), "row": record["row"],
                # THE CASE NAME IS NOT ALWAYS THE ROW NAME, AND THE DIFFERENCE COST
                # THIS CELL A ROW. A `parameterized` test is lifted by the census
                # under its DECORATED name -- `TestEigCoeffs.test_binary_grating_
                # oblique__idx0` -- while the row it files is the expanded one,
                # `..._oblique_0_0_0`. Handing `--child-cases` the row name asks for a
                # case the module does not define: the child exits 0 having selected
                # nothing and writes an EMPTY LIST, which reads as "the harness could
                # not measure this row" rather than as a name it got wrong. Measured
                # 2026-09-07: 1 unmeasured row of 78, and it is the cell's only
                # parametrised one.
                "case": record.get("case") or record["row"],
                "module": record.get("module") or seam_row.get("module"),
                "interpreter": (record.get("interpreter") or record.get("python")
                                or seam_row.get("interpreter") or sys.executable),
                "grid_cells": record.get("grid_cells"),
                "grid_shape": record.get("grid_shape"),
                "withdraw_in_seam": bool((seam_row.get("h_to_d_seam") or {})
                                         .get("withdraw_in_seam")),
                "census_selected": selected,
            })
    unique = {row["label"]: row for row in rows}
    facts = {"census": CENSUS, "seam_record": SEAM_RECORD, "cell_arms": list(CELL_ARMS),
             "rows_in_cell": len(unique),
             "rows_with_a_standing_withdraw": sorted(
                 label for label, row in unique.items() if row["withdraw_in_seam"])}
    return list(unique.values()), facts


def _carries_a_measurement(path: Path) -> bool:
    """Did a child's record actually measure this row? Not "does the file exist"."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable record is not a measurement
        return False
    candidates = payload if isinstance(payload, list) else [payload]
    return any("metal_folded_fused_hd_pair_gate" in (entry or {})
               for entry in candidates)


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Every corpus row in the cell, re-lifted in its own interpreter and driven."""
    import gate_provenance  # noqa: PLC0415
    import measure_predicate_coverage as census  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(HERE / "results")
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["label"] in wanted]
    lift_dir = out_dir / "lift"
    per_row = lift_dir / "per_row"
    per_row.mkdir(parents=True, exist_ok=True)
    progress_log = lift_dir / "steps.progress.log"
    examples_dir = Path(census.EXAMPLES_DIR)
    tests_dir = Path(census.TESTS_DIR)
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        return {"passed": False, "facts": facts,
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; "
                           f"set MEEP_GPU_CORPUS_ROOT to the checkout the census was "
                           f"cut over. Refused rather than measured on nothing")}
    probe_path = API_ROOT / census.PROBE
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": str(API_ROOT),
                        "MEEP_GPU_SUBNORMAL_POLICY": "flush",
                        "MEEP_GPU_FOLDED_HD_GATE_STEPS": str(steps),
                        "MEEP_GPU_FOLDED_HD_GATE_PROGRESS": str(progress_log)})
    if max_cells is not None:
        environment["MEEP_GPU_FOLDED_HD_GATE_MAX_CELLS"] = str(max_cells)
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    # THE CENSUS'S OWN SHIM, for the six MEEP test modules that decorate with
    # ``parameterized`` (not installed here). Without it every row from one of those
    # modules dies on the IMPORT and comes back ``measured: false`` -- a row scored as
    # unmeasurable by the harness rather than by the engine.
    shim_path = Path(census.__file__).resolve().parent / "shim"
    needs_shim: set = set()
    for row in rows:
        module_name = row.get("module")
        if row["leg"] == "examples" or not module_name:
            continue
        module_file = tests_dir / module_name
        if module_file.is_file() and "import parameterized" in module_file.read_text(
                encoding="utf-8", errors="replace"):
            needs_shim.add(module_name)
    measured: List[dict] = []
    for index, row in enumerate(rows, start=1):
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_")
                                 + ".json")
        environment["MEEP_GPU_FOLDED_HD_GATE_LABEL"] = row["label"]
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{API_ROOT}"
                                     if row.get("module") in needs_shim
                                     else str(API_ROOT))
        started = time.time()
        # A RESUME THAT REUSES AN EMPTY RECORD CAN NEVER RECOVER. The child writes its
        # JSON whatever happens, so "the file exists" is not "the row was measured" --
        # an empty list or a payload with no gate block is exactly the state a resume
        # exists to retry.
        reusable = resume and record_path.exists() and _carries_a_measurement(
            record_path)
        if not reusable:
            if row["leg"] == "examples":
                command = [row["interpreter"], "-u",
                           str(Path(census.__file__).resolve()),
                           "--leg", "examples", "--child-script",
                           str(examples_dir / row["row"]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", Path(__file__).stem]
            elif not row["module"]:
                log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module "
                    f"recorded for this tests row")
                measured.append({**row, "measured": False,
                                 "note": "no module recorded"})
                continue
            else:
                command = [row["interpreter"], "-u",
                           str(Path(census.__file__).resolve()),
                           "--leg", "tests", "--child-module",
                           str(tests_dir / row["module"]),
                           "--child-cases", json.dumps([row.get("case")
                                                        or row["row"]]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", Path(__file__).stem]
            log(f"lift {index}/{len(rows)} {row['label']} start "
                f"(cells {row.get('grid_cells')})")
            stderr_text = ""
            note = "child died"
            try:
                completed = subprocess.run(command, cwd=str(work), env=environment,
                                           timeout=timeout, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, check=False)
                stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired as expired:
                note = "child timeout"
                stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                               if expired.stderr else "")
            if not record_path.exists():
                died = {"row": row["row"], "measured": False, "note": note,
                        "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str), encoding="utf-8")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else [payload]
        wanted = {row["row"], row.get("case") or row["row"]}
        record = next((r for r in candidates if r.get("row") in wanted),
                      candidates[0] if candidates else {})
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get("metal_folded_fused_hd_pair_gate") or {}
        log(f"lift {index}/{len(rows)} {row['label']}: "
            f"measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} codes={block.get('codes_label')} "
            f"steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    with (lift_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record, default=str) + "\n")

    blocks = {r["label"]: (r.get("metal_folded_fused_hd_pair_gate") or {})
              for r in measured}
    admitted = sorted(label for label, b in blocks.items()
                      if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items()
                     if b.get("predicate_admits") is False)
    refused_by_the_withdraw = sorted(
        label for label in refused
        if any("standing integrated" in r
               for r in blocks[label].get("predicate_reasons", ())))
    expected_refused = sorted(facts["rows_with_a_standing_withdraw"])
    if only:
        expected_refused = sorted(set(expected_refused) & set(blocks))
    unmeasured = sorted(label for label, b in blocks.items()
                        if "predicate_admits" not in b)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    undefined = {label: blocks[label].get("why_not_driven")
                 for label in admitted
                 if blocks[label].get("undefined_by_construction")}
    not_driven = {label: blocks[label].get("why_not_driven")
                  for label in admitted
                  if not blocks[label].get("driven") and label not in undefined}
    diverged = sorted(label for label in driven
                      if not blocks[label].get("bit_identical"))
    below_floor = sorted(label for label in driven
                         if blocks[label].get("bit_identical")
                         and int(blocks[label].get("steps_compared") or 0)
                         < LIFT_CLEAN_STEP_FLOOR)
    words_compared = int(sum(int(blocks[label].get("words_compared") or 0)
                             for label in driven))
    steps_compared = int(sum(int(blocks[label].get("steps_compared") or 0)
                             for label in driven))
    codes_seen: Dict[str, int] = {}
    for label in blocks:
        key = blocks[label].get("codes_label")
        if key:
            codes_seen[key] = codes_seen.get(key, 0) + 1
    return {
        "passed": bool(not unmeasured and not diverged
                       and refused_by_the_withdraw == expected_refused
                       and sorted(refused) == expected_refused
                       and driven and not not_driven),
        "facts": facts,
        "rows_measured": len(measured),
        "admitted": len(admitted), "refused": len(refused),
        "refused_labels": refused,
        "refused_by_the_standing_withdraw": refused_by_the_withdraw,
        "expected_refused": expected_refused,
        "unmeasured": unmeasured,
        "driven": len(driven), "driven_and_identical": len(passed_rows),
        "diverged": diverged,
        "undefined_by_construction": undefined,
        "admitted_but_not_driven": not_driven,
        "rows_below_the_clean_step_floor": below_floor,
        "clean_step_floor": LIFT_CLEAN_STEP_FLOOR,
        "complete_driver_steps_compared": steps_compared,
        "words_compared": words_compared,
        "boundary_triples_driven": dict(sorted(codes_seen.items())),
        "rows_jsonl": str(lift_dir / "rows.jsonl"),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="artifact JSON path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--legs", default="all",
                        help="comma-separated leg names, a group name (host, "
                             "compile, device), or 'all'")
    parser.add_argument("--lift-steps", type=int, default=12)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=900.0)
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--lift-only", default=None,
                        help="comma-separated row labels")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.legs == "all":
        legs = ALL_LEGS
    elif args.legs in LEG_GROUPS:
        legs = LEG_GROUPS[args.legs]
    else:
        legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())
    unknown = sorted(set(legs) - set(ALL_LEGS))
    if unknown:
        raise SystemExit(f"unknown legs {unknown}; known legs are {list(ALL_LEGS)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    jsonl.write_text("", encoding="utf-8")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    def record(leg: str, payload: Dict[str, Any]) -> None:
        row = {"leg": leg, **payload}
        rows.append(row)
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        log(f"LEG {leg}: {'PASS' if row.get('passed') else 'FAIL'} "
            f"({time.perf_counter() - started:.1f}s)")

    codes = codes_of(build_driver(dict(CASES)[MUTATION_CASE], 1, scale_bits=0))
    wall_codes = codes_of(build_driver(dict(CASES)[WALL_MUTATION_CASE], 1,
                                       scale_bits=0))
    log(f"mutation specialisation {MUTATION_CASE} -> {codes_label(codes)} {codes}; "
        f"wall {WALL_MUTATION_CASE} -> {codes_label(wall_codes)} {wall_codes}")
    report = subnormal.mps_policy_report()

    if "driver_order" in legs:
        record("driver_order", leg_driver_order())
    if "transcription" in legs:
        record("transcription", leg_transcription(codes))
    if "fixture_shape" in legs:
        record("fixture_shape", leg_fixture_shape())
    if "purity" in legs:
        record("purity", leg_purity())
    if "phase_blind" in legs:
        record("phase_blind", leg_phase_blind())
    if "refusal" in legs:
        record("refusal", leg_refusal())
    if "binding_ceiling" in legs:
        record("binding_ceiling", leg_binding_ceiling())
    if "mutants_compile" in legs:
        record("mutants_compile", leg_mutants_compile(codes, wall_codes))
    if "product" in legs:
        record("product", leg_product(args.steps))
    if "codes_source" in legs:
        record("codes_source", leg_codes_source(args.steps))
    if "sub_step" in legs:
        record("sub_step", leg_sub_step())
    if "race" in legs:
        launcher, why = HOST_MUTATIONS["H_and_f_w_H_written_in_place"]
        record("race", leg_host_defect(args.steps, "H_and_f_w_H_written_in_place",
                                       launcher, why,
                                       [name for name, _ in CASES]))
    if "rotation" in legs:
        launcher, why = HOST_MUTATIONS["rotation_skipped"]
        record("rotation", leg_host_defect(args.steps, "rotation_skipped", launcher,
                                           why, SUB_STEP_CASES))
    if "arbitration" in legs:
        record("arbitration", leg_arbitration())
    if "sync" in legs:
        record("sync", leg_sync(args.steps))
    if "launch_structure" in legs:
        record("launch_structure", leg_launch_structure(args.steps))
    if "lift" in legs:
        record("lift", leg_lift(args.out.parent, args.lift_steps,
                                args.lift_max_cells, args.lift_timeout,
                                args.lift_resume,
                                (args.lift_only.split(",") if args.lift_only
                                 else None)))
    if "byte_neutral" in legs:
        record("byte_neutral", leg_byte_neutral(args.steps, codes))
    if "mutation" in legs:
        record("mutation", leg_mutation(args.steps, codes, wall_codes))
    if "disarm" in legs:
        record("disarm", leg_disarm(args.steps, codes))

    ran = tuple(dict.fromkeys(row["leg"] for row in rows))
    complete = all(leg in ran for leg in ALL_LEGS)
    all_passed = bool(rows) and all(row.get("passed") for row in rows)
    if complete:
        verdict = "PASS" if all_passed else "FAIL"
    else:
        verdict = ("INCOMPLETE: not every leg ran -- "
                   f"{'all requested legs passed' if all_passed else 'a requested leg FAILED'}; "
                   f"missing {sorted(set(ALL_LEGS) - set(ran))}")
    import torch  # noqa: PLC0415

    import gate_provenance  # noqa: PLC0415

    mutation_row = next((row for row in rows if row["leg"] == "mutation"), {})
    lift_row = next((row for row in rows if row["leg"] == "lift"), {})
    result = {
        "verdict": verdict,
        "complete": complete,
        "legs_requested": list(legs),
        "legs_run": list(ran),
        "legs_missing": sorted(set(ALL_LEGS) - set(ran)),
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "counts": {
            "product_cases": len(CASES) if "product" in ran else 0,
            "shader_mutations_armed": mutation_row.get("armed"),
            "shader_mutations_caught": mutation_row.get("caught"),
            "host_mutations": len(HOST_MUTATIONS) if "mutation" in ran else 0,
            "predicted_null": len(mutation_row.get("predicted_null") or {}),
            "byte_neutral_controls": 1 if "byte_neutral" in ran else 0,
            "sub_step_cases": len(SUB_STEP_CASES) if "sub_step" in ran else 0,
            "lift_rows_in_cell": (lift_row.get("facts") or {}).get("rows_in_cell"),
            "lift_rows_admitted": lift_row.get("admitted"),
            "lift_rows_refused": lift_row.get("refused"),
            "lift_rows_driven": lift_row.get("driven"),
            "lift_rows_driven_and_identical": lift_row.get("driven_and_identical"),
            "lift_complete_driver_steps": lift_row.get(
                "complete_driver_steps_compared"),
            "lift_words_compared": lift_row.get("words_compared"),
        },
        "mutation_case": MUTATION_CASE,
        "wall_mutation_case": WALL_MUTATION_CASE,
        "mutation_specialisation": {"codes": list(codes),
                                    "codes_label": codes_label(codes),
                                    "wall_codes": list(wall_codes),
                                    "wall_codes_label": codes_label(wall_codes)},
        "references": {
            "1": "the array path under the driver's own loop",
            "2": "the certified folded singles: plan_folded_constitutive('H') then "
                 "plan_folded_pml_curl('step_D'), dispatched",
            "3": "plan_step(fuse=True): the composition the composer installs today",
            "4": "plan_step(fuse=False): the same slots dispatched unfused",
        },
        "signature": {"pointers": family.PACKED_BINDINGS - 1,
                      "packed_bindings": family.PACKED_BINDINGS,
                      "ceiling": MAX_BUFFER_BINDINGS,
                      "headroom": MAX_BUFFER_BINDINGS - family.PACKED_BINDINGS},
        "product": {"family": family.FAMILY, "slot": family.SLOT,
                    "replaces": list(family.REPLACES), "seam": family.SEAM,
                    "installable": family.INSTALLABLE,
                    "hoists_the_withdraw": family.HOISTS_THE_WITHDRAW,
                    "carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
                    "weld_owed": family.WELD_OWED,
                    "absorb_row": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
                    "registered_in_FAMILY_MODULES": _in_family_modules()},
        "corpus": {"census": CENSUS, "seam_record": SEAM_RECORD,
                   "cell_arms": list(CELL_ARMS)},
        "subnormal_policy": report,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "jsonl": str(jsonl),
        "what_this_does_not_license": (
            "installing the product (the composer refuses it by name and the "
            "arbitration leg pins that: the released folded_fused_magnetic_pair holds "
            "update_H on 78 of the cell's 78 rows), flipping HOISTS_THE_WITHDRAW (no "
            "leg here drives a hoisted launch, so the 3 withdraw rows stay refused by "
            "name), any throughput claim (launch counts are not time and no timing "
            "exists for this shape), the (ordinary -> PML) cell, complex storage under "
            "a fold, a folded beta or BFAST run, cylindrical storage folded or not, a "
            "keep-resolved run, or any step past a row's own first banded step -- the "
            "byte claim is over the steps the flush precondition held for, and each "
            "row's count is recorded beside it"),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    gate_provenance.stamp(result)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    log(f"VERDICT {verdict} in {result['elapsed_seconds']:.2f}s; artifact {args.out}")
    if not all_passed:
        return 1
    return 0 if complete else EXIT_INCOMPLETE


def _in_family_modules() -> bool:
    from meep_gpu.metal_kernels import registry  # noqa: PLC0415

    return family.FAMILY in registry.FAMILY_MODULES or \
        Path(family.__file__).stem in registry.FAMILY_MODULES


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
