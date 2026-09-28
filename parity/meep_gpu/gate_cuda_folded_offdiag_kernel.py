"""Byte-identity gate for the FOLDED off-diagonal ``update_E`` CUDA kernel.

THE CLAIM UNDER TEST. ``meep_gpu/cuda_kernels/folded_offdiag_kernels.py`` emits
``update_E_pml_real_folded_offdiag``: the certified off-diagonal
``update_E`` with a MIRROR arm in ``coord_dn`` and a parity weight on that lane
alone. Does it reproduce ``stepping.update_E``, word for word, on a mirror-folded
grid -- and does it still reduce to the certified family where no axis is folded?

WHY THIS GATE EXISTS AND WHAT SPECIFIED IT. ``coverage.FOLDED_OFFDIAG_ROW_MASK_
ADMISSION`` records a 2026-08-20 device sweep of the SHIPPED kernel on folded
grids: 96/96 identical on the configurations where the fold cannot reach the
arithmetic, 352/352 divergent on the complementary ones, and every one of the
~132,514 differing words on the ghost-delta planes with ZERO elsewhere. That
record names the remedy in as many words (``what_would_move_the_twenty``) and
counts what it is worth (20 corpus slots, 19 of them reachable). This gate
measures whether the remedy works.

=============================================================================
WHAT MAKES A CASE NON-VACUOUS, AND WHAT REFUSES ONE
=============================================================================

A gate that cannot fail certifies nothing, and this sub-step has four ways to
be vacuous. Each is measured PER CASE and a case below its floor is REFUSED, not
passed:

* ``oracle_moved`` -- zero-init is a FIXED POINT of this recurrence, so a case
  that moved no output word passes for every tree;
* ``coupling_is_live`` -- ``f_w_Ex - Dx*inv_eps_Ex``, i.e. what the off-diagonal
  rows contributed. A case whose coupling is identically zero is testing the
  certified PLAIN constitutive kernel with extra steps;
* ``folded_axis_absorbs`` -- ``max|kps-1|``, ``max|kms-1|`` ON THE FOLDED AXIS. A
  folded axis whose half-integer profile is the identity everywhere cannot
  distinguish a coefficient-index error on the axis the fold moved, which is
  exactly what the retired refusal claimed would go wrong;
* ``mirror_ghost_is_live`` -- the ghost is ``parity * g[2]``. Where stored row 2
  of a partner volume is identically zero the mirror ghost EQUALS the metallic
  zero it is being distinguished from, and an exact result would be a fact about
  the fixture.

=============================================================================
THE SWEEP'S OWN TWO-SIDEDNESS
=============================================================================

Every case must be BIT-IDENTICAL -- unlike the sweep that specified this work,
whose whole finding was where the shipped kernel diverges. A gate whose every leg
is "must be identical" cannot show that its comparator works, so the two-sidedness
is carried by the MUTATION battery instead, and one mutation in particular:

    ``hand_the_fold_the_metallic_code`` is the SHIPPED kernel's only available
    answer for a folded axis, armed as a host defect. It must be CAUGHT where a
    folded axis is some live row slot's PARTNER and a NULL where it is not -- which
    is ``coverage.offdiag_fold_roles``' rule, re-measured on a device against THIS
    kernel rather than replayed from the earlier record. A discriminator with one
    arm is not a discriminator, so the sweep carries masks that put a folded axis
    in each of its three roles.

``drop_the_parity_weight`` is the second discriminator and the sharpest leg here:
the weight is ``-phase``, so on an EVEN plane it is -1 and dropping it must be
caught, while on an ODD plane it is +1 and dropping it is the identity. A battery
run at one parity would score it a catch and learn nothing about the SIGN.

=============================================================================
THE RELEASE VERDICT IS ITSELF FALSIFIED
=============================================================================

``--legs falsify`` re-runs the whole verdict computation over a sweep whose one
case carries a planted defect, and REQUIRES ``passed`` to come back False. A gate
that has never been shown to refuse is a gate whose pass means nothing.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=7 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_folded_offdiag_kernel.py --backend cupy \\
        --product full --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend EXECUTES the emitted device source through
the slice test module's own evaluator -- one grammar, one copy, imported rather
than duplicated. IT COMPILES NOTHING AND CERTIFIES NOTHING; what it settles is
whether the harness can fail and whether the expression tree is right::

    python -u gate_cuda_folded_offdiag_kernel.py --backend numpy \\
        --product reduced --out /tmp/folded_offdiag_kernel_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import hashlib
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
except ImportError:  # laptop: the evaluator backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_offdiag as flat  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402
from meep_gpu.cuda_kernels import folded_offdiag_kernels as folded  # noqa: E402

log = probe.log
save = probe.save
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census

SEED = 20260820

#: The six arrays this sub-step writes, and everything it touches. Imported from
#: the certified family's gate rather than respelled: two spellings of one output
#: list is how a gate quietly stops comparing an auxiliary.
OUTPUTS: Tuple[str, ...] = flat.OUTPUTS
STATE: Tuple[str, ...] = flat.STATE
COMPONENTS: Tuple[str, ...] = flat.COMPONENTS

#: Consecutive launches in the multi-step leg. 60 is the budget every certified
#: hand-CUDA record is cut at. "Identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: EVERY AXIS IS FOLDED SOMEWHERE, both plane parities appear, both declared
#: terminations appear on each axis, an odd full count appears, and one, two and
#: three simultaneous planes all appear. The two UNFOLDED controls carry the
#: reduction claim.
FOLD_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "unfolded_periodic", "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (8.0, 10.0, 12.0)},
    {"label": "unfolded_metallic", "axes": "", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"),
     "cell": (8.0, 10.0, 12.0)},

    # Folded X: the SLOWEST-stride coefficient index. Both terminations, both
    # plane parities, an odd full count.
    {"label": "fold_X_periodic", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_metallic", "axes": "X", "phase": 1,
     "boundaries": ("metallic", "periodic", "periodic"),
     "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_periodic_odd_plane", "axes": "X", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_metallic_odd_plane", "axes": "X", "phase": -1,
     "boundaries": ("metallic", "periodic", "periodic"),
     "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_periodic_odd_count", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (17.0, 8.0, 12.0)},

    # Folded Y, both terminations and both parities.
    {"label": "fold_Y_periodic", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_metallic_odd_plane", "axes": "Y", "phase": -1,
     "boundaries": ("periodic", "metallic", "periodic"),
     "cell": (8.0, 16.0, 12.0)},

    # Folded Z: the FASTEST-stride coefficient index. A decomposition defect that
    # confuses it with X is only visible if both are folded somewhere.
    {"label": "fold_Z_periodic_odd_plane", "axes": "Z", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (8.0, 12.0, 16.0)},
    {"label": "fold_Z_metallic", "axes": "Z", "phase": 1,
     "boundaries": ("periodic", "periodic", "metallic"),
     "cell": (8.0, 12.0, 16.0)},

    # A FOLD BESIDE A LIVE METALLIC WALL, which no other spec here carries: a
    # folded axis always reports ``wm = 0`` (``is_metallic and not is_mirrored``),
    # so a sweep of folded specs alone leaves every wall flag zero and the wall
    # mask untested on a fold. That is exactly the seam
    # ``offdiag_wall_mask_flags`` is kept separate from ``bc_*`` for.
    {"label": "fold_X_wall_Y", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"),
     "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_Z_wall_X", "axes": "Z", "phase": 1,
     "boundaries": ("metallic", "periodic", "periodic"),
     "cell": (8.0, 12.0, 16.0)},

    # TWO PLANES AT ONCE, one termination each way and both the same way.
    {"label": "fold_XY_mixed", "axes": "XY", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"),
     "cell": (16.0, 16.0, 10.0)},
    {"label": "fold_XZ_periodic_odd_plane", "axes": "XZ", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (16.0, 10.0, 16.0)},

    # THREE PLANES AT ONCE, both terminations. Every one of the six row slots
    # then takes a ghosted down shift.
    {"label": "fold_XYZ_periodic", "axes": "XYZ", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (16.0, 16.0, 16.0)},
    {"label": "fold_XYZ_metallic_odd_plane", "axes": "XYZ", "phase": -1,
     "boundaries": ("metallic", "metallic", "metallic"),
     "cell": (17.0, 16.0, 16.0)},
)

#: The row masks, CHOSEN so that a folded axis lands in each of its three roles.
#: A sweep over corpus masks alone would answer one third of the question and
#: report it as the whole.
ROW_MASK_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "full_tensor", "mask": (1, 1, 1, 1, 1, 1),
     "why": "every axis is some live slot's partner: the fold is always reachable"},
    {"label": "corpus_15row", "mask": (1, 0, 0, 1, 0, 0),
     "why": "the mask the 186-row corpus drives -- slots (Ex,Ey) and (Ey,Ex), so "
            "partner axes X and Y are live and Z is in NEITHER role"},
    {"label": "row_Ex_only", "mask": (1, 1, 0, 0, 0, 0),
     "why": "one row component: X is its OWN axis and never its partner, so a "
            "folded X exercises the own-axis up shift alone"},
    {"label": "single_Ey_Ez", "mask": (0, 0, 1, 0, 0, 0),
     "why": "one slot: own axis Y, partner axis Z -- a folded X is in neither "
            "role, which is the discriminators' NULL arm"},
)

#: The reduced product's specs, NAMED rather than sliced. A head slice of the
#: list is all X folds and no declared wall, which leaves the wall discriminators
#: with one arm and every axis question answered on one axis.
REDUCED_SPEC_LABELS: Tuple[str, ...] = (
    "unfolded_metallic", "fold_X_periodic", "fold_X_metallic_odd_plane",
    "fold_Z_metallic", "fold_X_wall_Y", "fold_XYZ_periodic")

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient, so it is a real second draw of the tables.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: Spatially VARYING row coefficients: a uniform coefficient is what makes the
#: registration rather than merely the rounding invisible.
ROW_FORM = "varying"

#: ``--fmad=false`` is CORRECTNESS on this sub-step, not tuning. The certified
#: sibling's unguarded control DIVERGED on 384/384; this family's row product is
#: that one plus a select, so the same control is carried here rather than
#: inherited.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = flat.GUARD_SETS


# ---------------------------------------------------------------------------
# The role a fold plays, and what that licenses
# ---------------------------------------------------------------------------

def fold_roles(grid, mask: Sequence[int]) -> Dict[str, Any]:
    """Which role each folded axis plays in this row mask.

    ``coverage.offdiag_fold_roles`` is the SHIPPED rule and is asked here rather
    than re-derived, because one of this gate's legs is whether that rule still
    describes a device -- now with a kernel that implements the mirror arm rather
    than one that lacks it.
    """
    mirrored = tuple(bool(grid.is_mirrored(axis)) for axis in range(3))
    roles = dict(coverage.offdiag_fold_roles(tuple(mask), mirrored))
    roles["folded_axes"] = list(roles["folded_axes"])
    roles["live_partner_axes"] = list(roles["live_partner_axes"])
    roles["live_own_axes"] = list(roles["live_own_axes"])
    roles["folded_partner_axes"] = list(roles["folded_partner_axes"])
    roles["folded_own_axes"] = list(roles["folded_own_axes"])
    return roles


def _plane_index(axis: int, index: int, shape) -> Tuple[Any, ...]:
    key: List[Any] = [slice(None)] * 3
    key[axis] = index if index >= 0 else shape[axis] - 1
    return tuple(key)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(xp, spec: Dict[str, Any], courant: float):
    """A frozen ``(fields, layer, grid)`` triple for one fold spec.

    THE THICKNESS RULE IS THE SIBLING FOLD GATES', not a new one: skip an axis too
    thin to hold a layer, and on a MIRRORED axis ask for the HIGH face only.
    ``PML._resolve_mirror_faces`` refuses a named low face on a folded axis
    outright -- cell 0 is the mirror plane, a boundary condition rather than a
    wall -- and ``stepping._require_consistent_pml`` raises if the two disagree.
    Asking for the high face is also what keeps ``folded_axis_absorbs`` above its
    floor.
    """
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                xp=xp, courant=courant)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def folded_axis_absorbs(layer, grid) -> Dict[str, Any]:
    """Does the FOLDED axis's HALF-INTEGER coefficient profile beat the identity?

    ``kps = kms = 1`` is the interior pass-through. An axis whose whole vector is
    that identity cannot distinguish a coefficient-index error on it, so a folded
    case at this floor is measuring every axis except the one the fold touched --
    which is precisely the axis the retired refusal was about.
    """
    tables = flat.half_integer_tables(layer)
    out: Dict[str, Any] = {"folded_axes": [], "max_deviation": 0.0}
    for axis, name in enumerate("xyz"):
        if not grid.is_mirrored(axis):
            continue
        deviation = 0.0
        for stem in ("kps", "kms"):
            values = to_host(tables[f"{stem}_{name}"]).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        out["folded_axes"].append({"axis": axis, "name": name,
                                   "max_deviation_from_identity": deviation})
        out["max_deviation"] = max(out["max_deviation"], deviation)
    out["meets_floor"] = (not out["folded_axes"]) or out["max_deviation"] > 0.0
    return out


def mirror_ghost_is_live(fields, grid) -> Dict[str, Any]:
    """Is the mirror ghost distinguishable from the metallic zero it replaces?

    The ghost is ``parity * g[2]`` (stepping.py:1870-1872, ``MIRROR_SOURCE_INDEX
    = 2``, asked of ``stepping`` rather than spelled: a second spelling of the row
    the ghost reflects from is a second place for it to drift, and this floor's
    whole job is to be about that row). Where stored row 2 of a partner volume is
    identically zero the ghost EQUALS the metallic zero and an exact result would
    be a fact about the fixture.
    """
    source_row = int(stepping.MIRROR_SOURCE_INDEX)
    out: Dict[str, Any] = {"source_row": source_row, "axes": [],
                           "min_max_abs": None}
    for axis in range(3):
        if not grid.is_mirrored(axis):
            continue
        for name in ("Dx", "Dy", "Dz"):
            volume = to_host(getattr(fields, name))
            plane = volume[_plane_index(axis, source_row, volume.shape)]
            value = float(np.max(np.abs(plane.astype(np.float64))))
            out["axes"].append({"axis": axis, "volume": name, "max_abs": value})
            out["min_max_abs"] = (value if out["min_max_abs"] is None
                                  else min(out["min_max_abs"], value))
    out["meets_floor"] = (not out["axes"]) or (out["min_max_abs"] or 0.0) > 0.0
    return out


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

_SLICE_TESTS: Optional[Any] = None


def slice_module():
    """This family's own test module: the evaluator AND the grammar it is anchored
    on. Imported, never copied -- that module is where the grammar is PINNED, so a
    second copy here could drift from it and the laptop leg would then be measuring
    a tree the merge bar does not check."""
    global _SLICE_TESTS
    if _SLICE_TESTS is None:
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            test_folded_offdiag as slice_tests)
        _SLICE_TESTS = slice_tests
    return _SLICE_TESTS


#: Template text the evaluator neither executes nor pins because it carries no
#: arithmetic. Kept in step with the slice test's own list.
_SCAFFOLDING: Tuple[str, ...] = (
    'extern "C" __global__', "float* __restrict__", "const float*",
    "int nx, int ny, int nz,", "int bc_x", "int wm_x", "float gw_x",
    "int idx = blockIdx.x", "if (idx >= nx * ny * nz) return;",
    "int nyz = ny * nz;", "int k = idx % nz;", "int j = (idx / nz) % ny;",
    "int i = idx / (ny * nz);", "int di = coord_dn", "int dj = coord_dn",
    "int dk = coord_dn", "int at_x = (i == 0)", ") {", "}", "{",
)


def evaluator_blind_lines(text: str) -> List[str]:
    """Lines of an emitted body the NumPy evaluator neither EXECUTES nor pins.

    THE HAZARD THIS EXISTS FOR, measured on the certified family's first laptop
    run: the evaluator is a ``finditer`` over an anchored grammar, so a mutation
    that rewrites a statement into a form the grammar does not match is not
    applied at all -- it is SILENTLY SKIPPED, and the leg then reports a verdict
    that says nothing about the defect. On the device backend the compiler
    executes every line and the question does not arise, which is why the two
    backends' catch tables differ and why both are reported.
    """
    tests = slice_module()
    grammar = (tests._TERM_CALL, tests._TOTAL_FIRST, tests._TOTAL_ADD,
               tests._MASK, tests._LANE, tests._GS, tests._US,
               tests._SRC_COUPLED, tests._SRC_PLAIN, tests._TAIL)
    body = text[text.index('extern "C"'):]
    covered = set()
    for pattern in grammar:
        for match in pattern.finditer(body):
            covered.update(range(match.start(), match.end()))
    blind, offset = [], 0
    for line in body.split("\n"):
        stripped = line.strip()
        span = set(range(offset, offset + len(line)))
        offset += len(line) + 1
        if not stripped or stripped.startswith("//"):
            continue
        if span & covered:
            continue
        if any(token in line for token in _SCAFFOLDING):
            continue
        blind.append(stripped)
    return blind


def classify_for_evaluator(pristine: str, mutated: str) -> Dict[str, Any]:
    """Can the NumPy evaluator see this defect at all, and if not, what guards it?

    THIS FAMILY'S EVALUATOR PARSES ITS GHOST RULES rather than transcribing them,
    which is a deliberate difference from the certified slice: ``coord_dn``,
    ``coord_up``, ``ghosted_mirror``, the ghost lanes and ``MIRROR_ROW`` are all
    read out of the emitted text and evaluated, because the mirror arm is the only
    new rule here and a pinned string is weaker evidence than an executed one. So
    the ``pinned_half`` bucket is much smaller here than there: only the helpers
    INHERITED from the certified family (``flat``, ``ghosted``,
    ``constitutive_apply``) are transcribed.
    """
    inherited = {name: folded._sibling_block(name)
                 for name in folded.SIBLING_HELPERS}
    broken = [name for name, body in inherited.items() if body not in mutated]
    if broken:
        return {"evaluator_sees_it": False, "why": "inherited_pinned_half",
                "broken_pins": broken}
    for pattern, label in ((slice_module()._COORD_DN, "coord_dn"),
                           (slice_module()._COORD_UP, "coord_up"),
                           (slice_module()._GHOST_MIRROR_BODY, "ghosted_mirror"),
                           (slice_module()._MIRROR_ROW, "MIRROR_ROW"),
                           (slice_module()._TERM_BODY, "folded_offdiag_term")):
        if pattern.search(mutated) is None:
            return {"evaluator_sees_it": False, "why": "outside_the_grammar",
                    "unparsable": label}
    blind = evaluator_blind_lines(mutated)
    if blind and blind != evaluator_blind_lines(pristine):
        return {"evaluator_sees_it": False, "why": "outside_the_grammar",
                "unmatched_lines": blind[:4]}
    return {"evaluator_sees_it": True, "why": "arithmetic"}


class EvaluatorBackend:
    """The kernel side, executed in NumPy from the emitted text. Compiles nothing.

    Present so the harness can be shown to FAIL where it must without a device. It
    answers ``certifies = False`` and every summary carries that through, so no run
    on this backend can be mistaken for a certification.
    """

    name = "numpy"
    #: NEVER. Every summary and the release verdict read this, so a laptop run
    #: cannot be mistaken for a certification even when every leg is green.
    certifies = False

    def __init__(self):
        self.source_transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self.sources_used: Dict[str, str] = {}

    def set_guard(self, guard: Sequence[str]) -> None:
        """No compiler, so no guard. Recorded rather than silently ignored."""

    def set_source_mutation(self, transform) -> None:
        self.source_transform = transform

    def pristine_source(self, mask) -> str:
        return folded.folded_offdiag_source(tuple(mask))

    def source_for(self, mask) -> Tuple[str, int]:
        text = self.pristine_source(mask)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def launch(self, fields, layer, rows, mask, tables, codes, walls,
               weights) -> None:
        text, _sites = self.source_for(mask)
        self.sources_used[str(tuple(mask))] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        slice_module().evaluate_folded_source(
            text, flat.device_arrays(fields, rows), tables, codes, walls,
            weights, tuple(fields.grid.shape))

    def compile_log(self) -> List[Dict[str, Any]]:
        return []

    def clear(self) -> int:
        return 0


class KernelBackend:
    """The shipped CuPy launcher, driven through its own gate door.

    THE GUARD AND THE MUTATION BOTH REACH NVRTC THROUGH THE SOURCE-KEYED MEMO:
    ``_get_kernel`` keys on ``(name, options, policy, source)``, so overriding
    ``_COMPILE_OPTIONS`` or installing a source override is a MISS and the compiler
    sees the bytes this leg intends. The memo is dropped at both ends anyway,
    because a leg that measured a guard it never applied is a failure family this
    track has three recorded instances of.

    THE SOURCE OVERRIDE GOES THROUGH ``set_kernel_source`` -- the module's own
    documented gate door -- rather than by monkeypatching the emitter function,
    which is what the certified family's gate has to do because its launcher calls
    the emitter by name.
    """

    name = "cupy"
    certifies = True

    def __init__(self):
        self.source_transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self.sources_used: Dict[str, str] = {}
        self._overridden: List[Tuple[int, ...]] = []

    def set_guard(self, guard: Sequence[str]) -> None:
        folded._COMPILE_OPTIONS = tuple(guard)
        folded.clear_kernel_cache()

    def set_source_mutation(self, transform) -> None:
        self.source_transform = transform
        for mask in self._overridden:
            folded.set_kernel_source(mask, None)
        self._overridden = []
        folded.clear_kernel_cache()

    def pristine_source(self, mask) -> str:
        """The UNMUTATED source. Site counting and digesting must never go through
        an installed override: that would apply the transform twice and the leg
        would hash a body it never compiled."""
        return folded.folded_offdiag_source(tuple(mask))

    def source_for(self, mask) -> Tuple[str, int]:
        text = self.pristine_source(mask)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def launch(self, fields, layer, rows, mask, tables, codes, walls,
               weights) -> None:
        key = tuple(mask)
        text, _sites = self.source_for(key)
        if self.source_transform is not None:
            folded.set_kernel_source(key, text)
            if key not in self._overridden:
                self._overridden.append(key)
        self.sources_used[str(key)] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        folded.update_E_folded_offdiag_fused_pml_real(
            fields, tables=tables, codes=codes, walls=walls, weights=weights)
        cp.cuda.runtime.deviceSynchronize()

    def compile_log(self) -> List[Dict[str, Any]]:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

        return list(compile_cache.compile_log())

    def clear(self) -> int:
        for mask in self._overridden:
            folded.set_kernel_source(mask, None)
        self._overridden = []
        return folded.clear_kernel_cache()


def build_backend(name: str):
    if name == "numpy":
        return EvaluatorBackend()
    if cp is None:
        raise SystemExit("--backend cupy needs CuPy; this host has none")
    return KernelBackend()


# ---------------------------------------------------------------------------
# THE DEFECTS
# ---------------------------------------------------------------------------
#
# Every one is a SILENT wrong answer -- a smooth, converged, plausible field with
# the wrong tensor in it -- and none would be caught by a magnitude comparison at
# any tolerance a physicist would accept.
#
# THE FIRST FIVE ARE THIS FAMILY'S OWN, and they are the whole reason the file
# exists: they are the mirror arm stated wrongly in each of the ways the reading
# invites. The rest are the certified family's defects re-anchored on this
# grammar, carried because this kernel emits that arithmetic too and a gate that
# only tested its own delta would certify a body it never checked.

_MIRROR_BRANCH = "if (bc == BC_MIRROR) return MIRROR_ROW;"
_TERM_RETURN = ("return 0.25f * ((near_pair * u[home])"
                " + (far_pair * ghosted(u, up)));")
_GHOST_RETURN = "return mg ? (w * value) : value;"


def _sub(pattern: str, replacement, text: str) -> Tuple[str, int]:
    new, count = re.subn(pattern, replacement, text)
    return new, count


def _replace(needle: str, replacement: str, text: str) -> Tuple[str, int]:
    return text.replace(needle, replacement), text.count(needle)


def m_mirror_ghost_as_the_metallic_zero(text):
    """The certified kernel's answer for a folded axis handed BC_METALLIC, planted
    as the defect it is. The 2026-08-20 sweep measured this arm DIVERGENT on
    352/352 guarded cases per policy."""
    return _replace(_MIRROR_BRANCH, "if (bc == BC_MIRROR) return -1;", text)


def m_mirror_ghost_as_the_periodic_wrap(text):
    """A fold read as a wrap -- the natural stand-in, measured WRONG on 32 of 480
    guarded cases per policy by the sweep that specified this family."""
    return _replace(_MIRROR_BRANCH, "if (bc == BC_MIRROR) return n - 1;", text)


def m_mirror_row_off_by_one(text):
    """The reflected row off by one. MEEP's halved origin at ``io = -2`` maps the
    ghost at -1 onto stored cell 2 (stepping.py:160); row 1 is a half-cell
    registration error on the fold plane -- smooth, converged and wrong."""
    return _replace("#define MIRROR_ROW 2", "#define MIRROR_ROW 1", text)


def m_flip_the_parity_sign(text):
    """``+phase`` instead of ``-phase``: the sign error a reader of
    ``mp.Mirror(direction, phase)`` makes, since the plane's declared phase and
    the component's parity about it are not the same number."""
    return _replace(_GHOST_RETURN, "return mg ? (-w * value) : value;", text)


def m_drop_the_parity_weight(text):
    """THE SHARPEST DISCRIMINATOR. The weight is ``-phase``: on an EVEN plane it
    is -1 and dropping it must be caught, on an ODD plane it is +1 and dropping it
    is the identity. Scored per case on the plane parity, so a battery at one
    parity cannot pass this by learning nothing about the sign."""
    return _replace(_GHOST_RETURN, "return value;", text)


def m_weight_every_lane(text):
    """The weight applied to every lane instead of the ghost lane alone. ``1.0f *
    x`` is the identity under ``-ftz=false``, so this is only observable where the
    weight is -1 -- which is the EVEN plane, and is why the module writes the rule
    as a select rather than as a blanket multiply."""
    return _replace(_GHOST_RETURN, "return w * value;", text)


def m_folded_own_axis_wraps(text):
    """``coord_up`` wrapping on a fold instead of serving the exact zero.
    ``_offdiagonal_terms`` passes no ``reflect_row`` (stepping.py:1248-1249) so
    both mirror terminations take :1828-1830. A DISCRIMINATOR: it can only bite
    where a folded axis is some live component's OWN axis."""
    return _replace("return (bc == BC_PERIODIC) ? 0 : -1;",
                    "return (bc == BC_METALLIC) ? -1 : 0;", text)


def m_ghost_lane_from_the_wall_flag(text):
    """The ``wm_*``/``bc_*`` split collapsed: the ghost lane read off the grid's
    DECLARATION instead of the resolved ghost rule. A folded axis reports
    ``wm = 0``, so the parity would never be applied at all."""
    return _sub(r"int mg_([xyz]) = \(bc_[xyz] == BC_MIRROR\) && at_[xyz];",
                r"int mg_\1 = wm_\1 && at_\1;", text)


def m_weight_the_own_axis_up_leg(text):
    """The parity applied to the OWN-axis up load as well. The array path weights
    exactly one plane, on the partner axis (stepping.py:1870-1872)."""
    return _replace(
        "float far_pair = ghosted(g, up) + ghosted_mirror(g, corner, mg, w);",
        "float far_pair = ghosted_mirror(g, up, mg, w)"
        " + ghosted_mirror(g, corner, mg, w);", text)


def m_couple_the_wrong_components(text):
    """Each term reads the NEXT axis's D volume instead of its partner's -- the
    defect an author makes by reading ``cycle_direction`` once and applying it to
    the volume as well as to the axis."""
    rotate = {"Dx": "Dy", "Dy": "Dz", "Dz": "Dx"}
    return _sub(r"folded_offdiag_term\(\n        (D[xyz]),",
                lambda m: f"folded_offdiag_term(\n        {rotate[m.group(1)]},",
                text)


def m_drop_a_row_from_the_volume_sum(text):
    """One partner's term silently missing from a component's total. Half a
    tensor: the remaining term is right, the field stays smooth, the anisotropy is
    wrong."""
    return _sub(r"\n    total_(\w+) = total_\1 \+ term_\w+;", "", text)


def m_invert_the_wall_mask(text):
    """The metallic wall-coupling select inverted: kept at the wall, zeroed
    inside."""
    return _sub(r"\? 0\.0f : total_(\w+);", r"? total_\1 : 0.0f;", text)


def m_drop_the_wall_mask(text):
    """The mask never applied. Its discriminator: a NULL on a grid that declares
    no metallic axis a live row masks against."""
    return _sub(r"\n *total_\w+ = \(wm_[xyz] && at_[xyz]\) \? 0\.0f : total_\w+;",
                "", text)


def m_hoist_the_coefficient(text):
    """The four-point average times ``u[home]``. The same ALGEBRA only for a
    uniform coefficient, and a different float32 number even then. The family's
    one genuinely new arithmetic element in the certified round, carried here
    because this kernel emits it."""
    return _replace(_TERM_RETURN,
                    "return 0.25f * ((near_pair + far_pair) * u[home]);", text)


def m_same_direction_shifts(text):
    """Both shifts UP: the half-cell registration error the opposite directions
    exist to prevent (stepping.py:1243-1249)."""
    rename = {"di": "ui", "dj": "uj", "dk": "uk"}
    return _sub(r"flat\((\w+), (\w+), (\w+), nyz, nz\)",
                lambda m: "flat({}, {}, {}, nyz, nz)".format(
                    *[rename.get(g, g) for g in m.groups()]), text)


def m_wrong_own_axis_table(text):
    """Every component reads the x coefficient table. MEEP's ``dsigw`` is the
    absorption a component accumulates along the direction it POINTS IN."""
    return _sub(r"kps_[xyz]\[[ijk]\], kms_[xyz]\[[ijk]\]", "kps_x[i], kms_x[i]",
                text)


def n_distribute_the_quarter(text):
    """NULL: 0.25 is an exact power of two, so scaling by it commutes with
    round-to-nearest AWAY FROM UNDERFLOW. Armed on the uniform class only, and the
    record says so rather than implying it is a null everywhere."""
    return _replace(_TERM_RETURN,
                    "return (0.25f * (near_pair * u[home]))"
                    " + (0.25f * (far_pair * ghosted(u, up)));", text)


def n_commute_the_near_product(text):
    """NULL: IEEE multiply commutes bitwise. Its discriminator is
    ``hoist_the_coefficient``, which edits the same expression's ASSOCIATION and
    must be caught -- so the pair says the comparator is sensitive to grouping and
    insensitive to operand order, which is a measurement rather than a claim."""
    return _replace("(near_pair * u[home])", "(u[home] * near_pair)", text)


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "mirror_ghost_as_the_metallic_zero": m_mirror_ghost_as_the_metallic_zero,
    "mirror_ghost_as_the_periodic_wrap": m_mirror_ghost_as_the_periodic_wrap,
    "mirror_row_off_by_one": m_mirror_row_off_by_one,
    "flip_the_parity_sign": m_flip_the_parity_sign,
    "drop_the_parity_weight": m_drop_the_parity_weight,
    "weight_every_lane": m_weight_every_lane,
    "weight_the_own_axis_up_leg": m_weight_the_own_axis_up_leg,
    "folded_own_axis_wraps": m_folded_own_axis_wraps,
    "ghost_lane_from_the_wall_flag": m_ghost_lane_from_the_wall_flag,
    "couple_the_wrong_components": m_couple_the_wrong_components,
    "drop_a_row_from_the_volume_sum": m_drop_a_row_from_the_volume_sum,
    "invert_the_wall_mask": m_invert_the_wall_mask,
    "drop_the_wall_mask": m_drop_the_wall_mask,
    "hoist_the_coefficient": m_hoist_the_coefficient,
    "same_direction_shifts": m_same_direction_shifts,
    "wrong_own_axis_table": m_wrong_own_axis_table,
    "distribute_the_quarter": n_distribute_the_quarter,
    "commute_the_near_product": n_commute_the_near_product,
}

#: Mutations whose verdict must be UNCAUGHT everywhere. Each is paired with a leg
#: editing the same kind of expression that must be caught, because a battery of
#: only-must-be-caught legs scores identically whether the comparator works or has
#: degenerated into failing everything.
NULL_MUTATIONS: Tuple[str, ...] = ("distribute_the_quarter",
                                   "commute_the_near_product")

#: Defects that can only bite where a folded axis is a live slot's PARTNER, which
#: is where the mirror ghost enters this sub-step at all
#: (``stepping._shift_down`` on ``axis = partner_axis``, :1214-1216). Scored as
#: DISCRIMINATORS: caught on that arm, NULL off it. Between them they re-measure
#: ``coverage.offdiag_fold_roles`` on a device, against a kernel that implements
#: the mirror arm rather than one that lacks it.
PARTNER_ARM_MUTATIONS: Tuple[str, ...] = (
    "mirror_ghost_as_the_metallic_zero", "mirror_ghost_as_the_periodic_wrap",
    "mirror_row_off_by_one", "flip_the_parity_sign", "weight_every_lane",
    "ghost_lane_from_the_wall_flag")

#: The same, for the OWN-axis arm: ``coord_up``'s mirror behaviour can only bite
#: where a folded axis is some live component's own axis.
OWN_ARM_MUTATIONS: Tuple[str, ...] = ("folded_own_axis_wraps",)

#: DEFECTS THAT ARE THE IDENTITY WHEN THE PARITY IS +1, and are therefore
#: two-sided in the PLANE PHASE as well as in the fold's role. The weight is
#: ``-phase`` (fields.py:117 collapsed over the D components), so on an ODD plane
#: it is exactly ``+1.0`` and any defect that only changes WHICH LANES the weight
#: multiplies -- or whether it is applied at all -- is bitwise inert there.
#:
#: MEASURED, NOT ASSUMED: scoring these as plain partner-arm catches made the
#: first laptop run report three DISCRIMINATOR VIOLATED at 3/8, and the missing
#: catch in every one was the odd-plane case. That is the leg working on a defect
#: that is genuinely not one there, and it is why both parities are swept: a
#: battery at even planes alone would score these CAUGHT and learn nothing about
#: the SIGN, while one at odd planes alone would score them NULL and learn nothing
#: at all.
PARITY_VISIBLE_MUTATIONS: Tuple[str, ...] = (
    "drop_the_parity_weight", "weight_every_lane", "weight_the_own_axis_up_leg",
    "ghost_lane_from_the_wall_flag")

#: Wall-mask defects that are DISCRIMINATORS: caught where the grid declares a
#: metallic wall that a live row component masks against, NULL where it does not
#: -- and a folded axis always reports ``wm = 0``, which is why the sweep carries
#: two fold-beside-a-wall specs so the catch arm is not empty.
#:
#: ``invert_the_wall_mask`` IS DELIBERATELY NOT HERE, and the reason is a
#: measurement rather than a reading: the first device run scored it
#: DISCRIMINATOR VIOLATED at 68/68 because inverting ``(wm && at) ? 0.0f : total``
#: to ``(wm && at) ? total : 0.0f`` zeroes the coupling EVERYWHERE the select is
#: false -- which, on a grid declaring no wall, is the whole volume. It is a
#: defect on every grid, so it belongs with the must-be-caught legs and scoring it
#: two-sided was a statement about the harness, not about the kernel.
WALL_DISCRIMINATORS: Tuple[str, ...] = ("drop_the_wall_mask",
                                        "drop_the_wall_flags")

#: Host defects: the kernel never chooses these and cannot choose them wrong, but
#: its caller can. Armed through the launcher's keyword-only gate door.
#:
#: ``hand_the_fold_the_metallic_code`` is the SHIPPED CERTIFIED KERNEL'S ONLY
#: AVAILABLE ANSWER, and it is this gate's two-sidedness: it must be CAUGHT
#: wherever the mirror ghost is reachable and a NULL where it is not.
HOST_MUTATIONS: Tuple[str, ...] = ("hand_the_fold_the_metallic_code",
                                   "flip_the_launched_ghost_weights",
                                   "reverse_folded_axis_coefficients",
                                   "swap_sub_lattice", "drop_the_wall_flags")


def apply_host_mutation(name: Optional[str], layer, grid, tables, codes, walls,
                        weights):
    """One host defect, on the arguments a launch binds. Returns the four."""
    if name is None:
        return tables, codes, walls, weights
    if name == "hand_the_fold_the_metallic_code":
        codes = tuple(coverage.BC_CODES["metallic"]
                      if code == folded.BC_MIRROR_CODE else code
                      for code in codes)
        return tables, codes, walls, weights
    if name == "flip_the_launched_ghost_weights":
        return tables, codes, walls, tuple(-w for w in weights)
    if name == "reverse_folded_axis_coefficients":
        # THE RETIRED REFUSAL, ARMED. "The fold changes the stored extent, and the
        # extent is what turns a cell index into a coefficient index" was the
        # certified predicate's stated reason for refusing a fold. If that
        # mattered to this sub-step, the coefficient a cell reads on the folded
        # axis would be wrong. This makes it wrong IN BOUNDS, by reversing the
        # vector rather than substituting a longer or shorter one, so the leg
        # measures a WRONG ANSWER and not a memory fault.
        mutated = dict(tables)
        xp = grid.xp
        for axis, name_ in enumerate("xyz"):
            if not grid.is_mirrored(axis):
                continue
            for stem in ("kps", "kms"):
                key = f"{stem}_{name_}"
                mutated[key] = xp.ascontiguousarray(mutated[key][::-1])
        return mutated, codes, walls, weights
    if name == "swap_sub_lattice":
        return flat.integer_tables(layer), codes, walls, weights
    if name == "drop_the_wall_flags":
        return tables, codes, (0, 0, 0), weights
    raise ValueError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_key(spec, mask_spec, courant, value_class, guard, steps) -> str:
    return "|".join([spec["label"], mask_spec["label"], f"C{courant}",
                     value_class, guard, f"n{steps}"])


def one_case(backend, xp, spec: Dict[str, Any], mask_spec: Dict[str, Any],
             courant: float, value_class: str, guard_label: str,
             guard: Sequence[str], steps: int,
             source_mutation: Optional[str] = None,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping.update_E`` ITSELF on real ``Grid``/``Fields``/``PML``
    objects with the rows installed through the PUBLIC installer, so the
    install-time validation, the zero-row drop and the stored-E switch are all in
    force. There is no second transcription on the oracle leg to drift.
    """
    started = time.time()
    key = case_key(spec, mask_spec, courant, value_class, guard_label, steps)
    # A STABLE PER-CASE SEED, from a DIGEST rather than from ``hash()``: Python
    # salts ``hash()`` of a str with PYTHONHASHSEED, so a run seeded from it draws
    # a different fixture every process and a failing case cannot be replayed.
    rng = np.random.default_rng(
        (SEED + int(hashlib.sha256(key.encode("ascii")).hexdigest()[:8], 16))
        % (2 ** 32))
    mask = tuple(mask_spec["mask"])

    fields, layer, grid = build(xp, spec, courant)
    rows = flat.install_rows(fields, grid, mask, ROW_FORM, rng)
    host = flat.seed_state(fields, grid, value_class, rng)

    roles = fold_roles(grid, mask)
    codes = folded.folded_offdiag_boundary_codes(grid)
    walls = coverage.offdiag_wall_mask_flags(grid)
    weights = folded.mirror_ghost_weights(grid)

    case: Dict[str, Any] = {
        "key": key,
        "label": spec["label"], "fold_axes": spec["axes"],
        "mirror_phase": spec["phase"], "boundaries": list(spec["boundaries"]),
        "row_mask_label": mask_spec["label"], "row_mask": list(mask),
        "installed_row_mask": list(coverage.offdiag_row_mask(fields)),
        "courant": courant, "value_class": value_class, "guard": guard_label,
        "steps": steps, "backend": backend.name,
        "source_mutation": source_mutation, "host_mutation": host_mutation,
        "shape": [int(n) for n in grid.shape],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
        "codes": list(codes), "walls": list(walls),
        "ghost_weights": [float(w) for w in weights],
        "roles": roles,
        "operand_census": operand_census(host),
    }
    case["fold_reaches_the_partner_arm"] = bool(
        roles["fold_is_a_live_partner_axis"])
    case["fold_reaches_the_own_arm"] = bool(roles["fold_is_a_live_own_axis"])
    case["wall_mask_bites"] = wall_mask_bites(mask, walls)

    # THE TWO PREDICATES' ANSWERS, recorded rather than acted on. The census asks
    # the composition verdict, and a record that did not carry both could not show
    # that the two families partition this configuration rather than sharing it.
    mine = folded.covers_folded_offdiag_composition(fields, layer, grid)
    theirs = coverage.covers_real_pml_offdiag_constitutive(fields, layer, grid)
    case["predicates"] = {
        "folded_composition": {"covered": bool(mine[0]), "reason": mine[1]},
        "certified_offdiag": {"covered": bool(theirs[0]), "reason": theirs[1]},
        "disjoint": not (bool(mine[0]) and bool(theirs[0])),
    }

    absorbs = folded_axis_absorbs(layer, grid)
    case["folded_axis_absorbs"] = absorbs
    ghost = mirror_ghost_is_live(fields, grid)
    case["mirror_ghost_is_live"] = ghost
    if not absorbs["meets_floor"]:
        case["skipped"] = ("the folded axis's half-integer coefficient profile is "
                           "the identity everywhere; this case cannot distinguish "
                           "a coefficient index error on the axis the fold moved")
        case["seconds"] = round(time.time() - started, 3)
        return case
    if not ghost["meets_floor"]:
        case["skipped"] = ("stored row 2 of a partner volume is identically zero "
                           "on a folded axis, so the mirror ghost equals the "
                           "metallic zero ghost and an exact result would be a "
                           "fact about the fixture")
        case["seconds"] = round(time.time() - started, 3)
        return case

    frozen = flat.snapshot(fields)

    # --- leg 1: the oracle -------------------------------------------------
    for _ in range(steps):
        stepping.update_E(fields, layer)
        flat.advance_sources(fields)
    oracle = {name: to_host(getattr(fields, name)).copy() for name in OUTPUTS}

    moved = sum(int(np.count_nonzero(
        oracle[name].ravel().view(np.uint32)
        != np.ascontiguousarray(frozen[name], dtype=np.float32).ravel().view(
            np.uint32)))
        for name in OUTPUTS)
    case["oracle_moved_words"] = moved
    case["oracle_total_words"] = sum(int(oracle[name].size) for name in OUTPUTS)
    case["oracle_moved"] = moved > 0
    diagonal = to_host(fields.Dx) * to_host(fields.inverse_epsilon_for("Ex"))
    coupling = to_host(fields.f_w_Ex) - diagonal
    case["coupling_max_abs"] = float(np.max(np.abs(coupling.astype(np.float64))))
    case["coupling_is_live"] = case["coupling_max_abs"] > 0.0

    # --- leg 2: the kernel, from the SAME frozen state ---------------------
    flat.restore(fields, frozen)
    backend.set_guard(guard)
    tables = flat.half_integer_tables(layer)
    tables, launch_codes, launch_walls, launch_weights = apply_host_mutation(
        host_mutation, layer, grid, tables, codes, walls, weights)
    launch_error = None
    try:
        for _ in range(steps):
            backend.launch(fields, layer, rows, mask, tables, launch_codes,
                           launch_walls, launch_weights)
            flat.advance_sources(fields)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, recorded
        launch_error = f"{type(exc).__name__}: {exc}"[:600]
    case["launch_error"] = launch_error

    if launch_error is None:
        parts = {name: bit_compare(oracle[name], getattr(fields, name))
                 for name in OUTPUTS}
        verdict = combine(parts)
        case["differing_words"] = int(verdict["differing_floats"])
        case["per_component"] = {
            name: int(part["differing_floats"]) for name, part in parts.items()}
    else:
        verdict = {"bit_identical": False, "differing_floats": 0,
                   "total_floats": 0, "per_component": {}}
        case["differing_words"] = 0
        case["per_component"] = {}
    case["bit_identical"] = bool(verdict["bit_identical"]) and launch_error is None
    case["total_floats"] = verdict["total_floats"]
    case["max_ulp"] = verdict.get("max_ulp")
    case["seconds"] = round(time.time() - started, 3)
    return case


def wall_mask_bites(mask: Sequence[int], walls: Sequence[int]) -> bool:
    """Does the metallic wall mask change ANY word on this (mask, grid) pair?

    A LIVE WALL FLAG IS NOT ENOUGH. ``_mask_metallic_wall_coupling`` zeroes face 0
    of the axes on which THE COMPONENT'S Yee shift is 0
    (:data:`coverage.OFFDIAG_WALL_MASK_AXES`), so ``wm_y`` reaches Ex and Ez and
    never reaches Ey. A grid that declares a wall on an axis no live row component
    masks against is one where dropping the mask is a genuine NULL, and scoring it
    a miss would be a statement about the fixture rather than about the kernel.
    """
    for slot, (row, _partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        component = COMPONENTS.index(row)
        if any(walls[axis] for axis in coverage.OFFDIAG_WALL_MASK_AXES[component]):
            return True
    return False


def case_is_valid(case: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """A case that could not distinguish anything is REFUSED, not passed."""
    if case.get("skipped"):
        return False, case["skipped"]
    if not case.get("oracle_moved"):
        return False, ("the array path changed no output word: zero-init is a "
                       "fixed point of this recurrence and a case that moved "
                       "nothing certifies nothing")
    if not case.get("coupling_is_live"):
        return False, ("the off-diagonal coupling is identically zero: this case "
                       "cannot distinguish any defect in this family")
    return True, None


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str, steps: int) -> List[Dict[str, Any]]:
    """The swept cases, before the guard axis multiplies them.

    THE REDUCED SUBSET IS NAMED, NOT SLICED. A head slice of this list is all X
    folds and no declared wall, which leaves the wall discriminators with one arm
    and every fold-axis question answered on one axis.
    """
    specs = (FOLD_SPECS if product == "full"
             else [s for s in FOLD_SPECS if s["label"] in REDUCED_SPEC_LABELS])
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else VALUE_CLASSES
    plan: List[Dict[str, Any]] = []
    for spec in specs:
        for mask_spec in ROW_MASK_SPECS:
            for courant in courants:
                for value_class in classes:
                    plan.append({"spec": spec, "mask_spec": mask_spec,
                                 "courant": courant,
                                 "value_class": value_class, "steps": steps})
    return plan


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [c for c in cases if c.get("case_is_valid")]
    per_guard: Dict[str, Dict[str, int]] = {}
    for case in scored:
        bucket = per_guard.setdefault(case["guard"], {
            "ran": 0, "identical": 0, "inexact_ran": 0, "inexact_identical": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])
        if case["courant"] == INEXACT_COURANT:
            bucket["inexact_ran"] += 1
            bucket["inexact_identical"] += int(case["bit_identical"])

    primary = [c for c in scored if c["guard"] == "fmad_false"]
    fold = [c for c in primary if c["fold_axes"]]
    unfolded = [c for c in primary if not c["fold_axes"]]
    partner_arm = [c for c in fold if c["fold_reaches_the_partner_arm"]]
    own_arm = [c for c in fold if c["fold_reaches_the_own_arm"]]

    by_fold_kind: Dict[str, Dict[str, int]] = {}
    for case in fold:
        kind = "folded_metallic" if any(
            case["metallic"][a] for a in case["roles"]["folded_axes"]
        ) else "folded_periodic"
        bucket = by_fold_kind.setdefault(kind, {"ran": 0, "identical": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])

    by_planes: Dict[str, Dict[str, int]] = {}
    for case in fold:
        bucket = by_planes.setdefault(str(len(case["roles"]["folded_axes"])),
                                      {"ran": 0, "identical": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])

    census = {"values": 0, "subnormals": 0, "negative_zeros": 0, "zeros": 0}
    for case in scored:
        if case["value_class"] != "subnormal_band":
            continue
        for key in census:
            census[key] += case["operand_census"][key]

    disjointness = [c["key"] for c in scored if not c["predicates"]["disjoint"]]
    return {
        "per_guard": per_guard,
        "by_fold_kind": by_fold_kind,
        "by_simultaneous_planes": by_planes,
        "scored": len(scored),
        "refused": len(cases) - len(scored),
        "folded_ran": len(fold),
        "folded_identical": sum(1 for c in fold if c["bit_identical"]),
        "unfolded_ran": len(unfolded),
        "unfolded_identical": sum(1 for c in unfolded if c["bit_identical"]),
        "partner_arm_ran": len(partner_arm),
        "partner_arm_identical": sum(1 for c in partner_arm if c["bit_identical"]),
        "own_arm_ran": len(own_arm),
        "own_arm_identical": sum(1 for c in own_arm if c["bit_identical"]),
        "differing_words_total": sum(c["differing_words"] for c in primary),
        "cases_that_diverged": [c["key"] for c in primary
                                if not c["bit_identical"]],
        "subnormal_band_operands": census,
        "subnormal_band_is_non_vacuous": census["subnormals"] > 0,
        "every_case_moved_the_oracle": all(c["oracle_moved"] for c in scored),
        "every_case_had_live_coupling": all(c["coupling_is_live"] for c in scored),
        "min_coupling_max_abs": (min(c["coupling_max_abs"] for c in scored)
                                 if scored else None),
        "min_folded_axis_deviation": min(
            [c["folded_axis_absorbs"]["max_deviation"] for c in scored
             if c["folded_axis_absorbs"]["folded_axes"]] or [None]),
        "predicates_disjoint_everywhere": not disjointness,
        "predicate_overlaps": disjointness,
    }


def run_sweep(backend, xp, results: Dict[str, Any], out_path: str, product: str,
              steps: int, leg_name: str,
              guards: Sequence[Tuple[str, Tuple[str, ...], bool]] = GUARD_SETS
              ) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product, steps)
    swept = [g for g in guards if backend.certifies or g[0] == "fmad_false"]
    total = len(plan) * len(swept)
    index = 0
    started = time.time()
    for guard_label, guard, _primary in swept:
        for entry in plan:
            index += 1
            case = one_case(backend, xp, entry["spec"], entry["mask_spec"],
                            entry["courant"], entry["value_class"], guard_label,
                            guard, entry["steps"])
            valid, why = case_is_valid(case)
            case["case_is_valid"] = valid
            case["refused_because"] = why
            cases.append(case)
            log(f"[{leg_name}] {index}/{total} {case['key']} "
                f"shape={case['shape']} "
                f"roles=P{int(case['fold_reaches_the_partner_arm'])}"
                f"O{int(case['fold_reaches_the_own_arm'])} "
                f"got={'IDENTICAL' if case.get('bit_identical') else 'DIVERGED'} "
                f"diff={case.get('differing_words')}"
                + ("" if valid else f" REFUSED: {why[:60]}")
                + f" ({case['seconds']} s)")
            results[leg_name] = {"cases": cases, "summary": summarize(cases),
                                 "seconds": round(time.time() - started, 1)}
            save(results, out_path)
    return results[leg_name]


# ---------------------------------------------------------------------------
# The mutation battery
# ---------------------------------------------------------------------------
#
# SCOPED TO CASES WHOSE UNMUTATED BASELINE WAS BIT-IDENTICAL, read off the sweep
# that just ran rather than tabulated, so no leg can be armed on an arm the device
# did not actually certify in this same process.

def mutation_plan(sweep_cases: Sequence[Dict[str, Any]], product: str
                  ) -> List[Dict[str, Any]]:
    """The (spec, mask) pairs a defect may be planted on.

    THE REDUCED SELECTION IS GREEDY, NOT A HEAD SLICE. Slicing the front of a case
    product landed the folded-curl gate's whole battery on its two UNFOLDED
    controls, so its one fold-specific mutation was skipped on every leg and
    scored 0/0 UNCAUGHT -- a harness defect that reads exactly like a kernel
    defect. This selection covers, in order: both arms of every discriminator
    (partner role, own role, neither role, wall / no wall), every distinct row
    mask, every distinct folded axis, and both plane parities.
    """
    seen, plan = set(), []
    for case in sweep_cases:
        if not case.get("case_is_valid") or case["guard"] != "fmad_false":
            continue
        if not case["bit_identical"]:
            continue
        if case["courant"] != INEXACT_COURANT or case["value_class"] != "uniform":
            continue
        key = (case["label"], case["row_mask_label"])
        if key in seen:
            continue
        seen.add(key)
        spec = next(s for s in FOLD_SPECS if s["label"] == case["label"])
        mask_spec = next(m for m in ROW_MASK_SPECS
                         if m["label"] == case["row_mask_label"])
        plan.append({"spec": spec, "mask_spec": mask_spec,
                     "partner_arm": bool(case["fold_reaches_the_partner_arm"]),
                     "own_arm": bool(case["fold_reaches_the_own_arm"]),
                     "has_wall": bool(case["wall_mask_bites"]),
                     "phase": case["mirror_phase"],
                     "folded_axes": tuple(case["roles"]["folded_axes"])})
    if product == "full":
        return plan
    chosen: List[Dict[str, Any]] = []
    covered = {"partner": set(), "own": set(), "wall": set(), "mask": set(),
               "axes": set(), "phase": set()}
    for entry in plan:
        wanted = (entry["partner_arm"] not in covered["partner"]
                  or entry["own_arm"] not in covered["own"]
                  or entry["has_wall"] not in covered["wall"]
                  or entry["mask_spec"]["label"] not in covered["mask"]
                  or entry["folded_axes"] not in covered["axes"]
                  or entry["phase"] not in covered["phase"])
        if not wanted:
            continue
        covered["partner"].add(entry["partner_arm"])
        covered["own"].add(entry["own_arm"])
        covered["wall"].add(entry["has_wall"])
        covered["mask"].add(entry["mask_spec"]["label"])
        covered["axes"].add(entry["folded_axes"])
        covered["phase"].add(entry["phase"])
        chosen.append(entry)
    return chosen


def expected_caught(name: str, case: Dict[str, Any]) -> bool:
    """Must THIS case diverge under THIS defect?

    Decided per case rather than per leg, because most of this family's defects
    are only defects where the fold plays a particular role -- and a leg scored
    "caught everywhere" would be false of the physics rather than of the kernel,
    while one scored "uncaught" on a grid where the arm is unreachable would be a
    null nothing could have violated.
    """
    if name in NULL_MUTATIONS:
        return False
    if name in WALL_DISCRIMINATORS:
        return bool(case["wall_mask_bites"])
    if name in PARITY_VISIBLE_MUTATIONS:
        return bool(case["fold_reaches_the_partner_arm"]
                    and case["mirror_phase"] == 1)
    if name in ("hand_the_fold_the_metallic_code",
                "flip_the_launched_ghost_weights"):
        return bool(case["fold_reaches_the_partner_arm"])
    if name in PARTNER_ARM_MUTATIONS:
        return bool(case["fold_reaches_the_partner_arm"])
    if name in OWN_ARM_MUTATIONS:
        return bool(case["fold_reaches_the_own_arm"])
    if name == "reverse_folded_axis_coefficients":
        return bool(case["fold_axes"])
    if name == "drop_a_row_from_the_volume_sum":
        return sum(case["row_mask"][0:2]) == 2 or sum(case["row_mask"][2:4]) == 2 \
            or sum(case["row_mask"][4:6]) == 2
    return True


#: Masks with a component carrying BOTH slots live -- the only ones
#: ``drop_a_row_from_the_volume_sum`` has anything to drop on.
BOTH_SLOTS_MASKS = {(1, 1, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0)}


def leg_armable(name: str, entry: Dict[str, Any]) -> bool:
    if name == "drop_a_row_from_the_volume_sum":
        return tuple(entry["mask_spec"]["mask"]) in BOTH_SLOTS_MASKS
    return True


def run_mutations(backend, xp, results: Dict[str, Any], out_path: str,
                  product: str, steps: int,
                  sweep_cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    plan = mutation_plan(sweep_cases, product)
    legs: List[Dict[str, Any]] = []
    started = time.time()
    names = [(name, "host") for name in HOST_MUTATIONS] + \
            [(name, "source") for name in sorted(SOURCE_MUTATIONS)]
    for name, kind in names:
        leg_started = time.time()
        entries = [e for e in plan if kind == "host" or leg_armable(name, e)]
        transform = SOURCE_MUTATIONS[name] if kind == "source" else None

        if not entries:
            legs.append({"mutation": name, "kind": kind, "ran": 0, "caught": 0,
                         "verdict": "NOT ARMABLE ON ANY LICENSED CASE",
                         "as_required": None, "cases": [], "seconds": 0.0})
            log(f"[mutations] {kind}:{name}: NOT ARMABLE on any licensed case")
            continue

        sites = None
        classification: Optional[Dict[str, Any]] = None
        if transform is not None:
            sites = 0
            for entry in entries:
                pristine = backend.pristine_source(
                    tuple(entry["mask_spec"]["mask"]))
                mutated_text, count = transform(pristine)
                sites += count
                if classification is None or classification["evaluator_sees_it"]:
                    classification = classify_for_evaluator(pristine, mutated_text)
            backend.set_source_mutation(transform)

        if (not backend.certifies and classification is not None
                and not classification["evaluator_sees_it"]):
            backend.set_source_mutation(None)
            legs.append({"mutation": name, "kind": kind, "ran": 0, "caught": 0,
                         "verdict": f"NOT MEASURABLE ON THIS BACKEND "
                                    f"({classification['why']})",
                         "as_required": None, "mutation_sites": sites,
                         "evaluator_classification": classification,
                         "cases": [], "seconds": 0.0})
            log(f"[mutations] {kind}:{name}: NOT MEASURABLE on backend="
                f"{backend.name} ({classification['why']}); sites={sites}")
            continue
        before_compiles = len(backend.compile_log())

        cases: List[Dict[str, Any]] = []
        for entry in entries:
            case = one_case(backend, xp, entry["spec"], entry["mask_spec"],
                            INEXACT_COURANT, "uniform", "fmad_false",
                            ("--fmad=false",), steps,
                            source_mutation=name if kind == "source" else None,
                            host_mutation=name if kind == "host" else None)
            valid, why = case_is_valid(case)
            case["case_is_valid"] = valid
            case["refused_because"] = why
            cases.append(case)

        compiles = backend.compile_log()[before_compiles:]
        mutated_digests = set()
        if transform is not None:
            for entry in entries:
                text, _n = transform(backend.pristine_source(
                    tuple(entry["mask_spec"]["mask"])))
                mutated_digests.add(
                    hashlib.sha256(text.encode("utf-8")).hexdigest())
        from_mutated = sum(1 for e in compiles
                           if e.get("source_sha256") in mutated_digests)
        if transform is not None:
            backend.set_source_mutation(None)
        backend.clear()

        scored = [c for c in cases if c["case_is_valid"]]
        for case in scored:
            case["expected_caught"] = expected_caught(name, case)
            case["was_caught"] = not case["bit_identical"]
            case["leg_as_required"] = case["expected_caught"] == case["was_caught"]
        caught = sum(1 for c in scored if c["was_caught"])
        catch_arm = [c for c in scored if c["expected_caught"]]
        null_arm = [c for c in scored if not c["expected_caught"]]
        discriminator = name in (WALL_DISCRIMINATORS + PARTNER_ARM_MUTATIONS
                                 + OWN_ARM_MUTATIONS + PARITY_VISIBLE_MUTATIONS
                                 + ("hand_the_fold_the_metallic_code",
                                    "flip_the_launched_ghost_weights",
                                    "reverse_folded_axis_coefficients"))
        if not scored:
            verdict, as_required = "NO LEGS", False
        elif discriminator and not (catch_arm and null_arm):
            # A DISCRIMINATOR WITH ONE ARM IS NOT A DISCRIMINATOR. Either half
            # alone is consistent with the rule being inert.
            verdict, as_required = "NO DISCRIMINATOR", False
        else:
            agreeing = sum(1 for c in scored if c["leg_as_required"])
            as_required = agreeing == len(scored)
            if not catch_arm:
                verdict = "NULL CONFIRMED" if as_required else "NULL VIOLATED"
            elif not null_arm:
                verdict = ("CAUGHT" if as_required
                           else "PARTIAL" if caught else "UNCAUGHT")
            else:
                verdict = ("CAUGHT ON THE ARM IT REACHES, NULL OFF IT"
                           if as_required else "DISCRIMINATOR VIOLATED")
        if transform is not None and sites == 0:
            as_required = False
            verdict = "REFUSED: the mutation matched nothing in any emitted source"
        if backend.certifies and transform is not None and from_mutated == 0:
            as_required = False
            verdict = ("REFUSED: the mutated bytes never reached the compiler; a "
                       "verdict for a mutation that never compiled is "
                       "indistinguishable from one that found nothing")

        record = {"mutation": name, "kind": kind,
                  "catch_arm": len(catch_arm), "null_arm": len(null_arm),
                  "ran": len(scored), "caught": caught,
                  "uncaught": len(scored) - caught, "verdict": verdict,
                  "as_required": bool(as_required), "mutation_sites": sites,
                  "is_discriminator": bool(discriminator),
                  "compiles_on_this_leg": len(compiles),
                  "compiles_from_mutated_source": from_mutated,
                  "evaluator_classification": classification,
                  "cases": cases, "seconds": round(time.time() - leg_started, 1)}
        legs.append(record)
        log(f"[mutations] {kind}:{name}: {verdict} ({caught}/{len(scored)}) "
            f"as_required={as_required} sites={sites} "
            f"compiles_from_mutated={from_mutated} ({record['seconds']} s)")
        results["mutations"] = mutation_summary(plan, legs, started)
        save(results, out_path)
    results["mutations"] = mutation_summary(plan, legs, started)
    save(results, out_path)
    return results["mutations"]


def mutation_summary(plan, legs, started) -> Dict[str, Any]:
    return {
        "plan": [{"label": e["spec"]["label"],
                  "row_mask": e["mask_spec"]["label"],
                  "partner_arm": e["partner_arm"], "own_arm": e["own_arm"],
                  "has_wall": e["has_wall"], "phase": e["phase"]}
                 for e in plan],
        "legs": legs,
        "legs_as_required":
            f"{sum(1 for l in legs if l['as_required'])}/"
            f"{sum(1 for l in legs if l['as_required'] is not None)}",
        "legs_not_scored": sum(1 for l in legs if l["as_required"] is None),
        "legs_not_armable": sum(
            1 for l in legs if l["verdict"].startswith("NOT ARMABLE")),
        "legs_not_measurable_on_this_backend": sum(
            1 for l in legs if l["verdict"].startswith("NOT MEASURABLE")),
        "all_legs_as_required": all(
            l["as_required"] for l in legs if l["as_required"] is not None),
        "seconds": round(time.time() - started, 1)}


# ---------------------------------------------------------------------------
# The verdict, and the leg that makes it refuse
# ---------------------------------------------------------------------------

def build_verdict(results: Dict[str, Any]) -> Dict[str, Any]:
    """The release verdict, as a FUNCTION of the legs, so it can be falsified.

    A gate whose verdict is computed inline can never be shown to refuse anything.
    ``--legs falsify`` re-runs this same function over a sweep carrying a planted
    defect and requires ``passed`` to come back False.
    """
    single = results.get("sweep", {}).get("summary", {})
    multi = results.get("multistep", {}).get("summary", {})
    mutations = results.get("mutations", {})
    control = single.get("per_guard", {}).get("default_no_options", {})
    verdict = {
        "folded_single_launch":
            f"{single.get('folded_identical', 0)}/{single.get('folded_ran', 0)}",
        "folded_multi_step":
            f"{multi.get('folded_identical', 0)}/{multi.get('folded_ran', 0)}",
        "unfolded_reduction_single_launch":
            f"{single.get('unfolded_identical', 0)}/{single.get('unfolded_ran', 0)}",
        "partner_arm":
            f"{single.get('partner_arm_identical', 0)}"
            f"/{single.get('partner_arm_ran', 0)}",
        "own_arm":
            f"{single.get('own_arm_identical', 0)}/{single.get('own_arm_ran', 0)}",
        "by_simultaneous_planes": single.get("by_simultaneous_planes"),
        "by_fold_kind": single.get("by_fold_kind"),
        "cases_that_diverged": single.get("cases_that_diverged"),
        "differing_words_total": single.get("differing_words_total"),
        "guard_control_identical_at_inexact_courant":
            f"{control.get('inexact_identical', 0)}/{control.get('inexact_ran', 0)}",
        "guard_control_diverged": (control.get("inexact_ran", 0) > 0
                                   and control.get("inexact_identical", 0)
                                   < control.get("inexact_ran", 0)),
        "subnormal_band_is_non_vacuous": single.get("subnormal_band_is_non_vacuous"),
        "every_case_had_live_coupling": single.get("every_case_had_live_coupling"),
        "every_case_moved_the_oracle": single.get("every_case_moved_the_oracle"),
        "predicates_disjoint_everywhere": single.get(
            "predicates_disjoint_everywhere"),
        "mutation_legs_as_required": mutations.get("legs_as_required"),
        "all_mutation_legs_as_required": mutations.get("all_legs_as_required"),
        "mutation_legs_not_measurable":
            mutations.get("legs_not_measurable_on_this_backend"),
        "mutation_legs_not_armable": mutations.get("legs_not_armable"),
    }
    verdict["passed"] = bool(
        results.get("certifies")
        and single.get("folded_ran", 0) > 0
        and single["folded_identical"] == single["folded_ran"]
        and single.get("unfolded_ran", 0) > 0
        and single["unfolded_identical"] == single["unfolded_ran"]
        and single.get("partner_arm_ran", 0) > 0
        and single["partner_arm_identical"] == single["partner_arm_ran"]
        and single.get("own_arm_ran", 0) > 0
        and single["own_arm_identical"] == single["own_arm_ran"]
        and multi.get("folded_ran", 0) > 0
        and multi["folded_identical"] == multi["folded_ran"]
        and single.get("subnormal_band_is_non_vacuous")
        and single.get("every_case_had_live_coupling")
        and single.get("every_case_moved_the_oracle")
        and single.get("predicates_disjoint_everywhere")
        and mutations.get("all_legs_as_required", False)
        and mutations.get("legs_not_measurable_on_this_backend", 1) == 0
        and mutations.get("legs_not_armable", 1) == 0)
    return verdict


def run_falsification(backend, xp, results: Dict[str, Any], out_path: str,
                      steps: int) -> Dict[str, Any]:
    """RUN THE VERDICT AGAINST A PLANTED DEFECT AND REQUIRE IT TO REFUSE.

    A pass that has never been shown to be refusable is a pass about nothing. The
    defect chosen is the SHIPPED certified kernel's own answer for a folded axis
    (``mirror_ghost_as_the_metallic_zero``), because if the verdict cannot refuse
    THAT then this family has no reason to exist.
    """
    started = time.time()
    spec = next(s for s in FOLD_SPECS if s["label"] == "fold_X_periodic")
    mask_spec = next(m for m in ROW_MASK_SPECS if m["label"] == "corpus_15row")
    backend.set_source_mutation(SOURCE_MUTATIONS["mirror_ghost_as_the_metallic_zero"])
    case = one_case(backend, xp, spec, mask_spec, INEXACT_COURANT, "uniform",
                    "fmad_false", ("--fmad=false",), steps,
                    source_mutation="mirror_ghost_as_the_metallic_zero")
    backend.set_source_mutation(None)
    backend.clear()
    valid, why = case_is_valid(case)
    case["case_is_valid"] = valid
    case["refused_because"] = why

    poisoned = dict(results)
    poisoned["certifies"] = True
    poisoned["sweep"] = {"cases": [case], "summary": summarize([case])}
    poisoned["multistep"] = poisoned["sweep"]
    poisoned["mutations"] = {"legs": [], "legs_as_required": "0/0",
                             "all_legs_as_required": True,
                             "legs_not_measurable_on_this_backend": 0,
                             "legs_not_armable": 0}
    poisoned_verdict = build_verdict(poisoned)
    leg = {
        "defect": "mirror_ghost_as_the_metallic_zero",
        "case": case["key"],
        "case_diverged": not case["bit_identical"],
        "differing_words": case["differing_words"],
        "verdict_under_the_defect": poisoned_verdict,
        "verdict_flipped": not poisoned_verdict["passed"],
        "seconds": round(time.time() - started, 1),
    }
    leg["as_required"] = bool(leg["case_diverged"] and leg["verdict_flipped"])
    results["falsification"] = leg
    log(f"[falsify] planted {leg['defect']}: diverged={leg['case_diverged']} "
        f"words={leg['differing_words']} verdict_flipped={leg['verdict_flipped']}")
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# The source manifest: a gate refuses to release against a tree it disagrees with
# ---------------------------------------------------------------------------

def manifest_reasons(out_path: str, imported: Dict[str, str]) -> List[str]:
    """Write or check ``source_sha256.txt`` beside the artifact.

    An append-only-in-scope campaign manifest. A later run in the same directory
    that imported DIFFERENT bytes for a path already recorded is refused: a single
    artifact directory mixing two trees is an artifact describing a run that never
    happened as one program.
    """
    path = os.path.join(os.path.dirname(out_path) or ".", "source_sha256.txt")
    existing: Dict[str, str] = {}
    reasons: List[str] = []
    if os.path.exists(path):
        with open(path, "r", encoding="ascii") as handle:
            for number, line in enumerate(handle.read().splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2 or len(parts[0]) != 64:
                    reasons.append(f"source manifest line {number} is malformed")
                    continue
                existing[parts[1]] = parts[0]
    for name, digest in sorted(imported.items()):
        if name in existing and existing[name] != digest:
            reasons.append(f"campaign source manifest disagrees for {name}")
    merged = dict(existing)
    merged.update(imported)
    body = "".join(f"{digest} {name}\n" for name, digest in sorted(merged.items()))
    with open(path + ".tmp", "w", encoding="ascii") as handle:
        handle.write(body)
    os.replace(path + ".tmp", path)
    return reasons


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

LEG_NAMES = ("sweep", "multistep", "mutations", "falsify")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "cupy", "numpy"))
    parser.add_argument("--legs", default=",".join(LEG_NAMES))
    parser.add_argument("--product", default="auto",
                        choices=("auto", "full", "reduced"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--multi-step-budget", type=int, default=MULTI_STEP_BUDGET)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("keep", "flush", "match_meep", "ieee"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    args = parser.parse_args(argv)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    legs = tuple(name for name in args.legs.split(",") if name)
    backend_name = args.backend
    if backend_name == "auto":
        backend_name = "cupy" if cp is not None else "numpy"
    product = args.product
    if product == "auto":
        product = "full" if backend_name == "cupy" else "reduced"

    results: Dict[str, Any] = {
        "gate": "cuda_folded_offdiag_kernel",
        "kernel": folded.KERNEL_NAME,
        "question": ("does the FOLDED off-diagonal update_E kernel -- the "
                     "certified emitter plus a MIRROR arm in coord_dn and a "
                     "parity weight on that lane -- reproduce stepping.update_E "
                     "word for word on a mirror-folded grid, and still reduce to "
                     "the certified family where no axis is folded?"),
        "specified_by": "coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend_name,
        "certifies": backend_name == "cupy",
        "product": product,
        "legs_requested": list(legs),
        "seed": SEED,
        "multi_step_budget": args.multi_step_budget,
        "emitter_corpus_digest": folded.corpus_digest(),
        "certified_sibling_corpus_digest": None,
        "gate_sha256": probe.source_digest(os.path.abspath(__file__)),
    }
    from meep_gpu.cuda_kernels import offdiag_emitter  # noqa: PLC0415
    results["certified_sibling_corpus_digest"] = offdiag_emitter.corpus_digest()

    # THE ORDER IS THE WHOLE POINT: the observer wraps NVRTC BELOW the strip so it
    # records the options NVRTC really received, and the policy installs before
    # the first compile.
    if backend_name == "cupy":
        results["nvrtc_binary_observer"] = probe.install_nvrtc_binary_observer()
    if args.import_meep_for_host_policy:
        results["meep_import_for_host_policy"] = probe.import_meep_for_host_policy()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    backend = build_backend(backend_name)
    xp = cp if backend_name == "cupy" else flat._NumpyWearingCupysName()
    if backend_name == "cupy":
        results["environment"] = probe.device_info()
    gate_provenance.stamp(results)
    save(results, out_path)

    if "sweep" in legs:
        run_sweep(backend, xp, results, out_path, product, args.steps, "sweep")
    if "multistep" in legs:
        # THE PRIMARY GUARD ONLY. The control's job is to show the guard is doing
        # work, and one launch per case already shows it; paying 60x for the same
        # demonstration buys nothing.
        run_sweep(backend, xp, results, out_path,
                  "reduced" if product == "full" else product,
                  args.multi_step_budget, "multistep", guards=GUARD_SETS[:1])
    if "mutations" in legs:
        run_mutations(backend, xp, results, out_path, product, args.steps,
                      results.get("sweep", {}).get("cases", []))
    if "falsify" in legs:
        run_falsification(backend, xp, results, out_path, args.steps)

    if backend_name == "cupy":
        results["nvrtc_binaries"] = probe.nvrtc_binary_report()

    verdict = build_verdict(results)
    falsify = results.get("falsification")
    verdict["falsification_as_required"] = (
        None if falsify is None else bool(falsify["as_required"]))
    if falsify is not None and not falsify["as_required"]:
        verdict["passed"] = False
    if "falsify" not in legs:
        verdict["passed"] = False
        verdict["why_not"] = ("the falsification leg did not run; a verdict never "
                              "shown to refuse a planted defect is not a verdict")
    if not results["certifies"]:
        verdict["why_not_certified"] = (
            "the numpy backend executes the emitted source and compiles nothing; "
            "it exercises the harness and the expression tree, it does not "
            "measure NVRTC's output")
    results["verdict"] = verdict

    gate_provenance.stamp(results)
    reasons = manifest_reasons(out_path, results.get(
        gate_provenance.IMPORTED_KEY, {}))
    results["release"] = {
        "released": bool(verdict["passed"] and not reasons),
        "reasons": reasons or ([] if verdict["passed"] else ["gate did not pass"]),
    }
    save(results, out_path)

    log(f"[done] backend={backend_name} "
        f"folded={verdict['folded_single_launch']} "
        f"multi={verdict['folded_multi_step']} "
        f"unfolded={verdict['unfolded_reduction_single_launch']} "
        f"partner={verdict['partner_arm']} own={verdict['own_arm']} "
        f"mutations={verdict['mutation_legs_as_required']} "
        f"falsified={verdict['falsification_as_required']} "
        f"released={results['release']['released']}")
    if not results["certifies"]:
        single = results.get("sweep", {}).get("summary", {})
        mutations = results.get("mutations", {})
        harness_ok = (
            single.get("folded_ran", 0) > 0
            and single.get("folded_identical") == single.get("folded_ran")
            and single.get("unfolded_identical") == single.get("unfolded_ran")
            and mutations.get("all_legs_as_required", True)
            and (falsify is None or falsify["as_required"]))
        return 0 if harness_ok else 1
    return 0 if results["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
