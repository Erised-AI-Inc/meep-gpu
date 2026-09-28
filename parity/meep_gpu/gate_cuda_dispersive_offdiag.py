"""Byte-identity gate for the DISPERSIVE off-diagonal ``update_E`` CUDA kernel.

THE CLAIM UNDER TEST. ``meep_gpu/cuda_kernels/dispersive_offdiag_update_e.py``
emits ``update_E_pml_real_folded_offdiag_dispersive``: MEEP's tensor row
product at ``update_E`` whose OPERANDS are ``D - sum P`` rather than ``D``, over a
possibly mirror-folded stored extent. Does it reproduce ``stepping.update_E``,
word for word -- and does it still reduce to the two certified off-diagonal
families where no pole drives anything?

WHY THIS GATE EXISTS AND WHAT SPECIFIED IT. The 2026-08-20 union census
(``results/cuda_predicate_coverage_2026-08-20_final/``) reads 753 / 759 slots
served by twenty shipped hand-CUDA families. Five of the six residual slots are
complex64. The sixth is ``examples/absorbed_power_density.py`` at ``update_E``,
and FOUR shipped families refuse it, each for a different true reason::

    cuda_constitutive     "dispersion: update_E's source is (D - sum P), not D"
    cuda_folded_offdiag   the same clause, same words
    cuda_dispersive       "off-diagonal chi1inv: the row product reads the other
                           components' (D - sum P) volumes at neighbouring cells"
    cuda_offdiag          "mirror symmetry: the fold changes the stored extent"

Read together those four ARE the specification. This gate measures whether the
kernel written to it works.

=============================================================================
THE FUSION QUESTION, SETTLED BY MEASUREMENT RATHER THAN BY PREFERENCE
=============================================================================

The census note says this slot may want "a new kernel OR a measured fusion of two
shipped ones". THE ``composition`` LEG IS THAT MEASUREMENT. It runs the only
route that exists -- materialize the three ``D - sum P`` volumes with the array
path, then launch the SHIPPED ``update_E_pml_real_folded_offdiag`` with
those buffers bound in place of ``D`` -- against the same oracle on the same
frozen state, and records:

* whether it is bit-identical (it should be: the shipped kernel reads its
  operands only through the three ``D`` pointers, and the materialization is the
  array path's own ``displacement_minus_polarization_volumes``);
* what it COSTS, in FULL-VOLUME PASSES and device allocations per step, counted
  exactly rather than timed. A shared box cannot give a trustworthy time; a pass
  count is exact and contention-free.

It is a fusion of ONE shipped kernel with the ARRAY PATH, not of two kernels, and
the leg records why: neither shipped kernel exposes ``D - sum P`` as an output.
``update_E_pml_real_dispersive`` writes ``(D - sum P) * inv_eps`` into
``f_w`` and accumulates ``E``; recovering the operand needs a DIVISION, which is
not the identity in float32, and the ``E`` accumulation would have to be undone.

=============================================================================
WHAT MAKES A CASE NON-VACUOUS, AND WHAT REFUSES ONE
=============================================================================

A gate that cannot fail certifies nothing, and this sub-step has five ways to be
vacuous. Each is measured PER CASE and a case below its floor is REFUSED:

* ``oracle_moved`` -- zero-init is a FIXED POINT of this recurrence;
* ``coupling_is_live`` -- what the off-diagonal rows contributed, from
  ``stepping._offdiagonal_terms`` itself over EVERY component. A case whose
  coupling is identically zero is testing the plain dispersive kernel;
* ``pole_chain_bites`` -- whether ``D - sum P`` differs BITWISE from ``D``. A case
  whose chain is inert is a NON-dispersive case wearing dispersive clothes, and
  every ordering leg would come back UNCAUGHT for a reason about the fixture;
* ``folded_axis_absorbs`` -- ``max|kps-1|`` ON THE FOLDED AXIS. A folded axis whose
  half-integer profile is the identity cannot show a coefficient-index error on
  the axis the fold moved;
* ``mirror_ghost_is_live`` -- the ghost is ``parity * (D - sum P)[2]``, so the
  floor is measured on the CHAIN and not on D. Where that row is identically zero
  the mirror ghost EQUALS the metallic zero it is being distinguished from.

=============================================================================
THE BATTERY'S OWN TWO-SIDEDNESS
=============================================================================

Every swept case must be BIT-IDENTICAL, so the sweep alone cannot show that the
comparator works. That is carried by the MUTATION battery, and by three legs in
particular:

* ``couple_the_raw_D`` -- the MIDDLE REFUSAL planted: the polarization subtracted
  from the diagonal term while the row product couples the raw ``D`` volumes. If
  this is not caught, this family has no reason to exist over ``cuda_folded_offdiag``;
* ``sum_then_subtract`` and ``reverse_pole_order`` -- DISCRIMINATORS in the ARITY:
  caught at two poles and above, a NULL at one, because both forms are the same
  expression at one pole. A battery at arity one would score them UNCAUGHT and be
  measuring the fixture;
* ``drop_the_parity_weight`` -- a DISCRIMINATOR in the PLANE PARITY: the weight is
  ``-phase``, so on an EVEN plane it is -1 and dropping it must be caught, while
  on an ODD plane it is +1 and dropping it is the identity.

=============================================================================
THE RELEASE VERDICT IS ITSELF FALSIFIED
=============================================================================

``--legs falsify`` re-runs the whole verdict computation over a sweep whose one
case carries a planted defect, and REQUIRES ``passed`` to come back False.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=2 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_dispersive_offdiag.py --backend cupy \\
        --product full --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend EXECUTES the emitted device source through
the slice test module's own evaluator -- one grammar, one copy, imported rather
than duplicated. IT COMPILES NOTHING AND CERTIFIES NOTHING::

    python -u gate_cuda_dispersive_offdiag.py --backend numpy \\
        --product reduced --out /tmp/dispersive_offdiag_local.json

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
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402
from meep_gpu.cuda_kernels import dispersive_offdiag_update_e as family  # noqa: E402
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
#: terminations appear, and one, two and three simultaneous planes all appear.
#: The two UNFOLDED controls carry the unfolded arm, which this family ADMITS --
#: unlike the folded off-diagonal family, there is no shipped competitor there.
FOLD_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "unfolded_periodic", "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (8.0, 10.0, 12.0)},
    {"label": "unfolded_metallic", "axes": "", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"),
     "cell": (8.0, 10.0, 12.0)},

    # Folded X: the SLOWEST-stride coefficient index. Both terminations, both
    # plane parities.
    {"label": "fold_X_periodic", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_metallic_odd_plane", "axes": "X", "phase": -1,
     "boundaries": ("metallic", "periodic", "periodic"),
     "cell": (16.0, 8.0, 12.0)},

    # Folded Y: the corpus row's own plane (absorbed_power_density.py).
    {"label": "fold_Y_periodic_corpus", "axes": "Y", "phase": 1,
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

    # A FOLD BESIDE A LIVE METALLIC WALL, which no other spec here carries: a
    # folded axis always reports ``wm = 0``, so a sweep of folded specs alone
    # leaves every wall flag zero and the wall mask untested on a fold.
    {"label": "fold_X_wall_Y", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"),
     "cell": (16.0, 8.0, 12.0)},

    # TWO PLANES AT ONCE, and THREE, both terminations.
    {"label": "fold_XY_mixed", "axes": "XY", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"),
     "cell": (16.0, 16.0, 10.0)},
    {"label": "fold_XYZ_periodic", "axes": "XYZ", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (16.0, 16.0, 16.0)},
)

#: The row masks, CHOSEN so that a folded axis lands in each of its three roles.
#: A sweep over the corpus mask alone would answer one third of the question and
#: report it as the whole.
ROW_MASK_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_15row", "mask": (1, 0, 0, 1, 0, 0),
     "why": "the mask absorbed_power_density.py drives -- slots (Ex,Ey) and "
            "(Ey,Ex), so partner axes X and Y are live and Z is in NEITHER role"},
    {"label": "full_tensor", "mask": (1, 1, 1, 1, 1, 1),
     "why": "every axis is some live slot's partner: the fold is always reachable"},
    {"label": "single_Ey_Ez", "mask": (0, 0, 1, 0, 0, 0),
     "why": "one slot: own axis Y, partner axis Z -- a folded X is in neither "
            "role, which is the discriminators' NULL arm, and Ex and Ez stay on "
            "the PLAIN dispersive arm beside a coupled component"},
)

#: The pole arities. THE FAMILY'S OWN LIST, imported, so the gate cannot sweep an
#: arity the emitter refuses -- or miss one the predicate admits.
ARITIES: Tuple[Tuple[int, int, int], ...] = family.POLE_COUNTS_SWEPT

#: The arity the corpus row drives, and the one the ordering discriminators need.
CORPUS_ARITY = (1, 1, 1)
MULTI_POLE_ARITIES = tuple(a for a in ARITIES if max(a) >= 2)

#: The reduced product's specs, NAMED rather than sliced. A head slice of the
#: list is all X folds and no declared wall, which leaves the wall discriminators
#: with one arm and every axis question answered on one axis.
REDUCED_SPEC_LABELS: Tuple[str, ...] = (
    "unfolded_metallic", "fold_Y_periodic_corpus", "fold_X_metallic_odd_plane",
    "fold_Z_periodic_odd_plane", "fold_X_wall_Y", "fold_XYZ_periodic")

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient AND every susceptibility coefficient, so it is a real
#: second draw of the tables.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: Spatially VARYING row coefficients: a uniform coefficient is what makes the
#: registration rather than merely the rounding invisible.
ROW_FORM = "varying"

#: ``--fmad=false`` is CORRECTNESS on this sub-step, not tuning. Both parent
#: families measured their unguarded controls diverging; this body is their union
#: plus the gathered subtraction chains, so the same control is carried here
#: rather than inherited.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = flat.GUARD_SETS


# ---------------------------------------------------------------------------
# The role a fold plays, and what that licenses
# ---------------------------------------------------------------------------

def fold_roles(grid, mask: Sequence[int]) -> Dict[str, Any]:
    """Which role each folded axis plays in this row mask.

    ``coverage.offdiag_fold_roles`` is the SHIPPED rule and is asked here rather
    than re-derived. It knows nothing about poles, and it does not need to: the
    fold acts on the SHIFT HELPERS while the poles act on the values they move.
    """
    mirrored = tuple(bool(grid.is_mirrored(axis)) for axis in range(3))
    roles = dict(coverage.offdiag_fold_roles(tuple(mask), mirrored))
    for key in ("folded_axes", "live_partner_axes", "live_own_axes",
                "folded_partner_axes", "folded_own_axes"):
        roles[key] = list(roles[key])
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


def register_poles(fields, grid, arity: Sequence[int], rng):
    """Build real ``PolarizationState`` objects at one arity, P seeded away from 0.

    THE ARITY IS BUILT FROM REAL OBJECTS, never faked: a per-component sigma of
    zero is what ``PolarizationState`` filters on (dispersion.py:641-643), which
    is the engine's own spelling of "this term does not drive that component" and
    is exactly what MEEP's ``needs_P`` / ``trivial_sigma`` pair does.

    ARITY (0, 0, 0) STILL REGISTERS ONE STATE. That is what the configuration IS:
    ``fields.polarizations`` truthy -- which both non-dispersive off-diagonal
    predicates refuse on -- with a chain of zero on every component, which the
    array path serves by ALIASING D. A fixture that registered nothing would build
    the shipped folded family's configuration and the reduction leg would be
    comparing that family to itself.
    """
    kinds = ("lorentzian", "drude")
    xp = grid.xp
    for index in range(max(1, max(arity))):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: (0.35 + 0.04 * index if index < arity[axis] else 0.0)
                 for axis, name in enumerate(COMPONENTS)}
        state = PolarizationState(term, sigma, grid, fields._field_dtype())
        for component in state.driven():
            state.P[component][...] = xp.asarray(np.ascontiguousarray(
                rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)))
            state.P_prev[component][...] = xp.asarray(np.ascontiguousarray(
                rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)))
        fields.polarizations.append(state)
    return fields.polarizations


def seed_pole_state(fields, grid, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Re-seed the pole buffers in the requested value class.

    THE MATERIAL AND THE COEFFICIENTS STAY NORMAL under the band class. Driving
    them into the band too would make every product underflow and the leg would
    measure the fixture rather than the policy's reach into this arithmetic.
    """
    if value_class != "subnormal_band":
        return {}
    xp = grid.xp
    shape = tuple(int(n) for n in grid.shape)
    names, arrays = [], []
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            names.append(f"P{index}_{component}")
            arrays.append((state, component))
    if not names:
        return {}
    host = probe.subnormal_band_hosts(names, shape, rng)
    for name, (state, component) in zip(names, arrays):
        state.P[component][...] = xp.asarray(np.ascontiguousarray(host[name]))
    return host


def pole_plan_arrays(plan: Dict[str, Any]) -> Dict[str, Any]:
    """``{'P_Ex_0': array, ...}`` in the kernel signature's own order.

    TAKES THE PLAN rather than the fields, so a leg that hands the launcher a
    deliberately reversed chain hands the evaluator the same reversed chain. A
    version that re-resolved from ``fields`` would leave the two backends
    measuring different permutations and the host ordering defect would score
    UNCAUGHT on one of them for a reason about the harness.
    """
    out: Dict[str, Any] = {}
    for target, _source, _axis in family.E_TERMS:
        for index, array in enumerate(plan[target]):
            out[f"P_{target}_{index}"] = array
    return out


def chain_volumes(fields, layer) -> Dict[str, np.ndarray]:
    """The three ``D - sum P`` volumes, on the HOST, copied.

    COPIED because ``update_E`` rewrites the same per-component scratch buffers
    (fields.py:1131), so a floor read after the loop would be a fact about the
    last step rather than about the state the comparison started from.
    """
    displacement = stepping._nonlinear_displacement(fields, layer)
    return {name: to_host(volume).copy()
            for name, volume in displacement["volumes"].items()}


def coupling_is_live(fields, layer) -> Dict[str, Any]:
    """What the off-diagonal rows contributed, over EVERY component.

    From ``stepping._offdiagonal_terms`` itself, not from a re-derivation, and
    over all three components rather than Ex alone: a mask whose only live slot is
    (Ey, Ez) leaves Ex's coupling identically zero, and a floor that read Ex would
    refuse a case perfectly able to discriminate.
    """
    displacement = stepping._nonlinear_displacement(fields, layer)
    out: Dict[str, Any] = {"per_component": {}, "max_abs": 0.0}
    for component in COMPONENTS:
        term = stepping._offdiagonal_terms(fields, component, displacement)
        value = (0.0 if term is None
                 else float(np.max(np.abs(to_host(term).astype(np.float64)))))
        out["per_component"][component] = value
        out["max_abs"] = max(out["max_abs"], value)
    out["meets_floor"] = out["max_abs"] > 0.0
    return out


def pole_chain_bites(fields, layer) -> Dict[str, Any]:
    """Does ``D - sum P`` differ BITWISE from ``D``?

    A case whose poles are all zero is a NON-dispersive case wearing dispersive
    clothes: ``sum_then_subtract``, ``reverse_pole_order``, ``drop_one_pole`` and
    ``couple_the_raw_D`` would every one come back UNCAUGHT, for a reason about
    the fixture rather than about the kernel. Arity (0, 0, 0) is EXPECTED to sit
    at this floor and is scored as the reduction arm instead.
    """
    volumes = chain_volumes(fields, layer)
    out: Dict[str, Any] = {"per_component": {}, "moved": 0, "total": 0}
    for target, source, _axis in family.E_TERMS:
        chain = np.ascontiguousarray(volumes[target], dtype=np.float32)
        raw = np.ascontiguousarray(to_host(getattr(fields, source)),
                                   dtype=np.float32)
        moved = int(np.count_nonzero(
            chain.ravel().view(np.uint32) != raw.ravel().view(np.uint32)))
        out["per_component"][target] = moved
        out["moved"] += moved
        out["total"] += int(chain.size)
    return out


def folded_axis_absorbs(layer, grid) -> Dict[str, Any]:
    """Does the FOLDED axis's HALF-INTEGER coefficient profile beat the identity?"""
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


def mirror_ghost_is_live(fields, layer, grid) -> Dict[str, Any]:
    """Is the mirror ghost distinguishable from the metallic zero it replaces?

    THE FLOOR IS MEASURED ON THE CHAIN, NOT ON D, and that is this family's own
    difference from the folded off-diagonal gate's version: the array path shifts
    ``volumes[partner]``, which is ``D - sum P``, so the ghost is
    ``parity * (D - sum P)[MIRROR_SOURCE_INDEX]``. A fixture whose D row 2 is
    nonzero but whose CHAIN row 2 cancels to zero would pass the D-side floor and
    still be measuring nothing.
    """
    source_row = int(stepping.MIRROR_SOURCE_INDEX)
    volumes = chain_volumes(fields, layer)
    out: Dict[str, Any] = {"source_row": source_row, "axes": [],
                           "min_max_abs": None}
    for axis in range(3):
        if not grid.is_mirrored(axis):
            continue
        for component in COMPONENTS:
            volume = volumes[component]
            plane = volume[_plane_index(axis, source_row, volume.shape)]
            value = float(np.max(np.abs(plane.astype(np.float64))))
            out["axes"].append({"axis": axis, "component": component,
                                "max_abs": value})
            out["min_max_abs"] = (value if out["min_max_abs"] is None
                                  else min(out["min_max_abs"], value))
    out["meets_floor"] = (not out["axes"]) or (out["min_max_abs"] or 0.0) > 0.0
    return out


def device_arrays(fields, rows, poles) -> Dict[str, Any]:
    """Device parameter name -> array, for the evaluator backend."""
    arrays = flat.device_arrays(fields, rows)
    arrays.update(poles)
    return arrays


def snapshot(fields) -> Dict[str, np.ndarray]:
    """Everything a leg must restore: the outputs AND the inputs it consumes."""
    frozen = {name: to_host(getattr(fields, name)).copy() for name in STATE}
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            frozen[f"P{index}_{component}"] = to_host(state.P[component]).copy()
    return frozen


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name in STATE:
        getattr(fields, name)[...] = xp.asarray(frozen[name])
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            key = f"P{index}_{component}"
            if key in frozen:
                state.P[component][...] = xp.asarray(frozen[key])


def advance_sources(fields) -> None:
    """Move D between launches, exactly the same way on both paths.

    Held fixed, ``f_w`` reaches a fixed point and 60 launches measure what one
    launch does. The POLES are deliberately NOT advanced here: ``update_P`` is
    another family's slot, and moving P with a rule this gate invented would put a
    transcription nobody certified on the oracle leg.
    """
    for name in ("Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = getattr(fields, name) * np.float32(0.97)


# ---------------------------------------------------------------------------
# The three kernel-side backends
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
            test_dispersive_offdiag_update_e as slice_tests)
        _SLICE_TESTS = slice_tests
    return _SLICE_TESTS


def evaluator_blind_lines(text: str) -> List[str]:
    """Lines of an emitted body the NumPy evaluator neither EXECUTES nor pins.

    THE HAZARD THIS EXISTS FOR, measured on the certified family's first laptop
    run: the evaluator is a ``finditer`` over an anchored grammar, so a mutation
    that rewrites a statement into a form the grammar does not match is not
    applied at all -- it is SILENTLY SKIPPED, and the leg then reports a verdict
    that says nothing. On the device backend the compiler executes every line and
    the question does not arise, which is why the two backends' catch tables
    differ and why both are reported.
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
        if any(token in line for token in tests._SCAFFOLDING):
            continue
        blind.append(stripped)
    return blind


def classify_for_evaluator(pristine: str, mutated: str) -> Dict[str, Any]:
    """Can the NumPy evaluator see this defect at all, and if not, what guards it?

    This family's evaluator PARSES its ghost rules, its pole chains, its two
    ghosted wrappers and its term body rather than transcribing them. Only the
    helpers INHERITED verbatim from the two certified emitters are pinned, so the
    ``pinned_half`` bucket is small here.
    """
    inherited = {}
    for kind, name, owner in family.prelude_provenance():
        donor = (family._flat.PRELUDE if owner == "offdiag_emitter"
                 else folded._OWN_PRELUDE)
        inherited[name] = (family._device_block(donor, name) if kind == "helper"
                           else family._define_block(donor, name))
    broken = [name for name, block in inherited.items() if block not in mutated]
    tests = slice_module()
    for pattern, label in ((tests._DMP_BODY, "dmp"),
                           (tests._DMP_GHOSTED_BODY, "dmp_ghosted"),
                           (tests._DMP_MIRROR_BODY, "dmp_ghosted_mirror"),
                           (tests._TERM_BODY, "term")):
        if pattern.search(mutated) is None:
            return {"evaluator_sees_it": False, "why": "outside_the_grammar",
                    "unparsable": label, "broken_pins": broken}
    if broken:
        # An inherited helper rewritten is a defect the evaluator CAN see through
        # ``coord_dn``/``coord_up`` (parsed) but NOT through ``flat``/``ghosted``/
        # ``constitutive_apply`` (transcribed). Both are reported.
        parsed = {"coord_up", "coord_dn", "MIRROR_ROW"}
        if not set(broken) <= parsed:
            return {"evaluator_sees_it": False, "why": "inherited_pinned_half",
                    "broken_pins": broken}
    blind = evaluator_blind_lines(mutated)
    if blind and blind != evaluator_blind_lines(pristine):
        return {"evaluator_sees_it": False, "why": "outside_the_grammar",
                "unmatched_lines": blind[:4]}
    return {"evaluator_sees_it": True, "why": "arithmetic", "broken_pins": broken}


class EvaluatorBackend:
    """The kernel side, executed in NumPy from the emitted text. Compiles nothing.

    Present so the harness can be shown to FAIL where it must without a device. It
    answers ``certifies = False`` and every summary carries that through, so no run
    on this backend can be mistaken for a certification.
    """

    name = "numpy"
    certifies = False
    kind = "kernel"

    def __init__(self):
        self.source_transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self.sources_used: Dict[str, str] = {}

    def set_guard(self, guard: Sequence[str]) -> None:
        """No compiler, so no guard. Recorded rather than silently ignored."""

    def set_source_mutation(self, transform) -> None:
        self.source_transform = transform

    def pristine_source(self, mask, arity) -> str:
        return family.dispersive_offdiag_source(tuple(mask), tuple(arity))

    def source_for(self, mask, arity) -> Tuple[str, int]:
        text = self.pristine_source(mask, arity)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def launch(self, fields, layer, rows, poles, mask, arity, tables, codes,
               walls, weights, plan=None) -> Dict[str, Any]:
        text, _sites = self.source_for(mask, arity)
        self.sources_used[str((tuple(mask), tuple(arity)))] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        slice_module().evaluate_dispersive_offdiag_source(
            text, device_arrays(fields, rows, poles), tables, codes, walls,
            weights, tuple(fields.grid.shape))
        return {"kernel_launches": 1, "extra_full_volume_passes": 0,
                "scratch_volumes": 0}

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
    """

    name = "cupy"
    certifies = True
    kind = "kernel"

    def __init__(self):
        self.source_transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self.sources_used: Dict[str, str] = {}
        self._overridden: List[Tuple[Any, Any]] = []

    def set_guard(self, guard: Sequence[str]) -> None:
        family._COMPILE_OPTIONS = tuple(guard)
        family.clear_kernel_cache()

    def set_source_mutation(self, transform) -> None:
        self.source_transform = transform
        for mask, arity in self._overridden:
            family.set_kernel_source(mask, arity, None)
        self._overridden = []
        family.clear_kernel_cache()

    def pristine_source(self, mask, arity) -> str:
        """The UNMUTATED source. Site counting and digesting must never go through
        an installed override: that would apply the transform twice and the leg
        would hash a body it never compiled."""
        return family.dispersive_offdiag_source(tuple(mask), tuple(arity))

    def source_for(self, mask, arity) -> Tuple[str, int]:
        text = self.pristine_source(mask, arity)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def launch(self, fields, layer, rows, poles, mask, arity, tables, codes,
               walls, weights, plan=None) -> Dict[str, Any]:
        key = (tuple(mask), tuple(arity))
        text, _sites = self.source_for(*key)
        if self.source_transform is not None:
            family.set_kernel_source(key[0], key[1], text)
            if key not in self._overridden:
                self._overridden.append(key)
        self.sources_used[str(key)] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        family.update_E_dispersive_offdiag_fused_pml_real(
            fields, tables=tables, codes=codes, walls=walls, weights=weights,
            plan=plan)
        cp.cuda.runtime.deviceSynchronize()
        return {"kernel_launches": 1, "extra_full_volume_passes": 0,
                "scratch_volumes": 0}

    def compile_log(self) -> List[Dict[str, Any]]:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

        return list(compile_cache.compile_log())

    def clear(self) -> int:
        for mask, arity in self._overridden:
            family.set_kernel_source(mask, arity, None)
        self._overridden = []
        return family.clear_kernel_cache()


class CompositionBackend:
    """THE MEASURED ALTERNATIVE: materialize ``D - sum P``, then the SHIPPED kernel.

    THIS IS NOT A FUSION OF TWO SHIPPED KERNELS AND THE RECORD SAYS SO. Neither
    shipped kernel exposes ``D - sum P`` as an output:
    ``update_E_pml_real_dispersive`` writes ``(D - sum P) * inv_eps`` into
    ``f_w`` and accumulates ``E`` (recovering the operand needs a DIVISION, which
    is not the identity in float32, and the accumulation would have to be undone),
    and ``update_E_pml_real_folded_offdiag`` has no pole parameters at all.
    So the only composition that exists is one shipped kernel plus the ARRAY PATH.

    WHAT IT COSTS, counted rather than timed, because a shared box cannot give a
    trustworthy time and a pass count is exact: ``sum(arity)`` extra full-volume
    subtraction passes plus one full-volume copy per driven component, and three
    scratch volumes held live. The fused kernel's numbers are zero and zero.

    THE D POINTERS ARE REBOUND AROUND THE LAUNCH and restored in a ``finally``.
    The shipped launcher reads ``fields.Dx`` directly, and there is no other way
    to hand it the materialized volumes without editing it -- which this file does
    not do.
    """

    name = "composition"
    #: NEVER. This backend measures an ALTERNATIVE, not the kernel under test.
    certifies = False
    kind = "composition"

    def __init__(self, inner):
        self.inner = inner
        self.source_transform = None
        self.sources_used: Dict[str, str] = {}

    def set_guard(self, guard: Sequence[str]) -> None:
        folded._COMPILE_OPTIONS = tuple(guard)
        folded.clear_kernel_cache()

    def set_source_mutation(self, transform) -> None:
        if transform is not None:
            raise ValueError(
                "the composition leg measures the SHIPPED folded off-diagonal "
                "kernel; mutating this family's emitter would not reach it")

    def pristine_source(self, mask, arity) -> str:
        return folded.folded_offdiag_source(tuple(mask))

    def source_for(self, mask, arity) -> Tuple[str, int]:
        return self.pristine_source(mask, arity), 0

    def launch(self, fields, layer, rows, poles, mask, arity, tables, codes,
               walls, weights, plan=None) -> Dict[str, Any]:
        volumes = fields.displacement_minus_polarization_volumes()
        originals = {name: getattr(fields, "D" + name[1])
                     for name in COMPONENTS}
        try:
            for component in COMPONENTS:
                setattr(fields, "D" + component[1], volumes[component])
            folded.update_E_folded_offdiag_fused_pml_real(
                fields, tables=tables, codes=codes, walls=walls,
                weights=weights)
            if cp is not None:
                cp.cuda.runtime.deviceSynchronize()
        finally:
            for component, array in originals.items():
                setattr(fields, "D" + component[1], array)
        driven = sum(1 for count in arity if count > 0)
        return {"kernel_launches": 1,
                "extra_full_volume_passes": int(sum(arity)) + driven,
                "scratch_volumes": driven}

    def compile_log(self) -> List[Dict[str, Any]]:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

        return list(compile_cache.compile_log())

    def clear(self) -> int:
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
# THE FIRST GROUP IS THIS FAMILY'S OWN: the four gathered field reads stated
# wrongly in each of the ways the reading invites. The rest are the two parent
# families' defects re-anchored on this grammar, carried because this kernel emits
# that arithmetic too and a gate that only tested its own delta would certify a
# body it never checked.

_TERM_RETURN = family.TERM_RETURN_LINE.strip()
_MIRROR_SELECT = family.MIRROR_SELECT_LINE.strip()
_MIRROR_BRANCH = "if (bc == BC_MIRROR) return MIRROR_ROW;"


def _sub(pattern: str, replacement, text: str) -> Tuple[str, int]:
    new, count = re.subn(pattern, replacement, text)
    return new, count


def _replace(needle: str, replacement: str, text: str) -> Tuple[str, int]:
    return text.replace(needle, replacement), text.count(needle)


def m_couple_the_raw_D(text):
    """THE MIDDLE REFUSAL, planted -- and SURGICALLY, which took one measurement.

    The defect is "subtract the polarization from the diagonal term, then couple
    the RAW D volumes". Only the COUPLING's four gathered reads change: the
    diagonal ``gs_c`` keeps its chain, and THE MIRROR PARITY IS LEFT EXACTLY
    WHERE IT WAS.

    THE FIRST VERSION OF THIS LEG ALSO DELETED THE MIRROR WRAPPER, and the laptop
    run scored it CAUGHT on two arity-(0,0,0) FOLDED cases -- where the chain is
    empty and there is nothing about ``D - sum P`` left to be wrong. It was
    catching the missing parity weight, not the missing chain, and a leg that
    conflates two defects measures neither. Three targeted rewrites instead:

    * ``dmp_c(g, POLES, home)``        -> ``g[home]``
    * ``dmp_ghosted_c(g, POLES, up)``  -> ``ghosted(g, up)``
    * ``dmp_ghosted_mirror_c``'s own load -> ``g[index]``

    At arity (0, 0, 0) all three are the IDENTITY by construction, which is what
    gives this leg its null arm.
    """
    out, first = _sub(r"dmp_(E[xyz])\(g((?:, P_\w+)*), (home|up|corner|down)\)",
                      r"g[\3]", text)
    out, second = _sub(
        r"dmp_ghosted_(E[xyz])\(g((?:, P_\w+)*), (home|up|corner|down)\)",
        r"ghosted(g, \3)", out)
    out, third = _sub(
        r"    float value = dmp_(E[xyz])\(g((?:, P_\w+)*), index\);",
        r"    float value = g[index];", out)
    return out, first + second + third


def m_diagonal_reads_raw_D(text):
    """``gs_c`` reading ``D_c[idx]`` instead of the chain: the non-dispersive body
    with a dispersive coupling bolted on -- the mirror image of the leg above."""
    return _sub(r"float gs_(E[xyz]) = dmp_\1\((D[xyz])(?:, P_\w+)*, idx\);",
                r"float gs_\1 = \2[idx];", text)


def m_up_leg_reads_raw_D(text):
    """THE HALF-DONE EDIT. ``dmp_ghosted_<c>`` is the OWN-axis up leg of every
    term; leaving its interior load as the raw D while the near leg reads the
    chain is what an author reaches by fixing the obvious read and missing the
    gathered one."""
    return _sub(
        r"    return \(index < 0\) \? 0\.0f : dmp_(E[xyz])\(g((?:, P_\w+)*), index\);",
        r"    return (index < 0) ? 0.0f : g[index];", text)


def m_sum_then_subtract(text):
    """``D - (P0 + P1 + ...)`` instead of ``((D - P0) - P1) - ...``.

    A DISCRIMINATOR IN THE ARITY, and written generically over it so that BOTH
    arms exist: float32 addition is not associative, but ``D - (P0)`` and
    ``(D - P0)`` are the same expression, so this must be CAUGHT at two poles and
    above and be a NULL at one. A battery armable only at two would have no null
    arm and could not tell a working comparator from one that fails everything.
    """
    def rewrite(match):
        poles = re.findall(r"P_\w+", match.group(0))
        summed = " + ".join(f"{name}[index]" for name in poles)
        return f"    float value = g[index] - ({summed});"

    return _sub(r"    float value = g\[index\];"
                r"(?:\n    value = value - P_\w+\[index\];)+", rewrite, text)


def m_reverse_pole_order(text):
    """The chain in ``fields.polarizations`` order REVERSED (fields.py:1125).

    The same arity discriminator as above, and written the same way so it has the
    same two arms: at ONE pole the reversal is the identity ON THE TEXT ITSELF,
    which is a site that cannot bite, and at two and above it is a different
    float32 number.
    """
    def rewrite(match):
        return "\n".join(reversed(match.group(0).split("\n")))

    return _sub(r"    value = value - P_\w+\[index\];"
                r"(?:\n    value = value - P_\w+\[index\];)*", rewrite, text)


def m_drop_one_pole(text):
    """One contributor silently missing from a chain. Half a susceptibility: the
    field stays smooth, the dispersion is wrong."""
    return _sub(r"\n    value = value - P_E[xyz]_1\[index\];", "", text)


def m_parity_weights_the_raw_D(text):
    """The mirror parity applied to ``D`` and the poles subtracted AFTER.

    The array path shifts the already-formed ``D - sum P`` volume and weights face
    0 (stepping.py:1870-1872), so the parity multiplies the CHAIN. Observable
    wherever the parity is -1, which is every EVEN plane -- and inert wherever the
    chain is empty, which is the other half of its discriminator.

    ONE ``re.sub`` WITH A REPLACEMENT FUNCTION rather than a loop over
    ``finditer`` on a string the loop is also rewriting: the first version did
    that and reported 3 sites where there are 9, because every match after the
    first was found at an offset the mutated text no longer had.
    """
    def rewrite(match):
        poles = re.findall(r"P_\w+", match.group(1))
        if not poles:
            return match.group(0)
        chain = "".join(f" - {name}[index]" for name in poles)
        return ("    float value = mg ? (w * g[index]) : g[index];\n"
                f"    return value{chain};")

    return _sub(r"    float value = dmp_E[xyz]\(g((?:, P_\w+)*), index\);\n"
                r"    return mg \? \(w \* value\) : value;", rewrite, text)


def m_couple_the_wrong_chain(text):
    """Each term reads its PARTNER's chain. Reading the ROW component's chain
    instead is the defect an author makes by carrying one component name through
    the helper suite -- a tensor with the wrong anisotropy in it."""
    rotate = {"Ex": "Ey", "Ey": "Ez", "Ez": "Ex"}
    return _sub(
        r"        (D[xyz])((?:, P_(?:Ex|Ey|Ez)_\d+)+), (chi1inv_\w+), idx,",
        lambda m: "        {}{}, {}, idx,".format(
            m.group(1),
            re.sub(r"P_(Ex|Ey|Ez)_", lambda n: f"P_{rotate[n.group(1)]}_",
                   m.group(2)),
            m.group(3)),
        text)


def m_mirror_ghost_as_the_metallic_zero(text):
    """The shipped CERTIFIED off-diagonal kernel's only available answer for a
    folded axis, planted. Caught wherever the mirror ghost is reachable, a NULL
    where it is not."""
    return _replace(_MIRROR_BRANCH, "if (bc == BC_MIRROR) return -1;", text)


def m_mirror_row_off_by_one(text):
    """The reflected row off by one -- a half-cell registration error on the fold
    plane: smooth, converged and wrong."""
    return _replace("#define MIRROR_ROW 2", "#define MIRROR_ROW 1", text)


def m_flip_the_parity_sign(text):
    """``+phase`` instead of ``-phase``: the sign error a reader of
    ``mp.Mirror(direction, phase)`` makes."""
    return _replace(_MIRROR_SELECT, "return mg ? (-w * value) : value;", text)


def m_drop_the_parity_weight(text):
    """THE SHARPEST DISCRIMINATOR IN THE PLANE PARITY. The weight is ``-phase``:
    on an EVEN plane it is -1 and dropping it must be caught, on an ODD plane it
    is +1 and dropping it is the identity."""
    return _replace(_MIRROR_SELECT, "return value;", text)


def m_folded_own_axis_wraps(text):
    """``coord_up`` wrapping on a fold instead of serving the exact zero.
    ``_offdiagonal_terms`` passes no ``reflect_row`` (stepping.py:1248-1249)."""
    return _replace("return (bc == BC_PERIODIC) ? 0 : -1;",
                    "return (bc == BC_METALLIC) ? -1 : 0;", text)


def m_ghost_lane_from_the_wall_flag(text):
    """The ``wm_*``/``bc_*`` split collapsed: the ghost lane read off the grid's
    DECLARATION instead of the resolved ghost rule. A folded axis reports
    ``wm = 0``, so the parity would never be applied at all."""
    return _sub(r"int mg_([xyz]) = \(bc_[xyz] == BC_MIRROR\) && at_[xyz];",
                r"int mg_\1 = wm_\1 && at_\1;", text)


def m_drop_a_row_from_the_volume_sum(text):
    """One partner's term silently missing from a component's total."""
    return _sub(r"\n    total_(\w+) = total_\1 \+ term_\w+;", "", text)


def m_invert_the_wall_mask(text):
    """The metallic wall-coupling select inverted: kept at the wall, zeroed
    inside. A defect on EVERY grid, which is why it is not a discriminator --
    measured that way on the folded family's first device run."""
    return _sub(r"\? 0\.0f : total_(\w+);", r"? total_\1 : 0.0f;", text)


def m_drop_the_wall_mask(text):
    """The mask never applied. Its discriminator: a NULL on a grid that declares
    no metallic axis a live row masks against."""
    return _sub(r"\n *total_\w+ = \(wm_[xyz] && at_[xyz]\) \? 0\.0f : total_\w+;",
                "", text)


def m_hoist_the_coefficient(text):
    """The four-point average times ``u[home]``. The same ALGEBRA only for a
    uniform coefficient, and a different float32 number even then."""
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
    round-to-nearest AWAY FROM UNDERFLOW. Armed on the uniform class only."""
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
    "couple_the_raw_D": m_couple_the_raw_D,
    "diagonal_reads_raw_D": m_diagonal_reads_raw_D,
    "up_leg_reads_raw_D": m_up_leg_reads_raw_D,
    "sum_then_subtract": m_sum_then_subtract,
    "reverse_pole_order": m_reverse_pole_order,
    "drop_one_pole": m_drop_one_pole,
    "parity_weights_the_raw_D": m_parity_weights_the_raw_D,
    "couple_the_wrong_chain": m_couple_the_wrong_chain,
    "mirror_ghost_as_the_metallic_zero": m_mirror_ghost_as_the_metallic_zero,
    "mirror_row_off_by_one": m_mirror_row_off_by_one,
    "flip_the_parity_sign": m_flip_the_parity_sign,
    "drop_the_parity_weight": m_drop_the_parity_weight,
    "folded_own_axis_wraps": m_folded_own_axis_wraps,
    "ghost_lane_from_the_wall_flag": m_ghost_lane_from_the_wall_flag,
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

#: DISCRIMINATORS IN THE POLE ARITY: the same expression at one pole, a different
#: float32 number at two and above. Caught where the case's arity has a chain of
#: two or more, a NULL where it does not.
ARITY_DISCRIMINATORS: Tuple[str, ...] = ("sum_then_subtract",
                                         "reverse_pole_order")

#: DEFECTS THAT NEED A LIVE POLE CHAIN AT ALL. At arity (0, 0, 0) the emitted tree
#: is the folded off-diagonal family's and every one of these is the identity --
#: which is the reduction arm, measured rather than claimed.
CHAIN_MUTATIONS: Tuple[str, ...] = (
    "couple_the_raw_D", "diagonal_reads_raw_D", "up_leg_reads_raw_D",
    "parity_weights_the_raw_D")

#: Defects that HAVE NO SITE without a live chain, so their null arm is the
#: REFUSAL ("the mutation matched nothing in this case's emitted body") rather
#: than a scored inert case. They are must-be-caught legs, and calling them
#: discriminators would demand a null arm that cannot exist -- which the first
#: laptop run reported as NO DISCRIMINATOR on both.
CHAIN_ONLY_MUTATIONS: Tuple[str, ...] = ("drop_one_pole",
                                         "couple_the_wrong_chain")

#: Defects that can only bite where a folded axis is a live slot's PARTNER, which
#: is where the mirror ghost enters this sub-step at all.
PARTNER_ARM_MUTATIONS: Tuple[str, ...] = (
    "mirror_ghost_as_the_metallic_zero", "mirror_row_off_by_one",
    "flip_the_parity_sign", "ghost_lane_from_the_wall_flag",
    "parity_weights_the_raw_D")

#: The same, for the OWN-axis arm.
OWN_ARM_MUTATIONS: Tuple[str, ...] = ("folded_own_axis_wraps",)

#: DEFECTS THAT ARE THE IDENTITY WHEN THE PARITY IS +1. The weight is ``-phase``,
#: so on an ODD plane it is exactly ``+1.0`` and any defect that only changes
#: whether or how the weight is applied is bitwise inert there. Both parities are
#: swept: a battery at even planes alone would score these CAUGHT and learn
#: nothing about the SIGN, one at odd planes alone would score them NULL and learn
#: nothing at all.
PARITY_VISIBLE_MUTATIONS: Tuple[str, ...] = (
    "drop_the_parity_weight", "ghost_lane_from_the_wall_flag",
    "parity_weights_the_raw_D")

#: Wall-mask defects that are DISCRIMINATORS: caught where the grid declares a
#: metallic wall that a live row component masks against, NULL where it does not.
WALL_DISCRIMINATORS: Tuple[str, ...] = ("drop_the_wall_mask",
                                        "drop_the_wall_flags")

#: Host defects: the kernel never chooses these and cannot choose them wrong, but
#: its caller can. Armed through the launcher's keyword-only gate doors.
HOST_MUTATIONS: Tuple[str, ...] = ("hand_the_fold_the_metallic_code",
                                   "flip_the_launched_ghost_weights",
                                   "reverse_folded_axis_coefficients",
                                   "swap_sub_lattice", "drop_the_wall_flags",
                                   "reverse_the_launched_pole_plan")


def apply_host_mutation(name: Optional[str], layer, grid, tables, codes, walls,
                        weights):
    """One host defect, on the arguments a launch binds. Returns the four."""
    if name is None or name == "reverse_the_launched_pole_plan":
        return tables, codes, walls, weights
    if name == "hand_the_fold_the_metallic_code":
        codes = tuple(coverage.BC_CODES["metallic"]
                      if code == family.BC_MIRROR_CODE else code
                      for code in codes)
        return tables, codes, walls, weights
    if name == "flip_the_launched_ghost_weights":
        return tables, codes, walls, tuple(-w for w in weights)
    if name == "reverse_folded_axis_coefficients":
        # The retired refusal, armed IN BOUNDS: reversing the vector rather than
        # substituting a longer or shorter one, so the leg measures a WRONG ANSWER
        # and not a memory fault.
        mutated = dict(tables)
        xp = grid.xp
        for axis, letter in enumerate("xyz"):
            if not grid.is_mirrored(axis):
                continue
            for stem in ("kps", "kms"):
                key = f"{stem}_{letter}"
                mutated[key] = xp.ascontiguousarray(mutated[key][::-1])
        return mutated, codes, walls, weights
    if name == "swap_sub_lattice":
        return flat.integer_tables(layer), codes, walls, weights
    if name == "drop_the_wall_flags":
        return tables, codes, (0, 0, 0), weights
    raise ValueError(f"unknown host mutation {name!r}")


def launched_pole_plan(fields, host_mutation: Optional[str]):
    """The plan the launch binds -- or a deliberately reversed one.

    THE ORDERING DEFECT THAT ACTUALLY THREATENS THIS KERNEL, armed on the HOST
    side rather than in the source: the launcher resolves
    ``fields.polarizations`` filtered by ``drives`` on every call, and a launcher
    that resolved it in the wrong order would compute ``((D - P1) - P0)``.
    """
    plan = family.resolve_pole_plan(fields)
    if host_mutation == "reverse_the_launched_pole_plan":
        return {name: list(reversed(chain)) for name, chain in plan.items()}
    return plan


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_key(spec, mask_spec, arity, courant, value_class, guard, steps) -> str:
    return "|".join([spec["label"], mask_spec["label"],
                     "a" + "".join(str(n) for n in arity), f"C{courant}",
                     value_class, guard, f"n{steps}"])


def one_case(backend, xp, spec: Dict[str, Any], mask_spec: Dict[str, Any],
             arity: Sequence[int], courant: float, value_class: str,
             guard_label: str, guard: Sequence[str], steps: int,
             source_mutation: Optional[str] = None,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping.update_E`` ITSELF on real ``Grid``/``Fields``/``PML``
    objects with the rows installed through the PUBLIC installer and real
    ``PolarizationState`` objects registered, so the install-time validation, the
    zero-row drop, the stored-E switch and the ``drives`` filter are all in force.
    There is no second transcription on the oracle leg to drift.
    """
    started = time.time()
    key = case_key(spec, mask_spec, arity, courant, value_class, guard_label,
                   steps)
    # A STABLE PER-CASE SEED, from a DIGEST rather than from ``hash()``: Python
    # salts ``hash()`` of a str with PYTHONHASHSEED, so a run seeded from it draws
    # a different fixture every process and a failing case cannot be replayed.
    rng = np.random.default_rng(
        (SEED + int(hashlib.sha256(key.encode("ascii")).hexdigest()[:8], 16))
        % (2 ** 32))
    mask = tuple(mask_spec["mask"])
    arity = tuple(int(n) for n in arity)

    fields, layer, grid = build(xp, spec, courant)
    rows = flat.install_rows(fields, grid, mask, ROW_FORM, rng)
    register_poles(fields, grid, arity, rng)
    host = flat.seed_state(fields, grid, value_class, rng)
    host.update(seed_pole_state(fields, grid, value_class, rng))
    # THE PLAN THE LAUNCH BINDS, resolved ONCE here and handed to the backend, so
    # the evaluator's ``P_<c>_<k>`` arrays and the CuPy launcher's argument order
    # are the same permutation -- including when ``reverse_the_launched_pole_plan``
    # deliberately makes it the wrong one.
    launch_plan = launched_pole_plan(fields, host_mutation)
    poles = pole_plan_arrays(launch_plan)

    roles = fold_roles(grid, mask)
    codes = family.dispersive_offdiag_boundary_codes(grid)
    walls = coverage.offdiag_wall_mask_flags(grid)
    weights = family.mirror_ghost_weights(grid)

    case: Dict[str, Any] = {
        "key": key,
        "label": spec["label"], "fold_axes": spec["axes"],
        "mirror_phase": spec["phase"], "boundaries": list(spec["boundaries"]),
        "row_mask_label": mask_spec["label"], "row_mask": list(mask),
        "installed_row_mask": list(coverage.offdiag_row_mask(fields)),
        "arity": list(arity),
        "resolved_arity": list(family.pole_counts_of(
            family.resolve_pole_plan(fields))),
        "courant": courant, "value_class": value_class, "guard": guard_label,
        "steps": steps, "backend": backend.name, "backend_kind": backend.kind,
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
    case["chain_is_live"] = max(arity) > 0
    case["has_a_multi_pole_chain"] = max(arity) >= 2
    # THE MIRROR GHOST ENTERS THROUGH ONE LEG ONLY -- the PARTNER-axis down shift
    # (stepping.py:1243-1245) -- so a defect in how the parity meets the CHAIN can
    # only bite where the folded axis is a live slot's partner AND THAT PARTNER
    # COMPONENT ITSELF CARRIES POLES.
    #
    # MEASURED, NOT ARGUED. The 2026-08-20 keep leg scored
    # ``parity_weights_the_raw_D`` DISCRIMINATOR VIOLATED on exactly two cases:
    # arity (2, 0, 3) folded on Y, where the folded partner is Ey and Ey's chain is
    # empty, so the rewrite had nothing to rewrite there. Expecting a catch from
    # ``chain_is_live`` alone was a statement about the fixture.
    case["folded_partner_chain_is_live"] = any(
        arity[axis] > 0 for axis in roles["folded_partner_axes"])

    # THE FIVE PREDICATES' ANSWERS, recorded rather than acted on. A record that
    # did not carry all five could not show that the families PARTITION this
    # configuration rather than sharing it.
    from meep_gpu.cuda_kernels import dispersive_kernels  # noqa: PLC0415

    verdicts = {
        "this": family.covers_real_pml_dispersive_offdiag_constitutive(
            fields, layer, grid),
        "cuda_constitutive": coverage.covers_real_pml_constitutive(
            fields, layer, grid, side="E"),
        "cuda_offdiag": coverage.covers_real_pml_offdiag_constitutive(
            fields, layer, grid),
        "cuda_folded_offdiag": folded.covers_folded_offdiag_composition(
            fields, layer, grid),
        "cuda_dispersive": (
            dispersive_kernels.covers_real_pml_dispersive_constitutive(
                fields, layer, grid)),
    }
    case["predicates"] = {name: {"covered": bool(value[0]), "reason": value[1]}
                          for name, value in verdicts.items()}
    case["predicates_disjoint"] = sum(
        1 for value in verdicts.values() if value[0]) <= 1
    case["this_family_covers"] = bool(verdicts["this"][0])

    absorbs = folded_axis_absorbs(layer, grid)
    case["folded_axis_absorbs"] = absorbs
    ghost = mirror_ghost_is_live(fields, layer, grid)
    case["mirror_ghost_is_live"] = ghost
    coupling = coupling_is_live(fields, layer)
    case["coupling"] = coupling
    chain = pole_chain_bites(fields, layer)
    case["pole_chain"] = chain
    case["pole_chain_bites"] = chain["moved"] > 0

    if not absorbs["meets_floor"]:
        case["skipped"] = ("the folded axis's half-integer coefficient profile is "
                           "the identity everywhere; this case cannot distinguish "
                           "a coefficient index error on the axis the fold moved")
    elif not ghost["meets_floor"]:
        case["skipped"] = (f"stored row {ghost['source_row']} of a partner "
                           f"D - sum P volume is identically zero on a folded "
                           f"axis, so the mirror ghost equals the metallic zero "
                           f"ghost and an exact result would be a fact about the "
                           f"fixture")
    elif max(arity) > 0 and not case["pole_chain_bites"]:
        case["skipped"] = ("D - sum P is bitwise equal to D everywhere at a "
                           "nonzero arity: the chain is inert and every ordering "
                           "leg would be measuring the fixture")
    if case.get("skipped"):
        case["seconds"] = round(time.time() - started, 3)
        return case

    frozen = snapshot(fields)

    # --- leg 1: the oracle -------------------------------------------------
    for _ in range(steps):
        stepping.update_E(fields, layer)
        advance_sources(fields)
    oracle = {name: to_host(getattr(fields, name)).copy() for name in OUTPUTS}

    moved = sum(int(np.count_nonzero(
        oracle[name].ravel().view(np.uint32)
        != np.ascontiguousarray(frozen[name], dtype=np.float32).ravel().view(
            np.uint32)))
        for name in OUTPUTS)
    case["oracle_moved_words"] = moved
    case["oracle_total_words"] = sum(int(oracle[name].size) for name in OUTPUTS)
    case["oracle_moved"] = moved > 0

    # --- leg 2: the kernel, from the SAME frozen state ---------------------
    restore(fields, frozen)
    backend.set_guard(guard)
    tables = flat.half_integer_tables(layer)
    tables, launch_codes, launch_walls, launch_weights = apply_host_mutation(
        host_mutation, layer, grid, tables, codes, walls, weights)
    launch_error = None
    cost: Dict[str, Any] = {}
    try:
        for _ in range(steps):
            cost = backend.launch(fields, layer, rows, poles, mask, arity,
                                  tables, launch_codes, launch_walls,
                                  launch_weights, plan=launch_plan)
            advance_sources(fields)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, recorded
        launch_error = f"{type(exc).__name__}: {exc}"[:600]
    case["launch_error"] = launch_error
    case["cost"] = cost

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
    of the axes on which THE COMPONENT'S Yee shift is 0, so ``wm_y`` reaches Ex
    and Ez and never reaches Ey.
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
    if not case.get("coupling", {}).get("meets_floor"):
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
    arities = ARITIES if product == "full" else (
        (0, 0, 0), CORPUS_ARITY, (2, 0, 3))
    plan: List[Dict[str, Any]] = []
    for spec in specs:
        for mask_spec in ROW_MASK_SPECS:
            for arity in arities:
                for courant in courants:
                    for value_class in VALUE_CLASSES:
                        plan.append({"spec": spec, "mask_spec": mask_spec,
                                     "arity": arity, "courant": courant,
                                     "value_class": value_class,
                                     "steps": steps})
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
    chained = [c for c in primary if c["chain_is_live"]]
    reduction = [c for c in primary if not c["chain_is_live"]]
    multi_pole = [c for c in primary if c["has_a_multi_pole_chain"]]
    corpus = [c for c in primary
              if c["row_mask_label"] == "corpus_15row"
              and tuple(c["arity"]) == CORPUS_ARITY and c["fold_axes"]]

    by_arity: Dict[str, Dict[str, int]] = {}
    for case in primary:
        bucket = by_arity.setdefault("".join(str(n) for n in case["arity"]),
                                     {"ran": 0, "identical": 0})
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

    overlaps = [c["key"] for c in scored if not c["predicates_disjoint"]]
    uncovered = [c["key"] for c in scored if not c["this_family_covers"]]
    return {
        "per_guard": per_guard,
        "by_arity": by_arity,
        "by_simultaneous_planes": by_planes,
        "scored": len(scored),
        "refused": len(cases) - len(scored),
        "folded_ran": len(fold),
        "folded_identical": sum(1 for c in fold if c["bit_identical"]),
        "unfolded_ran": len(unfolded),
        "unfolded_identical": sum(1 for c in unfolded if c["bit_identical"]),
        "live_chain_ran": len(chained),
        "live_chain_identical": sum(1 for c in chained if c["bit_identical"]),
        "reduction_ran": len(reduction),
        "reduction_identical": sum(1 for c in reduction if c["bit_identical"]),
        "multi_pole_ran": len(multi_pole),
        "multi_pole_identical": sum(1 for c in multi_pole if c["bit_identical"]),
        "corpus_shape_ran": len(corpus),
        "corpus_shape_identical": sum(1 for c in corpus if c["bit_identical"]),
        "differing_words_total": sum(c["differing_words"] for c in primary),
        "cases_that_diverged": [c["key"] for c in primary
                                if not c["bit_identical"]],
        "subnormal_band_operands": census,
        "subnormal_band_is_non_vacuous": census["subnormals"] > 0,
        "every_case_moved_the_oracle": all(c["oracle_moved"] for c in scored),
        "every_case_had_live_coupling": all(
            c["coupling"]["meets_floor"] for c in scored),
        "every_live_chain_case_bit_the_chain": all(
            c["pole_chain_bites"] for c in scored if c["chain_is_live"]),
        "min_coupling_max_abs": (min(c["coupling"]["max_abs"] for c in scored)
                                 if scored else None),
        "predicates_disjoint_everywhere": not overlaps,
        "predicate_overlaps": overlaps,
        "cases_this_family_does_not_cover": uncovered,
        "extra_full_volume_passes_total": sum(
            int(c.get("cost", {}).get("extra_full_volume_passes", 0))
            for c in scored),
        "scratch_volumes_max": max(
            [int(c.get("cost", {}).get("scratch_volumes", 0)) for c in scored]
            or [0]),
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
                            entry["arity"], entry["courant"],
                            entry["value_class"], guard_label, guard,
                            entry["steps"])
            valid, why = case_is_valid(case)
            case["case_is_valid"] = valid
            case["refused_because"] = why
            cases.append(case)
            log(f"[{leg_name}] {index}/{total} {case['key']} "
                f"shape={case['shape']} "
                f"roles=P{int(case['fold_reaches_the_partner_arm'])}"
                f"O{int(case['fold_reaches_the_own_arm'])} "
                f"chain={case['pole_chain']['moved']} "
                f"got={'IDENTICAL' if case.get('bit_identical') else 'DIVERGED'} "
                f"diff={case.get('differing_words')}"
                + ("" if valid else f" REFUSED: {why[:60]}")
                + f" ({case['seconds']} s)")
            results[leg_name] = {"cases": cases, "summary": summarize(cases),
                                 "seconds": round(time.time() - started, 1)}
            save(results, out_path)
    return results[leg_name]


# ---------------------------------------------------------------------------
# THE COMPOSITION LEG: the alternative, measured
# ---------------------------------------------------------------------------

def run_composition(backend, xp, results: Dict[str, Any], out_path: str,
                    product: str, steps: int) -> Dict[str, Any]:
    """Materialize ``D - sum P``, then launch the SHIPPED folded off-diagonal kernel.

    The question this answers is the census note's: "a new kernel OR a measured
    fusion of two shipped ones". It runs the only composition that exists, on the
    same specs and the same frozen states, and reports BOTH its bit verdict and
    its cost. Nothing here changes what the release verdict is about -- the
    kernel -- but the numbers are what a future planner decides on.
    """
    if backend.kind != "kernel" or not backend.certifies:
        # The composition launches the SHIPPED CuPy kernel, which needs a device.
        # Recorded rather than raised, so a laptop harness run still exercises
        # every other leg and the artifact says WHY this one is empty.
        results["composition"] = {
            "ran": False,
            "why_not": ("the composition launches the shipped CuPy "
                        "update_E_pml_real_folded_offdiag, which needs a "
                        "device; the numpy backend compiles nothing"),
            "cases": [], "summary": {}}
        save(results, out_path)
        log("[composition] SKIPPED: needs a device (the shipped kernel is CuPy)")
        return results["composition"]
    composition = CompositionBackend(backend)
    cases: List[Dict[str, Any]] = []
    plan = [entry for entry in case_product(
        "reduced" if product == "full" else product, steps)
        if entry["courant"] == INEXACT_COURANT
        and entry["value_class"] == "uniform"]
    started = time.time()
    for index, entry in enumerate(plan, 1):
        case = one_case(composition, xp, entry["spec"], entry["mask_spec"],
                        entry["arity"], entry["courant"], entry["value_class"],
                        "fmad_false", ("--fmad=false",), entry["steps"])
        valid, why = case_is_valid(case)
        case["case_is_valid"] = valid
        case["refused_because"] = why
        cases.append(case)
        log(f"[composition] {index}/{len(plan)} {case['key']} "
            f"got={'IDENTICAL' if case.get('bit_identical') else 'DIVERGED'} "
            f"passes={case.get('cost', {}).get('extra_full_volume_passes')} "
            f"({case['seconds']} s)")
        results["composition"] = {
            "what_it_is": (
                "materialize the three D - sum P volumes with the ARRAY PATH "
                "(Fields.displacement_minus_polarization_volumes), then launch "
                "the SHIPPED update_E_pml_real_folded_offdiag with those "
                "buffers bound in place of D"),
            "why_it_is_not_a_fusion_of_two_shipped_kernels": (
                "neither shipped kernel exposes D - sum P as an output: "
                "update_E_pml_real_dispersive writes (D - sum P) * inv_eps "
                "into f_w and accumulates E, so recovering the operand needs a "
                "float32 DIVISION and the accumulation would have to be undone; "
                "update_E_pml_real_folded_offdiag has no pole parameters"),
            "cases": cases, "summary": summarize(cases),
            "seconds": round(time.time() - started, 1)}
        save(results, out_path)
    return results["composition"]


# ---------------------------------------------------------------------------
# The mutation battery
# ---------------------------------------------------------------------------
#
# SCOPED TO CASES WHOSE UNMUTATED BASELINE WAS BIT-IDENTICAL, read off the sweep
# that just ran rather than tabulated, so no leg can be armed on an arm the device
# did not actually certify in this same process.

def mutation_plan(sweep_cases: Sequence[Dict[str, Any]], product: str
                  ) -> List[Dict[str, Any]]:
    """The (spec, mask, arity) triples a defect may be planted on.

    THE REDUCED SELECTION IS GREEDY, NOT A HEAD SLICE. Slicing the front of a case
    product landed the folded-curl gate's whole battery on its two UNFOLDED
    controls, so its one fold-specific mutation was skipped on every leg and
    scored 0/0 UNCAUGHT -- a harness defect that reads exactly like a kernel
    defect. This selection covers, in order: both arms of every discriminator
    (partner role, own role, neither role, wall / no wall, chain / no chain,
    multi-pole / single pole), every distinct row mask, every distinct folded
    axis, and both plane parities.
    """
    seen, plan = set(), []
    for case in sweep_cases:
        if not case.get("case_is_valid") or case["guard"] != "fmad_false":
            continue
        if not case["bit_identical"]:
            continue
        if case["courant"] != INEXACT_COURANT or case["value_class"] != "uniform":
            continue
        key = (case["label"], case["row_mask_label"], tuple(case["arity"]))
        if key in seen:
            continue
        seen.add(key)
        spec = next(s for s in FOLD_SPECS if s["label"] == case["label"])
        mask_spec = next(m for m in ROW_MASK_SPECS
                         if m["label"] == case["row_mask_label"])
        plan.append({"spec": spec, "mask_spec": mask_spec,
                     "arity": tuple(case["arity"]),
                     "partner_arm": bool(case["fold_reaches_the_partner_arm"]),
                     "own_arm": bool(case["fold_reaches_the_own_arm"]),
                     "has_wall": bool(case["wall_mask_bites"]),
                     "chain": bool(case["chain_is_live"]),
                     "multi_pole": bool(case["has_a_multi_pole_chain"]),
                     "folded_partner_chain": bool(
                         case["folded_partner_chain_is_live"]),
                     "phase": case["mirror_phase"],
                     "folded_axes": tuple(case["roles"]["folded_axes"])})
    if product == "full":
        return plan
    # THE COVERAGE KEYS ARE CROSSES, NOT INDEPENDENT AXES, and that is a
    # measurement rather than a preference. The first laptop run covered each axis
    # separately, and because arity (0, 0, 0) is first in the list every FOLDED
    # entry was already satisfied by it -- so no case in the plan was folded AND
    # carried a chain, and ``parity_weights_the_raw_D`` (which needs both) came
    # back with an empty catch arm. The role and the chain are covered TOGETHER.
    chosen: List[Dict[str, Any]] = []
    roles, shapes = set(), set()
    for entry in plan:
        role = (entry["partner_arm"], entry["own_arm"], entry["has_wall"],
                entry["chain"], entry["multi_pole"],
                entry["folded_partner_chain"])
        shape = (entry["mask_spec"]["label"], entry["folded_axes"],
                 entry["phase"], entry["chain"])
        if role in roles and shape in shapes:
            continue
        roles.add(role)
        shapes.add(shape)
        chosen.append(entry)
    return chosen


def expected_caught(name: str, case: Dict[str, Any]) -> bool:
    """Must THIS case diverge under THIS defect?

    Decided per case rather than per leg, because most of this family's defects
    are only defects where the chain is live, or where the fold plays a particular
    role -- and a leg scored "caught everywhere" would be false of the physics
    rather than of the kernel.
    """
    if name in NULL_MUTATIONS:
        return False
    if name in ARITY_DISCRIMINATORS:
        return bool(case["has_a_multi_pole_chain"])
    if name == "reverse_the_launched_pole_plan":
        return bool(case["has_a_multi_pole_chain"])
    if name in WALL_DISCRIMINATORS:
        return bool(case["wall_mask_bites"])
    if name == "parity_weights_the_raw_D":
        # The parity meets the chain on the PARTNER-axis down leg alone, so this
        # needs the folded partner component to carry poles -- not merely SOME
        # component somewhere. See ``folded_partner_chain_is_live``.
        return bool(case["folded_partner_chain_is_live"]
                    and case["mirror_phase"] == 1)
    if name in CHAIN_ONLY_MUTATIONS:
        return True
    if name in CHAIN_MUTATIONS:
        return bool(case["chain_is_live"])
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
BOTH_SLOTS_MASKS = {(1, 1, 1, 1, 1, 1)}

#: Every leg that edits a POLE CHAIN needs an arity that HAS one; the multi-pole
#: legs need two. Armed only where the site exists, so an unarmable leg is
#: reported as such rather than as an uncaught defect.
DISCRIMINATOR_NAMES = (WALL_DISCRIMINATORS + PARTNER_ARM_MUTATIONS
                       + OWN_ARM_MUTATIONS + PARITY_VISIBLE_MUTATIONS
                       + ARITY_DISCRIMINATORS + CHAIN_MUTATIONS
                       + ("hand_the_fold_the_metallic_code",
                          "flip_the_launched_ghost_weights",
                          "reverse_folded_axis_coefficients",
                          "reverse_the_launched_pole_plan"))


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
                    tuple(entry["mask_spec"]["mask"]), tuple(entry["arity"]))
                mutated_text, count = transform(pristine)
                sites += count
                if count and (classification is None
                              or classification["evaluator_sees_it"]):
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
                            entry["arity"], INEXACT_COURANT, "uniform",
                            "fmad_false", ("--fmad=false",), steps,
                            source_mutation=name if kind == "source" else None,
                            host_mutation=name if kind == "host" else None)
            valid, why = case_is_valid(case)
            case["case_is_valid"] = valid
            case["refused_because"] = why
            # A SOURCE LEG WHOSE TRANSFORM MATCHED NOTHING ON THIS CASE'S BODY IS
            # NOT SCORED. The arity axis makes the sites case-dependent -- a
            # chain-editing defect has no site at arity (0,0,0) -- and scoring a
            # rewrite that never happened as UNCAUGHT is the dead-branch trap.
            if transform is not None:
                _text, count = transform(backend.pristine_source(
                    tuple(entry["mask_spec"]["mask"]), tuple(entry["arity"])))
                case["mutation_sites_here"] = count
                if count == 0:
                    case["case_is_valid"] = False
                    case["refused_because"] = (
                        "the mutation matched nothing in this case's emitted "
                        "body; scoring it would report a defect never planted")
            cases.append(case)

        compiles = backend.compile_log()[before_compiles:]
        mutated_digests = set()
        if transform is not None:
            for entry in entries:
                text, count = transform(backend.pristine_source(
                    tuple(entry["mask_spec"]["mask"]), tuple(entry["arity"])))
                if count:
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
        discriminator = name in DISCRIMINATOR_NAMES
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
                  "arity": list(e["arity"]),
                  "partner_arm": e["partner_arm"], "own_arm": e["own_arm"],
                  "has_wall": e["has_wall"], "chain": e["chain"],
                  "multi_pole": e["multi_pole"], "phase": e["phase"],
                  "folded_partner_chain": e["folded_partner_chain"]}
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
    """The release verdict, as a FUNCTION of the legs, so it can be falsified."""
    single = results.get("sweep", {}).get("summary", {})
    multi = results.get("multistep", {}).get("summary", {})
    mutations = results.get("mutations", {})
    composition = results.get("composition", {}).get("summary", {})
    control = single.get("per_guard", {}).get("default_no_options", {})
    verdict = {
        "live_chain_single_launch":
            f"{single.get('live_chain_identical', 0)}"
            f"/{single.get('live_chain_ran', 0)}",
        "live_chain_multi_step":
            f"{multi.get('live_chain_identical', 0)}"
            f"/{multi.get('live_chain_ran', 0)}",
        "corpus_shape":
            f"{single.get('corpus_shape_identical', 0)}"
            f"/{single.get('corpus_shape_ran', 0)}",
        "folded":
            f"{single.get('folded_identical', 0)}/{single.get('folded_ran', 0)}",
        "unfolded":
            f"{single.get('unfolded_identical', 0)}"
            f"/{single.get('unfolded_ran', 0)}",
        "zero_arity_reduction":
            f"{single.get('reduction_identical', 0)}"
            f"/{single.get('reduction_ran', 0)}",
        "multi_pole":
            f"{single.get('multi_pole_identical', 0)}"
            f"/{single.get('multi_pole_ran', 0)}",
        "by_arity": single.get("by_arity"),
        "by_simultaneous_planes": single.get("by_simultaneous_planes"),
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
        "every_live_chain_case_bit_the_chain": single.get(
            "every_live_chain_case_bit_the_chain"),
        "predicates_disjoint_everywhere": single.get(
            "predicates_disjoint_everywhere"),
        "cases_this_family_does_not_cover": single.get(
            "cases_this_family_does_not_cover"),
        "mutation_legs_as_required": mutations.get("legs_as_required"),
        "all_mutation_legs_as_required": mutations.get("all_legs_as_required"),
        "mutation_legs_not_measurable":
            mutations.get("legs_not_measurable_on_this_backend"),
        "mutation_legs_not_armable": mutations.get("legs_not_armable"),
        # THE ALTERNATIVE, reported beside the verdict and NOT part of it: what
        # the composition costs is a planner's decision, not a release condition.
        "composition_identical":
            (f"{composition.get('scored', 0) - len(composition.get('cases_that_diverged', []))}"
             f"/{composition.get('scored', 0)}" if composition else None),
        "composition_extra_full_volume_passes":
            composition.get("extra_full_volume_passes_total"),
        "composition_scratch_volumes_max":
            composition.get("scratch_volumes_max"),
        "kernel_extra_full_volume_passes":
            single.get("extra_full_volume_passes_total"),
    }
    verdict["passed"] = bool(
        results.get("certifies")
        and single.get("live_chain_ran", 0) > 0
        and single["live_chain_identical"] == single["live_chain_ran"]
        and single.get("reduction_ran", 0) > 0
        and single["reduction_identical"] == single["reduction_ran"]
        and single.get("folded_ran", 0) > 0
        and single["folded_identical"] == single["folded_ran"]
        and single.get("unfolded_ran", 0) > 0
        and single["unfolded_identical"] == single["unfolded_ran"]
        and single.get("multi_pole_ran", 0) > 0
        and single["multi_pole_identical"] == single["multi_pole_ran"]
        and single.get("corpus_shape_ran", 0) > 0
        and single["corpus_shape_identical"] == single["corpus_shape_ran"]
        and multi.get("live_chain_ran", 0) > 0
        and multi["live_chain_identical"] == multi["live_chain_ran"]
        and single.get("subnormal_band_is_non_vacuous")
        and single.get("every_case_had_live_coupling")
        and single.get("every_case_moved_the_oracle")
        and single.get("every_live_chain_case_bit_the_chain")
        and single.get("predicates_disjoint_everywhere")
        and not single.get("cases_this_family_does_not_cover")
        and mutations.get("all_legs_as_required", False)
        and mutations.get("legs_not_measurable_on_this_backend", 1) == 0
        and mutations.get("legs_not_armable", 1) == 0)
    return verdict


def run_falsification(backend, xp, results: Dict[str, Any], out_path: str,
                      steps: int) -> Dict[str, Any]:
    """RUN THE VERDICT AGAINST A PLANTED DEFECT AND REQUIRE IT TO REFUSE.

    The defect chosen is ``couple_the_raw_D`` -- the MIDDLE REFUSAL -- because if
    the verdict cannot refuse THAT then this family has no reason to exist over
    the shipped folded off-diagonal kernel.
    """
    started = time.time()
    spec = next(s for s in FOLD_SPECS if s["label"] == "fold_Y_periodic_corpus")
    mask_spec = next(m for m in ROW_MASK_SPECS if m["label"] == "corpus_15row")
    backend.set_source_mutation(SOURCE_MUTATIONS["couple_the_raw_D"])
    case = one_case(backend, xp, spec, mask_spec, CORPUS_ARITY, INEXACT_COURANT,
                    "uniform", "fmad_false", ("--fmad=false",), steps,
                    source_mutation="couple_the_raw_D")
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
        "defect": "couple_the_raw_D",
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

LEG_NAMES = ("sweep", "multistep", "composition", "mutations", "falsify")


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
        "gate": "cuda_dispersive_offdiag",
        "kernel": family.KERNEL_NAME,
        "question": ("does the DISPERSIVE off-diagonal update_E kernel -- MEEP's "
                     "tensor row product whose four gathered field reads per term "
                     "are the PARTNER component's D - sum P chain, over a possibly "
                     "mirror-folded stored extent -- reproduce stepping.update_E "
                     "word for word, and still reduce to the two certified "
                     "off-diagonal families where no pole drives anything?"),
        "specified_by": ("the four refusals on examples/absorbed_power_density.py "
                         "in results/cuda_predicate_coverage_2026-08-20_final"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend_name,
        "certifies": backend_name == "cupy",
        "product": product,
        "legs_requested": list(legs),
        "seed": SEED,
        "multi_step_budget": args.multi_step_budget,
        "arities_swept": [list(a) for a in ARITIES],
        "pole_count_cap": family.POLE_COUNT_CAP,
        "emitter_corpus_digest": family.corpus_digest(),
        "folded_sibling_corpus_digest": folded.corpus_digest(),
        "gate_sha256": probe.source_digest(os.path.abspath(__file__)),
    }
    from meep_gpu.cuda_kernels import offdiag_emitter  # noqa: E402,PLC0415
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
    if "composition" in legs:
        run_composition(backend, xp, results, out_path, product, args.steps)
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
        f"chain={verdict['live_chain_single_launch']} "
        f"multi={verdict['live_chain_multi_step']} "
        f"corpus={verdict['corpus_shape']} "
        f"reduction={verdict['zero_arity_reduction']} "
        f"folded={verdict['folded']} unfolded={verdict['unfolded']} "
        f"mutations={verdict['mutation_legs_as_required']} "
        f"falsified={verdict['falsification_as_required']} "
        f"composition={verdict['composition_identical']} "
        f"released={results['release']['released']}")
    if not results["certifies"]:
        single = results.get("sweep", {}).get("summary", {})
        mutations = results.get("mutations", {})
        harness_ok = (
            single.get("live_chain_ran", 0) > 0
            and single.get("live_chain_identical") == single.get("live_chain_ran")
            and single.get("folded_identical") == single.get("folded_ran")
            and single.get("unfolded_identical") == single.get("unfolded_ran")
            and mutations.get("all_legs_as_required", True)
            and (falsify is None or falsify["as_required"]))
        return 0 if harness_ok else 1
    return 0 if results["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
