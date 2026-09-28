"""Sub-step byte-identity gate for the NO-ABSORBER CUDA curl family.

THE QUESTION
============
``meep_gpu/cuda_kernels/no_pml_curl.py`` carries three kernels for the curl
sub-steps of a run with NO ACTIVE ABSORBER. None of them has ever run on a
device. This gate asks, per sub-step and per arm, whether each is byte-identical
to ``stepping.step_B`` / ``stepping.step_D`` from ONE frozen state -- at one
launch and at 60 -- across periodic, metallic and mixed walls, a mirror fold at
both terminations, 1-D / 2-D / 3-D shapes, both float32 subnormal policies, both
value classes, and both courants.

WHAT COLLAPSES, AND WHAT DOES NOT
=================================
``stepping._apply_curl`` (stepping.py:489-539) has four tails. With
``_pml_is_active(pml)`` False and ``fields.condfac_for(target)`` None the tail is
one line, ``target -= curl`` (:510): no ``fu`` auxiliary, no ``kms``/``sinv``
pair, no split-field recurrence. The CURL is unchanged -- term table, ghost rule,
grouping, wall masks -- so ``step_{B,D}_no_pml_real`` are the certified
``step_{B,D}_pml_real`` blocks verbatim with only the tail replaced.

THE THIRD KERNEL EXISTS BECAUSE OF A MEASUREMENT, NOT A READING
===============================================================
``stepping.step_B`` reads its operands through ``_read_component``
(stepping.py:2438-2454), whose E branch derives ``D * inv_eps`` whenever
``fields.stores_E`` is False -- which, without an absorber and without a
susceptibility, is the normal case: ``fields.Ex`` is None. Measured on a real
(6, 8, 10) Grid/Fields pair with an inhomogeneous inverse epsilon,
``stepping.step_B(fields, None)`` is bit-identical as uint32 words to the forward
curl taken OF ``D * inv_eps``. So a kernel binding Ex/Ey/Ez cannot serve that
configuration, and ``step_B_no_pml_real_derived`` -- which forms the product
at the point of use -- is the arm that can. On the 186-row corpus that arm is FOUR
of the five reachable ``step_B`` slots.

``stepping.step_D`` differences the B ARRAYS: ``Fields.get_H`` returns them
directly without an absorber (fields.py:1184-1186, mu = 1, nothing stored).
Measured bit-identical on the same fixture. That is why ``step_D`` has ONE arm and
why E storage is swept as a free variable on it -- the claim "step_D does not read
E" is a claim, and the sweep is what turns it into a measurement.

THE ARMS
========
* ``step_B`` / ``stored_E``   -- ``step_B_no_pml_real``          (binds Ex/Ey/Ez)
* ``step_B`` / ``derived_E``  -- ``step_B_no_pml_real_derived``  (binds D and inv_eps)
* ``step_D`` / ``magnetic``   -- ``step_D_no_pml_real``          (binds the B arrays)

and the absorber itself is swept two ways, because ``_pml_is_active`` is False for
both and the predicate admits both: ``pml=None`` (no object at all) and an
INERT LAYER -- a ``PML`` with every face at zero thickness, whose ``is_active`` is
False (pml.py:427-434) and which ``stepping`` therefore routes down the same tail.
A run that swept only one would leave "an inert layer steps like no layer" an
inference.

WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
================================================================
* ``oracle_moved`` -- the fraction of output words the array path changed from the
  frozen input. A case that moved nothing is SKIPPED, not passed.
* ``masked_planes_are_live`` -- every plane the kernel ZEROES must carry a
  nonzero UNMASKED curl. This gate's licensed arm has no mask delta (the codes are
  the shipped resolution), so the folded sibling's mask-delta floor is vacuous
  here and this is its replacement: without it ``drop_metallic_mask`` and the two
  fold-mask mutations could come back UNCAUGHT for a reason about the fixture.
* ``epsilon_profile`` (derived arm) -- ``max|inv_eps - 1| > 0``, a nonzero spread
  WITHIN each volume, and the three volumes pairwise distinct. Against a table of
  ones ``drop_derived_inverse_epsilon`` is invisible; against one shared volume
  ``alias_derived_inv_eps_to_z`` is invisible; against a constant volume
  ``derived_wrap_uses_own_inv_eps`` is invisible.
* ``sources_are_distinct`` -- the three source volumes differ pairwise, so a
  component-binding defect has somewhere to show.
* The mutation battery, INCLUDING legs that must come back UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything.
* ``verdict_flips_against_planted_defect`` -- the release verdict is recomputed
  against a results record with a defect planted in it, and the run fails if the
  verdict does not flip. A gate whose verdict cannot go red is not a gate.

THE FOUR DEAD-PRELUDE NULLS, WHICH ARE EVIDENCE RATHER THAN BOOKKEEPING
======================================================================
``no_pml_curl.py`` shares ``_REAL_PML_PRELUDE`` with the certified pair verbatim,
so its ``pml_apply`` helper -- the split-field recurrence -- is compiled into every
kernel here and CALLED BY NONE of them. The shared battery's four ``pml_apply``
mutations (``drop_fu_store``, ``read_fprev_after_store``, ``swap_dsig_dsigu``,
``reload_fu_from_memory``) therefore arm, at one site each, and MUST come back
UNCAUGHT. That they do is the measurement that this family's tail really is
``target -= curl`` and not the recurrence -- a claim otherwise resting on reading
the source.

WHY A SUB-STEP GATE AND NOT A WHOLE-STEP ONE
============================================
A whole-step gate cannot see a mask: the driver's boundary passes overwrite
exactly the planes a missing mask would corrupt, and the sibling track measured a
kernel carrying NEITHER fold mask as bytewise-identical after a complete step.
Every comparison here is ONE SUB-STEP from ONE frozen state.

RUNNING IT
==========
Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_no_pml_curl.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the
same floors and the same HOST mutations against a transcription of the device
tree. IT COMPILES NOTHING AND CERTIFIES NOTHING -- what it settles is whether the
harness can localize a divergence and whether it can fail::

    python -u gate_cuda_no_pml_curl.py --backend numpy --out /tmp/no_pml_curl.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: the NumPy backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields, IYEE_SHIFTS  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage, no_pml_curl  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260820

#: Consecutive launches in the multi-step leg, the budget both certified
#: hand-CUDA records are cut at. "Identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the operands moving. Held fixed, the curl reads
#: the same numbers every launch and a 60-launch leg becomes a slow single-launch
#: leg. The same exact float32 scale is applied on both paths.
_ADVANCE = np.float32(0.97)

BC_PERIODIC = 0
BC_METALLIC = 1
BC_MIRROR_PERIODIC = 2

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: Which E storage each sub-step is swept under. ``step_B``'s two values ARE its
#: two arms; ``step_D``'s are a free variable, and sweeping both is what turns
#: "step_D does not read E" from a reading into a measurement.
E_STORAGE: Tuple[str, ...] = ("stored", "derived")


def arm_of(sub_step: str, e_storage: str) -> str:
    if sub_step == "step_D":
        return "magnetic"
    return "stored_E" if e_storage == "stored" else "derived_E"


#: What each (sub_step, arm) writes and reads, spelled as the ARRAYS a launch
#: binds. ``step_D``'s sources are the B arrays and are written that way on
#: purpose: "Hx" is the name that made the first version of this family's
#: predicate refuse every run it exists to serve.
ARM_ARRAYS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("step_B", "stored_E"): {"targets": ("Bx", "By", "Bz"),
                             "sources": ("Ex", "Ey", "Ez"), "backward": False},
    ("step_B", "derived_E"): {"targets": ("Bx", "By", "Bz"),
                              "sources": ("Dx", "Dy", "Dz"), "backward": False},
    ("step_D", "magnetic"): {"targets": ("Dx", "Dy", "Dz"),
                             "sources": ("Bx", "By", "Bz"), "backward": True},
}

#: The six curl terms as the KERNELS spell them, transcribed from
#: ``no_pml_curl.py``'s device text. ``(target, first source, first axis, second
#: source, second axis)``, where a source is named by its E/H component and the
#: launcher resolves which array carries it. Written out rather than derived from
#: ``stepping`` so the NumPy leg transcribes the DEVICE source rather than
#: comparing the oracle with itself; :func:`check_terms_against_stepping` pins the
#: correspondence so the transcription cannot drift silently.
CURL_TERMS: Dict[str, Tuple[Tuple[str, str, int, str, int], ...]] = {
    "step_B": (("Bx", "Ez", 1, "Ey", 2),
               ("By", "Ex", 2, "Ez", 0),
               ("Bz", "Ey", 0, "Ex", 1)),
    "step_D": (("Dx", "Hz", 1, "Hy", 2),
               ("Dy", "Hx", 2, "Hz", 0),
               ("Dz", "Hy", 0, "Hx", 1)),
}

#: Which array carries each source name, per arm. On ``derived_E`` the pair is
#: ``(D array, inverse-epsilon volume)`` and the operand is their product.
_SOURCE_INDEX = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 0, "Hy": 1, "Hz": 2}

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: ``--fmad=false`` is carried as CORRECTNESS here, not tuning, and on paper MORE
#: so on the derived arm than on the stored one: there ``sf - f1`` is a difference
#: of two PRODUCTS, a textbook fused-multiply-add candidate, while the array path
#: rounds the multiply and the subtraction separately. THE READING WAS WRONG ABOUT
#: THIS DEVICE, and the record says so rather than implying evidence it does not
#: have: measured 2026-08-20 on an RTX A6000 under NVRTC 11.6, the unguarded build
#: was bit-identical to the guarded one on all 112 comparable cases at the inexact
#: courant, on every arm. ``guard_control.reading`` reports MEASURED DECORATIVE,
#: which is a statement about this compiler on this body and not a licence to drop
#: the flag.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: How the boundary code triple is chosen. Only the first is a triple a dispatch
#: could produce, and only the first is gated on.
#:
#: * ``resolved``          -- ``coverage.real_curl_boundary_codes``, the shipped
#:   resolution and the licensed arm.
#: * ``mirror_as_metallic`` -- a folded PERIODIC axis handed BC_METALLIC. The two
#:   codes share a ghost rule (both take the zero far face), so this changes ONLY
#:   the Yee-shift-1 top-plane mask, and its divergence must be CONFINED to the
#:   mask-delta planes. That confinement is what says the top-plane mask is what
#:   the licensed arm's identity rests on.
#: * ``all_periodic``      -- the blunt misdeclaration: no mask and a wrapping
#:   ghost on every axis. It must diverge wherever the grid carries a non-periodic
#:   code; its footprint is reported, not asserted, because it moves two things.
SUBSTITUTIONS: Tuple[str, ...] = ("resolved", "mirror_as_metallic", "all_periodic")
LICENSED_SUBSTITUTION = "resolved"

#: The grids. EVERY WALL KIND appears, the fold appears at BOTH terminations, and
#: 1-D / 2-D / 3-D shapes all appear -- the corpus's real no-absorber rows are
#: (1,1,400), (200,200,1), (25,25,1), (100,100,1), (5,5,1) and (1,1,1), so a
#: 3-D-only sweep would measure a shape family the corpus does not contain.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "3d_periodic", "cell": (8.0, 10.0, 12.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"), "axes": "", "phase": 1,
     "absorber": "none"},
    {"label": "3d_metallic", "cell": (8.0, 10.0, 12.0), "dimensions": 3,
     "boundaries": ("metallic", "metallic", "metallic"), "axes": "", "phase": 1,
     "absorber": "none"},
    {"label": "3d_mixed", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("metallic", "periodic", "metallic"), "axes": "", "phase": 1,
     "absorber": "none"},
    # THE INERT LAYER, on a mixed grid: a PML object present but every face at
    # zero thickness. ``_pml_is_active`` is False, so ``stepping`` takes the same
    # tail -- and the predicate admits it. Swept because "inert is the same as
    # absent" is otherwise an inference about pml.py rather than a measurement.
    {"label": "3d_mixed_inert_layer", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("periodic", "metallic", "metallic"), "axes": "", "phase": 1,
     "absorber": "inert_layer"},
    # THE FOLD, at both terminations, on the slowest and the fastest axis.
    {"label": "fold_Y_periodic", "cell": (8.0, 16.0, 12.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"), "axes": "Y", "phase": 1,
     "absorber": "none"},
    {"label": "fold_Y_metallic", "cell": (8.0, 16.0, 12.0), "dimensions": 3,
     "boundaries": ("periodic", "metallic", "periodic"), "axes": "Y", "phase": 1,
     "absorber": "none"},
    {"label": "fold_X_periodic_odd_plane", "cell": (16.0, 8.0, 12.0),
     "dimensions": 3, "boundaries": ("periodic", "periodic", "periodic"),
     "axes": "X", "phase": -1, "absorber": "none"},
    {"label": "fold_Z_metallic", "cell": (8.0, 12.0, 16.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "metallic"), "axes": "Z", "phase": -1,
     "absorber": "none"},
    {"label": "fold_XY_mixed", "cell": (16.0, 16.0, 10.0), "dimensions": 3,
     "boundaries": ("periodic", "metallic", "periodic"), "axes": "XY", "phase": 1,
     "absorber": "none"},
    # 2-D and 1-D. An invariant axis is always PERIODIC and n = 1, so its wrap
    # returns the reading cell and the difference is an exact zero -- which is
    # what MEEP's ``stride(d) = 0`` computes for a direction it does not have. A
    # gate that never ran one would not have measured that.
    {"label": "2d_metallic_walls", "cell": (12.0, 14.0, 0.0), "dimensions": 2,
     "boundaries": ("metallic", "metallic", "periodic"), "axes": "", "phase": 1,
     "absorber": "none"},
    {"label": "2d_periodic", "cell": (12.0, 14.0, 0.0), "dimensions": 2,
     "boundaries": ("periodic", "periodic", "periodic"), "axes": "", "phase": 1,
     "absorber": "none"},
    {"label": "2d_fold_X_periodic", "cell": (16.0, 14.0, 0.0), "dimensions": 2,
     "boundaries": ("periodic", "periodic", "periodic"), "axes": "X", "phase": 1,
     "absorber": "none"},
    {"label": "1d_metallic", "cell": (0.0, 0.0, 24.0), "dimensions": 1,
     "boundaries": ("periodic", "periodic", "metallic"), "axes": "", "phase": 1,
     "absorber": "none"},
    {"label": "1d_periodic_inert_layer", "cell": (0.0, 0.0, 24.0), "dimensions": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "axes": "", "phase": 1,
     "absorber": "inert_layer"},
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    Everything else the fixture exercises -- the fold, the stored extent, the
    dtype, the contiguity, the inverse-epsilon volumes -- is a real object either
    way.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float, e_storage: str, rng):
    """A frozen ``(fields, layer, grid)`` triple for one spec and E storage."""
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                dimensions=spec.get("dimensions", 3), xp=xp, courant=courant)
    fields = Fields(grid=grid)
    if e_storage == "stored":
        fields.enable_field_storage()
    else:
        # THREE DISTINCT, INHOMOGENEOUS inverse-epsilon volumes, drawn away from
        # 1.0. Against a table of ones the multiply is invisible; against one
        # shared volume the component-aliasing defect is invisible; against a
        # constant volume the wrap's inverse-epsilon index is invisible. Each of
        # those is a mutation in this gate's battery, and ``epsilon_profile`` is
        # the floor that refuses a case where they could not bite.
        inverse = {}
        epsilon = {}
        for component in ("Ex", "Ey", "Ez"):
            values = rng.uniform(0.2, 0.9, size=grid.shape).astype(np.float32)
            inverse[component] = xp.asarray(np.ascontiguousarray(values))
            epsilon[component] = xp.asarray(
                np.ascontiguousarray((1.0 / values).astype(np.float32)))
        fields.set_epsilon_volumes(epsilon, inverse)
    layer = (PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
             if spec["absorber"] == "inert_layer" else None)
    return fields, layer, grid


def state_names(sub_step: str, arm: str) -> Tuple[str, ...]:
    spec = ARM_ARRAYS[(sub_step, arm)]
    # dict.fromkeys: step_D's sources ARE its own family's B arrays, and on the
    # derived B arm the sources are D -- neither list may name an array twice or
    # `restore` would write it under two keys.
    return tuple(dict.fromkeys(tuple(spec["targets"]) + tuple(spec["sources"])))


def seed_state(fields, grid, sub_step: str, arm: str, value_class: str,
               rng) -> Dict[str, np.ndarray]:
    xp = grid.xp
    names = state_names(sub_step, arm)
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
                for name in names}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(names, tuple(grid.shape), rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def snapshot(fields, sub_step: str, arm: str) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy()
            for name in state_names(sub_step, arm)}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, sub_step: str, arm: str) -> None:
    """Move the curl operands between launches, identically on both paths."""
    for name in ARM_ARRAYS[(sub_step, arm)]["sources"]:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


def outputs(sub_step: str, arm: str) -> Tuple[str, ...]:
    """The three arrays this sub-step writes. THERE IS NO AUXILIARY HERE.

    The certified pair compares ``fu`` on every case because a tree that gets the
    field right and the auxiliary wrong is correct for exactly one launch. This
    family writes no auxiliary at all -- that absence IS the family -- and the
    predicate refuses a run where one is allocated, so the comparison has nothing
    to add and its absence is not an omission.
    """
    return tuple(ARM_ARRAYS[(sub_step, arm)]["targets"])


# ---------------------------------------------------------------------------
# The boundary codes, and the substitution that is the experimental variable
# ---------------------------------------------------------------------------

def boundary_codes_for(grid, substitution: str):
    """``(codes, resolved_codes, kinds, why_skip)`` for one substitution."""
    resolved, refusal = coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        return None, None, None, f"the shipped resolution refuses this grid: {refusal}"
    kinds = coverage.real_pml_boundary_kinds(grid)
    resolved = tuple(int(c) for c in resolved)
    if substitution == "resolved":
        return resolved, resolved, kinds, None
    if substitution == "mirror_as_metallic":
        if BC_MIRROR_PERIODIC not in resolved:
            return None, resolved, kinds, ("no axis resolves to BC_MIRROR_PERIODIC; "
                                           "this control has nothing to misdeclare "
                                           "on this grid")
        codes = tuple(BC_METALLIC if c == BC_MIRROR_PERIODIC else c for c in resolved)
        return codes, resolved, kinds, None
    if substitution == "all_periodic":
        if all(c == BC_PERIODIC for c in resolved):
            return None, resolved, kinds, ("every axis already resolves to "
                                           "BC_PERIODIC; this control is the "
                                           "shipped triple on this grid")
        return (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC), resolved, kinds, None
    raise ValueError(f"unknown substitution {substitution!r}")


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------

def masked_planes(target: str, codes: Sequence[int], shape) -> List[Tuple[int, int]]:
    """The (axis, index) planes the KERNEL zeroes the curl on, for these codes.

    Transcribed from the device text, not from ``stepping``: a component masks
    cell 0 on an axis whose Yee shift there is 0 and whose code is METALLIC or
    MIRROR_PERIODIC, and the LAST stored slot on an axis whose shift is 1 and
    whose code is MIRROR_PERIODIC.
    """
    iyee = IYEE_SHIFTS[target]
    planes = []
    for axis in range(3):
        if iyee[axis] == 0:
            if codes[axis] in (BC_METALLIC, BC_MIRROR_PERIODIC):
                planes.append((axis, 0))
        elif codes[axis] == BC_MIRROR_PERIODIC:
            planes.append((axis, int(shape[axis]) - 1))
    return planes


def _plane_index(axis: int, index: int, shape) -> Tuple[Any, ...]:
    selector: List[Any] = [slice(None)] * 3
    selector[axis] = index
    return tuple(selector)


def plane_mask_array(planes: Sequence[Tuple[int, int]], shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for axis, index in planes:
        mask[_plane_index(axis, index, shape)] = True
    return mask


def localize(reference: np.ndarray, produced: np.ndarray, target: str,
             codes: Sequence[int], resolved: Sequence[int]) -> Dict[str, Any]:
    """Where the two paths differ, against the planes the two code triples disagree on."""
    a = np.ascontiguousarray(reference, dtype=np.float32).view(np.uint32)
    b = np.ascontiguousarray(produced, dtype=np.float32).view(np.uint32)
    differing = a != b
    shape = a.shape
    used = set(masked_planes(target, codes, shape))
    shipped = set(masked_planes(target, resolved, shape))
    delta = plane_mask_array(sorted(used ^ shipped), shape)
    total = int(np.count_nonzero(differing))
    on_delta = int(np.count_nonzero(differing & delta))
    return {
        "differing_words": total,
        "differing_on_mask_delta_planes": on_delta,
        "differing_elsewhere": total - on_delta,
        "mask_delta_planes": sorted(int(x) for pair in sorted(used ^ shipped)
                                    for x in pair) if used ^ shipped else [],
    }


# ---------------------------------------------------------------------------
# The kernel-side backends
# ---------------------------------------------------------------------------

def _source_arrays(fields, sub_step: str, arm: str):
    """The three operand arrays, and the three inverse-epsilon volumes or None."""
    names = ARM_ARRAYS[(sub_step, arm)]["sources"]
    arrays = tuple(getattr(fields, name) for name in names)
    if arm == "derived_E":
        return arrays, tuple(fields.inverse_epsilon_for(c)
                             for c in ("Ex", "Ey", "Ez"))
    return arrays, None


def run_kernel_cuda(sub_step: str, arm: str, fields, codes, dtdx: float,
                    inverse_epsilon=None) -> None:
    """The SHIPPED kernel, through its own launch wrapper, with codes handed in."""
    no_pml_curl.step_no_pml_curl(
        fields, sub_step, tuple(np.int32(c) for c in codes), dtdx, arm=arm,
        inverse_epsilon=inverse_epsilon)
    cp.cuda.runtime.deviceSynchronize()


def _shift_up_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    """``shift_up`` from ``_REAL_PML_PRELUDE``: neighbour one up, far-face rule.

    ONLY PERIODIC WRAPS. Both METALLIC and MIRROR_PERIODIC take the zero face,
    which is the shipped ternary ``(bc == BC_PERIODIC) ? g[wrap] : 0.0f``.
    """
    shifted = np.roll(field, -1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane_index(axis, -1, field.shape)] = np.float32(0.0)
    return shifted


def _shift_dn_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    shifted = np.roll(field, 1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane_index(axis, 0, field.shape)] = np.float32(0.0)
    return shifted


def run_kernel_numpy(sub_step: str, arm: str, fields, codes, dtdx: float,
                     inverse_epsilon=None) -> None:
    """The device tree, transcribed, in float32 -- the laptop backend.

    THE GROUPING IS THE SHIPPED ONE and it is load-bearing::

        float curl = dtdx * ((sf - f1) + (f2 - ss));

    NOT ``dtdx * (sf - f1 + f2 - ss)``, which C associates left to right; float
    addition is not associative and the two differ in the last bits.

    THE DERIVED OPERAND IS FORMED FIRST AND SHIFTED AFTER, which is the order
    ``stepping._read_component`` uses and the order ``derived_up`` implements: the
    far-face ghost of a non-periodic axis is an exact zero rather than the ghost
    cell's product, and a periodic wrap carries the WRAPPED cell's inverse
    epsilon.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the NVRTC
    contraction guard exists for and it is not the shipped bytes. What it settles
    is whether the harness can localize a divergence and whether it can fail.
    """
    shift = _shift_dn_numpy if ARM_ARRAYS[(sub_step, arm)]["backward"] else _shift_up_numpy
    scale = np.float32(dtdx)
    arrays, inverses = _source_arrays(fields, sub_step, arm)
    if inverse_epsilon is not None:
        inverses = inverse_epsilon
    operands = []
    for index in range(3):
        if inverses is None:
            operands.append(np.asarray(arrays[index], dtype=np.float32))
        else:
            operands.append((np.asarray(arrays[index], dtype=np.float32)
                             * np.asarray(inverses[index], dtype=np.float32)
                             ).astype(np.float32))
    for target, first, first_axis, second, second_axis in CURL_TERMS[sub_step]:
        f1 = operands[_SOURCE_INDEX[first]]
        f2 = operands[_SOURCE_INDEX[second]]
        sf = shift(f1, first_axis, codes[first_axis])
        ss = shift(f2, second_axis, codes[second_axis])
        curl = (scale * ((sf - f1) + (f2 - ss))).astype(np.float32)
        for axis, index in masked_planes(target, codes, curl.shape):
            curl[_plane_index(axis, index, curl.shape)] = np.float32(0.0)
        field = getattr(fields, target)
        field[...] = (np.asarray(field, dtype=np.float32) - curl).astype(np.float32)


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                 sub_step: str, arm: str) -> float:
    """Fraction of output WORDS the array path changed from the frozen input.

    Compared as raw uint32, never with ``allclose``: ``-0.0 == 0.0`` and
    ``NaN != NaN`` both lie, and the subnormal-band class puts signed zeros in the
    operands deliberately.
    """
    moved = total = 0
    for name in outputs(sub_step, arm):
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def unmasked_curl(frozen: Dict[str, np.ndarray], fields, sub_step: str, arm: str,
                  codes, dtdx: float) -> Dict[str, np.ndarray]:
    """Each term's curl BEFORE the wall/fold masks, on the host, from the frozen state."""
    shift = _shift_dn_numpy if ARM_ARRAYS[(sub_step, arm)]["backward"] else _shift_up_numpy
    scale = np.float32(dtdx)
    names = ARM_ARRAYS[(sub_step, arm)]["sources"]
    if arm == "derived_E":
        inverses = [to_host(fields.inverse_epsilon_for(c)).astype(np.float32)
                    for c in ("Ex", "Ey", "Ez")]
        operands = [(frozen[names[i]].astype(np.float32) * inverses[i]).astype(np.float32)
                    for i in range(3)]
    else:
        operands = [frozen[names[i]].astype(np.float32) for i in range(3)]
    out = {}
    for target, first, first_axis, second, second_axis in CURL_TERMS[sub_step]:
        f1 = operands[_SOURCE_INDEX[first]]
        f2 = operands[_SOURCE_INDEX[second]]
        sf = shift(f1, first_axis, codes[first_axis])
        ss = shift(f2, second_axis, codes[second_axis])
        out[target] = (scale * ((sf - f1) + (f2 - ss))).astype(np.float32)
    return out


def masked_planes_are_live(curls: Dict[str, np.ndarray], sub_step: str, arm: str,
                           codes) -> Dict[str, Any]:
    """Every plane the kernel ZEROES must carry a nonzero unmasked curl.

    THIS IS THIS GATE'S REPLACEMENT FOR THE FOLDED SIBLING'S MASK-DELTA FLOOR.
    Under the licensed substitution the codes ARE the shipped resolution, so the
    mask delta is empty and that floor is vacuous; a mask whose plane carries an
    identically zero curl is a mask no mutation can be scored against, and
    ``drop_metallic_mask`` would come back UNCAUGHT for a reason about the
    fixture.

    IT GATES THE MUTATION LEGS AND NOT THE BASELINE SWEEP, which is a distinction
    the sibling gates did not need and this one does. A reduced-dimension grid can
    make a mask genuinely unobservable -- on a 1-D z grid ``curl_z = dEy/dx -
    dEx/dy`` is identically zero because both transverse axes are invariant, so
    Bz's metallic cell-0 mask masks nothing that was ever nonzero. That is a true
    fact about 1-D, not a degenerate fixture, and skipping the whole case for it
    would drop the 1-D shape family out of the identity evidence to protect a
    mutation score. So the case is measured and the floor is RECORDED; the triple
    is then excluded from mutation scoring by :func:`scorable_baselines`, with the
    exclusion written into ``mutation_scope``.
    """
    entries = []
    for target in outputs(sub_step, arm):
        curl = curls[target]
        for axis, index in masked_planes(target, codes, curl.shape):
            plane = curl[_plane_index(axis, index, curl.shape)]
            live = int(np.count_nonzero(
                np.ascontiguousarray(plane, dtype=np.float32).view(np.uint32)))
            entries.append({"target": target, "axis": axis, "index": index,
                            "nonzero_words": live, "words": int(plane.size)})
    return {"planes": entries,
            "meets_floor": all(e["nonzero_words"] > 0 for e in entries),
            "planes_masked": len(entries)}


def epsilon_profile(fields, arm: str) -> Dict[str, Any]:
    """Is the inverse epsilon a fixture a mutation could bite on? (derived arm only)"""
    if arm != "derived_E":
        return {"applies": False, "meets_floor": True}
    volumes = [to_host(fields.inverse_epsilon_for(c)).astype(np.float64)
               for c in ("Ex", "Ey", "Ez")]
    deviation = max(float(np.max(np.abs(v - 1.0))) for v in volumes)
    spread = min(float(np.std(v)) for v in volumes)
    distinct = all(not np.array_equal(volumes[i], volumes[j])
                   for i in range(3) for j in range(i + 1, 3))
    return {"applies": True,
            "max_deviation_from_identity": deviation,
            "min_within_volume_spread": spread,
            "three_volumes_pairwise_distinct": bool(distinct),
            "meets_floor": deviation > 0.0 and spread > 0.0 and bool(distinct)}


def sources_are_distinct(frozen: Dict[str, np.ndarray], sub_step: str,
                         arm: str) -> Dict[str, Any]:
    """The three operand volumes differ pairwise, so a component swap has somewhere to show."""
    names = ARM_ARRAYS[(sub_step, arm)]["sources"]
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    same = [f"{names[i]}=={names[j]}" for i, j in pairs
            if np.array_equal(frozen[names[i]], frozen[names[j]])]
    return {"identical_pairs": same, "meets_floor": not same}


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "shape_full": [int(n) for n in grid.shape_full],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "stored_cells": [int(grid.stored_cells(a)) for a in range(3)],
        "owned_cells": [int(grid.owned_cells(a)) for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
        "dimensions": int(getattr(grid, "dimensions", 3)),
    }


def check_terms_against_stepping() -> Dict[str, Any]:
    """Pin :data:`CURL_TERMS` against ``stepping``'s own term tables.

    The table above is a transcription of the DEVICE source and the NumPy leg
    consumes it. If it drifted from ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS``
    the laptop leg would compare a wrong stencil against the oracle and report a
    divergence that says nothing about the kernel.
    """
    out: Dict[str, Any] = {"agreed": True, "checked": [], "disagreements": []}
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        for mine, theirs in zip(CURL_TERMS[sub_step], source):
            record = {
                "sub_step": sub_step, "target": mine[0],
                "mine": list(mine),
                "stepping": [theirs.target, theirs.first, theirs.first_axis,
                             theirs.second, theirs.second_axis],
            }
            record["agrees"] = record["mine"] == record["stepping"]
            out["checked"].append(record)
            if not record["agrees"]:
                out["agreed"] = False
                out["disagreements"].append(record)
    return out


def check_arm_selection_against_stepping() -> Dict[str, Any]:
    """``no_pml_curl_arm`` must branch on the flag ``stepping`` branches on.

    ``_read_component`` (stepping.py:2440) reads ``fields.stores_E``; the arm
    table reads the same attribute. An arm chosen off anything else -- the
    presence of ``fields.Ex``, a polarization count -- would be right on today's
    corpus and wrong on the first configuration that separates them.
    """
    import inspect  # noqa: PLC0415 - a one-shot provenance check
    source = inspect.getsource(stepping._read_component)
    return {
        "stepping_read_component_mentions_stores_E": "fields.stores_E" in source,
        "arm_source_mentions_stores_E":
            'stores_E' in inspect.getsource(no_pml_curl.no_pml_curl_arm),
        "agreed": ("fields.stores_E" in source
                   and "stores_E" in inspect.getsource(no_pml_curl.no_pml_curl_arm)),
    }


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def _flip_tail_sign(source: str) -> Tuple[str, int]:
    """``f = f - curl`` becomes ``f = f + curl``: MEEP's sign, inverted.

    The tail IS this family -- everything above it is the certified curl -- so a
    gate that could not catch a wrong sign here would be measuring the sibling's
    kernel and calling it this one's. MUST BE CAUGHT.
    """
    pattern = r"(\b\w+)\[idx\] = \1\[idx\] - curl;"
    return re.subn(pattern, r"\1[idx] = \1[idx] + curl;", source)[0], \
        len(re.findall(pattern, source))


def _drop_tail_store(source: str) -> Tuple[str, int]:
    """Never write the target: the whole sub-step becomes a no-op. MUST BE CAUGHT."""
    pattern = r"[ ]*(\b\w+)\[idx\] = \1\[idx\] - curl;\n"
    return re.subn(pattern, "", source)[0], len(re.findall(pattern, source))


def _tail_as_fma(source: str) -> Tuple[str, int]:
    """A NULL: ``fmaf(-1, curl, f)`` in place of ``f - curl``.

    ``-1.0f * curl`` is exact (a sign flip) and the fused add rounds once, which
    is what the subtraction does. Its discriminating sibling on the same line is
    :func:`_flip_tail_sign`, which MUST be caught; without the pair, "the tail is
    right" would rest on a leg nobody showed could fail. MUST BE UNCAUGHT.
    """
    pattern = r"(\b\w+)\[idx\] = \1\[idx\] - curl;"
    return re.subn(pattern, r"\1[idx] = __fmaf_rn(-1.0f, curl, \1[idx]);", source)[0], \
        len(re.findall(pattern, source))


def _drop_derived_inverse_epsilon(source: str) -> Tuple[str, int]:
    """DERIVED ARM: the operand becomes D instead of ``D * inv_eps``.

    Invisible against a table of ones, which is why the fixture draws the three
    volumes in [0.2, 0.9) and ``epsilon_profile`` refuses a case where it could
    not bite. MUST BE CAUGHT.
    """
    pattern = r"d\[(idx(?: [-+] ia \* stride)?)\] \* ie\[\1\]"
    return re.subn(pattern, r"d[\1]", source)[0], len(re.findall(pattern, source))


def _derived_operand_order(source: str) -> Tuple[str, int]:
    """A NULL: ``inv_eps * D`` in place of ``D * inv_eps``. IEEE multiply commutes.

    ``get_E`` writes ``D * self.inverse_epsilon_for(component)`` (fields.py:1035)
    with D on the left and the kernel keeps that order for transcription
    discipline, not because it moves a bit. Paired with
    :func:`_drop_derived_inverse_epsilon` on the same expression. MUST BE UNCAUGHT.
    """
    pattern = r"d\[(idx(?: [-+] ia \* stride)?)\] \* ie\[\1\]"
    return re.subn(pattern, r"ie[\1] * d[\1]", source)[0], \
        len(re.findall(pattern, source))


def _alias_derived_inv_eps_to_z(source: str) -> Tuple[str, int]:
    """DERIVED ARM: every component reads Ez's inverse-epsilon volume.

    The defect a wrapper that passed one "representative material array" would
    carry in -- ``Fields.set_epsilon_volumes`` allows three genuinely different
    volumes (diagonal anisotropy, or one isotropic geometry sampled at each
    component's own Yee site), and this makes Ex and Ey use Ez's. EXACTLY
    BIT-IDENTICAL under one shared volume, which is why the fixture draws three
    and ``epsilon_profile`` refuses a case that collapsed them. MUST BE CAUGHT.
    """
    pattern = r"(derived_(?:at|up)\(D[xy], )ie[xy](,)"
    return re.subn(pattern, r"\1iez\2", source)[0], len(re.findall(pattern, source))


def _derived_far_ghost_wraps(source: str) -> Tuple[str, int]:
    """DERIVED ARM: the non-periodic far ghost becomes the wrapped product.

    ``_shift_up``'s METALLIC / folded branch writes an exact zero
    (stepping.py:1828-1830); this makes the helper wrap on every code, which is a
    wrong answer on the far face of every metallic and every folded axis and
    IDENTICAL on an all-periodic grid -- so the leg is scored only where the grid
    carries a non-periodic code. MUST BE CAUGHT.
    """
    needle = ("    return (bc == BC_PERIODIC) ? d[idx - ia * stride] * "
              "ie[idx - ia * stride] : 0.0f;\n")
    replacement = "    return d[idx - ia * stride] * ie[idx - ia * stride];\n"
    return source.replace(needle, replacement), source.count(needle)


def _derived_wrap_uses_own_inv_eps(source: str) -> Tuple[str, int]:
    """DERIVED ARM: the periodic wrap multiplies by the READING cell's inverse epsilon.

    i.e. ``shift_up(D) * inv_eps[idx]`` rather than ``shift_up(D * inv_eps)`` --
    the derive-after-shift defect, localized to the one place the two orders can
    be told apart. Invisible against a constant inverse epsilon; the fixture draws
    an inhomogeneous one and ``epsilon_profile`` refuses a case that did not.
    MUST BE CAUGHT.
    """
    needle = "d[idx - ia * stride] * ie[idx - ia * stride]"
    replacement = "d[idx - ia * stride] * ie[idx]"
    return source.replace(needle, replacement), source.count(needle)


#: Local spellings, on top of the shared battery. The GATE owns which defects a
#: sub-step must be proof against; the spellings live here rather than in the
#: shared probe because these lines exist only in this family's device text.
LOCAL_SOURCE_MUTATIONS = {
    "flip_tail_sign": _flip_tail_sign,
    "drop_tail_store": _drop_tail_store,
    "tail_as_fma": _tail_as_fma,
    "drop_derived_inverse_epsilon": _drop_derived_inverse_epsilon,
    "derived_operand_order": _derived_operand_order,
    "alias_derived_inv_eps_to_z": _alias_derived_inv_eps_to_z,
    "derived_far_ghost_wraps": _derived_far_ghost_wraps,
    "derived_wrap_uses_own_inv_eps": _derived_wrap_uses_own_inv_eps,
}

def index_decomposition_is_observable(shape) -> bool:
    """Can ``fortran_order_index_decomposition`` be seen on a grid of this shape?

    COMPUTED, not argued. The mutation swaps the C-order linear-index
    decomposition for the Fortran-order one; on a grid where only ONE axis has
    extent the two agree cell for cell -- ``(1, 1, 24)`` gives ``(0, 0, idx)``
    either way -- so the edit rewrites the block into itself and no leg can catch
    it. Rather than reason about which shapes those are, both decompositions are
    evaluated over every index and compared.

    MEASURED 2026-08-20 on the GPU host (keep leg, GPU 6, v4): CAUGHT on 16 legs and
    IDENTICAL on the two ``1d_metallic`` ones, shape (1, 1, 24). That is MEEP's
    ``stride(d) = 0`` for a direction it does not have showing up in a mutation
    score, and it gets a NAME (the twin below) rather than a filter, so the
    inertness is measured instead of hidden.
    """
    nx, ny, nz = (int(n) for n in shape)
    index = np.arange(nx * ny * nz)
    c_order = (index // (ny * nz), (index // nz) % ny, index % nz)
    f_order = (index % nx, (index // nx) % ny, index // (nx * ny))
    return any(not np.array_equal(a, b) for a, b in zip(c_order, f_order))


#: THE THIRD STRUCTURAL NULL. Same transform, the shapes where it cannot bite.
LOCAL_SOURCE_MUTATIONS["fortran_order_index_decomposition_on_a_flat_grid"] = \
    probe.SOURCE_MUTATIONS["fortran_order_index_decomposition"]

#: THE SECOND STRUCTURAL NULL, for the same reason and from the same measurement.
#: ``derived_wrap_uses_own_inv_eps`` edits the PERIODIC branch of ``derived_up``,
#: which returns ``d[idx - ia*stride] * ie[idx - ia*stride]``. ON AN INVARIANT AXIS
#: ``ia`` is 0 and ``na`` is 1, so ``idx - ia*stride`` IS ``idx`` and the mutation
#: rewrites the expression into itself. Measured 2026-08-20 on the GPU host (keep leg,
#: GPU 6): CAUGHT on six grids and IDENTICAL on the seventh, and the seventh's only
#: periodic axes are invariant ones. That is MEEP's ``stride(d) = 0`` for a
#: direction it does not have, showing up in a mutation score -- so it gets a name
#: and must come back UNCAUGHT, rather than being filtered out of sight.
LOCAL_SOURCE_MUTATIONS["derived_wrap_uses_own_inv_eps_on_an_invariant_axis"] = \
    _derived_wrap_uses_own_inv_eps

#: THE SAME TRANSFORM UNDER A SECOND NAME, scored as a NULL on the grids where it
#: is INERT. Measured 2026-08-20 on the GPU host (smoke leg, GPU 6): dropping the zero
#: far ghost from ``derived_up`` was CAUGHT on ``3d_metallic`` (505 differing
#: words) and ``3d_mixed`` (370) and came back IDENTICAL on ``fold_Y_periodic``,
#: codes (0, 2, 0). That is not a hole -- it is the folded sibling's reading 1
#: ("the folded ghost values are dead") measured on THIS family: the only axis
#: whose far ghost the mutation changes is the folded PERIODIC one, and the two B
#: components that read a y-shifted operand (Bx off Ez, Bz off Ex) both have Yee
#: shift 1 on y, so ``BC_MIRROR_PERIODIC`` masks the very plane that consumes it.
#: Running it as an unnamed leg would have scored ESCAPED for a reason about the
#: fold; running it under a name that MUST come back UNCAUGHT turns the same
#: measurement into evidence, and a leg that ever catches it is a finding.
LOCAL_SOURCE_MUTATIONS["derived_far_ghost_wraps_on_a_folded_axis"] = \
    _derived_far_ghost_wraps

SOURCE_MUTATIONS_ALL = dict(probe.SOURCE_MUTATIONS, **LOCAL_SOURCE_MUTATIONS)

#: The curl body is the certified pair's, so the certified pair's curl battery is
#: the right battery for it, plus the six the tail and the derive add.
SOURCE_MUTATIONS: Tuple[str, ...] = (
    # The certified curl's own defects, on a body copied verbatim from it.
    "regroup_stencil",
    "drop_metallic_mask",
    "drop_folded_periodic_top_mask",
    "drop_folded_periodic_near_mask",
    "fortran_order_index_decomposition",
    # THE TAIL, which is the only thing this family changed.
    "flip_tail_sign",
    "drop_tail_store",
    # THE DERIVE, which is the only new arithmetic in the package.
    "drop_derived_inverse_epsilon",
    "alias_derived_inv_eps_to_z",
    "derived_far_ghost_wraps",
    "derived_wrap_uses_own_inv_eps",
    # THE NULLS. Each edits the same expression as one of the above.
    "commute_dtdx_scale",
    "tail_as_fma",
    "derived_operand_order",
    "derived_far_ghost_wraps_on_a_folded_axis",
    "derived_wrap_uses_own_inv_eps_on_an_invariant_axis",
    "fortran_order_index_decomposition_on_a_flat_grid",
    # THE FOUR DEAD-PRELUDE NULLS. They edit ``pml_apply``, which this family
    # compiles and never calls; UNCAUGHT is the measurement that the tail is not
    # the split-field recurrence.
    "drop_fu_store",
    "read_fprev_after_store",
    "swap_dsig_dsigu",
    "reload_fu_from_memory",
)

NULL_SOURCE_MUTATIONS: Tuple[str, ...] = (
    "commute_dtdx_scale", "tail_as_fma", "derived_operand_order",
    "derived_far_ghost_wraps_on_a_folded_axis",
    "derived_wrap_uses_own_inv_eps_on_an_invariant_axis",
    "fortran_order_index_decomposition_on_a_flat_grid",
    "drop_fu_store", "read_fprev_after_store", "swap_dsig_dsigu",
    "reload_fu_from_memory",
)

#: A mutation is scored only on legs where its lines can FIRE. Dropping the
#: metallic mask on a grid with no metallic-coded axis changes nothing -- an
#: UNCAUGHT that is a fact about the fixture, not about the kernel. Each entry is
#: a predicate over the grid's SHIPPED code triple.
#:
#: THIS IS NOT A WAY TO EXCUSE AN UNCAUGHT LEG. Every mutation still has to be
#: CAUGHT on every leg that does pass its filter, and ``_score`` reports NO LEGS --
#: never "inert" -- if a filter leaves none. Where a mutation is inert for a
#: STRUCTURAL reason rather than a fixture one, it is given a second NAME and
#: scored as a null on exactly those grids (see
#: ``derived_far_ghost_wraps_on_a_folded_axis``), so the inertness is measured
#: instead of filtered away.
#: EACH FILTER TAKES ``(codes, shape)``. The shape is not decoration: an
#: INVARIANT axis is always coded PERIODIC and holds one cell, so its wrap
#: returns the reading cell itself and a mutation of the wrap rewrites an
#: expression into itself. A filter over the codes alone cannot see that, and it
#: cost an ESCAPED leg on the first device run.
SOURCE_MUTATION_LEG_FILTER = {
    "drop_metallic_mask": lambda codes, shape: BC_METALLIC in codes,
    "drop_folded_periodic_top_mask":
        lambda codes, shape: BC_MIRROR_PERIODIC in codes,
    "drop_folded_periodic_near_mask":
        lambda codes, shape: BC_MIRROR_PERIODIC in codes,
    # THE ZERO FAR GHOST IS OBSERVABLE ONLY ON A METALLIC AXIS. On a folded
    # PERIODIC one the plane that consumes it is the plane the top-plane mask
    # drops -- measured, see the null leg below.
    "derived_far_ghost_wraps": lambda codes, shape: BC_METALLIC in codes,
    "derived_far_ghost_wraps_on_a_folded_axis":
        lambda codes, shape: (BC_MIRROR_PERIODIC in codes
                              and BC_METALLIC not in codes),
    # The wrap branch exists only on a PERIODIC axis, AND only where that axis
    # actually wraps -- i.e. holds more than one cell.
    "derived_wrap_uses_own_inv_eps":
        lambda codes, shape: any(codes[a] == BC_PERIODIC and shape[a] > 1
                                 for a in range(3)),
    "derived_wrap_uses_own_inv_eps_on_an_invariant_axis":
        lambda codes, shape: (BC_PERIODIC in codes
                              and not any(codes[a] == BC_PERIODIC and shape[a] > 1
                                          for a in range(3))),
    # The two index decompositions coincide when only one axis has extent.
    "fortran_order_index_decomposition":
        lambda codes, shape: index_decomposition_is_observable(shape),
    "fortran_order_index_decomposition_on_a_flat_grid":
        lambda codes, shape: not index_decomposition_is_observable(shape),
}

#: The same, in words, for the artifact -- a filter a reader cannot see is a
#: filter a reader cannot check.
SOURCE_MUTATION_LEG_REQUIREMENT: Dict[str, str] = {
    "drop_metallic_mask": "an axis coded BC_METALLIC",
    "drop_folded_periodic_top_mask": "an axis coded BC_MIRROR_PERIODIC",
    "drop_folded_periodic_near_mask": "an axis coded BC_MIRROR_PERIODIC",
    "derived_far_ghost_wraps": "an axis coded BC_METALLIC",
    "derived_far_ghost_wraps_on_a_folded_axis":
        "a folded PERIODIC axis and no metallic one",
    "derived_wrap_uses_own_inv_eps":
        "an axis coded BC_PERIODIC that holds more than one cell",
    "derived_wrap_uses_own_inv_eps_on_an_invariant_axis":
        "a periodic axis, and every periodic axis invariant (one cell)",
    "fortran_order_index_decomposition":
        "a shape on which the C-order and Fortran-order decompositions differ",
    "fortran_order_index_decomposition_on_a_flat_grid":
        "a shape on which the two decompositions coincide (one axis with extent)",
}

#: Mutations whose lines exist only in one kernel. ``derived_*`` edit the derive
#: helpers, which only the derived B kernel has.
SOURCE_MUTATION_REQUIRES_ARM: Dict[str, str] = {
    "drop_derived_inverse_epsilon": "derived_E",
    "derived_operand_order": "derived_E",
    "alias_derived_inv_eps_to_z": "derived_E",
    "derived_far_ghost_wraps": "derived_E",
    "derived_far_ghost_wraps_on_a_folded_axis": "derived_E",
    "derived_wrap_uses_own_inv_eps": "derived_E",
    "derived_wrap_uses_own_inv_eps_on_an_invariant_axis": "derived_E",
}


def host_mutated_inverse_epsilon(name: str, fields, grid):
    """The three inverse-epsilon volumes, corrupted. ``None`` where the leg is not derived.

    THIS FAMILY HAS NO COEFFICIENT TABLES -- that absence is the family -- so the
    certified gates' host mutations (reverse the folded axis's kms, swap the
    sub-lattice) have nothing to corrupt here. The three inverse-epsilon volumes
    ARE this family's host input, and the code triple is the other; both are
    mutated, and the absence of a third is stated rather than left as a gap the
    reader has to notice.
    """
    xp = grid.xp
    volumes = [fields.inverse_epsilon_for(c) for c in ("Ex", "Ey", "Ez")]
    hosts = [to_host(v).astype(np.float32) for v in volumes]
    if name == "reverse_inverse_epsilon_volumes":
        mutated = [np.ascontiguousarray(h[::-1, ::-1, ::-1]) for h in hosts]
    elif name == "alias_inverse_epsilon_to_Ez":
        mutated = [np.ascontiguousarray(hosts[2]) for _ in range(3)]
    elif name == "identity_rebind":
        # THE NULL: the same numbers, through the same copy path. A host-mutation
        # harness that perturbed a leg by the act of rebinding would score every
        # leg CAUGHT and mean nothing by it.
        mutated = [np.ascontiguousarray(h) for h in hosts]
    else:
        raise ValueError(f"unknown host mutation {name!r}")
    return tuple(xp.asarray(m) for m in mutated)


HOST_MUTATIONS: Tuple[str, ...] = ("reverse_inverse_epsilon_volumes",
                                   "alias_inverse_epsilon_to_Ez",
                                   "identity_rebind")
NULL_HOST_MUTATIONS: Tuple[str, ...] = ("identity_rebind",)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], sub_step: str, e_storage: str,
             courant: float, value_class: str, guard: str, substitution: str,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML`` objects.
    There is no second transcription on the oracle leg to drift: the kernel is
    compared against the thing it claims to reproduce, byte for byte, from ONE
    frozen state -- oracle runs, state is restored, kernel runs.
    """
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    arm = arm_of(sub_step, e_storage)
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{sub_step}|{e_storage}|{courant}|{value_class}".encode()).digest()[:4], "big"))

    fields, layer, grid = build(xp, spec, courant, e_storage, rng)
    host = seed_state(fields, grid, sub_step, arm, value_class, rng)
    dtdx = float(grid.dt / grid.dx)
    codes, resolved, kinds, why_skip = boundary_codes_for(grid, substitution)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "arm": arm,
        "e_storage": e_storage, "absorber": spec["absorber"],
        "courant": courant, "value_class": value_class, "guard": guard,
        "backend": backend, "substitution": substitution,
        "host_mutation": host_mutation,
        "fold_axes": spec["axes"], "mirror_phase": spec["phase"],
        "boundaries": list(spec["boundaries"]),
        "resolved_codes": None if resolved is None else [int(c) for c in resolved],
        "boundary_codes": None if codes is None else [int(c) for c in codes],
        "resolved_kinds": None if kinds is None else list(kinds),
        "dtdx": dtdx,
        "structure": structure_facts(grid),
        "operand_census": operand_census(host),
    }
    if why_skip is not None:
        case["skipped"] = why_skip
        case["seconds"] = time.time() - started
        return case

    # THE PREDICATE'S ANSWER, recorded rather than acted on. The gate exists to
    # decide whether it should be believed, so it must not gate the measurement.
    covered, reason = no_pml_curl.covers_no_pml_curl(fields, layer, grid, sub_step)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason,
                               "arm": no_pml_curl.no_pml_curl_arm(fields, sub_step)}

    profile = epsilon_profile(fields, arm)
    case["epsilon_profile"] = profile
    if not profile["meets_floor"]:
        case["skipped"] = ("the inverse-epsilon fixture is degenerate (identity, "
                           "constant, or one volume shared); the derive mutations "
                           "could not bite on this case")
        case["seconds"] = time.time() - started
        return case

    frozen = snapshot(fields, sub_step, arm)

    distinct = sources_are_distinct(frozen, sub_step, arm)
    case["sources_are_distinct"] = distinct
    if not distinct["meets_floor"]:
        case["skipped"] = ("two source volumes are identical; a component-binding "
                           "defect would be invisible on this case")
        case["seconds"] = time.time() - started
        return case

    curls = unmasked_curl(frozen, fields, sub_step, arm, resolved, dtdx)
    case["masked_planes_are_live"] = masked_planes_are_live(curls, sub_step, arm,
                                                            resolved)

    # Leg 1: the oracle. ``layer`` is None or the INERT layer, and stepping takes
    # the same tail for both -- which is the arm this spec axis exists to measure.
    if sub_step == "step_B":
        stepping.step_B(fields, layer)
    else:
        stepping.step_D(fields, layer)
    reference = snapshot(fields, sub_step, arm)

    moved = oracle_moved(frozen, reference, sub_step, arm)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    inverse_epsilon = (host_mutated_inverse_epsilon(host_mutation, fields, grid)
                       if (host_mutation is not None and arm == "derived_E") else None)
    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(sub_step, arm, fields, codes, dtdx, inverse_epsilon)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(sub_step, arm)}
    case["single_launch"] = combine(parts)

    case["localization"] = {
        name: localize(reference[name], to_host(getattr(fields, name)), name,
                       codes, resolved)
        for name in outputs(sub_step, arm)
    }
    case["localization_summary"] = {
        key: sum(v[key] for v in case["localization"].values())
        for key in ("differing_words", "differing_on_mask_delta_planes",
                    "differing_elsewhere")
    }

    # Leg 3: the multi-step, on the LICENSED substitution only -- 60 launches of a
    # deliberately misdeclared kernel measure how a wrong answer compounds, which
    # is not a question this gate asks.
    if host_mutation is None and substitution == LICENSED_SUBSTITUTION:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            if sub_step == "step_B":
                stepping.step_B(fields, layer)
            else:
                stepping.step_D(fields, layer)
            advance_sources(fields, sub_step, arm)
        oracle_multi = snapshot(fields, sub_step, arm)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(sub_step, arm, fields, codes, dtdx, None)
            advance_sources(fields, sub_step, arm)
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs(sub_step, arm)}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str):
    specs = SPECS if product == "full" else SPECS[:5]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    substitutions = SUBSTITUTIONS if product == "full" else (LICENSED_SUBSTITUTION,)
    out = []
    for spec in specs:
        for sub_step in SUB_STEPS:
            for e_storage in E_STORAGE:
                for courant in courants:
                    for value_class in classes:
                        for substitution in substitutions:
                            out.append((spec, sub_step, e_storage, courant,
                                        value_class, substitution))
    return out


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              product: str, guard: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, entry in enumerate(plan, start=1):
        spec, sub_step, e_storage, courant, value_class, substitution = entry
        case = one_case(backend, spec, sub_step, e_storage, courant, value_class,
                        guard, substitution)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"{case['arm']} {substitution} SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        summary = case["localization_summary"]
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"{case['arm']} c={courant} {value_class} {substitution} "
            f"shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"diff={summary['differing_words']} "
            f"on_delta={summary['differing_on_mask_delta_planes']} "
            f"elsewhere={summary['differing_elsewhere']} "
            f"({case['seconds']:.1f} s)")
    return cases


#: The legs a mutation is scored on. CHOSEN, not sliced off the front of the case
#: product: a leg taken from the head landed entirely on unfolded controls in a
#: sibling gate, so the fold-only mutation was skipped on every leg and scored
#: 0/0 UNCAUGHT -- a harness defect that reads exactly like a kernel defect.
#: ``1d_periodic_inert_layer`` is here for ONE leg and would otherwise not be:
#: ``fortran_order_index_decomposition_on_a_flat_grid`` needs a grid where the two
#: index decompositions coincide AND whose baseline is scorable, and the other
#: flat grid (``1d_metallic``) is excluded at ``step_B`` because Bz's metallic
#: cell-0 mask is dead there. Without it that null would score NO LEGS on the two
#: B arms -- which this gate treats as a failure, not as silence.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "3d_metallic", "3d_mixed", "fold_Y_periodic", "fold_Y_metallic",
    "fold_XY_mixed", "2d_metallic_walls", "2d_fold_X_periodic", "1d_metallic",
    "1d_periodic_inert_layer", "3d_periodic",
)


def _spec_by_label(label: str) -> Dict[str, Any]:
    for spec in SPECS:
        if spec["label"] == label:
            return spec
    raise KeyError(label)


_SPEC_FACTS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}


def _spec_facts(spec: Dict[str, Any]):
    """``(codes, shape)`` for a spec's grid, memoized. Built on NumPy: no device needed."""
    label = spec["label"]
    if label not in _SPEC_FACTS:
        planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
        grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                    boundaries=tuple(spec["boundaries"]), symmetry=planes,
                    dimensions=spec.get("dimensions", 3), xp=np, courant=0.5)
        codes, refusal = coverage.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise RuntimeError(f"{label}: {refusal}")
        _SPEC_FACTS[label] = (tuple(int(c) for c in codes),
                              tuple(int(n) for n in grid.shape))
    return _SPEC_FACTS[label]


def spec_boundary_codes(spec: Dict[str, Any]) -> Tuple[int, int, int]:
    """The SHIPPED code triple a spec's grid resolves to."""
    return _spec_facts(spec)[0]


def spec_shape(spec: Dict[str, Any]) -> Tuple[int, int, int]:
    """The STORED extent of a spec's grid -- what tells an invariant axis from a real one."""
    return _spec_facts(spec)[1]


def scorable_baselines(results: Dict[str, Any]):
    """The (label, sub_step, arm) triples whose LICENSED baseline was identical.

    A mutation leg answers "can this gate see this defect", and only an arm whose
    unmutated baseline is bit-identical can answer it: on an arm that already
    diverges every mutation scores CAUGHT for free.

    A TRIPLE WHOSE MASKED PLANES ARE DEAD IS EXCLUDED TOO, and for the opposite
    reason: there every mask mutation scores UNCAUGHT for free. Both exclusions
    are recorded, because "this leg was never asked" and "this leg was asked and
    was inert" are different findings and only one of them is about the kernel.
    """
    scorable, excluded = set(), []
    dead_masks = set()
    for case in results.get("sweep", {}).get("fmad_false", []):
        if case["substitution"] != LICENSED_SUBSTITUTION or case.get("skipped"):
            continue
        key = (case["label"], case["sub_step"], case["arm"])
        if not case.get("masked_planes_are_live", {}).get("meets_floor", True):
            dead_masks.add(key)
        if case["single_launch"]["bit_identical"]:
            scorable.add(key)
        else:
            excluded.append({"key": list(key), "why": "baseline diverged"})
    for key in sorted(dead_masks):
        scorable.discard(key)
        excluded.append({
            "key": list(key),
            "why": ("a plane this arm masks carries an identically zero unmasked "
                    "curl on this grid, so a mask mutation would score UNCAUGHT "
                    "for a reason about the shape rather than the kernel")})
    return scorable, excluded


def mutation_plan(product: str, scorable):
    plan = []
    for label in MUTATION_SPEC_LABELS:
        if product != "full" and label not in {s["label"] for s in SPECS[:5]}:
            continue
        spec = _spec_by_label(label)
        for sub_step in SUB_STEPS:
            for e_storage in E_STORAGE:
                arm = arm_of(sub_step, e_storage)
                if scorable is not None and (label, sub_step, arm) not in scorable:
                    continue
                plan.append((spec, sub_step, e_storage, INEXACT_COURANT, "uniform"))
    return plan


def leg_caught(case: Dict[str, Any]) -> bool:
    if case.get("skipped"):
        return False
    single = not case["single_launch"]["bit_identical"]
    multi = case.get("multi_step", {}).get("bit_identical")
    return bool(single or multi is False)


def _score(legs: List[Dict[str, Any]], must_be_caught: bool) -> Dict[str, Any]:
    ran = [c for c in legs if not c.get("skipped")]
    caught = [c for c in ran if leg_caught(c)]
    if not ran:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = "CAUGHT" if len(caught) == len(ran) else "ESCAPED"
    else:
        verdict = "NULL CONFIRMED" if not caught else "NULL BROKEN"
    return {"ran": len(ran), "caught": len(caught), "verdict": verdict}


def run_host_mutations(results: Dict[str, Any], out_path: str, backend: str,
                       product: str, scorable) -> Dict[str, Any]:
    """Corrupt the three inverse-epsilon volumes a launch binds, and re-run.

    DERIVED LEGS ONLY, by construction: the stored arm and ``step_D`` take no
    inverse epsilon, so there is nothing on the host to corrupt. That is recorded
    rather than silently skipped -- see :func:`host_mutated_inverse_epsilon`.
    """
    out: Dict[str, Any] = {}
    plan = [entry for entry in mutation_plan(product, scorable)
            if arm_of(entry[1], entry[2]) == "derived_E"]
    for name in HOST_MUTATIONS:
        legs = [one_case(backend, spec, sub_step, e_storage, courant, value_class,
                         "fmad_false", LICENSED_SUBSTITUTION, host_mutation=name)
                for spec, sub_step, e_storage, courant, value_class in plan]
        scored = _score(legs, name not in NULL_HOST_MUTATIONS)
        out[name] = dict(scored, applies_to_arm="derived_E", cases=legs)
        log(f"[host-mut] {name}: caught {scored['caught']}/{scored['ran']} "
            f"-> {scored['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def _leg_carries(mutation: str, spec: Dict[str, Any], arm: str) -> bool:
    required_arm = SOURCE_MUTATION_REQUIRES_ARM.get(mutation)
    if required_arm is not None and arm != required_arm:
        return False
    leg_filter = SOURCE_MUTATION_LEG_FILTER.get(mutation)
    return leg_filter is None or bool(
        leg_filter(spec_boundary_codes(spec), spec_shape(spec)))


def run_source_mutations(results: Dict[str, Any], out_path: str, product: str,
                         scorable) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED strings and recompiled.

    The compile memo is keyed through the SOURCE, so a mutated body is a miss and
    reaches NVRTC. Every leg records how many constructions came from the mutated
    bytes, because a leg reporting a pass for a mutation it never applied is worse
    than no leg -- three measured instances on the sibling track.
    """
    from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

    originals = {name: no_pml_curl.kernel_source(name)
                 for name in no_pml_curl.NO_PML_KERNELS}
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)

    try:
        for sub_step in SUB_STEPS:
            for e_storage in E_STORAGE:
                arm = arm_of(sub_step, e_storage)
                if sub_step == "step_D" and e_storage == "derived":
                    continue  # one kernel, already scored under 'stored'
                kernel_name = no_pml_curl.ARM_KERNELS[(sub_step, arm)]
                for name in SOURCE_MUTATIONS:
                    transform = SOURCE_MUTATIONS_ALL[name]
                    mutated, sites = transform(originals[kernel_name])
                    key = f"{sub_step}:{arm}:{name}"
                    leg_plan = [entry for entry in plan
                                if entry[1] == sub_step
                                and arm_of(entry[1], entry[2]) == arm
                                and _leg_carries(name, entry[0], arm)]
                    if sites == 0 or mutated == originals[kernel_name]:
                        out[key] = {
                            "armed": False, "sites": sites,
                            "expected_absent": bool(
                                SOURCE_MUTATION_REQUIRES_ARM.get(name)
                                not in (None, arm)),
                            "why": (f"matched {sites} site(s) and changed nothing "
                                    f"in {kernel_name}"),
                        }
                        log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                        results["source_mutations"] = out
                        save(results, out_path)
                        continue
                    no_pml_curl.set_kernel_source(kernel_name, mutated)
                    no_pml_curl.clear_kernel_cache()
                    compile_cache.clear_compile_log()
                    digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                    legs = [one_case("cuda", spec, sub, storage, courant,
                                     value_class, "fmad_false",
                                     LICENSED_SUBSTITUTION)
                            for spec, sub, storage, courant, value_class in leg_plan]
                    no_pml_curl.set_kernel_source(kernel_name,
                                                  originals[kernel_name])
                    no_pml_curl.clear_kernel_cache()
                    from_mutated = sum(1 for entry in compile_cache.compile_log()
                                       if entry["source_sha256"] == digest)
                    scored = _score(legs, name not in NULL_SOURCE_MUTATIONS)
                    out[key] = dict(
                        scored, armed=True, sites=sites, kernel=kernel_name,
                        leg_requirement=SOURCE_MUTATION_LEG_REQUIREMENT.get(name),
                        requires_arm=SOURCE_MUTATION_REQUIRES_ARM.get(name),
                        legs_in_plan=len([e for e in plan
                                          if arm_of(e[1], e[2]) == arm
                                          and e[1] == sub_step]),
                        legs_carrying_the_code=len(leg_plan),
                        mutated_source_sha256=digest,
                        kernel_constructions_from_mutated_bytes=from_mutated,
                        cases=legs)
                    if from_mutated == 0 and scored["ran"]:
                        out[key]["verdict"] = "UNACCOUNTED"
                        out[key]["why"] = ("no kernel construction used the "
                                           "mutated bytes; this leg measured the "
                                           "SHIPPED kernel")
                    log(f"[src-mut] {key}: caught {scored['caught']}/{scored['ran']} "
                        f"builds_from_mutated={from_mutated} -> {out[key]['verdict']}")
                    results["source_mutations"] = out
                    save(results, out_path)
    finally:
        for name, source in originals.items():
            no_pml_curl.set_kernel_source(name, source)
        no_pml_curl.clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    """The finding, partitioned by arm, with every clause it rests on named."""
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]

    arms: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        key = f"{case['sub_step']}::{case['arm']}::{case['substitution']}"
        entry = arms.setdefault(key, {
            "cases": 0, "single_identical": 0, "multi_cases": 0,
            "multi_identical": 0, "differing_words": 0,
            "differing_on_mask_delta_planes": 0, "differing_elsewhere": 0,
            "labels": set(), "absorbers": set(), "e_storage": set()})
        entry["cases"] += 1
        entry["labels"].add(case["label"])
        entry["absorbers"].add(case["absorber"])
        entry["e_storage"].add(case["e_storage"])
        if case["single_launch"]["bit_identical"]:
            entry["single_identical"] += 1
        if "multi_step" in case:
            entry["multi_cases"] += 1
            if case["multi_step"]["bit_identical"]:
                entry["multi_identical"] += 1
        for field in ("differing_words", "differing_on_mask_delta_planes",
                      "differing_elsewhere"):
            entry[field] += case["localization_summary"][field]
    for entry in arms.values():
        for field in ("labels", "absorbers", "e_storage"):
            entry[field] = sorted(entry[field])
        entry["all_identical"] = (entry["single_identical"] == entry["cases"]
                                  and entry["multi_identical"] == entry["multi_cases"])
        entry["divergence_confined_to_mask_delta"] = (
            entry["differing_elsewhere"] == 0 and entry["differing_words"] > 0)

    reasons: List[str] = []

    # THE THREE ARMS ARE ALL REQUIRED. A run that swept only the stored B arm
    # would release on one of the five reachable corpus slots and say nothing
    # about the four the derived arm carries.
    present_arms = {(c["sub_step"], c["arm"]) for c in scored}
    for required in (("step_B", "stored_E"), ("step_B", "derived_E"),
                     ("step_D", "magnetic")):
        if required not in present_arms:
            reasons.append(f"the sweep contained no {required[0]}/{required[1]} case")

    licensed_keys = sorted(k for k in arms if k.endswith(f"::{LICENSED_SUBSTITUTION}"))
    for key in licensed_keys:
        licensed = arms[key]
        if licensed["single_identical"] != licensed["cases"]:
            reasons.append(
                f"{key}: single-launch divergence on "
                f"{licensed['cases'] - licensed['single_identical']} of "
                f"{licensed['cases']} cases")
        if licensed["multi_identical"] != licensed["multi_cases"]:
            reasons.append(
                f"{key}: multi-step divergence on "
                f"{licensed['multi_cases'] - licensed['multi_identical']} of "
                f"{licensed['multi_cases']} cases")

    # BOTH ABSORBER SHAPES must appear somewhere in the licensed sweep, or "an
    # inert layer steps like no layer" stays an inference about pml.py.
    licensed_absorbers = {c["absorber"] for c in scored
                          if c["substitution"] == LICENSED_SUBSTITUTION}
    for shape in ("none", "inert_layer"):
        if shape not in licensed_absorbers:
            reasons.append(f"no licensed case ran with absorber={shape}")

    # step_D UNDER BOTH E STORAGE SHAPES, which is what makes "step_D does not
    # read E" a measurement.
    d_storage = {c["e_storage"] for c in scored
                 if c["sub_step"] == "step_D"
                 and c["substitution"] == LICENSED_SUBSTITUTION}
    if d_storage != set(E_STORAGE):
        reasons.append(f"step_D was scored under E storage {sorted(d_storage)} "
                       f"only; both are required")

    terms = results.get("curl_terms_vs_stepping", {})
    if not terms.get("agreed", False):
        reasons.append("the transcribed curl term table disagrees with stepping's")
    if not results.get("arm_selection_vs_stepping", {}).get("agreed", False):
        reasons.append("the arm table does not branch on the flag stepping branches on")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is not inert")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            if leg.get("expected_absent"):
                continue  # a derive mutation on a kernel with no derive helpers
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']} "
                           f"({leg.get('caught')}/{leg.get('ran')})")

    # WHAT A RELEASE RUN MUST CONTAIN AT ALL. Every clause above judges evidence
    # that is present; these two judge its ABSENCE, which is the failure mode a
    # gate cannot see from inside a green summary -- and it is also what keeps the
    # planted-defect check applicable rather than quietly skipped.
    if results.get("backend") == "cuda" and results.get("product") == "full":
        if results.get("skip_mutations"):
            reasons.append("--skip-mutations: no mutation evidence in this record")
        elif not results.get("source_mutations"):
            reasons.append("no source-mutation section was written; the device "
                           "battery did not run")
        if not any(c["substitution"] != LICENSED_SUBSTITUTION for c in scored):
            reasons.append("no misdeclaration control was scored; the licensed "
                           "arm's identity is not isolated from the fixture")

    # THE MISDECLARATION CONTROLS. A gate whose deliberate defects also came back
    # identical would be reporting a property of its fixture, not of the kernel.
    controls: Dict[str, Any] = {}
    for substitution in SUBSTITUTIONS:
        if substitution == LICENSED_SUBSTITUTION:
            continue
        legs = [c for c in scored if c["substitution"] == substitution]
        diverged = [c for c in legs if not c["single_launch"]["bit_identical"]]
        confined = all(c["localization_summary"]["differing_elsewhere"] == 0
                       for c in diverged) if diverged else None
        controls[substitution] = {
            "cases": len(legs), "diverged": len(diverged),
            "confined_to_mask_delta": confined,
            "differing_words": sum(c["localization_summary"]["differing_words"]
                                   for c in legs),
        }
        if legs and not diverged:
            reasons.append(
                f"the misdeclaration control {substitution} went IDENTICAL on all "
                f"{len(legs)} cases; a deliberately wrong code triple that changes "
                f"nothing means this gate is not measuring the boundary handling")
    if ("mirror_as_metallic" in controls
            and controls["mirror_as_metallic"]["cases"]
            and controls["mirror_as_metallic"]["confined_to_mask_delta"] is False):
        reasons.append(
            "mirror_as_metallic diverged OFF the mask-delta planes; the two codes "
            "share a ghost rule, so a difference elsewhere means the control moved "
            "something other than the top-plane mask")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT
               and c["substitution"] == LICENSED_SUBSTITUTION]
    guarded_identical = {
        (c["label"], c["sub_step"], c["arm"], c["courant"], c["value_class"])
        for c in scored if c["substitution"] == LICENSED_SUBSTITUTION
        and c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["arm"], c["courant"],
                      c["value_class"]) in guarded_identical]
    control_diverged = [c for c in comparable if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(control_diverged),
        "diverged_by_arm": {
            arm: sum(1 for c in control_diverged if c["arm"] == arm)
            for arm in sorted({c["arm"] for c in comparable})},
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored where "
                    "the guarded leg was identical" if not comparable else
                    "the contraction guard is load-bearing on this family"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here"),
    }

    return {
        "released": not reasons,
        "reasons": reasons,
        "licensed_arms": licensed_keys,
        "licensed_substitution": LICENSED_SUBSTITUTION,
        "misdeclaration_controls": controls,
        "arms": arms,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "claim": ("the no-absorber CUDA curl family -- step_B_no_pml_real, "
                  "step_B_no_pml_real_derived and step_D_no_pml_real -- "
                  "is byte-identical to stepping.step_B / stepping.step_D per "
                  "sub-step from one frozen state, at one launch and at "
                  f"{MULTI_STEP_BUDGET}, over periodic / metallic / mixed walls, a "
                  "mirror fold at both terminations, 1-D / 2-D / 3-D shapes, an "
                  "absent absorber and an inert layer, both float32 subnormal "
                  "policies, both value classes and both courants"),
        "does_not_claim": [
            "nothing dispatches these kernels; this gate licenses a predicate, "
            "not a wiring",
            "a conductivity is refused, not measured: MEEP's Absorber layer IS a "
            "conductivity and _apply_curl routes it to _apply_conductive_update",
            "complex storage, cylindrical coordinates, a Bloch phase, BFAST, "
            "special_kz and chi2/chi3 are refused by inherited clauses and "
            "untested here",
            "three simultaneous mirror planes were not swept and are refused",
            "the constitutive sub-steps are no_pml_constitutive.py's null plan, "
            "not this gate's",
            "no throughput claim: this is a correctness gate and times nothing",
        ],
    }


def verdict_flips_against_planted_defect(results: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the verdict against a record with a defect planted in it.

    A GATE WHOSE VERDICT CANNOT GO RED IS NOT A GATE. Three independent defects
    are planted, one at a time, into a COPY of the finished record: a licensed
    case flipped to diverging, a must-be-caught source mutation flipped to
    ESCAPED, and the misdeclaration control flipped to identical. Each must make
    ``released`` False on its own.

    A PLANT THE RECORD CANNOT CARRY IS REPORTED, NOT PASSED AND NOT FAILED. On the
    NumPy backend there are no source mutations to flip, and on ``--product
    reduced`` there is no misdeclaration control; those plants come back
    ``applicable: False`` with the reason, and ``all_flipped`` is over the plants
    that could be applied. A FULL DEVICE RUN CARRIES ALL THREE -- that is what
    :func:`summarize`'s mutation and control clauses are for -- so an inapplicable
    plant there is itself a finding, and ``inapplicable`` is printed rather than
    tucked into a boolean.
    """
    out: Dict[str, Any] = {"plants": [], "all_flipped": True, "inapplicable": []}
    base = summarize(results)

    def plant(name: str, mutate, why_absent: str) -> None:
        copy_of = copy.deepcopy(results)
        applied = mutate(copy_of)
        verdict = summarize(copy_of) if applied else None
        flipped = bool(applied) and not verdict["released"]
        out["plants"].append({
            "plant": name, "applicable": bool(applied),
            "why_not_applicable": None if applied else why_absent,
            "released_after_plant": None if verdict is None else verdict["released"],
            "flipped": flipped,
            "first_reason": None if verdict is None or verdict["released"]
            else verdict["reasons"][0][:160],
        })
        if not applied:
            out["inapplicable"].append(name)
        elif not flipped:
            out["all_flipped"] = False

    def flip_a_licensed_case(record) -> bool:
        for case in record.get("sweep", {}).get("fmad_false", []):
            if (not case.get("skipped")
                    and case["substitution"] == LICENSED_SUBSTITUTION
                    and case["single_launch"]["bit_identical"]):
                case["single_launch"]["bit_identical"] = False
                return True
        return False

    def flip_a_caught_mutation(record) -> bool:
        for leg in record.get("source_mutations", {}).values():
            if leg.get("armed") and leg.get("verdict") == "CAUGHT":
                leg["verdict"] = "ESCAPED"
                leg["caught"] = 0
                return True
        return False

    def silence_the_control(record) -> bool:
        touched = False
        for case in record.get("sweep", {}).get("fmad_false", []):
            if (not case.get("skipped")
                    and case["substitution"] != LICENSED_SUBSTITUTION):
                case["single_launch"]["bit_identical"] = True
                case["localization_summary"] = {
                    "differing_words": 0, "differing_on_mask_delta_planes": 0,
                    "differing_elsewhere": 0}
                touched = True
        return touched

    plant("a_licensed_case_diverges", flip_a_licensed_case,
          "no licensed case in the record was bit-identical to begin with")
    plant("a_must_be_caught_mutation_escapes", flip_a_caught_mutation,
          "the record carries no armed CAUGHT source mutation (NumPy backend, or "
          "--skip-mutations)")
    plant("the_misdeclaration_control_goes_identical", silence_the_control,
          "the record carries no non-licensed substitution case (--product reduced)")
    out["released_unplanted"] = base["released"]
    return out


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_no_pml_curl",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "skip_mutations": bool(args.skip_mutations),
        "question": ("are the three no-absorber CUDA curl kernels byte-identical "
                     "to stepping.step_B / stepping.step_D per sub-step and per "
                     "arm -- including the DERIVED-E arm, which forms D * inv_eps "
                     "at the point of use because that is what the array path "
                     "differences when nothing stores E?"),
        "kernels": list(no_pml_curl.NO_PML_KERNELS),
        "curl_terms_vs_stepping": check_terms_against_stepping(),
        "arm_selection_vs_stepping": check_arm_selection_against_stepping(),
        "kernel_source_sha256": {
            name: hashlib.sha256(
                no_pml_curl.kernel_source(name).encode("utf-8")).hexdigest()
            for name in no_pml_curl.NO_PML_KERNELS},
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        observer = probe.install_nvrtc_binary_observer()
        results["nvrtc_observer"] = observer
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": ("NumPy backend: compiles nothing, "
                                           "certifies nothing")}
    save(results, args.out)

    for guard, options, _is_primary in GUARD_SETS:
        if args.backend == "numpy" and guard != "fmad_false":
            continue  # there is no compiler on this leg to guard
        if args.backend == "cuda":
            no_pml_curl._COMPILE_OPTIONS = tuple(options)
            no_pml_curl.clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        no_pml_curl._COMPILE_OPTIONS = ("--fmad=false",)
        no_pml_curl.clear_kernel_cache()

    if not args.skip_mutations:
        scorable, excluded = scorable_baselines(results)
        results["mutation_scope"] = {
            "substitution": LICENSED_SUBSTITUTION,
            "scorable": sorted("::".join(k) for k in scorable),
            "excluded": excluded,
            "why": ("a mutation leg answers 'can this gate see this defect', and "
                    "only an arm whose unmutated baseline is bit-identical can "
                    "answer it"),
        }
        log(f"[mut-scope] {len(scorable)} scorable (label, sub_step, arm) triples; "
            f"{len(excluded)} excluded because the baseline already diverges")
        save(results, args.out)
        run_host_mutations(results, args.out, args.backend, args.product, scorable)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.product, scorable)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["verdict_flips_against_planted_defect"] = \
        verdict_flips_against_planted_defect(results)
    if not results["verdict_flips_against_planted_defect"]["all_flipped"]:
        results["summary"]["released"] = False
        results["summary"]["reasons"].append(
            "the release verdict did not flip against every planted defect; a "
            "verdict that cannot go red is not a verdict")
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']}")
    for name, arm in sorted(verdict["arms"].items()):
        log(f"[verdict]   {name}: single {arm['single_identical']}/{arm['cases']} "
            f"multi {arm['multi_identical']}/{arm['multi_cases']} "
            f"diff={arm['differing_words']} "
            f"on_delta={arm['differing_on_mask_delta_planes']} "
            f"elsewhere={arm['differing_elsewhere']}")
    for plant in results["verdict_flips_against_planted_defect"]["plants"]:
        log(f"[verdict]   plant {plant['plant']}: "
            f"applicable={plant['applicable']} flipped={plant['flipped']}"
            + ("" if plant["applicable"]
               else f" ({plant['why_not_applicable']})"))
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
