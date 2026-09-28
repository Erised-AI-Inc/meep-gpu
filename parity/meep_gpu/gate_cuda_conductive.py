"""Sub-step byte-identity gate for the CONDUCTIVE CUDA curl family.

THE QUESTION
============
``meep_gpu/cuda_kernels/conductive_kernels.py`` carries five kernels for the two
tails of ``stepping._apply_curl`` that read ``fields.condfac_for(term.target)``
(stepping.py:508) -- the no-absorber conductive update (:536-537) and the
four-case conductivity + PML recurrence (:520-534). None of them has ever run on
a device. This gate asks, per sub-step, per arm and PER CONDUCTIVITY MASK, whether
each is byte-identical to ``stepping.step_B`` / ``stepping.step_D`` from ONE
frozen state -- at one launch and at 60 -- across periodic, metallic and mixed
walls, a mirror fold at both terminations, 1-D / 2-D / 3-D shapes, an absent
absorber, an inert layer and an ACTIVE one, both float32 subnormal policies, both
value classes and both courants.

THE MASK IS THE EXPERIMENTAL VARIABLE, AND THAT IS THE POINT
============================================================
``_apply_curl`` reads the conductivity PER TARGET COMPONENT, so one lossy
component and two lossless take DIFFERENT tails inside one sub-step. The kernel
answers that with three compile-time constants, and "the lossless component is
identical by construction" is a claim about the preprocessor. Five masks are swept
-- all three lossy, each one lossy alone, and two of three -- so the claim is
measured on the device rather than read off the source. A gate that swept only the
all-lossy mask would say nothing about the configuration MEEP's own allocation
granularity makes ordinary (``s->conductivity[c][d]``).

WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
================================================================
* ``oracle_moved`` -- the fraction of output words the array path changed from the
  frozen input. A case that moved nothing is SKIPPED, not passed.
* ``conductivity_profile`` -- ``max|condfac - 1| > 0``, a nonzero spread WITHIN
  each volume, and the live volumes pairwise distinct. Against a table of ones the
  whole tail is invisible; against a CONSTANT sigma a coefficient-index defect is;
  against one shared volume a component-binding defect is. Each is a mutation in
  this gate's battery.
* ``pml_case_census`` -- on the PML arms, EVERY ONE of MEEP's four subchunk cases
  must carry a nonzero cell count somewhere in the case. Without it a green run
  could mean "case A was right and the other three were never reached", which is
  precisely the shape of the defect the four-case tail exists to avoid.
* ``masked_planes_are_live`` -- every plane the kernel ZEROES must carry a nonzero
  UNMASKED curl, or ``drop_metallic_mask`` comes back UNCAUGHT for a reason about
  the fixture.
* ``sources_are_distinct`` and ``epsilon_profile`` (derived arm) -- the sibling's
  floors, unchanged, for the sibling's reasons.
* The mutation battery, INCLUDING legs that must come back UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything.
* ``verdict_flips_against_planted_defect`` -- the release verdict is recomputed
  against a record with a defect planted in it, and the run fails if the verdict
  does not flip.

THE NEAR-ONE COEFFICIENT TABLE, WHICH IS A FIXTURE AND NOT A DEFECT
====================================================================
The four-case selection turns on an EXACT comparison, ``kms != 1.0``
(stepping.py:2055-2058). A kernel written with a tolerance is a defect the REAL
tables cannot expose: on the sibling track's real ``2d_cond_pml`` layout ``kms_x``
has 561/640 entries exactly 1.0 and 561 within 1e-7 of 1.0, so there is no
near-one band for a tolerance to mis-classify. This gate therefore carries a
``near_one`` coefficient-table variant -- ONE entry of one axis's ``kms`` set to
``0.99999994f``, the largest float32 below 1.0 -- installed on the PML object so
the ORACLE and the KERNEL both see it. The
``tolerance_instead_of_exact_predicate`` source mutation must be CAUGHT on that
variant and is expected UNCAUGHT on the shipped one; both are scored and the
contrast is the measurement.

RUNNING IT
==========
Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_conductive.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the
same floors and the same HOST mutations against a transcription of the device
tree. IT COMPILES NOTHING AND CERTIFIES NOTHING -- what it settles is whether the
harness can localize a divergence and whether it can fail::

    python -u gate_cuda_conductive.py --backend numpy --out /tmp/conductive.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible, which
matters here more than elsewhere: the array path's four-case conductive curl is
the engine's most expensive, and a correctness result is not a speed result.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

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
from meep_gpu.fields import IYEE_SHIFTS, Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import conductive_kernels, coverage  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260820

#: Consecutive launches in the multi-step leg, the budget every hand-CUDA record
#: is cut at. "Identical for N steps" is a claim about N -- and it is a sharper
#: claim here than on the lossless families, because the conductive PML tail
#: writes TWO histories whose errors compound only across launches.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the operands moving. Held fixed, the curl reads
#: the same numbers every launch and a 60-launch leg becomes a slow single-launch
#: leg. The same exact float32 scale is applied on both paths.
_ADVANCE = np.float32(0.97)

BC_PERIODIC = 0
BC_METALLIC = 1
BC_MIRROR_PERIODIC = 2

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")
E_STORAGE: Tuple[str, ...] = ("stored", "derived")

#: The per-component conductivity masks swept, in ``CONDUCTIVE_SUB_STEPS`` order.
#: ``(False, False, False)`` is NOT here and must not be: the predicate refuses it
#: by name, because a lossless curl belongs to the certified pair or to
#: ``no_pml_curl`` and admitting it here would be an OVERLAP the union census
#: scores as a widening.
COND_MASKS: Tuple[Tuple[bool, bool, bool], ...] = (
    (True, True, True), (True, False, False), (False, True, False),
    (False, False, True), (True, False, True),
)
MASK_IDS = {mask: "".join("C" if flag else "-" for flag in mask)
            for mask in COND_MASKS}

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: ``--fmad=false`` is carried as CORRECTNESS, not tuning: every conductive tail
#: is a chain of multiply-then-subtract and ``(f * cf) - curl`` is a textbook
#: fused-multiply-add candidate the array path rounds in two operations. The
#: unguarded control is scored at the inexact courant and its effect is REPORTED,
#: not asserted.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

SUBSTITUTIONS: Tuple[str, ...] = ("resolved", "mirror_as_metallic", "all_periodic")
LICENSED_SUBSTITUTION = "resolved"

#: The coefficient-table variants. Only ``shipped`` is a table a run could
#: produce; ``near_one`` is a FIXTURE that puts one entry in the band an
#: exact-comparison kernel and a tolerance kernel disagree about, and it is
#: installed on the PML object so the oracle sees it too. Both are licensed --
#: neither is a misdeclaration -- and the mutation battery is what reads the
#: contrast between them.
COEFFICIENT_TABLES: Tuple[str, ...] = ("shipped", "near_one")

#: The largest float32 strictly below 1.0. ``1.0 - 1e-7`` is NOT this number: it
#: rounds to exactly 1.0 in float32 (eps is 1.19e-7), which would make the
#: near-one fixture a no-op that reads as a passing leg.
NEAR_ONE = np.float32(0.99999994)

#: The grids. Every wall kind appears, the fold appears at BOTH terminations, and
#: 1-D / 2-D / 3-D shapes all appear -- the corpus's own conductive rows are
#: (1,1,400) periodic/periodic/metallic, (200,200,1) metallic/metallic/periodic
#: and (150,150,1) metallic/metallic/periodic under an ACTIVE layer, so a
#: 3-D-only sweep would measure a shape family the corpus does not contain.
#:
#: ``absorber`` is "none" (no object), "inert_layer" (a PML with every face at
#: zero thickness, whose ``is_active`` is False) or a thickness triple (ACTIVE).
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
    {"label": "3d_mixed_inert_layer", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("periodic", "metallic", "metallic"), "axes": "", "phase": 1,
     "absorber": "inert_layer"},
    {"label": "fold_Y_periodic", "cell": (8.0, 16.0, 12.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"), "axes": "Y", "phase": 1,
     "absorber": "none"},
    {"label": "fold_Z_metallic", "cell": (8.0, 12.0, 16.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "metallic"), "axes": "Z", "phase": -1,
     "absorber": "none"},
    {"label": "2d_metallic_walls", "cell": (12.0, 14.0, 0.0), "dimensions": 2,
     "boundaries": ("metallic", "metallic", "periodic"), "axes": "", "phase": 1,
     "absorber": "none"},
    {"label": "1d_metallic", "cell": (0.0, 0.0, 24.0), "dimensions": 1,
     "boundaries": ("periodic", "periodic", "metallic"), "axes": "", "phase": 1,
     "absorber": "none"},
    # THE ACTIVE-LAYER SPECS. Which of MEEP's four subchunk cases a target can
    # reach is decided by whether BOTH of its ladder axes carry an absorbing face:
    # ``dsig`` and ``dsigu`` are that component's own cycle (``vec.hpp``), so a
    # layer absent from one of them leaves that predicate False everywhere and two
    # of the four cases empty. ``pml_case_census`` refuses a case that reached
    # fewer than four on every conductive target, and the specs below are chosen so
    # some do and some do not -- the ones that do not are a real fact about a
    # reduced-dimension grid and are SKIPPED with the census printed, not passed.
    {"label": "3d_pml_two_axes", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("metallic", "metallic", "periodic"), "axes": "", "phase": 1,
     "absorber": ((2, 2), (2, 2), (0, 0))},
    {"label": "2d_pml_two_axes", "cell": (12.0, 14.0, 0.0), "dimensions": 2,
     "boundaries": ("metallic", "metallic", "periodic"), "axes": "", "phase": 1,
     "absorber": ((3, 3), (3, 3), (0, 0))},
    {"label": "2d_pml_mixed_walls", "cell": (12.0, 14.0, 0.0), "dimensions": 2,
     "boundaries": ("periodic", "metallic", "periodic"), "axes": "", "phase": 1,
     "absorber": ((3, 3), (0, 3), (0, 0))},
    {"label": "1d_pml_metallic", "cell": (0.0, 0.0, 24.0), "dimensions": 1,
     "boundaries": ("periodic", "periodic", "metallic"), "axes": "", "phase": 1,
     "absorber": ((0, 0), (0, 0), (3, 3))},
    {"label": "3d_pml_fold_Y", "cell": (8.0, 16.0, 12.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "metallic"), "axes": "Y", "phase": 1,
     "absorber": ((2, 2), (0, 2), (0, 0))},
    # A LAYER ON ALL THREE AXES. This is the only spec on which a SINGLE-component
    # mask reaches all four subchunk cases: a target's two ladder axes are
    # ``vec.hpp``'s cycle of its own, so a layer absent from one of them leaves
    # that target's dsigu (or dsig) inactive everywhere and cases A and B empty.
    # The layer occupies the boundary cells only, so the interior is case D --
    # "PML everywhere" would be a different fixture and is not this one.
    {"label": "3d_pml_all_axes", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("metallic", "periodic", "metallic"), "axes": "", "phase": 1,
     "absorber": ((2, 2), (2, 2), (2, 2))},
)


def spec_is_active(spec: Dict[str, Any]) -> bool:
    return spec["absorber"] not in ("none", "inert_layer")


def layer_kind(spec: Dict[str, Any]) -> str:
    return "active" if spec_is_active(spec) else "none"


def arm_of(spec: Dict[str, Any], sub_step: str, e_storage: str) -> str:
    """The arm this (spec, sub_step, storage) takes, from the shipped classifier's rule."""
    if sub_step == "step_D":
        return "magnetic"
    if spec_is_active(spec):
        return "stored_E"   # enable_pml_storage stores E unconditionally
    return "stored_E" if e_storage == "stored" else "derived_E"


#: What each ``(sub_step, arm, layer)`` writes and reads, spelled as the ARRAYS a
#: launch binds. ``step_D``'s no-absorber sources are the B arrays and are written
#: that way on purpose: "Hx" is the name that made the first version of the
#: no-absorber predicate refuse every run it exists to serve.
def arm_arrays(sub_step: str, arm: str, layer: str) -> Dict[str, Any]:
    if sub_step == "step_B":
        targets = ("Bx", "By", "Bz")
        sources = ("Ex", "Ey", "Ez") if arm == "stored_E" else ("Dx", "Dy", "Dz")
        return {"targets": targets, "sources": sources, "backward": False}
    targets = ("Dx", "Dy", "Dz")
    sources = ("Hx", "Hy", "Hz") if layer == "active" else ("Bx", "By", "Bz")
    return {"targets": targets, "sources": sources, "backward": True}


#: The six curl terms as the KERNELS spell them, transcribed from the device text.
#: ``(target, first source index, first axis, second source index, second axis)``.
#: Written out rather than derived from ``stepping`` so the NumPy leg transcribes
#: the DEVICE source rather than comparing the oracle with itself;
#: :func:`check_terms_against_stepping` pins the correspondence.
CURL_TERMS: Dict[str, Tuple[Tuple[str, int, int, int, int], ...]] = {
    "step_B": (("Bx", 2, 1, 1, 2), ("By", 0, 2, 2, 0), ("Bz", 1, 0, 0, 1)),
    "step_D": (("Dx", 2, 1, 1, 2), ("Dy", 0, 2, 2, 0), ("Dz", 1, 0, 0, 1)),
}

#: ``vec.hpp``'s cycle, per target: ``(dsig axis, dsigu axis)`` as axis indices.
COEFFICIENT_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def conductive_side(sub_step: str) -> str:
    return "B" if sub_step == "step_B" else "D"


def build(xp, spec: Dict[str, Any], sub_step: str, mask, courant: float,
          e_storage: str, coefficient_table: str, rng):
    """A frozen ``(fields, layer, grid)`` triple for one point of the product.

    THE SIGMA IS INHOMOGENEOUS AND PER COMPONENT. A uniform sigma makes a
    coefficient-index defect invisible; one shared volume makes a
    component-binding defect invisible. ``conductivity_profile`` is the floor that
    refuses a case where they could not bite.

    BOTH SIDES ARE INSTALLED WHERE THE MASK ASKS FOR THEM, but only THIS
    sub-step's targets matter to the kernel -- ``_apply_curl`` reads the
    conductivity per term, so a B sigma cannot reach ``step_D``. The other side is
    left lossless so the case is unambiguous about which sigma it measured.
    """
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                dimensions=spec.get("dimensions", 3), xp=xp, courant=courant)
    fields = Fields(grid=grid)
    if not spec_is_active(spec) and e_storage == "derived":
        # THREE DISTINCT, INHOMOGENEOUS inverse-epsilon volumes, drawn away from
        # 1.0 -- the derived arm's own floor, unchanged from the sibling gate.
        inverse, epsilon = {}, {}
        for component in ("Ex", "Ey", "Ez"):
            values = rng.uniform(0.2, 0.9, size=grid.shape).astype(np.float32)
            inverse[component] = xp.asarray(np.ascontiguousarray(values))
            epsilon[component] = xp.asarray(
                np.ascontiguousarray((1.0 / values).astype(np.float32)))
        fields.set_epsilon_volumes(epsilon, inverse)

    layer = None
    if spec["absorber"] == "inert_layer":
        layer = PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
    elif spec_is_active(spec):
        layer = PML(grid=grid, thickness=spec["absorber"])
        if coefficient_table == "near_one":
            install_near_one(layer)

    if layer is not None and layer.is_active:
        fields.enable_pml_storage()
    elif e_storage == "stored":
        fields.enable_field_storage()

    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    sigma = {}
    for component, flag in zip(targets, mask):
        if flag:
            sigma[component] = xp.asarray(np.ascontiguousarray(
                rng.uniform(0.05, 0.9, size=grid.shape).astype(np.float32)))
    setter = (fields.set_b_conductivity if sub_step == "step_B"
              else fields.set_d_conductivity)
    setter(sigma)
    return fields, layer, grid


def install_near_one(layer) -> Dict[str, Any]:
    """Put ONE coefficient entry in the band a tolerance would mis-classify.

    Written into the PML OBJECT, so ``stepping`` reads it through
    ``_curl_coefficients`` and the launcher reads it through
    ``real_pml_curl_tables``. Both paths therefore see the same table and the
    experiment is about the PREDICATE inside the tail, not about two different
    tables.

    The entry chosen is one that is currently EXACTLY 1.0 -- i.e. outside the
    absorbing region, where the array path takes case D or C -- so the change
    moves a cell ACROSS the boundary the predicate draws. Perturbing an entry that
    is already far from 1.0 would change a number and not a classification.
    """
    record: Dict[str, Any] = {"installed": False, "entries": []}
    for attribute in ("kms_x", "kms_y", "kms_z", "kms_x_h", "kms_y_h", "kms_z_h"):
        vector = getattr(layer, attribute, None)
        if vector is None:
            continue
        flat = to_host(vector).reshape(-1)
        exact = np.flatnonzero(flat == np.float32(1.0))
        if exact.size == 0:
            continue
        index = int(exact[exact.size // 2])
        host = to_host(vector).copy()
        host.reshape(-1)[index] = NEAR_ONE
        getattr(layer, attribute)[...] = layer.grid.xp.asarray(host)
        record["entries"].append({"attribute": attribute, "index": index,
                                  "value": float(NEAR_ONE)})
        record["installed"] = True
    return record


def state_names(sub_step: str, arm: str, layer: str, mask) -> Tuple[str, ...]:
    spec = arm_arrays(sub_step, arm, layer)
    names = list(spec["targets"]) + list(spec["sources"])
    if layer == "active":
        names += [f"fu_{t}" for t in spec["targets"]]
        names += [f"f_cond_{t}" for t, flag in zip(spec["targets"], mask) if flag]
    return tuple(dict.fromkeys(names))


def outputs(sub_step: str, arm: str, layer: str, mask) -> Tuple[str, ...]:
    """Every array this sub-step WRITES, auxiliaries and histories included.

    A tree that gets the field right and an auxiliary wrong is correct for exactly
    one launch, and the conductive PML tail writes TWO of them under predicates.
    Comparing only the field would make ``drop_history_store`` invisible at a
    single launch -- measured on the sibling track as a 120/120 UNCAUGHT.
    """
    spec = arm_arrays(sub_step, arm, layer)
    names = list(spec["targets"])
    if layer == "active":
        names += [f"fu_{t}" for t in spec["targets"]]
        names += [f"f_cond_{t}" for t, flag in zip(spec["targets"], mask) if flag]
    return tuple(names)


def seed_state(fields, grid, names, value_class: str, rng) -> Dict[str, np.ndarray]:
    xp = grid.xp
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


def snapshot(fields, names) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in names}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, sub_step: str, arm: str, layer: str) -> None:
    for name in arm_arrays(sub_step, arm, layer)["sources"]:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# The boundary codes, and the substitution that is the experimental variable
# ---------------------------------------------------------------------------

def boundary_codes_for(grid, substitution: str):
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
    """The (axis, index) planes the KERNEL zeroes the curl on, for these codes."""
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
    a = np.ascontiguousarray(reference, dtype=np.float32).view(np.uint32)
    b = np.ascontiguousarray(produced, dtype=np.float32).view(np.uint32)
    differing = a != b
    shape = a.shape
    stem = target.split("_")[-1]
    used = set(masked_planes(stem, codes, shape))
    shipped = set(masked_planes(stem, resolved, shape))
    delta = plane_mask_array(sorted(used ^ shipped), shape)
    total = int(np.count_nonzero(differing))
    on_delta = int(np.count_nonzero(differing & delta))
    return {"differing_words": total,
            "differing_on_mask_delta_planes": on_delta,
            "differing_elsewhere": total - on_delta}


# ---------------------------------------------------------------------------
# The kernel-side backends
# ---------------------------------------------------------------------------

def curl_tables(layer, sub_step: str):
    """``real_pml_curl_tables`` for this sub-step's sub-lattice, or None."""
    if layer is None or not layer.is_active:
        return None
    from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
    return step_curl_kernels.real_pml_curl_tables(
        layer, half_integer=(sub_step == "step_B"))


def run_kernel_cuda(spec, sub_step: str, arm: str, fields, layer, codes,
                    dtdx: float, inverse_epsilon=None, conductivity=None) -> None:
    """The SHIPPED kernel, through its own launch wrapper, with codes handed in."""
    conductive_kernels.step_conductive_curl(
        fields, layer, sub_step, tuple(np.int32(c) for c in codes), dtdx,
        tables=curl_tables(layer, sub_step),
        arm=(layer_kind(spec), arm), inverse_epsilon=inverse_epsilon,
        conductivity=conductivity)
    cp.cuda.runtime.deviceSynchronize()


def _shift_up_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    shifted = np.roll(field, -1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane_index(axis, -1, field.shape)] = np.float32(0.0)
    return shifted


def _shift_dn_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    shifted = np.roll(field, 1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane_index(axis, 0, field.shape)] = np.float32(0.0)
    return shifted


def _operand_volumes(fields, sub_step: str, arm: str, layer: str,
                     inverse_epsilon=None) -> List[np.ndarray]:
    names = arm_arrays(sub_step, arm, layer)["sources"]
    arrays = [np.asarray(to_host(getattr(fields, n)), dtype=np.float32) for n in names]
    if arm != "derived_E":
        return arrays
    inverses = inverse_epsilon or tuple(fields.inverse_epsilon_for(c)
                                        for c in ("Ex", "Ey", "Ez"))
    return [(arrays[i] * np.asarray(to_host(inverses[i]), dtype=np.float32)
             ).astype(np.float32) for i in range(3)]


def unmasked_curls(fields, sub_step: str, arm: str, layer: str, codes,
                   dtdx: float, inverse_epsilon=None) -> Dict[str, np.ndarray]:
    """Each term's curl BEFORE the wall/fold masks, on the host, from the current state."""
    shift = _shift_dn_numpy if arm_arrays(sub_step, arm, layer)["backward"] \
        else _shift_up_numpy
    scale = np.float32(dtdx)
    operands = _operand_volumes(fields, sub_step, arm, layer, inverse_epsilon)
    out = {}
    for target, first, first_axis, second, second_axis in CURL_TERMS[sub_step]:
        f1, f2 = operands[first], operands[second]
        sf = shift(f1, first_axis, codes[first_axis])
        ss = shift(f2, second_axis, codes[second_axis])
        out[target] = (scale * ((sf - f1) + (f2 - ss))).astype(np.float32)
    return out


def run_kernel_numpy(spec, sub_step: str, arm: str, fields, layer, codes,
                     dtdx: float, inverse_epsilon=None, conductivity=None) -> None:
    """The device tree, transcribed, in float32 -- the laptop backend.

    THE GROUPING IS THE SHIPPED ONE and it is load-bearing everywhere:
    ``dtdx * ((sf - f1) + (f2 - ss))`` in the curl, ``((f * cf) - curl) * ci`` in
    the conductive tail, and the four case expressions of ``cond_pml_apply``.

    THE BRANCH IS SELECTED, NOT ACCUMULATED. This is what the array path does not
    do -- it evaluates all four cases over the whole volume and copies -- and it
    is the step whose equivalence the gate exists to measure.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING.
    """
    kind = layer_kind(spec)
    curls = unmasked_curls(fields, sub_step, arm, kind, codes, dtdx, inverse_epsilon)
    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    supplied = conductivity or {}
    for index, target in enumerate(targets):
        curl = curls[target]
        for axis, plane in masked_planes(target, codes, curl.shape):
            curl[_plane_index(axis, plane, curl.shape)] = np.float32(0.0)
        field = np.asarray(to_host(getattr(fields, target)), dtype=np.float32)
        condfac = supplied.get(f"condfac_{target}", fields.condfac_for(target))
        condinv = supplied.get(f"condinv_{target}", fields.condinv_for(target))
        if kind != "active":
            if condfac is None:
                new = (field - curl).astype(np.float32)
            else:
                cf = np.asarray(to_host(condfac), dtype=np.float32)
                ci = np.asarray(to_host(condinv), dtype=np.float32)
                new = (((field * cf) - curl) * ci).astype(np.float32)
            getattr(fields, target)[...] = fields.grid.xp.asarray(new)
            continue

        first_axis, second_axis = COEFFICIENT_AXES[index]
        km1, si1 = _coefficient_pair(layer, first_axis, sub_step)
        km2, si2 = _coefficient_pair(layer, second_axis, sub_step)
        aux = np.asarray(to_host(getattr(fields, "fu_" + target)), dtype=np.float32)
        dsig = np.broadcast_to((km1 != 1.0) | (si1 != 1.0), field.shape)
        dsigu = np.broadcast_to((km2 != 1.0) | (si2 != 1.0), field.shape)
        if condfac is None:
            new_aux = (((aux * km1) - curl) * si1).astype(np.float32)
            new = ((((field * km2) + new_aux) - aux) * si2).astype(np.float32)
            getattr(fields, target)[...] = fields.grid.xp.asarray(new)
            getattr(fields, "fu_" + target)[...] = fields.grid.xp.asarray(new_aux)
            continue
        cf = np.asarray(to_host(condfac), dtype=np.float32)
        ci = np.asarray(to_host(condinv), dtype=np.float32)
        history = np.asarray(to_host(getattr(fields, "f_cond_" + target)),
                             dtype=np.float32)
        c_new = (((history * cf) - curl) * ci).astype(np.float32)
        u_cond = (((aux * cf) - curl) * ci).astype(np.float32)
        u_split = ((((aux * km1) + c_new) - history) * si1).astype(np.float32)
        u_new = np.where(dsig, u_split, u_cond).astype(np.float32)
        f_split = ((((field * km2) + u_new) - aux) * si2).astype(np.float32)
        f_first = ((((field * km1) + c_new) - history) * si1).astype(np.float32)
        f_direct = (((field * cf) - curl) * ci).astype(np.float32)
        new = np.where(dsigu, f_split,
                       np.where(dsig, f_first, f_direct)).astype(np.float32)
        getattr(fields, target)[...] = fields.grid.xp.asarray(new)
        getattr(fields, "fu_" + target)[...] = fields.grid.xp.asarray(
            np.where(dsigu, u_new, aux).astype(np.float32))
        getattr(fields, "f_cond_" + target)[...] = fields.grid.xp.asarray(
            np.where(dsig, c_new, history).astype(np.float32))


def _coefficient_pair(layer, axis: int, sub_step: str):
    """``(kms, sinv)`` for one axis at this sub-step's sub-lattice, as host arrays."""
    name = ("x", "y", "z")[axis]
    kms, sinv = stepping._curl_coefficients(layer, name,
                                            half_integer=(sub_step == "step_B"))
    return (np.asarray(to_host(kms), dtype=np.float32),
            np.asarray(to_host(sinv), dtype=np.float32))


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                 names: Sequence[str]) -> float:
    moved = total = 0
    for name in names:
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def conductivity_profile(fields, sub_step: str, mask) -> Dict[str, Any]:
    """Is the conductivity a fixture a mutation could bite on?

    ``max|condfac - 1| > 0`` -- otherwise the whole tail is an identity and
    ``drop_condfac`` is invisible. A nonzero spread WITHIN each volume -- otherwise
    a coefficient-INDEX defect (a component reading its neighbour's table, a
    Fortran-order index decomposition) reads the same number everywhere and is
    invisible. And the live volumes pairwise distinct -- otherwise a
    component-binding defect is.
    """
    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    live = [t for t, flag in zip(targets, mask) if flag]
    volumes = [np.asarray(to_host(fields.condfac_for(t)), dtype=np.float64)
               for t in live]
    inverses = [np.asarray(to_host(fields.condinv_for(t)), dtype=np.float64)
                for t in live]
    deviation = max(float(np.max(np.abs(v - 1.0))) for v in volumes)
    spread = min(float(np.std(v)) for v in volumes)
    distinct = all(not np.array_equal(volumes[i], volumes[j])
                   for i in range(len(volumes)) for j in range(i + 1, len(volumes)))
    inverse_deviation = max(float(np.max(np.abs(v - 1.0))) for v in inverses)
    return {"live_components": live,
            "max_condfac_deviation_from_identity": deviation,
            "max_condinv_deviation_from_identity": inverse_deviation,
            "min_within_volume_spread": spread,
            "live_volumes_pairwise_distinct": bool(distinct),
            "meets_floor": (deviation > 0.0 and inverse_deviation > 0.0
                            and spread > 0.0 and bool(distinct))}


def pml_case_census(layer, sub_step: str, shape, mask) -> Dict[str, Any]:
    """How many cells each of MEEP's four subchunk cases claims, per target.

    THE NON-VACUITY FLOOR THIS FAMILY NEEDS AND THE LOSSLESS ONES DO NOT. The
    four-case tail is four expressions and a two-level selection; a fixture whose
    layer puts every cell in case A would exercise one of them, and a green run
    would be a statement about the layer. The floor is that every case is reached
    on at least one CONDUCTIVE target -- a lossless target takes the plain
    recurrence and its census says nothing about this tail.
    """
    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    per_target: Dict[str, Dict[str, int]] = {}
    for index, (target, flag) in enumerate(zip(targets, mask)):
        if not flag:
            continue
        first_axis, second_axis = COEFFICIENT_AXES[index]
        km1, si1 = _coefficient_pair(layer, first_axis, sub_step)
        km2, si2 = _coefficient_pair(layer, second_axis, sub_step)
        dsig = np.broadcast_to((km1 != 1.0) | (si1 != 1.0), shape)
        dsigu = np.broadcast_to((km2 != 1.0) | (si2 != 1.0), shape)
        per_target[target] = {"A": int((dsig & dsigu).sum()),
                              "B": int((~dsig & dsigu).sum()),
                              "C": int((dsig & ~dsigu).sum()),
                              "D": int((~dsig & ~dsigu).sum())}
    reached = {case: max((counts[case] for counts in per_target.values()), default=0)
               for case in ("A", "B", "C", "D")}
    return {"per_target": per_target, "reached": reached,
            "meets_floor": all(count > 0 for count in reached.values())}


def masked_planes_are_live(curls: Dict[str, np.ndarray], sub_step: str,
                           codes) -> Dict[str, Any]:
    """Every plane the kernel ZEROES must carry a nonzero unmasked curl."""
    entries = []
    for target in conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]:
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
    if arm != "derived_E":
        return {"applies": False, "meets_floor": True}
    volumes = [to_host(fields.inverse_epsilon_for(c)).astype(np.float64)
               for c in ("Ex", "Ey", "Ez")]
    deviation = max(float(np.max(np.abs(v - 1.0))) for v in volumes)
    spread = min(float(np.std(v)) for v in volumes)
    distinct = all(not np.array_equal(volumes[i], volumes[j])
                   for i in range(3) for j in range(i + 1, 3))
    return {"applies": True, "max_deviation_from_identity": deviation,
            "min_within_volume_spread": spread,
            "three_volumes_pairwise_distinct": bool(distinct),
            "meets_floor": deviation > 0.0 and spread > 0.0 and bool(distinct)}


def sources_are_distinct(frozen, sub_step: str, arm: str, layer: str) -> Dict[str, Any]:
    names = arm_arrays(sub_step, arm, layer)["sources"]
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    same = [f"{names[i]}=={names[j]}" for i, j in pairs
            if np.array_equal(frozen[names[i]], frozen[names[j]])]
    return {"identical_pairs": same, "meets_floor": not same}


def structure_facts(grid) -> Dict[str, Any]:
    return {"shape": [int(n) for n in grid.shape],
            "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
            "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
            "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
            "dimensions": int(getattr(grid, "dimensions", 3))}


def check_terms_against_stepping() -> Dict[str, Any]:
    """Pin :data:`CURL_TERMS` and :data:`COEFFICIENT_AXES` against ``stepping``."""
    axis_of = {"x": 0, "y": 1, "z": 2}
    index_of = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 0, "Hy": 1, "Hz": 2}
    out: Dict[str, Any] = {"agreed": True, "checked": [], "disagreements": []}
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        for mine, theirs, axes in zip(CURL_TERMS[sub_step], source, COEFFICIENT_AXES):
            record = {
                "sub_step": sub_step, "target": mine[0],
                "mine": list(mine) + list(axes),
                "stepping": [theirs.target, index_of[theirs.first], theirs.first_axis,
                             index_of[theirs.second], theirs.second_axis,
                             axis_of[theirs.dsig], axis_of[theirs.dsigu]],
            }
            record["agrees"] = record["mine"] == record["stepping"]
            out["checked"].append(record)
            if not record["agrees"]:
                out["agreed"] = False
                out["disagreements"].append(record)
    return out


def check_mask_against_stepping() -> Dict[str, Any]:
    """``conductive_targets`` must read the attribute ``_apply_curl`` reads.

    A mask taken off ``fields.has_conductivity`` -- which is per SIDE -- would be
    right on today's corpus, where every conductive row carries all three
    components, and wrong on the first run that does not.
    """
    import ast  # noqa: PLC0415 - a one-shot provenance check
    import inspect  # noqa: PLC0415

    def code_of(function) -> str:
        """The function's SOURCE WITH ITS DOCSTRING REMOVED.

        The docstring is where the reason lives, and the reason names the
        attribute this check exists to forbid. Matching on the raw source would
        make an accurate comment fail the gate.
        """
        tree = ast.parse(inspect.getsource(function).lstrip())
        body = tree.body[0].body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        return "\n".join(ast.unparse(node) for node in body)

    apply_curl = code_of(stepping._apply_curl)
    mask = code_of(conductive_kernels.conductive_targets)
    return {"apply_curl_reads_condfac_for": "condfac_for(term.target)" in apply_curl,
            "mask_reads_condfac_for": "condfac_for" in mask,
            "mask_avoids_has_conductivity": "has_conductivity" not in mask,
            "agreed": ("condfac_for(term.target)" in apply_curl
                       and "condfac_for" in mask
                       and "has_conductivity" not in mask)}


# ---------------------------------------------------------------------------
# The source-mutation battery
# ---------------------------------------------------------------------------

def _drop_condfac(source: str) -> Tuple[str, int]:
    """``(f[idx] * cf[idx])`` -> ``f[idx]`` in the no-PML tail. MUST BE CAUGHT."""
    needle = "f[idx] = ((f[idx] * cf[idx]) - curl) * ci[idx];"
    return (source.replace(needle, "f[idx] = (f[idx] - curl) * ci[idx];"),
            source.count(needle))


def _drop_condinv(source: str) -> Tuple[str, int]:
    """Drop ``* ci[idx]`` from the no-PML tail. MUST BE CAUGHT."""
    needle = "f[idx] = ((f[idx] * cf[idx]) - curl) * ci[idx];"
    return (source.replace(needle, "f[idx] = ((f[idx] * cf[idx]) - curl);"),
            source.count(needle))


def _swap_condfac_condinv(source: str) -> Tuple[str, int]:
    """The two coefficients exchanged. MUST BE CAUGHT.

    ``condfac = 1 - sigma*dt/2`` and ``condinv = 1/(1 + sigma*dt/2)`` are DIFFERENT
    functions of the same sigma (fields.py:821-822), so this is a wrong absorber
    rather than a broken one -- smooth, converged and entirely wrong, which is the
    class of defect that survives an eyeball check.
    """
    out = source
    count = 0
    for needle, replacement in (
        ("f[idx] = ((f[idx] * cf[idx]) - curl) * ci[idx];",
         "f[idx] = ((f[idx] * ci[idx]) - curl) * cf[idx];"),
        ("    float cfv = cf[idx];\n    float civ = ci[idx];\n",
         "    float cfv = ci[idx];\n    float civ = cf[idx];\n"),
    ):
        count += out.count(needle)
        out = out.replace(needle, replacement)
    return out, count


def _cond_operand_order(source: str) -> Tuple[str, int]:
    """A NULL: ``cf[idx] * f[idx]`` for ``f[idx] * cf[idx]``. IEEE multiply commutes.

    ``stepping._apply_conductive_update`` writes ``field *= condfac`` -- the field
    on the left -- and the kernel keeps that order for transcription discipline,
    not because it changes a bit. Its discriminating sibling on the same
    expression is :func:`_drop_condfac`, which MUST be caught; without that pair,
    "the operand order is inert" would be a statement about a leg nobody showed
    could fail. MUST BE UNCAUGHT.
    """
    needle = "f[idx] = ((f[idx] * cf[idx]) - curl) * ci[idx];"
    return (source.replace(needle, "f[idx] = ((cf[idx] * f[idx]) - curl) * ci[idx];"),
            source.count(needle))


def _tolerance_instead_of_exact_predicate(source: str) -> Tuple[str, int]:
    """``!= 1.0f`` replaced by a 1e-7 tolerance. MUST BE CAUGHT on ``near_one``.

    THE DEFECT THE REAL TABLES CANNOT EXPOSE. ``stepping.py:2055-2058`` is a bare
    ``!= 1.0``; a real PML table has no entry within 1e-7 of 1.0 that is not
    exactly 1.0, so on the shipped table this mutation is expected UNCAUGHT and
    that is not evidence of anything. It is scored on BOTH coefficient-table
    variants and the CONTRAST is the measurement.
    """
    out = source
    count = 0
    for needle, replacement in (
        ("bool dsig = (km1 != 1.0f) || (si1 != 1.0f);",
         "bool dsig = (fabsf(km1 - 1.0f) > 1e-7f) || (fabsf(si1 - 1.0f) > 1e-7f);"),
        ("bool dsigu = (km2 != 1.0f) || (si2 != 1.0f);",
         "bool dsigu = (fabsf(km2 - 1.0f) > 1e-7f) || (fabsf(si2 - 1.0f) > 1e-7f);"),
    ):
        count += out.count(needle)
        out = out.replace(needle, replacement)
    return out, count


def _flatten_split_grouping(source: str) -> Tuple[str, int]:
    """``(((fv*km2) + u_new) - uv)`` flattened to ``((fv*km2) + (u_new - uv))``.

    The array path accumulates in place, one operation at a time
    (stepping.py:2085-2089), and float addition is not associative. MUST BE
    CAUGHT -- the sibling track's own recon control went 0/18 on NumPy alone.
    """
    needle = "float f_split  = (((fv * km2) + u_new) - uv) * si2;"
    return (source.replace(needle,
                           "float f_split  = ((fv * km2) + (u_new - uv)) * si2;"),
            source.count(needle))


def _drop_history_store(source: str) -> Tuple[str, int]:
    """Never write ``f_cond``. MUST BE CAUGHT -- and only because it is COMPARED.

    ``f`` is bit-identical on the first launch (``cv`` is loaded before the store),
    so this is invisible to a field-only comparison and lethal from the second
    launch on. It is the measurement of whether the history comparison is
    load-bearing.
    """
    needle = "    if (dsig) c[idx] = c_new;\n"
    return source.replace(needle, ""), source.count(needle)


def _store_history_unconditionally(source: str) -> Tuple[str, int]:
    """Write ``f_cond`` everywhere, ignoring ``dsig``. MUST BE CAUGHT.

    "UNCHANGED" in MEEP's case table is a REQUIREMENT, not a don't-care: the array
    path restores the prior value in the inactive region (stepping.py:2107-2109)
    so a diagnostic read cannot mistake unused arithmetic for state.
    """
    needle = "    if (dsig) c[idx] = c_new;\n"
    return source.replace(needle, "    c[idx] = c_new;\n"), source.count(needle)


def _store_auxiliary_unconditionally(source: str) -> Tuple[str, int]:
    """Same, for ``fu``. MUST BE CAUGHT."""
    needle = "    if (dsigu) u[idx] = u_new;\n"
    return source.replace(needle, "    u[idx] = u_new;\n"), source.count(needle)


def _swap_case_precedence(source: str) -> Tuple[str, int]:
    """``dsigu`` and ``dsig`` exchanged in the two-level selection. MUST BE CAUGHT.

    MEEP's precedence is ``dsigu`` first (stepping.py:2101-2105 copies
    ``first_only`` and ``direct`` only where ``dsigu`` is INACTIVE). Swapping it
    picks case C where the array path picks case A, which is wrong only inside the
    two-axis overlap -- a corner region, which is exactly where a reader would not
    look.
    """
    needle = "    f[idx] = dsigu ? f_split : (dsig ? f_first : f_direct);"
    return (source.replace(needle,
                           "    f[idx] = dsig ? f_first : (dsigu ? f_split : f_direct);"),
            source.count(needle))


def _swap_cond_dsig_dsigu(source: str) -> Tuple[str, int]:
    """The two coefficient pairs exchanged AT THE CALL SITE. MUST BE CAUGHT.

    A half-cell/axis confusion: the ``dsig`` ladder and the ``dsigu`` ladder run
    along different axes (``vec.hpp``'s cycle), so this is the conductive tail's
    version of the highest-consequence defect the constitutive kernels name --
    a smooth, converged, entirely wrong absorber.
    """
    pattern = (r"(cond_pml_apply\([^;]*?idx, curl, )"
               r"(kms_[xyz]\[[ijk]\], sinv_[xyz]\[[ijk]\]), "
               r"(kms_[xyz]\[[ijk]\], sinv_[xyz]\[[ijk]\])\)")
    return re.subn(pattern, r"\1\3, \2)", source)[0], \
        len(re.findall(pattern, source))


def _cond_pml_scale_commutes(source: str) -> Tuple[str, int]:
    """A NULL: ``cfv * cv`` for ``cv * cfv`` in the history expression.

    Paired with :func:`_drop_history_store` and :func:`_flatten_split_grouping` on
    the same helper, both of which MUST be caught -- so the pair says the
    comparator is sensitive to the ASSOCIATION of these expressions and
    insensitive to the operand order of a multiply, which is what IEEE-754 says.
    MUST BE UNCAUGHT.
    """
    needle = "float c_new    = ((cv * cfv) - curl) * civ;"
    return (source.replace(needle, "float c_new    = ((cfv * cv) - curl) * civ;"),
            source.count(needle))


LOCAL_SOURCE_MUTATIONS: Dict[str, Any] = {
    "drop_condfac": _drop_condfac,
    "drop_condinv": _drop_condinv,
    "swap_condfac_condinv": _swap_condfac_condinv,
    "cond_operand_order": _cond_operand_order,
    "tolerance_instead_of_exact_predicate": _tolerance_instead_of_exact_predicate,
    "flatten_split_grouping": _flatten_split_grouping,
    "drop_history_store": _drop_history_store,
    "store_history_unconditionally": _store_history_unconditionally,
    "store_auxiliary_unconditionally": _store_auxiliary_unconditionally,
    "swap_case_precedence": _swap_case_precedence,
    "swap_cond_dsig_dsigu": _swap_cond_dsig_dsigu,
    "cond_pml_scale_commutes": _cond_pml_scale_commutes,
}


#: Everything armed. The first five are the SHARED curl defects -- they must still
#: be caught here, because the whole safety argument is that the curl half is the
#: sibling's bytes and a transformation that damaged a mask would be invisible to
#: a byte-equality test that stripped the same lines. The rest are this family's.
SOURCE_MUTATIONS: Tuple[str, ...] = (
    "regroup_stencil",
    "drop_metallic_mask",
    "drop_folded_periodic_top_mask",
    "fortran_order_index_decomposition",
    "fortran_order_index_decomposition_on_a_flat_grid",
    "commute_dtdx_scale",
    "drop_condfac",
    "drop_condinv",
    "swap_condfac_condinv",
    "cond_operand_order",
    "tolerance_instead_of_exact_predicate",
    "flatten_split_grouping",
    "drop_history_store",
    "store_history_unconditionally",
    "store_auxiliary_unconditionally",
    "swap_case_precedence",
    "swap_cond_dsig_dsigu",
    "cond_pml_scale_commutes",
)

#: MUST BE UNCAUGHT. Each is paired with a defect on the SAME expression that must
#: be caught, because "inert" only means something beside a demonstration that the
#: leg can fail at all.
NULL_SOURCE_MUTATIONS: Tuple[str, ...] = (
    # paired with fortran_order_index_decomposition, which MUST be caught and is
    # the SAME transform on the shapes where it is observable
    "fortran_order_index_decomposition_on_a_flat_grid",
    "commute_dtdx_scale",         # paired with regroup_stencil
    "cond_operand_order",         # paired with drop_condfac
    "cond_pml_scale_commutes",    # paired with flatten_split_grouping
)

#: Mutations that only exist in one of the two tails. A leg where the site is
#: absent is recorded as NOT ARMED with ``expected_absent`` True rather than
#: scored, so "unasked" and "inert" stay different findings.
SOURCE_MUTATION_REQUIRES_LAYER = {
    "drop_condfac": "none", "drop_condinv": "none",
    "cond_operand_order": "none",
    "tolerance_instead_of_exact_predicate": "active",
    "flatten_split_grouping": "active", "drop_history_store": "active",
    "store_history_unconditionally": "active",
    "store_auxiliary_unconditionally": "active",
    "swap_case_precedence": "active", "swap_cond_dsig_dsigu": "active",
    "cond_pml_scale_commutes": "active",
    "drop_folded_periodic_top_mask": None,
}

#: WHICH LEGS CARRY THE CODE A MUTATION EDITS. A mask mutation on a grid with no
#: such wall edits a line the compiler keeps and no cell ever reaches, so it comes
#: back UNCAUGHT for a reason about the FIXTURE -- which reads exactly like a
#: kernel defect and is not one. The filter is on the resolved code triple, so it
#: cannot drift from what the launcher actually passes.
def index_decomposition_is_observable(shape) -> bool:
    """Can ``fortran_order_index_decomposition`` be seen on a grid of this shape?

    COMPUTED, not argued. The mutation swaps the C-order linear-index
    decomposition for the Fortran-order one; on a grid where only ONE axis has
    extent the two agree cell for cell -- ``(1, 1, 24)`` gives ``(0, 0, idx)``
    either way -- so the edit rewrites the block into itself and no leg can catch
    it. Rather than reason about which shapes those are, both decompositions are
    evaluated over every index and compared.

    MEASURED 2026-08-20 on the GPU host (keep leg, GPU 3): scored on 20 legs of
    ``step_D_no_pml_conductive``, CAUGHT on 16 and IDENTICAL on the four
    ``1d_metallic`` ones, shape (1, 1, 24) -- the same four the sibling gate
    measured, and MEEP's ``stride(d) = 0`` for a direction it does not have
    showing up in a mutation score. The twin below gives that inertness a NAME so
    it is measured rather than filtered out of sight.
    """
    nx, ny, nz = (int(n) for n in shape)
    index = np.arange(nx * ny * nz)
    c_order = (index // (ny * nz), (index // nz) % ny, index % nz)
    f_order = (index % nx, (index // nx) % ny, index // (nx * ny))
    return any(not np.array_equal(a, b) for a, b in zip(c_order, f_order))


#: THE STRUCTURAL NULL: the same transform under a second name, scored on exactly
#: the shapes where it CANNOT bite, and required to come back UNCAUGHT.
LOCAL_SOURCE_MUTATIONS["fortran_order_index_decomposition_on_a_flat_grid"] = \
    probe.SOURCE_MUTATIONS["fortran_order_index_decomposition"]

#: BUILT AFTER THE LAST ADDITION TO :data:`LOCAL_SOURCE_MUTATIONS`, and that
#: ordering is load-bearing: a name registered below this line is invisible to the
#: battery and raises KeyError mid-run. Measured 2026-08-20 on the GPU host -- a device
#: leg died forty minutes in with exactly that KeyError, which is why
#: ``test_every_named_mutation_resolves_and_arms`` now runs on the laptop.
SOURCE_MUTATIONS_ALL = dict(probe.SOURCE_MUTATIONS, **LOCAL_SOURCE_MUTATIONS)

SOURCE_MUTATION_LEG_FILTER = {
    "drop_metallic_mask": lambda codes, shape: BC_METALLIC in codes,
    "drop_folded_periodic_top_mask":
        lambda codes, shape: BC_MIRROR_PERIODIC in codes,
    "fortran_order_index_decomposition":
        lambda codes, shape: index_decomposition_is_observable(shape),
    "fortran_order_index_decomposition_on_a_flat_grid":
        lambda codes, shape: not index_decomposition_is_observable(shape),
}

#: What each filtered mutation REQUIRES of a leg, in words, for the record. A
#: mutation with NO requirement and no legs is a failure; one WITH a requirement
#: and no legs on a given kernel is recorded as unasked-on-that-kernel, and the
#: verdict then checks it was asked somewhere.
SOURCE_MUTATION_LEG_REQUIREMENT = {
    "drop_metallic_mask": "at least one axis resolving to BC_METALLIC",
    "drop_folded_periodic_top_mask":
        "at least one axis resolving to BC_MIRROR_PERIODIC (a folded PERIODIC axis)",
    "fortran_order_index_decomposition":
        "a shape on which the C-order and Fortran-order index decompositions differ",
    "fortran_order_index_decomposition_on_a_flat_grid":
        "a shape on which the two index decompositions COINCIDE, so the edit "
        "rewrites the block into itself (only a 1-D grid does, and a 1-D grid "
        "cannot reach all four PML subchunk cases, so this has legs on the "
        "no-absorber kernels alone)",
}

_SPEC_CODES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}


def spec_codes_and_shape(spec: Dict[str, Any]):
    """``(codes, shape)`` for a spec's grid, memoized. Built on NumPy: no device needed."""
    label = spec["label"]
    if label not in _SPEC_CODES:
        planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
        grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                    boundaries=tuple(spec["boundaries"]), symmetry=planes,
                    dimensions=spec.get("dimensions", 3), xp=np, courant=0.5)
        codes, refusal = coverage.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise RuntimeError(f"{label}: {refusal}")
        _SPEC_CODES[label] = (tuple(int(c) for c in codes),
                              tuple(int(n) for n in grid.shape))
    return _SPEC_CODES[label]


def leg_carries(mutation: str, spec: Dict[str, Any]) -> bool:
    leg_filter = SOURCE_MUTATION_LEG_FILTER.get(mutation)
    if leg_filter is None:
        return True
    codes, shape = spec_codes_and_shape(spec)
    return bool(leg_filter(codes, shape))


#: ``tolerance_instead_of_exact_predicate`` is expected UNCAUGHT on the shipped
#: table and CAUGHT on the near-one one. It is therefore scored per table, and the
#: verdict clause reads the CONTRAST rather than either number alone.
TABLE_SENSITIVE_MUTATIONS: Tuple[str, ...] = (
    "tolerance_instead_of_exact_predicate",)


# ---------------------------------------------------------------------------
# The host-mutation battery
# ---------------------------------------------------------------------------

def host_mutated_conductivity(name: Optional[str], fields, sub_step: str, mask):
    """Corrupt the conductivity volumes a LAUNCH BINDS, leaving the oracle's alone.

    A host mutation plants its defect on the POINTERS rather than on the fields
    object every other leg shares, which is what makes it a statement about the
    binding rather than about the fixture.
    """
    if name is None:
        return None
    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    live = [t for t, flag in zip(targets, mask) if flag]
    out: Dict[str, Any] = {}
    if name == "identity_rebind":
        for target in live:
            out[f"condfac_{target}"] = fields.condfac_for(target)
            out[f"condinv_{target}"] = fields.condinv_for(target)
        return out
    if name == "swap_condfac_and_condinv_volumes":
        for target in live:
            out[f"condfac_{target}"] = fields.condinv_for(target)
            out[f"condinv_{target}"] = fields.condfac_for(target)
        return out
    if name == "alias_every_component_to_the_first":
        first = live[0]
        for target in live:
            out[f"condfac_{target}"] = fields.condfac_for(first)
            out[f"condinv_{target}"] = fields.condinv_for(first)
        return out
    raise ValueError(f"unknown host mutation {name!r}")


HOST_MUTATIONS: Tuple[str, ...] = ("swap_condfac_and_condinv_volumes",
                                   "alias_every_component_to_the_first",
                                   "identity_rebind")
#: ``identity_rebind`` hands back the SAME volumes and MUST BE UNCAUGHT: it is
#: what says the other two legs measure the substitution rather than the act of
#: passing a ``conductivity`` argument at all.
NULL_HOST_MUTATIONS: Tuple[str, ...] = ("identity_rebind",)
#: ``alias_every_component_to_the_first`` cannot bite where only one component is
#: conductive, and that is a fact about the mask rather than about the kernel.
HOST_MUTATION_NEEDS_TWO_LIVE = ("alias_every_component_to_the_first",)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], sub_step: str, mask,
             e_storage: str, courant: float, value_class: str,
             coefficient_table: str, guard: str, substitution: str,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML`` objects.
    There is no second transcription on the oracle leg to drift: the kernel is
    compared against the thing it claims to reproduce, byte for byte, from ONE
    frozen state -- oracle runs, state is restored, kernel runs.
    """
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    arm = arm_of(spec, sub_step, e_storage)
    kind = layer_kind(spec)
    # SEEDED FROM A DIGEST, NOT FROM hash(). Python salts hash() of a tuple
    # containing strings with PYTHONHASHSEED, so such a gate draws a different
    # fixture every process and a failing case cannot be replayed.
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{sub_step}|{MASK_IDS[mask]}|{e_storage}|{courant}|"
            f"{value_class}|{coefficient_table}".encode()).digest()[:4], "big"))

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "arm": arm, "layer": kind,
        "mask": MASK_IDS[mask], "e_storage": e_storage,
        "absorber": "active" if spec_is_active(spec) else spec["absorber"],
        "coefficient_table": coefficient_table,
        "courant": courant, "value_class": value_class, "guard": guard,
        "backend": backend, "substitution": substitution,
        "host_mutation": host_mutation,
        "fold_axes": spec["axes"], "boundaries": list(spec["boundaries"]),
    }
    if coefficient_table == "near_one" and kind != "active":
        case["skipped"] = ("the near-one coefficient table only exists on a layer; "
                           "the no-absorber tail reads no coefficients")
        case["seconds"] = time.time() - started
        return case
    if kind == "active" and e_storage == "derived":
        case["skipped"] = ("enable_pml_storage stores E unconditionally, so there "
                           "is no derived arm under an active layer")
        case["seconds"] = time.time() - started
        return case

    fields, layer, grid = build(xp, spec, sub_step, mask, courant, e_storage,
                                coefficient_table, rng)
    names = state_names(sub_step, arm, kind, mask)
    host = seed_state(fields, grid, names, value_class, rng)
    dtdx = float(grid.dt / grid.dx)
    codes, resolved, kinds, why_skip = boundary_codes_for(grid, substitution)
    case.update({
        "resolved_codes": None if resolved is None else [int(c) for c in resolved],
        "boundary_codes": None if codes is None else [int(c) for c in codes],
        "resolved_kinds": None if kinds is None else list(kinds),
        "dtdx": dtdx, "structure": structure_facts(grid),
        "operand_census": operand_census(host),
        "outputs": list(outputs(sub_step, arm, kind, mask)),
    })
    if why_skip is not None:
        case["skipped"] = why_skip
        case["seconds"] = time.time() - started
        return case

    # THE PREDICATE'S ANSWER, recorded rather than acted on. The gate exists to
    # decide whether it should be believed, so it must not gate the measurement.
    covered, reason = conductive_kernels.covers_conductive_curl(
        fields, layer, grid, sub_step)
    case["predicate_today"] = {
        "covered": bool(covered), "reason": reason,
        "arm": list(conductive_kernels.conductive_arm(fields, layer, sub_step)),
        "mask": list(conductive_kernels.conductive_targets(fields, sub_step))}

    case["conductivity_profile"] = conductivity_profile(fields, sub_step, mask)
    if not case["conductivity_profile"]["meets_floor"]:
        case["skipped"] = ("the conductivity fixture is degenerate (identity, "
                           "constant, or one volume shared); the conductive "
                           "mutations could not bite on this case")
        case["seconds"] = time.time() - started
        return case

    case["epsilon_profile"] = epsilon_profile(fields, arm)
    if not case["epsilon_profile"]["meets_floor"]:
        case["skipped"] = "the inverse-epsilon fixture is degenerate"
        case["seconds"] = time.time() - started
        return case

    if kind == "active":
        case["pml_case_census"] = pml_case_census(layer, sub_step, tuple(grid.shape),
                                                  mask)
        if not case["pml_case_census"]["meets_floor"]:
            case["skipped"] = (
                "the layer did not put all four of MEEP's subchunk cases into "
                "this volume; a pass would be a statement about the layer, not "
                f"about the kernel: {case['pml_case_census']['reached']}")
            case["seconds"] = time.time() - started
            return case

    frozen = snapshot(fields, names)
    case["sources_are_distinct"] = sources_are_distinct(frozen, sub_step, arm, kind)
    if not case["sources_are_distinct"]["meets_floor"]:
        case["skipped"] = "two source volumes are identical"
        case["seconds"] = time.time() - started
        return case

    curls = unmasked_curls(fields, sub_step, arm, kind, resolved, dtdx)
    case["masked_planes_are_live"] = masked_planes_are_live(curls, sub_step, resolved)

    # Leg 1: the oracle.
    getattr(stepping, sub_step)(fields, layer)
    reference = snapshot(fields, names)
    moved = oracle_moved(frozen, reference, outputs(sub_step, arm, kind, mask))
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    conductivity = host_mutated_conductivity(host_mutation, fields, sub_step, mask)
    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(spec, sub_step, arm, fields, layer, codes, dtdx, None, conductivity)

    written = outputs(sub_step, arm, kind, mask)
    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in written}
    case["single_launch"] = combine(parts)
    case["localization"] = {
        name: localize(reference[name], to_host(getattr(fields, name)), name,
                       codes, resolved) for name in written}
    case["localization_summary"] = {
        key: sum(v[key] for v in case["localization"].values())
        for key in ("differing_words", "differing_on_mask_delta_planes",
                    "differing_elsewhere")}

    # Leg 3: the multi-step, on the LICENSED substitution only.
    if host_mutation is None and substitution == LICENSED_SUBSTITUTION:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            getattr(stepping, sub_step)(fields, layer)
            advance_sources(fields, sub_step, arm, kind)
        oracle_multi = snapshot(fields, names)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(spec, sub_step, arm, fields, layer, codes, dtdx, None, None)
            advance_sources(fields, sub_step, arm, kind)
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in written}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str, guard: str = "fmad_false"):
    """The case plan for one guard.

    SHAPED, NOT A BARE CROSS PRODUCT, and every trim is stated:

    * The LICENSED substitution gets the whole grid -- every spec, both sub-steps,
      all five masks, both courants, both value classes, both coefficient tables
      where they exist. That is the evidence the release verdict rests on.
    * The MISDECLARATION CONTROLS answer "is this gate measuring the boundary
      handling at all", which is a question about the code triple and not about
      the conductivity. They run at ONE mask, one courant, one value class and one
      table -- running them at all five masks would multiply a control that
      already answers its question.
    * The UNGUARDED guard set exists only for the contraction control, which is
      scored at the INEXACT courant (0.5 is exactly representable and cannot
      separate a contracted expression from an uncontracted one), so it runs at
      that courant alone.
    """
    primary = guard == "fmad_false"
    specs = (SPECS if product == "full"
             else tuple(_spec_by_label(label) for label in
                        ("3d_mixed", "3d_mixed_inert_layer", "3d_pml_all_axes")))
    masks = COND_MASKS if product == "full" else (COND_MASKS[0], COND_MASKS[1])
    courants = COURANTS if (product == "full" and primary) else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    out = []
    for spec in specs:
        active = spec_is_active(spec)
        storages = ("stored",) if active else E_STORAGE
        tables = COEFFICIENT_TABLES if active else ("shipped",)
        for sub_step in SUB_STEPS:
            for mask in masks:
                for e_storage in storages:
                    for courant in courants:
                        for value_class in classes:
                            for table in tables:
                                out.append((spec, sub_step, mask, e_storage,
                                            courant, value_class, table,
                                            LICENSED_SUBSTITUTION))
            if product != "full" or not primary:
                continue
            for substitution in SUBSTITUTIONS:
                if substitution == LICENSED_SUBSTITUTION:
                    continue
                out.append((spec, sub_step, COND_MASKS[0], storages[0],
                            INEXACT_COURANT, "uniform", "shipped", substitution))
    return out


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              product: str, guard: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product, guard)
    for index, entry in enumerate(plan, start=1):
        case = one_case(backend, entry[0], entry[1], entry[2], entry[3], entry[4],
                        entry[5], entry[6], guard, entry[7])
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {case['label']} "
                f"{case['sub_step']} {case['mask']} SKIPPED: {case['skipped'][:60]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        summary = case["localization_summary"]
        log(f"[{guard}] case {index}/{len(plan)} {case['label']} {case['sub_step']} "
            f"{case['arm']}/{case['layer']} mask={case['mask']} c={case['courant']} "
            f"{case['value_class']} {case['coefficient_table']} "
            f"{case['substitution']} shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"diff={summary['differing_words']} "
            f"elsewhere={summary['differing_elsewhere']} ({case['seconds']:.1f} s)")
    return cases


#: The legs a mutation is scored on. CHOSEN, not sliced off the front of the case
#: product: a leg taken from the head would land entirely on no-absorber specs and
#: every PML mutation would score 0/0 UNCAUGHT -- a harness defect that reads
#: exactly like a kernel defect.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "3d_metallic", "3d_mixed", "fold_Y_periodic", "2d_metallic_walls",
    "1d_metallic", "3d_pml_two_axes", "2d_pml_two_axes", "2d_pml_mixed_walls",
    "3d_pml_fold_Y", "3d_pml_all_axes",
)

#: The masks a mutation is scored on: the all-lossy one and one single-component
#: one, so a defect that only shows on a MIXED launch has a leg -- and so
#: ``alias_every_component_to_the_first`` has a leg with two live components.
MUTATION_MASKS: Tuple[Tuple[bool, bool, bool], ...] = (COND_MASKS[0], COND_MASKS[1])


def _spec_by_label(label: str) -> Dict[str, Any]:
    for spec in SPECS:
        if spec["label"] == label:
            return spec
    raise KeyError(label)


def scorable_baselines(results: Dict[str, Any]):
    """``(scorable, dead_masks, excluded)`` -- and the two exclusions are DIFFERENT.

    * ``scorable`` is every licensed key whose unmutated baseline was
      bit-identical. On a key that already diverges every mutation scores CAUGHT
      for free, so a mutation leg there answers nothing.
    * ``dead_masks`` is every key on which a plane the kernel ZEROES carries an
      identically zero unmasked curl. There a MASK mutation scores UNCAUGHT for
      free -- but only a mask mutation. Excluding the key from every mutation, as
      the first cut of this gate did, silently removed the only flat-grid leg the
      index-decomposition null had, and the run then failed for "a leg filter
      fired everywhere" -- a harness verdict about a harness decision.

    MEASURED 2026-08-20 on the GPU host: under ``keep`` the ``1d_metallic``/``step_D``
    keys were live and under ``meep_x86_flush`` they were dead, because that leg
    imports MEEP for the host policy and MEEP's build sets the x86 FTZ/DAZ bits --
    so the HOST NumPy that computes this floor flushes the subnormal curl on that
    plane to exactly zero. The floor is right on both legs; scoping it to the
    mutations it protects is what makes the two legs agree.
    """
    scorable, dead_masks, excluded = set(), set(), []
    for case in results.get("sweep", {}).get("fmad_false", []):
        if case["substitution"] != LICENSED_SUBSTITUTION or case.get("skipped"):
            continue
        key = (case["label"], case["sub_step"], case["mask"], case["arm"],
               case["coefficient_table"])
        if not case.get("masked_planes_are_live", {}).get("meets_floor", True):
            dead_masks.add(key)
        if case["single_launch"]["bit_identical"]:
            scorable.add(key)
        else:
            excluded.append({"key": list(key), "why": "baseline diverged"})
    for key in sorted(dead_masks):
        excluded.append({"key": list(key),
                         "why": ("a plane this arm masks carries an identically "
                                 "zero unmasked curl on this grid: excluded from "
                                 "the MASK mutations only")})
    return scorable, dead_masks, excluded


#: The mutations a DEAD MASK makes unscoreable, and only these. Every other
#: mutation edits arithmetic a masked plane has nothing to do with.
MASK_MUTATIONS: Tuple[str, ...] = ("drop_metallic_mask",
                                   "drop_folded_periodic_top_mask")


def mutation_plan(product: str, scorable, dead_masks=frozenset(),
                  mutation: Optional[str] = None):
    """The legs a mutation is scored on.

    ``mutation`` scopes the dead-mask exclusion: a key whose masked plane is dead
    is dropped for a MASK mutation and kept for every other, because "this plane
    carries no curl" says nothing about the conductive tail or the index
    decomposition.
    """
    plan = []
    labels = (MUTATION_SPEC_LABELS if product == "full"
              else ("3d_mixed", "3d_pml_all_axes"))
    for label in labels:
        spec = _spec_by_label(label)
        for sub_step in SUB_STEPS:
            for mask in MUTATION_MASKS:
                storages = ("stored",) if spec_is_active(spec) else E_STORAGE
                tables = (COEFFICIENT_TABLES if spec_is_active(spec)
                          else ("shipped",))
                for e_storage in storages:
                    for table in tables:
                        arm = arm_of(spec, sub_step, e_storage)
                        key = (label, sub_step, MASK_IDS[mask], arm, table)
                        if scorable is not None and key not in scorable:
                            continue
                        if mutation in MASK_MUTATIONS and key in dead_masks:
                            continue
                        plan.append((spec, sub_step, mask, e_storage,
                                     INEXACT_COURANT, "uniform", table))
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
                       product: str, scorable, dead_masks=frozenset()) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable, dead_masks)
    for name in HOST_MUTATIONS:
        legs = []
        for spec, sub_step, mask, e_storage, courant, value_class, table in plan:
            if name in HOST_MUTATION_NEEDS_TWO_LIVE and sum(mask) < 2:
                continue
            legs.append(one_case(backend, spec, sub_step, mask, e_storage, courant,
                                 value_class, table, "fmad_false",
                                 LICENSED_SUBSTITUTION, host_mutation=name))
        scored = _score(legs, name not in NULL_HOST_MUTATIONS)
        out[name] = dict(scored, cases=legs)
        log(f"[host-mut] {name}: caught {scored['caught']}/{scored['ran']} "
            f"-> {scored['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def kernel_for(spec, sub_step: str, arm: str) -> str:
    return conductive_kernels.ARM_KERNELS[(sub_step, layer_kind(spec), arm)]


def run_source_mutations(results: Dict[str, Any], out_path: str, product: str,
                         scorable, dead_masks=frozenset()) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED templates and recompiled.

    The compile memo is keyed through the SOURCE, so a mutated body is a miss and
    reaches NVRTC. Every leg records how many constructions came from the mutated
    bytes, because a leg reporting a pass for a mutation it never applied is worse
    than no leg -- three measured instances on the sibling track.
    """
    from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

    originals = {name: conductive_kernels.kernel_template(name)
                 for name in conductive_kernels.CONDUCTIVE_KERNELS}
    out: Dict[str, Any] = {}

    try:
        for kernel_name in conductive_kernels.CONDUCTIVE_KERNELS:
            for name in SOURCE_MUTATIONS:
                transform = SOURCE_MUTATIONS_ALL[name]
                plan = mutation_plan(product, scorable, dead_masks, name)
                kernel_legs = [entry for entry in plan
                               if kernel_for(entry[0], entry[1],
                                             arm_of(entry[0], entry[1], entry[3]))
                               == kernel_name]
                legs_for_this = [entry for entry in kernel_legs
                                 if leg_carries(name, entry[0])]
                # THE MUTATION IS APPLIED TO THE TEMPLATE, which carries BOTH
                # preprocessor arms; whether the site survives into the compiled
                # body is the mask's business and is recorded per leg.
                mutated, sites = transform(originals[kernel_name])
                key = f"{kernel_name}:{name}"
                required = SOURCE_MUTATION_REQUIRES_LAYER.get(name, "unset")
                kernel_layer = ("active"
                                if kernel_name in conductive_kernels._PML_KERNELS
                                else "none")
                if sites == 0 or mutated == originals[kernel_name]:
                    out[key] = {
                        "armed": False, "sites": sites,
                        "expected_absent": bool(required not in ("unset", None)
                                                and required != kernel_layer),
                        "why": (f"matched {sites} site(s) and changed nothing in "
                                f"{kernel_name}")}
                    log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                if not legs_for_this and name in SOURCE_MUTATION_LEG_FILTER:
                    out[key] = {
                        "armed": False, "sites": sites, "kernel": kernel_name,
                        "expected_absent": True,
                        "legs_in_plan": len(kernel_legs),
                        "legs_carrying_the_code": 0,
                        "leg_requirement": SOURCE_MUTATION_LEG_REQUIREMENT.get(name),
                        "why": (f"the code is present ({sites} site(s)) but no "
                                f"scorable leg on {kernel_name} carries it: it "
                                f"requires "
                                f"{SOURCE_MUTATION_LEG_REQUIREMENT.get(name)}")}
                    log(f"[src-mut] {key}: UNASKED ON THIS KERNEL "
                        f"(0 of {len(kernel_legs)} legs carry the code)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                conductive_kernels.set_kernel_template(kernel_name, mutated)
                conductive_kernels.clear_kernel_cache()
                compile_cache.clear_compile_log()
                # THE COMPILE LOG RECORDS THE SOURCE HANDED TO NVRTC, which is the
                # template WITH its three ``#define COND`` lines -- not the template
                # alone. Digesting the template would count zero constructions from
                # the mutated bytes on every leg and report UNACCOUNTED, which is
                # what one device run did before this was fixed: a battery scoring
                # its own accounting rather than the kernel.
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                built = {hashlib.sha256(
                    conductive_kernels.kernel_source(kernel_name, mask
                                                     ).encode("utf-8")).hexdigest()
                    for mask in COND_MASKS}
                legs = [one_case("cuda", spec, sub, mask, storage, courant,
                                 value_class, table, "fmad_false",
                                 LICENSED_SUBSTITUTION)
                        for spec, sub, mask, storage, courant, value_class, table
                        in legs_for_this]
                conductive_kernels.set_kernel_template(kernel_name,
                                                       originals[kernel_name])
                conductive_kernels.clear_kernel_cache()
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] in built)
                if name in TABLE_SENSITIVE_MUTATIONS:
                    # SCORED PER TABLE. The exact-comparison claim is a CONTRAST:
                    # UNCAUGHT on the shipped table is expected and is evidence of
                    # nothing; CAUGHT on the near-one table is the measurement.
                    by_table = {}
                    for table in COEFFICIENT_TABLES:
                        subset = [c for c in legs
                                  if c.get("coefficient_table") == table]
                        by_table[table] = _score(subset, must_be_caught=False)
                    near = by_table.get("near_one", {})
                    verdict = ("CAUGHT" if near.get("ran") and
                               near.get("caught") == near.get("ran")
                               else "ESCAPED" if near.get("ran") else "NO LEGS")
                    out[key] = {"armed": True, "sites": sites, "kernel": kernel_name,
                                "by_coefficient_table": by_table,
                                "ran": near.get("ran", 0),
                                "caught": near.get("caught", 0),
                                "verdict": verdict,
                                "scored_on": "near_one",
                                "why": ("a real PML table carries no entry within "
                                        "1e-7 of 1.0 that is not exactly 1.0, so "
                                        "the shipped-table legs are expected "
                                        "UNCAUGHT and are recorded, not scored"),
                                "mutated_source_sha256": digest,
                                "kernel_constructions_from_mutated_bytes": from_mutated,
                                "cases": legs}
                else:
                    scored = _score(legs, name not in NULL_SOURCE_MUTATIONS)
                    out[key] = dict(
                        scored, armed=True, sites=sites, kernel=kernel_name,
                        leg_requirement=SOURCE_MUTATION_LEG_REQUIREMENT.get(name),
                        legs_in_plan=len(kernel_legs),
                        legs_carrying_the_code=len(legs_for_this),
                        mutated_source_sha256=digest,
                        kernel_constructions_from_mutated_bytes=from_mutated,
                        cases=legs)
                if from_mutated == 0 and out[key].get("ran"):
                    out[key]["verdict"] = "UNACCOUNTED"
                    out[key]["why"] = ("no kernel construction used the mutated "
                                       "bytes; this leg measured the SHIPPED kernel")
                log(f"[src-mut] {key}: caught {out[key].get('caught')}/"
                    f"{out[key].get('ran')} builds_from_mutated={from_mutated} "
                    f"-> {out[key]['verdict']}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for name, source in originals.items():
            conductive_kernels.set_kernel_template(name, source)
        conductive_kernels.clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]

    arms: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        key = (f"{case['sub_step']}::{case['layer']}::{case['arm']}::"
               f"{case['substitution']}")
        entry = arms.setdefault(key, {
            "cases": 0, "single_identical": 0, "multi_cases": 0,
            "multi_identical": 0, "differing_words": 0,
            "differing_elsewhere": 0, "masks": set(), "labels": set(),
            "tables": set()})
        entry["cases"] += 1
        entry["masks"].add(case["mask"])
        entry["labels"].add(case["label"])
        entry["tables"].add(case["coefficient_table"])
        if case["single_launch"]["bit_identical"]:
            entry["single_identical"] += 1
        if "multi_step" in case:
            entry["multi_cases"] += 1
            if case["multi_step"]["bit_identical"]:
                entry["multi_identical"] += 1
        entry["differing_words"] += case["localization_summary"]["differing_words"]
        entry["differing_elsewhere"] += case["localization_summary"]["differing_elsewhere"]
    for entry in arms.values():
        for field in ("masks", "labels", "tables"):
            entry[field] = sorted(entry[field])
        entry["all_identical"] = (entry["single_identical"] == entry["cases"]
                                  and entry["multi_identical"] == entry["multi_cases"])

    reasons: List[str] = []

    # ALL FIVE ARMS ARE REQUIRED. A run that swept only the no-absorber arms would
    # release on six of the seven reachable corpus slots and say nothing about the
    # seventh, which is the four-case recurrence -- the harder half by far.
    present = {(c["sub_step"], c["layer"], c["arm"]) for c in scored}
    for required in (("step_B", "none", "stored_E"), ("step_B", "none", "derived_E"),
                     ("step_D", "none", "magnetic"), ("step_B", "active", "stored_E"),
                     ("step_D", "active", "magnetic")):
        if required not in present:
            reasons.append(f"the sweep contained no {'/'.join(required)} case")

    # EVERY MASK MUST APPEAR, or "one kernel serves a mixed run" is untested.
    swept_masks = {c["mask"] for c in scored if c["substitution"] == LICENSED_SUBSTITUTION}
    for mask in COND_MASKS:
        if MASK_IDS[mask] not in swept_masks:
            reasons.append(f"conductivity mask {MASK_IDS[mask]} was never swept")

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

    # BOTH ABSORBER SHAPES on the no-PML arms, or "an inert layer steps like no
    # layer" stays an inference about pml.py.
    licensed_absorbers = {c["absorber"] for c in scored
                          if c["substitution"] == LICENSED_SUBSTITUTION}
    for shape in ("none", "inert_layer", "active"):
        if shape not in licensed_absorbers:
            reasons.append(f"no licensed case ran with absorber={shape}")

    # ALL FOUR SUBCHUNK CASES REACHED SOMEWHERE. The per-case floor skips a case
    # that reached fewer; this is what refuses a RUN that never reached one.
    reached = {"A": 0, "B": 0, "C": 0, "D": 0}
    for case in scored:
        census = case.get("pml_case_census")
        if census:
            for name, count in census["reached"].items():
                reached[name] = max(reached[name], count)
    if any(c["layer"] == "active" for c in scored):
        for name, count in reached.items():
            if count == 0:
                reasons.append(f"MEEP subchunk case {name} was never reached by any "
                               f"PML case; that branch of the tail is untested")

    if not results.get("curl_terms_vs_stepping", {}).get("agreed", False):
        reasons.append("the transcribed curl term table disagrees with stepping's")
    if not results.get("conductivity_mask_vs_stepping", {}).get("agreed", False):
        reasons.append("the conductivity mask does not read the attribute "
                       "_apply_curl reads")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is not inert")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            if leg.get("expected_absent"):
                continue
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] == "NO LEGS":
            reasons.append(
                f"source mutation {key} was armed and scored on NO LEGS"
                + (f" (it requires {leg['leg_requirement']}, and no scorable leg "
                   f"had one)" if leg.get("leg_requirement") else "")
                + "; unasked is not inert")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']} "
                           f"({leg.get('caught')}/{leg.get('ran')})")

    # A MUTATION EXCLUDED ON EVERY KERNEL WAS NEVER MEASURED AT ALL. The per-kernel
    # exclusion above is legitimate -- a leg filter is what stops a structurally
    # inert leg from reading as a kernel defect -- but if it fires on EVERY kernel
    # the battery has quietly dropped a defect, which is the failure mode the
    # exclusion machinery itself could otherwise create.
    if results.get("source_mutations"):
        asked: Dict[str, bool] = {}
        for key, leg in results["source_mutations"].items():
            name = key.split(":", 1)[1]
            asked[name] = asked.get(name, False) or bool(leg.get("ran"))
        for name in SOURCE_MUTATION_LEG_FILTER:
            if name in asked and not asked[name]:
                reasons.append(
                    f"source mutation {name} was excluded on EVERY kernel and "
                    f"therefore never measured; a leg filter that fires "
                    f"everywhere has dropped a defect rather than protected one")

    if results.get("backend") == "cuda" and results.get("product") == "full":
        if results.get("skip_mutations"):
            reasons.append("--skip-mutations: no mutation evidence in this record")
        elif not results.get("source_mutations"):
            reasons.append("no source-mutation section was written; the device "
                           "battery did not run")
        if not any(c["substitution"] != LICENSED_SUBSTITUTION for c in scored):
            reasons.append("no misdeclaration control was scored")

    controls: Dict[str, Any] = {}
    for substitution in SUBSTITUTIONS:
        if substitution == LICENSED_SUBSTITUTION:
            continue
        legs = [c for c in scored if c["substitution"] == substitution]
        diverged = [c for c in legs if not c["single_launch"]["bit_identical"]]
        controls[substitution] = {
            "cases": len(legs), "diverged": len(diverged),
            "differing_words": sum(c["localization_summary"]["differing_words"]
                                   for c in legs)}
        if legs and not diverged:
            reasons.append(
                f"the misdeclaration control {substitution} went IDENTICAL on all "
                f"{len(legs)} cases; a deliberately wrong code triple that changes "
                f"nothing means this gate is not measuring the boundary handling")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT
               and c["substitution"] == LICENSED_SUBSTITUTION]
    guarded_identical = {
        (c["label"], c["sub_step"], c["mask"], c["arm"], c["courant"],
         c["value_class"], c["coefficient_table"])
        for c in scored if c["substitution"] == LICENSED_SUBSTITUTION
        and c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["mask"], c["arm"], c["courant"],
                      c["value_class"], c["coefficient_table"]) in guarded_identical]
    control_diverged = [c for c in comparable
                        if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored where "
                    "the guarded leg was identical" if not comparable else
                    "the contraction guard is load-bearing on this family"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here")}

    return {
        "released": not reasons,
        "reasons": reasons,
        "licensed_arms": licensed_keys,
        "licensed_substitution": LICENSED_SUBSTITUTION,
        "misdeclaration_controls": controls,
        "subchunk_cases_reached": reached,
        "arms": arms,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "claim": ("the conductive CUDA curl family -- three no-absorber kernels and "
                  "the four-case conductivity + PML pair -- is byte-identical to "
                  "stepping.step_B / stepping.step_D per sub-step, per arm and per "
                  "per-component conductivity mask, from one frozen state, at one "
                  f"launch and at {MULTI_STEP_BUDGET}"),
        "does_not_claim": [
            "nothing dispatches these kernels; this gate licenses a predicate, "
            "not a wiring",
            "the constitutive sub-steps of a conductive run: fields.condfac_for is "
            "read in _apply_curl and nowhere else, so those belong to the "
            "constitutive families and are untested here",
            "complex storage, cylindrical coordinates, a Bloch phase, BFAST, "
            "special_kz and chi2/chi3 are refused by inherited clauses",
            "no throughput claim: this is a correctness gate and times nothing, "
            "which matters here because the array path's four-case conductive "
            "curl is the engine's most expensive and a correctness result is not "
            "a speed result",
        ],
    }


def verdict_flips_against_planted_defect(results: Dict[str, Any]) -> Dict[str, Any]:
    """A GATE WHOSE VERDICT CANNOT GO RED IS NOT A GATE."""
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
            else verdict["reasons"][0][:160]})
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

    def drop_a_subchunk_case(record) -> bool:
        touched = False
        for case in record.get("sweep", {}).get("fmad_false", []):
            census = case.get("pml_case_census")
            if census and census["reached"].get("D"):
                census["reached"]["D"] = 0
                touched = True
        return touched

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
    plant("a_subchunk_case_was_never_reached", drop_a_subchunk_case,
          "the record carries no PML case census")
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
        "gate": "cuda_conductive",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "skip_mutations": bool(args.skip_mutations),
        "question": ("are the five conductive CUDA curl kernels byte-identical to "
                     "stepping.step_B / stepping.step_D per sub-step, per arm and "
                     "per PER-COMPONENT conductivity mask -- including the "
                     "four-case conductivity + PML recurrence, which the array "
                     "path evaluates over the whole volume and selects with "
                     "copyto and the kernel selects per cell?"),
        "kernels": list(conductive_kernels.CONDUCTIVE_KERNELS),
        "masks_swept": [MASK_IDS[m] for m in COND_MASKS],
        "curl_terms_vs_stepping": check_terms_against_stepping(),
        "conductivity_mask_vs_stepping": check_mask_against_stepping(),
        "kernel_template_sha256": {
            name: hashlib.sha256(
                conductive_kernels.kernel_template(name).encode("utf-8")).hexdigest()
            for name in conductive_kernels.CONDUCTIVE_KERNELS},
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
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
            continue
        if args.backend == "cuda":
            conductive_kernels._COMPILE_OPTIONS = tuple(options)
            conductive_kernels.clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        conductive_kernels._COMPILE_OPTIONS = ("--fmad=false",)
        conductive_kernels.clear_kernel_cache()

    if not args.skip_mutations:
        scorable, dead_masks, excluded = scorable_baselines(results)
        results["mutation_scope"] = {
            "substitution": LICENSED_SUBSTITUTION,
            "scorable": sorted("::".join(k) for k in scorable),
            "dead_masks": sorted("::".join(k) for k in dead_masks),
            "mask_mutations_scoped_out_of_dead_masks": list(MASK_MUTATIONS),
            "excluded": excluded,
            "why": ("a mutation leg answers 'can this gate see this defect', and "
                    "only an arm whose unmutated baseline is bit-identical can "
                    "answer it")}
        log(f"[mut-scope] {len(scorable)} scorable keys; {len(excluded)} excluded")
        save(results, args.out)
        run_host_mutations(results, args.out, args.backend, args.product, scorable,
                           dead_masks)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.product, scorable,
                                 dead_masks)

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
            f"diff={arm['differing_words']} elsewhere={arm['differing_elsewhere']}")
    log(f"[verdict]   subchunk cases reached: {verdict['subchunk_cases_reached']}")
    for plant in results["verdict_flips_against_planted_defect"]["plants"]:
        log(f"[verdict]   plant {plant['plant']}: applicable={plant['applicable']} "
            f"flipped={plant['flipped']}"
            + ("" if plant["applicable"] else f" ({plant['why_not_applicable']})"))
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
