"""Sub-step byte-identity gate for the CONDUCTIVE CUDA curl family.

THE QUESTION
============
``meep_gpu/cuda_kernels/conductive_kernels.py`` carries five kernels for the two
tails of ``stepping._apply_curl`` that read ``fields.condfac_for(term.target)``
(stepping.py:548) -- the no-absorber conductive update (:584-585) and MEEP's
conductive forms under a PML, chosen per structure chunk (:554-580). This gate asks,
per sub-step, per arm and PER CONDUCTIVITY MASK, whether
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

THE FORMS ARE CHOSEN PER STRUCTURE CHUNK, AND EACH CHOICE IS DATA
=================================================================
Since 0.10 the kernels step MEEP's forms (bit_parity_0_10 KERNELS.md section 1): the
curl in MEEP's grouping with its partner-presence forms, and, under a layer, the form
of each conductive component chosen per structure chunk from the chunk record ``cb``
(``chunk_map.chunk_bounds_record``: which cells' chunks carry the dsig and dsigu
layers) and from which chunks keep the component's conductivity
(``RegionPlan.conductive_chunks``: a chunk whose every sample is zero keeps none and
steps the non-conductive forms). Every case hands the kernel the record of its own
one-rank division (:func:`case_record`), the division the oracle's region plan steps
with, and the NumPy backend reads the same record. The two ``half_sigma`` specs put
the conductivity on the high half of the first layered axis only, so the chunks of
that axis's low layer keep none of it and their dsig cells step the non-conductive
form while the rest step the conductive ones: ``pml_case_census`` refuses such a case
unless it reaches both kinds of chunk. The coefficient-identity predicate this family
carried until 0.10 (``kms != 1``) is gone, and with it the near-one table that armed
its tolerance mutation (DESIGN.md D13: retired, not inverted); the B battery
(KERNELS.md section 7.2) arms the membership instead -- the predicate on coefficient
identity, a bound off by one, the two lattices' records swapped, the flags swapped,
an auxiliary stored on a plain cell, the conductive chunks ignored -- and the
increment's old grouping is the inverted must-catch (section 7.1).

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
import itertools
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
import gate_chunk_records  # noqa: E402
import gate_needles  # noqa: E402
import gate_provenance  # noqa: E402
from gate_needles import Needle  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import (RECORD_PRESENCE_WORD,  # noqa: E402
                                membership_from_record, record_index)
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

#: Where a spec's conductivity lives. ``everywhere``: drawn on every cell, so every
#: chunk keeps it. ``half_space``: the high half of the first layered axis only, so
#: the chunks of that axis's low layer keep none of it (``RegionPlan
#: .conductive_chunks``) and their dsig cells step the non-conductive form, while the
#: other chunks step all four conductive forms -- MEEP's per-chunk choice, which a
#: fixture with the conductivity everywhere never reaches.
SIGMA_LAYOUTS: Tuple[str, ...] = ("everywhere", "half_space")

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
    # THE CONDUCTIVITY ON A HALF-SPACE (``sigma``: ``half_space``). The chunks of the
    # x layer's low face keep none of it, so their dsig cells step the non-conductive
    # forms while every other chunk steps the conductive ones -- the configuration of
    # the corpus's conductive-PML rows (a lossy block, or an adjoint design region,
    # under a PML), where a layer's chunks keep no conductivity.
    {"label": "3d_pml_all_axes_half_sigma", "cell": (12.0, 13.0, 14.0),
     "dimensions": 3, "boundaries": ("metallic", "periodic", "metallic"), "axes": "",
     "phase": 1, "absorber": ((2, 2), (2, 2), (2, 2)), "sigma": "half_space"},
    {"label": "2d_pml_two_axes_half_sigma", "cell": (14.0, 16.0, 0.0),
     "dimensions": 2, "boundaries": ("metallic", "metallic", "periodic"), "axes": "",
     "phase": 1, "absorber": ((3, 3), (3, 3), (0, 0)), "sigma": "half_space"},
)


def sigma_layout(spec: Dict[str, Any]) -> str:
    """Where the spec's conductivity lives (:data:`SIGMA_LAYOUTS`)."""
    return spec.get("sigma", "everywhere")


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
          e_storage: str, rng):
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

    if layer is not None and layer.is_active:
        fields.enable_pml_storage()
    elif e_storage == "stored":
        fields.enable_field_storage()

    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    box = sigma_box(spec, tuple(int(n) for n in grid.shape))
    sigma = {}
    for component, flag in zip(targets, mask):
        if flag:
            values = rng.uniform(0.05, 0.9, size=grid.shape).astype(np.float32)
            if box is not None:
                values = np.where(box, values, np.float32(0.0)).astype(np.float32)
            sigma[component] = xp.asarray(np.ascontiguousarray(values))
    setter = (fields.set_b_conductivity if sub_step == "step_B"
              else fields.set_d_conductivity)
    setter(sigma)
    return fields, layer, grid


def sigma_box(spec: Dict[str, Any], shape) -> Optional[np.ndarray]:
    """The cells a ``half_space`` spec's conductivity occupies; None for ``everywhere``.

    The rows at or above the middle of the first axis that carries a layer face, every
    row of the other axes. The chunks of that axis's low layer allocate only rows
    below its thickness plus one (their owned samples plus one,
    ``ChunkLayout.loop_rows``), so they keep no conductivity, and every other chunk
    keeps it; ``pml_case_census`` measures both rather than trusting this.
    """
    if sigma_layout(spec) != "half_space":
        return None
    for axis, (low, high) in enumerate(spec["absorber"]):
        n = int(shape[axis])
        if n <= 1 or (int(low) == 0 and int(high) == 0):
            continue
        line = np.zeros(n, dtype=bool)
        line[n // 2:] = True
        if n // 2 <= int(low) + 1:
            raise ValueError(f"{spec['label']}: the half-space reaches the low layer's "
                             f"chunk along axis {axis}")
        selector = [None, None, None]
        selector[axis] = slice(None)
        return np.broadcast_to(line[tuple(selector)], tuple(shape)).copy()
    raise ValueError(f"{spec['label']}: a half-space conductivity needs a layered axis")


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


def case_record(grid, layer) -> np.ndarray:
    """The chunk record of the case's own division: the one-rank division of ``grid``
    with ``layer`` (None, the inert layer, or the active one).

    It is the division the oracle steps with: ``stepping``'s first PML sub-step plans
    the one-rank layout of the grid and layer when no driver installed another
    (``stepping._region_plan``). The no-absorber kernels read its partner-presence
    word only; the PML pair also its membership runs.
    """
    return np.ascontiguousarray(gate_chunk_records.layer_record(grid, layer),
                                dtype=np.int32)


def record_presence(record) -> Tuple[bool, bool, bool]:
    """The three partner-presence bits of a chunk record, as the kernels decode them."""
    word = int(np.asarray(record)[RECORD_PRESENCE_WORD])
    return tuple(bool((word >> axis) & 1) for axis in range(3))


def membership(record, target: str, axis: int, shape) -> np.ndarray:
    """``target``'s membership flag of ``axis``'s layer, as the kernel decodes it.

    On ``target``'s own Yee sub-lattice along ``axis`` (``conductive_kernels
    .membership_flags`` names the same flag), from the record's runs, broadcast to the
    field's shape.
    """
    half = IYEE_SHIFTS[target][axis] == 1
    line = membership_from_record(record, half, axis, int(shape[axis]))
    view = [1, 1, 1]
    view[axis] = int(shape[axis])
    return np.broadcast_to(np.asarray(line, dtype=bool).reshape(view), tuple(shape))


def run_kernel_cuda(spec, sub_step: str, arm: str, fields, layer, codes,
                    dtdx: float, record, inverse_epsilon=None, conductivity=None,
                    keeps=None) -> None:
    """The SHIPPED kernel, through its own launch wrapper, with codes and record handed in.

    ``record`` is the case's host chunk record (:func:`case_record`), uploaded here as
    the one contiguous ``int32[25]`` device vector the launcher requires. ``keeps`` is
    a host mutation's corrupted conductive-chunk binding, or None for the run's own.
    """
    conductive_kernels.step_conductive_curl(
        fields, layer, sub_step, tuple(np.int32(c) for c in codes), dtdx,
        tables=curl_tables(layer, sub_step),
        arm=(layer_kind(spec), arm), inverse_epsilon=inverse_epsilon,
        conductivity=conductivity,
        chunk_bounds=cp.asarray(np.ascontiguousarray(record, dtype=np.int32)),
        keeps=keeps)
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


def _curl_numpy(f1: np.ndarray, sf: np.ndarray, f2: np.ndarray, ss: np.ndarray,
                scale: np.float32, first_present: bool,
                second_present: bool) -> np.ndarray:
    """``curl_stencil`` from the certified prelude, transcribed in float32.

    MEEP's ``step_curl`` (step_generic.cpp:72-76, :104-108), as C associates it: both
    partners ``dtdx * (((sf - f1) + f2) - ss)``; the second absent ``dtdx * (sf - f1)``;
    the first absent ``(-dtdx) * (ss - f2)``; neither, an exact zero.
    """
    if first_present and second_present:
        return (scale * (((sf - f1) + f2) - ss)).astype(np.float32)
    if first_present:
        return (scale * (sf - f1)).astype(np.float32)
    if second_present:
        return ((-scale) * (ss - f2)).astype(np.float32)
    return np.zeros_like(f1, dtype=np.float32)


def unmasked_curls(fields, sub_step: str, arm: str, layer: str, codes,
                   dtdx: float, record, inverse_epsilon=None) -> Dict[str, np.ndarray]:
    """Each term's curl BEFORE the wall/fold masks, on the host, from the current state.

    The partners' presence is the record's presence word, as the kernel reads it.
    """
    shift = _shift_dn_numpy if arm_arrays(sub_step, arm, layer)["backward"] \
        else _shift_up_numpy
    scale = np.float32(dtdx)
    presence = record_presence(record)
    operands = _operand_volumes(fields, sub_step, arm, layer, inverse_epsilon)
    out = {}
    for target, first, first_axis, second, second_axis in CURL_TERMS[sub_step]:
        f1, f2 = operands[first], operands[second]
        sf = shift(f1, first_axis, codes[first_axis])
        ss = shift(f2, second_axis, codes[second_axis])
        out[target] = _curl_numpy(f1, sf, f2, ss, scale, presence[first_axis],
                                  presence[second_axis])
    return out


def run_kernel_numpy(spec, sub_step: str, arm: str, fields, layer, codes,
                     dtdx: float, record, inverse_epsilon=None, conductivity=None,
                     keeps=None) -> None:
    """The device tree, transcribed, in float32 -- the laptop backend.

    THE GROUPING IS THE SHIPPED ONE and it is load-bearing everywhere: MEEP's stencil
    with its presence forms (``curl_stencil``), ``((f * cf) - curl) * ci`` in the
    no-absorber tail, and ``cond_pml_apply``'s forms with the increment
    ``(c' - c)`` formed first.

    THE FORM IS SELECTED PER CELL FROM THE SAME DATA THE KERNEL READS: the dsig and
    dsigu flags decoded from ``record`` on the target's own sub-lattice
    (:func:`membership`), and which chunks keep the conductivity from the launcher's
    own binding (``conductive_kernels.conductive_keeps``) or a host mutation's
    ``keeps``. Where a chunk keeps none, the certified ``pml_apply``: the recurrence
    where dsigu, ``((f * km1) - curl) * si1`` elsewhere. Auxiliaries keep their words
    off their forms' cells.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING.
    """
    kind = layer_kind(spec)
    curls = unmasked_curls(fields, sub_step, arm, kind, codes, dtdx, record,
                           inverse_epsilon)
    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    supplied = conductivity or {}
    keep, masks = (True, True, True), (None, None, None)
    if kind == "active":
        keep, masks = (keeps if keeps is not None
                       else conductive_kernels.conductive_keeps(fields, layer, sub_step))
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
        dsig = membership(record, target, first_axis, field.shape)
        dsigu = membership(record, target, second_axis, field.shape)
        # pml_apply: the recurrence on dsigu cells, the dsig-only form elsewhere.
        plain_aux = (((aux * km1) - curl) * si1).astype(np.float32)
        plain = np.where(
            dsigu, ((((field * km2) + plain_aux) - aux) * si2),
            (((field * km1) - curl) * si1)).astype(np.float32)
        plain_aux = np.where(dsigu, plain_aux, aux).astype(np.float32)
        if condfac is None:
            getattr(fields, target)[...] = fields.grid.xp.asarray(plain)
            getattr(fields, "fu_" + target)[...] = fields.grid.xp.asarray(plain_aux)
            continue
        cf = np.asarray(to_host(condfac), dtype=np.float32)
        ci = np.asarray(to_host(condinv), dtype=np.float32)
        history = np.asarray(to_host(getattr(fields, "f_cond_" + target)),
                             dtype=np.float32)
        kept = (np.ones(field.shape, dtype=bool) if keep[index]
                else np.asarray(to_host(masks[index]), dtype=bool))
        c_new = (((history * cf) - curl) * ci).astype(np.float32)
        increment = (c_new - history).astype(np.float32)
        u_new = np.where(dsig, (((aux * km1) + increment) * si1),
                         (((aux * cf) - curl) * ci)).astype(np.float32)
        value = np.where(
            dsigu, ((((field * km2) + u_new) - aux) * si2),
            np.where(dsig, (((field * km1) + increment) * si1),
                     (((field * cf) - curl) * ci))).astype(np.float32)
        new = np.where(kept, value, plain).astype(np.float32)
        new_aux = np.where(kept, np.where(dsigu, u_new, aux), plain_aux).astype(np.float32)
        new_history = np.where(kept & dsig, c_new, history).astype(np.float32)
        getattr(fields, target)[...] = fields.grid.xp.asarray(new)
        getattr(fields, "fu_" + target)[...] = fields.grid.xp.asarray(new_aux)
        getattr(fields, "f_cond_" + target)[...] = fields.grid.xp.asarray(new_history)


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


def pml_case_census(fields, layer, sub_step: str, shape, mask, record,
                    sigma: str = "everywhere") -> Dict[str, Any]:
    """How many cells each of MEEP's four conductive forms claims, per target.

    THE NON-VACUITY FLOOR THIS FAMILY NEEDS AND THE LOSSLESS ONES DO NOT. The tail is
    four forms and a two-level selection, chosen per structure chunk; a fixture whose
    layer puts every cell in one would exercise one of them, and a green run would be
    a statement about the layer. The floor is that every form is reached on at least
    one CONDUCTIVE target -- a lossless target takes the plain forms and its census
    says nothing about this tail. The forms are counted on the chunks that keep the
    conductivity, from the record's membership runs, as the kernel chooses them.

    A ``half_space`` spec must also reach the dsig cells of a chunk that keeps NO
    conductivity (``unkept_dsig``), where the non-conductive form stores a different
    word; that is what the fixture exists for.
    """
    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    keep, masks = conductive_kernels.conductive_keeps(fields, layer, sub_step)
    per_target: Dict[str, Dict[str, int]] = {}
    for index, (target, flag) in enumerate(zip(targets, mask)):
        if not flag:
            continue
        first_axis, second_axis = COEFFICIENT_AXES[index]
        dsig = membership(record, target, first_axis, shape)
        dsigu = membership(record, target, second_axis, shape)
        kept = (np.ones(shape, dtype=bool) if keep[index]
                else np.asarray(to_host(masks[index]), dtype=bool))
        per_target[target] = {"A": int((kept & dsig & dsigu).sum()),
                              "B": int((kept & ~dsig & dsigu).sum()),
                              "C": int((kept & dsig & ~dsigu).sum()),
                              "D": int((kept & ~dsig & ~dsigu).sum()),
                              "unkept_dsig": int((~kept & dsig).sum()),
                              "unkept": int((~kept).sum())}
    reached = {case: max((counts[case] for counts in per_target.values()), default=0)
               for case in ("A", "B", "C", "D")}
    unkept_dsig = max((counts["unkept_dsig"] for counts in per_target.values()),
                      default=0)
    meets = all(count > 0 for count in reached.values())
    if sigma == "half_space":
        meets = meets and unkept_dsig > 0
    return {"per_target": per_target, "reached": reached,
            "unkept_dsig_reached": unkept_dsig, "sigma": sigma,
            "meets_floor": meets}


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


def _flatten_split_grouping(source: str) -> Tuple[str, int]:
    """``(((fv*km2) + u_new) - uv)`` flattened to ``((fv*km2) + (u_new - uv))``.

    The array path accumulates in place, one operation at a time
    (stepping.py:2631-2634), and float addition is not associative. MUST BE CAUGHT.
    """
    needle = "dsigu ? ((((fv * km2) + u_new) - uv) * si2)"
    return (source.replace(needle, "dsigu ? (((fv * km2) + (u_new - uv)) * si2)"),
            source.count(needle))


def _increment_regrouped_in_the_auxiliary(source: str) -> Tuple[str, int]:
    """INVERTED (KERNELS.md section 7.1): ``((u*km1) + (c' - c))`` -> ``((u*km1) + c') - c``.

    MEEP forms the increment first and then adds it (step_generic.cpp:209), and so
    does the array path (stepping.py:2615-2628); the two accumulations left to right
    are what this family carried until 0.10 and a different float32 number. Reached
    on the cells whose chunk carries dsig and dsigu. MUST BE CAUGHT.
    """
    needle = "float u_new = dsig ? (((uv * km1) + inc) * si1)"
    return (source.replace(needle,
                           "float u_new = dsig ? ((((uv * km1) + c_new) - cv) * si1)"),
            source.count(needle))


def _increment_regrouped_in_the_dsig_only_form(source: str) -> Tuple[str, int]:
    """INVERTED: the dsig-only form's ``((f*km1) + (c' - c))`` -> ``((f*km1) + c') - c``.

    step_generic.cpp:167; the array path's ``dsig_form`` (stepping.py:2636-2640). The
    old package grouping, on the cells whose chunk carries dsig and not dsigu. MUST BE
    CAUGHT.
    """
    needle = ": (dsig ? (((fv * km1) + inc) * si1)"
    return (source.replace(needle, ": (dsig ? ((((fv * km1) + c_new) - cv) * si1)"),
            source.count(needle))


def _predicate_on_coefficient_identity(source: str) -> Tuple[str, int]:
    """The form chosen by ``kms != 1`` instead of chunk membership. MUST BE CAUGHT.

    KERNELS.md section 7.2: the predicate this family carried until 0.10. A cell
    inside a layer's chunk whose own sigma is zero (its coefficients exactly 1) takes
    a different form, and so does one whose coefficient differs from 1 outside every
    chunk that carries the layer.
    """
    needle = "    float cfv = cf[idx];\n    float civ = ci[idx];\n"
    return (source.replace(
        needle, "    dsig = (km1 != 1.0f) || (si1 != 1.0f);\n"
                "    dsigu = (km2 != 1.0f) || (si2 != 1.0f);\n" + needle),
        source.count(needle))


def _swap_membership_flags(source: str) -> Tuple[str, int]:
    """The dsig and dsigu flags exchanged at the call site. MUST BE CAUGHT.

    KERNELS.md section 7.2: dsig-only against dsigu-only, on every cell whose chunk
    carries exactly one of the two layers.
    """
    pattern = (r"(cond_pml_apply\([^;]*?sinv_[xyz]\[[ijk]\], )"
               r"(in_[ih]_[xyz]), (in_[ih]_[xyz]), (KEPT[012])\)")
    return (re.subn(pattern, r"\1\3, \2, \4)", source)[0],
            len(re.findall(pattern, source)))


def _conductive_chunks_ignored(source: str) -> Tuple[str, int]:
    """Every chunk treated as keeping the conductivity. MUST BE CAUGHT on a half_space spec.

    MEEP steps the non-conductive forms in a chunk that keeps no conductivity
    (structure.cpp:883-900; ``RegionPlan.conductive_chunks``). On the dsig cells of
    such a chunk the conductive form runs the ``f_cond`` stage, a different word.
    """
    out, count = source, 0
    for index in range(3):
        needle = f"#define KEPT{index} (ck{index}[idx] != 0)\n"
        count += out.count(needle)
        out = out.replace(needle, f"#define KEPT{index} 1\n")
    return out, count


def _auxiliary_load_unguarded(source: str) -> Tuple[str, int]:
    """A DECLARED NULL (KERNELS.md section 7.2): the ``u`` load issued on every cell.

    The store stays guarded, and the loaded word feeds only the forms of the cells
    whose chunk carries dsigu, so no stored word changes. Pinned instead by the static
    needle in the module's own test. MUST BE UNCAUGHT.
    """
    needle = "    float uv = dsigu ? u[idx] : 0.0f;\n"
    return source.replace(needle, "    float uv = u[idx];\n"), source.count(needle)


def _history_load_unguarded(source: str) -> Tuple[str, int]:
    """A DECLARED NULL: the ``f_cond`` load issued on every cell, the store guarded.

    The loaded word feeds only the dsig forms. MUST BE UNCAUGHT.
    """
    needle = "    float cv = dsig ? c[idx] : 0.0f;\n"
    return source.replace(needle, "    float cv = c[idx];\n"), source.count(needle)


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
    """``dsig`` given precedence over ``dsigu`` in the field's selection. MUST BE CAUGHT.

    MEEP's precedence is ``dsigu`` first (stepping.py:2648-2649 copies the dsig-only
    form only where ``dsigu`` is INACTIVE). Taking the dsig-only form on the cells
    whose chunk carries both layers is wrong only inside the two-axis overlap -- a
    corner region, which is exactly where a reader would not look.
    """
    needle = "    float value = dsigu ? ((((fv * km2) + u_new) - uv) * si2)"
    return (source.replace(
        needle, "    float value = (dsigu && !dsig) ? ((((fv * km2) + u_new) - uv) * si2)"),
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
               r"(kms_[xyz]\[[ijk]\], sinv_[xyz]\[[ijk]\]), ")
    return re.subn(pattern, r"\1\3, \2, ", source)[0], \
        len(re.findall(pattern, source))


def _cond_pml_scale_commutes(source: str) -> Tuple[str, int]:
    """A NULL: ``cfv * cv`` for ``cv * cfv`` in the history expression.

    Paired with :func:`_drop_history_store` and :func:`_flatten_split_grouping` on
    the same helper, both of which MUST be caught -- so the pair says the
    comparator is sensitive to the ASSOCIATION of these expressions and
    insensitive to the operand order of a multiply, which is what IEEE-754 says.
    MUST BE UNCAUGHT.
    """
    needle = "float c_new = ((cv * cfv) - curl) * civ;"
    return (source.replace(needle, "float c_new = ((cfv * cv) - curl) * civ;"),
            source.count(needle))


LOCAL_SOURCE_MUTATIONS: Dict[str, Any] = {
    "drop_condfac": _drop_condfac,
    "drop_condinv": _drop_condinv,
    "swap_condfac_condinv": _swap_condfac_condinv,
    "cond_operand_order": _cond_operand_order,
    "flatten_split_grouping": _flatten_split_grouping,
    "increment_regrouped_in_the_auxiliary": _increment_regrouped_in_the_auxiliary,
    "increment_regrouped_in_the_dsig_only_form":
        _increment_regrouped_in_the_dsig_only_form,
    "predicate_on_coefficient_identity": _predicate_on_coefficient_identity,
    "swap_membership_flags": _swap_membership_flags,
    "conductive_chunks_ignored": _conductive_chunks_ignored,
    "auxiliary_load_unguarded": _auxiliary_load_unguarded,
    "history_load_unguarded": _history_load_unguarded,
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
    "flatten_split_grouping",
    "increment_regrouped_in_the_auxiliary",
    "increment_regrouped_in_the_dsig_only_form",
    "predicate_on_coefficient_identity",
    "swap_membership_flags",
    "conductive_chunks_ignored",
    "drop_history_store",
    "store_history_unconditionally",
    "store_auxiliary_unconditionally",
    "swap_case_precedence",
    "swap_cond_dsig_dsigu",
    "cond_pml_scale_commutes",
    "auxiliary_load_unguarded",
    "history_load_unguarded",
    # MEEP's partner-presence forms of the stencil, and their structural nulls.
    "full_bracket_over_an_absent_partner",
    "full_bracket_over_an_absent_partner_on_a_resolved_grid",
    "regroup_stencil_where_no_component_has_both_partners",
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
    # The auxiliary contract's two loads, each paired with the store of the same
    # word (store_auxiliary_unconditionally, store_history_unconditionally).
    "auxiliary_load_unguarded",
    "history_load_unguarded",
    # The presence forms' structural nulls, paired with full_bracket_over_an_absent_
    # partner and regroup_stencil, which must be caught on the shapes that reach them.
    "full_bracket_over_an_absent_partner_on_a_resolved_grid",
    "regroup_stencil_where_no_component_has_both_partners",
)

#: Mutations that only exist in one of the two tails. A leg where the site is
#: absent is recorded as NOT ARMED with ``expected_absent`` True rather than
#: scored, so "unasked" and "inert" stay different findings.
SOURCE_MUTATION_REQUIRES_LAYER = {
    "drop_condfac": "none", "drop_condinv": "none",
    "cond_operand_order": "none",
    "flatten_split_grouping": "active", "drop_history_store": "active",
    "increment_regrouped_in_the_auxiliary": "active",
    "increment_regrouped_in_the_dsig_only_form": "active",
    "predicate_on_coefficient_identity": "active",
    "swap_membership_flags": "active", "conductive_chunks_ignored": "active",
    "auxiliary_load_unguarded": "active", "history_load_unguarded": "active",
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

#: MEEP's one-partner stencil form, defeated: the second-partner-absent arm of
#: ``curl_stencil`` takes the full bracket over the absent partner, which rounds where
#: MEEP's form returns the one difference (step_generic.cpp:108). Reached only where
#: some axis is invariant. MUST BE CAUGHT; scored as a NULL on the shapes with every
#: axis resolved, where the arm is never taken.
_FULL_BRACKET_OVER_AN_ABSENT_PARTNER = Needle(
    find="if (pr1) return dtdx * (sf - f1);",
    replace="if (pr1) return dtdx * (((sf - f1) + f2) - ss);",
    why=("MEEP's g2-absent form dtdx * (g1s - g1) replaced by the full bracket over "
         "the absent partner. Reached only on a grid with an invariant axis. MUST BE "
         "CAUGHT."))
LOCAL_SOURCE_MUTATIONS["full_bracket_over_an_absent_partner"] = \
    _FULL_BRACKET_OVER_AN_ABSENT_PARTNER
LOCAL_SOURCE_MUTATIONS["full_bracket_over_an_absent_partner_on_a_resolved_grid"] = \
    _FULL_BRACKET_OVER_AN_ABSENT_PARTNER

#: THE STENCIL GROUPING DEFECT'S STRUCTURAL NULL. Every kernel here carries the
#: certified pair's ``curl_stencil`` verbatim, the text the probe's shared default
#: spells ``regroup_stencil`` and its null on (both borrowed): the defect is inverted
#: -- it plants the pairwise grouping these kernels carried until 0.10 -- and stays
#: must-catch where some component has both partners; on a grid where none has, the
#: same transform is scored as a NULL under this name.
LOCAL_SOURCE_MUTATIONS["regroup_stencil_where_no_component_has_both_partners"] = \
    probe.SOURCE_MUTATIONS["regroup_stencil"]

#: BUILT AFTER THE LAST ADDITION TO :data:`LOCAL_SOURCE_MUTATIONS`, and that
#: ordering is load-bearing: a name registered below this line is invisible to the
#: battery and raises KeyError mid-run. Measured 2026-08-20 on the GPU host -- a device
#: leg died forty minutes in with exactly that KeyError, which is why
#: ``test_every_named_mutation_resolves_and_arms`` now runs on the laptop.
#: THIS GATE'S NEEDLE TABLE: every armed name, this family's own spellings from
#: :data:`LOCAL_SOURCE_MUTATIONS` and the shared curl defects borrowed from the
#: probe's shared default (``gate_needles.py``). A shared defect this gate
#: re-spells for its own kernels becomes a local entry and moves no other gate.
SOURCE_MUTATIONS_ALL = gate_needles.gate_table(
    "gate_cuda_conductive", probe.SOURCE_MUTATIONS, SOURCE_MUTATIONS,
    own=LOCAL_SOURCE_MUTATIONS)

SOURCE_MUTATION_LEG_FILTER = {
    "drop_metallic_mask": lambda codes, shape: BC_METALLIC in codes,
    "drop_folded_periodic_top_mask":
        lambda codes, shape: BC_MIRROR_PERIODIC in codes,
    "fortran_order_index_decomposition":
        lambda codes, shape: index_decomposition_is_observable(shape),
    "fortran_order_index_decomposition_on_a_flat_grid":
        lambda codes, shape: not index_decomposition_is_observable(shape),
    # THE PRESENCE FORMS. A one-cell axis of these specs is exactly an invariant one,
    # so the shape says which partners are present: the one-partner arm is reached
    # only where an axis is invariant, the both-partner form only where two axes are
    # resolved.
    #
    # SCORED ON A GRID WITH TWO RESOLVED AXES, MEASURED 2026-10-08 on the GPU host: on
    # the 1-D (1, 1, 24) spec the one-partner arm is reached by a line of 24 cells, and
    # the bracket ``(a + g2) - g2`` rounds away from ``a`` on 0 to 2 of them per launch;
    # one 1-D leg (step_D_no_pml_conductive, the derived-storage seed) stored the same
    # words at 1 and at 60 launches. On a 2-D grid the arm is a plane of cells and every
    # leg diverged (11 to 17 words at one launch).
    "full_bracket_over_an_absent_partner":
        lambda codes, shape: (any(int(n) == 1 for n in shape)
                              and sum(int(n) > 1 for n in shape) >= 2),
    "full_bracket_over_an_absent_partner_on_a_resolved_grid":
        lambda codes, shape: all(int(n) > 1 for n in shape),
    "regroup_stencil":
        lambda codes, shape: sum(int(n) > 1 for n in shape) >= 2,
    "regroup_stencil_where_no_component_has_both_partners":
        lambda codes, shape: sum(int(n) > 1 for n in shape) < 2,
}

#: Mutations scored only on the specs whose conductivity lives in an interior box: the
#: per-chunk keep is observable only where some chunk keeps none.
SOURCE_MUTATION_SIGMA_FILTER = {
    "conductive_chunks_ignored": "half_space",
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
    "full_bracket_over_an_absent_partner":
        "an invariant (one-cell) axis on a grid with two resolved axes (a plane of cells "
        "takes the one-partner form; on the 1-D spec it is a line of 24)",
    "full_bracket_over_an_absent_partner_on_a_resolved_grid":
        "every axis resolved (no partner is ever absent)",
    "regroup_stencil": "two or more resolved axes (some component has both partners)",
    "regroup_stencil_where_no_component_has_both_partners":
        "fewer than two resolved axes (no component has both partners)",
    "conductive_chunks_ignored":
        "a conductivity on a half-space, so some chunk keeps none of it",
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
    required = SOURCE_MUTATION_SIGMA_FILTER.get(mutation)
    if required is not None and sigma_layout(spec) != required:
        return False
    leg_filter = SOURCE_MUTATION_LEG_FILTER.get(mutation)
    if leg_filter is None:
        return True
    codes, shape = spec_codes_and_shape(spec)
    return bool(leg_filter(codes, shape))


# ---------------------------------------------------------------------------
# The host-mutation battery
# ---------------------------------------------------------------------------

def host_mutated_conductivity(name: Optional[str], fields, sub_step: str, mask):
    """Corrupt the conductivity volumes a LAUNCH BINDS, leaving the oracle's alone.

    A host mutation plants its defect on the POINTERS rather than on the fields
    object every other leg shares, which is what makes it a statement about the
    binding rather than about the fixture.
    """
    if name is None or name in RECORD_HOST_MUTATIONS or name == "conductive_chunks_unbound":
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


def host_mutated_record(name: Optional[str], record: np.ndarray, grid) -> Optional[np.ndarray]:
    """Corrupt the chunk record a LAUNCH BINDS, leaving the oracle's division alone.

    KERNELS.md section 7.2: ``record_bound_off_by_one`` moves each layered axis's low
    run end one row up on both lattices (the boundary plane changes form);
    ``record_lattices_swapped`` hands each lattice the other's runs (the boundary plane
    moves by one cell). None for any other name.
    """
    if name not in RECORD_HOST_MUTATIONS:
        return None
    out = np.array(record, dtype=np.int32, copy=True)
    for axis in range(3):
        if int(grid.shape[axis]) <= 1:
            continue
        for half in (False, True):
            a0 = out[record_index(half, axis, 0)]
            a1 = out[record_index(half, axis, 1)]
            if name == "record_bound_off_by_one" and a1 > a0:
                out[record_index(half, axis, 1)] = a1 + 1
        if name == "record_lattices_swapped":
            for word in range(4):
                low = record_index(False, axis, word)
                high = record_index(True, axis, word)
                out[low], out[high] = record[high], record[low]
    return out


def host_mutated_keeps(name: Optional[str], fields, layer, sub_step: str):
    """``conductive_chunks_unbound``: the launch told every chunk keeps the conductivity."""
    if name != "conductive_chunks_unbound":
        return None
    return ((True, True, True), (None, None, None))


HOST_MUTATIONS: Tuple[str, ...] = ("swap_condfac_and_condinv_volumes",
                                   "alias_every_component_to_the_first",
                                   "identity_rebind",
                                   "record_bound_off_by_one",
                                   "record_lattices_swapped",
                                   "conductive_chunks_unbound")
#: ``identity_rebind`` hands back the SAME volumes and MUST BE UNCAUGHT: it is
#: what says the other two legs measure the substitution rather than the act of
#: passing a ``conductivity`` argument at all.
NULL_HOST_MUTATIONS: Tuple[str, ...] = ("identity_rebind",)
#: ``alias_every_component_to_the_first`` cannot bite where only one component is
#: conductive, and that is a fact about the mask rather than about the kernel.
HOST_MUTATION_NEEDS_TWO_LIVE = ("alias_every_component_to_the_first",)
#: The record's membership runs choose a form only under a layer; the no-absorber
#: kernels read its presence word alone.
RECORD_HOST_MUTATIONS: Tuple[str, ...] = ("record_bound_off_by_one",
                                          "record_lattices_swapped")
HOST_MUTATION_NEEDS_LAYER = RECORD_HOST_MUTATIONS + ("conductive_chunks_unbound",)
#: The per-chunk keep is observable only where some chunk keeps none.
HOST_MUTATION_NEEDS_HALF_SPACE_SIGMA = ("conductive_chunks_unbound",)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], sub_step: str, mask,
             e_storage: str, courant: float, value_class: str,
             guard: str, substitution: str,
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
            f"{value_class}".encode()).digest()[:4], "big"))

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "arm": arm, "layer": kind,
        "mask": MASK_IDS[mask], "e_storage": e_storage,
        "absorber": "active" if spec_is_active(spec) else spec["absorber"],
        "sigma": sigma_layout(spec),
        "courant": courant, "value_class": value_class, "guard": guard,
        "backend": backend, "substitution": substitution,
        "host_mutation": host_mutation,
        "fold_axes": spec["axes"], "boundaries": list(spec["boundaries"]),
    }
    if kind == "active" and e_storage == "derived":
        case["skipped"] = ("enable_pml_storage stores E unconditionally, so there "
                           "is no derived arm under an active layer")
        case["seconds"] = time.time() - started
        return case

    fields, layer, grid = build(xp, spec, sub_step, mask, courant, e_storage, rng)
    names = state_names(sub_step, arm, kind, mask)
    host = seed_state(fields, grid, names, value_class, rng)
    dtdx = float(grid.dt / grid.dx)
    record = case_record(grid, layer)
    codes, resolved, kinds, why_skip = boundary_codes_for(grid, substitution)
    case.update({
        "resolved_codes": None if resolved is None else [int(c) for c in resolved],
        "boundary_codes": None if codes is None else [int(c) for c in codes],
        "resolved_kinds": None if kinds is None else list(kinds),
        "dtdx": dtdx, "structure": structure_facts(grid),
        "chunk_record": [int(word) for word in record],
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
        case["pml_case_census"] = pml_case_census(
            fields, layer, sub_step, tuple(grid.shape), mask, record,
            sigma_layout(spec))
        if not case["pml_case_census"]["meets_floor"]:
            case["skipped"] = (
                "the layer did not put all four of MEEP's subchunk cases into "
                "this volume (or, on a half-space conductivity, no dsig cell of a "
                "chunk that keeps none); a pass would be a statement about the layer, not "
                f"about the kernel: {case['pml_case_census']['reached']}, unkept "
                f"dsig {case['pml_case_census']['unkept_dsig_reached']}")
            case["seconds"] = time.time() - started
            return case

    frozen = snapshot(fields, names)
    case["sources_are_distinct"] = sources_are_distinct(frozen, sub_step, arm, kind)
    if not case["sources_are_distinct"]["meets_floor"]:
        case["skipped"] = "two source volumes are identical"
        case["seconds"] = time.time() - started
        return case

    curls = unmasked_curls(fields, sub_step, arm, kind, resolved, dtdx, record)
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
    mutated_record = host_mutated_record(host_mutation, record, grid)
    launch_record = record if mutated_record is None else mutated_record
    keeps = host_mutated_keeps(host_mutation, fields, layer, sub_step)
    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(spec, sub_step, arm, fields, layer, codes, dtdx, launch_record, None,
           conductivity, keeps)

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
            runner(spec, sub_step, arm, fields, layer, codes, dtdx, record)
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
      all five masks, both courants, both value classes. That is the evidence the
      release verdict rests on.
    * The MISDECLARATION CONTROLS answer "is this gate measuring the boundary
      handling at all", which is a question about the code triple and not about
      the conductivity. They run at ONE mask, one courant and one value class --
      running them at all five masks would multiply a control that
      already answers its question.
    * The UNGUARDED guard set exists only for the contraction control, which is
      scored at the INEXACT courant (0.5 is exactly representable and cannot
      separate a contracted expression from an uncontracted one), so it runs at
      that courant alone.
    """
    primary = guard == "fmad_false"
    specs = (SPECS if product == "full"
             else tuple(_spec_by_label(label) for label in
                        ("3d_mixed", "3d_mixed_inert_layer", "3d_pml_all_axes",
                         "3d_pml_all_axes_half_sigma")))
    masks = COND_MASKS if product == "full" else (COND_MASKS[0], COND_MASKS[1])
    courants = COURANTS if (product == "full" and primary) else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    out = []
    for spec in specs:
        active = spec_is_active(spec)
        storages = ("stored",) if active else E_STORAGE
        for sub_step in SUB_STEPS:
            for mask in masks:
                for e_storage in storages:
                    for courant in courants:
                        for value_class in classes:
                            out.append((spec, sub_step, mask, e_storage,
                                        courant, value_class,
                                        LICENSED_SUBSTITUTION))
            if product != "full" or not primary:
                continue
            for substitution in SUBSTITUTIONS:
                if substitution == LICENSED_SUBSTITUTION:
                    continue
                out.append((spec, sub_step, COND_MASKS[0], storages[0],
                            INEXACT_COURANT, "uniform", substitution))
    return out


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              product: str, guard: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product, guard)
    for index, entry in enumerate(plan, start=1):
        case = one_case(backend, entry[0], entry[1], entry[2], entry[3], entry[4],
                        entry[5], guard, entry[6])
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
            f"{case['value_class']} sigma={case['sigma']} "
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
    "3d_pml_fold_Y", "3d_pml_all_axes", "3d_pml_all_axes_half_sigma",
    "2d_pml_two_axes_half_sigma",
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
        key = (case["label"], case["sub_step"], case["mask"], case["arm"])
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
              else ("3d_mixed", "3d_pml_all_axes", "3d_pml_all_axes_half_sigma"))
    for label in labels:
        spec = _spec_by_label(label)
        for sub_step in SUB_STEPS:
            for mask in MUTATION_MASKS:
                storages = ("stored",) if spec_is_active(spec) else E_STORAGE
                for e_storage in storages:
                    arm = arm_of(spec, sub_step, e_storage)
                    key = (label, sub_step, MASK_IDS[mask], arm)
                    if scorable is not None and key not in scorable:
                        continue
                    if mutation in MASK_MUTATIONS and key in dead_masks:
                        continue
                    plan.append((spec, sub_step, mask, e_storage,
                                 INEXACT_COURANT, "uniform"))
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
        for spec, sub_step, mask, e_storage, courant, value_class in plan:
            if name in HOST_MUTATION_NEEDS_TWO_LIVE and sum(mask) < 2:
                continue
            if name in HOST_MUTATION_NEEDS_LAYER and not spec_is_active(spec):
                continue
            if (name in HOST_MUTATION_NEEDS_HALF_SPACE_SIGMA
                    and sigma_layout(spec) != "half_space"):
                continue
            legs.append(one_case(backend, spec, sub_step, mask, e_storage, courant,
                                 value_class, "fmad_false",
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
                if not legs_for_this and (name in SOURCE_MUTATION_LEG_FILTER
                                          or name in SOURCE_MUTATION_SIGMA_FILTER):
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
                # template WITH its ``#define COND`` (and, on the PML pair, ``KEEP``)
                # lines -- not the template alone. Digesting the template would count zero constructions from
                # the mutated bytes on every leg and report UNACCOUNTED, which is
                # what one device run did before this was fixed: a battery scoring
                # its own accounting rather than the kernel.
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                built = {hashlib.sha256(
                    conductive_kernels.kernel_source(kernel_name, mask, keep
                                                     ).encode("utf-8")).hexdigest()
                    for mask in COND_MASKS
                    for keep in itertools.product((True, False), repeat=3)}
                legs = [one_case("cuda", spec, sub, mask, storage, courant,
                                 value_class, "fmad_false",
                                 LICENSED_SUBSTITUTION)
                        for spec, sub, mask, storage, courant, value_class
                        in legs_for_this]
                conductive_kernels.set_kernel_template(kernel_name,
                                                       originals[kernel_name])
                conductive_kernels.clear_kernel_cache()
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] in built)
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
            "sigma": set()})
        entry["cases"] += 1
        entry["masks"].add(case["mask"])
        entry["labels"].add(case["label"])
        entry["sigma"].add(case["sigma"])
        if case["single_launch"]["bit_identical"]:
            entry["single_identical"] += 1
        if "multi_step" in case:
            entry["multi_cases"] += 1
            if case["multi_step"]["bit_identical"]:
                entry["multi_identical"] += 1
        entry["differing_words"] += case["localization_summary"]["differing_words"]
        entry["differing_elsewhere"] += case["localization_summary"]["differing_elsewhere"]
    for entry in arms.values():
        for field in ("masks", "labels", "sigma"):
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
    unkept = max((case.get("pml_case_census", {}).get("unkept_dsig_reached", 0)
                  for case in scored), default=0)
    if any(c["layer"] == "active" for c in scored):
        for name, count in reached.items():
            if count == 0:
                reasons.append(f"MEEP subchunk case {name} was never reached by any "
                               f"PML case; that branch of the tail is untested")
        if unkept == 0:
            reasons.append("no PML case reached a dsig cell of a chunk that keeps no "
                           "conductivity; the per-chunk keep is untested")

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
        for name in (*SOURCE_MUTATION_LEG_FILTER, *SOURCE_MUTATION_SIGMA_FILTER):
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
         c["value_class"])
        for c in scored if c["substitution"] == LICENSED_SUBSTITUTION
        and c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["mask"], c["arm"], c["courant"],
                      c["value_class"]) in guarded_identical]
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
        "unkept_dsig_cells_reached": unkept,
        "arms": arms,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "claim": ("the conductive CUDA curl family -- three no-absorber kernels and "
                  "the conductivity + PML pair, its forms chosen per structure chunk "
                  "from the chunk record -- is byte-identical to "
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
                     "per PER-COMPONENT conductivity mask -- including MEEP's "
                     "conductive forms under a PML, chosen per structure chunk "
                     "from the chunk record and from which chunks keep the "
                     "conductivity, which the array path evaluates over the whole "
                     "volume and selects with copyto and the kernel selects per "
                     "cell?"),
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
