"""Sub-step byte-identity gate for the SHIPPED CUDA constitutive pair ON A FOLD.

THE CLAIM UNDER TEST, and it is a claim about a PREDICATE, not about a kernel.
``meep_gpu/cuda_kernels/constitutive_kernels.py`` ships two certified kernels,
``update_H_pml_real`` and ``update_E_pml_real``. This gate does not
change one byte of either. It asks whether the refusal
``covers_real_pml_constitutive`` states at ``coverage.py:624-626`` --

    "mirror symmetry: the fold changes the stored extent, and the extent is what
     turns a cell index into a coefficient index"

-- is a refusal the arithmetic requires, or a refusal inherited from the CURL
family that shares the predicate's shape.

WHY THE READING SAYS IT IS NOT REQUIRED (all four facts read off the source, and
the last one measured on a laptop before this file was written):

1. The sub-step reads NO NEIGHBOUR. ``stepping._apply_constitutive_pml``
   (stepping.py:2112-2143) is ``fw_previous = fw.copy(); fw[...] = source;
   field += kps*fw; field -= kms*fw_previous`` -- four whole-array operations, no
   shift, no ghost, no mask. A fold changes the GHOST RULE and the OWNERSHIP
   MASK; this sub-step has neither. ``_mask_non_owned_cells`` (stepping.py:1912)
   has exactly three call sites -- ``step_B`` (:397), ``step_D`` (:479) and
   ``_bfast_term`` (:902) -- and no constitutive function is among them.
2. ``update_H`` (stepping.py:907-922) and ``update_E`` (:926-995) never mention
   symmetry, a mirror, a parity or a fold. The fold's repairs are separate
   passes the DRIVER runs (``fill_symmetry_bc_B``/``_D``,
   ``fill_folded_far_ghosts_B``/``_D``), outside this sub-step on both sides.
3. THE EXTENT MOVES BOTH OPERANDS TOGETHER. ``PML._compute_coefficients``
   (pml.py:693-695, :706-708) builds every kps/kms vector from ``self.grid.nx``,
   ``grid.ny``, ``grid.nz`` -- the STORED extent, which is what ``Grid``'s fold
   halves (grid.py:570-575). The field arrays are allocated on the same
   ``grid.shape``. So the kernel's ``i = idx/(ny*nz)`` indexes a table of exactly
   the length the fold left, and the array path's ``(n,1,1)`` broadcast is the
   same map. The refusal's PREMISE is true and its CONCLUSION does not follow.
4. The sibling Triton track reached this conclusion first and wired it:
   ``triton_kernels/symmetry.py:3-6`` says the folded slice "reuses
   ``kernels.constitutive_step``, because it is element-wise and its PML
   coefficient vectors already use the folded grid's stored extent", and
   ``folded_constitutive_coverage`` (symmetry.py:789) is the predicate that
   admits it. CUDA predicates have matched Triton within 0-1 rows on all three
   certified families so far, which is a reason to MEASURE this one, not to
   assume it.

=============================================================================
WHY A SUB-STEP GATE AND NOT A WHOLE-STEP ONE
=============================================================================

A WHOLE-STEP GATE CANNOT SEE A MASK. The driver's fold repairs overwrite exactly
the planes a missing mask would corrupt (the same fact ``symmetry.py`` records
for the folded curl: "THE MASKS ARE INVISIBLE AT WHOLE-STEP GRANULARITY... a
whole-step in-session check therefore certifies a mask-less symmetry kernel as
correct"). This sub-step is claimed to need no mask at all, and a whole-step
check is precisely the instrument that cannot tell "needs none" from "is missing
one". So every comparison here is ONE SUB-STEP from ONE frozen state.

=============================================================================
WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
=============================================================================

* ``oracle_moved`` -- the fraction of output words the array path changed from
  the frozen input. Zero-init is a fixed point of this recurrence; a case that
  moved nothing is refused, not passed.
* ``folded_axis_absorbs`` -- ``max|kps-1|`` and ``max|kms-1|`` ON THE FOLDED
  AXIS. If the folded axis's coefficient vector is the identity everywhere, a
  coefficient-index error on that axis is invisible and the case is testing
  every axis except the one the fold touched. Measured on the laptop before this
  file: a folded Y periodic axis carries ``kms_y = [1...1, 0.1905, -2.238]``, so
  the floor is met -- but it is met because ``PML._resolve_mirror_faces``
  (pml.py:387-415) drops the LOW face and keeps the high one, and a fixture that
  asked for no absorber there would silently lose the whole question.
* ``stored_past_owned`` -- ``stored_cells - owned_cells`` on the folded axis.
  A folded PERIODIC axis stores ONE SLOT PAST MEEP's owned window and a folded
  METALLIC one does not (grid.py:570-575), and that extra slot is where the
  deepest absorber coefficient lands. Both terminations are swept because they
  are structurally different arrays, and the field records which is which.
* The mutation battery, including two that MUST BE UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or
  has degenerated into failing everything.

=============================================================================
THE FOLD-SPECIFIC MUTATION
=============================================================================

``reverse_folded_axis_coefficients`` is this gate's own, and it is the refusal
reason armed as a defect: the folded axis's kps/kms vectors are reversed before
they are handed to the kernel, which is what "the kernel read the profile at the
wrong extent/orientation" looks like IN BOUNDS. It must be CAUGHT. Without it,
"the fold does not move the coefficient index" would be a claim resting on a leg
nobody showed could fail on a folded grid.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=3 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_folded_constitutive.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the
same floors and the same HOST mutations against a transcription of the shipped
device tree. IT COMPILES NOTHING AND CERTIFIES NOTHING -- what it settles is
whether the coefficient tables at the folded stored extent reproduce
``stepping`` at all, which is the reading's fourth fact, and whether the harness
can fail::

    python -u gate_cuda_folded_constitutive.py --backend numpy \\
        --out /tmp/folded_constitutive_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
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
except ImportError:  # laptop: the NumPy backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260819

#: Consecutive launches in the multi-step leg. 60 is the budget both certified
#: hand-CUDA records are cut at. "Identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence live. Held fixed, ``f_w`` settles
#: and a 60-step leg becomes a slow single-step leg. The same exact float32 scale
#: is applied on both paths, so it cancels out of the comparison.
_ADVANCE = np.float32(0.97)

SIDE_ARRAYS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "H": {"targets": ("Hx", "Hy", "Hz"),
          "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz")},
    "E": {"targets": ("Ex", "Ey", "Ez"),
          "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz")},
}


def outputs(side: str) -> Tuple[str, ...]:
    """The six arrays this sub-step writes.

    ``f_w`` IS STATE. A tree that gets the field right and ``f_w`` wrong is
    correct for exactly one launch and wrong forever after, so the auxiliary is
    compared on every case, not only in the multi-step leg. The sibling track
    measured a dropped auxiliary store at 120/120 UNCAUGHT when ``fu`` was not
    compared.
    """
    spec = SIDE_ARRAYS[side]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def state_names(side: str) -> Tuple[str, ...]:
    return outputs(side) + tuple(SIDE_ARRAYS[side]["sources"])


# ---------------------------------------------------------------------------
# The case product
# ---------------------------------------------------------------------------
#
# EVERY AXIS IS FOLDED SOMEWHERE IN THIS LIST, and both terminations appear on
# each. ``PML._resolve_mirror_faces`` reads ``Grid.is_mirrored`` per axis, and
# the note above it records that enumerating X and Y alone once left a folded Z
# graded from BOTH walls -- so a sweep carrying one folded axis proves nothing
# about the other two.
#
# ``cell`` is in units of 1/resolution and the fixture runs at resolution 1.0, so
# the numbers ARE the full cell counts. An ODD full count is carried because
# MEEP's fold lands on a grid point at both parities but shifts the window, and
# the stored extent is ``n - n//2 + 1`` either way -- a different arithmetic from
# the even case even though nothing here reads it.

FOLD_SPECS: Tuple[Dict[str, Any], ...] = (
    # The unfolded control. Already certified; carried so the harness is shown to
    # reproduce the standing result under its own fixture rather than only under
    # the fixture that produced the record.
    {"label": "unfolded_periodic", "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "unfolded_metallic", "axes": "", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (8.0, 10.0, 12.0)},

    # Folded Y, both terminations, both parities of the plane, even full count.
    {"label": "fold_Y_periodic", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_metallic", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_periodic_odd_plane", "axes": "Y", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_metallic_odd_plane", "axes": "Y", "phase": -1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 16.0, 12.0)},

    # ODD FULL COUNT on the folded axis, both terminations.
    {"label": "fold_Y_periodic_odd_count", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 17.0, 12.0)},
    {"label": "fold_Y_metallic_odd_count", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 17.0, 12.0)},

    # Folded X and folded Z: the axis whose coefficient index is the SLOWEST
    # stride and the one whose index is the FASTEST. A decomposition defect that
    # confuses them is only visible if both are folded somewhere in the sweep.
    {"label": "fold_X_periodic", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_metallic", "axes": "X", "phase": 1,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_Z_periodic", "axes": "Z", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 12.0, 16.0)},
    {"label": "fold_Z_metallic", "axes": "Z", "phase": -1,
     "boundaries": ("periodic", "periodic", "metallic"), "cell": (8.0, 12.0, 16.0)},

    # TWO PLANES AT ONCE: the doubly-folded corner, one axis terminated each way.
    {"label": "fold_XY_mixed", "axes": "XY", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (16.0, 16.0, 10.0)},
    {"label": "fold_XZ_mixed", "axes": "XZ", "phase": -1,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 10.0, 16.0)},

    # THREE PLANES AT ONCE, added 2026-08-20. Until this round
    # :data:`CONSTITUTIVE_FOLD_ADMISSION` recorded ``folded_planes_swept = 2`` BY
    # CONSTRUCTION -- the number came from this tuple -- and the predicate quoted
    # it back as a refusal, so a three-plane fold was refused by the shape of the
    # case table rather than by any measurement. The corpus drives exactly one
    # triply-folded row (``TestLDOS.test_ldos_3D``, mirrored and metallic on all
    # three axes) and the refusal costs it two slots here and two on the curl.
    #
    # ALL THREE TERMINATION ARMS, not only the corpus's all-metallic one: the
    # folded PERIODIC axis is the one that stores a slot past MEEP's owned window,
    # so three of them at once is the shape where the deepest coefficient of every
    # axis lands in an extra slot simultaneously, and a mixed arm is the only one
    # where the three axes' vectors have different lengths RELATIVE to their owned
    # windows.
    #
    # THE EXTENTS ARE DELIBERATELY UNEQUAL (9, 10, 11 stored). A cube would let an
    # index decomposition confuse i, j and k and stay bit-identical, which is what
    # ``fortran_order_index_decomposition`` and ``own_axis_to_x_for_all_three``
    # plant.
    {"label": "fold_XYZ_metallic", "axes": "XYZ", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"),
     "cell": (16.0, 18.0, 20.0)},
    {"label": "fold_XYZ_periodic", "axes": "XYZ", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (16.0, 18.0, 20.0)},
    {"label": "fold_XYZ_mixed", "axes": "XYZ", "phase": -1,
     "boundaries": ("metallic", "periodic", "metallic"),
     "cell": (16.0, 18.0, 20.0)},
)

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient, so it is a real second draw of the tables and not only
#: a rounding probe.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

SIDES: Tuple[str, ...] = ("H", "E")

#: ``--fmad=false`` is CORRECTNESS on this sub-step, not tuning: both
#: accumulations and the E side's ``D*inv_eps`` are contraction candidates the
#: array path rounds twice. The control is EXPECTED TO DIVERGE at the inexact
#: courant; if it does not, the guard is decorative here and the record must say
#: so rather than implying evidence it does not have.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: Source mutations, borrowed from the shared probe so this gate cannot own a
#: second spelling of a defect the certified record was cut against. Device
#: backend only: they rewrite the module's device strings.
SOURCE_MUTATIONS_BY_SIDE: Dict[str, Tuple[str, ...]] = {
    "H": ("regroup_constitutive", "drop_fw_store", "store_fw_before_reading_prev",
          "own_axis_to_x_for_all_three", "fortran_order_index_decomposition",
          "commute_constitutive_scale"),
    "E": ("regroup_constitutive", "drop_fw_store", "store_fw_before_reading_prev",
          "drop_inverse_epsilon", "own_axis_to_x_for_all_three",
          "fortran_order_index_decomposition", "bind_Ez_inv_eps_for_all_three",
          "inv_eps_left", "commute_constitutive_scale"),
}

#: MUST BE UNCAUGHT, each paired with a leg editing the same expression that must
#: be caught (``drop_inverse_epsilon`` and ``regroup_constitutive``). A gate whose
#: whole battery must be caught scores identically whether the comparator works or
#: has degenerated into failing everything.
NULL_SOURCE_MUTATIONS: Tuple[str, ...] = ("inv_eps_left", "commute_constitutive_scale")

#: HOST mutations: they corrupt the TABLES rather than the device text, so they
#: run on both backends. ``reverse_folded_axis_coefficients`` is this gate's own
#: and is the refusal reason armed as a defect (see the header).
HOST_MUTATIONS: Tuple[str, ...] = (
    "reverse_folded_axis_coefficients",
    "swap_constitutive_sublattice",
    "swap_kps_kms",
)

#: Which host mutations are only meaningful on a folded case. Scoring
#: ``reverse_folded_axis_coefficients`` on the unfolded control would be scoring a
#: no-op as a miss.
FOLD_ONLY_HOST_MUTATIONS: Tuple[str, ...] = ("reverse_folded_axis_coefficients",)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    Everything else the fixture exercises -- the fold, the stored extent, the
    coefficient vector lengths, the dtype and contiguity -- is a real object
    either way. Same shim the sibling gates use, so the three tracks' laptop legs
    are commensurable.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float):
    """A frozen ``(fields, layer, grid)`` triple for one fold spec.

    THE THICKNESS RULE IS THE SLICE'S OWN (``test_constitutive_pml_real.build``),
    not a new one: skip an axis too thin to hold a layer, and on a MIRRORED axis
    ask for the HIGH face only. ``PML._resolve_mirror_faces`` (pml.py:387-415)
    refuses a named low face on a folded axis outright -- cell 0 is the mirror
    plane, a boundary condition rather than a wall -- so this is the shape the
    engine accepts, and it is also what keeps ``folded_axis_absorbs`` above its
    floor: the folded axis still grades from its far wall.
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


def install_epsilon(fields, grid, rng) -> None:
    """THREE INDEPENDENT inverse-epsilon volumes, never one array bound thrice.

    Binding one volume for all three components is defect 2 of the complex
    template (``update_E_pml_complex`` passes ``fields.inv_eps``, the Ez
    view, fields.py:1259-1260). A fixture that handed one array to all three
    could not see it, and ``bind_Ez_inv_eps_for_all_three`` would come back
    UNCAUGHT for a reason that says nothing about the kernel.

    Drawn AWAY FROM 1.0 for the reason the sibling fixture states: against a
    table of ones, ``drop_inverse_epsilon`` is bit-identical.
    """
    xp = grid.xp
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray((np.float32(1.0) / values).astype(np.float32))
    fields.set_epsilon_volumes(epsilon, inverse)


def seed_state(fields, grid, side: str, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Physical-band or subnormal-band values in every array the sub-step touches.

    THE AUXILIARIES START NONZERO in both classes. A zero ``f_w`` makes
    ``kms * prev`` exactly zero on the first launch whatever ``kms`` holds, so a
    mis-indexed coefficient would only show from step two -- precisely what the
    multi-step leg exists to catch, and no reason to hide it from the single
    launch as well.

    THE MATERIAL STAYS NORMAL under the band class: driving inverse epsilon into
    the band too would make every product underflow and the leg would measure the
    fixture rather than the policy's reach into this arithmetic.
    """
    xp = grid.xp
    names = state_names(side)
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


def snapshot(fields, side: str) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in state_names(side)}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, side: str) -> None:
    """Move the source between launches, exactly the same way on both paths.

    A real run's curl rewrites B/D before every constitutive call. Held fixed,
    ``f_w`` reaches a fixed point and 60 launches measure what one launch does.
    """
    for name in SIDE_ARRAYS[side]["sources"]:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# The coefficient tables, and the host mutations that corrupt them
# ---------------------------------------------------------------------------

def tables_for(side: str, layer) -> Dict[str, Any]:
    """The six flattened kps/kms views this side reads.

    ``constitutive_sub_lattice`` is asked here rather than hard-coding the
    suffix, because it is the same function the PREDICATE asks -- the pairing
    cannot disagree between the two, and getting it backwards is a half-cell
    error in the absorber profile: converged, smooth and wrong.
    """
    suffix = "_h" if coverage.constitutive_sub_lattice(side) else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def wrong_sub_lattice_tables(side: str, layer) -> Dict[str, Any]:
    """The OTHER side's sub-lattice: a half-cell error, not a crash."""
    suffix = "" if coverage.constitutive_sub_lattice(side) else "_h"
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def reversed_folded_axis_tables(side: str, layer, grid) -> Dict[str, Any]:
    """THE REFUSAL REASON, ARMED. The folded axis's profile read backwards.

    "The fold changes the stored extent, and the extent is what turns a cell
    index into a coefficient index" is the predicate's stated reason for refusing
    a fold. If that mattered to this sub-step, the coefficient a cell reads on the
    folded axis would be wrong. This mutation makes it wrong -- IN BOUNDS, by
    reversing the vector rather than by substituting a longer or shorter one, so
    the leg measures a WRONG ANSWER and not a memory fault.

    It is also the only thing that makes ``folded_axis_absorbs`` load-bearing: a
    folded axis whose profile is the identity everywhere reverses to itself, and
    the leg would come back UNCAUGHT for a reason about the fixture rather than
    about the kernel. The floor refuses such a case before it is scored.
    """
    tables = dict(tables_for(side, layer))
    xp = grid.xp
    for axis, name in enumerate("xyz"):
        if not grid.is_mirrored(axis):
            continue
        for stem in ("kps", "kms"):
            key = f"{stem}_{name}"
            tables[key] = xp.ascontiguousarray(tables[key][::-1])
    return tables


def swapped_kps_kms_tables(side: str, layer) -> Dict[str, Any]:
    """``(kap+sig)`` and ``(kap-sig)`` exchanged: the absorber runs backwards."""
    base = tables_for(side, layer)
    out = dict(base)
    for axis in "xyz":
        out[f"kps_{axis}"] = base[f"kms_{axis}"]
        out[f"kms_{axis}"] = base[f"kps_{axis}"]
    return out


def host_mutated_tables(name: str, side: str, layer, grid) -> Dict[str, Any]:
    if name == "reverse_folded_axis_coefficients":
        return reversed_folded_axis_tables(side, layer, grid)
    if name == "swap_constitutive_sublattice":
        return wrong_sub_lattice_tables(side, layer)
    if name == "swap_kps_kms":
        return swapped_kps_kms_tables(side, layer)
    raise KeyError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

def run_kernel_cuda(side: str, fields, tables: Dict[str, Any]) -> None:
    """The SHIPPED kernel, through its own public entry point.

    ``tables`` is passed explicitly -- the entry point's keyword-only gate door,
    which exists so a harness can hand in deliberately mis-paired tables. Passing
    the layer instead would derive them correctly and disarm every host mutation.
    """
    from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
    constitutive_kernels.update_fused_pml_real(side, fields, tables=tables)
    cp.cuda.runtime.deviceSynchronize()


def _broadcast(vector, axis: int):
    shape = [1, 1, 1]
    shape[axis] = int(vector.size)
    return np.asarray(vector).reshape(shape)


def run_kernel_numpy(side: str, fields, tables: Dict[str, Any]) -> None:
    """The device tree, transcribed, in float32 -- the laptop backend.

    FOUR LINES, and they are the four the slice test pins in the shipped device
    source (``test_the_grouping_is_two_separate_accumulations``,
    ``test_prev_is_loaded_before_the_auxiliary_is_stored``)::

        float prev = fw[idx];
        fw[idx] = src;
        float a = f[idx] + kps * src;
        f[idx] = a - kms * prev;

    COMPONENT c READS AXIS c's TABLE (``H_CONSTITUTIVE_TERMS`` /
    ``E_CONSTITUTIVE_TERMS``, stepping.py:227-228), which the kernel spells as
    ``kps_x[i]`` / ``kps_y[j]`` / ``kps_z[k]`` and this spells as a broadcast on
    the same axis. That correspondence is the whole fold question: the broadcast
    length here is ``tables[...]``'s length, i.e. the STORED extent, exactly as
    the device index is bounded by the stored shape.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the NVRTC
    contraction guard exists for, it does not exercise the subnormal policy, and
    it is not the shipped bytes. What it settles is (a) whether the tables at the
    folded extent reproduce ``stepping`` at all and (b) whether this harness can
    fail -- both on a laptop, before a device slot is spent.
    """
    spec = SIDE_ARRAYS[side]
    for axis, name in enumerate("xyz"):
        target = getattr(fields, spec["targets"][axis])
        aux = getattr(fields, spec["aux"][axis])
        source = getattr(fields, spec["sources"][axis])
        if side == "E":
            src = (source * fields.inverse_epsilon_for(spec["targets"][axis])
                   ).astype(np.float32)
        else:
            src = source.astype(np.float32)
        kps = _broadcast(tables[f"kps_{name}"], axis)
        kms = _broadcast(tables[f"kms_{name}"], axis)
        prev = aux.copy()
        aux[...] = src
        a = (target + kps * src).astype(np.float32)
        target[...] = (a - kms * prev).astype(np.float32)


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                 side: str) -> float:
    """Fraction of output WORDS the array path changed from the frozen input.

    Compared as raw uint32, never with ``allclose``: ``-0.0 == 0.0`` and
    ``NaN != NaN`` both lie, and the subnormal-band class puts signed zeros in
    the operands deliberately.
    """
    moved = total = 0
    for name in outputs(side):
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def folded_axis_absorbs(side: str, layer, grid) -> Dict[str, Any]:
    """Does the FOLDED axis's coefficient profile differ from the identity?

    ``kps = kms = 1`` is the interior pass-through. An axis whose whole vector is
    that identity cannot distinguish a coefficient-index error on it, so a folded
    case at this floor is measuring every axis except the one the fold touched.
    """
    tables = tables_for(side, layer)
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


def structure_facts(grid) -> Dict[str, Any]:
    """The shape facts that make a folded case structurally different.

    ``stored_past_owned`` is 1 on a folded PERIODIC axis and 0 on a folded
    METALLIC one (grid.py:570-575), and the extra slot is where the deepest
    absorber coefficient lands. Recorded per case so a reader can see that both
    terminations were really swept and were really different arrays.
    """
    return {
        "shape": [int(n) for n in grid.shape],
        "shape_full": [int(n) for n in grid.shape_full],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "stored_cells": [int(grid.stored_cells(a)) for a in range(3)],
        "owned_cells": [int(grid.owned_cells(a)) for a in range(3)],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
    }


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], side: str, courant: float,
             value_class: str, guard: str,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML``
    objects. There is no second transcription on the oracle leg to drift: the
    kernel is compared against the thing it claims to reproduce, byte for byte,
    from ONE frozen state -- oracle runs, state is restored, kernel runs.
    """
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{side}|{courant}|{value_class}".encode()).digest()[:4], "big"))

    fields, layer, grid = build(xp, spec, courant)
    if side == "E":
        install_epsilon(fields, grid, rng)
    host = seed_state(fields, grid, side, value_class, rng)

    case: Dict[str, Any] = {
        "label": spec["label"], "side": side, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend,
        "host_mutation": host_mutation,
        "fold_axes": spec["axes"], "mirror_phase": spec["phase"],
        "boundaries": list(spec["boundaries"]),
        "structure": structure_facts(grid),
        "operand_census": operand_census(host),
    }

    # THE PREDICATE'S CURRENT ANSWER, recorded rather than acted on. This gate
    # exists to decide whether that answer should change, so it must not be the
    # thing that gates the measurement -- but a record that did not carry it
    # could not show WHICH clause the run was taken against.
    covered, reason = coverage.covers_real_pml_constitutive(fields, layer, grid, side)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    absorbs = folded_axis_absorbs(side, layer, grid)
    case["folded_axis_absorbs"] = absorbs
    if not absorbs["meets_floor"]:
        case["skipped"] = ("the folded axis's coefficient profile is the identity "
                           "everywhere; this case cannot distinguish a coefficient "
                           "index error on the axis the fold moved")
        case["seconds"] = time.time() - started
        return case

    frozen = snapshot(fields, side)

    # Leg 1: the oracle.
    if side == "H":
        stepping.update_H(fields, layer)
    else:
        stepping.update_E(fields, layer)
    reference = snapshot(fields, side)

    moved = oracle_moved(frozen, reference, side)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; zero-init is a fixed point of this recurrence "
                           "and a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    if host_mutation is None:
        tables = tables_for(side, layer)
    else:
        tables = host_mutated_tables(host_mutation, side, layer, grid)
    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(side, fields, tables)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(side)}
    case["single_launch"] = combine(parts)

    # Leg 3: the multi-step. The auxiliary is STATE, and a tree that gets the
    # field right and ``f_w`` wrong is correct for exactly one launch and wrong
    # forever after.
    #
    # ONE ``Fields`` OBJECT, RUN TWICE FROM THE SAME FROZEN STATE, rather than a
    # second object built beside it. Two objects means two epsilon draws unless
    # the volumes are copied across, and a fixture that solved two different
    # problems would report a divergence that says nothing about the kernel.
    if host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            if side == "H":
                stepping.update_H(fields, layer)
            else:
                stepping.update_E(fields, layer)
            advance_sources(fields, side)
        oracle_multi = snapshot(fields, side)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(side, fields, tables)
            advance_sources(fields, side)
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs(side)}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str) -> List[Tuple[Dict[str, Any], str, float, str]]:
    specs = FOLD_SPECS if product == "full" else FOLD_SPECS[:6]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    out = []
    for spec in specs:
        for side in SIDES:
            for courant in courants:
                for value_class in classes:
                    out.append((spec, side, courant, value_class))
    return out


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              product: str, guard: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, (spec, side, courant, value_class) in enumerate(plan, start=1):
        case = one_case(backend, spec, side, courant, value_class, guard)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {side} "
                f"c={courant} {value_class} SKIPPED: {case['skipped'][:60]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {side} "
            f"c={courant} {value_class} shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"moved={case['oracle_moved']:.3f} ({case['seconds']:.1f} s)")
    return cases


#: The specs a mutation leg is scored on. CHOSEN, not sliced off the front of the
#: sweep: a leg taken from the head of the case product landed entirely on the two
#: UNFOLDED controls, so ``reverse_folded_axis_coefficients`` -- the one mutation
#: that only exists on a fold -- was skipped on every leg and scored 0/0 UNCAUGHT.
#: That is a harness defect that reads exactly like a kernel defect, which is why
#: the selection is written out.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "unfolded_periodic",     # the control: a table defect must bite here too
    "fold_Y_periodic",       # a stored slot past the owned window
    "fold_Y_metallic",       # the halved count, no extra slot
    "fold_X_periodic",       # the slowest-stride coefficient index
    "fold_Z_metallic",       # the fastest-stride one
    "fold_XY_mixed",         # two planes, one termination each
    # ADDED 2026-08-20 with the third fold plane. A three-plane case that were
    # swept but never MUTATED would raise the cap on the strength of an unarmed
    # arm: "identical at three planes" is consistent with a comparator that
    # stopped looking there, and the only thing that separates the two is a defect
    # planted on a three-plane grid and shown to bite.
    "fold_XYZ_metallic",     # three planes, the corpus row's own arm
    "fold_XYZ_periodic",     # three stored slots past the owned window at once
)


def mutation_plan(product: str) -> List[Tuple[Dict[str, Any], str, float, str]]:
    """One case per (spec, side) for the mutation legs, at the INEXACT courant.

    Held to one courant and one value class deliberately: a mutation leg answers
    "can this gate see this defect at all", and multiplying it by the whole sweep
    buys repetitions of that answer rather than a second question.
    """
    by_label = {spec["label"]: spec for spec in FOLD_SPECS}
    labels = MUTATION_SPEC_LABELS if product == "full" else MUTATION_SPEC_LABELS[:3]
    return [(by_label[label], side, INEXACT_COURANT, "uniform")
            for label in labels for side in SIDES]


def leg_caught(case: Dict[str, Any]) -> bool:
    """Did this mutated leg diverge from the array path at EITHER granularity?

    A defect that leaves the field right and the auxiliary wrong is bit-identical
    on the target at launch one; a defect in the auxiliary alone shows in ``f_w``
    immediately and in the field from launch two. Both legs are compared, and a
    mutation caught by either is caught -- scoring only the single launch would
    call a defect inert that the 60-launch leg was built to find.
    """
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def run_host_mutations(results: Dict[str, Any], out_path: str, backend: str,
                       product: str) -> Dict[str, Any]:
    """Every table-level defect, on both backends. A gate that cannot fail certifies nothing."""
    out: Dict[str, Any] = {}
    plan = mutation_plan(product)
    for name in HOST_MUTATIONS:
        legs: List[Dict[str, Any]] = []
        for spec, side, courant, value_class in plan:
            if name in FOLD_ONLY_HOST_MUTATIONS and not spec["axes"]:
                continue
            case = one_case(backend, spec, side, courant, value_class,
                            "fmad_false", host_mutation=name)
            legs.append(case)
        scored = [c for c in legs if not c.get("skipped")]
        caught = sum(1 for c in scored if leg_caught(c))
        # NO LEGS is its own verdict, never "UNCAUGHT". A mutation that was never
        # run has not been shown inert; it has not been asked. Collapsing the two
        # is how a gate acquires a silent hole, and this one had it: the first
        # full run scored ``reverse_folded_axis_coefficients`` 0/0 UNCAUGHT
        # because every selected leg happened to be an unfolded control.
        if not scored:
            verdict = "NO LEGS"
        elif caught == len(scored):
            verdict = "CAUGHT"
        elif caught:
            verdict = "PARTIAL"
        else:
            verdict = "UNCAUGHT"
        out[name] = {
            "ran": len(scored),
            "caught": caught,
            "uncaught": len(scored) - caught,
            "must_be_caught": True,
            "verdict": verdict,
            "cases": legs,
        }
        log(f"[host-mut] {name}: caught {caught}/{len(scored)} -> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results: Dict[str, Any], out_path: str,
                         product: str) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED strings and recompiled.

    The compile memo is keyed through the SOURCE (``compile_cache.kernel_cache_key``),
    so a mutated body is a miss and reaches NVRTC. Every leg records how many
    constructions came from the mutated bytes, because a leg reporting a pass for
    a mutation it never applied is worse than no leg -- three measured instances
    on the sibling track.
    """
    from meep_gpu.cuda_kernels import compile_cache, constitutive_kernels  # noqa: PLC0415

    attribute_for = {"H": "_update_H_pml_real_kernel_code",
                     "E": "_update_E_pml_real_kernel_code"}
    originals = {side: getattr(constitutive_kernels, attribute)
                 for side, attribute in attribute_for.items()}
    out: Dict[str, Any] = {}
    # FOLDED LEGS ONLY. Every one of these defects is already scored on unfolded
    # grids by the certified record; what this gate adds is whether the same
    # battery still bites once the stored extent is halved.
    plan = [entry for entry in mutation_plan(product) if entry[0]["axes"]]

    try:
        for side in SIDES:
            attribute = attribute_for[side]
            for name in SOURCE_MUTATIONS_BY_SIDE[side]:
                transform = probe.SOURCE_MUTATIONS[name]
                mutated, sites = transform(originals[side])
                key = f"{side}:{name}"
                if sites == 0 or mutated == originals[side]:
                    out[key] = {"armed": False,
                                "why": (f"matched {sites} site(s) and changed "
                                        f"nothing; the mutation and the kernel "
                                        f"have drifted apart")}
                    log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                setattr(constitutive_kernels, attribute, mutated)
                compile_cache.clear_compile_log()
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                legs = []
                for spec, leg_side, courant, value_class in plan:
                    if leg_side != side:
                        continue
                    legs.append(one_case("cuda", spec, side, courant, value_class,
                                         "fmad_false"))
                setattr(constitutive_kernels, attribute, originals[side])
                scored = [c for c in legs if not c.get("skipped")]
                caught = sum(1 for c in scored if leg_caught(c))
                caught_single = sum(1 for c in scored
                                    if not c["single_launch"]["bit_identical"])
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] == digest)
                must_be_caught = name not in NULL_SOURCE_MUTATIONS
                if not scored:
                    verdict = "NO LEGS"
                elif must_be_caught:
                    verdict = ("CAUGHT" if caught == len(scored)
                               else "PARTIAL" if caught else "UNCAUGHT")
                else:
                    verdict = "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"
                out[key] = {
                    "armed": True, "sites": sites,
                    "mutated_source_sha256": digest,
                    "kernel_constructions_from_mutated_bytes": from_mutated,
                    "ran": len(scored), "caught": caught,
                    "caught_at_single_launch": caught_single,
                    "must_be_caught": must_be_caught,
                    "verdict": verdict, "cases": legs,
                }
                if from_mutated == 0 and scored:
                    # A leg reporting a pass for a mutation it never applied is
                    # worse than no leg. The rewrite matched, the memo is keyed
                    # through the source, and yet nothing was built from these
                    # bytes -- so this leg measured the SHIPPED kernel and its
                    # verdict is about nothing.
                    out[key]["verdict"] = "UNACCOUNTED"
                    out[key]["why"] = ("no kernel construction used the mutated "
                                       "bytes; this leg did not exercise the "
                                       "mutation")
                log(f"[src-mut] {key}: caught {caught}/{len(scored)} "
                    f"builds_from_mutated={from_mutated} -> {verdict}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for side, attribute in attribute_for.items():
            setattr(constitutive_kernels, attribute, originals[side])
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    """The release decision, with every clause it rests on named."""
    reasons: List[str] = []
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    folded = [c for c in scored if c["fold_axes"]]
    single_ok = [c for c in scored if c["single_launch"]["bit_identical"]]
    multi_cases = [c for c in scored if "multi_step" in c]
    multi_ok = [c for c in multi_cases if c["multi_step"]["bit_identical"]]

    if not folded:
        reasons.append("no folded case was scored at all")
    if len(single_ok) != len(scored):
        reasons.append(f"single-launch divergence on {len(scored) - len(single_ok)} "
                       f"of {len(scored)} cases")
    if len(multi_ok) != len(multi_cases):
        reasons.append(f"multi-step divergence on {len(multi_cases) - len(multi_ok)} "
                       f"of {len(multi_cases)} cases")

    # Both terminations, on a folded axis, must have been scored -- they are
    # structurally different arrays (one slot past the owned window or not).
    terminations = set()
    for case in folded:
        for axis, mirrored in enumerate(case["structure"]["mirrored"]):
            if mirrored:
                terminations.add("metallic" if case["structure"]["metallic"][axis]
                                 else "periodic")
    if terminations != {"periodic", "metallic"}:
        reasons.append(f"folded terminations scored were {sorted(terminations)}, "
                       f"not both of periodic and metallic")

    # THE PLANE COUNT IS MEASURED FROM WHAT RAN, never quoted from the case table.
    # :data:`coverage.CONSTITUTIVE_FOLD_ADMISSION`'s ``folded_planes_swept`` may be
    # raised to this number and to no other -- the field it replaced was taken from
    # the spec tuple, which is how the predicate came to refuse a plane count on
    # the strength of a list rather than of a measurement.
    plane_counts = sorted({sum(1 for a in range(3) if c["structure"]["mirrored"][a])
                           for c in folded})
    max_planes = max(plane_counts) if plane_counts else 0
    planes_by_termination: Dict[str, set] = {}
    for case in folded:
        structure = case["structure"]
        axes = [a for a in range(3) if structure["mirrored"][a]]
        arm = ("metallic" if all(structure["metallic"][a] for a in axes)
               else "periodic" if not any(structure["metallic"][a] for a in axes)
               else "mixed")
        planes_by_termination.setdefault(arm, set()).add(len(axes))
    planes_by_termination = {arm: sorted(counts)
                             for arm, counts in planes_by_termination.items()}
    # A cap of N rests on N-plane cases at BOTH folded terminations. A folded
    # PERIODIC axis stores one slot past MEEP's owned window and a METALLIC one
    # does not, so they are structurally different arrays and a cap resting on one
    # of them is the same over-claim, one plane further out.
    for arm in ("metallic", "periodic"):
        if arm in planes_by_termination and max(planes_by_termination[arm]) != max_planes:
            reasons.append(
                f"folded {arm} was scored at up to "
                f"{max(planes_by_termination[arm])} simultaneous fold planes but "
                f"the sweep reached {max_planes}; a plane cap must rest on both "
                f"terminations at the count it names")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored on any case; "
                           f"unasked is not inert")
        elif leg["verdict"] != "CAUGHT":
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']}")

    # THE GUARD CONTROL, reported rather than gated on. ``--fmad=false`` is
    # claimed to be CORRECTNESS on this sub-step; the unguarded leg is expected to
    # DIVERGE at the inexact courant, where a contracted and an uncontracted
    # expression round differently. If it does not, the guard is decorative here
    # and this field is what says so, instead of the record implying evidence it
    # does not have. NOT a release clause: the guard's necessity is a separate
    # claim from the kernel's identity under it.
    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    control_diverged = [c for c in control if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "diverged": len(control_diverged),
        # NOT MEASURED is its own reading. The NumPy backend has no compiler to
        # guard, so it scores nothing here, and a run that scored nothing must
        # not read as a run that found nothing.
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored at "
                    "the inexact courant" if not control else
                    "the contraction guard is load-bearing on this sub-step"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here"),
    }

    return {
        "released": not reasons,
        "reasons": reasons,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "folded_cases": len(folded),
        "single_launch_identical": len(single_ok),
        "multi_step_identical": len(multi_ok),
        "multi_step_cases": len(multi_cases),
        "folded_terminations": sorted(terminations),
        # THE CAP, MEASURED. The only number
        # ``CONSTITUTIVE_FOLD_ADMISSION["folded_planes_swept"]`` may be set from.
        "folded_planes_scored": plane_counts,
        "max_folded_planes_scored": max_planes,
        "folded_planes_by_termination": planes_by_termination,
        "claim": ("the SHIPPED CUDA constitutive kernels are byte-identical to "
                  "stepping.update_H/update_E on a folded stored extent, per "
                  "sub-step, at both folded terminations and up to "
                  f"{max_planes} simultaneous fold planes, with no change to the "
                  "device source"),
        "does_not_claim": [
            f"more than {max_planes} simultaneous mirror planes: not swept, and "
            f"the predicate refuses them by name. Three is every plane a 3-D grid "
            f"has, so at max_planes = 3 that clause is unreachable rather than "
            f"merely unmet",
            "nothing dispatches these kernels; no module in meep_gpu imports "
            "cuda_kernels at all",
            "the predicate is NOT changed by this gate; the change it licenses "
            "is reported, not made",
            "the folded CURL is a separate family and is not measured here",
        ],
    }


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
    parser.add_argument("--import-meep-for-host-policy", action="store_true",
                        help=("import MEEP first so the HOST half of a 'flush' "
                              "policy can be attained; strict install refuses "
                              "otherwise"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_folded_constitutive",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("does covers_real_pml_constitutive's mirror-symmetry refusal "
                     "describe a divergence, or is the shipped kernel already "
                     "exact on a folded stored extent?"),
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        # THE OBSERVER GOES IN BEFORE THE POLICY, and the order is the design:
        # under 'keep' the policy's strip wraps this, so it records the option
        # tuple NVRTC was really given (post-strip). Installed afterwards it
        # would sit outside the strip and record the pre-strip tuple -- the one
        # thing that would make the two policy legs look alike.
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
            from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
            constitutive_kernels._COMPILE_OPTIONS = tuple(options)
            constitutive_kernels._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        constitutive_kernels._COMPILE_OPTIONS = ("--fmad=false",)
        constitutive_kernels._clear_kernel_cache()

    if not args.skip_mutations:
        run_host_mutations(results, args.out, args.backend, args.product)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.product)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} "
        f"scored={verdict['scored_cases']} folded={verdict['folded_cases']} "
        f"single_identical={verdict['single_launch_identical']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
