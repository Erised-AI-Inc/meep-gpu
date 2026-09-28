"""Sub-step byte-identity gate for the CUDA COMPLEX constitutive pair ON A FOLD.

THE CENSUS REFUSAL THIS ANSWERS, quoted from ``coverage._complex_grid_refusal``::

    mirror symmetry: different ghost rule, a parity mask and two fill passes

``_complex_grid_refusal`` is ONE function with TWO callers --
``covers_real_pml_complex_curl`` and ``covers_real_pml_complex_constitutive`` --
and this clause fires for both. FOR THE CURL IT IS REAL AND STAYS: the emitted
complex curl reads neighbours through ``cshift_up``/``cshift_dn``, a fold changes
its ghost rule, its ownership mask and its stored extent at once, and it takes
``bc_x/bc_y/bc_z`` as runtime arguments with no code for a mirror. The
CONSTITUTIVE pair inherited the refusal through the function the two share, and
inheritance is not evidence. This is the same shape the Dcyl clause had before
``gate_cuda_complex_cylindrical_constitutive.py`` measured it.

WHAT THE READING SAYS, and every fact is read off a named line:

1. ``complex_pml_kernels.update_fused_pml_complex`` (complex_pml_kernels.py:427-470)
   TAKES NO BOUNDARY CODES AND NO MASK. Its argument list is the target triple,
   the auxiliary triple, the source triple, the three inverse-epsilon volumes on
   the E side, ``nx, ny, nz`` and the six kps/kms vectors. No ``bc_*``, no
   ``ph_*``, no phase pair, no ``dtdx``. Compare its sibling
   ``curl_fused_pml_complex`` (:380-425), which passes all of them. There is no
   place in the constitutive template for a ghost rule to be wrong.
2. ``stepping._apply_constitutive_pml`` (stepping.py:2112-2145) reads NO
   NEIGHBOUR: ``fw_previous = fw.copy(); fw[...] = source; field += kps*fw;
   field -= kms*fw_previous``. ``_mask_non_owned_cells`` (:1865) has exactly
   three call sites -- ``step_B``, ``step_D`` and ``_bfast_term`` -- and no
   constitutive function is among them.
3. THE EXTENT MOVES BOTH OPERANDS TOGETHER. ``PML._compute_coefficients``
   (pml.py:693-695, :706-708) builds every kps/kms vector from ``grid.nx/ny/nz``
   -- the STORED extent, which is what the fold halves (grid.py:570-575) -- and
   the complex volumes are allocated on the same ``grid.shape``.
4. THE REAL PAIR'S FOLD WAS MEASURED AND NEEDED NO DEVICE CODE
   (:data:`coverage.CONSTITUTIVE_FOLD_ADMISSION`: 112/112 per policy on 96
   folded cases, the fold-specific mutation caught 10/10).

FACT 4 IS NOT INHERITED EITHER. The complex kernels are different bytes --
complex64 word pairs, the zero cross terms, an expansion arm the real pair does
not take -- and this project does not transfer a verdict across kernels. That is
the rule the Dcyl round stated and followed, and it is why this gate exists
rather than a sentence citing the real one.

=============================================================================
WHAT THIS IS WORTH, COUNTED BEFORE IT WAS RUN
=============================================================================

The 2026-08-20 closeout census carries EIGHT complex, PML-active, folded rows,
all thirty-two of whose slots are unserved. Widening this pair moves at most
SEVEN of them, and the arithmetic is worth writing down because the fold clause
SHORT-CIRCUITS and hides what lies behind it:

* only ``update_H``/``update_E`` are this pair's, so the sixteen curl slots are
  untouched -- the complex curl's fold refusal is real and stays;
* three of the eight rows carry ``grid.beta != 0`` (two ``TestEigCoeffs``
  gratings and ``TestSpecialKz.test_eigsrc_kz_0_complex``) and are refused for
  special_kz on every sub-step -- six slots that stay refused;
* three carry an off-diagonal epsilon (``solve-cw.py``,
  ``TestArrayMetadata.test_array_metadata``, ``TestHoleyWvgBands.test_fields_at_kx``)
  whose ``update_E`` is refused by a clause of its own -- three more.

16 - 6 - 3 = 7. NOT ONE OF THE EIGHT BECOMES A ROW THE HAND-CUDA TRACK CAN STEP
END TO END, because their curls stay refused. A gate that reported "eight rows
gained" would be reporting the fold clause's short circuit as coverage.

=============================================================================
WHY A SUB-STEP GATE AND NOT A WHOLE-STEP ONE
=============================================================================

A WHOLE-STEP GATE CANNOT SEE A MASK. The driver's fold repairs
(``fill_symmetry_bc_B``/``_D``, ``fill_folded_far_ghosts_B``/``_D``) overwrite
exactly the planes a missing mask would corrupt, so a whole-step in-session check
certifies a mask-less symmetry kernel as correct. This sub-step is claimed to
need no mask at all, and a whole-step check is precisely the instrument that
cannot tell "needs none" from "is missing one". Every comparison here is ONE
SUB-STEP from ONE frozen state.

=============================================================================
WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
=============================================================================

* ``oracle_moved`` -- the fraction of output WORDS the array path changed from
  the frozen input. Zero-init is a fixed point of this recurrence.
* ``folded_axis_absorbs`` -- ``max|kps-1|`` and ``max|kms-1|`` ON THE FOLDED
  AXIS. If the folded axis's profile is the identity everywhere, a
  coefficient-index error there is invisible and the case is testing every axis
  except the one the fold touched. It is also what makes
  ``reverse_folded_axis_coefficients`` -- the refusal's own premise armed -- a
  real defect rather than a no-op, since a constant vector reverses to itself.
* ``stored_past_owned`` -- recorded per case. A folded PERIODIC axis stores ONE
  SLOT PAST MEEP's owned window and a folded METALLIC one does not
  (grid.py:570-575); both terminations are swept because they are structurally
  different arrays, and the record shows which each case was.
* the signed-zero and subnormal word censuses, so the zero cross terms and the
  float32 subnormal policy are shown to have had something to act on.
* the mutation battery, including the NULLS. A battery whose every leg must be
  caught scores identically whether the comparator works or has degenerated into
  failing everything.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=3 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_complex_folded_constitutive.py --backend cupy \\
        --subnormal-policy keep --expansion-probe $PROBE --out $OUT/keep/gate.json

Laptop (no CUDA): ``--backend numpy`` runs the same oracle, fixture and floors
through the harness backend and CERTIFIES NOTHING; ``--backend host`` compiles
the emitted text with a host C++ compiler and certifies nothing about the device
either. Both exist so the fixture can be shown to work before a device slot is
spent.

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
except ImportError:  # laptop: the host-compiled and harness legs still run
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_complex as base  # noqa: E402
#: THE COMPLEX CONSTITUTIVE MACHINERY IS THE Dcyl GATE'S, imported rather than
#: respelled: the seeding of complex volumes plane by plane (which must never
#: form ``re + 1j*im``, or every negative zero is destroyed), the word census,
#: the table views, the sub-lattice mutation, the adjudicator and the source
#: mutations are the same questions asked of the same kernels. A second spelling
#: is a second thing that must be kept in step with the emitter forever. What
#: THIS file owns is the fold: the fixture that builds one, the floor that shows
#: the folded axis absorbs, and the mutation that arms the refusal.
import gate_cuda_complex_cylindrical_constitutive as cyl  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter, coverage  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host

SEED = 20260820

MULTI_STEP_BUDGET = cyl.MULTI_STEP_BUDGET
SIDE_ARRAYS = cyl.SIDE_ARRAYS
outputs = cyl.outputs
state_names = cyl.state_names
seed_state = cyl.seed_state
word_census = cyl.word_census
snapshot = cyl.snapshot
restore = cyl.restore
advance_sources = cyl.advance_sources
tables_for = cyl.tables_for
wrong_sub_lattice_tables = cyl.wrong_sub_lattice_tables
swapped_kps_kms_tables = cyl.swapped_kps_kms_tables
oracle_moved = cyl.oracle_moved
leg_caught = cyl.leg_caught
adjudicate = cyl.adjudicate
constitutive_sites = cyl.constitutive_sites
_licence_stub = cyl._licence_stub
_NumpyWearingCupysName = cyl._NumpyWearingCupysName

COURANTS = cyl.COURANTS
INEXACT_COURANT = cyl.INEXACT_COURANT
VALUE_CLASSES = cyl.VALUE_CLASSES
SIDES = cyl.SIDES
GUARD_SETS = cyl.GUARD_SETS
MUTATION_VALUE_CLASSES = cyl.MUTATION_VALUE_CLASSES


# ---------------------------------------------------------------------------
# THE CASE TABLE
# ---------------------------------------------------------------------------
#
# EVERY AXIS IS FOLDED SOMEWHERE IN THIS LIST, and both terminations appear.
# ``PML._resolve_mirror_faces`` reads ``Grid.is_mirrored`` per axis, and the
# sibling gates' note records that enumerating X and Y alone once left a folded Z
# graded from BOTH walls -- so a sweep carrying one folded axis proves nothing
# about the other two.
#
# THE TWO TERMINATIONS ARE NOT TWO SPELLINGS OF ONE CASE. A folded PERIODIC axis
# stores ONE SLOT PAST MEEP's owned window and a folded METALLIC one does not
# (grid.py:570-575), so their coefficient vectors are structurally different
# arrays and the deepest absorber coefficient lands in a different place. The
# refusal is about the extent; the extent is what differs.
#
# BOTH ARE IN THE CORPUS, which is why both are here rather than for symmetry:
# ``solve-cw.py`` and ``TestArrayMetadata.test_array_metadata`` fold X and Y with
# METALLIC terminations, and the four grating/waveguide rows fold one axis with a
# PERIODIC one.
#
# THE UNPHASED AND PHASED CASES ARE BOTH CARRIED. Five of the eight corpus rows
# record ``has_bloch`` TRUE with the phase on an UNFOLDED axis (a mirror plane
# forces that axis's k component to zero), so a sweep of unphased grids alone
# would leave the corpus's own shape unmeasured. The constitutive kernel takes no
# phase argument at all -- which is exactly why a phased case must be run rather
# than argued away.
#
# ``cell`` is in units of 1/resolution at resolution 1.0, so the numbers ARE the
# full cell counts. An ODD full count is carried because MEEP's fold lands on a
# grid point at both parities but shifts the window.

FOLD_SPECS: Tuple[Dict[str, Any], ...] = (
    # The unfolded controls. Already certified by the complex family's own round;
    # carried so a divergence in the folded rows cannot be blamed on the fixture.
    {"label": "unfolded_periodic", "axes": "", "phase": 1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "unfolded_metallic", "axes": "", "phase": 1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (8.0, 10.0, 12.0)},

    # Folded Y, both terminations, both plane parities, even full count.
    {"label": "fold_Y_periodic", "axes": "Y", "phase": 1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_metallic", "axes": "Y", "phase": 1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_periodic_odd_plane", "axes": "Y", "phase": -1,
     "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 12.0)},

    # ODD FULL COUNT on the folded axis.
    {"label": "fold_Y_periodic_odd_count", "axes": "Y", "phase": 1,
     "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 17.0, 12.0)},

    # Folded X and folded Z: the axis whose coefficient index is the SLOWEST
    # stride and the one whose index is the FASTEST. A decomposition defect that
    # confuses them is only visible if both are folded somewhere in the sweep.
    {"label": "fold_X_metallic", "axes": "X", "phase": 1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_Z_periodic", "axes": "Z", "phase": -1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 12.0, 16.0)},
    {"label": "fold_Z_metallic", "axes": "Z", "phase": -1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "metallic"), "cell": (8.0, 12.0, 16.0)},

    # A PHASE ON AN UNFOLDED AXIS -- the corpus's own shape. Five of the eight
    # rows this gate is about are Bloch-phased, always on an axis the fold does
    # not touch: a mirror plane forces its axis's k component to zero, which is
    # what ``TestHoleyWvgBands.test_fields_at_kx`` (fold Y, k on X) and
    # ``TestModeDecomposition.test_triangular_lattice_oblique`` (fold X, k on Y
    # and Z) record. The constitutive kernel takes no phase argument, so these
    # cases ask whether a phase the kernel cannot see nevertheless moves a word.
    {"label": "fold_Y_periodic_phased_x", "axes": "Y", "phase": 1,
     "k_point": (0.23, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_X_metallic_phased_yz", "axes": "X", "phase": 1,
     "k_point": (0.0, -0.17, 0.31),
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},

    # TWO PLANES AT ONCE, both terminations. ``solve-cw.py`` and
    # ``TestArrayMetadata.test_array_metadata`` fold X and Y together with
    # METALLIC terminations, which is exactly the first of these.
    {"label": "fold_XY_metallic", "axes": "XY", "phase": 1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("metallic", "metallic", "periodic"), "cell": (16.0, 16.0, 10.0)},
    {"label": "fold_XY_periodic", "axes": "XY", "phase": -1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 16.0, 10.0)},

    # THREE PLANES AT ONCE. No complex corpus row carries one -- the corpus's
    # triply-folded row is real storage -- but the two REAL families' plane caps
    # are being raised to three in the same round, and a complex pair capped at
    # two while its siblings sit at three would be an asymmetry resting on which
    # gate happened to sweep what rather than on any property of the kernels.
    # THE EXTENTS ARE DELIBERATELY UNEQUAL (9, 10, 11 stored): a cube would let an
    # index decomposition confuse i, j and k and stay bit-identical, which is what
    # ``column_major_index`` plants.
    {"label": "fold_XYZ_metallic", "axes": "XYZ", "phase": 1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (16.0, 18.0, 20.0)},
    {"label": "fold_XYZ_periodic", "axes": "XYZ", "phase": 1, "k_point": (0.0, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 18.0, 20.0)},
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(xp, spec: Dict[str, Any], courant: float, rng):
    """A frozen ``(fields, layer, grid)`` triple for one fold spec, complex64.

    ``force_complex_fields=True`` ALWAYS. Every row this gate is about records it
    TRUE in the census, and it is half of what the complex family exists for; the
    other half, a nonzero ``k_point``, is carried by the two phased specs.

    THE THICKNESS RULE IS THE SIBLING GATES' (``test_constitutive_pml_real.build``
    and ``gate_cuda_folded_constitutive.build``), restated rather than imported so
    the gates cannot drift apart silently through a shared helper: skip an axis too
    thin to hold a layer, and on a MIRRORED axis ask for the HIGH face only.
    ``PML._resolve_mirror_faces`` (pml.py:387-415) refuses a named low face on a
    folded axis outright -- cell 0 is the mirror plane, a boundary condition rather
    than a wall -- so this is the shape the engine accepts, and it is also what
    keeps ``folded_axis_absorbs`` above its floor: the folded axis still grades
    from its far wall.

    THREE INDEPENDENT inverse-epsilon volumes on the E side, never one array bound
    thrice: binding one volume for all three components is the defect
    ``fields.inv_eps`` (the Ez view, fields.py:1259-1260) invites, and a fixture
    that handed one array to all three could not see it. Drawn AWAY FROM 1.0 --
    against a table of ones a dropped inverse epsilon is bit-identical.
    """
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                k_point=tuple(spec.get("k_point") or (0.0, 0.0, 0.0)),
                xp=xp, courant=courant)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray((np.float32(1.0) / values).astype(np.float32))
    fields.set_epsilon_volumes(epsilon, inverse)
    return fields, layer, grid


# ---------------------------------------------------------------------------
# The fold's own host mutation and floor
# ---------------------------------------------------------------------------

def reversed_folded_axis_tables(side: str, layer, grid) -> Dict[str, Any]:
    """THE REFUSAL REASON, ARMED. The folded axis's profile read backwards.

    The complex clause's stated reason is a ghost rule, a parity mask and two fill
    passes -- none of which the constitutive template contains, since it takes no
    boundary argument at all. What a fold COULD still move in an element-wise
    sub-step is the coefficient INDEX, because the fold halves the stored extent
    the kps/kms vectors are built on. This mutation makes that index wrong -- IN
    BOUNDS, by reversing the vector rather than substituting a longer or shorter
    one, so the leg measures a WRONG ANSWER and not a memory fault.

    It is also the only thing that makes ``folded_axis_absorbs`` load-bearing: a
    folded axis whose profile is the identity everywhere reverses to itself, and
    the leg would come back UNCAUGHT for a reason about the fixture. The floor
    refuses such a case before it is scored.

    ON AN UNFOLDED CONTROL there is no folded axis and this is a no-op, which is
    why the control is a bar rather than a scored leg (``dead_must_be_identical``).
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


def host_mutated_tables(name: str, side: str, layer, grid) -> Dict[str, Any]:
    if name == "reverse_folded_axis_coefficients":
        return reversed_folded_axis_tables(side, layer, grid)
    if name == "swap_constitutive_sublattice":
        return wrong_sub_lattice_tables(side, layer)
    if name == "swap_kps_kms":
        return swapped_kps_kms_tables(side, layer)
    raise KeyError(f"unknown host mutation {name!r}")


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
    terminations were really swept and were really different arrays -- which is
    the claim's load-bearing half, because the refusal is about the extent.
    """
    return {
        "shape": [int(n) for n in grid.shape],
        "shape_full": [int(n) for n in grid.shape_full],
        "cells": int(np.prod([int(n) for n in grid.shape])),
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "has_symmetry": bool(grid.has_symmetry),
        "has_bloch": bool(getattr(grid, "has_bloch", False)),
        "k_point": [float(k) for k in getattr(grid, "k_point", (0.0, 0.0, 0.0))],
        "stored_cells": [int(grid.stored_cells(a)) for a in range(3)],
        "owned_cells": [int(grid.owned_cells(a)) for a in range(3)],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
        "folded_planes": sum(1 for a in range(3) if grid.is_mirrored(a)),
    }


# ---------------------------------------------------------------------------
# The mutation legs
# ---------------------------------------------------------------------------
#
# "caught"            -> every scored case must DIVERGE
# "caught_where_live" -> must diverge on every case the defect can REACH
# "uncaught"          -> every scored case must stay identical, and that IS the
#                        measurement
# "measure"           -> a number is reported, not a bar

SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = dict(
    cyl.SOURCE_MUTATIONS)

MUTATION_LEGS: Tuple[Dict[str, Any], ...] = (
    {"leg": "s1_fold_the_zero_cross_terms", "mutation": "fold_the_zero_cross_terms",
     "kind": "source", "expect": "caught_where_live", "reach": "signed_zero",
     "why": "the zero cross terms carry the sign of the other plane into an "
            "addend that is exactly zero; folding them to a literal is the "
            "transformation the compiler must NOT be doing for us"},
    {"leg": "s2_plane_wise_scaling", "mutation": "plane_wise_scaling",
     "kind": "source", "expect": "caught_where_live", "reach": "signed_zero",
     "why": "np.multiply carries no scalar-times-complex loop, so a real "
            "coefficient IS a full complex multiply on the array path"},
    {"leg": "s3_flatten_the_constitutive_tail",
     "mutation": "flatten_the_constitutive_tail", "kind": "source",
     "expect": "caught",
     "why": "two separate accumulations, left to right, is what the array path's "
            "two += statements fix"},
    {"leg": "s4_store_fw_before_reading_prev",
     "mutation": "store_fw_before_reading_prev", "kind": "source",
     "expect": "caught",
     "why": "prev must be read before the fw store; the wrong order is a slightly "
            "weaker absorber, wrong INSIDE THE PML ONLY -- and the PML on a folded "
            "axis is graded from the far wall alone"},
    {"leg": "s5_word_pair_transposed", "mutation": "word_pair_transposed",
     "kind": "source", "expect": "caught",
     "why": "a complex64 volume read as (im, re) pairs is a scrambled field, not "
            "a launch failure"},
    {"leg": "s6_column_major_index", "mutation": "column_major_index",
     "kind": "source", "expect": "caught",
     "why": "THE COEFFICIENT-INDEX LEG. i and k swapped means every kps/kms "
            "lookup names the wrong axis -- the defect class the fold refusal "
            "could still be about, planted in the device text rather than in the "
            "tables. The unequal extents in the case table are what make it "
            "visible"},
    {"leg": "s7_half_plane_field_store", "mutation": "half_plane_field_store",
     "kind": "source", "expect": "caught",
     "why": "complex64 is a float2 and a store that writes one plane is a real "
            "defect class in this tree; on an element-wise sub-step it is the "
            "only shape a plane-selective defect can take"},
    {"leg": "s8_half_plane_auxiliary_store", "mutation": "half_plane_auxiliary_store",
     "kind": "source", "expect": "caught",
     "why": "the same defect on the AUXILIARY, which is STATE: a half-written f_w "
            "is wrong forever after, and is invisible to a gate that compares "
            "only the field"},
    {"leg": "h1_reverse_folded_axis_coefficients",
     "mutation": "reverse_folded_axis_coefficients", "kind": "host",
     "expect": "caught_where_live", "reach": "folded",
     "dead_must_be_identical": True,
     "why": "THE REFUSAL'S OWN PREMISE, ARMED. If the fold moved a coefficient "
            "index the way the refusal implies, this is the mutation that would "
            "show it. The dead half is a bar because on an unfolded control the "
            "mutation is a no-op and must be one"},
    {"leg": "h2_swap_constitutive_sublattice",
     "mutation": "swap_constitutive_sublattice", "kind": "host",
     "expect": "caught",
     "why": "the half-cell error the sub-lattice pairing exists to prevent: "
            "converged, smooth and wrong"},
    {"leg": "h3_swap_kps_kms", "mutation": "swap_kps_kms", "kind": "host",
     "expect": "caught",
     "why": "(kap+sig) and (kap-sig) exchanged: the absorber runs backwards"},
    {"leg": "h4_the_other_expansion_arm", "mutation": "the_other_expansion_arm",
     "kind": "host", "expect": "measure",
     "why": "the arm this platform was NOT licensed for. Where the two arms are "
            "measurably apart this must DIVERGE; where they coincide the "
            "coincidence is the finding, and the number of cases in each bucket "
            "is what the licence's discrimination claim is worth on THIS "
            "sub-step and on a FOLDED grid"},
    {"leg": "n1_reload_the_constitutive_store",
     "mutation": "reload_the_constitutive_store", "kind": "source",
     "expect": "uncaught",
     "why": "a float32 stored to global memory and loaded back is the identity on "
            "the bits; a leg that caught this would mean the harness reports "
            "differences that are not arithmetic"},
    {"leg": "n2_commute_the_plane_sum", "mutation": "commute_the_plane_sum",
     "kind": "source", "expect": "uncaught",
     "why": "IEEE addition is commutative on the bits -- it is associativity that "
            "is not -- so this separates 'the gate sees grouping' from 'the gate "
            "sees any edit at all'"},
)

#: The specs a mutation leg is scored on. CHOSEN, not sliced off the front of the
#: case product: a leg taken from the head of the sweep landed entirely on the
#: UNFOLDED controls on a sibling round, so the one mutation that only exists on a
#: fold was skipped on every leg and scored 0/0 UNCAUGHT. That is a harness defect
#: that reads exactly like a kernel defect, which is why the selection is written
#: out and why one unfolded control is kept in it.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "unfolded_periodic",        # the control: a table defect must bite here too,
                                # and the fold mutation must be a no-op here
    "fold_Y_periodic",          # a stored slot past the owned window
    "fold_X_metallic",          # the slowest-stride coefficient index, no extra slot
    "fold_Z_periodic",          # the fastest-stride one
    "fold_Y_periodic_phased_x",  # a Bloch phase the kernel cannot see
    "fold_XY_metallic",         # two planes, the corpus's own shape
    "fold_XYZ_periodic",        # three planes, three extra slots at once
)

HOST_MUTATION_NAMES = ("reverse_folded_axis_coefficients",
                       "swap_constitutive_sublattice", "swap_kps_kms",
                       "the_other_expansion_arm")


def _reaches(leg: Dict[str, Any], case: Dict[str, Any]) -> bool:
    """Whether this case is one the leg's defect can be seen on at all.

    A REACH IS A CLAIM AND EACH ONE IS DERIVED, not guessed:

    * ``folded`` -- the mutation edits the FOLDED axis's coefficient vectors. An
      unfolded control has no folded axis and the mutation is the identity there,
      so the dead half is a bar: it must be silent, or the mutation is editing
      something it does not claim to.
    * ``signed_zero`` -- the ONE class on which the zero cross terms can change a
      stored word: ``fma(c, z.re, -0.0f)`` differs from ``fma(c, z.re, +0.0f)``
      exactly when ``c * z.re`` is ``-0.0f``, and the class is what puts negative
      zeros in the plane the cross term reads.
    """
    reach = leg.get("reach")
    if reach is None:
        return True
    if reach == "folded":
        return bool(case["fold_axes"])
    if reach == "signed_zero":
        return case["value_class"] == "signed_zero"
    raise ValueError(f"unknown reach {reach!r}")


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend, xp, spec: Dict[str, Any], side: str, courant: float,
             value_class: str, guard: str, expansion, policy: Optional[str],
             host_mutation: Optional[str] = None,
             multi_step: bool = True) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML``
    objects. There is no second transcription on the oracle leg to drift: the
    kernel is compared against the thing it claims to reproduce, word for word,
    from ONE frozen state -- oracle runs, state is restored, kernel runs.
    """
    started = time.time()
    # SEEDED FROM A DIGEST, never from ``hash()``: Python salts ``hash()`` of a
    # tuple containing strings with PYTHONHASHSEED, so a gate seeded that way
    # draws a different fixture every process and a failing case cannot be
    # replayed.
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{side}|{courant}|{value_class}".encode()).digest()[:4],
            "big"))

    fields, layer, grid = build(xp, spec, courant, rng)
    host = seed_state(fields, grid, side, value_class, rng)

    arm_override = None
    if host_mutation == "the_other_expansion_arm":
        arm_override = base.other_arm(
            complex_emitter.EXPANSION_NAMES[
                complex_emitter.normalized_expansion(expansion)])

    case: Dict[str, Any] = {
        "label": spec["label"], "side": side, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend.name,
        "host_mutation": host_mutation,
        "fold_axes": spec["axes"], "mirror_phase": spec["phase"],
        "k_point": list(spec.get("k_point") or (0.0, 0.0, 0.0)),
        "boundaries": list(spec["boundaries"]),
        "structure": structure_facts(grid),
        "operand_census": word_census(host),
        "arm": complex_emitter.EXPANSION_NAMES[
            complex_emitter.normalized_expansion(arm_override or expansion)],
    }

    # THE PREDICATE'S CURRENT ANSWER, recorded rather than acted on. This gate
    # exists to decide whether that answer should change, so it must not be the
    # thing that gates the measurement -- but a record that did not carry it could
    # not show WHICH clause the run was taken against.
    covered, reason = coverage.covers_real_pml_complex_constitutive(
        fields, layer, grid, side, license=_licence_stub(expansion, policy),
        subnormal_policy=policy)
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
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    tables = (tables_for(side, layer)
              if host_mutation in (None, "the_other_expansion_arm")
              else host_mutated_tables(host_mutation, side, layer, grid))
    launch_arm = arm_override if arm_override is not None else expansion
    backend.launch_constitutive(side, fields, layer, grid, launch_arm, tables=tables)
    if backend.name == "cupy":
        cp.cuda.runtime.deviceSynchronize()

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(side)}
    case["single_launch"] = combine(parts)

    # Leg 3: the multi-step. The auxiliary is STATE, and a tree that gets the
    # field right and ``f_w`` wrong is correct for exactly one launch and wrong
    # forever after. ONE ``Fields`` OBJECT, RUN TWICE FROM THE SAME FROZEN STATE.
    if multi_step:
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
            backend.launch_constitutive(side, fields, layer, grid, launch_arm,
                                        tables=tables)
            advance_sources(fields, side)
        if backend.name == "cupy":
            cp.cuda.runtime.deviceSynchronize()
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
    classes = VALUE_CLASSES if product == "full" else ("uniform", "signed_zero")
    out = []
    for spec in specs:
        for side in SIDES:
            for courant in courants:
                for value_class in classes:
                    out.append((spec, side, courant, value_class))
    return out


def run_sweep(backend, xp, results: Dict[str, Any], out_path: str,
              product: str, guard: str, expansion, policy) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, (spec, side, courant, value_class) in enumerate(plan, start=1):
        case = one_case(backend, xp, spec, side, courant, value_class, guard,
                        expansion, policy)
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
            f"planes={case['structure']['folded_planes']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"moved={case['oracle_moved']:.3f} ({case['seconds']:.1f} s)")
    return cases


def mutation_plan(product: str) -> List[Tuple[Dict[str, Any], str, float, str]]:
    """One case per (spec, side, value class) for the mutation legs.

    Held to ONE COURANT deliberately: a mutation leg answers "can this gate see
    this defect at all", and multiplying it by both courants buys repetitions of
    that answer rather than a second question. The VALUE CLASSES are not collapsed
    the same way, because ``signed_zero`` is the reach of two of the legs and
    ``subnormal_band`` is the class the family's certified round measured to
    separate the two EXPANSION ARMS.
    """
    by_label = {spec["label"]: spec for spec in FOLD_SPECS}
    labels = MUTATION_SPEC_LABELS if product == "full" else MUTATION_SPEC_LABELS[:3]
    return [(by_label[label], side, INEXACT_COURANT, value_class)
            for label in labels for side in SIDES
            for value_class in MUTATION_VALUE_CLASSES]


def run_mutations(backend, xp, results: Dict[str, Any], out_path: str,
                  product: str, expansion, policy) -> Dict[str, Any]:
    """Every leg, source and host. A gate that cannot fail certifies nothing.

    THE COMPILE MEMO IS KEYED THROUGH THE SOURCE
    (``compile_cache.kernel_cache_key`` over ``complex_emitter.complex_source``'s
    output), so a mutated body is a miss and reaches the compiler. Every source
    leg records how many sites the rewrite matched IN EACH CONSTITUTIVE SIDE,
    because a leg reporting a pass for a mutation it never applied is worse than
    no leg -- three measured instances on the sibling track.
    """
    out: Dict[str, Any] = {}
    plan = mutation_plan(product)
    for leg in MUTATION_LEGS:
        name = leg["mutation"]
        key = leg["leg"]
        sites: Dict[str, int] = {}
        live_sides = set(SIDES)
        if leg["kind"] == "source":
            transform = SOURCE_MUTATIONS[name]
            sites = constitutive_sites(transform, expansion)
            live_sides = {side for side, count in sites.items() if count > 0}
            if not live_sides:
                out[key] = {"armed": False, "mutation": name, "sites": sites,
                            "why": ("the rewrite matched 0 sites in either "
                                    "constitutive source; the mutation and the "
                                    "emitted text have drifted apart"),
                            "verdict": "NOT ARMED", "as_required": False}
                log(f"[mut] {key}: NOT ARMED (sites {sites})")
                results["mutations"] = out
                save(results, out_path)
                continue
            backend.set_source_transform(transform)
        legs: List[Dict[str, Any]] = []
        try:
            for spec, side, courant, value_class in plan:
                if side not in live_sides:
                    continue
                host_mutation = name if leg["kind"] == "host" else None
                legs.append(one_case(backend, xp, spec, side, courant, value_class,
                                     "fmad_false", expansion, policy,
                                     host_mutation=host_mutation))
        finally:
            if leg["kind"] == "source":
                backend.set_source_transform(None)
        verdict = _adjudicate(leg, legs)
        out[key] = {"armed": True, "mutation": name, "kind": leg["kind"],
                    "sites": sites, "sides_scored": sorted(live_sides),
                    "why": leg["why"], **verdict, "cases": legs}
        log(f"[mut] {key}: {verdict['verdict']} "
            f"caught {verdict['caught']}/{verdict['ran']} "
            f"sides={sorted(live_sides)} sites={sites} "
            f"as_required={verdict['as_required']}")
        results["mutations"] = out
        save(results, out_path)
    return out


def _adjudicate(leg: Dict[str, Any], cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The Dcyl gate's adjudicator, with THIS gate's reach.

    ``adjudicate`` resolves ``_reaches`` through its own module, so a
    ``caught_where_live`` leg with reach ``folded`` would raise there. The
    reach-free branches are identical and are delegated; only the live/dead split
    is re-decided here, against :func:`_reaches` above.
    """
    if leg["expect"] != "caught_where_live":
        return adjudicate(leg, cases)
    scored = [c for c in cases if not c.get("skipped")]
    caught = [c for c in scored if leg_caught(c)]
    out: Dict[str, Any] = {"ran": len(scored), "caught": len(caught),
                           "uncaught": len(scored) - len(caught),
                           "expect": leg["expect"]}
    if not scored:
        # NO LEGS is its own verdict, never "as required". A mutation that was
        # never run has not been shown inert; it has not been asked.
        out["verdict"] = "NO LEGS"
        out["as_required"] = False
        return out
    live = [c for c in scored if _reaches(leg, c)]
    dead = [c for c in scored if not _reaches(leg, c)]
    live_caught = [c for c in live if leg_caught(c)]
    dead_caught = [c for c in dead if leg_caught(c)]
    out["reach"] = leg["reach"]
    out["live"] = f"{len(live_caught)}/{len(live)}"
    out["dead"] = f"{len(dead_caught)}/{len(dead)}"
    out["dead_is_a_bar"] = bool(leg.get("dead_must_be_identical"))
    out["as_required"] = bool(
        live and len(live_caught) == len(live)
        and (not dead_caught if leg.get("dead_must_be_identical") else True))
    out["verdict"] = ("CAUGHT WHERE LIVE" if out["as_required"]
                      else "NO LIVE LEGS" if not live else "PARTIAL")
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any], backend_certifies: bool) -> Dict[str, Any]:
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

    # EVERY CASE MUST HAVE BEEN COMPLEX STORAGE. A gate for the complex family
    # that quietly built real volumes would be measuring the other kernel, and
    # nothing else in the artifact would say so.
    if any(c["predicate_today"]["reason"].startswith("real float32 storage")
           for c in scored):
        reasons.append("a scored case was refused as real float32 storage; the "
                       "fixture did not build complex volumes")
    # AND EVERY FOLDED ONE MUST HAVE REACHED THE FOLD CLAUSE. If a folded case
    # were refused for something upstream of the fold, the run would be measuring
    # a configuration the widening does not touch and reporting it as though it
    # did.
    off_clause = sorted({c["predicate_today"]["reason"] for c in folded
                         if not c["predicate_today"]["reason"].startswith(
                             ("mirror symmetry", "axis "))
                         and not c["predicate_today"]["covered"]})
    if off_clause:
        reasons.append(f"folded cases were refused by clauses other than the "
                       f"fold's: {off_clause}")

    # BOTH FOLDED TERMINATIONS. A folded PERIODIC axis stores one slot past MEEP's
    # owned window and a METALLIC one does not, so they are structurally different
    # arrays and a sweep carrying one would answer half the question.
    terminations = set()
    for case in folded:
        structure = case["structure"]
        for axis, mirrored in enumerate(structure["mirrored"]):
            if mirrored:
                terminations.add("metallic" if structure["metallic"][axis]
                                 else "periodic")
    if terminations != {"periodic", "metallic"}:
        reasons.append(f"folded terminations scored were {sorted(terminations)}, "
                       f"not both of periodic and metallic")

    # EVERY AXIS MUST HAVE BEEN FOLDED SOMEWHERE. The coefficient index runs on
    # the slowest stride for X and the fastest for Z, and a decomposition defect
    # that confuses them is only visible if both were folded.
    folded_axes = set()
    for case in folded:
        for axis, mirrored in enumerate(case["structure"]["mirrored"]):
            if mirrored:
                folded_axes.add("xyz"[axis])
    if folded_axes != {"x", "y", "z"}:
        reasons.append(f"folded axes scored were {sorted(folded_axes)}, not all "
                       f"three")

    # A PHASED FOLDED CASE. Five of the eight corpus rows carry a Bloch phase on
    # an unfolded axis, and the constitutive kernel takes no phase argument -- so
    # a sweep of unphased grids alone would license the corpus's own shape by
    # extrapolation from one it never ran.
    phased = [c for c in folded if c["structure"]["has_bloch"]]
    if not phased:
        reasons.append("no PHASED folded case was scored; five of the eight corpus "
                       "rows this gate is about carry a Bloch phase on an unfolded "
                       "axis")

    # THE PLANE COUNT IS MEASURED FROM WHAT RAN, never quoted from the case table.
    plane_counts = sorted({c["structure"]["folded_planes"] for c in folded})
    max_planes = max(plane_counts) if plane_counts else 0

    # THE SIGNED-ZERO CLASS MUST HAVE CONTAINED SIGNED ZEROS. Without them the two
    # zero-cross-term legs have no reach and their verdicts are about nothing.
    signed = [c for c in scored if c["value_class"] == "signed_zero"]
    negative_zeros = sum(int(c["operand_census"].get("negative_zeros", 0))
                         for c in signed)
    if signed and negative_zeros == 0:
        reasons.append("the signed_zero cases carried no negative zero words; the "
                       "zero cross terms had nothing to act on")
    band = [c for c in scored if c["value_class"] == "subnormal_band"]
    subnormals = sum(int(c["operand_census"].get("subnormals", 0)) for c in band)
    if band and subnormals == 0:
        reasons.append("the subnormal_band cases carried no subnormal words")

    for key, leg in results.get("mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"mutation {key} was not armed: {leg.get('why')}")
        elif not leg.get("as_required"):
            reasons.append(f"mutation {key} is {leg['verdict']} "
                           f"({leg.get('caught')}/{leg.get('ran')}), "
                           f"expected {leg.get('expect')}")

    licence = results.get("expansion_licence") or {}
    if not licence.get("usable"):
        reasons.append("no usable expansion licence: the complex multiply's arm "
                       "is a measured platform fact and may not be guessed")

    if not backend_certifies:
        reasons.append("this backend compiles nothing through NVRTC and certifies "
                       "nothing; a laptop leg measures the transcription only")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    control_diverged = [c for c in control if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored at the "
                    "inexact courant" if not control else
                    "the contraction guard changed an answer on this sub-step"
                    if control_diverged else
                    "MEASURED INERT on these cases: the unguarded leg was "
                    "bit-identical too. Read together with the compiled-image "
                    "comparison, which says whether the flag had anything to do"),
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
        "folded_axes": sorted(folded_axes),
        "phased_folded_cases": len(phased),
        # THE CAP, MEASURED. The only number a predicate's plane clause may be
        # set from.
        "folded_planes_scored": plane_counts,
        "max_folded_planes_scored": max_planes,
        "value_classes": sorted({case["value_class"] for case in scored}),
        "negative_zero_words_in_signed_zero_cases": negative_zeros,
        "subnormal_words_in_band_cases": subnormals,
        "claim": ("the SHIPPED CUDA COMPLEX constitutive kernels "
                  "(update_H_pml_complex_bloch / ..._E_...) are "
                  "bit-identical to stepping.update_H/update_E on a MIRROR-FOLDED "
                  "complex64 grid, per sub-step, at both folded terminations, over "
                  "folded X/Y/Z, both plane parities, odd and even full counts, "
                  f"phased and unphased, and up to {max_planes} simultaneous fold "
                  "planes, with no change to the emitted source"),
        "does_not_claim": [
            "nothing dispatches these kernels; no module in meep_gpu imports "
            "cuda_kernels at all, so a widened predicate licenses a MEASUREMENT "
            "and not a production step",
            "the predicate is NOT changed by this gate; the change it licenses is "
            "reported, not made",
            "the complex CURL on a fold is REFUSED for a real reason and nothing "
            "here touches it: cshift_up/cshift_dn are a neighbour stencil, a fold "
            "changes the ghost rule and the ownership mask, and the emitted curl "
            "takes bc_x/bc_y/bc_z with no code for a mirror. All sixteen curl "
            "slots on the eight corpus rows stay refused, so not one of them "
            "becomes a row the hand-CUDA track can step end to end",
            "the REAL constitutive pair's fold is a different kernel and a "
            "different gate (gate_cuda_folded_constitutive.py); nothing here "
            "changes covers_real_pml_constitutive",
            f"more than {max_planes} simultaneous mirror planes were not swept",
            "no throughput claim: this is a correctness gate and times nothing",
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
    parser.add_argument("--backend", choices=("cupy", "host", "numpy"),
                        default="cupy")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--expansion-probe", default=None,
                        help="the probe artifact the expansion licence is cut from")
    parser.add_argument("--import-meep-for-host-policy", action="store_true",
                        help=("import MEEP first so the HOST half of a 'flush' "
                              "policy can be attained; strict install refuses "
                              "otherwise"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--falsify", default=None, choices=tuple(SOURCE_MUTATIONS),
                        help=("plant this defect for the WHOLE RUN, sweep included, "
                              "so the RELEASE VERDICT ITSELF is shown to flip. A "
                              "gate whose individual legs can fail but whose "
                              "verdict cannot has not been shown to be a gate"))
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_complex_folded_constitutive",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("does _complex_grid_refusal's mirror-symmetry clause describe "
                     "a divergence in the CONSTITUTIVE arithmetic, or is it "
                     "inherited from the complex CURL through the one function "
                     "the two predicates share?"),
        "licence_subject": probe.source_digest(os.path.join(
            _REPO_API, "meep_gpu", "cuda_kernels", "complex_emitter.py")),
    }

    if args.backend == "cupy":
        if cp is None:
            log("[fatal] --backend cupy but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        # THE OBSERVER GOES IN BEFORE THE POLICY, and the order is the design:
        # under 'keep' the policy's strip wraps this, so it records the option
        # tuple NVRTC was really given (post-strip).
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {
            "python": sys.version.split()[0], "numpy_version": np.__version__,
            "note": ("this backend compiles nothing through NVRTC and certifies "
                     "nothing about the device")}
    save(results, args.out)

    licence = base.load_licence(args.expansion_probe, args.subnormal_policy)
    results["expansion_licence"] = licence
    if not licence["usable"]:
        # A REFUSAL RATHER THAN A DEFAULT. The arm is a measured platform fact and
        # this family derives it from nothing; a run without one would be a
        # guessed constexpr compiled into a binary.
        results["status"] = "refused: no usable expansion licence"
        results["summary"] = {"released": False, "reasons": [
            "no usable expansion licence: cut a probe artifact with "
            "cut_expansion_probe.py under the SAME subnormal policy this run "
            "installs, and pass it with --expansion-probe"],
            "licence_refusals": (licence.get("verdict") or {}).get("refusals", []),
            "policy_reasons": licence.get("policy_reasons", [])}
        save(results, args.out)
        log("[done] REFUSED: no usable expansion licence")
        return 2
    expansion = licence["arm"]
    results["expansion_arm"] = expansion
    results["expansion_arm_basis"] = licence.get("basis")
    log(f"[licence] arm={expansion} basis={licence.get('basis')} "
        f"record={(licence.get('record_sha256') or '')[:12]}")

    backend = base.build_backend(args.backend)
    xp = cp if args.backend == "cupy" else _NumpyWearingCupysName()
    results["backend_certifies"] = bool(backend.certifies)
    save(results, args.out)

    try:
        if args.falsify:
            # THE VERDICT'S OWN FALSIFICATION LEG. Every mutation leg below shows
            # that a LEG can fail; this shows that the RELEASE DECISION can.
            sites = constitutive_sites(SOURCE_MUTATIONS[args.falsify], expansion)
            backend.set_source_transform(SOURCE_MUTATIONS[args.falsify])
            results["falsification"] = {
                "planted": args.falsify, "sites": sites,
                "requires": "summary.released must come out FALSE"}
            log(f"[falsify] planted {args.falsify} sites={sites}: the release "
                f"verdict must now be FALSE")
            save(results, args.out)

        for guard, options, _is_primary in GUARD_SETS:
            if args.backend != "cupy" and guard != "fmad_false":
                continue  # there is no NVRTC on this leg to guard
            backend.set_guard(options)
            log(f"[guard] {guard} options={options}")
            run_sweep(backend, xp, results, args.out, args.product, guard,
                      expansion, args.subnormal_policy)

        backend.set_guard(("--fmad=false",))
        if args.backend == "cupy":
            # THE COMPILED-IMAGE COMPARISON, which is what makes an identical
            # output control readable: two bit-identical NVRTC images cannot
            # produce different words, so "inert" and "unmeasured" come apart.
            results["guard_effect"] = backend.guard_effect(expansion)
            save(results, args.out)

        if not args.skip_mutations and not args.falsify:
            # NOT UNDER --falsify: ``run_mutations`` clears the source transform in
            # its own ``finally``, which would silently disarm the planted defect
            # partway through the run.
            run_mutations(backend, xp, results, args.out, args.product,
                          expansion, args.subnormal_policy)
    finally:
        backend.restore()

    if args.backend == "cupy":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results, bool(backend.certifies))
    if args.falsify:
        results["summary"]["falsification"] = {
            "planted": args.falsify,
            "verdict_flipped": not results["summary"]["released"],
            "reading": ("this run is a FALSIFICATION and is not a release under "
                        "any reading; what it measures is whether the release "
                        "decision can come out False at all"),
        }
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} "
        f"scored={verdict['scored_cases']} folded={verdict['folded_cases']} "
        f"planes={verdict['folded_planes_scored']} "
        f"terminations={verdict['folded_terminations']} "
        f"phased={verdict['phased_folded_cases']} "
        f"single_identical={verdict['single_launch_identical']} "
        f"multi_identical={verdict['multi_step_identical']}/"
        f"{verdict['multi_step_cases']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
