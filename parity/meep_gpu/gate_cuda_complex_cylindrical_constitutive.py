"""Sub-step byte-identity gate for the SHIPPED CUDA COMPLEX constitutive pair ON A Dcyl GRID.

THE CLAIM UNDER TEST, and it is a claim about a PREDICATE, not about a kernel.
``meep_gpu/cuda_kernels/complex_pml_kernels.py`` ships ``update_fused_pml_complex``,
which launches ``update_H_pml_complex_bloch`` /
``update_E_pml_complex_bloch`` from ``complex_emitter``'s
``_CONSTITUTIVE_TEMPLATE``. This gate does not change one byte of either. It asks
whether the refusal ``covers_real_pml_complex_constitutive`` inherits from
``coverage._complex_grid_refusal``, the one clause function it shares with
``covers_real_pml_complex_curl`` --

    "cylindrical (Dcyl): prefix-sum radial derivative and axis-row rules"

-- is a refusal the CONSTITUTIVE arithmetic requires, or a refusal inherited from
the complex CURL, which shares that one function with it. For the curl the refusal
is REAL and stays: ``cshift_up``/``cshift_dn`` are a neighbour stencil and Dcyl
replaces the radial derivative with a prefix sum and adds the axis-row rules. The
constitutive sub-step has no neighbour to shift.

WHY THE READING SAYS IT IS NOT REQUIRED HERE (every fact read off the source, and
checked line by line before this file was written):

1. THE SUB-STEP READS NO NEIGHBOUR. ``stepping._apply_constitutive_pml``
   (stepping.py:2112-2145) is ``fw_previous = fw.copy(); fw[...] = source;
   field += kps*fw; field -= kms*fw_previous`` -- four whole-array operations, no
   shift, no ghost, no mask. Dcyl changes the RADIAL DERIVATIVE and the AXIS ROW;
   this sub-step has neither.
2. ``update_H`` (stepping.py:907-925) and ``update_E`` (:926-995) contain ZERO
   occurrences of "cylindrical", "is_axis", "m" or "axis_zero". The Dcyl repairs
   are separate passes ``step_B`` and ``step_D`` run: ``_cylindrical_axis_zero_B``
   is called at :376 and ``_cylindrical_axis_zero_D`` at :457, both inside the
   CURL, and nowhere else in the module.
3. ``_mask_non_owned_cells`` (stepping.py:1912) has exactly three call sites --
   ``step_B`` (:397), ``step_D`` (:479) and ``_bfast_term`` (:929). No
   constitutive function is among them, on any coordinate system.
4. THE EMITTED KERNEL SAYS THE SAME THING IN ITS OWN CHARACTERS.
   ``complex_emitter._CONSTITUTIVE_TEMPLATE`` takes no ``bc_*``, no ``ph_*``, no
   phase pair and no ``dtdx``: its whole body is three ``constitutive_apply``
   calls on ``kps_x[i]``/``kps_y[j]``/``kps_z[k]``. There is no place in it for a
   ghost rule to be wrong, which is why the fold, the wrap and the axis row cannot
   reach it -- and why the ONLY thing Dcyl could still move is the coefficient
   INDEX, which is what this gate arms.
5. THE EXTENT MOVES BOTH OPERANDS TOGETHER, which is the refusal's premise where
   it applies to an index at all, and is TRUE. ``PML._compute_coefficients`` builds
   every kps/kms vector from ``grid.nx``/``ny``/``nz`` -- the STORED extent -- and
   the field arrays are allocated on the same ``grid.shape``. On Dcyl the phi axis
   is invariant with n = 1 and r and z are ordinary extents, so the kernel's
   ``i = idx/(ny*nz)`` indexes a table of exactly the length the grid left.
6. THE REAL PAIR WAS MEASURED ON THIS EXACT QUESTION AND ADMITTED
   (:data:`coverage.CONSTITUTIVE_CYLINDRICAL_ADMISSION`, 96/96 on the GPU host,
   2026-08-20). THAT IS NOT TRANSFERABLE. These are different kernel bytes --
   complex64 word pairs, the zero cross terms, the arm -- emitted by a different
   template and compiled under an expansion licence the real pair does not take.
   The sibling Triton track measured its own complex Dcyl constitutive at 480/480
   identity rows with 0 differing uint32 words
   (``triton_kernels/cylindrical_complex.py``). That is a reason to measure this
   one, not to inherit either verdict.

=============================================================================
WHAT IS ARMED SO THE ADMISSION CANNOT BE A NO-OP
=============================================================================

``reverse_radial_axis_coefficients`` is the refusal reason ITSELF armed as a
defect: it reverses the r axis's kps/kms vectors IN BOUNDS, so a cell reads the
wrong coefficient without reading out of memory. If the extent really did move a
coefficient index the way the refusal says, this is the mutation that would show
it -- and a gate that admits Dcyl without catching this would be admitting on an
absence of evidence. ``radial_axis_absorbs`` is the floor that keeps it
load-bearing: a profile that is the identity everywhere reverses to itself, and
such a case is refused before it is scored rather than counted as a miss. It is
the same mutation, spelled the same way, as the real pair's round, so the two are
commensurable to the case.

THE COMPLEX HALF-PLANE DEFECT CLASS IS ARMED TOO, and it is this family's own.
``complex64`` is a float2 and every store in the emitted kernel is two word
stores; a store that writes one plane and leaves the other is a real defect class
in this tree (a wall clear that zeroes only the real plane), and on an
element-wise sub-step it is the only shape a plane-selective defect can take.
``half_plane_field_store`` and ``half_plane_auxiliary_store`` are those two
stores, each crippled to the real plane alone.

``fold_the_zero_cross_terms`` and ``plane_wise_scaling`` are carried from the
family's own certified round unchanged, because they are what distinguishes a
transcribed complex multiply from a plane-wise one, and the ``signed_zero`` value
class is the one that can see them.

USAGE

    python -u gate_cuda_complex_cylindrical_constitutive.py \
        --backend cupy --product full --subnormal-policy keep \
        --expansion-probe <probe>.json \
        --out results/cuda_complex_cylindrical_constitutive_<date>/gate.json

    python -u gate_cuda_complex_cylindrical_constitutive.py --backend host \
        --product full --expansion-probe \
        results/expansion_probe_2026-08-17/expansion_probe_keep.json \
        --out /tmp/complex_cyl_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten after
every case, so an interrupted run keeps everything up to the failure.
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

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter, coverage  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260820

#: Consecutive launches in the multi-step leg. 60 is the budget every certified
#: hand-CUDA record in this tree is cut at, including the complex family's own
#: 2026-08-19 release and the real Dcyl constitutive round this gate is derived
#: from. "Identical for N steps" is a claim about N, and the auxiliary ``f_w`` is
#: STATE: a tree that gets the field right and ``f_w`` wrong is correct for
#: exactly one launch and wrong forever after.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence live. Held fixed, ``f_w`` settles
#: and a 60-step leg becomes a slow single-step leg. The same exact complex64
#: scale is applied to the SAME bytes on both paths, so it cancels out of the
#: comparison; not exactly representable, deliberately -- an exact scale would
#: leave the mantissas untouched.
_ADVANCE = np.complex64(complex(0.97, -0.03))

SIDE_ARRAYS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "H": {"targets": ("Hx", "Hy", "Hz"),
          "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz")},
    "E": {"targets": ("Ex", "Ey", "Ez"),
          "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz")},
}


def outputs(side: str) -> Tuple[str, ...]:
    """The six complex volumes this sub-step writes, compared as float32 words."""
    spec = SIDE_ARRAYS[side]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def state_names(side: str) -> Tuple[str, ...]:
    return outputs(side) + tuple(SIDE_ARRAYS[side]["sources"])


# ---------------------------------------------------------------------------
# THE CASE TABLE
# ---------------------------------------------------------------------------
#
# Dcyl grids over BOTH z terminations and BOTH |m| classes, plus the Cartesian
# controls the certified complex round already covers, carried so a divergence in
# the Dcyl rows cannot be blamed on the fixture.
#
# ``m`` IS SWEPT EVEN THOUGH THIS SUB-STEP CANNOT READ IT. That is the point: the
# claim is that update_H/update_E are element-wise and carry no cylindrical
# branch, and a sweep that fixed m at 0 would leave "except at |m| >= 1" open. The
# |m| classes differ in the CURL (axis-row increment at |m| = 1, near-axis zeroing
# at |m| >= 2), so if any of that leaked into the constitutive sub-step it would
# show as an m-dependent divergence here.
#
# NEGATIVE m IS SWEPT because seven of the sixteen corpus rows this refusal costs
# carry m = -1, and a sweep that took |m| as the whole story would be an
# assumption about ``Grid`` rather than a measurement.
#
# ``cell`` is in units of 1/resolution at resolution 1.0, so the numbers ARE the
# cell counts. Grid owns the r axis's boundary pair -- the axis at r = 0 and a
# metallic wall at r_max -- so only z is chosen (grid.py:498-508).
#
# EVERY CASE IS COMPLEX STORAGE AND UNPHASED, and that is the corpus, not a
# simplification: all sixteen Dcyl complex rows in the 2026-08-20 census carry
# ``has_bloch`` FALSE and ``force_complex_fields`` TRUE. Dcyl Bloch is z-only in
# MEEP (grid.py:654-657) and no corpus row uses it; a phased Dcyl case would be
# measuring a configuration the corpus does not contain, and the constitutive
# kernel takes no phase argument in any event.

CYL_SPECS: Tuple[Dict[str, Any], ...] = (
    # The Cartesian controls. Already certified by the complex family's own round;
    # carried so a divergence in the Dcyl rows cannot be blamed on the fixture.
    {"label": "cartesian_periodic", "cylindrical": False, "m": 0,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "cartesian_metallic", "cylindrical": False, "m": 0,
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (8.0, 10.0, 12.0)},

    # m = 0, both z terminations.
    {"label": "cyl_m0_z_metallic", "cylindrical": True, "m": 0,
     "z_kind": "metallic", "cell": (12.0, 0.0, 16.0)},
    {"label": "cyl_m0_z_periodic", "cylindrical": True, "m": 0,
     "z_kind": "periodic", "cell": (12.0, 0.0, 16.0)},

    # |m| = 1: the axis-row increment class, BOTH SIGNS.
    {"label": "cyl_m1_z_metallic", "cylindrical": True, "m": 1,
     "z_kind": "metallic", "cell": (12.0, 0.0, 16.0)},
    {"label": "cyl_mneg1_z_metallic", "cylindrical": True, "m": -1,
     "z_kind": "metallic", "cell": (12.0, 0.0, 16.0)},
    {"label": "cyl_mneg1_z_periodic", "cylindrical": True, "m": -1,
     "z_kind": "periodic", "cell": (12.0, 0.0, 16.0)},

    # |m| >= 2: the near-axis zeroing class. m = 2, 3 and 5 all appear in the
    # corpus (perturbation_theory, ring-cyl, test_pml_cyl).
    {"label": "cyl_m2_z_metallic", "cylindrical": True, "m": 2,
     "z_kind": "metallic", "cell": (12.0, 0.0, 16.0)},
    {"label": "cyl_m3_z_periodic", "cylindrical": True, "m": 3,
     "z_kind": "periodic", "cell": (12.0, 0.0, 16.0)},
    {"label": "cyl_m5_z_metallic", "cylindrical": True, "m": 5,
     "z_kind": "metallic", "cell": (14.0, 0.0, 10.0)},

    # ODD CELL COUNTS on both resolved axes, and a shape whose r and z extents
    # differ from every case above: a coefficient-index defect that happens to be
    # invisible at one aspect ratio is not invisible at two. 13*1*17 = 221 complex
    # cells is also not a multiple of the 256-lane block, so the tail guard
    # ``if (idx >= nx*ny*nz) return;`` decides real threads there.
    {"label": "cyl_m1_odd_counts", "cylindrical": True, "m": 1,
     "z_kind": "metallic", "cell": (13.0, 0.0, 17.0)},
    {"label": "cyl_m0_wide_r", "cylindrical": True, "m": 0,
     "z_kind": "periodic", "cell": (20.0, 0.0, 8.0)},
)

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient, so it is a real second draw of the tables and not only
#: a rounding probe.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

#: ``uniform`` is the physical band. ``subnormal_band`` is what a float32 subnormal
#: policy decides the fate of, and it is the class the complex family's own round
#: measured to separate the two expansion arms on a zero-imaginary orientation.
#: ``signed_zero`` is the class the ZERO CROSS TERMS exist for, and it is a real
#: operand class rather than a contrivance: a complex word whose real plane is an
#: exact zero and whose imaginary plane is not is what a flushed subnormal, a
#: freshly-initialised volume and a wall plane all produce, and it is the only
#: class on which ``z.im * 0.0f`` can change a stored word.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "signed_zero")

SIDES: Tuple[str, ...] = ("H", "E")

#: ``--fmad=false`` is CORRECTNESS on this sub-step in principle -- both
#: accumulations and the E side's ``D*inv_eps`` are contraction candidates the
#: array path rounds twice. The control is REPORTED, NOT GATED, for the reason the
#: family's own release records: under the licensed ``FMA_V1`` arm every product
#: in the emitted source is already an explicit ``__fmaf_rn``, which no contraction
#: flag reaches, so the flag has nothing here to do and the images come out
#: identical. An identical control is a FINDING on this family and a failure on
#: the real one; the artifact says which it measured rather than implying evidence
#: it does not have.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    Everything else the fixture exercises -- the stored extent, the coefficient
    vector lengths, the dtype and contiguity, the complex64 word view -- is a real
    object either way. Same shim the sibling gates use, so the tracks' laptop legs
    are commensurable.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float, rng):
    """A frozen ``(fields, layer, grid)`` triple for one case, Dcyl or Cartesian.

    ``force_complex_fields=True`` ALWAYS. Every Dcyl row this gate is about is a
    complex-storage-at-k=0 row (the 2026-08-20 census records ``has_bloch`` FALSE
    on all sixteen), which is exactly the class ``force_complex_fields`` names and
    half of what the complex family exists for.

    ON Dcyl, ``Grid`` OWNS THE r AXIS'S BOUNDARY PAIR -- the axis at r = 0 and a
    metallic wall at r_max -- so only z is chosen (grid.py:498-508), and the PML
    is asked for the HIGH r face only for the same reason: cell 0 is the axis, a
    boundary condition rather than a wall. Same rule the sibling Dcyl gates use,
    restated rather than imported so the gates cannot drift apart silently through
    a shared helper.

    THREE INDEPENDENT inverse-epsilon volumes on the E side, never one array bound
    thrice: binding one volume for all three components is the defect
    ``fields.inv_eps`` (the Ez view, fields.py:1259-1260) invites, and a fixture
    that handed one array to all three could not see it. Drawn AWAY FROM 1.0 --
    against a table of ones a dropped inverse epsilon is bit-identical.
    """
    if not spec["cylindrical"]:
        grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                    boundaries=tuple(spec["boundaries"]),
                    xp=xp, courant=courant)
        thickness = tuple((0, 0) if grid.shape[axis] < 6 else (2, 2)
                          for axis in range(3))
    else:
        grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                    cylindrical=True, m=int(spec["m"]),
                    boundaries={"z": spec["z_kind"]},
                    xp=xp, courant=courant)
        thickness = {"x": (0, max(2, int(grid.shape[0]) // 4)),
                     "z": max(2, int(grid.shape[2]) // 4)}
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


def seed_state(fields, grid, side: str, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Seed every complex volume the sub-step touches; return the host words.

    THE AUXILIARIES START NONZERO in every class. A zero ``f_w`` makes
    ``kms * prev`` exactly zero on the first launch whatever ``kms`` holds, so a
    mis-indexed coefficient would only show from step two -- precisely what the
    multi-step leg exists to catch, and no reason to hide it from the single
    launch as well.

    THE MATERIAL STAYS NORMAL under the band class: driving inverse epsilon into
    the band too would make every product underflow and the leg would measure the
    fixture rather than the policy's reach into this arithmetic.

    THE TWO PLANES ARE ASSIGNED, NEVER COMBINED AS ``re + 1j*im``. ``1j * im``
    carries a real part of ``0.0 * im``, so ``re + 1j*im`` computes
    ``(-0.0) + (+0.0) = +0.0`` and DESTROYS every negative zero in the real plane
    -- and the signed-zero class exists precisely to put them there. Measured on
    the sibling round before it was reasoned about.
    """
    xp = grid.xp
    shape = tuple(grid.shape)
    names = state_names(side)
    if value_class == "uniform":
        planes = {name: (rng.uniform(-1.0, 1.0, size=shape),
                         rng.uniform(-1.0, 1.0, size=shape)) for name in names}
    elif value_class == "subnormal_band":
        real = subnormal_band_hosts(names, shape, rng)
        imag = subnormal_band_hosts(names, shape, rng)
        planes = {name: (real[name], imag[name]) for name in names}
    elif value_class == "signed_zero":
        planes = {}
        for name in names:
            values = rng.uniform(-1.0, 1.0, size=shape)
            zeroed = rng.integers(0, 2, size=shape) == 0
            signs = np.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0)
            planes[name] = (np.where(zeroed, signs, values),
                            rng.uniform(-1.0, 1.0, size=shape))
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    host: Dict[str, np.ndarray] = {}
    for name, (real_plane, imag_plane) in planes.items():
        values = np.empty(shape, dtype=np.complex64)
        values.real = real_plane.astype(np.float32)
        values.imag = imag_plane.astype(np.float32)
        host[name] = values
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def word_census(host: Dict[str, np.ndarray]) -> Dict[str, int]:
    """The operand census over the float32 WORDS of the complex volumes.

    ``operand_census`` counts subnormals, zeros and negative zeros in float32
    arrays; a complex volume's words are exactly that array, so this class's
    non-vacuity is measured on the same instrument the real families use.
    """
    return probe.operand_census({name: values.view(np.float32).ravel()
                                 for name, values in host.items()})


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

    ``complex_emitter.HALF_INTEGER`` is asked here rather than hard-coding the
    suffix, because it is the same table the SHIPPED launcher asks
    (``complex_pml_kernels.complex_constitutive_tables``) -- the pairing cannot
    disagree between the two, and getting it backwards is a half-cell error in the
    absorber profile: converged, smooth and wrong.
    """
    suffix = "_h" if complex_emitter.HALF_INTEGER[side] else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def wrong_sub_lattice_tables(side: str, layer) -> Dict[str, Any]:
    """The OTHER side's sub-lattice: a half-cell error, not a crash."""
    suffix = "" if complex_emitter.HALF_INTEGER[side] else "_h"
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def reversed_radial_axis_tables(side: str, layer, grid) -> Dict[str, Any]:
    """THE REFUSAL REASON, ARMED. The r axis's profile read backwards.

    "cylindrical (Dcyl): prefix-sum radial derivative and axis-row rules" is the
    predicate's stated reason for refusing Dcyl. The prefix-sum half cannot reach
    an element-wise sub-step at all; the half that could is the coefficient INDEX
    on the radial axis. This mutation makes that index wrong -- IN BOUNDS, by
    reversing the vector rather than by substituting a longer or shorter one, so
    the leg measures a WRONG ANSWER and not a memory fault.

    It is also the only thing that makes ``radial_axis_absorbs`` load-bearing: an
    axis whose profile is the identity everywhere reverses to itself, and the leg
    would come back UNCAUGHT for a reason about the fixture rather than about the
    kernel. The floor refuses such a case before it is scored.

    ON A CARTESIAN CONTROL there is no radial axis and this is a no-op, which is
    why the control is excluded from scoring it.
    """
    tables = dict(tables_for(side, layer))
    if not bool(getattr(grid, "cylindrical", False)):
        return tables
    xp = grid.xp
    for stem in ("kps", "kms"):
        key = f"{stem}_x"
        tables[key] = xp.ascontiguousarray(tables[key][::-1])
    return tables


def swapped_kps_kms_tables(side: str, layer) -> Dict[str, Any]:
    """``(kap+sig)`` and ``(kap-sig)`` exchanged: the absorber runs backwards."""
    base_tables = tables_for(side, layer)
    out = dict(base_tables)
    for axis in "xyz":
        out[f"kps_{axis}"] = base_tables[f"kms_{axis}"]
        out[f"kms_{axis}"] = base_tables[f"kps_{axis}"]
    return out


def host_mutated_tables(name: str, side: str, layer, grid) -> Dict[str, Any]:
    if name == "reverse_radial_axis_coefficients":
        return reversed_radial_axis_tables(side, layer, grid)
    if name == "swap_constitutive_sublattice":
        return wrong_sub_lattice_tables(side, layer)
    if name == "swap_kps_kms":
        return swapped_kps_kms_tables(side, layer)
    raise KeyError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# The source mutations this family's constitutive half needs
# ---------------------------------------------------------------------------
#
# SIX ARE THE FAMILY'S OWN, imported from ``gate_cuda_complex`` rather than
# respelled: a second spelling of a defect the certified record was cut against is
# one too many, and a rewrite that drifted from the shipped text would report NOT
# ARMED for a reason about this file. Two are NEW and are this gate's, because the
# complex half-plane store has no analogue on the real track.

def m_half_plane_field_store(text: str) -> Tuple[str, int]:
    """Store only the REAL plane of the constitutive result; keep the old imaginary.

    THE COMPLEX FAMILY'S OWN DEFECT CLASS, and on an element-wise sub-step it is
    the only shape a plane-selective defect can take. ``complex64`` is a float2 and
    ``cf_store`` is two word stores; a store that writes one plane and leaves the
    other is a real defect in this tree (a wall clear that zeroed only the real
    plane), and it is invisible to any comparison that looks at magnitudes.

    Wrong on EVERY cell whose imaginary part the step would have moved -- which is
    every cell with a nonzero imaginary source -- so the ``uniform`` class alone
    catches it and the reach is universal.
    """
    return base._sub(
        r"    a = cf_sub\(a, mul_coefficient_left\(kms, prev\)\);\n"
        r"    cf_store\(f, idx, a\);",
        "    a = cf_sub(a, mul_coefficient_left(kms, prev));\n"
        "    a.im = cf_load(f, idx).im;\n"
        "    cf_store(f, idx, a);", text)


def m_half_plane_auxiliary_store(text: str) -> Tuple[str, int]:
    """Store only the REAL plane of ``f_w``; keep the previous imaginary word.

    The same defect one line up, on the AUXILIARY. It is the more dangerous half:
    ``f_w`` is state, so a half-written auxiliary is a defect the field does not
    show until the next launch reads it back, which is what the 60-launch leg is
    for. On the FIRST launch it shows immediately because ``f_w`` is compared on
    every case -- the sibling track measured a dropped auxiliary store at 120/120
    UNCAUGHT when the auxiliary was not compared.
    """
    return base._sub(
        r"    cf prev = cf_load\(fw, idx\);\n    cf_store\(fw, idx, src\);",
        "    cf prev = cf_load(fw, idx);\n"
        "    { cf half = src; half.im = prev.im; cf_store(fw, idx, half); }", text)


def n_reload_the_constitutive_store(text: str) -> Tuple[str, int]:
    """A NULL mutation: store the target, load it back, store it again.

    A float32 written to global memory and read back is the identity on the bits.
    A leg that CAUGHT this would mean the harness reports differences that are not
    arithmetic, which is the thing an uncaught null is there to rule out. The
    family's own ``reload_the_target_from_memory`` needles the CURL's store and
    would not match here at all -- which is why this one exists rather than being
    borrowed and reported NOT ARMED.
    """
    return base._sub(r"    cf_store\(f, idx, a\);\n\}",
                     "    cf_store(f, idx, a);\n"
                     "    cf_store(f, idx, cf_load(f, idx));\n}", text)


#: The device-text defects. Every one must MATCH the constitutive template on BOTH
#: sides -- a mutation whose needle only appears in the curl would report a pass
#: for a leg that exercised the shipped bytes, which is worse than no leg, and
#: :func:`constitutive_sites` is what refuses to score such a leg.
SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "fold_the_zero_cross_terms": base.SOURCE_MUTATIONS["fold_the_zero_cross_terms"],
    "plane_wise_scaling": base.SOURCE_MUTATIONS["plane_wise_scaling"],
    "flatten_the_constitutive_tail":
        base.SOURCE_MUTATIONS["flatten_the_constitutive_tail"],
    "store_fw_before_reading_prev":
        base.SOURCE_MUTATIONS["store_fw_before_reading_prev"],
    "word_pair_transposed": base.SOURCE_MUTATIONS["word_pair_transposed"],
    "column_major_index": base.SOURCE_MUTATIONS["column_major_index"],
    "half_plane_field_store": m_half_plane_field_store,
    "half_plane_auxiliary_store": m_half_plane_auxiliary_store,
    "commute_the_plane_sum": base.SOURCE_MUTATIONS["commute_the_plane_sum"],
    "reload_the_constitutive_store": n_reload_the_constitutive_store,
}


# ---------------------------------------------------------------------------
# The mutation legs and what each one requires
# ---------------------------------------------------------------------------
#
# "caught"            -> every scored case must DIVERGE
# "caught_where_live" -> must diverge on every case the defect can REACH
# "uncaught"          -> every scored case must stay identical, and that IS the
#                        measurement
# "measure"           -> a number is reported, not a bar

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
            "weaker absorber, wrong INSIDE THE PML ONLY"},
    {"leg": "s5_word_pair_transposed", "mutation": "word_pair_transposed",
     "kind": "source", "expect": "caught",
     "why": "a complex64 volume read as (im, re) pairs is a scrambled field, not "
            "a launch failure"},
    {"leg": "s6_column_major_index", "mutation": "column_major_index",
     "kind": "source", "expect": "caught",
     "why": "THE COEFFICIENT-INDEX LEG. i and k swapped means every kps/kms "
            "lookup names the wrong axis -- the defect class the Dcyl refusal is "
            "about, planted in the device text rather than in the tables"},
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
    {"leg": "h1_reverse_radial_axis_coefficients",
     "mutation": "reverse_radial_axis_coefficients", "kind": "host",
     "expect": "caught_where_live", "reach": "cylindrical",
     "dead_must_be_identical": True,
     "why": "THE REFUSAL'S OWN PREMISE, ARMED. If the Dcyl extent moved a "
            "coefficient index the way the refusal says, this is the mutation "
            "that would show it. The dead half is a bar because on a Cartesian "
            "control the mutation is a no-op and must be one"},
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
            "sub-step and THIS coordinate system"},
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
#: case product: a leg taken from the head of the sweep landed entirely on the two
#: UNFOLDED controls on the sibling round, so the one mutation that only exists on
#: a Dcyl grid was skipped on every leg and scored 0/0 UNCAUGHT. That is a harness
#: defect that reads exactly like a kernel defect, which is why the selection is
#: written out and why one Cartesian control is kept in it.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "cartesian_periodic",    # the control: a table defect must bite here too,
                             # and the radial mutation must be a no-op here
    "cyl_m0_z_periodic",     # a stored slot past the owned window
    "cyl_m0_z_metallic",     # no extra slot
    "cyl_mneg1_z_metallic",  # the axis-row increment class, negative m
    "cyl_m3_z_periodic",     # the near-axis zeroing class
    "cyl_m0_wide_r",         # a radial extent unlike every other case
)

#: The value classes a mutation leg is scored on. ALL THREE, and each earns its
#: place: ``signed_zero`` is the reach of the two zero-cross-term legs and a plan
#: without it would score them 0/0; ``subnormal_band`` is the class the family's
#: own certified round measured to separate the two EXPANSION ARMS on a
#: zero-imaginary orientation, and a plan without it sends the arm leg back
#: COINCIDENT EVERYWHERE for a reason about the plan rather than about the
#: platform -- measured on this gate's first reduced run, which is why the class
#: is here and this sentence is with it.
MUTATION_VALUE_CLASSES: Tuple[str, ...] = ("uniform", "signed_zero", "subnormal_band")


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                 side: str) -> float:
    """Fraction of output WORDS the array path changed from the frozen input.

    Compared as raw uint32, never with ``allclose``: ``-0.0 == 0.0`` and
    ``NaN != NaN`` both lie, and both the signed-zero and the subnormal-band
    classes put signed zeros in the operands deliberately.
    """
    moved = total = 0
    for name in outputs(side):
        a = np.ascontiguousarray(before[name]).view(np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name]).view(np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def radial_axis_absorbs(side: str, layer, grid) -> Dict[str, Any]:
    """Does the RADIAL axis's coefficient profile differ from the identity?

    ``kps = kms = 1`` is the interior pass-through. An axis whose whole vector is
    that identity cannot distinguish a coefficient-index error on it, so a Dcyl
    case at this floor is measuring every axis except the one Dcyl touched.
    """
    tables = tables_for(side, layer)
    out: Dict[str, Any] = {"radial_axis": None, "max_deviation": 0.0}
    if bool(getattr(grid, "cylindrical", False)):
        deviation = 0.0
        for stem in ("kps", "kms"):
            values = to_host(tables[f"{stem}_x"]).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        out["radial_axis"] = {"axis": 0, "name": "x",
                              "max_deviation_from_identity": deviation}
        out["max_deviation"] = deviation
    out["meets_floor"] = (out["radial_axis"] is None) or out["max_deviation"] > 0.0
    return out


def structure_facts(grid) -> Dict[str, Any]:
    """The shape facts that make a Dcyl case structurally different.

    ``stored_past_owned`` is 1 on a PERIODIC axis and 0 on a METALLIC one
    (grid.py:570-575), and the extra slot is where the deepest absorber
    coefficient lands. Recorded per case so a reader can see that both z
    terminations were really swept and were really different arrays -- which is
    the claim's load-bearing half, because the refusal is about the extent.
    """
    return {
        "shape": [int(n) for n in grid.shape],
        "shape_full": [int(n) for n in grid.shape_full],
        "cells": int(np.prod([int(n) for n in grid.shape])),
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "is_axis": [bool(grid.is_axis(a)) for a in range(3)],
        "stored_cells": [int(grid.stored_cells(a)) for a in range(3)],
        "owned_cells": [int(grid.owned_cells(a)) for a in range(3)],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
        "coefficient_vector_lengths": {
            axis: int(getattr(grid, "shape")[index])
            for index, axis in enumerate("xyz")},
    }


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
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{side}|{courant}|{value_class}".encode()).digest()[:4], "big"))

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
        "cylindrical": bool(spec["cylindrical"]), "m": int(spec["m"]),
        "z_kind": spec.get("z_kind"),
        "boundaries": list(spec.get("boundaries") or ()),
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

    absorbs = radial_axis_absorbs(side, layer, grid)
    case["radial_axis_absorbs"] = absorbs
    if not absorbs["meets_floor"]:
        case["skipped"] = ("the radial axis's coefficient profile is the identity "
                           "everywhere; this case cannot distinguish a coefficient "
                           "index error on the axis Dcyl moved")
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
    tables = (tables_for(side, layer) if host_mutation in (None, "the_other_expansion_arm")
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
    # forever after.
    #
    # ONE ``Fields`` OBJECT, RUN TWICE FROM THE SAME FROZEN STATE, rather than a
    # second object built beside it. Two objects means two epsilon draws unless the
    # volumes are copied across, and a fixture that solved two different problems
    # would report a divergence that says nothing about the kernel.
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


def _licence_stub(expansion, policy: Optional[str]) -> Dict[str, Any]:
    """The verdict shape ``complex_expansion_refusal`` checks, for the record leg.

    NOT A LICENCE AND NOT USED TO BIND AN ARM. The arm comes from
    ``complex_fields.expansion_license`` over a probe artifact, upstream of here
    (:func:`main`); this is only what the predicate is HANDED so its answer is
    about the corpus clause under test rather than about the harness. Passing a
    placeholder string instead is refused with "the expansion licence is str, not
    the verdict dict", which is a refusal about the harness reported as one about
    the corpus.
    """
    arm = complex_emitter.normalized_expansion(expansion)
    return {"arm": complex_emitter.EXPANSION_NAMES[arm], "expansion": arm,
            "basis": "measured", "refusals": [], "policy_resolved": policy}


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str) -> List[Tuple[Dict[str, Any], str, float, str]]:
    specs = CYL_SPECS if product == "full" else CYL_SPECS[:6]
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
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"moved={case['oracle_moved']:.3f} ({case['seconds']:.1f} s)")
    return cases


def mutation_plan(product: str) -> List[Tuple[Dict[str, Any], str, float, str]]:
    """One case per (spec, side, value class) for the mutation legs.

    Held to ONE COURANT deliberately: a mutation leg answers "can this gate see
    this defect at all", and multiplying it by both courants buys repetitions of
    that answer rather than a second question. The VALUE CLASSES are not collapsed
    the same way, because ``signed_zero`` is the reach of two of the legs.
    """
    by_label = {spec["label"]: spec for spec in CYL_SPECS}
    labels = MUTATION_SPEC_LABELS if product == "full" else MUTATION_SPEC_LABELS[:3]
    return [(by_label[label], side, INEXACT_COURANT, value_class)
            for label in labels for side in SIDES
            for value_class in MUTATION_VALUE_CLASSES]


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


def _reaches(leg: Dict[str, Any], case: Dict[str, Any]) -> bool:
    """Whether this case is one the leg's defect can be seen on at all.

    A REACH IS A CLAIM AND EACH ONE IS DERIVED, not guessed:

    * ``cylindrical`` -- the mutation edits the RADIAL axis's coefficient vector.
      A Cartesian control has no radial axis and the mutation is the identity
      there, so the dead half is a bar: it must be silent, or the mutation is
      editing something it does not claim to.
    * ``signed_zero`` -- the ONE class on which the zero cross terms can change a
      stored word: ``fma(c, z.re, -0.0f)`` differs from ``fma(c, z.re, +0.0f)``
      exactly when ``c * z.re`` is ``-0.0f``, and the class is what puts negative
      zeros in the plane the cross term reads.
    """
    reach = leg.get("reach")
    if reach is None:
        return True
    if reach == "cylindrical":
        return bool(case["cylindrical"])
    if reach == "signed_zero":
        return case["value_class"] == "signed_zero"
    raise ValueError(f"unknown reach {reach!r}")


def adjudicate(leg: Dict[str, Any], cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Did this leg's verdict come out the way the leg requires?

    NO LEGS is its own verdict, never "as required". A mutation that was never run
    has not been shown inert; it has not been asked. Collapsing the two is how a
    gate acquires a silent hole, and the sibling round had exactly that: the
    radial mutation scored 0/0 UNCAUGHT because every selected leg happened to be
    an unfolded control.
    """
    scored = [c for c in cases if not c.get("skipped")]
    caught = [c for c in scored if leg_caught(c)]
    out: Dict[str, Any] = {"ran": len(scored), "caught": len(caught),
                           "uncaught": len(scored) - len(caught),
                           "expect": leg["expect"]}
    if not scored:
        out["verdict"] = "NO LEGS"
        out["as_required"] = False
        return out
    expect = leg["expect"]
    if expect == "caught":
        out["as_required"] = len(caught) == len(scored)
        out["verdict"] = ("CAUGHT" if out["as_required"]
                          else "PARTIAL" if caught else "UNCAUGHT")
    elif expect == "uncaught":
        out["as_required"] = not caught
        out["verdict"] = "NULL CONFIRMED" if out["as_required"] else "NULL VIOLATED"
    elif expect == "caught_where_live":
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
    elif expect == "measure":
        out["diverged_on"] = sorted({
            f"{c['label']}/{c['side']}/{c['value_class']}" for c in caught})
        out["coincided_on"] = sorted({
            f"{c['label']}/{c['side']}/{c['value_class']}"
            for c in scored if not leg_caught(c)})
        out["max_differing_words"] = max(
            (c["single_launch"]["differing_floats"] for c in scored), default=0)
        # A measurement leg passes when it MEASURED something -- and, for the arm
        # swap specifically, when at least one case could tell the arms apart. A
        # leg on which nothing ever diverges has not shown the arm matters here,
        # and that is a finding rather than a pass.
        out["as_required"] = bool(caught)
        out["verdict"] = ("SEPARATED" if caught else "COINCIDENT EVERYWHERE")
    else:
        raise ValueError(f"unknown expectation {expect!r}")
    return out


def constitutive_sites(transform: Callable[[str], Tuple[str, int]],
                       expansion) -> Dict[str, int]:
    """How many sites the rewrite matches in EACH constitutive source, per side.

    THE BACKEND'S OWN SITE COUNT IS NOT ENOUGH HERE and this is not pedantry.
    ``set_source_transform`` returns the MAXIMUM over all four sub-steps, so a
    mutation that matches only in the CURL comes back with a nonzero count and a
    leg would then be scored on constitutive cases running the SHIPPED bytes --
    reporting "uncaught" for a reason about the harness. This counts each
    constitutive side separately, and a side that matches nothing is excluded from
    scoring by name rather than being silently counted as a miss.

    THE ARM IS PASSED THROUGH because the two arms are different text: the
    zero-cross-term needles exist in both but spelled differently, and counting on
    one arm would arm a leg for a binary the other arm compiles.
    """
    out: Dict[str, int] = {}
    for side in SIDES:
        sub_step = "update_H" if side == "H" else "update_E"
        source = complex_emitter.complex_source(sub_step, expansion)
        _, count = transform(source)
        out[side] = int(count)
    return out


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
        verdict = adjudicate(leg, legs)
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


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any], backend_certifies: bool) -> Dict[str, Any]:
    """The release decision, with every clause it rests on named."""
    reasons: List[str] = []
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    cylindrical = [c for c in scored if c["cylindrical"]]
    single_ok = [c for c in scored if c["single_launch"]["bit_identical"]]
    multi_cases = [c for c in scored if "multi_step" in c]
    multi_ok = [c for c in multi_cases if c["multi_step"]["bit_identical"]]

    if not cylindrical:
        reasons.append("no Dcyl case was scored at all")
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

    # BOTH z TERMINATIONS AND BOTH |m| CLASSES must have been scored. The z
    # termination changes the stored extent (one slot past the owned window or
    # not) and is what makes the coefficient vectors structurally different
    # arrays; the |m| classes are what the CURL treats differently, and a sweep
    # that scored only one of them would leave "except at |m| >= 1" open.
    terminations = {case["z_kind"] for case in cylindrical if case["z_kind"]}
    if terminations != {"periodic", "metallic"}:
        reasons.append(f"z terminations scored were {sorted(terminations)}, "
                       f"not both of periodic and metallic")
    m_classes = set()
    for case in cylindrical:
        m = abs(int(case["m"]))
        m_classes.add("m0" if m == 0 else "m1" if m == 1 else "m2plus")
    if m_classes != {"m0", "m1", "m2plus"}:
        reasons.append(f"|m| classes scored were {sorted(m_classes)}, not all of "
                       f"m = 0, |m| = 1 and |m| >= 2")
    if not [case for case in cylindrical if int(case["m"]) < 0]:
        reasons.append("no NEGATIVE m was scored; seven of the sixteen corpus "
                       "rows are m = -1")

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

    # THE GUARD CONTROL, REPORTED RATHER THAN GATED ON, for the reason the complex
    # family's own release records: under the licensed FMA_V1 arm every product in
    # the emitted source is an explicit __fmaf_rn, which no contraction flag
    # reaches, so identical images are the EXPECTED reading here and a divergent
    # control would be the surprise. The byte comparison in
    # ``guard_effect`` is what separates "inert because there is nothing to do"
    # from "inert because the gate cannot see"; it is recorded beside this.
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
        "cylindrical_cases": len(cylindrical),
        "single_launch_identical": len(single_ok),
        "multi_step_identical": len(multi_ok),
        "multi_step_cases": len(multi_cases),
        "z_terminations": sorted(terminations),
        "m_classes": sorted(m_classes),
        "m_values": sorted({int(case["m"]) for case in cylindrical}),
        "value_classes": sorted({case["value_class"] for case in scored}),
        "negative_zero_words_in_signed_zero_cases": negative_zeros,
        "subnormal_words_in_band_cases": subnormals,
        "claim": ("the SHIPPED CUDA COMPLEX constitutive kernels "
                  "(update_H_pml_complex_bloch / ..._E_...) are "
                  "bit-identical to stepping.update_H/update_E on a complex64 "
                  "Dcyl grid, per sub-step, at every |m| class, both signs of m "
                  "and both z terminations, with no change to the emitted source"),
        "does_not_claim": [
            "nothing dispatches these kernels; no module in meep_gpu imports "
            "cuda_kernels at all, so a widened predicate licenses a MEASUREMENT "
            "and not a production step",
            "the predicate is NOT changed by this gate; the change it licenses is "
            "reported, not made",
            "the complex Dcyl CURL is a separate family and is REFUSED for a real "
            "reason -- cshift_up/cshift_dn are a neighbour stencil and Dcyl "
            "replaces the radial derivative with a prefix sum and adds the "
            "axis-row rules. Nothing here touches it",
            "a PHASED Dcyl grid is not measured: no corpus row carries one "
            "(all sixteen record has_bloch FALSE) and Dcyl Bloch is z-only in "
            "MEEP. The constitutive kernel takes no phase argument in any case",
            "the REAL constitutive pair is a different kernel and was scored by a "
            "different gate; nothing here changes covers_real_pml_constitutive",
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
        "gate": "cuda_complex_cylindrical_constitutive",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("does covers_real_pml_complex_constitutive's Dcyl refusal "
                     "describe a divergence in the CONSTITUTIVE arithmetic, or is "
                     "it inherited from the complex CURL through the one clause "
                     "_complex_grid_refusal shares with it?"),
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
        # tuple NVRTC was really given (post-strip). Installed afterwards it would
        # sit outside the strip and record the pre-strip tuple -- the one thing
        # that would make the two policy legs look alike.
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
            # that a LEG can fail; this shows that the RELEASE DECISION can. The
            # defect is planted for the whole run, sweep included, so a run under
            # --falsify that still reported released=True would mean the summary
            # is not reading the cases it claims to.
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
            # partway through the run and leave an artifact that measured two
            # different programs.
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
        f"scored={verdict['scored_cases']} dcyl={verdict['cylindrical_cases']} "
        f"m={verdict['m_values']} z={verdict['z_terminations']} "
        f"single_identical={verdict['single_launch_identical']} "
        f"multi_identical={verdict['multi_step_identical']}/"
        f"{verdict['multi_step_cases']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
