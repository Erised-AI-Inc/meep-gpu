"""Sub-step byte-identity gate for the SHIPPED CUDA constitutive pair ON A Dcyl GRID.

THE CLAIM UNDER TEST, and it is a claim about a PREDICATE, not about a kernel.
``meep_gpu/cuda_kernels/constitutive_kernels.py`` ships two certified kernels,
``update_H_pml_real`` and ``update_E_pml_real``, and
``complex_pml_kernels.py`` the complex pair. This gate does not change one byte of
any of them. It asks whether the refusal ``covers_real_pml_constitutive`` and
``covers_real_pml_complex_constitutive`` both state --

    "cylindrical (Dcyl): the axial extent moves every coefficient index"

-- is a refusal the arithmetic requires, or a refusal inherited from the CURL
family that shares the predicate's shape. The predicate's own docstring says as
much in as many words: "The cylindrical half stands, unmeasured and therefore
refused" (coverage.py:854). This is the measurement.

WHY THE READING SAYS IT IS NOT REQUIRED (every fact read off the source, and
checked line by line before this file was written):

1. The sub-step reads NO NEIGHBOUR. ``stepping._apply_constitutive_pml``
   (stepping.py:2112-2145) is ``fw_previous = fw.copy(); fw[...] = source;
   field += kps*fw; field -= kms*fw_previous`` -- four whole-array operations, no
   shift, no ghost, no mask. Dcyl changes the RADIAL DERIVATIVE and the AXIS ROW;
   this sub-step has neither.
2. ``update_H`` (stepping.py:907-925) and ``update_E`` (:926-995) contain ZERO
   occurrences of "cylindrical", "is_axis", "m" or "axis_zero" -- counted, not
   eyeballed. The Dcyl repairs are separate passes ``step_B`` and ``step_D`` run:
   ``_cylindrical_axis_zero_B`` is called at :376 and ``_cylindrical_axis_zero_D``
   at :457, both inside the CURL, and nowhere else in the module.
3. ``_mask_non_owned_cells`` (stepping.py:1912) has exactly three call sites --
   ``step_B`` (:397), ``step_D`` (:479) and ``_bfast_term`` (:929). No
   constitutive function is among them, on any coordinate system.
4. THE EXTENT MOVES BOTH OPERANDS TOGETHER, which is the refusal's premise and is
   TRUE. ``PML._compute_coefficients`` builds every kps/kms vector from
   ``grid.nx``/``ny``/``nz`` -- the STORED extent -- and the field arrays are
   allocated on the same ``grid.shape``. On Dcyl the phi axis is invariant with
   n = 1 and r and z are ordinary extents, so the kernel's ``i = idx/(ny*nz)``
   indexes a table of exactly the length the grid left. The premise is true and
   the conclusion does not follow.
5. The sibling Triton track reached this conclusion first AND MEASURED IT:
   ``triton_kernels/cylindrical_complex.py`` records 480/480 identity rows with 0
   differing uint32 words for ``update_H``/``update_E`` on complex Dcyl against a
   plain element-wise dsigw reference carrying no cylindrical clause. That is a
   reason to measure this one, not to inherit it: the CUDA kernels are different
   bytes, compiled by a different toolchain, and this project does not transfer a
   verdict across backends.

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
such a case is refused before it is scored rather than counted as a miss.

Every other leg is the folded sibling's, unchanged, so the two rounds are
commensurable to the case: same source mutations, same null pair, same guard
sets, same value classes, same courant pair.

USAGE

    python -u gate_cuda_cylindrical_constitutive.py \
        --backend cuda --out results/cuda_cylindrical_constitutive_<date>/gate.json

    python -u gate_cuda_cylindrical_constitutive.py --backend numpy \
        --out /tmp/cyl_constitutive_local.json
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
# THE CASE TABLE
# ---------------------------------------------------------------------------
#
# Dcyl grids over BOTH z terminations and BOTH |m| classes, plus the Cartesian
# controls the folded sibling carries so this harness is shown to reproduce the
# standing result under its own fixture.
#
# ``m`` IS SWEPT EVEN THOUGH THIS SUB-STEP CANNOT READ IT. That is the point: the
# claim is that update_H/update_E are element-wise and carry no cylindrical
# branch, and a sweep that fixed m at 0 would leave "except at |m| >= 1" open.
# The |m| classes differ in the CURL (axis-row increment at |m| = 1, near-axis
# zeroing at |m| >= 2), so if any of that leaked into the constitutive sub-step
# it would show as an m-dependent divergence here.
#
# ``cell`` is in units of 1/resolution at resolution 1.0, so the numbers ARE the
# cell counts. Grid owns the r axis's boundary pair -- the axis at r = 0 and a
# metallic wall at r_max -- so only z is chosen (grid.py:498-508).

CYL_SPECS: Tuple[Dict[str, Any], ...] = (
    # The Cartesian controls. Already certified; carried so a divergence in the
    # Dcyl rows cannot be blamed on the fixture.
    {"label": "cartesian_periodic", "cylindrical": False, "m": 0,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "cartesian_metallic", "cylindrical": False, "m": 0,
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (8.0, 10.0, 12.0)},

    # m = 0: the REAL-storage Dcyl class, both z terminations.
    {"label": "cyl_m0_z_metallic", "cylindrical": True, "m": 0,
     "z_kind": "metallic", "cell": (12.0, 0.0, 16.0)},
    {"label": "cyl_m0_z_periodic", "cylindrical": True, "m": 0,
     "z_kind": "periodic", "cell": (12.0, 0.0, 16.0)},

    # |m| = 1: the axis-row increment class, BOTH SIGNS. Seven of the sixteen
    # corpus rows are m = -1 and one is m = +1.
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
    # invisible at one aspect ratio is not invisible at two.
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
#: run on both backends. ``reverse_radial_axis_coefficients`` is this gate's own
#: and is the refusal reason armed as a defect (see the header).
HOST_MUTATIONS: Tuple[str, ...] = (
    "reverse_radial_axis_coefficients",
    "swap_constitutive_sublattice",
    "swap_kps_kms",
)

#: Which host mutations are only meaningful on a folded case. Scoring
#: ``reverse_radial_axis_coefficients`` on the unfolded control would be scoring a
#: no-op as a miss.
CYLINDRICAL_ONLY_HOST_MUTATIONS: Tuple[str, ...] = ("reverse_radial_axis_coefficients",)


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
    """A frozen ``(fields, layer, grid)`` triple for one case, Dcyl or Cartesian.

    ON Dcyl, ``Grid`` OWNS THE r AXIS'S BOUNDARY PAIR -- the axis at r = 0 and a
    metallic wall at r_max -- so only z is chosen (grid.py:498-508), and the PML
    is asked for the HIGH r face only for the same reason: cell 0 is the axis, a
    boundary condition rather than a wall. Same rule the sibling Dcyl curl gate
    (``gate_cuda_cylindrical_real.build``) uses, restated rather than imported so
    the two gates cannot drift apart silently through a shared helper.

    The Cartesian controls take the folded sibling's thickness rule unchanged.
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


def reversed_radial_axis_tables(side: str, layer, grid) -> Dict[str, Any]:
    """THE REFUSAL REASON, ARMED. The r axis's profile read backwards.

    "cylindrical (Dcyl): the axial extent moves every coefficient index" is the
    predicate's stated reason for refusing Dcyl. If that mattered to this
    sub-step, the coefficient a cell reads on the radial axis would be wrong.
    This mutation makes it wrong -- IN BOUNDS, by reversing the vector rather
    than by substituting a longer or shorter one, so the leg measures a WRONG
    ANSWER and not a memory fault.

    It is also the only thing that makes ``radial_axis_absorbs`` load-bearing: an
    axis whose profile is the identity everywhere reverses to itself, and the leg
    would come back UNCAUGHT for a reason about the fixture rather than about the
    kernel. The floor refuses such a case before it is scored.

    ON A CARTESIAN CONTROL there is no radial axis and this is a no-op, which is
    why the control is excluded from scoring it -- see
    :data:`CYLINDRICAL_ONLY_HOST_MUTATIONS`.
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
    base = tables_for(side, layer)
    out = dict(base)
    for axis in "xyz":
        out[f"kps_{axis}"] = base[f"kms_{axis}"]
        out[f"kms_{axis}"] = base[f"kps_{axis}"]
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
        "cylindrical": bool(spec["cylindrical"]), "m": int(spec["m"]),
        "z_kind": spec.get("z_kind"),
        "boundaries": list(spec.get("boundaries") or ()),
        "structure": structure_facts(grid),
        "operand_census": operand_census(host),
    }

    # THE PREDICATE'S CURRENT ANSWER, recorded rather than acted on. This gate
    # exists to decide whether that answer should change, so it must not be the
    # thing that gates the measurement -- but a record that did not carry it
    # could not show WHICH clause the run was taken against.
    covered, reason = coverage.covers_real_pml_constitutive(fields, layer, grid, side)
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
    specs = CYL_SPECS if product == "full" else CYL_SPECS[:6]
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
#: UNFOLDED controls, so ``reverse_radial_axis_coefficients`` -- the one mutation
#: that only exists on a fold -- was skipped on every leg and scored 0/0 UNCAUGHT.
#: That is a harness defect that reads exactly like a kernel defect, which is why
#: the selection is written out.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "cartesian_periodic",    # the control: a table defect must bite here too
    "cyl_m0_z_periodic",     # a stored slot past the owned window
    "cyl_m0_z_metallic",     # no extra slot
    "cyl_mneg1_z_metallic",  # the axis-row increment class, negative m
    "cyl_m3_z_periodic",     # the near-axis zeroing class
    "cyl_m0_wide_r",         # a radial extent unlike every other case
)


def mutation_plan(product: str) -> List[Tuple[Dict[str, Any], str, float, str]]:
    """One case per (spec, side) for the mutation legs, at the INEXACT courant.

    Held to one courant and one value class deliberately: a mutation leg answers
    "can this gate see this defect at all", and multiplying it by the whole sweep
    buys repetitions of that answer rather than a second question.
    """
    by_label = {spec["label"]: spec for spec in CYL_SPECS}
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
            if name in CYLINDRICAL_ONLY_HOST_MUTATIONS and not spec["cylindrical"]:
                continue  # a radial mutation on a Cartesian control is a no-op
            case = one_case(backend, spec, side, courant, value_class,
                            "fmad_false", host_mutation=name)
            legs.append(case)
        scored = [c for c in legs if not c.get("skipped")]
        caught = sum(1 for c in scored if leg_caught(c))
        # NO LEGS is its own verdict, never "UNCAUGHT". A mutation that was never
        # run has not been shown inert; it has not been asked. Collapsing the two
        # is how a gate acquires a silent hole, and this one had it: the first
        # full run scored ``reverse_radial_axis_coefficients`` 0/0 UNCAUGHT
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
    # Dcyl LEGS ONLY. Every one of these defects is already scored on Cartesian
    # grids by the certified record; what this gate adds is whether the same
    # battery still bites on a Dcyl extent.
    plan = [entry for entry in mutation_plan(product) if entry[0]["cylindrical"]]

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

    # BOTH z TERMINATIONS AND BOTH |m| CLASSES must have been scored. The z
    # termination changes the stored extent (one slot past the owned window or
    # not) and is what makes the coefficient vectors structurally different
    # arrays; the |m| classes are what the CURL treats differently, and a sweep
    # that scored only one of them would leave "except at |m| >= 1" open, which
    # is the whole reason m is in the case table for a sub-step that cannot read
    # it.
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
        "cylindrical_cases": len(cylindrical),
        "single_launch_identical": len(single_ok),
        "multi_step_identical": len(multi_ok),
        "multi_step_cases": len(multi_cases),
        "z_terminations": sorted(terminations),
        "m_classes": sorted(m_classes),
        "m_values": sorted({int(case["m"]) for case in cylindrical}),
        "claim": ("the SHIPPED CUDA constitutive kernels are byte-identical to "
                  "stepping.update_H/update_E on a Dcyl grid, per sub-step, at "
                  "every |m| class and both z terminations, with no change to "
                  "the device source"),
        "does_not_claim": [
            "nothing dispatches these kernels; no module in meep_gpu imports "
            "cuda_kernels at all",
            "the predicate is NOT changed by this gate; the change it licenses "
            "is reported, not made",
            "the Dcyl CURL is a separate family and is not measured here -- it "
            "genuinely needs new kernels (the radial prefix-sum derivative and "
            "the axis-row rules), which cylindrical_kernels.py carries for m = 0 "
            "and nothing carries for |m| >= 1",
            "the COMPLEX constitutive pair is a separate kernel and is not "
            "measured here; this gate scores the real pair only",
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
        "gate": "cuda_cylindrical_constitutive",
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
        f"scored={verdict['scored_cases']} dcyl={verdict['cylindrical_cases']} "
        f"m={verdict['m_values']} z={verdict['z_terminations']} "
        f"single_identical={verdict['single_launch_identical']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
