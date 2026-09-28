"""Sub-step byte-identity gate for the THREE DISPERSIVE hand-CUDA arms.

WHAT IS UNDER TEST, and it is three different claims rather than one:

* **arm1** -- ``dispersive_kernels.update_E_pml_real_dispersive`` reproduces
  ``stepping.update_E`` under an active absorber when the source is
  ``(D - sum P) * inv_eps`` (stepping.py:1010-1018). A NEW KERNEL.
* **arm2** -- ``dispersive_kernels.update_E_no_pml_real_dispersive`` reproduces
  ``stepping.update_E`` with NO absorber and stored E, where the sub-step is the
  plain store at stepping.py:1022. A SECOND NEW KERNEL, not a flag on the first:
  no auxiliary, no coefficient vector, no history, no per-axis index.
* **arm3** -- the SHIPPED ``ade_kernels.update_P_fused_pml_real`` reproduces
  ``stepping.update_P`` on a LAYERLESS run. NO NEW KERNEL AT ALL. This leg is
  what turns "a predicate widening, not a new kernel" from a reading of
  ``Fields.drive_field`` (fields.py:1140-1163) into a measurement, and it carries
  ``arm3_wrong_drive_CONTROL`` -- the same shipped launcher with ``D`` bound as
  the drive -- which MUST DIVERGE. A leg whose control cannot fail has not shown
  that binding the right array is what produced the pass.

=============================================================================
THE VACUITY TRAP THIS FAMILY SITS ON TOP OF
=============================================================================

A zero-initialised constitutive or ADE step leaves every output word at ``+0.0``
forever, so a reference with deliberately swapped coefficients still reports
IDENTICAL. ``triton_kernels/cylindrical_complex.py`` records exactly that: half
its first cut could not fail. FOUR FLOORS stand between this gate and the same
result, and every one of them is checked PER CASE and refuses the case rather
than scoring it:

1. ``oracle_moved`` -- the fraction of output WORDS the array path changed from
   the frozen input, compared as raw uint32 (``-0.0 == 0.0`` and ``NaN != NaN``
   both lie, and two of the three value classes put signed zeros in the operands
   deliberately). Zero means the case is a fixed point and certifies nothing.
2. ``pole_subtraction_bites`` -- the fraction of words where ``D - sum P``
   differs BITWISE from ``D``. A case whose poles are all zero is a
   non-dispersive case wearing dispersive clothes: every pole mutation would come
   back UNCAUGHT for a reason about the fixture. Required on every arity above 0.
3. ``coefficient_profile_bites`` (arm1 only) -- the absorber profile must differ
   from the interior identity ``kps = kms = 1`` on at least one axis, or a
   coefficient-INDEX error is invisible and the case measures nothing about the
   thing ``own_axis_to_x_for_all_three`` is armed for.
4. ``inverse_epsilon_bites`` -- the material must differ from 1.0, or
   ``drop_inverse_epsilon_dispersive`` is bit-identical to the shipped kernel.

=============================================================================
THE POLE-ORDER PAIR IS INVISIBLE AT ONE POLE, AND THAT IS MEASURED
=============================================================================

``sum_then_subtract`` (``D - (P0 + P1)``) and ``reverse_pole_order`` are EXACTLY
bit-identical to the shipped chain at ONE pole -- there is only one association
of ``D - P0`` -- and lethal at two. Measured on the laptop before this gate was
written, on an 8x10x12 grid at courant 0.35: 0 differing words at one pole, 1253
and 1297 at two, 2245 and 2400 at five
(``meep_gpu/cuda_kernels/test_dispersive.py`` re-measures it every merge).

So :data:`MUTATION_ARITIES` pins every pole-mutation leg to an arity of at least
two, and a leg that would have been scored at one pole is reported ``NO LEGS``
rather than ``UNCAUGHT``. Unasked is not inert, and the sibling Dcyl gate's first
full run is the standing example of what happens when the two are collapsed.

USAGE

    python -u gate_cuda_dispersive.py --backend cuda --subnormal-policy keep \\
        --out results/cuda_dispersive_<date>/keep/gate.json
    python -u gate_cuda_dispersive.py --backend cuda --subnormal-policy flush \\
        --import-meep-for-host-policy \\
        --out results/cuda_dispersive_<date>/flush/gate.json

    python -u gate_cuda_dispersive.py --backend numpy --out /tmp/dispersive.json
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
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import dispersive_kernels as dispersive  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census

SEED = 20260820

#: Consecutive launches in the multi-step leg. 60 is the budget both certified
#: hand-CUDA records are cut at. "Identical for N steps" is a claim about N, and
#: on THIS family it is the only leg that can see a stale pole plan: the buffers
#: rotate at update_P (dispersion.py:689-691), so a launcher that cached its
#: plan is exact for one launch and reads last step's polarization forever after.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence live. Held fixed, ``f_w`` settles
#: and a 60-step leg becomes a slow single-step leg. The same exact float32 scale
#: is applied on both paths, so it cancels out of the comparison.
_ADVANCE = np.float32(0.97)

ARMS: Tuple[str, ...] = ("arm1", "arm2", "arm3")

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient AND every ADE recurrence coefficient, so it is a real
#: second draw of both tables and not only a rounding probe.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

#: ``cancellation`` is this family's own class and no other CUDA gate has it.
#: Each pole carries an equal share of D, so the sequential subtraction walks the
#: sum down towards zero and the LAST subtraction is the one that cancels -- which
#: is where ``D - sum P`` lands in and around the subnormal band and where the
#: two float32 policies can differ. Without it the gate's verdict would say
#: nothing about the band the pole chain actually visits.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "cancellation")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)


# ---------------------------------------------------------------------------
# THE CASE TABLE
# ---------------------------------------------------------------------------
#
# Every spec is a configuration the ELEVEN corpus rows this family answers for
# actually carry, plus the structural controls that keep an indexing defect from
# hiding at one aspect ratio. Shapes are in units of 1/resolution at resolution
# 1.0, so the numbers ARE the cell counts.

SPECS: Tuple[Dict[str, Any], ...] = (
    # --- arm 1's own configurations -------------------------------------------
    # stochastic_emitter*.py: 2-D, all-periodic, no fold, six poles.
    {"label": "pml_2d_periodic", "pml": True, "mirrors": (),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (10.0, 12.0, 1.0)},
    # TestLoadDump.*_2d: 2-D, metallic x and y, FOLDED on y, five poles.
    {"label": "pml_2d_folded_metallic", "pml": True, "mirrors": (1,),
     "boundaries": ("metallic", "metallic", "periodic"), "cell": (12.0, 14.0, 1.0)},
    # absorbed_power_density.py's shape class: periodic/mirror/periodic.
    {"label": "pml_2d_folded_periodic", "pml": True, "mirrors": (1,),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (12.0, 14.0, 1.0)},
    # TWO SIMULTANEOUS PLANES -- the ceiling CONSTITUTIVE_FOLD_ADMISSION swept and
    # the ceiling this family's predicate inherits. Admitting it without scoring
    # it would be inheriting a verdict across families.
    {"label": "pml_3d_folded_two_planes", "pml": True, "mirrors": (0, 1),
     "boundaries": ("metallic", "periodic", "metallic"), "cell": (12.0, 14.0, 10.0)},
    # 3-D, all-metallic: a stored extent with NO slot past the owned window on any
    # axis, against the periodic cases which have one on every axis.
    {"label": "pml_3d_metallic", "pml": True, "mirrors": (),
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (9.0, 11.0, 13.0)},
    # ODD counts on all three axes and an aspect ratio unlike every case above:
    # a coefficient-index defect invisible at one shape is not invisible at two.
    {"label": "pml_3d_odd_counts", "pml": True, "mirrors": (),
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (7.0, 15.0, 9.0)},

    # --- arm 2 and arm 3's configurations -------------------------------------
    # absorber-1d.py and TestAbsorber.test_absorber: 1-D along z, metallic z.
    {"label": "no_pml_1d_metallic_z", "pml": False, "mirrors": (),
     "boundaries": ("periodic", "periodic", "metallic"), "cell": (1.0, 1.0, 40.0)},
    # material-dispersion.py: the fully periodic degenerate cell.
    {"label": "no_pml_all_periodic", "pml": False, "mirrors": (),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    # A FOLD WITHOUT AN ABSORBER. No corpus row carries it; the predicate admits
    # it, so it is scored rather than admitted on the unfolded cases' evidence.
    {"label": "no_pml_folded", "pml": False, "mirrors": (1,),
     "boundaries": ("metallic", "periodic", "metallic"), "cell": (10.0, 12.0, 8.0)},
)

#: Which arms each spec can carry. arm2 and arm3 need an inert layer; arm1 needs
#: an active one. Written out rather than derived from ``spec["pml"]`` so a spec
#: added later has to say which claims it is evidence for.
ARMS_BY_SPEC: Dict[str, Tuple[str, ...]] = {
    "pml_2d_periodic": ("arm1",),
    "pml_2d_folded_metallic": ("arm1",),
    "pml_2d_folded_periodic": ("arm1",),
    "pml_3d_folded_two_planes": ("arm1",),
    "pml_3d_metallic": ("arm1",),
    "pml_3d_odd_counts": ("arm1",),
    "no_pml_1d_metallic_z": ("arm2", "arm3"),
    "no_pml_all_periodic": ("arm2", "arm3"),
    "no_pml_folded": ("arm2", "arm3"),
}

#: The pole arities swept. The module's own list, so the gate and the emitter
#: cannot disagree about what "measured up to the cap" means.
ARITIES: Tuple[Tuple[int, int, int], ...] = dispersive.POLE_COUNTS_SWEPT

#: THE ARITIES A POLE MUTATION MAY BE SCORED AT. Two and above, always -- see the
#: header. A leg planted at arity (0,0,0) or (1,1,1) is reported NO LEGS.
MUTATION_ARITIES: Tuple[Tuple[int, int, int], ...] = ((2, 2, 2), (5, 5, 5))

#: Host mutations: they corrupt the TABLES or the PLAN rather than the device
#: text, so they run on both backends. ``reverse_pole_order`` and
#: ``drop_one_pole_from_plan`` are HOST rather than source mutations
#: deliberately: the PLAN fills the kernel's pole slots, so corrupting it there
#: is the faithful spelling of "the launcher resolved fields.polarizations order
#: backwards", which is the defect that actually threatens this kernel.
HOST_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "swap_constitutive_sublattice": {
        "arms": ("arm1",), "pole_defect": False, "must_be_caught": True,
        "why": "the INTEGER sub-lattice where update_E reads the half-integer one "
               "(stepping.py:1015) -- a half-cell error in the absorber profile"},
    "swap_kps_kms": {
        "arms": ("arm1",), "pole_defect": False, "must_be_caught": True,
        "why": "(kap+sig) and (kap-sig) exchanged: the absorber runs backwards"},
    "reverse_pole_order": {
        "arms": ("arm1", "arm2"), "pole_defect": True, "must_be_caught": True,
        "why": "the chain walked backwards; float32 subtraction is not "
               "commutative across a chain and this is INVISIBLE at one pole"},
    "drop_one_pole_from_plan": {
        "arms": ("arm1", "arm2"), "pole_defect": True, "must_be_caught": True,
        "why": "the last contributor silently missing from the chain"},
    "arm3_wrong_drive": {
        "arms": ("arm3",), "pole_defect": False, "must_be_caught": True,
        "why": "THE CONTROL that makes arm 3 a measurement: D bound as the drive "
               "instead of the stored E. The sibling Triton track measured the "
               "mirror-image defect at 3072/15360 words differing "
               "(B_ade_wrong_drive_CONTROL, triton_kernels/no_pml_ade.py)"},
    "arm3_stale_pole_plan": {
        "arms": ("arm3",), "pole_defect": False, "must_be_caught": True,
        "why": "the P/P_prev/_scratch views resolved BEFORE the loop and reused. "
               "The three buffers rotate PER COMPONENT inside one call "
               "(dispersion.py:689-691) and the retired history becomes the NEXT "
               "component's scratch, so a hoisted plan writes this component's "
               "result over the buffer the previous one just claimed. Scored "
               "against a LOCAL COPY of the shipped loop, paired with "
               "arm3_launcher_copy_NULL below so the copy is shown equivalent to "
               "the shipped launcher before the hoist is blamed for anything"},
    "arm3_launcher_copy_NULL": {
        "arms": ("arm3",), "pole_defect": False, "must_be_caught": False,
        # CUDA ONLY, and reported NOT APPLICABLE rather than NULL CONFIRMED
        # elsewhere: on the NumPy backend the "copy of the launcher" IS the
        # transcription the whole leg already runs, so a null there would be
        # true by construction -- a vacuous pass of exactly the kind this gate's
        # floors exist to refuse.
        "backends": ("cuda",),
        "why": "THE DISCRIMINATING CONTROL for arm3_stale_pole_plan: the same "
               "local copy of the launcher loop WITHOUT the hoist. It must be "
               "UNCAUGHT, which is what makes 'the hoist is the defect' a "
               "measurement rather than a statement about a leg written by hand"},
}

#: Source mutations: they rewrite the EMITTED device text through
#: ``dispersive_kernels.SOURCE_TRANSFORM``. Device backend only.
#:
#: The NULLS must come back UNCAUGHT, each paired with a leg editing the SAME
#: expression that must be caught -- ``inv_eps_left`` with
#: ``drop_inverse_epsilon_dispersive``, ``commute_constitutive_scale`` with
#: ``regroup_constitutive_dispersive``. A battery of only-must-be-caught legs
#: scores identically whether the comparator works or has degenerated into
#: failing everything.
SOURCE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "sum_then_subtract": {
        "arms": ("arm1", "arm2"), "pole_defect": True, "must_be_caught": True},
    "drop_one_pole": {
        "arms": ("arm1", "arm2"), "pole_defect": True, "must_be_caught": True},
    "regroup_constitutive_dispersive": {
        "arms": ("arm1",), "pole_defect": False, "must_be_caught": True},
    "drop_fw_store_dispersive": {
        "arms": ("arm1",), "pole_defect": False, "must_be_caught": True},
    "store_fw_before_reading_prev_dispersive": {
        "arms": ("arm1",), "pole_defect": False, "must_be_caught": True},
    "drop_inverse_epsilon_dispersive": {
        "arms": ("arm1", "arm2"), "pole_defect": False, "must_be_caught": True},
    "own_axis_to_x_for_all_three": {
        "arms": ("arm1",), "pole_defect": False, "must_be_caught": True},
    "fortran_order_index_decomposition": {
        "arms": ("arm1",), "pole_defect": False, "must_be_caught": True},
    "bind_Ez_inv_eps_for_all_three": {
        "arms": ("arm1", "arm2"), "pole_defect": False, "must_be_caught": True},
    "inv_eps_left": {
        "arms": ("arm1", "arm2"), "pole_defect": False, "must_be_caught": False},
    "commute_constitutive_scale": {
        "arms": ("arm1",), "pole_defect": False, "must_be_caught": False},
}


# ---------------------------------------------------------------------------
# THE SOURCE MUTATIONS, spelled against the EMITTED text
# ---------------------------------------------------------------------------

def _mutate_sum_then_subtract(source: str) -> Tuple[str, int]:
    """``((D - P0) - P1)`` becomes ``D - (P0 + P1)``. INVISIBLE AT ONE POLE."""
    sites = 0
    out = source
    for _target, _src, axis in dispersive.ELECTRIC_TERMS:
        chain = re.findall(rf"^    s_{axis} = s_{axis} - (\w+)\[idx\];$",
                           out, flags=re.M)
        if len(chain) < 2:
            continue
        block = "\n".join(f"    s_{axis} = s_{axis} - {name}[idx];" for name in chain)
        total = f"    float t_{axis} = {chain[0]}[idx];\n"
        total += "\n".join(f"    t_{axis} = t_{axis} + {name}[idx];"
                           for name in chain[1:])
        total += f"\n    s_{axis} = s_{axis} - t_{axis};"
        out = out.replace(block, total)
        sites += 1
    return out, sites


def _mutate_drop_one_pole(source: str) -> Tuple[str, int]:
    """The LAST contributor silently missing from every chain that has one."""
    sites = 0
    out = source
    for _target, _src, axis in dispersive.ELECTRIC_TERMS:
        chain = re.findall(rf"^    s_{axis} = s_{axis} - (\w+)\[idx\];$",
                           out, flags=re.M)
        if not chain:
            continue
        needle = f"    s_{axis} = s_{axis} - {chain[-1]}[idx];\n"
        if needle in out:
            out = out.replace(needle, "", 1)
            sites += 1
    return out, sites


def _mutate_regroup(source: str) -> Tuple[str, int]:
    needle = ("    float a = f[idx] + kps * src;\n"
              "    f[idx] = a - kms * prev;\n")
    return source.replace(needle, "    f[idx] = f[idx] + (kps * src - kms * prev);\n"), \
        source.count(needle)


def _mutate_drop_fw_store(source: str) -> Tuple[str, int]:
    needle = "    fw[idx] = src;\n"
    return source.replace(needle, ""), source.count(needle)


def _mutate_store_before_read(source: str) -> Tuple[str, int]:
    needle = ("    float prev = fw[idx];\n"
              "    fw[idx] = src;\n")
    replacement = ("    fw[idx] = src;\n"
                   "    float prev = fw[idx];\n")
    return source.replace(needle, replacement), source.count(needle)


#: THE LEFT OPERAND OF THE INVERSE-EPSILON MULTIPLY HAS TWO SPELLINGS, and the
#: first cut of these two rewrites knew only one. At arity 0 the emitted line is
#: ``float src_x = Dx[idx] * inv_eps_Ex[idx];`` and at every arity above it is
#: ``float src_x = s_x * inv_eps_Ex[idx];`` -- the chain's register, with no
#: ``[idx]``. A pattern requiring ``[idx]`` on the left matched NOTHING at the
#: arities the mutation plan scores (all of which are 2 and above), and the
#: 2026-08-20 keep leg reported both legs NOT ARMED with sites=0. That is the
#: gate working: an unarmed leg is refused by name rather than counted as a pass.
_INV_EPS_OPERANDS = r"(\w+(?:\[idx\])?)"


def _mutate_drop_inverse_epsilon(source: str) -> Tuple[str, int]:
    """The source becomes ``(D - sum P)``, unscaled. MUST BE CAUGHT."""
    pattern = rf"= {_INV_EPS_OPERANDS} \* inv_eps_E([xyz])\[idx\];"
    return re.subn(pattern, r"= \1;", source)[0], len(re.findall(pattern, source))


def _mutate_inv_eps_left(source: str) -> Tuple[str, int]:
    """A NULL: IEEE multiply commutes. Its discriminating sibling drops it."""
    pattern = rf"= {_INV_EPS_OPERANDS} \* (inv_eps_E[xyz]\[idx\]);"
    return re.subn(pattern, r"= \2 * \1;", source)[0], len(re.findall(pattern, source))


def _mutate_commute_scale(source: str) -> Tuple[str, int]:
    """A NULL on the accumulation. Its sibling regroups the same two lines."""
    needle = ("    float a = f[idx] + kps * src;\n"
              "    f[idx] = a - kms * prev;\n")
    replacement = ("    float a = f[idx] + src * kps;\n"
                   "    f[idx] = a - prev * kms;\n")
    return source.replace(needle, replacement), source.count(needle)


def _mutate_own_axis(source: str) -> Tuple[str, int]:
    pattern = r"kps_([yz])\[([jk])\], kms_\1\[\2\]"
    return re.subn(pattern, "kps_x[i], kms_x[i]", source)[0], \
        len(re.findall(pattern, source))


def _mutate_fortran_index(source: str) -> Tuple[str, int]:
    needle = ("    int k = idx % nz;\n"
              "    int j = (idx / nz) % ny;\n"
              "    int i = idx / (ny * nz);\n")
    replacement = ("    int i = idx % nx;\n"
                   "    int j = (idx / nx) % ny;\n"
                   "    int k = idx / (nx * ny);\n")
    return source.replace(needle, replacement), source.count(needle)


def _mutate_bind_Ez_inv_eps(source: str) -> Tuple[str, int]:
    pattern = r"inv_eps_E([xy])\[idx\]"
    return re.subn(pattern, "inv_eps_Ez[idx]", source)[0], \
        len(re.findall(pattern, source))


SOURCE_TRANSFORMS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "sum_then_subtract": _mutate_sum_then_subtract,
    "drop_one_pole": _mutate_drop_one_pole,
    "regroup_constitutive_dispersive": _mutate_regroup,
    "drop_fw_store_dispersive": _mutate_drop_fw_store,
    "store_fw_before_reading_prev_dispersive": _mutate_store_before_read,
    "drop_inverse_epsilon_dispersive": _mutate_drop_inverse_epsilon,
    "inv_eps_left": _mutate_inv_eps_left,
    "commute_constitutive_scale": _mutate_commute_scale,
    "own_axis_to_x_for_all_three": _mutate_own_axis,
    "fortran_order_index_decomposition": _mutate_fortran_index,
    "bind_Ez_inv_eps_for_all_three": _mutate_bind_Ez_inv_eps,
}


# ---------------------------------------------------------------------------
# THE FIXTURE
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``. Same shim every CUDA gate uses."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def arm_arrays(arm: str) -> Tuple[str, ...]:
    """The output arrays this arm writes -- what the comparison covers."""
    if arm == "arm1":
        return ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")
    if arm == "arm2":
        return ("Ex", "Ey", "Ez")
    return ()  # arm3's outputs are the P buffers, enumerated per case


def build(xp, spec: Dict[str, Any], courant: float, arity: Tuple[int, int, int],
          rng) -> Tuple[Any, Any, Any]:
    """A frozen ``(fields, layer, grid)`` triple at one spec and one pole arity.

    THE ARITY IS BUILT FROM REAL ``PolarizationState`` OBJECTS, never faked: a
    per-component sigma of zero is what ``PolarizationState`` filters on
    (dispersion.py:645-647), which is the engine's own spelling of "this term
    does not drive that component" and is exactly what MEEP's ``needs_P`` /
    ``trivial_sigma`` pair does. A fixture that instead poked ``_driven`` would
    be measuring an object the engine cannot build.
    """
    kwargs: Dict[str, Any] = {}
    if spec["mirrors"]:
        kwargs["symmetry"] = [Mirror("XYZ"[axis]) for axis in spec["mirrors"]]
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), xp=xp, courant=courant,
                **kwargs)
    layer = None
    if spec["pml"]:
        # THE LOW FACE IS DROPPED ON A MIRRORED AXIS: cell 0 lies ON the mirror
        # plane, a boundary condition rather than an outer wall, and an absorber
        # there would eat the mirrored half of the domain (pml.py:399-412).
        thickness = {
            "xyz"[axis]: ((0, 0) if int(grid.shape[axis]) < 6
                          else (0, max(2, int(grid.shape[axis]) // 4))
                          if grid.is_mirrored(axis)
                          else (max(2, int(grid.shape[axis]) // 4),) * 2)
            for axis in range(3)}
        layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    if spec["pml"]:
        fields.enable_pml_storage()

    # THREE INDEPENDENT inverse-epsilon volumes, never one array bound thrice --
    # binding one is defect 2 of the complex template and
    # ``bind_Ez_inv_eps_for_all_three`` could not be caught by a fixture that
    # collapsed them. Drawn AWAY from 1.0 or dropping the multiply is identity.
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray((np.float32(1.0) / values).astype(np.float32))
    fields.set_epsilon_volumes(epsilon, inverse)

    poles = max(arity)
    kinds = ("lorentzian", "drude")
    for index in range(poles):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: (0.35 + 0.04 * index
                        if index < arity[axis] else 0.0)
                 for axis, name in enumerate(("Ex", "Ey", "Ez"))}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))
    return fields, layer, grid


def state_names(fields, arm: str) -> Tuple[str, ...]:
    names = ["Ex", "Ey", "Ez", "Dx", "Dy", "Dz"]
    if arm == "arm1":
        names += ["f_w_Ex", "f_w_Ey", "f_w_Ez"]
    return tuple(names)


def seed_state(fields, grid, arm: str, value_class: str, arity, rng) -> Dict[str, np.ndarray]:
    """Seed every array the sub-step reads or writes, in the requested class.

    ``uniform``          -- [-1, 1), auxiliaries NONZERO so a mis-indexed
                            coefficient shows at launch one rather than two.
    ``subnormal_band``   -- the shared band draw, with the needles forced in.
    ``cancellation``     -- THIS FAMILY'S OWN. Each pole carries an equal share of
                            D so the chain walks the sum down towards zero, plus a
                            perturbation sweeping 1e-45..1e-30 on the LAST pole,
                            plus signed zeros and both ends of the subnormal
                            boundary in one cell in sixteen. The point is not
                            realism: it is that the verdict otherwise says nothing
                            about the band ``D - sum P`` actually visits.

    THE MATERIAL STAYS NORMAL under every class. Driving inverse epsilon into the
    band too would make every product underflow and the leg would measure the
    fixture rather than the policy's reach into this arithmetic.
    """
    xp = grid.xp
    shape = tuple(int(n) for n in grid.shape)
    names = state_names(fields, arm)
    pole_arrays: List[Tuple[Any, str]] = []
    for state in fields.polarizations:
        for component in state.driven():
            pole_arrays.append((state, component))

    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                for name in names}
    elif value_class == "subnormal_band":
        host = probe.subnormal_band_hosts(names, shape, rng)
    elif value_class == "cancellation":
        host = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                for name in names}
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")

    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))

    pole_host: Dict[str, np.ndarray] = {}
    by_component: Dict[str, List[Tuple[Any, str]]] = {"Ex": [], "Ey": [], "Ez": []}
    for state, component in pole_arrays:
        by_component[component].append((state, component))
    for component, entries in by_component.items():
        if not entries:
            continue
        if value_class == "cancellation":
            displacement = host["D" + component[1]]
            share = (displacement / np.float32(len(entries))).astype(np.float32)
            exponents = rng.uniform(-45.0, -30.0, size=shape)
            signs = np.where(rng.integers(0, 2, size=shape) == 0, -1.0, 1.0)
            perturbation = (signs * np.power(10.0, exponents)).astype(np.float32)
            values = [share.copy() for _ in entries]
            values[-1] = (share - perturbation).astype(np.float32)
        elif value_class == "subnormal_band":
            values = list(probe.subnormal_band_hosts(
                [f"P{index}" for index in range(len(entries))], shape, rng).values())
        else:
            values = [rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                      for _ in entries]
        for (state, _component), array in zip(entries, values):
            state.P[component][...] = xp.asarray(np.ascontiguousarray(array))
            state.P_prev[component][...] = xp.asarray(np.ascontiguousarray(
                rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)))
            pole_host[f"P[{component}][{id(state)}]"] = array

    if value_class == "cancellation":
        # Exact zeros, negative zeros and both ends of the subnormal boundary in
        # one cell in sixteen, in every array the sub-step reads or writes except
        # the material -- a zero inverse epsilon is not a configuration the
        # engine builds. The comparison stays BYTEWISE and never asks for a
        # magnitude, because CuPy's own flush-to-zero is a measurement hazard here.
        specials = np.array([0.0, -0.0, np.float32(1.1754944e-38),
                             np.float32(1.1754942e-38), np.float32(1e-45),
                             np.float32(-1e-45)], dtype=np.float32)
        for name in names:
            array = to_host(getattr(fields, name)).copy()
            picks = rng.integers(0, 16, size=shape) == 0
            choice = specials[rng.integers(0, specials.size, size=shape)]
            array = np.where(picks, choice, array).astype(np.float32)
            getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(array))
            host[name] = array

    host.update(pole_host)
    return host


def snapshot(fields, arm: str) -> Dict[str, np.ndarray]:
    if arm == "arm3":
        out = {}
        for index, state in enumerate(fields.polarizations):
            for component in state.driven():
                out[f"P{index}_{component}"] = to_host(state.P[component]).copy()
                out[f"Pprev{index}_{component}"] = to_host(
                    state.P_prev[component]).copy()
        return out
    return {name: to_host(getattr(fields, name)).copy() for name in arm_arrays(arm)}


def full_state(fields, arm: str) -> Dict[str, Any]:
    """Everything a leg must restore: the outputs AND the inputs it consumes."""
    frozen = {name: to_host(getattr(fields, name)).copy()
              for name in state_names(fields, arm)}
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            frozen[f"P{index}_{component}"] = to_host(state.P[component]).copy()
            frozen[f"Pprev{index}_{component}"] = to_host(
                state.P_prev[component]).copy()
    return frozen


def restore(fields, frozen: Dict[str, Any]) -> None:
    xp = fields.grid.xp
    for name in state_names(fields, "arm1"):
        if name in frozen:
            getattr(fields, name)[...] = xp.asarray(frozen[name])
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            key, prev = f"P{index}_{component}", f"Pprev{index}_{component}"
            if key in frozen:
                state.P[component][...] = xp.asarray(frozen[key])
            if prev in frozen:
                state.P_prev[component][...] = xp.asarray(frozen[prev])


def advance_sources(fields, arm: str) -> None:
    """Move the input between launches, exactly the same way on both paths.

    A real run's curl rewrites D before every constitutive call. Held fixed,
    ``f_w`` reaches a fixed point and 60 launches measure what one launch does.
    On arm 3 the DRIVE is what has to move, and the drive is the stored E.
    """
    names = ("Dx", "Dy", "Dz") if arm != "arm3" else ("Ex", "Ey", "Ez")
    for name in names:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# THE FLOORS
# ---------------------------------------------------------------------------

def moved_fraction(before: Dict[str, np.ndarray],
                   after: Dict[str, np.ndarray]) -> float:
    moved = total = 0
    for name in before:
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def pole_subtraction_bites(fields, plan) -> Dict[str, Any]:
    """Does ``D - sum P`` differ BITWISE from ``D``?

    A case whose poles are all zero is a NON-DISPERSIVE case wearing dispersive
    clothes: ``drop_one_pole``, ``sum_then_subtract`` and ``reverse_pole_order``
    would every one come back UNCAUGHT, for a reason about the fixture rather
    than about the kernel. This is the floor that refuses such a case before it
    is scored.
    """
    out: Dict[str, Any] = {"per_component": {}, "moved": 0, "total": 0}
    for target, source, _axis in dispersive.ELECTRIC_TERMS:
        chain = plan.get(target, ())
        displacement = to_host(getattr(fields, source)).astype(np.float32)
        value = displacement.copy()
        for array in chain:
            value = (value - to_host(array).astype(np.float32)).astype(np.float32)
        a = np.ascontiguousarray(displacement).ravel().view(np.uint32)
        b = np.ascontiguousarray(value).ravel().view(np.uint32)
        moved = int(np.count_nonzero(a != b))
        out["per_component"][target] = {"contributors": len(chain), "moved": moved,
                                        "total": int(a.size)}
        out["moved"] += moved
        out["total"] += int(a.size)
    return out


def coefficient_profile_bites(layer) -> Dict[str, Any]:
    """Does the absorber profile differ from the interior identity anywhere?

    ``kps = kms = 1`` is the pass-through. A case whose whole profile is that
    identity cannot distinguish a coefficient-INDEX error on any axis, so
    ``own_axis_to_x_for_all_three``, ``fortran_order_index_decomposition`` and
    ``swap_constitutive_sublattice`` would all be no-ops on it.
    """
    deviation = 0.0
    per_axis = {}
    for axis in "xyz":
        axis_max = 0.0
        for stem in ("kps", "kms"):
            values = to_host(getattr(layer, f"{stem}_{axis}_h")).astype(np.float64)
            axis_max = max(axis_max, float(np.max(np.abs(values - 1.0))))
        per_axis[axis] = axis_max
        deviation = max(deviation, axis_max)
    axes_absorbing = sum(1 for value in per_axis.values() if value > 0.0)
    return {"per_axis": per_axis, "max_deviation": deviation,
            "axes_absorbing": axes_absorbing, "meets_floor": deviation > 0.0}


def inverse_epsilon_bites(fields) -> Dict[str, Any]:
    deviation = 0.0
    distinct = set()
    for target, _source, _axis in dispersive.ELECTRIC_TERMS:
        volume = to_host(fields.inverse_epsilon_for(target)).astype(np.float64)
        deviation = max(deviation, float(np.max(np.abs(volume - 1.0))))
        distinct.add(hashlib.sha256(
            np.ascontiguousarray(volume, dtype=np.float64).tobytes()).hexdigest())
    return {"max_deviation_from_identity": deviation,
            "distinct_volumes": len(distinct),
            "meets_floor": deviation > 0.0 and len(distinct) == 3}


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "shape_full": [int(n) for n in grid.shape_full],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
    }


# ---------------------------------------------------------------------------
# THE TWO KERNEL-SIDE BACKENDS
# ---------------------------------------------------------------------------

def tables_for(layer, host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """The six HALF-INTEGER kps/kms views, or a deliberately corrupted set."""
    suffix = "_h"
    if host_mutation == "swap_constitutive_sublattice":
        suffix = ""  # the INTEGER sub-lattice: a half-cell error, not a crash
    tables = {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
              for axis in "xyz" for stem in ("kps", "kms")}
    if host_mutation == "swap_kps_kms":
        base = dict(tables)
        for axis in "xyz":
            tables[f"kps_{axis}"] = base[f"kms_{axis}"]
            tables[f"kms_{axis}"] = base[f"kps_{axis}"]
    return tables


def plan_for(fields, host_mutation: Optional[str] = None) -> Dict[str, List[Any]]:
    plan = dispersive.resolve_pole_plan(fields)
    if host_mutation == "reverse_pole_order":
        plan = {target: list(reversed(chain)) for target, chain in plan.items()}
    elif host_mutation == "drop_one_pole_from_plan":
        plan = {target: list(chain[:-1]) if chain else list(chain)
                for target, chain in plan.items()}
    return plan


def run_kernel_cuda(arm: str, fields, layer, tables, plan) -> None:
    """The SHIPPED launcher, through its own public entry point.

    ``tables`` and ``plan`` are passed explicitly -- the entry points'
    keyword-only gate doors, which exist so a harness can hand in deliberately
    mis-paired ones. Passing the layer instead would derive them correctly and
    disarm every host mutation.
    """
    if arm == "arm1":
        dispersive.update_E_fused_pml_real_dispersive(fields, tables=tables, plan=plan)
    else:
        dispersive.update_E_no_pml_real_dispersive(fields, plan=plan)
    cp.cuda.runtime.deviceSynchronize()


def _broadcast(vector, axis: int):
    shape = [1, 1, 1]
    shape[axis] = int(np.asarray(vector).size)
    return np.asarray(to_host(vector)).reshape(shape)


def run_kernel_numpy(arm: str, fields, layer, tables, plan,
                     defect: Optional[str] = None) -> None:
    """The EMITTED device body, transcribed to NumPy float32, line for line.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the NVRTC
    contraction guard exists for, it does not exercise the subnormal policy, and
    it is not the shipped bytes. What it settles is (a) whether the transcription
    reproduces ``stepping`` at all and (b) whether this harness can fail -- both
    on a laptop, before a device slot is spent.
    """
    for axis_index, (target, source, axis) in enumerate(dispersive.ELECTRIC_TERMS):
        chain = [to_host(array).astype(np.float32) for array in plan[target]]
        if defect == "sum_then_subtract" and len(chain) >= 2:
            total = chain[0]
            for array in chain[1:]:
                total = (total + array).astype(np.float32)
            value = (to_host(getattr(fields, source)).astype(np.float32)
                     - total).astype(np.float32)
        else:
            if defect == "drop_one_pole" and chain:
                chain = chain[:-1]
            value = to_host(getattr(fields, source)).astype(np.float32)
            for array in chain:
                value = (value - array).astype(np.float32)
        material = to_host(fields.inverse_epsilon_for(target)).astype(np.float32)
        if defect == "drop_inverse_epsilon_dispersive":
            product = value
        elif defect == "inv_eps_left":
            product = (material * value).astype(np.float32)
        else:
            product = (value * material).astype(np.float32)
        if arm == "arm2":
            getattr(fields, target)[...] = fields.grid.xp.asarray(product)
            continue
        kps = _broadcast(tables[f"kps_{axis}"], axis_index)
        kms = _broadcast(tables[f"kms_{axis}"], axis_index)
        auxiliary = getattr(fields, "f_w_" + target)
        field = getattr(fields, target)
        previous = to_host(auxiliary).copy()
        auxiliary[...] = fields.grid.xp.asarray(product)
        accumulated = (to_host(field).astype(np.float32)
                       + kps * product).astype(np.float32)
        field[...] = fields.grid.xp.asarray(
            (accumulated - kms * previous).astype(np.float32))


# --- arm 3 -----------------------------------------------------------------

def arm3_drive(fields, host_mutation: Optional[str]) -> Callable[[str], Any]:
    """The bound method the launcher binds, or the CONTROL that binds the wrong array.

    The shipped default is ``fields.drive_field``, which without a layer returns
    the STORED E (fields.py:1162). ``arm3_wrong_drive`` binds ``D`` instead --
    the array a reader who conflated "the drive" with "the displacement" would
    reach for, and the mirror image of the sibling track's measured
    ``B_ade_wrong_drive_CONTROL``.
    """
    if host_mutation == "arm3_wrong_drive":
        return lambda component: getattr(fields, "D" + component[1])
    return fields.drive_field


def _arm3_local_loop(fields, drive, hoisted: bool) -> int:
    """``ade_kernels.update_P_fused_pml_real``'s loop, copied, with a hoist switch.

    WHY A COPY EXISTS AT ALL. The stale-plan defect is a defect in the LAUNCHER's
    pointer resolution, and the shipped launcher has no door for it -- it re-reads
    ``state.P``/``P_prev``/``_scratch`` inside its own loop and there is no
    argument that makes it stop. Monkeypatching it would test the patch.

    So the defect is planted in a local copy, and the copy is made accountable by
    a NULL control: ``hoisted=False`` runs the same copy unhoisted and MUST come
    back bit-identical to ``stepping.update_P``. Only with that control does
    ``hoisted=True`` diverging say anything about the hoist rather than about the
    hand-written copy. The DEVICE KERNEL is the shipped one in both cases -- this
    copies the Python loop around it, never the CUDA text.
    """
    from meep_gpu.cuda_kernels import ade_kernels  # noqa: PLC0415

    entries = [(state, component)
               for state in fields.polarizations for component in state.driven()]
    cached = None
    if hoisted:
        cached = {(id(state), component):
                  (state.P[component], state.P_prev[component], state._scratch)
                  for state, component in entries}
    launches = 0
    for state, component in entries:
        if cached is not None:
            p, p_prev, scratch = cached[(id(state), component)]
        else:
            p, p_prev, scratch = (state.P[component], state.P_prev[component],
                                  state._scratch)
        w = drive(component)
        sigma = state.sigma[component]
        volume = bool(ade_kernels.ade_sigma_is_volume(state, component))
        ade_kernels._launch(ade_kernels._get_kernel(volume), scratch, p, p_prev,
                            sigma, w, tuple(state._coefficients), volume)
        launches += 1
        state.P[component] = scratch
        state.P_prev[component] = p
        state._scratch = p_prev
    return launches


def run_arm3_cuda(fields, host_mutation: Optional[str]) -> int:
    """The SHIPPED launcher, except on the two legs that measure the launcher itself."""
    from meep_gpu.cuda_kernels import ade_kernels  # noqa: PLC0415
    drive = arm3_drive(fields, host_mutation)
    if host_mutation in ("arm3_stale_pole_plan", "arm3_launcher_copy_NULL"):
        launches = _arm3_local_loop(
            fields, drive, hoisted=host_mutation == "arm3_stale_pole_plan")
    else:
        launches = ade_kernels.update_P_fused_pml_real(fields, None, drive=drive)
    cp.cuda.runtime.deviceSynchronize()
    return launches


def run_arm3_numpy(fields, host_mutation: Optional[str]) -> int:
    """``dispersion.PolarizationState.update``'s expression, with the ROTATION.

    ``((p*c_now) + (c_prev*q)) + (c_drive*(s*w))`` -- left-associated across the
    array path's two ``+=`` (dispersion.py:684-686), with ``s*w`` formed before
    ``c_drive`` multiplies it, and the three buffers rotated only AFTER the write
    (:689-691)::

        state.P[component]      = scratch     # this step's result
        state.P_prev[component] = p           # what P held
        state._scratch          = p_prev      # the retired history

    ``arm3_stale_pole_plan`` resolves every buffer BEFORE the loop and reuses the
    views. That is EXACT for the first component of the first launch and wrong
    from the second onwards: the retired history becomes the NEXT component's
    scratch, so a cached plan writes this component's result over the buffer the
    previous component just claimed. The certified launcher carries the hazard in
    as many words and this is it, armed.
    """
    drive = arm3_drive(fields, host_mutation)
    xp = fields.grid.xp
    entries = [(state, component)
               for state in fields.polarizations for component in state.driven()]
    cached = None
    if host_mutation in ("arm3_stale_pole_plan",):
        cached = {(id(state), component):
                  (state.P[component], state.P_prev[component], state._scratch)
                  for state, component in entries}
    launches = 0
    for state, component in entries:
        if cached is not None:
            p_array, q_array, scratch = cached[(id(state), component)]
        else:
            p_array, q_array, scratch = (state.P[component],
                                         state.P_prev[component], state._scratch)
        c_now, c_prev, c_drive = state._coefficients
        w = to_host(drive(component)).astype(np.float32)
        p = to_host(p_array).astype(np.float32)
        q = to_host(q_array).astype(np.float32)
        sigma = state.sigma[component]
        s = (to_host(sigma).astype(np.float32) if hasattr(sigma, "shape")
             else np.float32(sigma))
        value = (((p * np.float32(c_now)) + (np.float32(c_prev) * q))
                 + (np.float32(c_drive) * (s * w))).astype(np.float32)
        scratch[...] = xp.asarray(value)
        # The rotation, transcribed, and only once the write landed.
        state.P[component] = scratch
        state.P_prev[component] = p_array
        state._scratch = q_array
        launches += 1
    return launches


# ---------------------------------------------------------------------------
# ONE CASE
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], arm: str, arity: Tuple[int, int, int],
             courant: float, value_class: str, guard: str,
             host_mutation: Optional[str] = None,
             numpy_defect: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML``/
    ``PolarizationState`` objects. There is no second transcription on the oracle
    leg to drift: the kernel is compared against the thing it claims to
    reproduce, byte for byte, from ONE frozen state.
    """
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    # SEEDED FROM A DIGEST, NEVER FROM hash(). Python salts hash() of a tuple
    # containing strings with PYTHONHASHSEED, so such a gate draws a different
    # fixture every process and a failing case cannot be replayed.
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{arm}|{'-'.join(str(v) for v in arity)}|"
            f"{courant}|{value_class}".encode()).digest()[:4], "big"))

    fields, layer, grid = build(xp, spec, courant, arity, rng)
    host = seed_state(fields, grid, arm, value_class, arity, rng)

    case: Dict[str, Any] = {
        "label": spec["label"], "arm": arm, "arity": list(arity),
        "courant": courant, "value_class": value_class, "guard": guard,
        "backend": backend, "host_mutation": host_mutation,
        "numpy_defect": numpy_defect,
        "boundaries": list(spec["boundaries"]), "mirrors": list(spec["mirrors"]),
        "structure": structure_facts(grid),
        # A ZERO-ARITY CASE IS A REDUCTION CONTROL, NOT A COVERAGE CLAIM. The
        # predicate refuses it ("no susceptibility is registered": that run is
        # coverage.covers_real_pml_constitutive's), and it is scored anyway
        # because the emitted body at arity (0,0,0) is character-identical to the
        # certified non-dispersive kernel's three source lines -- so this leg is
        # the emitter reproducing certified bytes, checked on a device.
        "is_reduction_control": max(arity) == 0,
        "operand_census": operand_census(
            {name: value for name, value in host.items()
             if isinstance(value, np.ndarray)}),
    }

    # THE PREDICATE'S ANSWER, recorded rather than acted on.
    if arm == "arm1":
        covered, reason = dispersive.covers_real_pml_dispersive_constitutive(
            fields, layer, grid)
    elif arm == "arm2":
        covered, reason = dispersive.covers_no_pml_dispersive_constitutive(
            fields, layer, grid)
    else:
        covered, reason = dispersive.covers_no_pml_ade_update_p(fields, layer, grid)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    plan = plan_for(fields, None)
    case["pole_subtraction_bites"] = pole_subtraction_bites(fields, plan)
    case["inverse_epsilon_bites"] = inverse_epsilon_bites(fields)
    if arm == "arm1":
        case["coefficient_profile_bites"] = coefficient_profile_bites(layer)

    floors: List[str] = []
    if max(arity) > 0 and case["pole_subtraction_bites"]["moved"] == 0:
        floors.append("D - sum P is bitwise equal to D everywhere; the pole chain "
                      "does not bite and every pole mutation would be a no-op")
    if not case["inverse_epsilon_bites"]["meets_floor"]:
        floors.append("the inverse-epsilon volumes are the identity or are not "
                      "three distinct arrays; drop_inverse_epsilon and "
                      "bind_Ez_inv_eps could not be caught")
    if arm == "arm1" and not case["coefficient_profile_bites"]["meets_floor"]:
        floors.append("the absorber profile is the interior identity everywhere; "
                      "a coefficient-index error is invisible on this case")
    if floors:
        case["skipped"] = floors[0]
        case["floors_failed"] = floors
        case["seconds"] = time.time() - started
        return case

    frozen = full_state(fields, arm)
    before = snapshot(fields, arm)

    # Leg 1: the oracle.
    if arm == "arm3":
        stepping.update_P(fields, layer)
    else:
        stepping.update_E(fields, layer)
    reference = snapshot(fields, arm)
    moved = moved_fraction(before, reference)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; zero-init is a fixed point of this recurrence "
                           "and a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case
    if arm == "arm3":
        # WHICH BUFFERS THE ORACLE LEFT WHERE. The rotation moves three names per
        # component, so the comparison has to be against the same permutation the
        # kernel path will produce, and the snapshot is taken by (state, component)
        # rather than by identity for exactly that reason.
        case["oracle_launches"] = sum(
            len(state.driven()) for state in fields.polarizations)

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    tables = tables_for(layer, host_mutation) if arm == "arm1" else None
    mutated_plan = plan_for(fields, host_mutation)
    if arm == "arm3":
        launches = (run_arm3_cuda(fields, host_mutation) if backend == "cuda"
                    else run_arm3_numpy(fields, host_mutation))
        case["kernel_launches"] = launches
    elif backend == "cuda":
        run_kernel_cuda(arm, fields, layer, tables, mutated_plan)
    else:
        run_kernel_numpy(arm, fields, layer, tables, mutated_plan,
                         defect=numpy_defect)
    got = snapshot(fields, arm)
    parts = {name: bit_compare(reference[name], got[name]) for name in reference}
    case["single_launch"] = combine(parts)

    # Leg 3: the multi-step. The auxiliary (arm 1) and the P history (arm 3) are
    # STATE, and a tree that gets the field right and the state wrong is correct
    # for exactly one launch and wrong forever after. ONE ``Fields`` object, run
    # twice from the same frozen state -- two objects would mean two epsilon draws
    # and a fixture solving two different problems.
    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        if arm == "arm3":
            stepping.update_P(fields, layer)
        else:
            stepping.update_E(fields, layer)
        advance_sources(fields, arm)
    oracle_multi = snapshot(fields, arm)

    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        if arm == "arm3":
            if backend == "cuda":
                run_arm3_cuda(fields, host_mutation)
            else:
                run_arm3_numpy(fields, host_mutation)
        elif backend == "cuda":
            run_kernel_cuda(arm, fields, layer, tables,
                            plan_for(fields, host_mutation))
        else:
            run_kernel_numpy(arm, fields, layer, tables,
                             plan_for(fields, host_mutation), defect=numpy_defect)
        advance_sources(fields, arm)
    multi = snapshot(fields, arm)
    case["multi_step"] = combine(
        {name: bit_compare(oracle_multi[name], multi[name]) for name in oracle_multi})
    case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# THE SWEEP
# ---------------------------------------------------------------------------

def case_product(product: str) -> List[Tuple[Dict[str, Any], str, Tuple[int, int, int],
                                             float, str]]:
    out = []
    specs = SPECS if product == "full" else SPECS[:2] + SPECS[6:8]
    arities = ARITIES if product == "full" else ((0, 0, 0), (2, 2, 2))
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform", "cancellation")
    for spec in specs:
        for arm in ARMS_BY_SPEC[spec["label"]]:
            for arity in arities:
                if arm == "arm3" and max(arity) == 0:
                    continue  # update_P with nothing driven is not this arm's slot
                for courant in courants:
                    for value_class in classes:
                        out.append((spec, arm, arity, courant, value_class))
    return out


def run_sweep(results, out_path, backend, product, guard):
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, (spec, arm, arity, courant, value_class) in enumerate(plan, start=1):
        case = one_case(backend, spec, arm, arity, courant, value_class, guard)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] {index}/{len(plan)} {spec['label']} {arm} "
                f"n={arity} c={courant} {value_class} SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] {index}/{len(plan)} {spec['label']} {arm} n={arity} "
            f"c={courant} {value_class} shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"moved={case['oracle_moved']:.3f} "
            f"poles_bite={case['pole_subtraction_bites']['moved']} "
            f"({case['seconds']:.1f} s)")
    return cases


def leg_caught(case: Dict[str, Any]) -> bool:
    """Did this mutated leg diverge at EITHER granularity?

    A defect that leaves the field right and the auxiliary wrong is bit-identical
    on the target at launch one; a stale pole plan is EXACT at launch one by
    construction. Both legs are compared, and a mutation caught by either is
    caught -- scoring only the single launch would call inert a defect the
    60-launch leg exists to find.
    """
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def mutation_plan(product: str, arms: Sequence[str], pole_defect: bool):
    """One case per (spec, arm) at the INEXACT courant, at an arity that can see it.

    POLE MUTATIONS ARE PINNED TO :data:`MUTATION_ARITIES`, all of which are two or
    above, because ``sum_then_subtract`` and ``reverse_pole_order`` are PROVABLY
    inert at one pole. A leg scored at arity 1 would report a harness defect as a
    kernel pass, which is the failure the header measures rather than asserts.
    """
    arities = MUTATION_ARITIES if pole_defect else ((2, 2, 2),)
    if product != "full":
        arities = arities[:1]
    out = []
    for spec in SPECS:
        for arm in ARMS_BY_SPEC[spec["label"]]:
            if arm not in arms:
                continue
            for arity in arities:
                out.append((spec, arm, arity, INEXACT_COURANT, "uniform"))
    if product != "full":
        out = out[:3]
    return out


def _score(legs: List[Dict[str, Any]], must_be_caught: bool) -> Dict[str, Any]:
    scored = [case for case in legs if not case.get("skipped")]
    caught = sum(1 for case in scored if leg_caught(case))
    # NO LEGS is its own verdict, never "UNCAUGHT". A mutation that was never run
    # has not been shown inert; it has not been asked. Collapsing the two is how
    # a gate acquires a silent hole.
    if not scored:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = ("CAUGHT" if caught == len(scored)
                   else "PARTIAL" if caught else "UNCAUGHT")
    else:
        verdict = "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"
    return {"ran": len(scored), "caught": caught, "uncaught": len(scored) - caught,
            "must_be_caught": must_be_caught, "verdict": verdict, "cases": legs}


def run_host_mutations(results, out_path, backend, product):
    """Every table-level and plan-level defect, on BOTH backends."""
    out: Dict[str, Any] = {}
    for name, spec in HOST_MUTATIONS.items():
        backends = spec.get("backends")
        if backends and backend not in backends:
            # NOT APPLICABLE is its own verdict and is neither a pass nor a
            # failure. A leg this backend cannot arm has not been shown inert.
            out[name] = {"ran": 0, "caught": 0, "uncaught": 0,
                         "must_be_caught": spec["must_be_caught"],
                         "verdict": "NOT APPLICABLE ON THIS BACKEND",
                         "why": spec["why"], "cases": [],
                         "applicable_backends": list(backends)}
            log(f"[host-mut] {name}: NOT APPLICABLE on backend {backend!r}")
            results["host_mutations"] = out
            save(results, out_path)
            continue
        legs = []
        for case_spec, arm, arity, courant, value_class in mutation_plan(
                product, spec["arms"], spec["pole_defect"]):
            legs.append(one_case(backend, case_spec, arm, arity, courant,
                                 value_class, "fmad_false", host_mutation=name))
        out[name] = dict(_score(legs, spec["must_be_caught"]), why=spec["why"],
                         pole_defect=spec["pole_defect"],
                         scored_arities=sorted({tuple(c["arity"]) for c in legs}))
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_numpy_defects(results, out_path, product):
    """The NumPy backend's answer to "can this harness fail at all?".

    The device leg plants defects in the emitted TEXT; a laptop has no compiler,
    so the same defects are planted in the transcription instead. This is not
    evidence about the shipped bytes and the record says so -- what it settles is
    that the comparator is sensitive to each defect BEFORE a device slot is spent,
    and that the two pole defects are invisible at one pole and lethal at two.
    """
    out: Dict[str, Any] = {}
    for name in ("sum_then_subtract", "drop_one_pole",
                 "drop_inverse_epsilon_dispersive", "inv_eps_left"):
        spec = SOURCE_MUTATIONS[name]
        legs = []
        for case_spec, arm, arity, courant, value_class in mutation_plan(
                product, spec["arms"], spec["pole_defect"]):
            legs.append(one_case("numpy", case_spec, arm, arity, courant,
                                 value_class, "fmad_false", numpy_defect=name))
        out[name] = _score(legs, spec["must_be_caught"])
        log(f"[numpy-defect] {name}: caught {out[name]['caught']}/"
            f"{out[name]['ran']} -> {out[name]['verdict']}")
        results["numpy_defects"] = out
        save(results, out_path)

    # THE INVISIBILITY MEASUREMENT, taken rather than quoted.
    invisibility = {}
    for name in ("sum_then_subtract",):
        rows = []
        for arity in ((1, 1, 1), (2, 2, 2), (5, 5, 5)):
            case = one_case("numpy", SPECS[0], "arm1", arity, INEXACT_COURANT,
                            "uniform", "fmad_false", numpy_defect=name)
            rows.append({"arity": list(arity), "skipped": case.get("skipped"),
                         "differing_floats":
                             None if case.get("skipped")
                             else case["single_launch"]["differing_floats"]})
        invisibility[name] = rows
        log(f"[invisibility] {name}: " + ", ".join(
            f"n={row['arity'][0]} -> {row['differing_floats']}" for row in rows))
    results["pole_defect_invisibility_at_one_pole"] = invisibility
    save(results, out_path)
    return out


def run_source_mutations(results, out_path, product):
    """Every device-text defect, applied to the EMITTED strings and recompiled.

    The compile memo is keyed through the SOURCE
    (``compile_cache.kernel_cache_key``), so a mutated body is a miss and reaches
    NVRTC. Every leg records how many constructions came from the mutated bytes,
    because a leg reporting a pass for a mutation it never applied is worse than
    no leg -- three measured instances on the sibling track.
    """
    from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

    out: Dict[str, Any] = {}
    try:
        for name, spec in SOURCE_MUTATIONS.items():
            transform = SOURCE_TRANSFORMS[name]
            sites_seen = {"count": 0}
            digests: Dict[str, str] = {}

            def hook(arm, counts, code, _transform=transform, _seen=sites_seen,
                     _digests=digests):
                mutated, sites = _transform(code)
                _seen["count"] += sites
                if sites and mutated != code:
                    _digests[hashlib.sha256(
                        mutated.encode("utf-8")).hexdigest()] = f"{arm}:{counts}"
                return mutated

            dispersive.SOURCE_TRANSFORM = hook
            dispersive._clear_kernel_cache()
            compile_cache.clear_compile_log()
            legs = []
            for case_spec, arm, arity, courant, value_class in mutation_plan(
                    product, spec["arms"], spec["pole_defect"]):
                legs.append(one_case("cuda", case_spec, arm, arity, courant,
                                     value_class, "fmad_false"))
            dispersive.SOURCE_TRANSFORM = None
            dispersive._clear_kernel_cache()

            from_mutated = sum(1 for entry in compile_cache.compile_log()
                               if entry["source_sha256"] in digests)
            entry = dict(_score(legs, spec["must_be_caught"]),
                         armed=bool(sites_seen["count"]),
                         sites=sites_seen["count"],
                         distinct_mutated_bodies=len(digests),
                         kernel_constructions_from_mutated_bytes=from_mutated,
                         pole_defect=spec["pole_defect"],
                         scored_arities=sorted({tuple(c["arity"]) for c in legs}))
            if not entry["armed"]:
                entry["verdict"] = "NOT ARMED"
                entry["why"] = ("the rewrite matched no site in any emitted body; "
                                "the mutation and the emitter have drifted apart")
            elif from_mutated == 0 and entry["ran"]:
                # A leg reporting a pass for a mutation it never applied is worse
                # than no leg: the rewrite matched, the memo is keyed through the
                # source, and yet nothing was built from these bytes.
                entry["verdict"] = "UNACCOUNTED"
                entry["why"] = ("no kernel construction used the mutated bytes; "
                                "this leg did not exercise the mutation")
            out[name] = entry
            log(f"[src-mut] {name}: caught {entry['caught']}/{entry['ran']} "
                f"sites={entry['sites']} builds_from_mutated={from_mutated} "
                f"-> {entry['verdict']}")
            results["source_mutations"] = out
            save(results, out_path)
    finally:
        dispersive.SOURCE_TRANSFORM = None
        dispersive._clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# THE VERDICT
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [case for case in primary if not case.get("skipped")]
    by_arm = {arm: [case for case in scored if case["arm"] == arm] for arm in ARMS}

    for arm in ARMS:
        if not by_arm[arm]:
            reasons.append(f"{arm} was not scored on a single case")
    single_ok = [case for case in scored if case["single_launch"]["bit_identical"]]
    multi_cases = [case for case in scored if "multi_step" in case]
    multi_ok = [case for case in multi_cases if case["multi_step"]["bit_identical"]]
    if len(single_ok) != len(scored):
        reasons.append(f"single-launch divergence on {len(scored) - len(single_ok)} "
                       f"of {len(scored)} cases")
    if len(multi_ok) != len(multi_cases):
        reasons.append(f"multi-step divergence on {len(multi_cases) - len(multi_ok)} "
                       f"of {len(multi_cases)} cases")

    # EVERY SWEPT ARITY MUST HAVE BEEN SCORED on arm 1 and arm 2, or the cap is
    # not a measured ceiling and POLE_COUNT_CAP is a number rather than evidence.
    for arm in ("arm1", "arm2"):
        arities = {tuple(case["arity"]) for case in by_arm[arm]}
        missing = [a for a in ARITIES if a not in arities
                   and not (arm == "arm3" and max(a) == 0)]
        if missing:
            reasons.append(f"{arm} did not score arities {missing}; "
                           f"POLE_COUNT_CAP claims every one of them was run")
    # A MULTI-POLE CASE ON EVERY ARM THAT HAS A CHAIN, or the whole pole-order
    # battery rests on cases where it is provably inert.
    for arm in ("arm1", "arm2"):
        if not [case for case in by_arm[arm] if max(case["arity"]) >= 2]:
            reasons.append(f"{arm} scored no case with two or more poles; every "
                           f"pole-order defect is provably inert below that")
    # BOTH FOLD CLASSES on arm 1: five of its eight corpus rows are folded.
    folded = [case for case in by_arm["arm1"] if case["mirrors"]]
    if not folded:
        reasons.append("arm1 scored no folded case; five of its eight corpus rows "
                       "are folded and the predicate admits a fold")
    if not [case for case in folded if len(case["mirrors"]) == 2]:
        reasons.append("arm1 scored no TWO-plane fold; the predicate admits two "
                       "and admitting the ceiling unmeasured is admitting by "
                       "argument alone")
    # THE CANCELLATION CLASS, which is the only one that puts D - sum P in the band.
    classes = {case["value_class"] for case in scored}
    if "cancellation" not in classes:
        reasons.append("the cancellation value class was never scored; without it "
                       "the verdict says nothing about the band the pole chain visits")
    if "subnormal_band" not in classes:
        reasons.append("the subnormal_band value class was never scored")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NOT APPLICABLE ON THIS BACKEND":
            continue  # recorded, neither a pass nor a failure -- see its "why"
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is not inert")
        elif not leg["must_be_caught"] and leg["verdict"] != "NULL CONFIRMED":
            reasons.append(f"host NULL {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']}); a null that fires is "
                           f"reporting a defect in the harness or the kernel")
        elif leg["must_be_caught"] and leg["verdict"] != "CAUGHT":
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for name, leg in results.get("source_mutations", {}).items():
        if leg.get("verdict") in ("NOT ARMED", "UNACCOUNTED", "NO LEGS"):
            reasons.append(f"source mutation {name} is {leg['verdict']}: "
                           f"{leg.get('why', '')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {name} is {leg['verdict']}")
    for name, leg in results.get("numpy_defects", {}).items():
        if leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"numpy defect leg {name} is {leg['verdict']}")

    control = [case for case in sweep.get("default_no_options", [])
               if not case.get("skipped") and case["courant"] == INEXACT_COURANT]
    control_diverged = [case for case in control
                        if not case["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored at the "
                    "inexact courant" if not control else
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
        "scored_by_arm": {arm: len(by_arm[arm]) for arm in ARMS},
        "single_launch_identical": len(single_ok),
        "multi_step_identical": len(multi_ok),
        "multi_step_cases": len(multi_cases),
        "arities_scored": sorted({tuple(case["arity"]) for case in scored}),
        "value_classes_scored": sorted(classes),
        "claim": (
            "arm1 and arm2 -- the two emitted dispersive update_E kernels -- are "
            "byte-identical to stepping.update_E on real engine objects at every "
            "swept pole arity, folded and unfolded, under both float32 subnormal "
            "policies; and arm3 -- the SHIPPED ade_kernels update_P pair, "
            "unchanged -- is byte-identical to stepping.update_P on a layerless "
            "run when Fields.drive_field hands it the stored E"),
        "does_not_claim": [
            "nothing dispatches any of these kernels; no module in meep_gpu "
            "imports cuda_kernels at all, so a predicate returning True licenses "
            "a MEASUREMENT and not a production step",
            "the off-diagonal dispersive row (absorbed_power_density.py) is NOT "
            "covered and is not measured here: it needs the offdiag row product "
            "crossed with the pole chain, and the shipped CUDA offdiag family "
            "additionally refuses a fold",
            "a Dcyl dispersive run is refused, not measured; the corpus drives "
            "none and the sibling constitutive family's Dcyl admission is not "
            "transferred here",
            "complex storage is a different family and is not measured here",
        ],
    }


def verdict_flips_against_planted_defect(results: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the verdict against a record with a defect planted in it.

    A GATE WHOSE VERDICT CANNOT GO RED IS NOT A GATE. Four independent defects
    are planted, one at a time, into a COPY of the finished record. Each must make
    ``released`` False ON ITS OWN.

    A PLANT THE RECORD CANNOT CARRY IS REPORTED, NOT PASSED AND NOT FAILED -- on
    the NumPy backend there are no source mutations to flip.
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
            else verdict["reasons"][0][:180]})
        if not applied:
            out["inapplicable"].append(name)
        elif not flipped:
            out["all_flipped"] = False

    def flip_a_case(record) -> bool:
        for case in record.get("sweep", {}).get("fmad_false", []):
            if not case.get("skipped") and case["single_launch"]["bit_identical"]:
                case["single_launch"]["bit_identical"] = False
                return True
        return False

    def flip_a_multi_step(record) -> bool:
        for case in record.get("sweep", {}).get("fmad_false", []):
            if (not case.get("skipped") and "multi_step" in case
                    and case["multi_step"]["bit_identical"]):
                case["multi_step"]["bit_identical"] = False
                return True
        return False

    def escape_a_mutation(record) -> bool:
        for leg in record.get("source_mutations", {}).values():
            if leg.get("armed") and leg.get("verdict") == "CAUGHT":
                leg["verdict"] = "UNCAUGHT"
                leg["caught"] = 0
                return True
        for leg in record.get("host_mutations", {}).values():
            if leg.get("verdict") == "CAUGHT":
                leg["verdict"] = "UNCAUGHT"
                leg["caught"] = 0
                return True
        return False

    def drop_the_multi_pole_cases(record) -> bool:
        """The clause the whole pole battery rests on: at least one case with >=2
        poles. Silencing it must go red, or a run that scored only single-pole
        rows would release with every pole defect provably inert."""
        cases = record.get("sweep", {}).get("fmad_false", [])
        touched = False
        for case in cases:
            if max(case.get("arity", [0])) >= 2:
                case["skipped"] = "planted: dropped for the falsification check"
                touched = True
        return touched

    plant("a_scored_case_diverges_at_one_launch", flip_a_case,
          "no scored case in the record was bit-identical to begin with")
    plant("a_scored_case_diverges_over_60_launches", flip_a_multi_step,
          "the record carries no bit-identical multi-step leg")
    plant("a_must_be_caught_mutation_escapes", escape_a_mutation,
          "the record carries no armed CAUGHT mutation (--skip-mutations)")
    plant("every_multi_pole_case_disappears", drop_the_multi_pole_cases,
          "the record carries no case with two or more poles")
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
        "gate": "cuda_dispersive",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("do the two emitted dispersive update_E kernels reproduce "
                     "stepping.update_E byte for byte at every swept pole arity, "
                     "and is the no-absorber ADE slot a predicate widening over "
                     "the SHIPPED update_P kernel rather than a new one?"),
        "emitted_corpus_digest": dispersive.corpus_digest(),
        "arities_swept": [list(a) for a in ARITIES],
        "pole_count_cap": dispersive.POLE_COUNT_CAP,
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        # THE OBSERVER GOES IN BEFORE THE POLICY: under 'keep' the policy's strip
        # wraps this, so it records the option tuple NVRTC was really given.
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

    for guard, options, _primary in GUARD_SETS:
        if args.backend == "numpy" and guard != "fmad_false":
            continue  # there is no compiler on this leg to guard
        if args.backend == "cuda":
            dispersive._COMPILE_OPTIONS = tuple(options)
            dispersive._clear_kernel_cache()
            from meep_gpu.cuda_kernels import ade_kernels  # noqa: PLC0415
            ade_kernels._COMPILE_OPTIONS = tuple(options)
            ade_kernels._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        dispersive._COMPILE_OPTIONS = ("--fmad=false",)
        dispersive._clear_kernel_cache()
        from meep_gpu.cuda_kernels import ade_kernels  # noqa: PLC0415
        ade_kernels._COMPILE_OPTIONS = ("--fmad=false",)
        ade_kernels._clear_kernel_cache()

    if not args.skip_mutations:
        run_host_mutations(results, args.out, args.backend, args.product)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.product)
        else:
            run_numpy_defects(results, args.out, args.product)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["verdict_flips_against_planted_defect"] = \
        verdict_flips_against_planted_defect(results)
    if not results["verdict_flips_against_planted_defect"]["all_flipped"]:
        results["summary"]["released"] = False
        results["summary"]["reasons"].append(
            "the release verdict did NOT flip against every applicable planted "
            "defect; a verdict that cannot go red is not a verdict")
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']} "
        f"by_arm={verdict['scored_by_arm']} "
        f"single_identical={verdict['single_launch_identical']} "
        f"multi_identical={verdict['multi_step_identical']}/{verdict['multi_step_cases']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
