"""Sub-step byte-identity gate for the hand-CUDA BFAST (``grid.bfast_active``) family.

THE QUESTION, in three parts, each measured separately:

1. **The curl, INCLUDING THE STATE.** Is ``cuda_kernels/bfast_curl.py``'s pair
   byte-identical, as uint32 words, to ``stepping.step_B`` / ``step_D`` on a grid
   whose ``bfast_scaled_k`` is nonzero -- at one launch and at 60, at an exactly
   representable courant and at one that is not, on physical-band and
   subnormal-band operands, under both float32 subnormal policies? The ``f_bfast``
   IIR state is compared as an OUTPUT on every case, beside the field and the PML
   auxiliary, because its homogeneous mode ``(-1)^n`` is UNDAMPED forever
   (stepping.py:872-877): a kernel that gets the field right and the state wrong is
   correct for exactly one launch and wrong from the second.
2. **The constitutive sides.** ``covers_bfast_constitutive`` admits the CERTIFIED
   constitutive pair on a BFAST run on the reading that ``update_H`` / ``update_E``
   read nothing BFAST-dependent. A reading is not evidence: the constitutive arm
   runs the certified kernels against the array path on a BFAST grid whose two
   curls have ALREADY advanced the state, per side, scored SEPARATELY.
3. **Is the BFAST term doing anything at all?** Two control runs of the ORACLE from
   the same frozen state:
   * ``bfast_off`` -- the term skipped entirely (``grid.bfast_active`` False, which
     is the only switch ``stepping`` reads at :364/:446);
   * ``k_zero`` -- ``bfast_scaled_k`` forced to (0,0,0) with the term STILL taken,
     so ``total`` is exactly zero and only the ``-2*F_prev`` half survives.
   The reference must differ from BOTH. The second control is what makes the k
   assignment load-bearing rather than the recurrence alone: a fixture that
   differed from ``bfast_off`` but not from ``k_zero`` would be measuring the IIR
   decay and nothing about ``k x E``.

THE FOLD IS NOT SWEPT AND THE PREDICATE REFUSES IT. The certified curl pair admits
a mirror fold since 2026-08-20 on a device verdict of its own, but that verdict
rests on "the folded ghost VALUES are dead because their consumer planes are
masked" -- an argument about a stencil that DIFFERENCES its ghosts. BFAST SUMS the
same operands, so the argument does not transfer, and no corpus row pairs BFAST
with a fold. Refusing costs nothing and is recorded in ``does_not_claim``.

WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
----------------------------------------------------------------
* ``oracle_moved`` over field, auxiliary AND state.
* ``bfast_term_is_live`` -- the two control runs above.
* ``state_starts_nonzero`` -- an ``f_bfast`` seeded to zero makes ``advance`` equal
  ``total`` at the first launch, which hides the ``2.0f * bprev`` term entirely.
  Measured per case rather than assumed from the seeding code.
* ``absorber_is_not_the_identity``.
* The mutation battery, including legs that MUST BE UNCAUGHT: ``curl + (-advance)``
  for ``curl - advance`` (IEEE-754 defines subtraction as addition of the negation)
  and ``(f1 + sf)`` for ``(sf + f1)`` (float add commutes exactly). Each is paired
  with a must-catch leg on the same expression.

RUNNING IT
----------
Device (the GPU host, ONE verified-empty GPU)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_bfast.py --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA), same oracle, same fixture, same floors, same HOST mutations,
against a transcription of the device tree; it compiles nothing and certifies
nothing::

    python -u gate_cuda_bfast.py --backend numpy --out /tmp/bfast.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402
from meep_gpu.cuda_kernels import bfast_curl as family  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260820
MULTI_STEP_BUDGET = 60
_ADVANCE = np.float32(0.97)

BC_PERIODIC = 0
BC_METALLIC = 1
BC_MIRROR_PERIODIC = 2

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")
SIDES: Tuple[str, ...] = ("H", "E")

SUB_STEP_ARRAYS: Dict[str, Dict[str, Any]] = {
    "step_B": {"targets": ("Bx", "By", "Bz"), "aux": ("fu_Bx", "fu_By", "fu_Bz"),
               "state": ("f_bfast_Bx", "f_bfast_By", "f_bfast_Bz"),
               "sources": ("Ex", "Ey", "Ez"), "half_integer": True,
               "backward": False},
    "step_D": {"targets": ("Dx", "Dy", "Dz"), "aux": ("fu_Dx", "fu_Dy", "fu_Dz"),
               "state": ("f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"),
               "sources": ("Hx", "Hy", "Hz"), "half_integer": False,
               "backward": True},
}

CONSTITUTIVE_ARRAYS: Dict[str, Dict[str, Any]] = {
    "H": {"targets": ("Hx", "Hy", "Hz"), "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz")},
    "E": {"targets": ("Ex", "Ey", "Ez"), "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz")},
}

#: ``(target, aux, state, first, first_axis, second, second_axis, dsig, dsigu)``
#: as the SHIPPED KERNEL spells it -- a transcription of the DEVICE source, which
#: the NumPy leg consumes, pinned against ``stepping`` by
#: :func:`check_terms_against_stepping`.
CURL_TERMS: Dict[str, Tuple[Tuple[Any, ...], ...]] = {
    "step_B": (
        ("Bx", "fu_Bx", "f_bfast_Bx", "Ez", 1, "Ey", 2, 1, 2),
        ("By", "fu_By", "f_bfast_By", "Ex", 2, "Ez", 0, 2, 0),
        ("Bz", "fu_Bz", "f_bfast_Bz", "Ey", 0, "Ex", 1, 0, 1),
    ),
    "step_D": (
        ("Dx", "fu_Dx", "f_bfast_Dx", "Hz", 1, "Hy", 2, 1, 2),
        ("Dy", "fu_Dy", "f_bfast_Dy", "Hx", 2, "Hz", 0, 2, 0),
        ("Dz", "fu_Dz", "f_bfast_Dz", "Hy", 0, "Hx", 1, 0, 1),
    ),
}

#: Every spec is UNFOLDED: the predicate refuses a fold by name (see the module
#: docstring). ``k`` is drawn three ways -- the corpus's single-axis vector, a
#: general one that puts a live term on every component, and a negative one.
BFAST_SPECS: Tuple[Dict[str, Any], ...] = (
    # The corpus row's own shape and k: one axis only, so Bx's pair is (0, 0) and
    # only By and Bz carry a k term. Sweeping it alone would leave the third
    # component's term unmeasured, which is why the general vectors are below.
    {"label": "corpus_single_axis_k", "k": (0.8169576958985646, 0.0, 0.0),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "general_k_periodic", "k": (0.81, -0.37, 0.52),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "general_k_metallic_x", "k": (0.81, -0.37, 0.52),
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "general_k_metallic_xyz", "k": (0.81, -0.37, 0.52),
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "negative_k_periodic", "k": (-0.63, 0.44, -0.21),
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (10.0, 8.0, 14.0)},
    # A THIN, INVARIANT AXIS. ``have_p``/``have_m`` are ``figure_out_step_plan``'s
    # flags and they zero a k on an axis the grid does not resolve
    # (stepping.py:909-912). The curl would swallow that by itself; the BFAST SUM
    # would not, which is the whole reason the gate carries this spec.
    {"label": "invariant_z_axis", "k": (0.81, -0.37, 0.52), "dimensions": 2,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 0.0)},
)

COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)


# ---------------------------------------------------------------------------
# THE FAMILY'S OWN SOURCE MUTATIONS
# ---------------------------------------------------------------------------

def _replace_all(source: str, pairs: Sequence[Tuple[str, str]]) -> Tuple[str, int]:
    sites = 0
    out = source
    for old, new in pairs:
        sites += out.count(old)
        out = out.replace(old, new)
    return out, sites


_TOTAL_LINES = tuple(
    f"        float total = (k1_{s} * (sf + f1)) - (k2_{s} * (ss + f2));\n"
    for s in ("a", "b", "c"))


def _drop_bfast_term(source: str) -> Tuple[str, int]:
    """The whole feature deleted: the certified kernel, on a BFAST run.

    The state store is kept (writing back what it read) so the mutation is a
    DROPPED TERM rather than a dropped array, which is the defect the predicate's
    inventory clause cannot see.
    """
    out, sites = _replace_all(source, [
        ("        curl = curl - advance;\n", ""),
    ])
    out, more = _replace_all(out, [
        ("        float advance = total - (2.0f * bprev);\n",
         "        float advance = 0.0f;\n"),
    ])
    return out, sites + more


def _bfast_difference_not_sum(source: str) -> Tuple[str, int]:
    """THE defect this family exists to be proof against.

    ``step_bfast`` SUMS the two samples where ``step_curl`` differences them
    (stepping.py:889-896). On an invariant axis the difference is an exact zero and
    the sum is ``2*g``, so the two are not close.
    """
    return _replace_all(source, [("(sf + f1)", "(sf - f1)"),
                                 ("(ss + f2)", "(ss - f2)")])


def _bfast_swap_k1_k2(source: str) -> Tuple[str, int]:
    """``k1`` and ``k2`` exchanged -- MEEP's "puts k1 in direction of g2" comment,
    read backwards (step_db.cpp:129-136). Silent: it merely moves the term to the
    wrong pair of components."""
    pairs = []
    for s in ("a", "b", "c"):
        pairs.append((f"(k1_{s} * (sf + f1)) - (k2_{s} * (ss + f2))",
                      f"(k2_{s} * (sf + f1)) - (k1_{s} * (ss + f2))"))
    return _replace_all(source, pairs)


def _bfast_added_not_subtracted(source: str) -> Tuple[str, int]:
    """``_bfast_term`` returns ``-advance`` (stepping.py:931) and the caller ADDS
    it (:396); a kernel that adds the unnegated advance has the whole feature at
    the wrong sign."""
    return _replace_all(source, [("curl = curl - advance;",
                                  "curl = curl + advance;")])


def _bfast_scaled_by_dtdx(source: str) -> Tuple[str, int]:
    """The term scaled by dtdx. MEEP passes ``dtdx`` to ``step_bfast`` and the body
    NEVER READS IT (stepping.py:863-865); the Tustin filter already carries the
    ``dt``."""
    pairs = []
    for line in _TOTAL_LINES:
        pairs.append((line, line.replace("float total = ", "float total = dtdx * ")))
    return _replace_all(source, pairs)


def _bfast_two_becomes_one(source: str) -> Tuple[str, int]:
    """``F_new = S - F_prev`` gives an advance of ``S - 2*F_prev`` (stepping.py:922).
    Writing ``S - F_prev`` is the filter without its pole, which is stable rather
    than marginally stable -- a plausible, smooth, wrong answer."""
    return _replace_all(source, [("total - (2.0f * bprev)", "total - bprev")])


def _bfast_state_regrouped(source: str) -> Tuple[str, int]:
    """``F += advance`` regrouped as ``F = total - F``.

    Algebraically identical (``F + (S - 2F) = S - F``) and NOT identical in
    float32: the array path rounds ``S - 2F`` and then ``F + that``, two roundings
    of different magnitudes. Scored as MUST BE CAUGHT because the grouping is the
    thing being transcribed; a leg that came back UNCAUGHT would mean the gate
    cannot see a regrouping of the state advance and would be reported as such
    rather than passed over.
    """
    return _replace_all(source, [("= bprev + advance;", "= total - bprev;")])


def _bfast_advance_not_masked(source: str) -> Tuple[str, int]:
    """The ownership mask dropped from ``advance`` but kept on ``curl``.

    ``stepping`` masks ``advance`` at :902, BEFORE the state absorbs it at :903,
    because MEEP writes ``F`` only inside its owned-cell loop. Drop that and the
    field is right on launch one and the state is wrong forever. Only bites where a
    mask line fires, so it is scored on legs whose grid carries a mask code.
    """
    out = []
    sites = 0
    for line in source.splitlines(keepends=True):
        if "advance = 0.0f;" in line and " == BC_" in line:
            sites += 1
            continue
        out.append(line)
    return "".join(out), sites


def _bfast_state_read_after_store(source: str) -> Tuple[str, int]:
    """The aliasing trap: read the previous state from memory AFTER writing it.

    The array path holds ``state`` and computes ``advance`` from it before the
    in-place ``+=`` (stepping.py:922-930). A kernel that re-read the array would
    see its own store.
    """
    pairs = []
    for target in ("fb_Bx", "fb_By", "fb_Bz", "fb_Dx", "fb_Dy", "fb_Dz"):
        pairs.append((f"        {target}[idx] = bprev + advance;\n"
                      f"        curl = curl - advance;\n",
                      f"        {target}[idx] = bprev + advance;\n"
                      f"        curl = curl - ({target}[idx] - bprev);\n"))
    return _replace_all(source, pairs)


def _bfast_subtract_as_add_negative(source: str) -> Tuple[str, int]:
    """NULL. IEEE-754 defines ``a - b`` as ``a + (-b)``; the two spellings are the
    same bits on every input, signed zeros included."""
    return _replace_all(source, [("curl = curl - advance;",
                                  "curl = curl + (-advance);")])


def _bfast_sum_operands_commuted(source: str) -> Tuple[str, int]:
    """NULL. float32 addition commutes exactly. stepping.py:917-918 writes
    ``shifted + at``; this leg measures that the order is a transcription
    convention rather than an arithmetic fact."""
    return _replace_all(source, [("(sf + f1)", "(f1 + sf)"),
                                 ("(ss + f2)", "(f2 + ss)")])


BFAST_SOURCE_MUTATIONS: Dict[str, Any] = {
    "drop_bfast_term": _drop_bfast_term,
    "bfast_difference_not_sum": _bfast_difference_not_sum,
    "bfast_swap_k1_k2": _bfast_swap_k1_k2,
    "bfast_added_not_subtracted": _bfast_added_not_subtracted,
    "bfast_scaled_by_dtdx": _bfast_scaled_by_dtdx,
    "bfast_two_becomes_one": _bfast_two_becomes_one,
    "bfast_state_regrouped": _bfast_state_regrouped,
    "bfast_advance_not_masked": _bfast_advance_not_masked,
    "bfast_state_read_after_store": _bfast_state_read_after_store,
    "bfast_subtract_as_add_negative": _bfast_subtract_as_add_negative,
    "bfast_sum_operands_commuted": _bfast_sum_operands_commuted,
}
BFAST_NULL_MUTATIONS: Tuple[str, ...] = ("bfast_subtract_as_add_negative",
                                         "bfast_sum_operands_commuted")
SOURCE_MUTATION_REQUIRES_MASK: Dict[str, bool] = {"bfast_advance_not_masked": True}

SHARED_SOURCE_MUTATIONS: Tuple[str, ...] = (
    "regroup_stencil", "drop_metallic_mask", "swap_dsig_dsigu", "drop_fu_store",
    "read_fprev_after_store", "fortran_order_index_decomposition",
    "commute_dtdx_scale", "reload_fu_from_memory",
)
SHARED_NULL_MUTATIONS: Tuple[str, ...] = probe.PML_NULL_MUTATIONS
SHARED_REQUIRES_CODE: Dict[str, int] = {"drop_metallic_mask": BC_METALLIC}

HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_k1_k2_scalars",
    "zero_k_scalars",
    "negate_k_scalars",
    "k_indexed_by_derivative_axis",
    "swap_curl_sublattice",
    "k_scalars_rounded_again",
)
#: MUST BE UNCAUGHT: rounding an already-float32 value to float32 is the identity.
HOST_NULL_MUTATIONS: Tuple[str, ...] = ("k_scalars_rounded_again",)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float):
    extra = ({} if spec.get("dimensions") is None
             else {"dimensions": int(spec["dimensions"])})
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), xp=xp, courant=courant,
                bfast_scaled_k=tuple(spec["k"]), **extra)
    thickness = tuple((0, 0) if grid.shape[axis] < 6 else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def outputs(sub_step: str) -> Tuple[str, ...]:
    spec = SUB_STEP_ARRAYS[sub_step]
    return tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["state"])


def state_names(sub_step: str) -> Tuple[str, ...]:
    return outputs(sub_step) + tuple(SUB_STEP_ARRAYS[sub_step]["sources"])


def constitutive_state_names(side: str) -> Tuple[str, ...]:
    spec = CONSTITUTIVE_ARRAYS[side]
    return tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])


def constitutive_outputs(side: str) -> Tuple[str, ...]:
    spec = CONSTITUTIVE_ARRAYS[side]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def seed_state(fields, grid, names: Sequence[str], value_class: str,
               rng) -> Dict[str, np.ndarray]:
    xp = grid.xp
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
                for name in names}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(tuple(names), tuple(grid.shape), rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def snapshot(fields, names: Sequence[str]) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in names}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, names: Sequence[str]) -> None:
    for name in names:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# Codes, tables and coefficients
# ---------------------------------------------------------------------------

def boundary_codes_for(grid):
    kinds = tuple(coverage.real_pml_boundary_kinds(grid))
    codes, refusal = coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(f"the shipped resolver refuses this grid: {refusal}")
    return tuple(int(code) for code in codes), kinds


def tables_for(sub_step: str, layer) -> Dict[str, Any]:
    suffix = "_h" if SUB_STEP_ARRAYS[sub_step]["half_integer"] else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def wrong_sub_lattice_tables(sub_step: str, layer) -> Dict[str, Any]:
    suffix = "" if SUB_STEP_ARRAYS[sub_step]["half_integer"] else "_h"
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def _k_indexed_by_derivative_axis(grid, sub_step: str):
    """``bfast_scaled_k`` indexed by the DERIVATIVE direction instead of the
    partner's own direction -- MEEP's ``component_index`` read wrongly
    (vec.hpp:445), the mistake stepping.py:814-823 calls the single easiest in the
    whole pass. Host-side, because that is where the assignment lives."""
    k = tuple(float(v) for v in grid.bfast_scaled_k)
    magnetic = sub_step == "step_B"
    out = []
    for _t, _a, _s, _first, first_axis, _second, second_axis, _dsig, _dsigu in \
            CURL_TERMS[sub_step]:
        k1 = k[second_axis] if not bool(grid.is_invariant(second_axis)) else 0.0
        k2 = k[first_axis] if not bool(grid.is_invariant(first_axis)) else 0.0
        if not magnetic:
            k1, k2 = -k1, -k2
        out.append((float(np.float32(k1)), float(np.float32(k2))))
    return tuple(out)


def host_mutated_inputs(name: Optional[str], sub_step: str, layer, grid,
                        dtdx: float):
    tables = tables_for(sub_step, layer)
    coefficients = family.bfast_curl_coefficients(grid, sub_step)
    if name is None:
        return tables, coefficients
    if name == "swap_curl_sublattice":
        return wrong_sub_lattice_tables(sub_step, layer), coefficients
    if name == "swap_k1_k2_scalars":
        return tables, tuple((k2, k1) for k1, k2 in coefficients)
    if name == "zero_k_scalars":
        return tables, tuple((0.0, 0.0) for _ in coefficients)
    if name == "negate_k_scalars":
        return tables, tuple((-k1, -k2) for k1, k2 in coefficients)
    if name == "k_indexed_by_derivative_axis":
        return tables, _k_indexed_by_derivative_axis(grid, sub_step)
    if name == "k_scalars_rounded_again":
        return tables, tuple((float(np.float32(k1)), float(np.float32(k2)))
                             for k1, k2 in coefficients)
    raise KeyError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

def run_kernel_cuda(sub_step: str, fields, tables, codes, dtdx, coefficients):
    family.step_bfast(sub_step, fields, tables,
                      tuple(np.int32(c) for c in codes), dtdx, coefficients)
    cp.cuda.runtime.deviceSynchronize()


def _plane(axis: int, index: int, shape) -> Tuple[Any, ...]:
    key: List[Any] = [slice(None)] * len(shape)
    key[axis] = index
    return tuple(key)


def _shift_up_numpy(field, axis, bc):
    shifted = np.roll(field, -1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane(axis, -1, field.shape)] = np.float32(0.0)
    return shifted


def _shift_dn_numpy(field, axis, bc):
    shifted = np.roll(field, 1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane(axis, 0, field.shape)] = np.float32(0.0)
    return shifted


def _broadcast(vector, axis):
    shape = [1, 1, 1]
    shape[axis] = int(np.asarray(vector).size)
    return np.asarray(vector, dtype=np.float32).reshape(shape)


def run_kernel_numpy(sub_step: str, fields, tables, codes, dtdx, coefficients):
    """The device tree, transcribed, in float32 -- the laptop backend.

    THE GROUPING IS THE SHIPPED ONE and it is load-bearing::

        float curl    = dtdx * ((sf - f1) + (f2 - ss));
        float total   = (k1 * (sf + f1)) - (k2 * (ss + f2));
        float bprev   = fb[idx];
        float advance = total - (2.0f * bprev);
        <mask lines on advance>
        fb[idx]       = bprev + advance;
        curl          = curl - advance;
        <mask lines on curl>
        pml_apply(...)

    COMPILES NOTHING AND CERTIFIES NOTHING.
    """
    spec = SUB_STEP_ARRAYS[sub_step]
    shift = _shift_dn_numpy if spec["backward"] else _shift_up_numpy
    scale = np.float32(dtdx)
    two = np.float32(2.0)
    names = "xyz"
    for term, (k1, k2) in zip(CURL_TERMS[sub_step], coefficients):
        (target, aux, state, first, first_axis, second, second_axis,
         dsig, dsigu) = term
        f = getattr(fields, target)
        fu = getattr(fields, aux)
        fb = getattr(fields, state)
        f1 = getattr(fields, first)
        f2 = getattr(fields, second)
        sf = shift(f1, first_axis, codes[first_axis])
        ss = shift(f2, second_axis, codes[second_axis])
        curl = (scale * ((sf - f1) + (f2 - ss))).astype(np.float32)
        total = ((np.float32(k1) * (sf + f1)).astype(np.float32)
                 - (np.float32(k2) * (ss + f2)).astype(np.float32)).astype(np.float32)
        bprev = fb.copy()
        advance = (total - (two * bprev).astype(np.float32)).astype(np.float32)
        iyee = IYEE_SHIFTS[target]

        def mask(volume):
            for axis in range(3):
                if iyee[axis] == 0:
                    if codes[axis] in (BC_METALLIC, BC_MIRROR_PERIODIC):
                        volume[_plane(axis, 0, volume.shape)] = np.float32(0.0)
                elif codes[axis] == BC_MIRROR_PERIODIC:
                    volume[_plane(axis, -1, volume.shape)] = np.float32(0.0)

        mask(advance)
        fb[...] = (bprev + advance).astype(np.float32)
        curl = (curl - advance).astype(np.float32)
        mask(curl)
        kms = _broadcast(tables[f"kms_{names[dsig]}"], dsig)
        sinv = _broadcast(tables[f"sinv_{names[dsig]}"], dsig)
        kms_u = _broadcast(tables[f"kms_{names[dsigu]}"], dsigu)
        sinv_u = _broadcast(tables[f"sinv_{names[dsigu]}"], dsigu)
        fprev = fu.copy()
        fu_new = (((fprev * kms).astype(np.float32) - curl).astype(np.float32)
                  * sinv).astype(np.float32)
        fu[...] = fu_new
        a = ((f * kms_u).astype(np.float32) + fu_new).astype(np.float32)
        f[...] = ((a - fprev).astype(np.float32) * sinv_u).astype(np.float32)


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def moved_fraction(before, after, names) -> float:
    moved = total = 0
    for name in names:
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def differing_words(a, b, name) -> int:
    x = np.ascontiguousarray(a[name], dtype=np.float32).ravel().view(np.uint32)
    y = np.ascontiguousarray(b[name], dtype=np.float32).ravel().view(np.uint32)
    return int(np.count_nonzero(x != y))


def absorber_is_not_the_identity(sub_step: str, layer) -> Dict[str, Any]:
    worst = 0.0
    for _key, vector in tables_for(sub_step, layer).items():
        values = to_host(vector).astype(np.float64)
        worst = max(worst, float(np.max(np.abs(values - 1.0))))
    return {"max_deviation_from_identity": worst, "meets_floor": worst > 0.0}


class _BfastOffGrid:
    """The grid with ``bfast_active`` False -- the term skipped entirely.

    ``stepping`` reads it once per sub-step (:336 / :422) and takes the term only
    ``if bfast`` (:364 / :446), so a run through this proxy is the SAME sub-step
    with the whole feature deleted and nothing else changed.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid) -> None:
        object.__setattr__(self, "_grid", grid)

    @property
    def bfast_active(self) -> bool:
        return False

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_grid"), item)


class _ZeroKGrid:
    """The grid with ``bfast_scaled_k`` all zero but the term STILL TAKEN.

    ``bfast_active`` is forced True so ``stepping`` enters ``_bfast_term``; the
    zero vector makes ``total`` exactly zero and leaves only ``-2*F_prev``. The
    difference between this and the real run is what the k assignment is worth,
    isolated from the IIR recurrence.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid) -> None:
        object.__setattr__(self, "_grid", grid)

    @property
    def bfast_active(self) -> bool:
        return True

    @property
    def bfast_scaled_k(self):
        return (0.0, 0.0, 0.0)

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_grid"), item)


def _oracle_through(fields, layer, grid, sub_step, frozen, proxy_class):
    saved = fields.grid
    try:
        object.__setattr__(fields, "grid", proxy_class(saved))
        restore(fields, frozen)
        if sub_step == "step_B":
            stepping.step_B(fields, layer)
        else:
            stepping.step_D(fields, layer)
        return snapshot(fields, state_names(sub_step))
    finally:
        object.__setattr__(fields, "grid", saved)


def bfast_term_is_live(fields, layer, grid, sub_step, frozen,
                       reference) -> Dict[str, Any]:
    """The two control runs, and what each of them proves."""
    off = _oracle_through(fields, layer, grid, sub_step, frozen, _BfastOffGrid)
    zero_k = _oracle_through(fields, layer, grid, sub_step, frozen, _ZeroKGrid)
    vs_off = {name: differing_words(reference, off, name)
              for name in outputs(sub_step)}
    vs_zero_k = {name: differing_words(reference, zero_k, name)
                 for name in outputs(sub_step)}
    return {
        "differing_words_vs_bfast_off": vs_off,
        "differing_words_vs_zero_k": vs_zero_k,
        "total_vs_bfast_off": sum(vs_off.values()),
        "total_vs_zero_k": sum(vs_zero_k.values()),
        "meets_floor": sum(vs_off.values()) > 0 and sum(vs_zero_k.values()) > 0,
    }


def state_starts_nonzero(frozen, sub_step) -> Dict[str, Any]:
    """An ``f_bfast`` seeded to zero hides ``2.0f * bprev`` at the first launch."""
    counts = {}
    for name in SUB_STEP_ARRAYS[sub_step]["state"]:
        values = np.ascontiguousarray(frozen[name], dtype=np.float32).ravel()
        counts[name] = int(np.count_nonzero(values.view(np.uint32)))
    return {"nonzero_words": counts, "meets_floor": all(v > 0 for v in counts.values())}


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "bfast_scaled_k": [float(v) for v in grid.bfast_scaled_k],
        "invariant": [bool(grid.is_invariant(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
    }


def check_terms_against_stepping() -> Dict[str, Any]:
    """Pin :data:`CURL_TERMS` and the k assignment against ``stepping``'s own.

    The stencil half is the certified gate's check. The k half re-derives
    ``stepping._bfast_term``'s ``k1``/``k2`` from ``stepping``'s OWN term tables and
    ``stepping._bfast_axis``, and compares them against what
    ``family.bfast_curl_coefficients`` returns for the same grid. Two
    transcriptions of the same six lines, compared -- because the version this gate
    consumes and the version the kernel is launched with are different files.
    """
    out: Dict[str, Any] = {"agreed": True, "checked": [], "disagreements": []}
    axis_of = {"x": 0, "y": 1, "z": 2}
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        for mine, theirs in zip(CURL_TERMS[sub_step], source):
            record = {
                "sub_step": sub_step, "target": mine[0],
                "mine": [mine[0], mine[3], mine[4], mine[5], mine[6], mine[7], mine[8]],
                "stepping": [theirs.target, theirs.first, theirs.first_axis,
                             theirs.second, theirs.second_axis,
                             axis_of[theirs.dsig], axis_of[theirs.dsigu]],
            }
            record["agrees"] = record["mine"] == record["stepping"]
            out["checked"].append(record)
            if not record["agrees"]:
                out["agreed"] = False
                out["disagreements"].append(record)

    # The k assignment, on a grid with a thin invariant axis so the have_p/have_m
    # gates are exercised rather than merely present.
    _f, _l, grid = build(_NumpyWearingCupysName(),
                         {"label": "probe", "k": (0.81, -0.37, 0.52),
                          "dimensions": 2, "boundaries": ("periodic",) * 3,
                          "cell": (8.0, 10.0, 0.0)}, 0.5)
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        mine = family.bfast_curl_coefficients(grid, sub_step)
        magnetic = sub_step == "step_B"
        for index, term in enumerate(source):
            have_p = not bool(grid.is_invariant(term.first_axis))
            have_m = not bool(grid.is_invariant(term.second_axis))
            k = grid.bfast_scaled_k
            k1 = k[stepping._bfast_axis(term.second)] if have_m else 0.0
            k2 = k[stepping._bfast_axis(term.first)] if have_p else 0.0
            if not magnetic:
                k1, k2 = -k1, -k2
            theirs = (float(np.float32(k1)), float(np.float32(k2)))
            record = {"sub_step": sub_step, "target": term.target,
                      "mine": list(mine[index]), "stepping": list(theirs),
                      "agrees": mine[index] == theirs, "kind": "k_assignment"}
            out["checked"].append(record)
            if not record["agrees"]:
                out["agreed"] = False
                out["disagreements"].append(record)
    return out


# ---------------------------------------------------------------------------
# One curl case
# ---------------------------------------------------------------------------

def one_case(backend, spec, sub_step, courant, value_class, guard,
             host_mutation=None) -> Dict[str, Any]:
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{sub_step}|{courant}|{value_class}".encode()).digest()[:4], "big"))

    fields, layer, grid = build(xp, spec, courant)
    seed_state(fields, grid, state_names(sub_step), value_class, rng)
    dtdx = float(grid.dt / grid.dx)
    codes, kinds = boundary_codes_for(grid)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend,
        "host_mutation": host_mutation, "boundaries": list(spec["boundaries"]),
        "boundary_codes": [int(c) for c in codes], "resolved_kinds": list(kinds),
        "dtdx": dtdx, "structure": structure_facts(grid),
    }
    covered, reason = family.covers_bfast_curl(fields, layer, grid, sub_step)
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    certified, certified_reason = coverage.covers_real_pml_curl(
        fields, layer, grid, sub_step)
    case["certified_predicate"] = {"covered": bool(certified),
                                   "reason": certified_reason}

    absorbs = absorber_is_not_the_identity(sub_step, layer)
    case["absorber_is_not_the_identity"] = absorbs
    if not absorbs["meets_floor"]:
        case["skipped"] = "every absorber coefficient is the identity"
        case["seconds"] = time.time() - started
        return case

    frozen = snapshot(fields, state_names(sub_step))
    case["operand_census"] = operand_census(frozen)
    nonzero = state_starts_nonzero(frozen, sub_step)
    case["state_starts_nonzero"] = nonzero
    if not nonzero["meets_floor"]:
        case["skipped"] = ("an f_bfast array is all zero: the 2.0f * bprev term "
                           "is invisible at the first launch")
        case["seconds"] = time.time() - started
        return case

    if sub_step == "step_B":
        stepping.step_B(fields, layer)
    else:
        stepping.step_D(fields, layer)
    reference = snapshot(fields, state_names(sub_step))

    moved = moved_fraction(frozen, reference, outputs(sub_step))
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = "the array path changed no output word"
        case["seconds"] = time.time() - started
        return case

    live = bfast_term_is_live(fields, layer, grid, sub_step, frozen, reference)
    case["bfast_term_is_live"] = live
    if not live["meets_floor"]:
        case["skipped"] = ("the BFAST term, or its k assignment, moved no output "
                           "word: this case measures the certified kernel")
        case["seconds"] = time.time() - started
        return case

    restore(fields, frozen)
    tables, coefficients = host_mutated_inputs(host_mutation, sub_step, layer,
                                               grid, dtdx)
    case["bfast_coefficients"] = [[float(a), float(b)] for a, b in coefficients]
    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(sub_step, fields, tables, codes, dtdx, coefficients)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(sub_step)}
    case["single_launch"] = combine(parts)
    case["single_launch_per_array"] = {
        name: parts[name]["differing_floats"] for name in outputs(sub_step)}

    if host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            if sub_step == "step_B":
                stepping.step_B(fields, layer)
            else:
                stepping.step_D(fields, layer)
            advance_sources(fields, SUB_STEP_ARRAYS[sub_step]["sources"])
        oracle_multi = snapshot(fields, state_names(sub_step))

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(sub_step, fields, tables, codes, dtdx, coefficients)
            advance_sources(fields, SUB_STEP_ARRAYS[sub_step]["sources"])
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs(sub_step)}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The constitutive arm
# ---------------------------------------------------------------------------

def one_constitutive_case(backend, spec, side, courant, value_class, guard):
    started = time.time()
    if backend != "cuda":
        return {"label": spec["label"], "side": side, "courant": courant,
                "value_class": value_class, "backend": backend,
                "skipped": ("the constitutive arm runs the CERTIFIED CUDA kernel; "
                            "there is no NumPy transcription of it in this gate")}
    from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415

    rng = np.random.default_rng(
        SEED + 613 + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{side}|{courant}|{value_class}".encode()).digest()[:4], "big"))
    fields, layer, grid = build(cp, spec, courant)
    every = tuple(dict.fromkeys(
        state_names("step_B") + state_names("step_D")
        + constitutive_state_names("H") + constitutive_state_names("E")))
    seed_state(fields, grid, every, value_class, rng)

    case: Dict[str, Any] = {
        "label": spec["label"], "side": side, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend,
        "structure": structure_facts(grid),
    }
    covered, reason = family.covers_bfast_constitutive(fields, layer, grid, side)
    case["predicate"] = {"covered": bool(covered), "reason": reason}

    # THE BFAST RUN'S OWN HISTORY: both curls, on the array path, so the state and
    # the fields this sub-step reads are what BFAST produced.
    stepping.step_B(fields, layer)
    stepping.step_D(fields, layer)

    frozen = snapshot(fields, constitutive_state_names(side))
    if side == "H":
        stepping.update_H(fields, layer)
    else:
        stepping.update_E(fields, layer)
    reference = snapshot(fields, constitutive_state_names(side))
    moved = moved_fraction(frozen, reference, constitutive_outputs(side))
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = "the array path changed no output word"
        case["seconds"] = time.time() - started
        return case

    restore(fields, frozen)
    constitutive_kernels.update_fused_pml_real(side, fields, layer)
    cp.cuda.runtime.deviceSynchronize()
    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in constitutive_outputs(side)}
    case["single_launch"] = combine(parts)

    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        if side == "H":
            stepping.update_H(fields, layer)
        else:
            stepping.update_E(fields, layer)
        advance_sources(fields, CONSTITUTIVE_ARRAYS[side]["sources"])
    oracle_multi = snapshot(fields, constitutive_state_names(side))
    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        constitutive_kernels.update_fused_pml_real(side, fields, layer)
        advance_sources(fields, CONSTITUTIVE_ARRAYS[side]["sources"])
    cp.cuda.runtime.deviceSynchronize()
    multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
             for name in constitutive_outputs(side)}
    case["multi_step"] = combine(multi)
    case["multi_step"]["launches"] = MULTI_STEP_BUDGET
    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweeps
# ---------------------------------------------------------------------------

def case_product(product: str):
    specs = BFAST_SPECS if product == "full" else BFAST_SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    return [(spec, sub_step, courant, value_class)
            for spec in specs for sub_step in SUB_STEPS
            for courant in courants for value_class in classes]


def run_sweep(results, out_path, backend, product, guard):
    cases = []
    plan = case_product(product)
    for index, (spec, sub_step, courant, value_class) in enumerate(plan, start=1):
        case = one_case(backend, spec, sub_step, courant, value_class, guard)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"c={courant} {value_class} shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"diff={case['single_launch']['differing_floats']} "
            f"k_live={case['bfast_term_is_live']['total_vs_zero_k']} "
            f"({case['seconds']:.1f} s)")
    return cases


def run_constitutive(results, out_path, backend, product, guard):
    cases = []
    specs = BFAST_SPECS if product == "full" else BFAST_SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    plan = [(spec, side, courant, value_class)
            for spec in specs for side in SIDES
            for courant in courants for value_class in classes]
    for index, (spec, side, courant, value_class) in enumerate(plan, start=1):
        case = one_constitutive_case(backend, spec, side, courant, value_class, guard)
        cases.append(case)
        results.setdefault("constitutive", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[const {guard}] {index}/{len(plan)} {spec['label']} update_{side} "
                f"SKIPPED: {case['skipped'][:60]}")
            continue
        log(f"[const {guard}] {index}/{len(plan)} {spec['label']} update_{side} "
            f"c={courant} {value_class} "
            f"single={'IDENTICAL' if case['single_launch']['bit_identical'] else 'DIVERGED'} "
            f"multi={'IDENTICAL' if case['multi_step']['bit_identical'] else 'DIVERGED'} "
            f"diff={case['single_launch']['differing_floats']} "
            f"({case['seconds']:.1f} s)")
    return cases


MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "corpus_single_axis_k", "general_k_periodic", "general_k_metallic_x",
    "general_k_metallic_xyz", "negative_k_periodic", "invariant_z_axis",
)


def spec_codes(spec) -> Tuple[int, int, int]:
    _f, _l, grid = build(_NumpyWearingCupysName(), spec, COURANTS[0])
    codes, refusal = coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(f"the shipped resolver refuses fixture {spec['label']!r}: "
                         f"{refusal}")
    return tuple(int(c) for c in codes)


def scorable_baselines(results):
    baseline = {}
    for case in results.get("sweep", {}).get("fmad_false", []):
        if (case.get("courant") != INEXACT_COURANT
                or case.get("value_class") != "uniform"):
            continue
        baseline[(case["label"], case["sub_step"])] = case
    scorable, excluded = set(), []
    for key, case in baseline.items():
        if case.get("skipped"):
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": f"baseline skipped: {case['skipped'][:80]}"})
            continue
        if (case["single_launch"]["bit_identical"]
                and case.get("multi_step", {}).get("bit_identical", True)):
            scorable.add(key)
        else:
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": "the unmutated baseline already diverges here",
                             "differing_words":
                                 case["single_launch"]["differing_floats"]})
    return scorable, excluded


def mutation_plan(product, scorable=None):
    by_label = {spec["label"]: spec for spec in BFAST_SPECS}
    labels = MUTATION_SPEC_LABELS if product == "full" else MUTATION_SPEC_LABELS[:3]
    plan = [(by_label[label], sub_step, INEXACT_COURANT, "uniform")
            for label in labels for sub_step in SUB_STEPS]
    if scorable is None:
        return plan
    return [e for e in plan if (e[0]["label"], e[1]) in scorable]


def leg_caught(case) -> bool:
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def _score(legs, must_be_caught: bool) -> Dict[str, Any]:
    scored = [c for c in legs if not c.get("skipped")]
    caught = sum(1 for c in scored if leg_caught(c))
    if not scored:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = ("CAUGHT" if caught == len(scored)
                   else "PARTIAL" if caught else "UNCAUGHT")
    else:
        verdict = "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"
    return {"ran": len(scored), "caught": caught, "uncaught": len(scored) - caught,
            "must_be_caught": must_be_caught, "verdict": verdict}


def run_host_mutations(results, out_path, backend, product, scorable):
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)
    for name in HOST_MUTATIONS:
        legs = [one_case(backend, spec, sub_step, courant, value_class,
                         "fmad_false", host_mutation=name)
                for spec, sub_step, courant, value_class in plan]
        out[name] = dict(_score(legs, name not in HOST_NULL_MUTATIONS), cases=legs)
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def _grid_carries_a_mask(codes) -> bool:
    return any(code in (BC_METALLIC, BC_MIRROR_PERIODIC) for code in codes)


def run_source_mutations(results, out_path, product, scorable):
    from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

    originals = {name: family.kernel_source(name) for name in family.BFAST_KERNELS}
    transforms = dict(BFAST_SOURCE_MUTATIONS)
    for name in SHARED_SOURCE_MUTATIONS:
        transforms[name] = probe.SOURCE_MUTATIONS[name]
    nulls = set(BFAST_NULL_MUTATIONS) | set(SHARED_NULL_MUTATIONS)
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)

    try:
        for sub_step in SUB_STEPS:
            kernel_name = family.KERNEL_FOR_SUB_STEP[sub_step]
            for name, transform in transforms.items():
                mutated, sites = transform(originals[kernel_name])
                key = f"{sub_step}:{name}"
                required = SHARED_REQUIRES_CODE.get(name)
                leg_plan = [e for e in plan if e[1] == sub_step]
                if required is not None:
                    leg_plan = [e for e in leg_plan if required in spec_codes(e[0])]
                if SOURCE_MUTATION_REQUIRES_MASK.get(name):
                    leg_plan = [e for e in leg_plan
                                if _grid_carries_a_mask(spec_codes(e[0]))]
                if sites == 0 or mutated == originals[kernel_name]:
                    out[key] = {"armed": False,
                                "why": (f"matched {sites} site(s) and changed "
                                        f"nothing; the mutation and the kernel "
                                        f"have drifted apart")}
                    log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                family.set_kernel_source(kernel_name, mutated)
                family._clear_kernel_cache()
                compile_cache.clear_compile_log()
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                legs = [one_case("cuda", spec, sub_step, courant, value_class,
                                 "fmad_false")
                        for spec, _s, courant, value_class in leg_plan]
                family.set_kernel_source(kernel_name, originals[kernel_name])
                family._clear_kernel_cache()
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] == digest)
                scored = _score(legs, name not in nulls)
                out[key] = dict(scored, armed=True, sites=sites,
                                requires_code=required,
                                legs_in_plan=len([e for e in plan
                                                  if e[1] == sub_step]),
                                legs_scored=len(leg_plan),
                                mutated_source_sha256=digest,
                                kernel_constructions_from_mutated_bytes=from_mutated,
                                cases=legs)
                if from_mutated == 0 and scored["ran"]:
                    out[key]["verdict"] = "UNACCOUNTED"
                    out[key]["why"] = ("no kernel construction used the mutated "
                                       "bytes; this leg did not exercise the "
                                       "mutation")
                log(f"[src-mut] {key}: caught {scored['caught']}/{scored['ran']} "
                    f"builds_from_mutated={from_mutated} -> {out[key]['verdict']}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for name in family.BFAST_KERNELS:
            family.set_kernel_source(name, originals[name])
        family._clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results) -> Dict[str, Any]:
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    reasons: List[str] = []

    arms: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        key = ("invariant_axis" if any(case["structure"]["invariant"])
               else "full_rank")
        entry = arms.setdefault(key, {"cases": 0, "single_identical": 0,
                                      "multi_cases": 0, "multi_identical": 0,
                                      "differing_words": 0, "sub_steps": set(),
                                      "labels": set()})
        entry["cases"] += 1
        entry["sub_steps"].add(case["sub_step"])
        entry["labels"].add(case["label"])
        entry["single_identical"] += bool(case["single_launch"]["bit_identical"])
        if "multi_step" in case:
            entry["multi_cases"] += 1
            entry["multi_identical"] += bool(case["multi_step"]["bit_identical"])
        entry["differing_words"] += case["single_launch"]["differing_floats"]
    for entry in arms.values():
        entry["sub_steps"] = sorted(entry["sub_steps"])
        entry["labels"] = sorted(entry["labels"])
        entry["all_identical"] = (entry["single_identical"] == entry["cases"]
                                  and entry["multi_identical"] == entry["multi_cases"])

    if not scored:
        reasons.append("no curl case was scored at all")
    for arm in ("full_rank", "invariant_axis"):
        if arm not in arms:
            reasons.append(f"the sweep contained no {arm} case; the have_p/have_m "
                           f"gates are only exercised on the second, and the k "
                           f"assignment only on the first")
            continue
        entry = arms[arm]
        if entry["single_identical"] != entry["cases"]:
            reasons.append(f"{arm}: single-launch divergence on "
                           f"{entry['cases'] - entry['single_identical']} of "
                           f"{entry['cases']} cases")
        if entry["multi_identical"] != entry["multi_cases"]:
            reasons.append(f"{arm}: multi-step divergence on "
                           f"{entry['multi_cases'] - entry['multi_identical']} of "
                           f"{entry['multi_cases']} cases")
        if sorted(entry["sub_steps"]) != sorted(SUB_STEPS):
            reasons.append(f"{arm}: only sub-steps {entry['sub_steps']} were scored")

    terms = results.get("curl_terms_vs_stepping", {})
    if not terms.get("agreed", False):
        reasons.append("the transcribed curl/k term table disagrees with stepping's")

    for case in scored:
        if not case["predicate"]["covered"]:
            reasons.append(f"{case['label']} {case['sub_step']}: the shipped "
                           f"predicate REFUSES a case this gate scored "
                           f"({case['predicate']['reason'][:80]})")
        if case["certified_predicate"]["covered"]:
            reasons.append(f"{case['label']} {case['sub_step']}: the CERTIFIED "
                           f"predicate also admits this case; the two families are "
                           f"meant to partition on bfast_active")

    constitutive = results.get("constitutive", {}).get("fmad_false", [])
    per_side: Dict[str, Dict[str, int]] = {}
    for case in constitutive:
        entry = per_side.setdefault(case["side"], {"cases": 0, "single": 0,
                                                   "multi_cases": 0, "multi": 0,
                                                   "skipped": 0})
        if case.get("skipped"):
            entry["skipped"] += 1
            continue
        entry["cases"] += 1
        entry["single"] += bool(case["single_launch"]["bit_identical"])
        entry["multi_cases"] += 1
        entry["multi"] += bool(case["multi_step"]["bit_identical"])
    for side in SIDES:
        entry = per_side.get(side)
        if entry is None or entry["cases"] == 0:
            reasons.append(f"update_{side} was never scored on a BFAST run; the "
                           f"constitutive admission would rest on the reading "
                           f"alone")
            continue
        if entry["single"] != entry["cases"] or entry["multi"] != entry["multi_cases"]:
            reasons.append(f"update_{side}: divergence on "
                           f"{entry['cases'] - entry['single']} of "
                           f"{entry['cases']} single-launch and "
                           f"{entry['multi_cases'] - entry['multi']} of "
                           f"{entry['multi_cases']} multi-step cases")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is "
                           f"not inert")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']}")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    guarded_identical = {(c["label"], c["sub_step"], c["courant"], c["value_class"])
                         for c in scored if c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["courant"], c["value_class"])
                  in guarded_identical]
    control_diverged = [c for c in comparable
                        if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run" if not comparable else
                    "the contraction guard is load-bearing on this pair"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too"),
    }

    return {
        "released": not reasons,
        "reasons": reasons,
        "arms": arms,
        "constitutive_per_side": per_side,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "claim": ("on a real Cartesian grid with bfast_scaled_k != 0 and real "
                  "float32 storage under an active PML, cuda_kernels/bfast_curl.py's "
                  "pair is byte-identical to stepping.step_B / step_D per sub-step "
                  "-- FIELD, PML AUXILIARY AND f_bfast IIR STATE -- at one launch "
                  "and at 60, with and without an invariant axis, at an exactly "
                  "representable courant and at one that is not, on physical-band "
                  "and subnormal-band operands; and the CERTIFIED constitutive pair "
                  "is byte-identical to stepping.update_H / update_E on the same "
                  "BFAST runs, scored per side"),
        "does_not_claim": [
            "nothing dispatches these kernels; this gate licenses a predicate "
            "clause, not a wiring",
            "A MIRROR FOLD IS REFUSED BY NAME AND NOT MEASURED: the certified "
            "pair's fold verdict is about a stencil that differences its ghosts, "
            "and BFAST sums them",
            "complex storage with BFAST is refused (one COMPLEX state per "
            "component) and untested",
            "cylindrical BFAST is refused by Grid itself (grid.py:699+) and is "
            "not reachable",
            "a conductivity, beta, a Bloch phase and dispersion are refused by "
            "inherited clauses and untested here",
            "no throughput claim: this is a correctness gate and times nothing",
        ],
    }


def save(results, out_path: str) -> None:
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
        "gate": "cuda_bfast",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("is the hand-CUDA BFAST curl pair byte-identical to "
                     "stepping.step_B / step_D -- field, auxiliary AND f_bfast "
                     "state -- on a BFAST run, and is the CERTIFIED constitutive "
                     "pair byte-identical to update_H / update_E on the same run?"),
        "curl_terms_vs_stepping": check_terms_against_stepping(),
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

    for guard, options, _primary in GUARD_SETS:
        if args.backend == "numpy" and guard != "fmad_false":
            continue
        if args.backend == "cuda":
            from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
            family._COMPILE_OPTIONS = tuple(options)
            family._clear_kernel_cache()
            constitutive_kernels._COMPILE_OPTIONS = tuple(options)
            constitutive_kernels._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)
        run_constitutive(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        family._COMPILE_OPTIONS = ("--fmad=false",)
        family._clear_kernel_cache()
        constitutive_kernels._COMPILE_OPTIONS = ("--fmad=false",)
        constitutive_kernels._clear_kernel_cache()

    if not args.skip_mutations:
        scorable, excluded = scorable_baselines(results)
        results["mutation_scope"] = {
            "scorable": sorted("::".join(k) for k in scorable),
            "excluded": excluded,
            "why": ("a mutation leg answers 'can this gate see this defect', and "
                    "only an arm whose unmutated baseline is bit-identical can "
                    "answer it"),
        }
        log(f"[mut-scope] {len(scorable)} scorable pairs; {len(excluded)} excluded")
        save(results, args.out)
        run_host_mutations(results, args.out, args.backend, args.product, scorable)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.product, scorable)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']}")
    for name, arm in sorted(verdict["arms"].items()):
        log(f"[verdict]   {name}: single {arm['single_identical']}/{arm['cases']} "
            f"multi {arm['multi_identical']}/{arm['multi_cases']} "
            f"diff={arm['differing_words']}")
    for side, entry in sorted(verdict["constitutive_per_side"].items()):
        log(f"[verdict]   update_{side}: single {entry['single']}/{entry['cases']} "
            f"multi {entry['multi']}/{entry['multi_cases']} "
            f"skipped {entry['skipped']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
