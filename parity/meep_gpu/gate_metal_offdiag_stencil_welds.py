"""The device gate for the FOUR Metal off-diagonal STENCIL welds.

WHAT THIS CERTIFIES. Four products landed in one round, and every one of them
closes a cell both fusion boards have carried as ``STRUCTURALLY UNFUSABLE ON ANY
BACKEND`` for the whole campaign::

    offdiag_fused_electric_pair                 PML x offdiag              8 of 16
    folded_offdiag_fused_electric_pair          folded x folded offdiag    9 of 19
    complex_no_pml_offdiag_fused_electric_pair  cx no-PML x cx offdiag     1 of 2
    folded_complex_offdiag_fused_electric_pair  folded cx x cx folded od   1 of 3

(the second number is the cell's rows, the first the subset CLEARING the source
seam — see leg E). The verdict they close reads "the off-diagonal constitutive
arm is a STENCIL over the curl arm's own in-place D output, and no grid-wide
barrier exists inside one launch". The premise is the IN-PLACE part: these
kernels write ``D``/``fu_D`` to launch-local SCRATCH, re-derive every foreign tap
from PRE-LAUNCH state, and the launcher rotates the buffers after the launch
returns. Nothing written is ever read, so no barrier is wanted.

THE COMPARISON IS PER COMPLETE DRIVER STEP, AS uint32 WORDS, over every stored
volume a step can touch — never ``allclose``, never a single sub-step — and it is
taken against BOTH references the standing Metal format requires:

    A  identity          the ARRAY PATH, on this host, every step;
    B  certified singles the DEVICE composition of the very kernels this weld
                         REPLACES (``launch.plan_step(..., fuse=False)``), so the
                         claim is not merely "matches NumPy" but "matches the
                         certified Metal kernels it stands in for";
    C  mutation          one arithmetic/index/offset/rotation line of the SHIPPED
                         source or the SHIPPED launcher replaced and driven
                         through the plan builder's own ``functions`` door. Each
                         MUST diverge; each declared equivalence must NOT. The
                         rotation nulls are this design's own: rotation skipped,
                         scratch aliased to the live buffer, a foreign tap read
                         from the LIVE buffer instead of re-derived, the near
                         redirect row wrong, the far redirect dropped, the parity
                         dropped;
    D  ceiling           shipped signatures COMPILE at their declared binding
                         counts, every refuted signature is REFUSED, and the
                         pointer ceiling is BISECTED on this host;
    E  seam refusal      an in-seam ELECTRIC source must be REFUSED BY NAME by
                         every one of the four predicates. This is the leg that
                         replaces the deposit leg the repairable families carry:
                         ``deposit_repair.repairable`` refuses an off-diagonal
                         chi1inv, so ``CARRIES_DEPOSIT_REPAIR`` is False on all
                         four and a source in the seam is a REFUSAL rather than
                         something a bracket carries. The null is the same
                         configuration with NO source, which must be admitted.

ONE SUBNORMAL POLICY, AND THAT IS A PLATFORM FACT RATHER THAN A GAP. The standing
format asks for both float32 subnormal policies "where the platform has two".
Metal has one: it flushes denormals natively and exposes no lever (both denormal
pragma spellings are compile errors), which is why every Metal predicate REFUSES
the ``keep`` policy by name. The gate records the resolved policy and asserts the
reference walk is subnormal-free at every step, which is the precondition that
refusal buys.

Rule 7: one flushed line per case. Runs locally on MPS, routed through
``metal_gate_runner``. Nothing here is dispatch: ``meep_gpu.fastpath
.plan_fast_path`` still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_no_pml_offdiag_fused_electric_pair as complex_no_pml_pair,
    folded_complex_offdiag_fused_electric_pair as folded_complex_pair,
    folded_offdiag_fused_electric_pair as folded_pair,
    launch as metal_launch,
    offdiag_fused_electric_pair as plain_pair,
    shaders,
    subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)

#: An ODD budget, for the reason the residue gate records: a period-2 mode is
#: byte-equal to its seed after any even number of steps, and the
#: every-array-moved floor would read it as untouched.
STEPS = 13
SEED = 20260901

ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "update_P": lambda f, p: stepping.update_P(f, p),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
}

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields: Any) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Any, right: Any) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


def seed_state(fields: Any, seed: int) -> Any:
    """Physical-band values everywhere, plus a NON-UNIFORM inverse epsilon and
    non-uniform off-diagonal rows — on a one-value-per-component fixture a
    wrong-cell coefficient read is invisible."""
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        real = rng.normal(0.0, 0.37, size=array.shape).astype(np.float32)
        if np.iscomplexobj(array):
            imag = rng.normal(0.0, 0.37, size=array.shape).astype(np.float32)
            array[...] = (real + 1j * imag).astype(array.dtype)
        else:
            array[...] = real.astype(array.dtype)
    rng_eps = np.random.default_rng(seed ^ 0x5EED)
    reader = getattr(fields, "inverse_epsilon_for", None)
    if callable(reader):
        for component in ("Ex", "Ey", "Ez"):
            volume = reader(component)
            if volume is None or not getattr(volume, "flags", None):
                continue
            if not volume.flags.writeable:
                continue
            volume[...] = (0.2 + 0.6 * rng_eps.random(
                volume.shape)).astype(volume.dtype)
    from meep_gpu.metal_kernels import offdiag_update_e as _offdiag  # noqa: PLC0415
    for index, row in enumerate(_offdiag.row_volumes_for(fields)):
        if row is None or not getattr(row, "flags", None) or not row.flags.writeable:
            continue
        local = np.random.default_rng(seed ^ 0x0A11 ^ index)
        row[...] = (-0.4 + 0.8 * local.random(row.shape)).astype(row.dtype)
    return fields


def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    return tuple(live)


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    hits = source.count(old)
    if hits != count:
        raise AssertionError(
            f"the mutation needle {old!r} matches {hits} times, expected {count}; "
            f"an unarmed mutation reports its defect as uncaught")
    return source.replace(old, new)


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------

def _plain_source(plan: Any, mode: str) -> str:
    return plain_pair.offdiag_fused_electric_pair_source(
        plan.row_mask, plan.codes, plan.zero_metal, mode)


def _folded_source(plan: Any, mode: str) -> str:
    grid = plan.fields.grid
    folded_axes, far_axes, _reflect = folded_pair.folded_axis_tables(grid)
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        folded_offdiag_update_e as _fod)
    negate = _fod.negated_axes(plan.codes, _fod.mirror_ghost_weights(grid))
    return folded_pair.folded_offdiag_fused_electric_pair_source(
        plan.row_mask, plan.codes, plan.zero_metal, negate, folded_axes, far_axes,
        mode)


def _complex_no_pml_source(plan: Any, mode: str) -> str:
    from meep_gpu.metal_kernels import complex_fields as _complex  # noqa: PLC0415
    grid = plan.fields.grid
    kinds = stepping._boundary_kinds(grid, None)
    phased, _values = _complex.phase_arguments(
        _complex.bloch_phase_table(grid, kinds), backward=True)
    expansion = _complex.expansion_from_probe(_complex.load_expansion_probe())
    return complex_no_pml_pair.complex_no_pml_offdiag_fused_electric_pair_source(
        plan.row_mask, plan.codes, plan.zero_metal, phased, expansion, mode)


def _folded_complex_source(plan: Any, mode: str) -> str:
    from meep_gpu.metal_kernels import complex_fields as _complex  # noqa: PLC0415
    from meep_gpu.metal_kernels import folded_complex as _fc  # noqa: PLC0415
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        folded_offdiag_update_e as _fod)
    grid = plan.fields.grid
    pml = plan.fields.__dict__.get("_gate_pml")
    kinds = stepping._boundary_kinds(grid, pml)
    phased, _values = _complex.phase_arguments(
        _complex.bloch_phase_table(grid, kinds), backward=True)
    negate = _fod.negated_axes(plan.codes, _fod.mirror_ghost_weights(grid))
    folded_axes, far_axes, _reflect = folded_pair.folded_axis_tables(grid)
    expansion = _fc.expansion_from_probe(_fc.load_expansion_probe())
    return (folded_complex_pair
            .folded_complex_offdiag_fused_electric_pair_source(
                plan.row_mask, plan.codes, plan.zero_metal, phased, negate,
                folded_axes, far_axes, expansion, mode))


#: THE MUTATIONS EVERY FAMILY CARRIES, spelled against text the shared builder
#: emits so one table serves all four. Each MUST move bytes.
COMMON_MUTATIONS: Tuple[Tuple[str, str, str], ...] = (
    ("rotation_skipped_in_the_kernel",
     "    d0s[ii] = dfin0; d1s[ii] = dfin1; d2s[ii] = dfin2;",
     "    d0s[ii] = dfin0; d1s[ii] = dfin1; d2s[ii] = dfin1;"),
)


FAMILIES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "offdiag_fused_electric_pair",
        "label": "off-diagonal fused electric D/E pair",
        "module": plain_pair,
        "plan": plain_pair.plan_metal_offdiag_fused_electric_pair,
        "coverage": plain_pair.metal_offdiag_fused_electric_pair_coverage,
        "takes_probe": False,
        "cases": (
            ("pml_periodic_one_row", lambda: matrix.cart(rows={"Ex": ("Ey",)})),
            ("pml_walled_one_row", lambda: matrix.cart(
                rows={"Ex": ("Ey",)}, boundaries="metallic")),
            ("pml_walled_all_rows", lambda: matrix.cart(
                boundaries="metallic",
                rows={"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"),
                      "Ez": ("Ex", "Ey")})),
            ("pml_mixed_boundaries", lambda: matrix.cart(
                boundaries={"x": "metallic"},
                rows={"Ey": ("Ez",), "Ez": ("Ex",)})),
        ),
        "mutation_case": "pml_walled_all_rows",
        "source": _plain_source,
        "entry": "offdiag_fused_electric_pair_step",
        "mutations": (
            ("foreign_tap_read_from_the_LIVE_buffer_not_re_derived",
             ", nxi, nyi, nzi, pf0, pf1, pf2, pu0, pu1, pu2, g0, g1, g2, kmx",
             ", nxi, nyi, nzi, d0s, d1s, d2s, pu0, pu1, pu2, g0, g1, g2, kmx",
             -1),
            ("own_cell_displacement_taken_from_the_scratch_it_just_wrote",
             "    float dfin0 = d_final_0(",
             "    float dfin0 = d0s[ii] + 0.0f * d_final_0("),
            ("split_field_scratch_store_dropped",
             "    n0s[ii] = own.n0; n1s[ii] = own.n1; n2s[ii] = own.n2;\n", ""),
            ("wall_clear_dropped_on_component_0",
             "    value = ((j == 0) || (k == 0)) ? 0.0f : value;", ""),
            ("curl_pack_offset_x_reads_the_y_vector",
             "device const float* kmx = cpack + prm.off_kmx;",
             "device const float* kmx = cpack + prm.off_kmy;"),
            ("material_pack_offset_reads_the_wrong_row",
             "device const float* u01 = mpack + prm.off_u01;",
             "device const float* u01 = mpack + prm.off_u02;"),
            ("inverse_epsilon_on_the_wrong_component",
             "    float us1 = e1[ii];", "    float us1 = e0[ii];"),
        ),
        "equivalences": (),
        "predicted_nulls": (),
        "refuted": (
            ("separate_scalars", plain_pair.refuted_separate_scalar_source),
            ("cpack_only", plain_pair.refuted_cpack_only_source),
        ),
        "shipped": lambda: plain_pair.offdiag_fused_electric_pair_source(
            (1, 0, 0, 0, 0, 0), (1, 1, 1), (1, 1, 1)),
        "seam_case": "pml_periodic_one_row",
        "seam_component": "Ez",
    },
    {
        "name": "folded_offdiag_fused_electric_pair",
        "label": "folded off-diagonal fused electric D/E pair",
        "module": folded_pair,
        "plan": folded_pair.plan_metal_folded_offdiag_fused_electric_pair,
        "coverage":
            folded_pair.metal_folded_offdiag_fused_electric_pair_coverage,
        "takes_probe": False,
        "cases": (
            ("fold_y_even", lambda: matrix.folded(rows={"Ex": ("Ey",)})),
            ("fold_y_odd", lambda: matrix.folded(phase=-1,
                                                 rows={"Ex": ("Ey",)})),
            ("fold_y_odd_walled_x", lambda: matrix.folded(
                phase=-1, boundaries={"x": "metallic"},
                rows={"Ex": ("Ey",)})),
            ("fold_y_3d", lambda: matrix.folded(
                depth=1.2, rows={"Ex": ("Ey",), "Ey": ("Ez",)})),
            ("fold_y_odd_count", lambda: matrix.folded(
                extent=2.1, rows={"Ex": ("Ey",)})),
            ("fold_xy_mixed_phase", lambda: matrix.folded(
                axis="XY", phase=(1, -1),
                rows={"Ex": ("Ey",), "Ez": ("Ex",)})),
            ("fold_y_metallic_termination", lambda: matrix.folded(
                boundaries={"y": "metallic"}, rows={"Ex": ("Ey",)})),
        ),
        "mutation_case": "fold_xy_mixed_phase",
        "source": _folded_source,
        "entry": "folded_offdiag_fused_electric_pair_step",
        "mutations": (
            ("near_redirect_row_wrong",
             "    j = near_y ? 2 : j;", "    j = near_y ? 1 : j;", -1),
            ("near_parity_dropped",
             "    value = near_y ? -value : value;", "", -1),
            ("far_redirect_dropped",
             "    j = was_far ? reflect_y : j;", ""),
            ("far_parity_dropped",
             "    value = was_far ? -value : value;", ""),
            ("foreign_tap_read_from_the_LIVE_buffer_not_re_derived",
             ", nxi, nyi, nzi, pf0, pf1, pf2, pu0, pu1, pu2, g0, g1, g2, kmx",
             ", nxi, nyi, nzi, d0s, d1s, d2s, pu0, pu1, pu2, g0, g1, g2, kmx",
             -1),
            ("split_field_scratch_store_dropped",
             "    n0s[ii] = own.n0; n1s[ii] = own.n1; n2s[ii] = own.n2;\n", ""),
            # ON A FIXTURE WHERE THE TWO VECTORS DIFFER. A two-axis fold gives x
            # and y the same extent AND the same one-sided absorber, so `kmx` and
            # `kmy` are the same words there and the swap is a MEASURED no-op —
            # 0 differing words, which would report this mutation as uncaught.
            # `fold_y_3d` is the fixture that separates them (x cross 1.6 with a
            # two-sided layer, y extent 2.0 with a one-sided one).
            ("curl_pack_offset_x_reads_the_y_vector",
             "device const float* kmx = cpack + prm.off_kmx;",
             "device const float* kmx = cpack + prm.off_kmy;", 1, "fold_y_3d"),
        ),
        "equivalences": (),
        "predicted_nulls": (),
        "refuted": (
            ("separate_scalars", folded_pair.refuted_separate_scalar_source),
            ("cpack_only", folded_pair.refuted_cpack_only_source),
        ),
        "shipped": lambda: (
            folded_pair.folded_offdiag_fused_electric_pair_source(
                (1, 0, 0, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 1, 0),
                {1: 1}, {1: 1})),
        "seam_case": "fold_y_even",
        "seam_component": "Ez",
    },
    {
        "name": "complex_no_pml_offdiag_fused_electric_pair",
        "label": "complex no-PML off-diagonal fused electric D/E pair",
        "module": complex_no_pml_pair,
        "plan": (complex_no_pml_pair
                 .plan_metal_complex_no_pml_offdiag_fused_electric_pair),
        "coverage": (complex_no_pml_pair
                     .metal_complex_no_pml_offdiag_fused_electric_pair_coverage),
        "takes_probe": "complex_probe",
        "cases": (
            ("bloch_offdiag", matrix.complex_lossless_offdiag_no_pml),
            ("walled_offdiag", lambda: matrix.cart(
                pml=0, storage=False, complex_storage=True,
                boundaries="metallic",
                rows={"Ex": ("Ey", "Ez"), "Ey": ("Ez",)})),
        ),
        "mutation_case": "bloch_offdiag",
        "source": _complex_no_pml_source,
        "entry": "complex_no_pml_offdiag_fused_electric_pair_step",
        "mutations": (
            ("foreign_tap_read_from_the_LIVE_buffer_not_re_derived",
             ", nxi, nyi, nzi, pf0, pf1, pf2, g0, g1, g2, dtdx",
             ", nxi, nyi, nzi, d0s, d1s, d2s, g0, g1, g2, dtdx", -1),
            ("curl_bloch_phase_dropped",
             "      b_x = wx ? c_mul(b_x, px) : b_x;", ""),
            ("inverse_epsilon_on_the_wrong_component",
             "c_mul_field_left(dfin1, e1[ii])",
             "c_mul_field_left(dfin1, e0[ii])"),
            ("scratch_store_dropped",
             "    d0s[ii] = dfin0; d1s[ii] = dfin1; d2s[ii] = dfin2;\n", ""),
            ("wall_clear_dropped_on_component_0",
             "    value = ((j == 0) || (k == 0)) ? float2(0.0f, 0.0f) : value;",
             "", 1, "walled_offdiag"),
        ),
        "equivalences": (),
        "predicted_nulls": (),
        "refuted": (
            ("separate_scalars",
             complex_no_pml_pair.refuted_separate_scalar_source),
        ),
        "shipped": lambda: (
            complex_no_pml_pair
            .complex_no_pml_offdiag_fused_electric_pair_source(
                (1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0), "FMA_V1")),
        "seam_case": "bloch_offdiag",
        "seam_component": "Ez",
    },
    {
        "name": "folded_complex_offdiag_fused_electric_pair",
        "label": "folded complex off-diagonal fused electric D/E pair",
        "module": folded_complex_pair,
        "plan": (folded_complex_pair
                 .plan_metal_folded_complex_offdiag_fused_electric_pair),
        "coverage": (folded_complex_pair
                     .metal_folded_complex_offdiag_fused_electric_pair_coverage),
        "takes_probe": "folded_complex_probe",
        "cases": (
            ("fold_complex_even", lambda: matrix.folded(
                complex_storage=True, rows={"Ey": ["Ez"]})),
            ("fold_complex_odd", lambda: matrix.folded(
                complex_storage=True, phase=-1, rows={"Ey": ["Ez"]})),
            ("fold_complex_bloch", lambda: matrix.folded(
                complex_storage=True, k_point=(0.3, 0.0, 0.0),
                rows={"Ey": ["Ez"]})),
            ("fold_complex_walled_x", lambda: matrix.folded(
                complex_storage=True, boundaries={"x": "metallic"},
                rows={"Ey": ["Ez"], "Ex": ["Ey"]})),
            ("fold_complex_xy_mixed_phase", lambda: matrix.folded(
                complex_storage=True, axis="XY", phase=(1, -1),
                rows={"Ey": ["Ez"], "Ex": ["Ey"]})),
        ),
        "mutation_case": "fold_complex_walled_x",
        "source": _folded_complex_source,
        "entry": "folded_complex_offdiag_fused_electric_pair_step",
        "mutations": (
            # THE AUDIT FINDING, ARMED ON DEVICE. The array path spells a complex
            # parity `phase * plane`; a plain copy at +1 and a bare negation at -1
            # are the REAL families' (correct) spelling and differ from it in ten
            # of a hundred engineered signed-zero pairs. The walled fixture is the
            # one that reaches such a pair: the far fill images a cell the wall
            # clear has already zeroed.
            ("complex_far_parity_spelled_as_a_bare_negation",
             "    value = was_far ? c_mul(float2(-1.0f, 0.0f), value) : value;",
             "    value = was_far ? -value : value;", 1),
            ("far_redirect_dropped",
             "    j = was_far ? reflect_y : j;", ""),
            ("foreign_tap_read_from_the_LIVE_buffer_not_re_derived",
             ", nxi, nyi, nzi, pf0, pf1, pf2, pu0, pu1, pu2, g0, g1, g2, kmx",
             ", nxi, nyi, nzi, d0s, d1s, d2s, pu0, pu1, pu2, g0, g1, g2, kmx",
             -1),
            ("split_field_scratch_store_dropped",
             "    n0s[ii] = own.n0; n1s[ii] = own.n1; n2s[ii] = own.n2;\n", ""),
            ("material_pack_offset_reads_the_wrong_row",
             "device const float* u01 = mpack + prm.off_u01;",
             "device const float* u01 = mpack + prm.off_u02;"),
        ),
        "equivalences": (),
        # A PREDICTED NULL IS A MEASUREMENT, AND IT IS NOT A DECLARED
        # EQUIVALENCE. The two spellings are NOT equal in general — the host probe
        # measures them differing in ten of a hundred engineered signed-zero pairs
        # — but the +1 arm cannot reach such a pair on any fixture this corpus
        # has: a near redirect reads stored cell 2, which the wall clear (stored
        # cell 0) never touches, so the value it multiplies is never an exact
        # zero. The FAR arm can, and is armed above as a must-catch at -1. If this
        # ever DID diverge the prediction was wrong and the leg fails.
        "predicted_nulls": (
            ("complex_near_parity_spelled_as_a_plain_copy_at_plus_one",
             "    value = near_y ? c_mul(float2(1.0f, 0.0f), value) : value;",
             "", -1,
             "the near redirect reads stored cell 2, which the wall clear never "
             "reaches, so no exact zero arrives at the one place a c_mul by "
             "(+1, +0) and a plain copy differ; the far arm at -1 CAN reach one "
             "and is armed as a must-catch"),
        ),
        "refuted": (
            ("separate_scalars",
             folded_complex_pair.refuted_separate_scalar_source),
            ("cpack_only", folded_complex_pair.refuted_cpack_only_source),
        ),
        "shipped": lambda: (
            folded_complex_pair
            .folded_complex_offdiag_fused_electric_pair_source(
                (0, 0, 1, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0),
                {1: 1}, {1: 1}, "FMA_V1")),
        "seam_case": "fold_complex_even",
        "seam_component": "Ez",
    },
)


def build_plan(spec: Mapping[str, Any], fields: Any, pml: Any,
               residency: Residency, probes: Mapping[str, Any],
               sources: Sequence[Any] = (),
               functions: Optional[Mapping[Any, Any]] = None) -> Any:
    kwargs: Dict[str, Any] = {}
    if spec["takes_probe"]:
        kwargs["probe"] = probes[spec["takes_probe"]]
    if functions is not None:
        kwargs["functions"] = functions
    plan = spec["plan"](fields, pml, sources=tuple(sources),
                        residency=residency, **kwargs)
    if plan is not None:
        # The folded-complex source rebuilder needs the PML to resolve the
        # boundary kinds; the plan does not carry one, so the gate parks it on
        # the fields object it already owns rather than widening the plan.
        fields.__dict__["_gate_pml"] = pml
    return plan


def _coverage_kwargs(spec: Mapping[str, Any],
                     probes: Mapping[str, Any]) -> Dict[str, Any]:
    return ({"probe": probes[spec["takes_probe"]]}
            if spec["takes_probe"] else {})


# ---------------------------------------------------------------------------
# Legs A and B: the two walks
# ---------------------------------------------------------------------------

class CountingFunction:
    """A launcher wrapper that counts calls INDEPENDENTLY of the plan.

    TWO COUNTERS OR NONE. ``plan.launches`` is the plan's own tally and a plan
    that never launched would report zero just as convincingly as one that did;
    this wrapper sits between the plan and the compiled entry point, so a launch
    the plan claims and did not make is a disagreement rather than a matching
    pair of zeros.
    """

    __slots__ = ("function", "calls")

    def __init__(self, function: Any) -> None:
        self.function = function
        self.calls = 0

    def __call__(self, *args: Any) -> Any:
        self.calls += 1
        return self.function(*args)


def metal_step(fields: Any, pml: Any, plans: Mapping[str, Any],
               residency: Residency, live: Sequence[str]) -> None:
    """One complete step: a slot a plan owns runs on the device, the rest on the
    array path bracketed by sync_out / sync_in."""
    ran: set = set()
    for name in live:
        plan = plans.get(name)
        if plan is not None:
            if id(plan) not in ran:
                plan.run()
                ran.add(id(plan))
            continue
        residency.sync_out()
        ARRAY_PATH[name](fields, pml)
        residency.sync_in()


def _fused_plans(plan: Any) -> Dict[str, Any]:
    return {name: plan for name in plan.replaces_sub_steps}


def run_case(spec: Mapping[str, Any], build: Callable[[], Tuple[Any, Any]],
             probes: Mapping[str, Any], steps: Optional[int] = None,
             functions: Optional[Mapping[Any, Any]] = None) -> Dict[str, Any]:
    """Step the array path and the fused device plan side by side."""
    steps = STEPS if steps is None else int(steps)
    reference, reference_pml = build()
    actual, actual_pml = build()
    seed_state(reference, SEED)
    seed_state(actual, SEED)

    drift = compare(reference, actual)
    if drift:
        return {"passed": False, "reason": "the two builds are not identical",
                "drift": drift}

    residency = Residency()
    plan = build_plan(spec, actual, actual_pml, residency, probes,
                      functions=functions)
    if plan is None:
        return {"passed": False, "reason": "the pair was refused",
                "refusals": list(spec["coverage"](
                    actual, actual_pml, (), residency,
                    **_coverage_kwargs(spec, probes)).reasons)[:8]}

    counter = CountingFunction(plan._functions[shaders.CONTRACT_OFF])
    plan._functions[shaders.CONTRACT_OFF] = counter

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    # A REPLACED PASS ABSENT FROM THE LIVE SET IS INERT, NOT MISSING. `zero_metal_D`
    # does no work on a grid with no metallic axis, and `live_sub_steps` leaves it
    # out there; the kernel carries the clear unconditionally (the specialisation
    # emits no guard when no axis is walled), so the two agree. What WOULD be a
    # defect is the pass the fusion starts at being absent, which would mean the
    # plan replaced nothing at all.
    inert = [name for name in plan.replaces_sub_steps if name not in live]
    missing = [name for name in (plan.replaces_sub_steps[0], "update_E")
               if name not in live]
    before = frozen(actual)
    residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
        metal_step(actual, actual_pml, _fused_plans(plan), residency, live)
        residency.sync_out()
        difference = compare(reference, actual)
        census = sum(subnormal.census(np.real(value))
                     + subnormal.census(np.imag(value))
                     if np.iscomplexobj(value) else subnormal.census(value)
                     for value in state_of(reference).values())
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": int(census)})
        if difference or census:
            break

    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    clean = all(row["reference_subnormals"] == 0 for row in per_step)
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step)
                   and counter.calls == plan.launches)
    return {
        "passed": bool(identical and clean and launches_ok and not still
                       and not missing),
        "bit_identical": identical,
        "subnormal_free": clean,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "steps_compared": len(per_step),
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "arrays_compared": len(before),
        "arrays_that_never_moved": still,
        "moved_words": int(sum(moved.values())),
        "runs": plan.runs, "launches": plan.launches,
        "independent_launch_count": counter.calls,
        "launches_per_run": plan.launches_per_run,
        "live_passes": list(live),
        "replaces": list(plan.replaces_sub_steps),
        "replaced_passes_inert_on_this_fixture": inert,
        "replaced_passes_missing_from_the_live_set": missing,
        "shape": list(plan.shape), "row_mask": list(plan.row_mask),
        "zero_metal": list(plan.zero_metal),
        "mirrors": len(residency.names),
    }


def run_against_certified_singles(
        spec: Mapping[str, Any], build: Callable[[], Tuple[Any, Any]],
        probes: Mapping[str, Any],
        steps: Optional[int] = None) -> Dict[str, Any]:
    """Leg B: the fused plan against the CERTIFIED DEVICE KERNELS it replaces.

    Both engines run on MPS. The reference composes ``launch.plan_step(...,
    fuse=False)`` — the shipped composer's own per-slot selection, which for
    these configurations picks exactly the certified curl, the certified
    ghost-fill plans and the certified off-diagonal constitutive — so a
    divergence here is the WELD's, not NumPy's.
    """
    steps = STEPS if steps is None else int(steps)
    reference, reference_pml = build()
    actual, actual_pml = build()
    seed_state(reference, SEED)
    seed_state(actual, SEED)

    reference_residency = Residency()
    single = metal_launch.plan_step(
        reference, reference_pml, residency=reference_residency, sources=(),
        complex_probe=probes["complex_probe"], beta_probe=probes["beta_probe"],
        folded_complex_probe=probes["folded_complex_probe"],
        cylindrical_complex_probe=probes["cylindrical_complex_probe"],
        fuse=False)
    residency = Residency()
    plan = build_plan(spec, actual, actual_pml, residency, probes)
    if plan is None:
        return {"passed": False, "reason": "the pair was refused"}

    live = live_passes(actual, actual_pml)
    covered = [name for name in plan.replaces_sub_steps
               if single.plans.get(name) is not None]
    reference_residency.sync_in()
    residency.sync_in()

    rows: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        metal_step(reference, reference_pml, single.plans, reference_residency,
                   live)
        reference_residency.sync_out()
        metal_step(actual, actual_pml, _fused_plans(plan), residency, live)
        residency.sync_out()
        difference = compare(reference, actual)
        rows.append({"step": step, "differing_words": sum(difference.values()),
                     "differing_arrays": dict(sorted(difference.items()))})
        if difference:
            break
    identical = len(rows) == steps and all(r["differing_words"] == 0 for r in rows)
    return {
        "passed": bool(identical and covered),
        "bit_identical": identical,
        "steps_compared": len(rows),
        "differing_words": rows[-1]["differing_words"],
        "differing_arrays": rows[-1]["differing_arrays"],
        "replaced_slots_the_singles_composition_covers_on_device": covered,
        "singles_selected": {name: single.selected.get(name)
                             for name in plan.replaces_sub_steps},
    }


def leg_identity(spec: Mapping[str, Any],
                 probes: Mapping[str, Any]) -> Dict[str, Any]:
    """A: the full mechanism against the array path, every case, every step."""
    rows: Dict[str, Any] = {}
    for label, build in spec["cases"]:
        started = time.time()
        rows[label] = run_case(spec, build, probes)
        log(f"    identity  {spec['name']:44s} {label:28s} "
            f"passed={rows[label]['passed']} "
            f"words={rows[label].get('differing_words')} "
            f"launches={rows[label].get('launches')}"
            f"/{rows[label].get('independent_launch_count')} "
            f"({time.time() - started:5.1f} s)")
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows}


def leg_singles(spec: Mapping[str, Any],
                probes: Mapping[str, Any]) -> Dict[str, Any]:
    """B: the fused plan against the certified DEVICE kernels it replaces."""
    rows: Dict[str, Any] = {}
    for label, build in spec["cases"]:
        started = time.time()
        rows[label] = run_against_certified_singles(spec, build, probes)
        log(f"    singles   {spec['name']:44s} {label:28s} "
            f"passed={rows[label]['passed']} "
            f"words={rows[label].get('differing_words')} "
            f"({time.time() - started:5.1f} s)")
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows}


def _mutated_functions(spec: Mapping[str, Any], plan: Any, old: str, new: str,
                       count: int = 1) -> Dict[Any, Any]:
    mode = shaders.CONTRACT_OFF
    source = spec["source"](plan, mode)
    if count < 0:
        hits = source.count(old)
        if hits < 1:
            raise AssertionError(
                f"the mutation needle {old!r} matched nothing; an unarmed "
                f"mutation reports its defect as uncaught")
        source = source.replace(old, new)
    else:
        source = needle(source, old, new, count)
    function = getattr(compile_source(source), spec["entry"])
    return {mode: function}


def leg_mutation(spec: Mapping[str, Any],
                 probes: Mapping[str, Any]) -> Dict[str, Any]:
    """C: the DEGENERATE mechanism. Every armed mutation MUST diverge."""
    cases = dict(spec["cases"])
    default = cases[spec["mutation_case"]]
    rows: Dict[str, Any] = {}
    for mutation in tuple(spec["mutations"]) + COMMON_MUTATIONS:
        # (label, old, new[, count[, case]]). A PER-MUTATION CASE is not a
        # convenience: an arm whose two operands happen to be equal on the default
        # fixture is a MEASURED no-op, and moving it to a fixture that separates
        # them is the difference between an armed mutation and a vacuous one.
        label, old, new = mutation[0], mutation[1], mutation[2]
        count = mutation[3] if len(mutation) > 3 else 1
        build = cases[mutation[4]] if len(mutation) > 4 else default
        reference, reference_pml = build()
        residency = Residency()
        seed_state(reference, SEED)
        plan = build_plan(spec, reference, reference_pml, residency, probes)
        if plan is None:
            rows[label] = {"passed": False, "reason": "the pair was refused"}
            continue
        try:
            functions = _mutated_functions(spec, plan, old, new, count)
        except AssertionError as error:
            rows[label] = {"passed": False, "reason": str(error)}
            log(f"    mutation  {spec['name']:44s} {label:56s} UNARMED")
            continue
        result = run_case(spec, build, probes, functions=functions)
        diverged = not result.get("bit_identical", True)
        rows[label] = {
            "passed": bool(diverged and result.get("launches")),
            "caught": diverged,
            "first_divergence": result.get("first_divergence"),
            "differing_words": result.get("differing_words"),
            "launches": result.get("launches"),
        }
        log(f"    mutation  {spec['name']:44s} {label:56s} "
            f"caught={diverged} words={result.get('differing_words')}")

    equivalences: Dict[str, Any] = {}
    for label, old, new in spec.get("equivalences", ()):
        reference, reference_pml = build()
        residency = Residency()
        seed_state(reference, SEED)
        plan = build_plan(spec, reference, reference_pml, residency, probes)
        functions = _mutated_functions(spec, plan, old, new)
        result = run_case(spec, build, probes, functions=functions)
        identical = bool(result.get("bit_identical"))
        equivalences[label] = {"passed": identical,
                               "differing_words": result.get("differing_words")}
        log(f"    equivalent{spec['name']:44s} {label:56s} identical={identical}")
    predicted: Dict[str, Any] = {}
    for label, old, new, count, reason in spec.get("predicted_nulls", ()):
        build_null = default
        reference, reference_pml = build_null()
        residency = Residency()
        seed_state(reference, SEED)
        plan = build_plan(spec, reference, reference_pml, residency, probes)
        functions = _mutated_functions(spec, plan, old, new, count)
        result = run_case(spec, build_null, probes, functions=functions)
        identical = bool(result.get("bit_identical"))
        predicted[label] = {
            "passed": identical, "predicted_null": True,
            "differing_words": result.get("differing_words"), "reason": reason}
        log(f"    predicted {spec['name']:44s} {label:56s} "
            f"null={identical} words={result.get('differing_words')}")

    return {"passed": (all(row["passed"] for row in rows.values())
                       and all(row["passed"] for row in equivalences.values())
                       and all(row["passed"] for row in predicted.values())),
            "mutations": rows, "declared_equivalences": equivalences,
            "predicted_nulls": predicted}


# ---------------------------------------------------------------------------
# Leg C2: the ROTATION, driven through the shipped launcher's own door
# ---------------------------------------------------------------------------

def leg_rotation(spec: Mapping[str, Any],
                 probes: Mapping[str, Any]) -> Dict[str, Any]:
    """The launcher's post-launch rotation, and the two ways it can be wrong.

    NOT A SOURCE MUTATION — this arm exercises the HOST half of the design
    through the shipped plan object, which is the only half a kernel mutation
    cannot reach:

    * ``rotation_skipped``   — the plan launches and does NOT swap the engine's
      references, so the next step reads the pre-launch displacement. MUST
      diverge.
    * ``scratch_aliased``    — the twin table is pointed back at the live host
      array, so read and write resolve to ONE tensor. The plan must REFUSE this
      at ``_resolve`` rather than launch it, which is the guard the design's
      "nothing written is read" rests on.
    """
    build = dict(spec["cases"])[spec["mutation_case"]]
    rows: Dict[str, Any] = {}

    reference, reference_pml = build()
    actual, actual_pml = build()
    seed_state(reference, SEED)
    seed_state(actual, SEED)
    residency = Residency()
    plan = build_plan(spec, actual, actual_pml, residency, probes)
    if plan is None:
        return {"passed": False, "reason": "the pair was refused"}
    live = live_passes(actual, actual_pml)
    residency.sync_in()
    diverged_at = None
    for step in range(1, 4):
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
        # THE SKIP: launch, then put the engine's references back where they were.
        held = {name: getattr(actual, name) for name in plan.rotated_names}
        metal_step(actual, actual_pml, _fused_plans(plan), residency, live)
        for name, value in held.items():
            plan.rotated[name] = getattr(actual, name)
            setattr(actual, name, value)
        residency.sync_out()
        if compare(reference, actual):
            diverged_at = step
            break
    rows["rotation_skipped"] = {"passed": diverged_at is not None,
                                "first_divergence": diverged_at}
    log(f"    rotation  {spec['name']:44s} rotation_skipped "
        f"caught={diverged_at is not None}")

    fresh, fresh_pml = build()
    seed_state(fresh, SEED)
    residency2 = Residency()
    plan2 = build_plan(spec, fresh, fresh_pml, residency2, probes)
    name = plan2.rotated_names[0]
    plan2.rotated[name] = getattr(fresh, name)
    refused = False
    message = ""
    try:
        residency2.sync_in()
        plan2.run()
    except RuntimeError as error:
        refused, message = True, str(error)[:200]
    rows["scratch_aliased_to_the_live_buffer"] = {
        "passed": refused, "refused": refused, "message": message}
    log(f"    rotation  {spec['name']:44s} scratch_aliased refused={refused}")
    return {"passed": all(row["passed"] for row in rows.values()), "arms": rows}


# ---------------------------------------------------------------------------
# Leg E: the in-seam electric source is REFUSED
# ---------------------------------------------------------------------------

def _volume_source(fields: Any, component: str) -> Any:
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                        envelope=ContinuousEnvelope(frequency=1.0,
                                                    is_integrated=False))


def leg_seam_refusal(spec: Mapping[str, Any],
                     probes: Mapping[str, Any]) -> Dict[str, Any]:
    """E: an in-seam ELECTRIC source must be refused BY NAME; none must not.

    THE NULL IS THE SAME CONFIGURATION WITH NO SOURCE. Without it, a predicate
    that refused everything would pass this leg.
    """
    build = dict(spec["cases"])[spec["seam_case"]]
    fields, pml = build()
    seed_state(fields, SEED)
    residency = Residency()
    source = _volume_source(fields, spec["seam_component"])
    kwargs = _coverage_kwargs(spec, probes)
    with_source = spec["coverage"](fields, pml, (source,), residency, **kwargs)
    without = spec["coverage"](fields, pml, (), residency, **kwargs)
    undeclared = spec["coverage"](fields, pml, None, residency, **kwargs)
    named = [reason for reason in with_source.reasons
             if "deposit_repair.repairable" in reason]
    planned = build_plan(spec, fields, pml, residency, probes,
                         sources=(source,))
    row = {
        "passed": bool(not with_source.covered and named and without.covered
                       and not undeclared.covered and planned is None),
        "refused_with_an_in_seam_electric_source": not with_source.covered,
        "refusal_names_the_repairable_clause": bool(named),
        "admitted_with_no_source": bool(without.covered),
        "refused_when_the_source_set_is_UNDECLARED": not undeclared.covered,
        "plan_is_None_with_a_source": planned is None,
        "carries_deposit_repair": bool(spec["module"].CARRIES_DEPOSIT_REPAIR),
        "reasons": list(with_source.reasons)[:4],
    }
    log(f"    seam      {spec['name']:44s} refused={not with_source.covered} "
        f"admitted_without={without.covered}")
    return row


# ---------------------------------------------------------------------------
# Leg D: the binding ceiling
# ---------------------------------------------------------------------------

def _synthetic_pointer_signature(pointers: int) -> str:
    """``pointers`` device pointers plus one packed ``Params&``, and a body that
    touches every one of them."""
    lines = [f"    device float* p{index:02d} [[buffer({index})]],"
             for index in range(pointers)]
    lines.append(f"    constant Params& prm [[buffer({pointers})]],")
    body = "\n".join(f"    p{index:02d}[idx] = p{index:02d}[idx] + total;"
                     for index in range(pointers))
    total = " + ".join(f"p{index:02d}[0]" for index in range(pointers))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "struct Params { uint n_elem; };",
        "kernel void bisect(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    float total = {total};",
        body,
        "}",
        "",
    ))


def leg_binding_ceiling() -> Dict[str, Any]:
    """D: shipped compiles at its declared count, refuted is refused, bisect."""
    rows: Dict[str, Any] = {}
    binding = re.compile(r"\[\[buffer\((\d+)\)\]\]")
    for spec in FAMILIES:
        name = spec["name"]
        source = spec["shipped"]()
        slots = sorted({int(number) for number in binding.findall(source)})
        declared = spec["module"].PACKED_BINDINGS
        try:
            compile_source(source)
            compiled, error = True, ""
        except Exception as exc:  # noqa: BLE001
            compiled, error = False, str(exc).strip().splitlines()[-1][:160]
        rows[f"{name}/shipped"] = {
            "passed": bool(compiled and len(slots) == declared
                           and slots == list(range(len(slots)))
                           and len(slots) <= MAX_BUFFER_BINDINGS),
            "compiled": compiled, "error": error,
            "bindings_parsed": len(slots), "bindings_declared": declared,
        }
        log(f"    ceiling   {name}/shipped bindings={len(slots)}"
            f"/{declared} compiled={compiled}")
        for label, builder in spec.get("refuted", ()):
            key = f"{name}/{label}"
            refused_source = builder()
            refused_slots = sorted({int(number)
                                    for number in binding.findall(refused_source)})
            try:
                compile_source(refused_source)
                rows[key] = {"passed": False, "compiled": True,
                             "bindings_parsed": len(refused_slots)}
            except Exception as error:  # noqa: BLE001 - the failure IS the result
                rows[key] = {"passed": True, "compiled": False,
                             "bindings_parsed": len(refused_slots),
                             "error": str(error).strip().splitlines()[-1][:160]}
            log(f"    ceiling   {key:70s} refused={not rows[key]['compiled']}")

    bisect: Dict[str, Any] = {}
    for pointers, must_compile in ((MAX_BUFFER_BINDINGS - 1, True),
                                   (MAX_BUFFER_BINDINGS, False)):
        source = _synthetic_pointer_signature(pointers)
        try:
            compile_source(source)
            compiled, error = True, ""
        except Exception as exc:  # noqa: BLE001
            compiled, error = False, str(exc).strip().splitlines()[-1][:160]
        bisect[f"{pointers}_pointers_plus_params"] = {
            "passed": compiled is must_compile,
            "compiled": compiled, "must_compile": must_compile, "error": error,
        }
        log(f"    bisect    {pointers} pointers + Params  compiled={compiled} "
            f"(must_compile={must_compile})")
    rows["ceiling_bisected_on_this_host"] = {
        "passed": all(row["passed"] for row in bisect.values()),
        "probes": bisect, "max_buffer_bindings": MAX_BUFFER_BINDINGS,
    }
    return {"passed": all(row["passed"] for row in rows.values()),
            "signatures": rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CORPUS_ROWS: Dict[str, Any] = {
    "census": "parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19",
    "offdiag_fused_electric_pair": {
        "cell": "D_to_E (PML, offdiag)", "cell_rows": 16,
        "seam_instances_reachable": 8,
        "note": "the other 8 carry an electric source inside the seam, which "
                "deposit_repair.repairable refuses for an off-diagonal chi1inv"},
    "folded_offdiag_fused_electric_pair": {
        "cell": "D_to_E (folded, folded offdiag)", "cell_rows": 19,
        "seam_instances_reachable": 9, "note": "same clause, 10 refused"},
    "complex_no_pml_offdiag_fused_electric_pair": {
        "cell": "D_to_E (complex no-PML curl, complex no-PML off-diagonal)",
        "cell_rows": 2, "seam_instances_reachable": 1,
        "note": "same clause, 1 refused"},
    "folded_complex_offdiag_fused_electric_pair": {
        "cell": "D_to_E (folded complex, complex folded off-diagonal PML E)",
        "cell_rows": 3, "seam_instances_reachable": 1,
        "note": "same clause, 2 refused"},
}


def main() -> int:
    global STEPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--only", default=None, help="run one family by name")
    parser.add_argument(
        "--legs", default="identity,singles,mutation,rotation,seam,ceiling")
    parser.add_argument("--steps", type=int, default=STEPS)
    arguments = parser.parse_args()
    STEPS = int(arguments.steps)

    environment = matrix.prepare_environment()
    import torch  # noqa: PLC0415
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")

    from meep_gpu.metal_kernels import complex_fields as _complex  # noqa: PLC0415
    from meep_gpu.metal_kernels import folded_complex as _folded  # noqa: PLC0415
    from meep_gpu.metal_kernels import special_kz as _beta  # noqa: PLC0415
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        cylindrical_complex as _cylindrical)

    paths = {
        "complex_probe":
            environment.get("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"),
        "beta_probe": environment.get("MEEP_GPU_METAL_EXPANSION_PROBE"),
        "folded_complex_probe":
            environment.get("MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE"),
        "cylindrical_complex_probe":
            environment.get("MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE"),
    }
    probes = {
        "complex_probe": _complex.load_expansion_probe(),
        "beta_probe": _beta.load_expansion_probe(),
        "folded_complex_probe": _folded.load_expansion_probe(),
        "cylindrical_complex_probe": _cylindrical.load_expansion_probe(),
    }
    for key in ("complex_probe", "folded_complex_probe"):
        if probes[key] is None:
            raise SystemExit(
                f"the {key} expansion artifact at {paths[key]!r} did not load; "
                f"the two complex families would refuse BY NAME and this gate "
                f"would measure the artifact's absence rather than arithmetic")

    legs = tuple(part.strip() for part in arguments.legs.split(",") if part.strip())
    started = time.time()
    record: Dict[str, Any] = {
        "gate": "metal_offdiag_stencil_welds",
        "host": {"platform": sys.platform, "torch": torch.__version__,
                 "mps_available": bool(torch.backends.mps.is_available())},
        "steps": arguments.steps, "seed": SEED, "probes": paths,
        "families": {}, "corpus_rows": CORPUS_ROWS,
        "subnormal_policy_note": (
            "Metal has ONE float32 subnormal policy: it flushes natively and "
            "exposes no lever, which is why every Metal predicate refuses "
            "'keep' by name. The identity leg asserts the reference walk is "
            "subnormal-free at every step, which is the precondition that "
            "refusal buys."),
    }

    for spec in FAMILIES:
        if arguments.only and spec["name"] != arguments.only:
            continue
        log(f"  {spec['name']}")
        entry: Dict[str, Any] = {}
        if "identity" in legs:
            entry["identity"] = leg_identity(spec, probes)
        if "singles" in legs:
            entry["certified_singles"] = leg_singles(spec, probes)
        if "mutation" in legs:
            entry["mutation"] = leg_mutation(spec, probes)
        if "rotation" in legs:
            entry["rotation"] = leg_rotation(spec, probes)
        if "seam" in legs:
            entry["seam_refusal"] = leg_seam_refusal(spec, probes)
        entry["passed"] = all(value.get("passed") for value in entry.values()
                              if isinstance(value, dict))
        record["families"][spec["name"]] = entry

    if "ceiling" in legs:
        record["binding_ceiling"] = leg_binding_ceiling()

    record["torch_version"] = torch.__version__
    record["numpy_version"] = np.__version__
    record["metal_frontend"] = metal_frontend_version()
    record["subnormal_policy"] = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")
    record["elapsed_s"] = round(time.time() - started, 2)
    record["verdict"] = "PASS" if all(
        value.get("passed") for value in record["families"].values()) and (
        record.get("binding_ceiling", {}).get("passed", True)) else "FAIL"
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(record)
    text = json.dumps(record, indent=1, sort_keys=True)
    if arguments.out:
        path = Path(arguments.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        log(f"  wrote {path}")
    else:
        print(text)
    log(f"  VERDICT {record['verdict']}  ({record['elapsed_s']} s)")
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
