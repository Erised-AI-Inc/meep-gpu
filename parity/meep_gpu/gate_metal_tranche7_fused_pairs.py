"""The device gate for the 2026-08-30 Metal fusion tranche — seven new fused pairs.

WHAT THIS CERTIFIES, AND ON WHAT. Seven products landed in one round, every one of
them a weld across a driver seam:

    complex_fused_electric_pair          complex/Bloch      D -> E   (16 corpus rows)
    folded_fused_dispersive_pair         folded x disp. E   D -> E   (4)
    folded_beta_complex_fused_pair       folded beta cplx   D -> E   (3)
    folded_complex_fused_pair            folded complex     D -> E   (2)
    beta_fused_electric_pair             special_kz real    D -> E   (1)
    folded_beta_real_fused_pair          folded beta real   D -> E   (1)
    folded_beta_real_fused_magnetic_pair folded beta real   B -> H   (1)

THE COMPARISON IS PER COMPLETE DRIVER STEP, AS uint32 WORDS, over every stored volume
a step can touch — never ``allclose`` and never a single sub-step. A fused pair right
for one launch and wrong forever after diverges only once ``fu_*`` and ``f_w_*``
accumulate, which is why the budget is twelve steps and why the B/H half is compared
on a D/E family (a corrupted E reaches B through ``step_B`` on the very next step).

THE THREE LEGS ARE ARMED, and a case where the negative leg does not diverge is
VACUOUS and is reported as such rather than counted:

    A  ``identity``   the full mechanism. REQUIRES 0 differing words at every step,
                      the declared launch count, and that every compared array MOVED
                      (a no-op agreeing with a no-op is not a pass);
    B  ``mutation``   the DEGENERATE mechanism — one arithmetic or index line of the
                      shipped source replaced, compiled, and handed to the plan
                      builder through its ``functions`` seam. Each MUST diverge;
    C  ``deposit``    the seam's in-seam source, with and without the repair bracket.
                      The bracketed walk must be bit-identical; the UNBRACKETED one —
                      the shipped plan with an empty source list, then injected anyway
                      — MUST diverge. That is "a product that declares
                      CARRIES_DEPOSIT_REPAIR without the wrappers", built out of
                      shipped code rather than a hand-written mutant.

THE BINDING CEILING IS COMPILED, NOT ASSERTED. Every family that spells a
``SEPARATE_SCALAR_BINDINGS`` or split-plane refutation has that source compiled here
and the FAILURE is required, so "the scalars must ride packed" is a measurement on
this toolchain rather than arithmetic on a number in a docstring.

Runs locally on MPS. Nothing here is dispatch: ``meep_gpu.fastpath.plan_fast_path``
still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parent.parent
for path in (str(HERE), str(API_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    beta_fused_electric_pair,
    complex_fused_electric_pair,
    folded_beta_complex_fused_pair,
    folded_beta_real_fused_magnetic_pair,
    folded_beta_real_fused_pair,
    folded_complex_fused_pair,
    folded_fused_dispersive_pair,
    launch as metal_launch,
    subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)

STEPS = 12
SEED = 20260830

#: Which array-path function each live pass is. One table, the driver's own order.
ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
    "update_P": lambda f, p: stepping.update_P(f, p),
}

#: Every stored volume a complete step can touch. The ``fu_*`` PML auxiliaries and the
#: ``f_w_*`` constitutive workspaces are STATE. The half a given family does not touch
#: is in the list deliberately: a fused pair that corrupted E reaches B through
#: ``step_B`` on the very next step, and a comparison blind to that reports on half the
#: engine.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose."""
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


def seed_state(fields: Any, seed: int, scale: float = 1.0) -> Any:
    """Physical-band values in every allocated state volume.

    NOT COSMETIC. A predicate reads dtype, contiguity and base address, and a
    zero-filled volume passes every one of them — but a plan built on zeros and
    launched is a no-op agreeing with a no-op, which is the vacuity this gate refuses.
    Complex volumes are seeded on both planes.
    """
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        real = rng.normal(0.0, 0.37, size=array.shape).astype(np.float32) * scale
        if np.iscomplexobj(array):
            imag = rng.normal(0.0, 0.37, size=array.shape).astype(np.float32) * scale
            array[...] = (real + 1j * imag).astype(array.dtype)
        else:
            array[...] = real.astype(array.dtype)
    # THE INVERSE EPSILON IS MADE NON-UNIFORM, identically on both sides, and that is
    # NOT decoration. `metal_composition_matrix._epsilon` fills ONE value per
    # component, so on that fixture a kernel that read the inverse permittivity at the
    # WRONG CELL -- the owning thread's instead of the imaged ghost's -- is
    # bit-identical to the right one, and a mutation arming exactly that defect comes
    # back uncaught for a reason that has nothing to do with the kernel. Measured:
    # `len(unique(inverse_epsilon_for('Ez'))) == 1` on `matrix.folded()`. A seeded
    # per-cell volume makes the index observable; the two engines are built from one
    # seed so the material stays bit-equal between them.
    rng_eps = np.random.default_rng(seed ^ 0x5EED)
    for component in ("Ex", "Ey", "Ez"):
        reader = getattr(fields, "inverse_epsilon_for", None)
        if not callable(reader):
            continue
        volume = reader(component)
        if volume is None or not getattr(volume, "flags", None):
            continue
        if not volume.flags.writeable:
            continue
        volume[...] = (0.2 + 0.6 * rng_eps.random(volume.shape)).astype(volume.dtype)

    for state in tuple(getattr(fields, "polarizations", ()) or ()):
        for label in ("P", "P_prev"):
            table = getattr(state, label, {}) or {}
            for component, array in table.items():
                array[...] = rng.normal(
                    0.0, 0.2, size=array.shape).astype(array.dtype)
    return fields


def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    """The composer's OWN live set, never a second model of what a step is."""
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    return tuple(live)


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------
#
# Each row names the module, the fixture builder, and the source mutations the
# ``mutation`` leg arms. A mutation is a (label, old, new) needle applied to the
# shipped source text: it must match EXACTLY ONCE, so a needle that stopped matching
# raises here rather than quietly arming nothing.


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    hits = source.count(old)
    if hits != count:
        raise AssertionError(
            f"the mutation needle {old!r} matches {hits} times, expected {count}; "
            f"an unarmed mutation reports its defect as uncaught")
    return source.replace(old, new)


def _folded_complex(**keywords: Any) -> Tuple[Any, Any]:
    return matrix.folded(complex_storage=True, **keywords)


#: family label -> everything the three legs need.
#:
#: ``probe`` is the expansion artifact a complex family binds its multiply arm from.
#: It is threaded rather than left to the environment so the gate measures the arm the
#: artifact licenses rather than whichever one an env var happened to name.
FAMILIES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "complex_fused_electric_pair",
        "label": "complex fused electric D/E pair",
        "module": complex_fused_electric_pair,
        "plan": complex_fused_electric_pair.plan_metal_complex_fused_electric_pair,
        "coverage":
            complex_fused_electric_pair.metal_complex_fused_electric_pair_coverage,
        "cases": (
            ("all_periodic_3d", lambda: matrix.cart(complex_storage=True)),
            ("wall_x_3d", lambda: matrix.cart(complex_storage=True,
                                              boundaries={"x": "metallic"})),
            ("wall_xyz_3d", lambda: matrix.cart(
                complex_storage=True,
                boundaries={"x": "metallic", "y": "metallic", "z": "metallic"})),
            ("bloch_x_3d", lambda: matrix.cart(complex_storage=True,
                                               k_point=(0.3, 0.0, 0.0))),
        ),
        "mutation_case": "wall_xyz_3d",
        "deposit_case": "wall_x_3d",
        "deposit_component": "Ez",
        "probe_key": "complex_probe",
        "source": lambda plan, mode, target=None: (
            complex_fused_electric_pair.complex_fused_electric_pair_source(
                plan.codes, plan.phased, plan.zero_metal, plan.expansion, mode)),
        "entry": "complex_fused_electric_pair_step",
        "mutations": (
            ("drop_one_wall_clear_row", None,
             "    v1 = at_x ? float2(0.0f, 0.0f) : v1;\n", ""),
            ("seam_reads_the_split_field_auxiliary", None,
             "float2 src0 = c_mul_field_left(v0, e0[ii]);",
             "float2 src0 = c_mul_field_left(u0[ii], e0[ii]);"),
            ("inverse_epsilon_on_the_wrong_component", None,
             "float2 src1 = c_mul_field_left(v1, e1[ii]);",
             "float2 src1 = c_mul_field_left(v1, e0[ii]);"),
        ),
        #: DECLARED EQUIVALENCES, run and reported but NOT scored as must-catch. Each
        #: is a rewrite the fused body makes bit-identical by CONSTRUCTION, and the
        #: leg proves that rather than assuming it: a rewrite believed harmless and
        #: never run is an untested belief.
        "equivalences": (
            ("seam_reloads_the_flux_it_just_stored", None,
             "float2 src0 = c_mul_field_left(v0, e0[ii]);",
             "float2 src0 = c_mul_field_left(f0[ii], e0[ii]);"),
        ),
        "refuted": (
            ("separate_scalars",
             complex_fused_electric_pair.refuted_separate_scalar_source),
            ("split_planes",
             complex_fused_electric_pair.split_plane_pair_signature),
        ),
    },
    {
        "name": "folded_fused_dispersive_pair",
        "label": "folded fused dispersive D/E pair",
        "module": folded_fused_dispersive_pair,
        "plan":
            folded_fused_dispersive_pair.plan_metal_folded_fused_dispersive_pair,
        "coverage":
            folded_fused_dispersive_pair
            .metal_folded_fused_dispersive_pair_coverage,
        "cases": (
            ("fold_y_periodic_2d",
             lambda: matrix.dispersive(matrix.folded())),
            ("fold_y_metallic_2d",
             lambda: matrix.dispersive(
                 matrix.folded(boundaries={"y": "metallic"}))),
            ("fold_y_odd_2d",
             lambda: matrix.dispersive(matrix.folded(phase=-1))),
            ("fold_xy_2d",
             lambda: matrix.dispersive(matrix.folded(axis="XY"))),
        ),
        "mutation_case": "fold_y_periodic_2d",
        "deposit_case": "fold_y_periodic_2d",
        "deposit_component": "Ez",
        "probe_key": None,
        "source": lambda plan, mode, target=2: (
            folded_fused_dispersive_pair.folded_fused_dispersive_pair_source(
                plan.codes, int(target), plan.entries[int(target)].pole_count,
                plan.phases, plan.zero_metal, mode)),
        "entry": "folded_fused_dispersive_pair_component",
        "per_component": 2,
        "mutations": (
            ("drop_the_pole_subtraction", 2,
             "    osource = osource - q0[ii];\n", ""),
            ("near_ghost_reads_the_owner_poles", 2,
             "gj_source = gj_source - q0[gj_i];",
             "gj_source = gj_source - q0[ii];"),
            ("far_ghost_loses_its_parity", 1,
             "float gf_v = -v1;", "float gf_v = v1;"),
        ),
        #: DECLARED EQUIVALENCE, MEASURED RATHER THAN BELIEVED. Reusing the owner
        #: thread's constitutive coefficient at the FAR ghost is bit-identical on
        #: every folded grid this engine builds -- not because the two cells share an
        #: entry in general, but because the HALF-INTEGER profile the E side reads
        #: SATURATES over its outermost two cells, so ``kps_a_h[reflect]`` and
        #: ``kps_a_h[last]`` are the same word. Swept here over 12 folded extents x 3
        #: PML thicknesses: 72 of 72 half-integer entries identical, and 72 of 72
        #: INTEGER entries (the B/H side's) DIFFERENT. The reload is kept because it
        #: is the destination's own entry by construction; this leg records that the
        #: D/E seam cannot observe it.
        "equivalences": (
            ("far_ghost_reuses_the_owner_coefficient_half_integer_saturates", 1,
             "int gf_coord = (nyi - 1);", "int gf_coord = j;"),
        ),
        "refuted": (
            ("separate_scalars",
             folded_fused_dispersive_pair.refuted_separate_scalar_source),
        ),
    },
    {
        "name": "folded_complex_fused_pair",
        "label": "folded complex fused D/E pair",
        "module": folded_complex_fused_pair,
        "plan": folded_complex_fused_pair.plan_metal_folded_complex_fused_pair,
        "coverage":
            folded_complex_fused_pair.metal_folded_complex_fused_pair_coverage,
        "cases": (
            ("fold_y_periodic_2d", lambda: _folded_complex()),
            ("fold_y_metallic_2d",
             lambda: _folded_complex(boundaries={"y": "metallic"})),
            ("fold_y_odd_2d", lambda: _folded_complex(phase=-1)),
            ("fold_xy_mixed_2d",
             lambda: _folded_complex(axis="XY", phase=(1, -1))),
        ),
        "mutation_case": "fold_y_periodic_2d",
        "deposit_case": "fold_y_periodic_2d",
        "deposit_component": "Ez",
        "probe_key": "folded_complex_probe",
        "source": lambda plan, mode, target=None: (
            folded_complex_fused_pair.folded_complex_fused_pair_source(
                plan.codes, plan.phased, plan.zero_metal, plan.expansion, mode)),
        "entry": "folded_complex_fused_pair_step",
        "mutations": (
            ("near_ghost_takes_the_far_parity", None,
             "float2 g0_j_v = c_mul(mp1, v0);",
             "float2 g0_j_v = c_mul(fp1, v0);"),
            ("ghost_constitutive_reads_the_owner_cell", None,
             "float2 g0_j_src = c_mul_field_left(g0_j_v, e0[g0_j_i]);",
             "float2 g0_j_src = c_mul_field_left(g0_j_v, e0[ii]);"),
        ),
        #: THE SAME MEASURED EQUIVALENCE the dispersive D/E family records: the
        #: half-integer PML profile saturates over its outermost two cells, so the far
        #: ghost's own entry and its owner's are the same word on every folded grid.
        "equivalences": (
            ("far_ghost_reuses_the_owner_coefficient_half_integer_saturates", None,
             "float g1_f_kp = kp1[(nyi - 1)], g1_f_km = km1[(nyi - 1)];",
             "float g1_f_kp = kp_1, g1_f_km = km_1;"),
        ),
        "refuted": (
            ("separate_scalars",
             folded_complex_fused_pair.refuted_separate_scalar_source),
        ),
    },
    {
        "name": "folded_beta_complex_fused_pair",
        "label": "folded beta complex fused D/E pair",
        "module": folded_beta_complex_fused_pair,
        "plan":
            folded_beta_complex_fused_pair
            .plan_metal_folded_beta_complex_fused_pair,
        "coverage":
            folded_beta_complex_fused_pair
            .metal_folded_beta_complex_fused_pair_coverage,
        "cases": (
            ("fold_y_beta_2d", lambda: _folded_complex(beta=0.3)),
            ("fold_y_beta_metallic_2d",
             lambda: _folded_complex(beta=0.3, boundaries={"y": "metallic"})),
            ("fold_y_beta_negative_2d", lambda: _folded_complex(beta=-0.685)),
        ),
        "mutation_case": "fold_y_beta_2d",
        "deposit_case": "fold_y_beta_2d",
        "deposit_component": "Ez",
        "probe_key": "folded_complex_probe",
        "beta_probe_key": "beta_probe",
        "source": lambda plan, mode, target=None: (
            folded_beta_complex_fused_pair.folded_beta_complex_fused_pair_source(
                plan.codes, plan.phased, plan.zero_metal, plan.expansion, mode)),
        "entry": "folded_beta_complex_fused_pair_step",
        "mutations": (
            ("drop_the_beta_insert", None,
             "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);\n", ""),
            ("beta_signs_swapped", None,
             "curl1 = curl1 - c_mul(float2(bmr, bmi), a);",
             "curl1 = curl1 - c_mul(float2(bpr, bpi), a);"),
            ("near_ghost_takes_the_far_parity", None,
             "float2 g0_j_v = c_mul(mp1, v0);",
             "float2 g0_j_v = c_mul(fp1, v0);"),
        ),
        "refuted": (),
    },
    {
        "name": "beta_fused_electric_pair",
        "label": "beta fused electric D/E pair",
        "module": beta_fused_electric_pair,
        "plan": beta_fused_electric_pair.plan_metal_beta_fused_electric_pair,
        "coverage":
            beta_fused_electric_pair.metal_beta_fused_electric_pair_coverage,
        "cases": (
            ("beta_2d", lambda: matrix.flat(beta=0.33)),
            ("beta_negative_2d", lambda: matrix.flat(beta=-0.685)),
            ("beta_metallic_2d",
             lambda: matrix.flat(beta=0.33, boundaries={"x": "metallic"})),
        ),
        "mutation_case": "beta_metallic_2d",
        "deposit_case": "beta_2d",
        "deposit_component": "Ez",
        "probe_key": None,
        "source": lambda plan, mode, target=None: (
            beta_fused_electric_pair.beta_fused_electric_pair_source(
                plan.codes, plan.zero_metal, mode)),
        "entry": "beta_fused_electric_pair_step",
        "mutations": (
            ("drop_the_beta_insert", None,
             "    curl0 = curl0 - (beta_plus * b);\n", ""),
            ("beta_signs_swapped", None,
             "curl1 = curl1 - (beta_minus * a);",
             "curl1 = curl1 - (beta_plus * a);"),
            ("seam_reads_the_split_field_auxiliary", None,
             "float src2 = v2 * ie2[ii];", "float src2 = u2[ii] * ie2[ii];"),
        ),
        "refuted": (
            ("separate_scalars",
             beta_fused_electric_pair.refuted_separate_scalar_source),
        ),
    },
    {
        "name": "folded_beta_real_fused_pair",
        "label": "folded beta real fused D/E pair",
        "module": folded_beta_real_fused_pair,
        "plan": folded_beta_real_fused_pair.plan_metal_folded_beta_real_fused_pair,
        "coverage":
            folded_beta_real_fused_pair.metal_folded_beta_real_fused_pair_coverage,
        "cases": (
            ("fold_y_beta_2d", lambda: matrix.folded(beta=0.3)),
            ("fold_y_beta_metallic_2d",
             lambda: matrix.folded(beta=0.3, boundaries={"y": "metallic"})),
            ("fold_y_beta_odd_2d", lambda: matrix.folded(beta=0.3, phase=-1)),
        ),
        "mutation_case": "fold_y_beta_2d",
        "deposit_case": "fold_y_beta_2d",
        "deposit_component": "Ez",
        "probe_key": None,
        "source": lambda plan, mode, target=None: (
            folded_beta_real_fused_pair.folded_beta_real_fused_pair_source(
                plan.codes, plan.phases, plan.zero_metal, mode)),
        "entry": "folded_beta_real_fused_pair_step",
        "mutations": (
            ("drop_the_beta_insert", None,
             "    curl0 = curl0 - (beta_plus * b);\n", ""),
            ("far_ghost_loses_its_parity", None,
             "float gf_v = -v1;", "float gf_v = v1;"),
        ),
        #: THE SAME MEASURED EQUIVALENCE the other two D/E folded families record.
        "equivalences": (
            ("far_ghost_reuses_the_owner_coefficient_half_integer_saturates", None,
             "float gf_kp = kp1[(nyi - 1)], gf_km = km1[(nyi - 1)];",
             "float gf_kp = kp_1, gf_km = km_1;"),
        ),
        "refuted": (),
    },
    {
        "name": "folded_beta_real_fused_magnetic_pair",
        "label": "folded beta real fused B/H pair",
        "module": folded_beta_real_fused_magnetic_pair,
        "plan":
            folded_beta_real_fused_magnetic_pair
            .plan_metal_folded_beta_real_fused_magnetic_pair,
        "coverage":
            folded_beta_real_fused_magnetic_pair
            .metal_folded_beta_real_fused_magnetic_pair_coverage,
        "cases": (
            ("fold_y_beta_2d", lambda: matrix.folded(beta=0.3)),
            ("fold_y_beta_metallic_2d",
             lambda: matrix.folded(beta=0.3, boundaries={"y": "metallic"})),
            ("fold_y_beta_odd_2d", lambda: matrix.folded(beta=0.3, phase=-1)),
        ),
        "mutation_case": "fold_y_beta_2d",
        "deposit_case": "fold_y_beta_2d",
        # AN Hz SOURCE IS REFUSED BY THE ENGINE ON AN EVEN Y PLANE -- that plane makes
        # Ey, Hx and Hz ODD, and a point source with no extent across it can carry no
        # odd parity at all. Hy is the component the plane leaves EVEN, so this is the
        # engine's own answer to "which magnetic component can a point deposit on this
        # fold declare", not a choice made to get a green leg.
        "deposit_component": "Hy",
        "probe_key": None,
        "source": lambda plan, mode, target=None: (
            folded_beta_real_fused_magnetic_pair
            .folded_beta_real_fused_magnetic_pair_source(
                plan.codes, plan.phases, plan.zero_metal, mode)),
        "entry": "folded_beta_real_fused_magnetic_pair_step",
        "mutations": (
            ("drop_the_beta_insert", None,
             "    curl1 = curl1 - (beta_minus * a);\n", ""),
            ("beta_signs_swapped", None,
             "curl0 = curl0 - (beta_plus * b);",
             "curl0 = curl0 - (beta_minus * b);"),
            ("far_ghost_loses_its_parity", None,
             "float g0_j_v = -v0;", "float g0_j_v = v0;"),
        ),
        #: DECLARED EQUIVALENCE, MEASURED. The NEAR ghost sits at stored 0 of a folded
        #: axis and its owner at stored 2, and a folded axis carries its absorber on
        #: the HIGH face only (``stepping._require_consistent_pml``) -- so both cells
        #: are outside the layer and every PML entry there is the identity 1.0, on the
        #: integer AND the half-integer lattice alike (measured on
        #: ``matrix.folded()``: kps_y[0] == kps_y[2] == kps_y_h[0] == kps_y_h[2] ==
        #: 1.0). The reload is kept because it is the destination's own entry by
        #: construction; this leg records that no folded grid this engine builds can
        #: observe it.
        "equivalences": (
            ("near_ghost_reuses_the_owner_coefficient_low_end_is_outside_the_pml",
             None,
             "float g1_n_kp = kp1[0], g1_n_km = km1[0];",
             "float g1_n_kp = kp_1, g1_n_km = km_1;"),
        ),
        "refuted": (),
    },
)


def probe_arguments(spec: Mapping[str, Any],
                    probes: Mapping[str, Any]) -> Dict[str, Any]:
    """The expansion artifacts this family's plan builder takes, by keyword."""
    out: Dict[str, Any] = {}
    if spec.get("probe_key"):
        out["probe"] = probes[spec["probe_key"]]
    if spec.get("beta_probe_key"):
        out["beta_probe"] = probes[spec["beta_probe_key"]]
    return out


def build_plan(spec: Mapping[str, Any], fields: Any, pml: Any,
               residency: Residency, probes: Mapping[str, Any],
               sources: Sequence[Any] = (),
               functions: Optional[Mapping[Any, Any]] = None) -> Any:
    kwargs = probe_arguments(spec, probes)
    if functions is not None:
        kwargs["functions"] = functions
    return spec["plan"](fields, pml, sources=tuple(sources),
                        residency=residency, **kwargs)


# ---------------------------------------------------------------------------
# Leg A / B: the walk
# ---------------------------------------------------------------------------

def metal_step(fields: Any, pml: Any, plan: Any, owned: Sequence[str],
               residency: Residency, live: Sequence[str]) -> None:
    """One complete step, on the device where the plan owns the pass.

    A pass no plan owns runs on the array path and is bracketed with an explicit
    ``sync_out`` / ``sync_in``. Without that bracket the next device launch would read
    the mirror's stale bytes, which is smooth, plausible and wrong.

    ``owned`` is the plan's ``replaces_sub_steps`` and NOT the slot it dispatches at:
    deriving the skip set from the dispatch would leave the carried in-seam passes
    running on the host on top of the carry the kernel already performed — which for
    the wall clear is IDEMPOTENT and would hide a dropped carry entirely.
    """
    skip = set(owned)
    for name in live:
        if name in skip:
            if name == owned[0]:
                plan.run()
            continue
        residency.sync_out()
        ARRAY_PATH[name](fields, pml)
        residency.sync_in()


def run_case(spec: Mapping[str, Any], build: Callable[[], Tuple[Any, Any]],
             probes: Mapping[str, Any], steps: int = STEPS,
             functions: Optional[Mapping[Any, Any]] = None) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
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
                    **probe_arguments(spec, probes)).reasons)}

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    before = frozen(actual)
    residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
        metal_step(actual, actual_pml, plan, plan.replaces_sub_steps, residency,
                   live)
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
                   and plan.launches == plan.launches_per_run * len(per_step))
    return {
        "passed": bool(identical and clean and launches_ok and not still),
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
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "live_passes": list(live),
        "replaces": list(plan.replaces_sub_steps),
        "shape": list(plan.shape),
        "mirrors": len(residency.names),
    }


def leg_identity(spec: Mapping[str, Any],
                 probes: Mapping[str, Any]) -> Dict[str, Any]:
    """A: the full mechanism. Every case must be bit-identical at every step."""
    rows: Dict[str, Any] = {}
    for label, build in spec["cases"]:
        started = time.time()
        rows[label] = run_case(spec, build, probes)
        log(f"    identity  {spec['name']:38s} {label:24s} "
            f"passed={rows[label]['passed']} "
            f"words={rows[label].get('differing_words')} "
            f"launches={rows[label].get('launches')} "
            f"({time.time() - started:5.1f} s)")
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows}


def _mutated_functions(spec: Mapping[str, Any], plan: Any, target: Optional[int],
                       old: str, new: str) -> Dict[Any, Any]:
    """The shipped source with ONE line replaced, compiled and keyed as the plan keys.

    THE MUTATION SEAM IS THE PLAN'S OWN ``functions`` ARGUMENT, so the broken kernel
    reaches the device through the shipped builder rather than through a second launch
    path this file would own. A family that dispatches per component is keyed
    ``(mode, axis)``; the single-launch families are keyed by mode alone.
    """
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    mode = shaders.CONTRACT_OFF
    per_component = spec.get("per_component")
    if target is not None:
        per_component = int(target)
    source = needle(spec["source"](plan, mode, per_component), old, new)
    function = getattr(compile_source(source), spec["entry"])
    if per_component is not None:
        # A PER-COMPONENT FAMILY IS KEYED (mode, axis) AND THE OTHER TWO COMPONENTS
        # MUST STAY SHIPPED. Handing back only the mutated key leaves the plan
        # builder to compile the rest itself, which is exactly what makes the leg
        # measure ONE line rather than one component.
        return {(mode, per_component): function}
    return {mode: function}


def leg_mutation(spec: Mapping[str, Any],
                 probes: Mapping[str, Any]) -> Dict[str, Any]:
    """B: the DEGENERATE mechanism. Every armed mutation MUST diverge.

    A mutation that does not diverge is reported as UNCAUGHT and fails the leg — it
    means the line it broke is not reached on this fixture, and a leg that scored it
    as a pass would be certifying an untested line.
    """
    build = dict(spec["cases"])[spec["mutation_case"]]
    rows: Dict[str, Any] = {}
    for label, target, old, new in spec["mutations"]:
        reference, reference_pml = build()
        residency = Residency()
        seed_state(reference, SEED)
        plan = build_plan(spec, reference, reference_pml, residency, probes)
        if plan is None:
            rows[label] = {"passed": False, "reason": "the pair was refused"}
            continue
        functions = _mutated_functions(spec, plan, target, old, new)
        result = run_case(spec, build, probes, functions=functions)
        diverged = not result.get("bit_identical", True)
        rows[label] = {
            "passed": bool(diverged and result.get("launches")),
            "caught": diverged,
            "first_divergence": result.get("first_divergence"),
            "differing_words": result.get("differing_words"),
            "launches": result.get("launches"),
        }
        log(f"    mutation  {spec['name']:38s} {label:40s} "
            f"caught={diverged} words={result.get('differing_words')}")

    # THE DECLARED EQUIVALENCES, run and reported but scored the OTHER way: each must
    # be bit-identical, because each is a rewrite the fused body makes harmless BY
    # CONSTRUCTION. `seam_reloads_the_flux_it_just_stored` is the one this tranche
    # found: the kernel stores `f[ii] = v` immediately before the constitutive half
    # reads the register, so reloading the stored word is the same word. Running it
    # is what turns that from a belief into a measurement -- and it is NOT counted as
    # a caught mutation, which is what "a case where B does not diverge is VACUOUS"
    # requires.
    equivalences: Dict[str, Any] = {}
    for label, target, old, new in spec.get("equivalences", ()):
        reference, reference_pml = build()
        residency = Residency()
        seed_state(reference, SEED)
        plan = build_plan(spec, reference, reference_pml, residency, probes)
        if plan is None:
            equivalences[label] = {"passed": False, "reason": "the pair was refused"}
            continue
        functions = _mutated_functions(spec, plan, target, old, new)
        result = run_case(spec, build, probes, functions=functions)
        identical = bool(result.get("bit_identical"))
        equivalences[label] = {
            "passed": identical, "bit_identical": identical,
            "differing_words": result.get("differing_words"),
            "launches": result.get("launches"),
        }
        log(f"    equivalent{spec['name']:38s} {label:40s} "
            f"identical={identical} words={result.get('differing_words')}")
    return {"passed": (all(row["passed"] for row in rows.values())
                       and all(row["passed"] for row in equivalences.values())),
            "mutations": rows, "declared_equivalences": equivalences}


# ---------------------------------------------------------------------------
# Leg C: the deposit, with and WITHOUT the repair bracket
# ---------------------------------------------------------------------------

def _volume_source(fields: Any, component: str,
                   size: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Any:
    """A REAL engine source, so ``field_type`` is the engine's own classification."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=size,
                        envelope=ContinuousEnvelope(frequency=1.0))


def _withdraw(sources: Sequence[Any], fields: Any) -> None:
    for source in sources:
        hook = getattr(source, "withdraw", None)
        if callable(hook):
            hook(fields)


def run_deposit_case(spec: Mapping[str, Any], probes: Mapping[str, Any],
                     repair: bool = True, steps: int = STEPS) -> Dict[str, Any]:
    """Complete driver steps with an in-seam source, byte compared.

    THE WALK IS THE DRIVER'S, NOT ``metal_step``'S:

      * the INJECTION is performed between the fused launch and the repair, because
        that is where the driver performs it;
      * every in-seam pass the driver runs UNCONDITIONALLY — the symmetry fill, the
        wall clear, the far ghost fill — runs ON THE HOST after the injection, even
        where the kernel carries it inline, because ``driver.step`` runs them behind
        no consult. That is what lets the repair read a final field;
      * the two slots are run as the two separate consults they are.

    ``repair=False`` builds the SHIPPED plan for an EMPTY source list — the bare pair
    plus the ``NoopPlan`` — and injects anyway. That is the NO-MECHANISM control and
    it must diverge.
    """
    pair = "B" if spec["deposit_component"].startswith("H") else "D"
    curl_slot, update_slot = (("step_B", "update_H") if pair == "B"
                              else ("step_D", "update_E"))
    build = dict(spec["cases"])[spec["deposit_case"]]
    reference, reference_pml = build()
    actual, actual_pml = build()
    seed_state(reference, SEED)
    seed_state(actual, SEED)
    drift = compare(reference, actual)
    if drift:
        return {"passed": False, "reason": "the two builds are not identical"}

    reference_sources = (_volume_source(reference, spec["deposit_component"]),)
    actual_sources = (_volume_source(actual, spec["deposit_component"]),)
    in_seam = deposit_repair.in_seam_sources(actual_sources, pair)
    if len(in_seam) != 1:
        return {"passed": False,
                "reason": f"the source is not in the {pair} seam; the walk would "
                          f"inject nothing and compare a no-op with a no-op"}

    residency = Residency()
    # THE SHIPPED COMPOSER BUILDS THE PLAN, not this file: reaching for the plan
    # builder directly would test the kernel and skip the wiring, and the wiring is
    # what this leg exists for.
    composed = metal_launch.plan_step(
        actual, actual_pml, residency=residency,
        sources=(actual_sources if repair else ()), fuse=True,
        # `plan_step` threads these into `StepContext.extra` and the arms hand them
        # to their own `expansion_from_probe`, which takes the RECORD. Passing a PATH
        # here returns None and every complex arm refuses BY NAME -- the composer then
        # selects nothing and this leg would report "the composer did not fuse the
        # seam" for a reason that has nothing to do with the seam.
        complex_probe=probes["complex_probe"],
        beta_probe=probes["beta_probe"],
        folded_complex_probe=probes["folded_complex_probe"],
        cylindrical_complex_probe=probes["cylindrical_complex_probe"])
    leading = composed.plans.get(curl_slot)
    trailing = composed.plans.get(update_slot)
    if leading is None or trailing is None:
        return {"passed": False, "reason": "the composer did not fuse the seam",
                "selected": dict(composed.selected),
                "refusals": [r for key, value in composed.reasons.items()
                             if key.startswith("fused_pair") for r in value][:6]}
    installed = (type(leading).__name__, type(trailing).__name__)
    if repair:
        installed_ok = installed == ("LeadingRepairPlan", "TrailingRepairPlan")
    else:
        installed_ok = (installed[0] != "LeadingRepairPlan"
                        and installed[1] == "NoopPlan")
    inner = getattr(leading, "absorbed_by", leading)
    if composed.selected.get(curl_slot) != spec["label"]:
        return {"passed": False,
                "reason": f"another product won {curl_slot}: "
                          f"{composed.selected.get(curl_slot)!r}"}

    live = live_passes(actual, actual_pml)
    before = frozen(actual)
    dt = float(actual.grid.dt)

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        when = (step - 1) * dt
        _withdraw(reference_sources, reference)
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
            if name == curl_slot:
                for source in reference_sources:
                    source.inject(reference, when)

        _withdraw(actual_sources, actual)
        for name in live:
            if name == curl_slot:
                residency.sync_in()
                leading.run()
                residency.sync_out()
                for source in actual_sources:
                    source.inject(actual, when)
            elif name == update_slot:
                trailing.run()
            else:
                ARRAY_PATH[name](actual, actual_pml)

        difference = compare(reference, actual)
        per_step.append({"step": step, "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items()))})
        if difference:
            break

    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    repairs = int(getattr(leading, "repairs", 0))
    launches_ok = (inner.runs == len(per_step)
                   and inner.launches == inner.launches_per_run * len(per_step))
    return {
        "passed": bool(identical and launches_ok and not still and installed_ok
                       and repairs > 0),
        "repair_wired": repair,
        "installed_plans": list(installed),
        "installed_as_expected": installed_ok,
        "deposit_points_repaired": repairs,
        "bit_identical": identical,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "arrays_that_never_moved": still,
        "seam": pair,
        "source_component": spec["deposit_component"],
        "runs": inner.runs, "launches": inner.launches,
        "launches_per_run": inner.launches_per_run,
        "selected": {slot: composed.selected.get(slot)
                     for slot in (curl_slot, update_slot)},
    }


def leg_deposit(spec: Mapping[str, Any],
                probes: Mapping[str, Any]) -> Dict[str, Any]:
    """C: the in-seam deposit, bracketed and unbracketed.

    The bracketed walk must be BIT-IDENTICAL and must have repaired at least one
    point; the unbracketed one MUST DIVERGE. A control that agreed would mean the
    deposit never reached the compared state and the whole leg would be vacuous.
    """
    carried = run_deposit_case(spec, probes, repair=True)
    control = run_deposit_case(spec, probes, repair=False)
    diverged = not control.get("bit_identical", True)
    log(f"    deposit   {spec['name']:38s} carried={carried['passed']} "
        f"words={carried.get('differing_words')} "
        f"repaired={carried.get('deposit_points_repaired')} "
        f"| control diverged={diverged} "
        f"words={control.get('differing_words')}")
    return {"passed": bool(carried.get("passed") and diverged),
            "vacuous": not diverged,
            "carried": carried, "null_control": control}


# ---------------------------------------------------------------------------
# The binding ceiling
# ---------------------------------------------------------------------------

def leg_binding_ceiling() -> Dict[str, Any]:
    """Every refuted signature must FAIL to compile on this toolchain.

    That is what turns each family's ``SEPARATE_SCALAR_BINDINGS`` from arithmetic on a
    number in a docstring into a measurement. The SHIPPED signatures are compiled here
    too and must SUCCEED, so the leg cannot pass by refusing everything.
    """
    rows: Dict[str, Any] = {}
    for spec in FAMILIES:
        for label, builder in spec.get("refuted", ()):
            key = f"{spec['name']}/{label}"
            try:
                compile_source(builder())
                rows[key] = {"passed": False, "compiled": True}
            except Exception as error:  # noqa: BLE001 - the failure IS the result
                rows[key] = {"passed": True, "compiled": False,
                             "error": str(error).strip().splitlines()[-1][:160]}
            log(f"    ceiling   {key:60s} refused={not rows[key]['compiled']}")
    return {"passed": all(row["passed"] for row in rows.values()), "signatures": rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # THE BUDGET IS A MODULE CONSTANT SO EVERY LEG READS ONE NUMBER. A `--steps` that
    # only reached `run_case` would leave the mutation and deposit legs on the default
    # and the record would name a budget two of the three legs never ran.
    global STEPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--only", default=None,
                        help="run one family by name")
    parser.add_argument("--legs", default="identity,mutation,deposit,ceiling")
    parser.add_argument("--steps", type=int, default=STEPS)
    arguments = parser.parse_args()
    STEPS = int(arguments.steps)

    environment = matrix.prepare_environment()
    # THE PROBES ARE LOADED RECORDS, NOT PATHS. Every family's plan builder passes its
    # ``probe`` argument straight to that family's ``expansion_from_probe``, which
    # takes the RECORD -- handing it a path returns None and the family refuses BY
    # NAME, which would make this gate measure the artifact's absence rather than the
    # kernel's arithmetic. Each record is read through its OWN family's loader, so the
    # gate binds the same artifact the shipped predicate binds.
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
    for key in ("complex_probe", "beta_probe", "folded_complex_probe"):
        if paths[key] is None:
            raise SystemExit(
                f"no {key} artifact; the complex and beta families would refuse BY "
                f"NAME and this gate would measure their absence rather than their "
                f"arithmetic")
    probes = {
        "complex_probe": _complex.load_expansion_probe(),
        "beta_probe": _beta.load_expansion_probe(),
        "folded_complex_probe": _folded.load_expansion_probe(),
        "cylindrical_complex_probe": _cylindrical.load_expansion_probe(),
    }
    for key in ("complex_probe", "beta_probe", "folded_complex_probe"):
        if probes[key] is None:
            raise SystemExit(f"the {key} artifact at {paths[key]!r} did not load")

    legs = tuple(part.strip() for part in arguments.legs.split(",") if part.strip())
    started = time.time()
    record: Dict[str, Any] = {
        "gate": "metal_tranche7_fused_pairs",
        "host": {"platform": sys.platform, "torch": None},
        "steps": arguments.steps,
        "seed": SEED,
        "probes": paths,
        "families": {},
    }
    try:
        import torch  # noqa: PLC0415

        record["host"]["torch"] = torch.__version__
        record["host"]["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover - reported, never fatal
        record["host"]["mps_available"] = None

    for spec in FAMILIES:
        if arguments.only and spec["name"] != arguments.only:
            continue
        log(f"  {spec['name']}")
        entry: Dict[str, Any] = {}
        if "identity" in legs:
            entry["identity"] = leg_identity(spec, probes)
        if "mutation" in legs:
            entry["mutation"] = leg_mutation(spec, probes)
        if "deposit" in legs and spec.get("label"):
            entry["deposit"] = leg_deposit(spec, probes)
        entry["passed"] = all(value.get("passed") for value in entry.values()
                              if isinstance(value, dict))
        record["families"][spec["name"]] = entry

    if "ceiling" in legs:
        record["binding_ceiling"] = leg_binding_ceiling()

    # THE TOOLCHAIN AND THE POLICY, at the top level and spelled the way every other
    # Metal gate spells them, because `mint_metal_weld` reads these four keys to build
    # a weld's `host` and `subnormal_policy` lines. Without them the minted entry says
    # "?" for the torch version, the Metal frontend and the subnormal policy -- a
    # record that passes `test_every_metal_weld_carries_its_metadata` (a "?" string is
    # truthy) while stating nothing, which is worse than a red test.
    record["torch_version"] = record["host"].get("torch")
    record["numpy_version"] = np.__version__
    record["metal_frontend"] = metal_frontend_version()
    record["subnormal_policy"] = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")
    record["elapsed_s"] = round(time.time() - started, 2)
    record["verdict"] = "PASS" if all(
        value.get("passed") for value in record["families"].values()) and (
        record.get("binding_ceiling", {}).get("passed", True)) else "FAIL"
    # THE PROVENANCE STAMP, and why its absence was a defect rather than a style
    # difference. Every other Metal gate calls this one line; this gate did not, so
    # its artifact carried no `imported_source_sha256` and no `canonical_verdict`.
    # Both are what the weld tooling reads: `mint_metal_weld` and
    # `rebind_metal_welds` bind the CURATED paths out of `imported_source_sha256`
    # and take the verdict from `canonical_verdict`/`release`. Without them the run
    # produced a PASS nothing could bind -- which is exactly how this gate presented
    # to `build_fusion_matrix`'s RELEASE_BINDING floor (its seven products credited,
    # named in no bindable row) and to `test_every_metal_family_is_welded` (a gate
    # with no weld). Adding it records the bytes THIS process imported; it does not
    # change a single measurement above.
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(record)  # bytes THIS process imported; see gate_provenance
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
    # ROUTED THROUGH THE RELEASE RUNNER, as every other `gate_metal_*.py` is. Called
    # bare, `main()` writes a verdict with no `release` block and no runtime source
    # weld, and `metal_gate_runner`'s docstring says why that is not a releasable
    # artifact: the gate's own `source_sha256` "need not name the source path the
    # process imported". The runner re-derives that set, welds it against
    # `source_sha256.txt`, and only then records `release.released`.
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
