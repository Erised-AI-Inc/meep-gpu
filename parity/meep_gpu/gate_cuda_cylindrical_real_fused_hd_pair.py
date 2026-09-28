#!/usr/bin/env python3
"""Device byte gate for the CUDA REAL m = 0 Dcyl H->D two-launch product.

WHAT IS BEING CERTIFIED. ``meep_gpu/cuda_kernels/cylindrical_real_fused_hd_pair.py``:
launch 1 (slot ``update_H``) computes the certified ``update_H_pml_real`` recompute at
the thread's own cell into launch-local scratch and RECOMPUTES it at the one-cell
backward radial neighbour to form the pre-``cumsum`` increment of
``stepping.cylindrical_rderiv_prefix(Hy_new, 0.5)``; ``xp.cumsum`` runs untouched on the
array path (it IS the oracle on this backend -- a column-serial device scan is measured
NOT to reproduce it, per case, below); the ``H``/``f_w_H`` bindings rotate; launch 2 is
the certified ``cyl_step_D_pml_real`` with the prefix handed in. Three launches per
seam where the D side takes seven today (two kernels plus the prefix's five statements).

THE ARITHMETIC IS MEASURED, NOT CHOSEN, and its measurement is re-run here as a leg:
the increment's spelling is ``(Hy_new[i]*w[i] - Hy_new[i-1]*w[i-1]) / d[i-1]`` with IEEE
``/`` under ``--fmad=false``; the reciprocal multiply, the distributed divide and the
unguarded build are ARMED mutations that must be CAUGHT, and the swapped multiply
operand is an armed NULL (IEEE multiplication commutes). The complex sibling inverts
the divide's spelling; this gate carries the inversion as a mutation rather than a
comment.

THE CLAIM is per COMPLETE DRIVER STEP, as uint32 words over every stored volume, against
FOUR references stepped from one seed on every fixture: the array path, the certified
singles (``update_H_pml_real`` then ``cyl_step_D_pml_real``), the composition the
composer installs on these rows TODAY (the released real cylindrical B->H and D->E
pairs, driven through their own entry points), and the four slots dispatched unfused.
Every weld step must advance THREE independent counters -- the family tally, the memo's
kernel count and a wrapper around ``cupy.cumsum`` -- so nothing passes by not executing.

STANDALONE. The family is not in ``fused_pairs``' tables; the wiring change is a patch
in the lane directory, not applied. Every arrangement here is planned from arrays by
this gate and the array path is driven by this gate. The arbitration leg measures the
composer as shipped (this product absent; the two released cylindrical pairs hold the
slots) and with the wiring rows patched in-process (this product refused BY NAME on its
own ``INSTALLABLE = False``), then removes them.

LEGS: see ``cylindrical_hd_gate_common.LEG_GROUPS``. Rule 7: a flushed line per case.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = str(next(parent for parent in Path(_HERE).parents
                     if (parent / "meep_gpu" / "cuda_kernels").is_dir()))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cylindrical_hd_gate_common as common  # noqa: E402
from cylindrical_hd_gate_common import (Adapter, case_rng, cp, differing,  # noqa: E402
                                        encodes_ascii, float_words, log,
                                        mutation_needles_resolve)

#: ``measure_predicate_coverage`` digests this package into every lifted row.
SUBJECT_PACKAGE = "cuda_kernels"
SEED = 20260907
FAMILY_NAME = "cuda_cylindrical_real_fused_hd_pair"
BLOCK_KEY = "cuda_cylindrical_real_fused_hd_pair_gate"
CELL_ARMS: Tuple[str, str] = ("cuda_constitutive/ordinary", "cuda_cylindrical/cylindrical")

#: The synthetic fixtures. Both z terminations, odd extents, a wide radial extent, a
#: tall z, ONE-COLUMN rows (three corpus rows have nz = 1), a corpus-sized extent, a
#: thin absorber and the inexact Courant beside the exact one.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "r16_z20_metallic", "shape": (16, 1, 20), "z_kind": "metallic"},
    {"label": "r16_z20_periodic", "shape": (16, 1, 20), "z_kind": "periodic"},
    {"label": "r16_z20_metallic_c05", "shape": (16, 1, 20), "z_kind": "metallic",
     "courant": 0.5},
    {"label": "r40_z16_metallic", "shape": (40, 1, 16), "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "signed_zero")},
    {"label": "r64_z12_metallic", "shape": (64, 1, 12), "z_kind": "metallic"},
    {"label": "r9_z17_metallic", "shape": (9, 1, 17), "z_kind": "metallic"},
    {"label": "r9_z17_periodic", "shape": (9, 1, 17), "z_kind": "periodic"},
    {"label": "r12_z48_metallic", "shape": (12, 1, 48), "z_kind": "metallic"},
    # ONE-COLUMN ROWS. A one-cell z axis must be PERIODIC (Grid refuses a metallic
    # unit axis by name: MEEP overrides a unit direction to periodic), which is what
    # the three one-column corpus rows carry.
    {"label": "r80_z1_periodic", "shape": (80, 1, 1), "z_kind": "periodic"},
    {"label": "r16_z20_thin_metallic", "shape": (16, 1, 20), "z_kind": "metallic",
     "thin_absorber": True},
    {"label": "r120_z120_metallic", "shape": (120, 1, 120), "z_kind": "metallic"},
)
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("r16_z20_metallic", "r9_z17_periodic",
                                         "r64_z12_metallic")
REDUCED_LABELS: Tuple[str, ...] = ("r16_z20_metallic", "r9_z17_periodic", "r12_z48_metallic")

#: ``{name: {"old", "new", "why", "live_on", ...}}`` -- device-source rewrites of
#: launch 1, each of which MUST make a scored fixture diverge unless declared null.
DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "reciprocal_multiply_divide": {
        "old": "    increment[idx] = diff / divisor[i - 1];",
        "new": "    increment[idx] = diff * (1.0f / divisor[i - 1]);",
        "why": ("numpy's a * (float32(1)/d) -- the spelling the Metal verdict prescribes "
                "for the COMPLEX scan and the one a reader would copy; on this backend the "
                "array path is CuPy's true_divide and the two differ on a quarter of words "
                "(1,816,293 / 1,802,148 of 7,193,552 in the probe)"),
        "live_on": "every fixture",
    },
    "distributed_divide": {
        "old": "    increment[idx] = diff / divisor[i - 1];",
        "new": "    increment[idx] = (wi / divisor[i - 1]) - (wim1 / divisor[i - 1]);",
        "why": "the divide distributed over the subtract: two roundings where the array path makes one",
        "live_on": "every fixture",
    },
    "fmad_default_build": {
        "old": None, "new": None, "options": (),
        "why": ("the same source built WITHOUT --fmad=false: NVRTC contracts the "
                "increment's subtract with one product (and the constitutive's two "
                "accumulations); 1,168,184 / 1,149,775 words in the probe"),
        "live_on": "every fixture",
    },
    "forward_difference": {
        "old": ("    raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);\n"
                "    float wi = own_h[1] * weights[i];\n"
                "    float wim1 = halo_h[1] * weights[i - 1];"),
        "new": ("    raw_update_H_cell((i + 1 < nx) ? idx + nyz : idx, weld, halo_h, halo_w);\n"
                "    float wi = own_h[1] * weights[i];\n"
                "    float wim1 = halo_h[1] * weights[(i + 1 < nx) ? i + 1 : i];"),
        "why": "the halo taken FORWARD along r: the wrong difference, converged and smooth",
        "live_on": "every fixture",
    },
    "neighbour_reads_stored_H": {
        "old": "    raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);",
        "new": ("    halo_h[0] = weld.Hx[idx - nyz]; halo_h[1] = weld.Hy[idx - nyz]; "
                "halo_h[2] = weld.Hz[idx - nyz];\n"
                "    halo_w[0] = 0.0f; halo_w[1] = 0.0f; halo_w[2] = 0.0f;"),
        "why": ("the neighbour's Hy read from the PRE-launch volume instead of recomputed: "
                "the array path differences update_H's OUTPUT, so the increment is one "
                "sub-step behind on one of its two terms"),
        "live_on": "every fixture",
    },
    "row0_not_zero": {
        "old": "        increment[idx] = 0.0f;",
        "new": "        increment[idx] = own_h[1] * weights[i];",
        "why": "the sum's start is the weighted row instead of +0.0: the whole prefix shifts",
        "live_on": "every fixture",
    },
    "row0_negative_zero": {
        "old": "        increment[idx] = 0.0f;",
        "new": "        increment[idx] = -0.0f;",
        "why": "a sign-carrying zero at the sum's start",
        "live_on": "predicted null",
        "predicted_null": (
            "the row-0 prefix word feeds only Dz's curl at rows 0 and 1: row 0 is zeroed by "
            "the r-axis ownership mask, and at row 1 the shifted operand -0.0 enters "
            "(sf - f1) + (f2 - ss), where -0.0 - x == +0.0 - x for every x but zero. "
            "cupy.cumsum absorbs a -0.0 addend the same way. Armed because a prefix scratch "
            "that carried the sign would be visible on a fixture with a zero Dz operand; "
            "recorded either way"),
    },
    "accumulations_regrouped": {
        "old": "    float a = f[idx] + kps * src;",
        "new": "    float a = f[idx];\n    src = src;   // regrouped below",
        "also": [("    return a - kms * prev;", "    return a + (kps * src - kms * prev);")],
        "why": ("the two SEPARATE accumulations become one expression, a different float32 "
                "number because addition is not associative (constitutive_kernels note 1)"),
        "live_on": "every fixture with an active absorber",
    },
    "coefficient_index_is_the_threads_own": {
        "old": ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                "    int i = idx / (ny * nz);\n\n    // Hx <- Bx"),
        "new": "    int k = 0;\n    int j = 0;\n    int i = 0;\n\n    // Hx <- Bx",
        "why": ("the recomputed cell's absorber coefficients are taken from the origin -- "
                "the half-cell class of error, converged and smooth"),
        "live_on": "every fixture with an active absorber",
    },
    "f_w_H_written_in_place": {
        "old": "    *fw_out = src;",
        "new": "    const_cast<float*>(fw)[idx] = src;",
        "why": ("the split-field history stored IN PLACE, so a foreign recompute reading fw "
                "at a cell another block already wrote gets B where its recurrence needs "
                "B_prev. RACY BY CONSTRUCTION; the deterministic twin below carries the "
                "arithmetic consequence"),
        "live_on": "every fixture (schedule-dependent)",
        "predicted_null": ("schedule-dependent by construction; whether a block observes "
                           "another's store is not a function of the source. The "
                           "deterministic twin foreign_history_reads_the_post_store_value "
                           "must be CAUGHT"),
    },
    "foreign_history_reads_the_post_store_value": {
        "old": "    float prev = fw[idx];",
        "new": "    float prev = src;",
        "why": "the recurrence reads the value the in-place arrangement would have handed it",
        "live_on": "every fixture with an active absorber (kms != 0)",
    },
    "swapped_multiply_operand": {
        "old": "    float wi = own_h[1] * weights[i];",
        "new": "    float wi = weights[i] * own_h[1];",
        "why": "coefficient-left multiply; IEEE-754 multiplication commutes bitwise",
        "live_on": "predicted null",
        "predicted_null": ("IEEE-754 multiplication commutes bitwise: field-left is a "
                           "documentation fact for the REAL multiply (0 / 0 in the probe) "
                           "and a rounding fact only for the complex one"),
    },
    "halo_weight_is_own_row": {
        "old": "    float wim1 = halo_h[1] * weights[i - 1];",
        "new": "    float wim1 = halo_h[1] * weights[i];",
        "why": "the neighbour weighted by the thread's own radius: a half-cell error in r",
        "live_on": "every fixture",
    },
    "divisor_index_shifted": {
        "old": "    increment[idx] = diff / divisor[i - 1];",
        "new": "    increment[idx] = diff / divisor[(i < nx - 1) ? i : i - 1];",
        "why": "the divisor ladder read one row late",
        "live_on": "every fixture",
    },
}

HOST_MUTATIONS: Tuple[str, ...] = (
    "rotation_skipped", "curl_launched_first", "stale_prefix", "ir0_zero_ladder",
    "swap_constitutive_sublattice", "scratch_aliased_to_storage")

BYTE_NEUTRAL: Dict[str, str] = {
    "old": "    float wi = own_h[1] * weights[i];",
    "new": "    float wi = Hy_out[idx] * weights[i];",
}


# ---------------------------------------------------------------------------
# The spelling leg: standalone kernels over the array path's own statements
# ---------------------------------------------------------------------------

_SPELL_TEMPLATE = r'''
extern "C" __global__ void spell(
    float* __restrict__ inc, const float* __restrict__ f,
    const float* __restrict__ w, const float* __restrict__ d,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;
    int nyz = ny * nz;
    int i = idx / nyz;
    if (i == 0) { inc[idx] = 0.0f; return; }
    float wi   = @MUL_HERE@;
    float wim1 = @MUL_HALO@;
    float diff = wi - wim1;
    inc[idx] = @DIVIDE@;
}
'''

_SPELL_VARIANTS: Dict[str, Dict[str, str]] = {
    "product": {"@MUL_HERE@": "f[idx] * w[i]", "@MUL_HALO@": "f[idx - nyz] * w[i - 1]",
                "@DIVIDE@": "diff / d[i - 1]"},
    "ctl_reciprocal_multiply": {"@MUL_HERE@": "f[idx] * w[i]",
                                "@MUL_HALO@": "f[idx - nyz] * w[i - 1]",
                                "@DIVIDE@": "diff * (1.0f / d[i - 1])"},
    "ctl_distributed_divide": {"@MUL_HERE@": "f[idx] * w[i]",
                               "@MUL_HALO@": "f[idx - nyz] * w[i - 1]",
                               "@DIVIDE@": "(wi / d[i - 1]) - (wim1 / d[i - 1])"},
    "null_swapped_multiply": {"@MUL_HERE@": "w[i] * f[idx]", "@MUL_HALO@": "w[i - 1] * f[idx - nyz]",
                              "@DIVIDE@": "diff / d[i - 1]"},
}


def edge_field(rng, shape: Tuple[int, int, int]) -> np.ndarray:
    """Bit-assembled: signed zeros, subnormals, tiny normals, ordinary, large, small.
    No host float op touches a subnormal, so keep and flush see identical inputs."""
    kind = rng.integers(0, 6, shape).astype(np.uint32)
    sign = rng.integers(0, 2, shape).astype(np.uint32) << np.uint32(31)
    mant = rng.integers(1, 1 << 23, shape).astype(np.uint32)
    exp = np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3, kind == 4],
        [np.zeros(shape, np.uint32), np.zeros(shape, np.uint32),
         rng.integers(1, 17, shape).astype(np.uint32),
         rng.integers(120, 135, shape).astype(np.uint32),
         np.full(shape, 226, np.uint32)],
        default=np.full(shape, 117, np.uint32)).astype(np.uint32)
    mant = np.where(kind == 0, np.uint32(0), mant).astype(np.uint32)
    return (sign | (exp << np.uint32(23)) | mant).astype(np.uint32).view(np.float32)


def spelling_field(rng, shape: Tuple[int, int, int], cls: str) -> np.ndarray:
    nr = shape[0]
    if cls == "uniform":
        return rng.standard_normal(shape).astype(np.float32)
    if cls == "wide_dynamic":
        return (rng.standard_normal(shape) * (10.0 ** rng.integers(-12, 12, shape))).astype(np.float32)
    if cls == "cancelling":
        sign = np.where(np.arange(nr).reshape(-1, 1, 1) % 2, 1.0, -1.0)
        return (rng.standard_normal(shape) * sign * 1e6).astype(np.float32)
    if cls == "edge":
        return edge_field(rng, shape)
    raise ValueError(cls)


def substitute(template: str, table: Dict[str, str]) -> str:
    src = template
    for macro, expr in table.items():
        assert src.count(macro) == 1, (macro, src.count(macro))
        src = src.replace(macro, expr)
    assert "@" not in src
    return src


def leg_spelling_real() -> Dict[str, Any]:
    """The real divide spelling on standalone kernels against the array path's own four
    statements: the product's spelling 0, the reciprocal multiply and the distributed
    divide bite, the swapped operand is a null, and the unguarded build bites."""
    from meep_gpu.fields import StepScratch  # noqa: PLC0415

    shapes = ((163, 1, 175), (80, 1, 1), (1050, 1, 115))
    classes = ("uniform", "wide_dynamic", "cancelling", "edge")
    cases: List[Dict[str, Any]] = []
    kernels: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
    for shape in shapes:
        for cls in classes:
            rng = case_rng(SEED, f"spelling/{shape}/{cls}")
            f_dev = cp.asarray(spelling_field(rng, shape, cls))
            increment_ap, _shipped = common.array_path_increment(f_dev, 0.5)
            scratch = StepScratch(cp)
            weights, divisor = scratch.constant(
                ("cyl_rderiv", shape[0], 0.5, np.dtype(np.float32)),
                lambda: common.stepping._cylindrical_rderiv_weights(  # noqa: SLF001
                    cp, shape[0], 0.5, np.dtype(np.float32)))
            entry: Dict[str, Any] = {"shape": list(shape), "class": cls,
                                     "words": int(f_dev.size), "variants": {}}
            for variant, table in _SPELL_VARIANTS.items():
                for guard, options in (("fmad_false", ("--fmad=false",)), ("fmad_default", ())):
                    key = (variant, options)
                    if key not in kernels:
                        kernels[key] = cp.RawKernel(substitute(_SPELL_TEMPLATE, table), "spell",
                                                    options=options)
                    out = cp.empty_like(f_dev)
                    cells = int(f_dev.size)
                    kernels[key](((cells + 255) // 256,), (256,),
                                 (out, f_dev, weights.reshape(-1), divisor.reshape(-1),
                                  np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2])))
                    cp.cuda.runtime.deviceSynchronize()
                    entry["variants"][f"{variant}::{guard}"] = differing(out, increment_ap)
            cases.append(entry)
            log(f"[spelling] {shape} {cls:12s} product={entry['variants']['product::fmad_false']}"
                f"/{entry['words']} recip={entry['variants']['ctl_reciprocal_multiply::fmad_false']} "
                f"dist={entry['variants']['ctl_distributed_divide::fmad_false']} "
                f"swap={entry['variants']['null_swapped_multiply::fmad_false']} "
                f"fmad_default={entry['variants']['product::fmad_default']}")
    total = sum(c["words"] for c in cases)

    def agg(name: str) -> int:
        return sum(c["variants"][name] for c in cases)

    out = {
        "cases": cases, "denominator": len(cases), "words": total,
        "product_differing": agg("product::fmad_false"),
        "controls": {
            "reciprocal_multiply": agg("ctl_reciprocal_multiply::fmad_false"),
            "distributed_divide": agg("ctl_distributed_divide::fmad_false"),
            "unguarded_build": agg("product::fmad_default"),
            "swapped_multiply_operand_null": agg("null_swapped_multiply::fmad_false"),
        },
        "every_case_has_a_biting_control": all(
            c["variants"]["ctl_reciprocal_multiply::fmad_false"] > 0
            or c["variants"]["ctl_distributed_divide::fmad_false"] > 0 for c in cases),
    }
    out["passed"] = bool(
        out["product_differing"] == 0 and out["controls"]["reciprocal_multiply"] > 0
        and out["controls"]["distributed_divide"] > 0 and out["controls"]["unguarded_build"] > 0
        and out["controls"]["swapped_multiply_operand_null"] == 0
        and out["every_case_has_a_biting_control"])
    return out


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class RealAdapter(Adapter):
    complex_storage = False
    cell_arms = CELL_ARMS
    specs = SPECS
    mutation_spec_labels = MUTATION_SPEC_LABELS
    reduced_labels = REDUCED_LABELS
    value_classes = ("uniform", "subnormal_band")
    seed = SEED
    device_mutations = DEVICE_MUTATIONS
    host_mutations = HOST_MUTATIONS
    byte_neutral = BYTE_NEUTRAL
    wiring_arms_row = ("ordinary", "cylindrical")

    def __init__(self, policy: Optional[str]) -> None:
        from meep_gpu.cuda_kernels import cylindrical_real_fused_hd_pair as family  # noqa: PLC0415

        self.family = family
        self.policy = policy
        self.license = None
        self.arm = None

    # -- the product -------------------------------------------------------------
    def predicate(self, fields, pml, grid, sources):
        return self.family.covers_cylindrical_real_fused_hd_pair(fields, pml, grid, sources)

    def resolve(self, fields, grid, pml, **overrides):
        state = self.family.resolve(fields, grid, pml)
        state.update(overrides)
        return state

    def launch1(self, fields, state, kernel=None, threads=None):
        return self.family.launch_constitutive_and_increment(
            fields, state, kernel, threads or self.family._FUSED_THREADS)  # noqa: SLF001

    def scan_rotate(self, fields, state, rotate=True):
        return self.family.scan_and_rotate(fields, state, rotate)

    def launch2(self, fields, state):
        return self.family.launch_curl(fields, state)

    def weld_state_mutation(self, name, fields, grid, pml, state):
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415

        if name == "ir0_zero_ladder":
            source = self.prefix_source_volume(fields)
            weights, divisor = common.stepping._cylindrical_rderiv_weights(  # noqa: SLF001
                cp, int(source.shape[0]), 0.0, source.real.dtype)
            state["weights"], state["divisor"] = weights.reshape(-1), divisor.reshape(-1)
            return {"applied": "row vectors rebuilt at ir0 = 0.0 (the B side's half-shift)"}
        if name == "swap_constitutive_sublattice":
            state["constitutive"] = constitutive_kernels.real_constitutive_tables(pml, True)
            return {"applied": "update_H's coefficients taken from the HALF-INTEGER sub-lattice"}
        if name == "scratch_aliased_to_storage":
            state["scratch"] = {vol: getattr(fields, vol) for vol in self.family.SCRATCH_VOLUMES}
            return {"applied": "the scratch twins bound to the live H/f_w_H (the in-place weld)"}
        if name == "stale_prefix":
            return {"applied": "the curl handed a prefix of the PRE-launch Hy (in the arrangement)"}
        raise KeyError(name)

    def kernel_source(self) -> str:
        return self.family.kernel_source()

    # -- the references ------------------------------------------------------------
    def singles(self, fields, grid, pml, dtdx):
        from meep_gpu.cuda_kernels import constitutive_kernels, cylindrical_kernels  # noqa: PLC0415

        tables_h = constitutive_kernels.constitutive_tables_for("H", pml)
        tables_e = constitutive_kernels.constitutive_tables_for("E", pml)
        curl_b = cylindrical_kernels.cylindrical_curl_tables(pml, True)
        curl_d = cylindrical_kernels.cylindrical_curl_tables(pml, False)
        codes = cylindrical_kernels.cylindrical_boundary_codes_for(grid)
        scratch = getattr(fields, "scratch", None)
        return {
            "step_B": lambda: cylindrical_kernels._step_B_fused_pml_cylindrical(  # noqa: SLF001
                fields, curl_b, codes, dtdx, scratch=scratch),
            "update_H": lambda: constitutive_kernels.update_fused_pml_real(
                "H", fields, tables=tables_h),
            "step_D": lambda: cylindrical_kernels._step_D_fused_pml_cylindrical(  # noqa: SLF001
                fields, curl_d, codes, dtdx, scratch=scratch),
            "update_E": lambda: constitutive_kernels.update_fused_pml_real(
                "E", fields, tables=tables_e),
        }

    def released_pairs(self, fields, grid, pml, dtdx):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            cylindrical_real_fused_electric_pair as electric,
            cylindrical_real_fused_magnetic_pair as magnetic)

        return {
            "step_B": lambda: magnetic.run_cylindrical_real_fused_magnetic_pair(
                fields, grid, pml, dtdx, sources=()),
            "step_D": lambda: electric.run_cylindrical_real_fused_electric_pair(
                fields, grid, pml, dtdx, sources=(), scratch=getattr(fields, "scratch", None)),
        }

    def released_pair_replaces(self):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            cylindrical_real_fused_electric_pair as electric,
            cylindrical_real_fused_magnetic_pair as magnetic)

        return {"step_B": tuple(magnetic.REPLACES), "step_D": tuple(electric.REPLACES)}

    def driver_kwargs(self):
        return {"force_complex_fields": False, "m": 0}

    # -- the family's own host legs -------------------------------------------------
    def leg_transcription(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                           cylindrical_kernels, fused_hd_pair,
                                           own_cell_hoist)

        family = self.family
        source = family.kernel_source()
        # The single is hoisted (register view); the weld lifts its statement form.
        certified_h = own_cell_hoist.unhoisted_kernel_code(
            constitutive_kernels._update_H_pml_real_kernel_code,  # noqa: SLF001
            "update_H_pml_real", "_update_H_pml_real_kernel_code")
        prelude = constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE  # noqa: SLF001
        raw_cell = fused_hd_pair.raw_update_H_cell_source()
        pieces = {
            "constitutive_prelude": fused_hd_pair.constitutive_prelude(),
            "weld_args_struct": fused_hd_pair.weld_args_struct(),
            "raw_update_H_cell": raw_cell,
            "signature": family.signature(),
            "increment": family.increment_source(),
        }
        # THE LIFT IS THE CERTIFIED H->D WELD'S OWN: the four emitters are IMPORTED, and
        # the emitted string is their concatenation plus this family's own three pieces.
        composed = (pieces["constitutive_prelude"] + pieces["weld_args_struct"]
                    + pieces["raw_update_H_cell"] + pieces["signature"])
        # THE PRELUDE IS STRIPPED FIRST: its helper closes its own parameter list on
        # "\n) {\n" too, and splitting the whole string there would hand back the
        # helper's body (whose two stores the PURE form moves) as if it were the kernel's.
        h_body = certified_h[len(prelude):].split("\n) {\n", 1)[1][: -len("}\n")]
        decode = ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                  "    int i = idx / (ny * nz);\n")
        constitutive_facts = {
            "source_starts_with_the_four_imported_emitters": source.startswith(composed),
            "decode_is_carried_verbatim": decode in raw_cell,
            "thread_preamble_is_gone": "blockIdx.x" not in raw_cell,
            "three_calls_captured": raw_cell.count("constitutive_apply_pure(") == 3,
            "no_storing_call_survives": "constitutive_apply(" not in raw_cell.replace(
                "constitutive_apply_pure(", ""),
            "the_two_accumulations_are_separate":
                "float a = f[idx] + kps * src;" in pieces["constitutive_prelude"]
                and "return a - kms * prev;" in pieces["constitutive_prelude"],
            "the_helper_stores_nothing":
                "fw[idx] = src;" not in pieces["constitutive_prelude"]
                and "f[idx] = a" not in pieces["constitutive_prelude"],
            "prev_is_read_before_the_store_moves":
                pieces["constitutive_prelude"].index("float prev = fw[idx];")
                < pieces["constitutive_prelude"].index("*fw_out = src;"),
            "certified_body_lines_survive": all(
                line in raw_cell for line in h_body.splitlines()
                if line.strip() and "blockIdx" not in line and "return;" not in line
                and "constitutive_apply(" not in line),
        }
        # THE INCREMENT: every declared line present exactly once, below the own-cell
        # store, and no stale magnetic read below the weld (only the recompute).
        below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
        increment_facts = {
            "declared_lines_present_once": {
                entry["line"]: source.count(entry["line"]) for entry in family.INCREMENT_SPELLING},
            "no_stale_magnetic_read_below_the_weld": {
                name: below.count(f"{name}[") + below.count(f"weld.{name}")
                for name in family.H_TARGETS},
            "halo_is_a_recompute": "raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);" in below,
            "divide_is_ieee": "increment[idx] = diff / divisor[i - 1];" in below,
            "row0_is_positive_zero": "increment[idx] = 0.0f;" in below,
            "no_reciprocal_multiply": "1.0f / divisor" not in source,
            "compile_options": list(family._COMPILE_OPTIONS),  # noqa: SLF001
        }
        curl_source = cylindrical_kernels._cyl_step_D_pml_real_kernel_code  # noqa: SLF001
        launch2_facts = {
            "curl_kernel_name": family.CURL_KERNEL_NAME,
            "curl_kernel_is_the_certified_one":
                f'void {family.CURL_KERNEL_NAME}(' in curl_source
                and family.CURL_KERNEL_NAME in cylindrical_kernels.CERTIFIED_KERNELS,
            "curl_source_sha256": hashlib.sha256(curl_source.encode("utf-8")).hexdigest(),
            "launcher_takes_a_prefix": "prefix=state[\"prefix\"]" in
                __import__("inspect").getsource(family.launch_curl),
        }
        facts = {
            "constitutive": constitutive_facts, "increment": increment_facts,
            "launch2": launch2_facts,
            "source_is_ascii": source.isascii(),
            "source_encodes_under_ascii": encodes_ascii(source),
            "one_device_string": list(family.device_sources()) == [family.KERNEL_NAME],
            "kernel_name_declared_once": source.count(f"void {family.KERNEL_NAME}(") == 1,
            "source_sha256": family.source_digest(), "source_bytes": len(source),
            "mutation_needles_resolve": mutation_needles_resolve(self, source),
            "byte_neutral_needle_resolves": source.count(BYTE_NEUTRAL["old"]) == 1,
        }
        facts["passed"] = bool(
            all(constitutive_facts.values())
            and all(n == 1 for n in increment_facts["declared_lines_present_once"].values())
            and sum(increment_facts["no_stale_magnetic_read_below_the_weld"].values()) == 0
            and increment_facts["halo_is_a_recompute"] and increment_facts["divide_is_ieee"]
            and increment_facts["row0_is_positive_zero"] and increment_facts["no_reciprocal_multiply"]
            and increment_facts["compile_options"] == ["--fmad=false"]
            and launch2_facts["curl_kernel_is_the_certified_one"]
            and launch2_facts["launcher_takes_a_prefix"]
            and facts["source_is_ascii"] and facts["source_encodes_under_ascii"]
            and facts["one_device_string"] and facts["kernel_name_declared_once"]
            and facts["mutation_needles_resolve"]["passed"]
            and facts["byte_neutral_needle_resolves"])
        return facts

    def leg_refusal(self) -> Dict[str, Any]:
        from meep_gpu.fields import Fields  # noqa: PLC0415
        from meep_gpu.grid import Grid  # noqa: PLC0415
        from meep_gpu.pml import PML  # noqa: PLC0415
        from meep_gpu.sources import (ContinuousEnvelope, GaussianEnvelope,  # noqa: PLC0415
                                      VolumeSource)

        family = self.family
        checks: Dict[str, Any] = {}

        def dcyl(m: int = 0, complex_storage: bool = False, shape=(16, 1, 20), **kwargs):
            grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                        cylindrical=True, m=m, boundaries={"z": "metallic"}, courant=0.35,
                        xp=cp, **kwargs)
            fields = Fields(grid=grid, force_complex_fields=complex_storage)
            fields.enable_field_storage()
            fields.enable_pml_storage()
            return fields, PML(grid=grid, thickness={"x": (0, 3), "z": 3}), grid

        def verdict(name: str, expected: bool, thunk, names: Optional[str] = None) -> None:
            try:
                fields, pml, grid, sources = thunk()
                covered, reason = family.covers_cylindrical_real_fused_hd_pair(
                    fields, pml, grid, sources)
            except Exception as error:  # noqa: BLE001 - a construction the engine refuses
                checks[name] = {"covered": False, "expected": expected,
                                "reason": f"construction refused: {error!r}"[:300],
                                "passed": expected is False,
                                "note": "refused by the engine before the predicate was asked"}
                return
            entry = {"covered": bool(covered), "expected": expected, "reason": reason}
            entry["passed"] = bool(covered) == expected
            if names is not None:
                entry["names_the_reason"] = names in reason
                entry["passed"] = entry["passed"] and entry["names_the_reason"]
            checks[name] = entry

        verdict("admits_a_plain_m0_real_dcyl_run", True, lambda: (*dcyl(), ()))
        verdict("refuses_an_undeclared_source_set", False, lambda: (*dcyl(), None), "was not declared")

        def integrated():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
            return fields, pml, grid, [source]

        verdict("refuses_a_standing_integrated_electric_withdraw", False, integrated,
                "standing integrated")

        def plain_source():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2))
            return fields, pml, grid, [source]

        verdict("admits_a_non_integrated_electric_source", True, plain_source)

        def magnetic():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Hz", center=(0.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
            return fields, pml, grid, [source]

        verdict("admits_an_integrated_magnetic_source", True, magnetic)
        verdict("refuses_complex_storage", False, lambda: (*dcyl(complex_storage=True), ()),
                "complex64")
        verdict("refuses_m_not_zero", False, lambda: (*dcyl(m=1, complex_storage=True), ()))

        def no_pml():
            fields, _pml, grid = dcyl()
            return fields, None, grid, ()

        verdict("refuses_a_run_with_no_pml", False, no_pml, "no active PML")

        def cartesian():
            grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 1.6), courant=0.35, xp=cp,
                        boundaries=("periodic", "periodic", "periodic"))
            fields = Fields(grid=grid)
            fields.enable_field_storage()
            fields.enable_pml_storage()
            return fields, PML(grid=grid, thickness=2), grid, ()

        verdict("refuses_a_cartesian_grid", False, cartesian, "cylindrical")
        verdict("refuses_a_bloch_phase", False,
                lambda: (*dcyl(k_point=(0.0, 0.0, 0.1)), ()))
        verdict("refuses_a_single_radial_row", False, lambda: (*dcyl(shape=(1, 1, 20)), ()))

        class _Beta:
            def __init__(self, grid):
                self._grid = grid

            def __getattr__(self, item):
                return getattr(self._grid, item)

            beta = 0.3

        def beta():
            fields, pml, grid = dcyl()
            return fields, pml, _Beta(grid), ()

        verdict("refuses_special_kz", False, beta, "beta")

        def missing_volume():
            fields, pml, grid = dcyl()

            class _NoHz:
                def __getattr__(self, item):
                    return getattr(fields, item)

                Hz = None

            return _NoHz(), pml, grid, ()

        verdict("refuses_an_unallocated_rotated_volume", False, missing_volume, "Hz")
        return {"checks": checks, "denominator": len(checks),
                "passed": all(entry["passed"] for entry in checks.values())}

    def leg_spelling(self) -> Dict[str, Any]:
        return leg_spelling_real()

    def wiring_product_row(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels.fused_pairs import CudaFusedPairPlan  # noqa: PLC0415
        from meep_gpu.triton_kernels.coverage import Coverage  # noqa: PLC0415

        family = self.family

        def coverage(context):
            covered, reason = family.covers_cylindrical_real_fused_hd_pair(
                context.fields, context.pml, context.grid, context.sources)
            return Coverage(bool(covered), (str(reason),))

        def plan(context):
            held: Dict[str, Any] = {}

            def resolve(ctx):
                if "state" not in held:
                    held["state"] = family.resolve(ctx.fields, ctx.grid, ctx.pml)
                return {"state": held["state"]}

            def launch(fields, arguments):
                return family.run_cylindrical_real_fused_hd_pair(
                    fields, fields.grid, context.pml, sources=context.sources,
                    state=arguments["state"])

            return CudaFusedPairPlan(family=family.FAMILY, label="cylindrical real fused H/D pair",
                                     kernel_label=family.KERNEL_NAME,
                                     replaces=tuple(family.REPLACES),
                                     slots=("update_H", "step_D"), context=context,
                                     resolve_launch_args=resolve, launch=launch)

        return {"curl_slot": "update_H", "coverage": coverage, "plan": plan,
                "module": "cylindrical_real_fused_hd_pair"}


def make_adapter(policy: Optional[str], _licence: Optional[Dict[str, Any]]) -> RealAdapter:
    return RealAdapter(policy)


def runtime_reasons() -> List[str]:
    return common.hd_gate.runtime_reasons()


def main(argv=None) -> int:
    return common.run_gate(make_adapter, os.path.abspath(__file__), FAMILY_NAME, BLOCK_KEY,
                           __doc__.splitlines()[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
