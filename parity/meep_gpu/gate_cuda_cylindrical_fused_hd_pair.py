#!/usr/bin/env python3
"""Device byte gate for the CUDA COMPLEX Dcyl H->D product (three launches), any m.

WHAT IS BEING CERTIFIED. ``meep_gpu/cuda_kernels/cylindrical_fused_hd_pair.py``: launch
1 (slot ``update_H``) computes the certified ``update_H_pml_complex_bloch`` body, lifted
PURE, at the thread's own cell into launch-local scratch, with MEEP's per-chunk
constitutive forms part by part; launch 2 is the column-serial radial scan
(``cylindrical_prefix``) over the SCRATCH Hy, one lane per stored part -- MEEP's
per-chunk ``f_rderiv_int`` of the Hp ``update_H`` just produced; the bindings rotate;
launch 3 is the certified ``cyl_step_D_pml_complex`` with the radial difference handed
in -- the ``i*m/r`` couplings, the |m| = 1 fold, the m = 0 axis post-add and the
|m| >= 2 zeroing all read the ROTATED H pointwise there. Three launches per seam, as the
D side's own baseline is three under the per-chunk sum: the product saves NO launch and
ships ``INSTALLABLE = False``.

THE ARITHMETIC IS THE CERTIFIED HALVES', and the battery says so: launch 1's
accumulation inverted to the package's old grouping (KERNELS.md section 7.1), the
section 7.2 B battery on its constitutive select, the unguarded build, and the scratch
stores; the scan's own battery is ``gate_cuda_cylindrical_scan.py``'s, re-read here on
complex storage and the family's shapes by the ``spelling`` leg.

Everything else is the real sibling's gate over complex64 storage, every integer m, both
z terminations, the ``accurate_fields_near_cylorigin`` branch, one-column rows and a
corpus-sized extent: four references per complete driver step, three launch counters
per weld step (the family tally, the memo's kernel count and the family's scan tally;
no ``cupy.cumsum`` may run), the floors, the scan stage per case, the driver-level sync
and seam legs (the seam leg drives the magnetic f_minus_p restore, hd_seam_restore), the
arbitration as shipped and wired in-process, and the corpus lift over the cell's 18 rows
(all 18 admitted and driven: the 2 the seam record flags withdraw_in_seam were refused
on their standing withdraw before entry 25, and since it no engine source stands one).

STANDALONE: every arrangement here is planned from arrays by this gate and the array
path is driven by this gate; the arbitration leg patches the product row in-process.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = str(next(parent for parent in Path(_HERE).parents
                     if (parent / "meep_gpu" / "cuda_kernels").is_dir()))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cylindrical_hd_gate_common as common  # noqa: E402
import hd_seam_restore  # noqa: E402
from cylindrical_hd_gate_common import (Adapter, cp, encodes_ascii,  # noqa: E402
                                        mutation_needles_resolve)

SUBJECT_PACKAGE = "cuda_kernels"
SEED = 20260907
FAMILY_NAME = "cuda_cylindrical_fused_hd_pair"
BLOCK_KEY = "cuda_cylindrical_fused_hd_pair_gate"
CELL_ARMS: Tuple[str, str] = ("cuda_complex/complex", "cuda_cyl_complex/cylindrical complex")

#: The fixtures: every m class, both z terminations, odd extents, the accurate-origin
#: branch, one-column rows, a thin absorber, a corpus-sized extent. ``plant`` marks the
#: fixtures that ALSO run the planted row-0 class.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "m0_r16_z20_metallic", "shape": (16, 1, 20), "m": 0, "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "planted")},
    {"label": "m0_r16_z20_periodic", "shape": (16, 1, 20), "m": 0, "z_kind": "periodic"},
    {"label": "m+1_r16_z20_metallic", "shape": (16, 1, 20), "m": 1, "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "planted", "signed_zero")},
    # THE DIVISOR-LADDER FIXTURE. On the D side the divisor is the INTEGER ladder
    # d = 1, 2, 3, ... (ir0 = 0.5), and CuPy's scaled divide reduces EXACTLY to
    # numpy's re * fl(1/d) wherever d * fl(1/d) rounds to 1 in float32 -- which is every
    # d < 41 (first exceptions 41, 47, 55, 61, 82, 83, 94, 97, ...; measured on the
    # host before this fixture was added). A 16-row fixture therefore cannot tell the
    # two spellings apart on non-zero data; this one reaches row 64.
    {"label": "m+1_r64_z12_metallic", "shape": (64, 1, 12), "m": 1, "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "signed_zero")},
    {"label": "m-1_r16_z20_periodic", "shape": (16, 1, 20), "m": -1, "z_kind": "periodic"},
    {"label": "m+1_r9_z17_metallic", "shape": (9, 1, 17), "m": 1, "z_kind": "metallic"},
    {"label": "m+2_r9_z17_metallic", "shape": (9, 1, 17), "m": 2, "z_kind": "metallic"},
    {"label": "m+3_r9_z17_periodic", "shape": (9, 1, 17), "m": 3, "z_kind": "periodic"},
    {"label": "m+5_r12_z48_metallic", "shape": (12, 1, 48), "m": 5, "z_kind": "metallic"},
    {"label": "m-1_r40_z16_metallic", "shape": (40, 1, 16), "m": -1, "z_kind": "metallic",
     "classes": ("uniform", "subnormal_band", "signed_zero")},
    # The accurate-origin branch drops the default treatment's zero rows near the axis,
    # and the array path refuses it BY NAME above courant 1/(|m| + 0.5) (0.2857 at |m| = 3):
    # the near-axis update is a smooth growing mode there, not a crash. The fixture-wide
    # default (0.35) tripped that refusal on the first campaign cut (attempt 1, all five
    # per-fixture legs errored on this row only), so the row carries its own courant.
    {"label": "m+3_accurate_r16_z20", "shape": (16, 1, 20), "m": 3, "z_kind": "metallic",
     "accurate": True, "courant": 0.25},
    # ONE-COLUMN ROWS: a one-cell z axis must be PERIODIC (Grid refuses a metallic unit
    # axis by name), which is what the three one-column corpus rows carry.
    {"label": "m+1_r80_z1_periodic", "shape": (80, 1, 1), "m": 1, "z_kind": "periodic"},
    {"label": "m0_thin_r16_z20_metallic", "shape": (16, 1, 20), "m": 0, "z_kind": "metallic",
     "thin_absorber": True, "classes": ("uniform", "subnormal_band", "planted")},
    {"label": "m+1_r120_z120_metallic", "shape": (120, 1, 120), "m": 1, "z_kind": "metallic"},
)
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("m+1_r16_z20_metallic", "m+1_r64_z12_metallic",
                                         "m0_thin_r16_z20_metallic", "m+2_r9_z17_metallic")
REDUCED_LABELS: Tuple[str, ...] = ("m0_r16_z20_metallic", "m+1_r16_z20_metallic",
                                   "m+2_r9_z17_metallic")

#: ``{name: {"old", "new", "why", "live_on", ...}}`` -- device-source rewrites of
#: launch 1, each of which MUST make a scored fixture diverge unless declared null.
DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    # INVERTED (bit_parity_0_10 KERNELS.md section 7.1 and Appendix A): the shipped
    # accumulation IS MEEP's grouping, part by part, so the must-catch edit installs the
    # package's old two left-to-right accumulations.
    "accumulation_regrouped_to_the_old_package_order": {
        "old": ("cf_add(cf_load(f, idx), cf_sub(mul_coefficient_left(kps, src), "
                "mul_coefficient_left(kms, prev)))"),
        "new": ("cf_sub(cf_add(cf_load(f, idx), mul_coefficient_left(kps, src)), "
                "mul_coefficient_left(kms, prev))"),
        "why": ("MEEP's one accumulation f + ((kps*src) - (kms*prev)) per part becomes the "
                "package's old (f + kps*src) - kms*prev, a different float32 number because "
                "addition is not associative"),
        "live_on": "every fixture with an active absorber",
    },
    "fmad_default_build": {
        "old": None, "new": None, "options": (),
        "why": ("the same source built WITHOUT --fmad=false: NVRTC may contract the "
                "accumulation's products with its subtraction"),
        "live_on": "every fixture",
    },
    "coefficient_index_is_the_threads_own": {
        "old": ("    // thread's own.\n\n    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                "    int i = idx / (ny * nz);\n"),
        "new": "    // thread's own.\n\n    int k = 0;\n    int j = 0;\n    int i = 0;\n",
        "why": ("the cell's absorber coefficients and chunk membership taken from the "
                "origin -- the half-cell class of error, converged and smooth"),
        "live_on": "every fixture with an active absorber",
    },
    "foreign_history_reads_the_post_store_value": {
        "old": "    cf prev = dsigw ? cf_load(fw, idx) : cf_zero();",
        "new": "    cf prev = dsigw ? src : cf_zero();",
        "why": ("the recurrence reads the value an in-place arrangement would have handed "
                "it: B where it needs B_prev"),
        "live_on": "every fixture with an active absorber (kms != 0)",
    },
    # --- MEEP's per-chunk forms (KERNELS.md section 7.2, the B battery) ---------------
    "constitutive_auxiliary_stored_on_a_plain_cell": {
        "old": "    *fw_out = dsigw ? src : cf_load(fw, idx);",
        "new": "    *fw_out = src;",
        "why": ("the auxiliary contract (C2): off the dsigw run the array path leaves "
                "f_w_H untouched, and the rotation makes the scratch the live f_w_H"),
        "live_on": "every fixture with an active absorber and a plain cell",
    },
    "plain_constitutive_takes_the_accumulation": {
        "old": "    cf updated = dsigw ? cf_add(",
        "new": "    cf updated = 1 ? cf_add(",
        "why": ("the select removed on update_H: the accumulation on every cell, where "
                "MEEP's plain chunk steps H = B"),
        "live_on": "every fixture with an active absorber and a plain cell",
    },
    "membership_bound_off_by_one": {
        # The r layer is the record's FIRST run on these grids ([13, 16) on the 16-row
        # fixtures: the r-high layer, there being no r-low one), so its END is the
        # extent and moving it changes nothing; its START moves a plane.
        "old": "const int in_i_x = (i >= cbi_x_a0 && i < cbi_x_a1)",
        "new": "const int in_i_x = (i >= cbi_x_a0 - 1 && i < cbi_x_a1)",
        "why": "the r layer's run start moved one row in: that plane changes form",
        "live_on": "every fixture",
    },
    # --- the scratch stores -------------------------------------------------------
    "scratch_store_dropped_for_Hy": {
        "old": "    cf_store(Hx_out, idx, own_h[0]); cf_store(Hy_out, idx, own_h[1]);",
        "new": "    cf_store(Hx_out, idx, own_h[0]);",
        "why": ("the scratch Hy keeps the word the rotation left in it, so the scan and the "
                "curl read an H two sub-steps old -- the volume the radial sum is OF"),
        "live_on": "every fixture",
    },
    "auxiliary_scratch_rows_swapped": {
        "old": "    cf_store(f_w_Hx_out, idx, own_w[0]); cf_store(f_w_Hy_out, idx, own_w[1]);",
        "new": "    cf_store(f_w_Hx_out, idx, own_w[1]); cf_store(f_w_Hy_out, idx, own_w[0]);",
        "why": "two components' split-field histories stored to each other's twin",
        "live_on": "every fixture with an active absorber",
    },
}

HOST_MUTATIONS: Tuple[str, ...] = (
    "rotation_skipped", "curl_launched_first", "stale_radial", "one_chunk_starts",
    "swap_constitutive_sublattice", "scratch_aliased_to_storage",
    "every_cell_chunk_record")

BYTE_NEUTRAL: Dict[str, str] = {
    "old": "    cf_store(Hx_out, idx, own_h[0]); cf_store(Hy_out, idx, own_h[1]);",
    "new": ("    cf hy_word = own_h[1];\n"
            "    cf_store(Hx_out, idx, own_h[0]); cf_store(Hy_out, idx, hy_word);"),
}


def leg_spelling_complex(adapter: "ComplexAdapter") -> Dict[str, Any]:
    """The scan's spelling on complex64 storage (two word columns per cell), on this
    family's shapes: ``gate_cuda_cylindrical_scan``'s identity comparison over the
    one-rank division of a high-r layer and a three-chunk line. The 0.9.2 increment
    spellings this leg used to measure are gone with the increment; the scan's own
    mutation battery is that gate's."""
    import gate_cuda_cylindrical_scan as scan_gate  # noqa: PLC0415

    del adapter
    shapes = tuple(tuple(spec["shape"]) for spec in SPECS if spec["shape"][2] >= 4)[:6]
    out = scan_gate.identity_leg(shapes, backend="cuda", storages=("complex",),
                                 divisions=("layer", "three_chunks"))
    out["passed"] = bool(out["all_identical"])
    return out


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class ComplexAdapter(Adapter):
    complex_storage = True
    cell_arms = CELL_ARMS
    specs = SPECS
    mutation_spec_labels = MUTATION_SPEC_LABELS
    reduced_labels = REDUCED_LABELS
    value_classes = ("uniform", "subnormal_band")
    seed = SEED
    device_mutations = DEVICE_MUTATIONS
    host_mutations = HOST_MUTATIONS
    byte_neutral = BYTE_NEUTRAL
    wiring_arms_row = ("complex", "cylindrical complex")

    def __init__(self, policy: Optional[str]) -> None:
        from meep_gpu.cuda_kernels import cylindrical_fused_hd_pair as family  # noqa: PLC0415

        self.family = family
        self.policy = policy

    @staticmethod
    def real_plane(name: str, shape: Tuple[int, int, int], value_class: str, rng) -> np.ndarray:
        # THE SIGNED-ZERO CLASS, DENSE ON THE RADIAL SUM'S OWN VOLUMES: the real planes
        # of Hy, By and f_w_Hy are signed zeros on EVERY cell (so Hy_new.re is a signed
        # zero everywhere and the scan's zero-sign classes are hit on every word),
        # the other real planes on half their cells, the imaginary planes uniform.
        if value_class == "signed_zero" and name in ("Hy", "By", "f_w_Hy"):
            return np.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0).astype(np.float32)
        return Adapter.real_plane(name, shape, value_class, rng)

    def plant(self, fields, grid, value_class: str) -> None:
        """THE ROW-0 TINY-NORMAL PLANT on top of a uniform seed. With no absorber on
        phi (kps_y = kms_y = 1 everywhere) ``Hy_new = (Hy + By) - f_w_Hy_prev`` exactly, so
        the pre-launch state is chosen to make ``Hy_new[0] = (-tiny_normal, negative)`` --
        the class where the scan's first weighted word ``Hy.re * 0.5`` underflows (a
        subnormal under keep, a signed zero under flush) -- and ``Hy_new[1].re = -0.0``
        with ``Hy_new[1].im > 0``, so the first difference subtracts that word from a
        signed zero. The scan must reproduce the array path's word under the policy the
        run was built for."""
        if value_class != "planted":
            return
        shape = tuple(int(n) for n in grid.shape)
        columns = (shape[1], shape[2])
        rng = np.random.default_rng(self.seed + 7)
        mant = rng.integers(0, 1 << 23, columns).astype(np.uint32)
        tiny = ((np.uint32(1) << np.uint32(31)) | (np.uint32(1) << np.uint32(23)) | mant)
        row0_by = np.empty(columns, dtype=np.complex64)
        row0_by.real = tiny.astype(np.uint32).view(np.float32)
        row0_by.imag = -(0.1 + np.abs(rng.standard_normal(columns))).astype(np.float32)
        row1_by = np.empty(columns, dtype=np.complex64)
        row1_by.real = np.full(columns, -0.0, dtype=np.float32)
        row1_by.imag = (0.1 + np.abs(rng.standard_normal(columns))).astype(np.float32)
        zeros = np.zeros(columns, dtype=np.complex64)
        row1_hy = np.empty(columns, dtype=np.complex64)
        row1_hy.real = np.full(columns, -0.0, dtype=np.float32)
        row1_hy.imag = (0.1 + np.abs(rng.standard_normal(columns))).astype(np.float32)
        fields.Hy[0] = cp.asarray(zeros)
        fields.f_w_Hy[0] = cp.asarray(zeros)
        fields.By[0] = cp.asarray(row0_by)
        fields.Hy[1] = cp.asarray(row1_hy)
        fields.f_w_Hy[1] = cp.asarray(zeros)
        fields.By[1] = cp.asarray(row1_by)

    # -- the product -------------------------------------------------------------
    def predicate(self, fields, pml, grid, sources):
        return self.family.covers_cylindrical_fused_hd_pair(fields, pml, grid, sources)

    @staticmethod
    def chunk_record(fields, pml):
        """The run's chunk record on the device: the region plan's division."""
        from meep_gpu.cuda_kernels import cylindrical_kernels  # noqa: PLC0415

        return cylindrical_kernels.device_chunk_record(fields, pml)

    def resolve(self, fields, grid, pml, **overrides):
        state = self.family.resolve(fields, grid, pml,
                                    chunk_bounds=self.chunk_record(fields, pml))
        state.update(overrides)
        return state

    def launch1(self, fields, state, kernel=None, threads=None):
        return self.family.launch_update_H(
            fields, state, kernel, threads or self.family._FUSED_THREADS)  # noqa: SLF001

    def scan_rotate(self, fields, state, rotate=True):
        return self.family.scan_and_rotate(fields, state, rotate)

    def launch2(self, fields, state):
        return self.family.launch_curl(fields, fields.grid, state["layer"], state)

    def kernel_source(self) -> str:
        return self.family.kernel_source()

    def source_digest(self) -> str:
        return self.family.source_digest()

    def weld_state_mutation(self, name, fields, grid, pml, state):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_prefix  # noqa: PLC0415

        if name == "one_chunk_starts":
            tables = dict(cylindrical_prefix.radial_scan_tables(fields, pml, "step_D"))
            tables["starts"] = np.zeros_like(np.asarray(tables["starts"]))
            tables["segs"] = np.zeros(1, dtype=np.int32)
            state["scan"] = cylindrical_prefix.radial_scan_device_tables(tables, cp)
            return {"applied": "the scan told the run is one chunk: one sum from the axis "
                               "over the whole r extent, where MEEP restarts per chunk"}
        if name == "swap_constitutive_sublattice":
            state["constitutive"] = complex_pml_kernels.complex_constitutive_tables(pml, True)
            return {"applied": "update_H's coefficients from the HALF-INTEGER sub-lattice"}
        if name == "scratch_aliased_to_storage":
            state["scratch"] = {vol: getattr(fields, vol) for vol in self.family.SCRATCH_VOLUMES}
            return {"applied": "the scratch twins bound to the live H/f_w_H"}
        if name == "every_cell_chunk_record":
            import gate_chunk_records  # noqa: PLC0415

            record = gate_chunk_records.every_cell_record(tuple(int(n) for n in grid.shape))
            state["chunk_bounds"] = cp.ascontiguousarray(cp.asarray(record, dtype=cp.int32))
            return {"applied": "launch 1 and the curl handed a division in which every cell "
                               "carries every layer: the 0.9.2 whole-grid forms"}
        if name == "stale_radial":
            return {"applied": "the curl handed the radial difference of the PRE-launch Hy "
                               "(in the arrangement)"}
        raise KeyError(name)

    # -- the references ------------------------------------------------------------
    def singles(self, fields, grid, pml, dtdx):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as cyl  # noqa: PLC0415

        tables_h = complex_pml_kernels.complex_constitutive_tables(pml, False)
        tables_e = complex_pml_kernels.complex_constitutive_tables(pml, True)
        scratch = getattr(fields, "scratch", None)
        cb = self.chunk_record(fields, pml)
        return {
            "step_B": lambda: cyl.step_cylindrical_complex("step_B", fields, grid, pml,
                                                           chunk_bounds=cb, scratch=scratch),
            "update_H": lambda: complex_pml_kernels.update_fused_pml_complex(
                "H", fields, tables=tables_h, chunk_bounds=cb),
            "step_D": lambda: cyl.step_cylindrical_complex("step_D", fields, grid, pml,
                                                           chunk_bounds=cb, scratch=scratch),
            "update_E": lambda: complex_pml_kernels.update_fused_pml_complex(
                "E", fields, tables=tables_e, chunk_bounds=cb),
        }

    def released_pairs(self, fields, grid, pml, dtdx):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            cylindrical_fused_electric_pair as electric,
            cylindrical_fused_magnetic_pair as magnetic)

        cb = self.chunk_record(fields, pml)
        return {
            "step_B": lambda: magnetic.run_cylindrical_fused_magnetic_pair(
                fields, grid, pml, chunk_bounds=cb, sources=()),
            "step_D": lambda: electric.run_cylindrical_fused_electric_pair(
                fields, grid, pml, chunk_bounds=cb, sources=()),
        }

    def released_pair_replaces(self):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            cylindrical_fused_electric_pair as electric,
            cylindrical_fused_magnetic_pair as magnetic)

        return {"step_B": tuple(magnetic.REPLACES), "step_D": tuple(electric.REPLACES)}

    def driver_kwargs(self):
        return {"force_complex_fields": True, "m": 1}

    def driver_source(self, integrated: bool):
        return {"component": "Ez", "center": (4.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                "frequency": 1.0,
                "source_type": "continuous" if integrated else "gaussian",
                **({"is_integrated": True} if integrated else {"fwidth": 0.2})}

    # -- the family's own host legs -------------------------------------------------
    def leg_transcription(self) -> Dict[str, Any]:
        import re  # noqa: PLC0415

        from meep_gpu.cuda_kernels import complex_emitter  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_fused_hd_pair as chd  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as cyl  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_prefix  # noqa: PLC0415

        family = self.family
        certified_prelude = complex_emitter.prelude()
        certified_body = complex_emitter.complex_source("update_H")[len(certified_prelude):]
        certified_body = certified_body.split("\n) {\n", 1)[1][: -len("}\n")]
        pure_prelude = chd.constitutive_prelude("plain")
        raw_cell = chd.raw_update_H_cell_source()
        composed = (pure_prelude + chd.weld_args_struct() + raw_cell + family.signature())
        decode = ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                  "    int i = idx / (ny * nz);\n")
        source = family.kernel_source()
        below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
        curl_source = cyl.cylindrical_complex_source("step_D")
        facts = {
            "source_starts_with_the_imported_emitters": source.startswith(composed),
            "decode_is_carried_verbatim": decode in raw_cell,
            "thread_preamble_is_gone": "blockIdx.x" not in raw_cell,
            "three_calls_captured": raw_cell.count("constitutive_apply_pure(") == 3,
            "no_storing_call_survives": "constitutive_apply(" not in raw_cell.replace(
                "constitutive_apply_pure(", ""),
            "the_accumulation_is_meeps": (
                "cf_add(cf_load(f, idx), cf_sub(mul_coefficient_left(kps, src), "
                "mul_coefficient_left(kms, prev)))") in pure_prelude,
            "certified_body_lines_survive": all(
                line in raw_cell for line in certified_body.splitlines()
                if line.strip() and "blockIdx" not in line and "return;" not in line
                and "constitutive_apply(" not in line),
            "the_thread_s_own_cell_only": source.count("raw_update_H_cell(idx") == 1,
            "no_magnetic_read_below_the_cell": {
                name: below.count(f"weld.{name}") for name in family.H_TARGETS},
            "six_scratch_stores": sum(
                len(re.findall(rf"cf_store\({name}_out, idx,", below))
                for name in family.SCRATCH_VOLUMES) == 6,
            "kernel_name_declared_once": source.count(f"void {family.KERNEL_NAME}(") == 1,
            "source_is_ascii": source.isascii() and encodes_ascii(source),
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "scan_and_curl": {
                "scan_kernel_is_the_shipped_one":
                    "get_radial_scan_kernel()" in inspect.getsource(family.scan_and_rotate),
                "scan_sums_the_scratch_hp": 'state["scratch"][PREFIX_SOURCE]'
                    in inspect.getsource(family.scan_and_rotate),
                "scan_kernel_name": cylindrical_prefix.RADIAL_SCAN_KERNEL,
                "curl_kernel_name": family.CURL_KERNEL_NAME,
                "curl_kernel_is_the_certified_one":
                    f"void {family.CURL_KERNEL_NAME}(" in curl_source
                    and family.CURL_KERNEL_NAME in cyl.CERTIFIED_KERNELS,
                "curl_source_sha256": hashlib.sha256(curl_source.encode("utf-8")).hexdigest(),
                "launcher_hands_the_radial_difference":
                    'radial=state["radial"]' in inspect.getsource(family.launch_curl),
            },
        }
        facts["passed"] = bool(
            facts["source_starts_with_the_imported_emitters"]
            and facts["decode_is_carried_verbatim"] and facts["thread_preamble_is_gone"]
            and facts["three_calls_captured"] and facts["no_storing_call_survives"]
            and facts["the_accumulation_is_meeps"]
            and facts["certified_body_lines_survive"]
            and facts["the_thread_s_own_cell_only"]
            and sum(facts["no_magnetic_read_below_the_cell"].values()) == 0
            and facts["six_scratch_stores"]
            and facts["kernel_name_declared_once"] and facts["source_is_ascii"]
            and facts["scan_and_curl"]["scan_kernel_is_the_shipped_one"]
            and facts["scan_and_curl"]["scan_sums_the_scratch_hp"]
            and facts["scan_and_curl"]["curl_kernel_is_the_certified_one"]
            and facts["scan_and_curl"]["launcher_hands_the_radial_difference"])
        out = {
            "transcription": facts,
            "compile_options": list(family._COMPILE_OPTIONS),  # noqa: SLF001
            "source_sha256": family.source_digest(),
            "mutation_needles_resolve": mutation_needles_resolve(self, source),
            "byte_neutral_needle_resolves": source.count(BYTE_NEUTRAL["old"]) == 1,
        }
        out["passed"] = bool(
            facts["passed"]
            and out["compile_options"] == ["--fmad=false"]
            and out["mutation_needles_resolve"]["passed"]
            and out["byte_neutral_needle_resolves"])
        return out

    def leg_refusal(self) -> Dict[str, Any]:
        from meep_gpu.fields import Fields  # noqa: PLC0415
        from meep_gpu.grid import Grid  # noqa: PLC0415
        from meep_gpu.pml import PML  # noqa: PLC0415
        from meep_gpu.sources import (ContinuousEnvelope, GaussianEnvelope,  # noqa: PLC0415
                                      VolumeSource)

        family = self.family
        checks: Dict[str, Any] = {}

        def dcyl(m: int = 1, complex_storage: bool = True, shape=(16, 1, 20), **kwargs):
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
                covered, reason = family.covers_cylindrical_fused_hd_pair(
                    fields, pml, grid, sources)
            except Exception as error:  # noqa: BLE001
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

        for m in (0, 1, -3):
            verdict(f"admits_a_plain_complex_dcyl_run_m{m}", True, lambda m=m: (*dcyl(m=m), ()))
        verdict("admits_the_accurate_origin_branch", True,
                lambda: (*dcyl(m=2, accurate_fields_near_cylorigin=True), ()))
        verdict("refuses_an_undeclared_source_set", False, lambda: (*dcyl(), None), "was not declared")

        def integrated():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Ez", center=(4.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
            return fields, pml, grid, [source]

        # Since entry 25 an engine integrated source enters through f_minus_p and stands
        # no withdraw, so it is ADMITTED; a source class bringing its own withdraw is
        # still refused by name (hd_seam_restore.withdraw_refusal_checks).
        i_fields, i_pml, i_grid, i_sources = integrated()
        checks.update(hd_seam_restore.withdraw_refusal_checks(
            lambda sources: family.covers_cylindrical_fused_hd_pair(
                i_fields, i_pml, i_grid, sources),
            i_sources[0]))

        def plain_source():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Ez", center=(4.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2))
            return fields, pml, grid, [source]

        verdict("admits_a_non_integrated_electric_source", True, plain_source)

        def magnetic():
            fields, pml, grid = dcyl()
            source = VolumeSource(grid=grid, component="Hz", center=(4.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0),
                                  envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
            return fields, pml, grid, [source]

        verdict("admits_an_integrated_magnetic_source", True, magnetic)
        verdict("refuses_real_storage", False, lambda: (*dcyl(m=0, complex_storage=False), ()),
                "real float32")

        def no_pml():
            fields, _pml, grid = dcyl()
            return fields, None, grid, ()

        verdict("refuses_a_run_with_no_pml", False, no_pml, "no active PML")

        def cartesian():
            grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 1.6), courant=0.35, xp=cp,
                        boundaries=("periodic", "periodic", "periodic"))
            fields = Fields(grid=grid, force_complex_fields=True)
            fields.enable_field_storage()
            fields.enable_pml_storage()
            return fields, PML(grid=grid, thickness=2), grid, ()

        verdict("refuses_a_cartesian_grid", False, cartesian, "cylindrical")
        verdict("refuses_a_bloch_phase", False, lambda: (*dcyl(k_point=(0.0, 0.0, 0.1)), ()))
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
        return leg_spelling_complex(self)

    def wiring_product_row(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels.fused_pairs import CudaFusedPairPlan  # noqa: PLC0415
        from meep_gpu.triton_kernels.coverage import Coverage  # noqa: PLC0415

        family = self.family

        def coverage(context):
            covered, reason = family.covers_cylindrical_fused_hd_pair(
                context.fields, context.pml, context.grid, context.sources)
            return Coverage(bool(covered), (str(reason),))

        def plan(context):
            held: Dict[str, Any] = {}

            def resolve(ctx):
                if "state" not in held:
                    held["state"] = family.resolve(ctx.fields, ctx.grid, ctx.pml,
                                                   chunk_bounds=ctx.chunk_bounds_device())
                return {"state": held["state"]}

            def launch(fields, arguments):
                return family.run_cylindrical_fused_hd_pair(
                    fields, fields.grid, context.pml, sources=context.sources,
                    state=arguments["state"])

            return CudaFusedPairPlan(family=family.FAMILY, label="cylindrical complex fused H/D pair",
                                     kernel_label=family.KERNEL_NAME,
                                     replaces=tuple(family.REPLACES),
                                     slots=("update_H", "step_D"), context=context,
                                     resolve_launch_args=resolve, launch=launch)

        return {"curl_slot": "update_H", "coverage": coverage, "plan": plan,
                "module": "cylindrical_fused_hd_pair"}


def make_adapter(policy: Optional[str]) -> ComplexAdapter:
    return ComplexAdapter(policy)


def runtime_reasons() -> List[str]:
    return common.hd_gate.runtime_reasons()


def main(argv=None) -> int:
    return common.run_gate(make_adapter, os.path.abspath(__file__), FAMILY_NAME, BLOCK_KEY,
                           __doc__.splitlines()[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
