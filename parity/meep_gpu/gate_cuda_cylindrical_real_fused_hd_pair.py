#!/usr/bin/env python3
"""Device byte gate for the CUDA REAL m = 0 Dcyl H->D product (three launches).

WHAT IS BEING CERTIFIED. ``meep_gpu/cuda_kernels/cylindrical_real_fused_hd_pair.py``:
launch 1 (slot ``update_H``) computes the certified ``update_H_pml_real`` body at the
thread's own cell into launch-local scratch, with MEEP's per-chunk constitutive forms;
launch 2 is the column-serial radial scan (``cylindrical_prefix``) over the SCRATCH Hy --
MEEP's per-chunk ``f_rderiv_int`` of the Hp ``update_H`` just produced; the ``H``/``f_w_H``
bindings rotate; launch 3 is the certified ``cyl_step_D_pml_real`` with the radial
difference handed in. Three launches per seam, as the D side's own baseline is three
under the per-chunk sum: the product saves NO launch and ships ``INSTALLABLE = False``.

THE ARITHMETIC IS THE CERTIFIED HALVES', and the battery says so: launch 1's
accumulation inverted to the package's old grouping (KERNELS.md section 7.1), the
section 7.2 B battery on its constitutive select, the unguarded build, and the scratch
stores; the scan's own battery is ``gate_cuda_cylindrical_scan.py``'s, re-read here on
the family's shapes by the ``spelling`` leg.

THE CLAIM is per COMPLETE DRIVER STEP, as uint32 words over every stored volume, against
FOUR references stepped from one seed on every fixture: the array path, the certified
singles (``update_H_pml_real`` then ``cyl_step_D_pml_real``), the composition the
composer installs on these rows TODAY (the released real cylindrical B->H and D->E
pairs, driven through their own entry points), and the four slots dispatched unfused.
Every weld step must advance THREE counters -- the family tally, the memo's kernel count
and the family's scan tally -- and no ``cupy.cumsum`` may run, so nothing passes by not
executing.

STANDALONE. Every arrangement here is planned from arrays by this gate and the array
path is driven by this gate. The arbitration leg measures the composer as shipped and
with the product row patched in-process (refused BY NAME on its own
``INSTALLABLE = False``), then removes it.

LEGS: see ``cylindrical_hd_gate_common.LEG_GROUPS``. Progress reporting: a flushed line per case.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
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
    # INVERTED (bit_parity_0_10 KERNELS.md section 7.1 and Appendix A): the shipped
    # accumulation IS MEEP's grouping, so the must-catch edit installs the package's old
    # two left-to-right accumulations.
    "accumulation_regrouped_to_the_old_package_order": {
        "old": "(f[idx] + ((kps * src) - (kms * prev)))",
        "new": "((f[idx] + (kps * src)) - (kms * prev))",
        "why": ("MEEP's one accumulation f + ((kps*src) - (kms*prev)) (step_generic.cpp's "
                "f[i] += ..., the right-hand side formed first) becomes the package's old "
                "(f + kps*src) - kms*prev, a different float32 number because addition is "
                "not associative"),
        "live_on": "every fixture with an active absorber",
    },
    "fmad_default_build": {
        "old": None, "new": None, "options": (),
        "why": ("the same source built WITHOUT --fmad=false: NVRTC may contract the "
                "accumulation's products with its subtraction"),
        "live_on": "every fixture",
    },
    "coefficient_index_is_the_threads_own": {
        "old": ("    // the thread's own.\n\n    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                "    int i = idx / (ny * nz);\n"),
        "new": "    // the thread's own.\n\n    int k = 0;\n    int j = 0;\n    int i = 0;\n",
        "why": ("the cell's absorber coefficients and chunk membership taken from the "
                "origin -- the half-cell class of error, converged and smooth"),
        "live_on": "every fixture with an active absorber",
    },
    "foreign_history_reads_the_post_store_value": {
        "old": "    float prev = dsigw ? fw[idx] : 0.0f;",
        "new": "    float prev = dsigw ? src : 0.0f;",
        "why": ("the recurrence reads the value an in-place arrangement would have handed "
                "it: B where it needs B_prev"),
        "live_on": "every fixture with an active absorber (kms != 0)",
    },
    # --- MEEP's per-chunk forms (KERNELS.md section 7.2, the B battery) ---------------
    "constitutive_auxiliary_stored_on_a_plain_cell": {
        "old": "    *fw_out = dsigw ? src : fw[idx];",
        "new": "    *fw_out = src;",
        "why": ("the auxiliary contract (C2): off the dsigw run the array path leaves "
                "f_w_H untouched, and the rotation makes the scratch the live f_w_H"),
        "live_on": "every fixture with an active absorber and a plain cell",
    },
    "plain_constitutive_takes_the_accumulation": {
        "old": "    float value = dsigw ? (f[idx] + ((kps * src) - (kms * prev))) : src;",
        "new": "    float value = (f[idx] + ((kps * src) - (kms * prev)));",
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
        "old": "    Hx_out[idx] = own_h[0]; Hy_out[idx] = own_h[1]; Hz_out[idx] = own_h[2];",
        "new": "    Hx_out[idx] = own_h[0]; Hz_out[idx] = own_h[2];",
        "why": ("the scratch Hy keeps the word the rotation left in it, so the scan and the "
                "curl read an H two sub-steps old -- the volume the radial sum is OF"),
        "live_on": "every fixture",
    },
    "auxiliary_scratch_rows_swapped": {
        "old": "    f_w_Hx_out[idx] = own_w[0]; f_w_Hy_out[idx] = own_w[1];",
        "new": "    f_w_Hx_out[idx] = own_w[1]; f_w_Hy_out[idx] = own_w[0];",
        "why": "two components' split-field histories stored to each other's twin",
        "live_on": "every fixture with an active absorber",
    },
}

HOST_MUTATIONS: Tuple[str, ...] = (
    "rotation_skipped", "curl_launched_first", "stale_radial", "one_chunk_starts",
    "swap_constitutive_sublattice", "scratch_aliased_to_storage",
    "every_cell_chunk_record")

BYTE_NEUTRAL: Dict[str, str] = {
    "old": "    Hx_out[idx] = own_h[0]; Hy_out[idx] = own_h[1]; Hz_out[idx] = own_h[2];",
    "new": ("    float hy_word = own_h[1];\n"
            "    Hx_out[idx] = own_h[0]; Hy_out[idx] = hy_word; Hz_out[idx] = own_h[2];"),
}


def leg_spelling_real() -> Dict[str, Any]:
    """The scan's spelling, on this family's shapes: ``gate_cuda_cylindrical_scan``'s
    identity comparison (real storage; the one-rank division of a high-r layer and a
    three-chunk line). The 0.9.2 increment spelling this leg used to measure is gone
    with the increment; the scan's own mutation battery is that gate's."""
    import gate_cuda_cylindrical_scan as scan_gate  # noqa: PLC0415

    shapes = tuple(tuple(spec["shape"]) for spec in SPECS if spec["shape"][2] >= 4)[:6]
    out = scan_gate.identity_leg(shapes, backend="cuda", storages=("real",),
                                 divisions=("layer", "three_chunks"))
    out["passed"] = bool(out["all_identical"])
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

    # -- the product -------------------------------------------------------------
    def predicate(self, fields, pml, grid, sources):
        return self.family.covers_cylindrical_real_fused_hd_pair(fields, pml, grid, sources)

    @staticmethod
    def chunk_record(fields, pml):
        """The run's chunk record on the device: the region plan's division, which is
        the division the array path steps this run with."""
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
        return self.family.launch_curl(fields, state)

    def weld_state_mutation(self, name, fields, grid, pml, state):
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_prefix  # noqa: PLC0415

        if name == "one_chunk_starts":
            tables = dict(cylindrical_prefix.radial_scan_tables(fields, pml, "step_D"))
            tables["starts"] = np.zeros_like(np.asarray(tables["starts"]))
            tables["segs"] = np.zeros(1, dtype=np.int32)
            state["scan"] = cylindrical_prefix.radial_scan_device_tables(tables, cp)
            return {"applied": "the scan told the run is one chunk: one sum from the axis "
                               "over the whole r extent, where MEEP restarts per chunk"}
        if name == "swap_constitutive_sublattice":
            state["constitutive"] = constitutive_kernels.real_constitutive_tables(pml, True)
            return {"applied": "update_H's coefficients taken from the HALF-INTEGER sub-lattice"}
        if name == "scratch_aliased_to_storage":
            state["scratch"] = {vol: getattr(fields, vol) for vol in self.family.SCRATCH_VOLUMES}
            return {"applied": "the scratch twins bound to the live H/f_w_H (the in-place weld)"}
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
        cb = self.chunk_record(fields, pml)
        return {
            "step_B": lambda: cylindrical_kernels._step_B_fused_pml_cylindrical(  # noqa: SLF001
                fields, curl_b, codes, dtdx, scratch=scratch, chunk_bounds=cb, pml=pml),
            "update_H": lambda: constitutive_kernels.update_fused_pml_real(
                "H", fields, tables=tables_h, chunk_bounds=cb),
            "step_D": lambda: cylindrical_kernels._step_D_fused_pml_cylindrical(  # noqa: SLF001
                fields, curl_d, codes, dtdx, scratch=scratch, chunk_bounds=cb, pml=pml),
            "update_E": lambda: constitutive_kernels.update_fused_pml_real(
                "E", fields, tables=tables_e, chunk_bounds=cb),
        }

    def released_pairs(self, fields, grid, pml, dtdx):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            cylindrical_real_fused_electric_pair as electric,
            cylindrical_real_fused_magnetic_pair as magnetic)

        cb = self.chunk_record(fields, pml)
        return {
            "step_B": lambda: magnetic.run_cylindrical_real_fused_magnetic_pair(
                fields, grid, pml, dtdx, chunk_bounds=cb, sources=()),
            "step_D": lambda: electric.run_cylindrical_real_fused_electric_pair(
                fields, grid, pml, dtdx, chunk_bounds=cb, sources=()),
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
        import inspect  # noqa: PLC0415
        import re  # noqa: PLC0415

        from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                           cylindrical_kernels, cylindrical_prefix,
                                           fused_hd_pair, own_cell_hoist)

        family = self.family
        source = family.kernel_source()
        # The single is hoisted (register view); the weld lifts its statement form.
        certified_h = own_cell_hoist.unhoisted_kernel_code(
            constitutive_kernels._update_H_pml_real_kernel_code,  # noqa: SLF001
            "update_H_pml_real", "_update_H_pml_real_kernel_code")
        prelude = constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE  # noqa: SLF001
        raw_cell = fused_hd_pair.raw_update_H_cell_source()
        pure_prelude = fused_hd_pair.constitutive_prelude()
        composed = (pure_prelude + fused_hd_pair.weld_args_struct() + raw_cell
                    + family.signature())
        # THE PRELUDE IS STRIPPED FIRST: its helper closes its own parameter list on
        # "\n) {\n" too, and splitting the whole string there would hand back the
        # helper's body (whose two stores the PURE form moves) as if it were the kernel's.
        h_body = certified_h[len(prelude):].split("\n) {\n", 1)[1][: -len("}\n")]
        decode = ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                  "    int i = idx / (ny * nz);\n")
        constitutive_facts = {
            "source_starts_with_the_imported_emitters": source.startswith(composed),
            "decode_is_carried_verbatim": decode in raw_cell,
            "thread_preamble_is_gone": "blockIdx.x" not in raw_cell,
            "three_calls_captured": raw_cell.count("constitutive_apply_pure(") == 3,
            "no_storing_call_survives": "constitutive_apply(" not in raw_cell.replace(
                "constitutive_apply_pure(", ""),
            "the_accumulation_is_meeps":
                "(f[idx] + ((kps * src) - (kms * prev)))" in pure_prelude,
            "prev_is_read_before_the_store_moves":
                pure_prelude.index("float prev = dsigw ? fw[idx] : 0.0f;")
                < pure_prelude.index("*fw_out = dsigw ? src : fw[idx];"),
            "certified_body_lines_survive": all(
                line in raw_cell for line in h_body.splitlines()
                if line.strip() and "blockIdx" not in line and "return;" not in line
                and "constitutive_apply(" not in line),
        }
        below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
        launch1_facts = {
            "the_thread_s_own_cell_only": source.count("raw_update_H_cell(idx") == 1,
            "no_magnetic_read_or_write_below_the_cell": {
                name: below.count(f"{name}[") + below.count(f"weld.{name}")
                for name in family.H_TARGETS},
            "six_scratch_stores": sum(len(re.findall(rf"\b{name}_out\[idx\] =", below))
                                      for name in family.SCRATCH_VOLUMES) == 6,
            "compile_options": list(family._COMPILE_OPTIONS),  # noqa: SLF001
        }
        curl_source = cylindrical_kernels._cyl_step_D_pml_real_kernel_code  # noqa: SLF001
        later_facts = {
            "scan_kernel_is_the_shipped_one":
                "get_radial_scan_kernel()" in inspect.getsource(family.scan_and_rotate),
            "scan_sums_the_scratch_hp": 'state["scratch"][PREFIX_SOURCE]'
                in inspect.getsource(family.scan_and_rotate),
            "scan_kernel_name": cylindrical_prefix.RADIAL_SCAN_KERNEL,
            "curl_kernel_name": family.CURL_KERNEL_NAME,
            "curl_kernel_is_the_certified_one":
                f'void {family.CURL_KERNEL_NAME}(' in curl_source
                and family.CURL_KERNEL_NAME in cylindrical_kernels.CERTIFIED_KERNELS,
            "curl_source_sha256": hashlib.sha256(curl_source.encode("utf-8")).hexdigest(),
            "launcher_hands_the_radial_difference": 'radial=state["radial"]' in
                inspect.getsource(family.launch_curl),
        }
        facts = {
            "constitutive": constitutive_facts, "launch1": launch1_facts,
            "scan_and_curl": later_facts,
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
            and launch1_facts["the_thread_s_own_cell_only"]
            and sum(launch1_facts["no_magnetic_read_or_write_below_the_cell"].values()) == 0
            and launch1_facts["six_scratch_stores"]
            and launch1_facts["compile_options"] == ["--fmad=false"]
            and later_facts["scan_kernel_is_the_shipped_one"]
            and later_facts["scan_sums_the_scratch_hp"]
            and later_facts["curl_kernel_is_the_certified_one"]
            and later_facts["launcher_hands_the_radial_difference"]
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

        # Since entry 25 an engine integrated source enters through f_minus_p and stands
        # no withdraw, so it is ADMITTED; a source class bringing its own withdraw is
        # still refused by name (hd_seam_restore.withdraw_refusal_checks).
        i_fields, i_pml, i_grid, i_sources = integrated()
        checks.update(hd_seam_restore.withdraw_refusal_checks(
            lambda sources: family.covers_cylindrical_real_fused_hd_pair(
                i_fields, i_pml, i_grid, sources),
            i_sources[0]))

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
                    held["state"] = family.resolve(ctx.fields, ctx.grid, ctx.pml,
                                                   chunk_bounds=ctx.chunk_bounds_device())
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


def make_adapter(policy: Optional[str]) -> RealAdapter:
    return RealAdapter(policy)


def runtime_reasons() -> List[str]:
    return common.hd_gate.runtime_reasons()


def main(argv=None) -> int:
    return common.run_gate(make_adapter, os.path.abspath(__file__), FAMILY_NAME, BLOCK_KEY,
                           __doc__.splitlines()[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
