#!/usr/bin/env python
"""DEVICE byte gate for the Triton Dcyl COMPLEX H->D product:
``meep_gpu/triton_kernels/cylindrical_fused_hd_pair.py``.

The complex-storage twin of ``gate_triton_cylindrical_real_fused_hd_pair.py``, on the
``(cylindrical complex -> cylindrical complex PML)`` H->D cell: every m the complex
family carries (m = 0 under complex64 storage, |m| = 1, |m| >= 2 with and without the
accurate axis branch) and both z terminations. Same shape -- complex ``update_H`` into
scratch in launch 1, then the certified ``CylindricalComplexCurlPlan`` over the scratch
``H``: its column-serial radial scan over both stored parts of ``Hy`` (launch 2) and its
curl (launch 3) -- same four reference engines, same evidence standard (uint32 words per
complete driver step, both subnormal policies from empty caches, every zero beside a
control that bites). Read the real gate and the kit first; this file states only what
the complex storage adds.

WHAT THE COMPLEX STORAGE ADDS. Launch 1 is the complex Cartesian weld's constitutive
block, statement for statement: ``complex_fused_hd_pair._h_cell_complex`` (the root's
``constitutive_form`` on each stored part, the chunk record, MEEP's start of H), IMPORTED.
No complex multiply, no multiply-add and no divide is spelled in this product -- the
pre-scan increment's FMA_V1 multiply and scaled complex divide left with the
elementwise increment stage when the scan became MEEP's per-chunk column-serial sum
(``cylindrical_complex``'s own scan, the certified plan's). What is armed is the scratch
choreography on both word planes, the auxiliary contract on the rotated history, the
start of H, and the scan and curl ordering, through the same host mutations as the real
gate.

BOTH POLICIES REACH THE COMPOSER. No complex predicate carries a policy clause of its
own (the clause went with the expansion licence it guarded), so under either policy the
composer selects this family's neighbours on the seam and every reference is measured;
``composer_reachable_under_this_policy`` in the record says so.

    CUDA_VISIBLE_DEVICES=<n> CUPY_CACHE_DIR=<empty, token ftz_stripped> \\
      TRITON_CACHE_DIR=<empty> PYTHONPATH=. \\
      python -u parity/meep_gpu/gate_triton_cylindrical_fused_hd_pair.py \\
      --subnormal-policy keep --out <fresh>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import triton_cylindrical_hd_gate_kit as kit  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.triton_kernels import complex_fields  # noqa: E402
from meep_gpu.triton_kernels import complex_fused_hd_pair as complex_fused  # noqa: E402
from meep_gpu.triton_kernels import cylindrical_complex as cx  # noqa: E402
from meep_gpu.triton_kernels import cylindrical_triton as cyl  # noqa: E402
from meep_gpu.triton_kernels import cylindrical_fused_hd_pair as family  # noqa: E402
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402
from meep_gpu.triton_kernels.offdiag_scratch_weld import twin_table  # noqa: E402

GATE = "triton_cylindrical_fused_hd_pair"
CELL_ARMS: Tuple[str, str] = family.ARMS
CENSUS = kit.CENSUS
SEAM_RECORD = kit.SEAM_RECORD
MODES = kit.MODES

SOURCES: Tuple[str, ...] = (
    "meep_gpu/triton_kernels/cylindrical_fused_hd_pair.py",
    "meep_gpu/triton_kernels/cylindrical_real_fused_hd_pair.py",
    "meep_gpu/triton_kernels/fused_hd_pair.py",
    "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
    "meep_gpu/triton_kernels/complex_fused_hd_pair.py",
    "meep_gpu/triton_kernels/cylindrical_complex.py",
    "meep_gpu/triton_kernels/cylindrical_triton.py",
    "meep_gpu/triton_kernels/complex_fields.py",
    "meep_gpu/triton_kernels/kernels.py",
    "meep_gpu/triton_kernels/forms.py",
    "meep_gpu/triton_kernels/coverage.py",
    "meep_gpu/triton_kernels/launch.py",
    "meep_gpu/stepping.py",
    "meep_gpu/withdraw_hoist.py",
    "meep_gpu/driver.py",
    "meep_gpu/fields.py",
    "meep_gpu/fastpath.py",
    "parity/meep_gpu/triton_cylindrical_hd_gate_kit.py",
    "parity/meep_gpu/gate_triton_cylindrical_fused_hd_pair.py",
)

#: The synthetic fixtures: the three m classes the kernel compiles (M_ZERO, M_ONE,
#: M_MANY with both near-axis row counts), both z terminations, one single-column
#: (nz = 1) periodic case on the ring-cyl / perturbation_theory shape class, and
#: Courant numbers below the |m| >= 2 accurate-branch ceiling where that branch is on.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("m0_zm", {"shape": (12, 16), "m": 0, "courant": 0.35, "pml": 3,
               "z_boundary": "metallic"}),
    ("m1_zm", {"shape": (12, 16), "m": 1, "courant": 0.37, "pml": 3,
               "z_boundary": "metallic"}),
    ("mneg1_zp", {"shape": (12, 16), "m": -1, "courant": 0.31, "pml": 3,
                  "z_boundary": "periodic"}),
    ("m2_acc_zm", {"shape": (14, 10), "m": 2, "courant": 0.25, "pml": 2,
                   "z_boundary": "metallic", "accurate": True}),
    ("m3_zm", {"shape": (12, 16), "m": 3, "courant": 0.25, "pml": 3,
               "z_boundary": "metallic"}),
    ("m1_nz1_zp", {"shape": (16, 1), "m": 1, "courant": 0.37, "pml": 3,
                   "z_boundary": "periodic"}),
    # THE MUTATION CASE: 96 radial rows so the ladder reaches the values where the two
    # wrong divide spellings part from the scaled algorithm (numpy's reciprocal
    # coincides with it wherever d * (1/d) rounds to exactly 1.0, which holds for
    # every d < 41 -- MEASURED NULL on a 12-row fixture 2026-09-07), and 192 cells so
    # the launch is still one program.
    ("m1_nr96_nz2", {"shape": (96, 2), "m": 1, "courant": 0.37, "pml": 3,
                     "z_boundary": "periodic"}),
    # FIVE PROGRAMS (1280 cells at BLOCK = 256): the halo recompute crosses program
    # boundaries here, as it does on every corpus row.
    ("m1_nr40_nz32", {"shape": (40, 32), "m": 1, "courant": 0.35, "pml": 4,
                      "z_boundary": "metallic"}),
    # THE RUN'S FIRST update_H on the mutation case's grid: unseparated, so every
    # arrangement performs MEEP's start of H at step 1. Every case above is seeded
    # mid-run.
    ("m1_zm_first_update", {"shape": (12, 16), "m": 1, "courant": 0.37, "pml": 3,
                            "z_boundary": "metallic", "h_started": False}),
)
#: THE MUTATION CASE: |m| = 1 with a walled z, so both Hx's (r) and Hz's (z) own axes
#: carry a layer and the auxiliary-contract rows have dsigw cells to land on; 192 cells,
#: one program.
MUTATION_CASE = "m1_zm"
FIRST_UPDATE_CASE = "m1_zm_first_update"

KERNEL_NAME = "cyl_complex_update_H_scratch"

# ---------------------------------------------------------------------------
# The mutations, each with the case it is scored on and why it must (or must not) bite
# ---------------------------------------------------------------------------

#: The scratch history store of component 0's real word (Hx, whose own axis r carries
#: the layer), as launch 1 spells it. Spelled once because two rows rewrite it.
_FW_STORE_0_RE = ("        tl.store(wo0 + 2 * idx, tl.where(dw0, src0_re, tl.load(\n"
                  "            wi0 + 2 * idx, mask=live & (dw0 == 0), other=0.0)), "
                  "mask=live)\n")

MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m_H_written_in_place": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(ho1 + 2 * idx, own1_re, mask=live)\n",
        "new": "        tl.store(hi1 + 2 * idx, own1_re, mask=live)\n",
        "why": "the stepped Hy's real word written into the PRE-LAUNCH buffer instead of "
               "the scratch: the scan and the curl read a scratch word nobody wrote this "
               "step, and the rotation publishes it",
    },
    "m_H_takes_B": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(ho0 + 2 * idx, own0_re, mask=live)\n",
        "new": "        tl.store(ho0 + 2 * idx, src0_re, mask=live)\n",
        "why": "the scratch Hx's real word takes the flux density the constitutive read "
               "rather than the stepped field. Armed on Hx, whose own axis r carries the "
               "layer: on Hy (phi) the stepped field IS B and the edit is a null by "
               "construction",
    },
    "m_imaginary_plane_takes_B": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(ho0 + 2 * idx + 1, own0_im, mask=live)\n",
        "new": "        tl.store(ho0 + 2 * idx + 1, src0_im, mask=live)\n",
        "why": "THE COMPLEX-SPECIFIC HAZARD: the scratch Hx is two words, and a weld that "
               "carried the stepped value on the real plane only would be wrong exactly "
               "where the imaginary part is live -- invisible to any real-field test",
    },
    "m_f_w_H_written_in_place": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(wo0 + 2 * idx, tl.where(dw0, src0_re, tl.load(\n",
        "new": "        tl.store(wi0 + 2 * idx, tl.where(dw0, src0_re, tl.load(\n",
        "why": "the split-field history written into the PRE-LAUNCH buffer: the rotation "
               "publishes an unwritten scratch history",
    },
    "b7_fw_scratch_stored_unmasked": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _FW_STORE_0_RE,
        "new": "        tl.store(wo0 + 2 * idx, src0_re, mask=live)\n",
        "why": "the scratch f_w_Hx real word takes B at every cell, as the 0.9.2 kernels "
               "wrote it, instead of B on the dsigw cells and the pre-launch word "
               "everywhere else; the rotation publishes the moved history",
    },
    "b8_fw_scratch_stored_on_the_layer_only": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _FW_STORE_0_RE,
        "new": "        tl.store(wo0 + 2 * idx, src0_re, mask=live & dw0)\n",
        "why": "the certified single's masked history store copied literally onto the "
               "rotated scratch: every plain cell of f_w_Hx is published as the twin's "
               "stale word instead of the kept word",
    },
    "b11_start_of_H_not_forwarded": {
        "target": "kernel", "expected": "CAUGHT", "case": FIRST_UPDATE_CASE,
        "old": "            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)\n",
        "new": "            kp0, kp1, kp2, kmx, kmy, kmz, cb, 0, ny, nz)\n",
        "why": "launch 1 hands the constitutive 0 for MEEP's start of H, so the run's "
               "first update_H accumulates onto the stored H instead of starting it from "
               "B on the dsigw cells. Scored on the first-update case",
    },
    "n1_fw_twin_load_unmasked": {
        "target": "kernel", "expected": "CAUGHT", "expected_override": "NULL",
        "case": MUTATION_CASE,
        "old": "            wi1 + 2 * idx, mask=live & (dw1 == 0), other=0.0)), mask=live)\n",
        "new": "            wi1 + 2 * idx, mask=live, other=0.0)), mask=live)\n",
        "why": "PREDICTED NULL (KERNELS.md 7.2, the unconditional auxiliary load): the "
               "pre-launch history word the twin keeps is read on every cell, but on the "
               "dsigw cells the select takes B and never consumes it, so no word moves. "
               "The guard is pinned by the lift instead (the store is the Cartesian "
               "weld's statement)",
    },
    "m_rotation_skipped": {
        "target": "host", "host": "rotation_skipped", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "the plan launches and does NOT move the engine's references onto the "
               "freshly written scratch, so the run publishes the pre-launch magnetic "
               "field",
    },
    "m_fp_fusion_on": {
        "target": "host", "host": "fusion_on", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "all three launches compiled with floating-point contraction enabled: the "
               "per-part constitutive accumulation, the scan's increment and the curl's "
               "forms each contract a product into an fma, a different float32 number",
    },
    "m_scan_skipped": {
        "target": "host", "host": "scan_skipped", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "launch 2 not performed: the curl differences whatever the scan's output "
               "held, where MEEP sums this step's Hy on both parts",
    },
    "m_scan_reads_the_pre_launch_Hy": {
        "target": "host", "host": "scan_reads_pre_launch_Hy", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "launch 2 sums the PRE-LAUNCH Hy -- the previous step's field -- while the "
               "curl reads this step's scratch H; the purity ledger is the floor that "
               "makes the stale sum differ",
    },
    "m_curl_reads_the_pre_launch_H": {
        "target": "host", "host": "curl_reads_pre_launch_H", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "launches 2 and 3 run the certified plan built over the PRE-LAUNCH H: the "
               "scan and the curl consume the state the seam's first half has not yet "
               "produced",
    },
    "m_ir0_zero_ladder": {
        "target": "host", "host": "ir0_zero_ladder", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "the scan's two invariant row vectors rebuilt at ir0 = 0.0 (the B side's "
               "ladder) where step_D sums Hp at 0.5",
    },
    "m_constitutive_takes_the_half_integer_lattice": {
        "target": "host", "host": "half_integer", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the constitutive half bound the `_h` PML sub-lattice instead of the "
               "integer one: a half-cell-wrong absorber profile, smooth and wrong",
    },
    "m_withdraw_after_step_D": {
        "target": "host", "host": "withdraw_after_step_D", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "the seam's one pass on the wrong side of the launch. Since entry 25 that "
               "pass is the magnetic f_minus_p restore (no electric withdraw stands), "
               "and the withdraw leg drives it moved before the launch, and dropped, on "
               "a fixture carrying integrated electric and magnetic sources; both must "
               "diverge (hd_seam_restore.SEAM_NULL_CONTROLS)",
    },
}

#: The one edit required NOT to diverge.
BYTE_NEUTRAL: Dict[str, Any] = {
    "tag": "m_scratch_word_stores_reordered",
    "old": ("        tl.store(ho0 + 2 * idx, own0_re, mask=live)\n"
            "        tl.store(ho0 + 2 * idx + 1, own0_im, mask=live)\n"),
    "new": ("        tl.store(ho0 + 2 * idx + 1, own0_im, mask=live)\n"
            "        tl.store(ho0 + 2 * idx, own0_re, mask=live)\n"),
    "why": "the scratch Hx's two words stored in the other order. PREDICTED NULL: they "
           "are independent words nothing in launch 1 reads, so only the instruction "
           "order may move",
}


# ---------------------------------------------------------------------------
# The family adapter
# ---------------------------------------------------------------------------

class Spec(kit.FamilySpec):
    gate = GATE
    family = family
    complex_storage = True
    cell_arms = CELL_ARMS
    sources = SOURCES
    cases = CASES
    mutation_case = MUTATION_CASE
    kernel_name = KERNEL_NAME
    mutations = MUTATIONS
    byte_neutral = BYTE_NEUTRAL

    def driver_keywords(self, keywords: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(keywords)

    def pml_spec(self, keywords: Mapping[str, Any]) -> Any:
        thickness = int(keywords.get("pml", 3))
        nr, nz = keywords["shape"]
        spec: Dict[str, Any] = {"x": {"high": thickness}}
        if keywords.get("z_boundary", "metallic") == "metallic" and int(nz) > 2 * thickness + 1:
            spec["z"] = thickness
        return spec

    def _from_arrays_inputs(self, driver: Any) -> Dict[str, Any]:
        fields, pml, grid = driver.fields, driver.pml, driver.fields.grid
        arrays = {name: getattr(fields, name)
                  for name in family.ROTATED + family.IN_PLACE + family.CONSTITUTIVE_SOURCES}
        for name, twin in twin_table(fields, family.ROTATED).items():
            arrays["scratch_" + name] = twin
        kinds = stepping._boundary_kinds(grid, pml)  # noqa: SLF001
        return {
            "arrays": arrays,
            "curl_flat": {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                          for axis in "xyz" for stem in ("kms", "sinv")},
            "constitutive_flat": {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                                  for axis in "xyz" for stem in ("kps", "kms")},
            "dtdx": grid.dt / grid.dx, "courant": float(grid.courant),
            "m": int(grid.m),
            "accurate": bool(grid.accurate_fields_near_cylorigin),
            "bcz": 1 if kinds[2] == "metallic" else 0,
        }

    def build_weld(self, driver: Any, kernel: Any = None, curl_kernel: Any = None,
                   forced: bool = False, scan_kernel: Any = None) -> Any:
        sources = tuple(getattr(driver, "_sources", ()))
        if not forced:
            plan = family.plan_cylindrical_fused_hd_pair(
                driver.fields, driver.pml, sources, kernel=kernel, curl_kernel=curl_kernel,
                scan_kernel=scan_kernel)
            if plan is not None:
                return plan
            # The withdraw leg's fixture is refused BY DESIGN (a standing integrated
            # electric source); the gate's from-arrays route builds the same plan there
            # so the hoist and its two null controls can be driven.
            verdict = family.cylindrical_fused_hd_pair_coverage(driver.fields, driver.pml,
                                                                sources)
            if not all("standing integrated" in reason for reason in verdict.reasons):
                raise RuntimeError("the product refused a fixture this gate expects it to "
                                   f"admit: {verdict.reasons}")
        # The same plan the engine route builds, so the same chunk record it binds.
        from gate_chunk_records import layer_record  # noqa: PLC0415

        inputs = self._from_arrays_inputs(driver)
        plan = family.plan_cylindrical_fused_hd_pair_from_arrays(
            inputs["arrays"], inputs["curl_flat"], inputs["constitutive_flat"],
            inputs["dtdx"], inputs["courant"], inputs["m"], inputs["accurate"],
            inputs["bcz"], driver.fields, driver.xp,
            kernel=kernel, curl_kernel=curl_kernel, scan_kernel=scan_kernel,
            chunk_bounds=layer_record(driver.fields.grid, driver.pml),
            scan_starts=cx.radial_scan_starts(driver.fields, driver.pml, "step_D", None))
        # The from-arrays route starts no H; this gate's fixtures say whether theirs
        # has, so the route takes the engine route's start-of-H state.
        plan._h_state = driver.fields  # noqa: SLF001
        return plan

    def singles(self, driver: Any, forced: bool = False) -> Tuple[Any, Any]:
        fields, pml = driver.fields, driver.pml
        if not forced:
            return (cx.plan_cylindrical_complex_constitutive(fields, pml, "H"),
                    cx.plan_cylindrical_complex_curl(fields, pml, "step_D"))
        from gate_chunk_records import layer_record  # noqa: PLC0415

        inputs = self._from_arrays_inputs(driver)
        record = layer_record(fields.grid, pml)
        side = complex_fields.CONSTITUTIVE_SIDES["H"]
        h_arrays = {name: getattr(fields, name)
                    for name in tuple(side["targets"]) + tuple(side["aux"]) + tuple(side["sources"])}
        constitutive = complex_fields.plan_complex_constitutive_from_arrays(
            "H", h_arrays, inputs["constitutive_flat"], chunk_bounds=record)
        # The engine route's start-of-H state (the plan's own slot).
        constitutive._h_state = fields  # noqa: SLF001
        spec = triton_launch.SUB_STEPS["step_D"]
        d_arrays = {name: getattr(fields, name) for name in spec["targets"] + spec["sources"]}
        d_arrays.update({"fu_" + name: getattr(fields, "fu_" + name) for name in spec["targets"]})
        curl = cx.plan_cylindrical_complex_curl_from_arrays(
            "step_D", d_arrays, inputs["curl_flat"], inputs["dtdx"], inputs["courant"],
            inputs["m"], inputs["accurate"], inputs["bcz"], driver.xp,
            chunk_bounds=record,
            scan_starts=cx.radial_scan_starts(fields, pml, "step_D", None))
        return constitutive, curl

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        return family.cylindrical_fused_hd_pair_coverage(fields, pml, sources)

    def default_kernels(self, owner: Any) -> Dict[str, Any]:
        from meep_gpu.triton_kernels import kernels as tkernels  # noqa: PLC0415
        from meep_gpu.triton_kernels import (  # noqa: PLC0415
            cylindrical_fused_electric_pair as pair_d,
            cylindrical_fused_magnetic_pair as pair_b,
        )

        name = type(owner).__name__
        scan = {"scan._kernel": cyl.radial_scan_kernel()}
        if name == "CylindricalFusedHdPairPlan":
            # Launch 1, and BOTH certified step_D plans' scan and curl: the run picks
            # one, and the unpicked plan's counters must stay at zero.
            slots: Dict[str, Any] = {
                "_kernel": family.cyl_complex_update_H_scratch_kernel()}
            for index in range(len(owner.step_d_plans)):
                slots[f"step_d_plans.{index}._kernel"] = cx.cyl_complex_pml_curl_step
                slots[f"step_d_plans.{index}.scan._kernel"] = cyl.radial_scan_kernel()
            return slots
        if name == "ComplexConstitutivePlan":
            return {"_kernel": complex_fields.bloch_constitutive_step}
        if name == "CylindricalComplexCurlPlan":
            return {"_kernel": cx.cyl_complex_pml_curl_step, **scan}
        if name == "ConstitutivePlan":
            return {"_kernel": tkernels.constitutive_step}
        if name == "PmlCurlPlan":
            return {"_kernel": tkernels.pml_curl_step}
        if name == "CylindricalFusedMagneticPairPlan":
            return {"_kernel": pair_b.cyl_complex_fused_curl_constitutive_B_kernel(), **scan}
        if name == "CylindricalFusedElectricPairPlan":
            return {"_kernel": pair_d.cyl_complex_fused_curl_constitutive_D_kernel(), **scan}
        return {}

    def refusal_cases(self, driver: Any) -> List[Dict[str, Any]]:
        fields, pml = driver.fields, driver.pml
        grid = fields.grid

        class _Inactive:
            is_active = False

        return [
            {"name": "undeclared_source_list", "fields": fields, "pml": pml, "sources": None,
             "must_refuse": True, "needle": "was not declared"},
            {"name": "no_sources", "fields": fields, "pml": pml, "sources": (),
             "must_refuse": False},
            {"name": "integrated_electric_withdraw", "fields": fields, "pml": pml,
             "sources": (kit.stand_in_source(),), "must_refuse": True,
             "needle": "standing integrated"},
            {"name": "non_integrated_electric_source", "fields": fields, "pml": pml,
             "sources": (kit.stand_in_source(integrated=False),), "must_refuse": False},
            {"name": "integrated_magnetic_source", "fields": fields, "pml": pml,
             "sources": (kit.stand_in_source(component="Hz", field_type="B"),),
             "must_refuse": False},
            {"name": "not_cylindrical", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, grid=kit.GridView(grid, cylindrical=False)),
             "needle": "not cylindrical"},
            {"name": "real_storage", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, force_complex_fields=False),
             "needle": "force_complex_fields is not set"},
            {"name": "one_radial_row", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, grid=kit.GridView(
                 grid, shape=(1, 1, int(grid.shape[2])))),
             "needle": "radial extent is 1"},
            {"name": "inactive_absorber", "fields": fields, "pml": _Inactive(), "sources": (),
             "must_refuse": True, "first_half": "cylindrical complex constitutive half"},
        ]

    def half_integer_rebind(self, plan: Any, pml: Any) -> Any:
        from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

        plan._kps = tuple(CupyPointer(_flat(getattr(pml, f"kps_{axis}_h")))  # noqa: SLF001
                          for axis in "xyz")
        plan._kms = tuple(CupyPointer(_flat(getattr(pml, f"kms_{axis}_h")))  # noqa: SLF001
                          for axis in "xyz")
        return plan

    def word_view(self, array: Any) -> Any:
        return complex_fields._word_view(array)  # noqa: SLF001

    def composition_refusal_needle(self) -> str:
        return "float32 subnormal policy"

    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        findings: List[str] = []
        measures: Dict[str, Any] = {}
        # THE CONSTITUTIVE IS IMPORTED, NOT COPIED: its lift is the complex Cartesian
        # product's, re-checked here because this family launches it.
        certified = complex_fused.certified_constitutive_tail()
        lifted = complex_fused.lifted_constitutive_tail()
        measures["constitutive_chars"] = len(certified)
        if certified != lifted:
            findings.append("complex_fused_hd_pair._h_cell_complex is no longer "
                            "complex_fields.bloch_constitutive_step's own body with the "
                            "declared edits")
        text = Path(family.__file__).read_text(encoding="utf-8")
        if "from .complex_fused_hd_pair import _h_cell_complex" not in text \
                or "def _h_cell_complex(" in text:
            findings.append("the complex constitutive is not imported from "
                            "complex_fused_hd_pair (or a copy is defined here)")
        # LAUNCH 1 IS THE COMPLEX WELD'S CONSTITUTIVE BLOCK, statement for statement.
        certified_block = family.certified_weld_block()
        lifted_block = family.lifted_weld_block()
        measures["launch1_statements"] = len(lifted_block)
        if certified_block != lifted_block:
            findings.append("launch 1 is not complex_fused_hd_pair's constitutive block, "
                            "statement for statement")
        if len(lifted_block) < 20:
            findings.append(f"launch 1 is only {len(lifted_block)} statements; the cut "
                            f"anchors have crossed")
        # NO ARITHMETIC OF ITS OWN: no complex product helper, no multiply-add, no divide.
        for needle in ("_mul_", "_rotate_", "_div_", "_connect", "tl.math.", " / "):
            if any(needle in line for line in lifted_block):
                findings.append(f"launch 1 spells {needle.strip()!r}")
        if family.MULTIPLY_HELPERS != ():
            findings.append(f"MULTIPLY_HELPERS is {family.MULTIPLY_HELPERS}, not empty")
        # THE RADIAL-SUM DECLARATION IS THE CERTIFIED FAMILY'S OWN TABLE, and the scan
        # and the curl are the certified plan's, built by the certified builder.
        spec = cx.PREFIX[family.CURL_SUB_STEP]
        if (family.PREFIX_COMPONENT, float(family.PREFIX_IR0)) != (
                spec["component"], float(spec["ir0"])):
            findings.append("the product's radial-sum declaration is not "
                            "cylindrical_complex's")
        measures["radial_sum"] = {"component": family.PREFIX_COMPONENT,
                                  "ir0": family.PREFIX_IR0}
        for needle in ("_cx.plan_cylindrical_complex_curl_from_arrays(",
                       "_cx.radial_scan_starts(fields, pml, CURL_SUB_STEP, None)"):
            if needle not in text:
                findings.append(f"the product no longer builds the certified step_D plan "
                                f"through {needle!r}")
        tree = ast.parse((API_ROOT / "meep_gpu" / "triton_kernels" / "kernels.py").read_text(
            encoding="utf-8"))
        certified_default = next(
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "DEFAULT_BLOCK")
        if family.DEFAULT_BLOCK != certified_default:
            findings.append(f"DEFAULT_BLOCK {family.DEFAULT_BLOCK} is not kernels.py's "
                            f"{certified_default}")
        try:
            measures["sub_lattice_suffixes"] = list(family._sub_lattice_suffixes())  # noqa: SLF001
        except AssertionError as error:
            findings.append(str(error))
        measures["launches_per_run"] = family.LAUNCHES_PER_RUN
        measures["kernel_launches_per_run"] = family.KERNEL_LAUNCHES_PER_RUN
        if family.LAUNCHES_PER_RUN != family.KERNEL_LAUNCHES_PER_RUN or \
                family.LAUNCHES_PER_RUN != 1 + kit.CYLINDRICAL_PLAN_KERNEL_LAUNCHES:
            findings.append("LAUNCHES_PER_RUN does not count launch 1 and the certified "
                            "plan's scan and curl")
        return findings, measures

    def ghost_observability(self) -> Tuple[List[str], Dict[str, Any]]:
        findings: List[str] = []
        measures: Dict[str, Any] = {}
        curl_text = Path(cx.__file__).read_text(encoding="utf-8")
        product_text = Path(family.__file__).read_text(encoding="utf-8")
        for needle in ("curl0_re =", "curl2_re =", "def cyl_complex_pml_curl_step",
                       "def cyl_radial_difference", "total + "):
            if needle in product_text:
                findings.append(f"the product module carries curl or scan text "
                                f"({needle!r})")
        measures["launches_2_and_3_are_the_certified_plans_run"] = (
            "plan.run(guard)" in product_text)
        if not measures["launches_2_and_3_are_the_certified_plans_run"]:
            findings.append("the plan does not run the certified complex step_D plan")
        reads_own = ("curl2_re = dtdx * tl.load(rad + 2 * idx, mask=live, other=0.0)"
                     in curl_text
                     and "curl2_im = dtdx * tl.load(rad + 2 * idx + 1, mask=live, other=0.0)"
                     in curl_text)
        masked = ("curl2_re = tl.where(at_r, 0.0, curl2_re)" in curl_text
                  and "curl2_im = tl.where(at_r, 0.0, curl2_im)" in curl_text)
        measures["curl_reads_the_scans_difference_at_its_own_cell"] = reads_own
        measures["target2_zeroed_at_the_axis_row_under_BACKWARD"] = masked
        if not reads_own:
            findings.append("the certified complex curl no longer reads the scan's "
                            "difference at its own cell")
        if not masked:
            findings.append("the certified complex curl no longer zeroes Dz's curl at the "
                            "axis row")
        measures["what_this_licenses"] = (
            "the certified complex curl takes Dz's radial term from the scan's difference "
            "at its own cell and zeroes it at the axis row on every m class, so no edit "
            "to what an r near ghost serves can be byte-visible on the D side; this "
            "product runs the certified plan over the scratch H")
        return findings, measures

    def arbitration_incumbents(self) -> Tuple[str, str]:
        return ("fused pair B (cylindrical complex)", "fused pair D (cylindrical complex)")

    def product_row(self) -> Dict[str, str]:
        return {"curl_slot": "update_H", "module": "cylindrical_fused_hd_pair",
                "coverage": "cylindrical_fused_hd_pair_coverage",
                "builder": "plan_cylindrical_fused_hd_pair",
                "label": "fused pair H->D (cylindrical complex)"}


SPEC = Spec()

leg_driver_order = lambda: kit.leg_driver_order(SPEC)  # noqa: E731
leg_transcription = lambda: kit.leg_transcription(SPEC)  # noqa: E731
leg_purity_ledger = lambda: kit.leg_purity_ledger(SPEC)  # noqa: E731
leg_ghost_observability = lambda: kit.leg_ghost_observability(SPEC)  # noqa: E731
lift_basis = lambda results: kit.lift_basis(SPEC, results)  # noqa: E731
run_product = kit.run_product
drive = kit.drive
evaluate_row = kit.evaluate_row
leg_lift = kit.leg_lift
_is_private = kit._is_private  # noqa: SLF001


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the gate; see the module docstring and the kit for what each leg measures."""
    return kit.main(SPEC, Path(__file__).resolve(), argv)


if __name__ == "__main__":
    sys.exit(main())
