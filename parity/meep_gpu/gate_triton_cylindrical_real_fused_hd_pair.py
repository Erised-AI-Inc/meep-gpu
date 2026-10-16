#!/usr/bin/env python
"""DEVICE byte gate for the Triton Dcyl m = 0 REAL H->D product:
``meep_gpu/triton_kernels/cylindrical_real_fused_hd_pair.py``.

THE FOURTH SEAM ON A CYLINDRICAL ROW, on this backend, for the three real-storage rows
of the ``(cylindrical -> cylindrical PML)`` H->D cell. The product is THREE launches:
launch 1 (slot ``update_H``) evaluates the certified ``update_H`` body at its own cell
into write-only scratch -- the Cartesian weld's constitutive block, statement for
statement, with the chunk record and MEEP's start of H; launches 2 and 3 (slot
``step_D``) are the certified ``CylindricalCurlPlan`` over the scratch ``H`` -- its
column-serial radial scan (MEEP's per-chunk ``f_rderiv_int`` of the stepped ``Hy``) and
its curl; the plan rotates the six ``H``/``f_w_H`` references. Three launches where the
two certified singles take three: the product is kept, not dispatched, for its form
alignment.

THE CLAIM is per COMPLETE DRIVER STEP, as uint32 words over EVERY stored volume, against
the array path and three more reference engines driven in lockstep from one seed (the
template's four: array, the two certified singles, the composition the composer installs
today -- both released cylindrical pairs -- and the same slots unfused), over a stated
budget of steps, under BOTH float32 subnormal policies from empty caches, with every
zero beside a control that moves words. Every leg is the H->D template gate's
(``gate_triton_fused_hd_pair.py``), written once in ``triton_cylindrical_hd_gate_kit.py``,
plus the cylindrical floors on every product row.

THE ARMED DEFECTS, each required to move words: the stepped H or its history written
into the pre-launch buffer, the history stored on every cell (the 0.9.2 store) or on the
layer only (the certified single's store copied onto a rotated twin), the start of H not
forwarded, the rotation skipped, contraction on, the scan skipped, the scan reading the
pre-launch ``Hy``, scan and curl both over the pre-launch ``H``, the ``ir0 = 0.0`` ladder,
the constitutive half on the half-integer lattice, the withdraw after ``step_D``. The
constitutive form battery (KERNELS.md 7.1 and 7.2) runs on ``fused_hd_pair``'s gate,
on the same ``_h_cell`` bytes this product launches. The byte-neutral edit swaps two
independent scratch stores.

    CUDA_VISIBLE_DEVICES=<n> CUPY_CACHE_DIR=<empty, token ftz_stripped> \\
      TRITON_CACHE_DIR=<empty> PYTHONPATH=. \\
      python -u parity/meep_gpu/gate_triton_cylindrical_real_fused_hd_pair.py \\
      --subnormal-policy keep --out <fresh>/gate.json

    # laptop: the host legs only
    PYTHONPATH=. python -u \\
      parity/meep_gpu/gate_triton_cylindrical_real_fused_hd_pair.py --no-device \\
      --out <fresh>/no_device.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import triton_cylindrical_hd_gate_kit as kit  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.triton_kernels import cylindrical_real_fused_hd_pair as family  # noqa: E402
from meep_gpu.triton_kernels import cylindrical_triton as cyl  # noqa: E402
from meep_gpu.triton_kernels import fused_hd_pair as cartesian  # noqa: E402
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402
from meep_gpu.triton_kernels.offdiag_scratch_weld import twin_table  # noqa: E402

GATE = "triton_cylindrical_real_fused_hd_pair"
CELL_ARMS: Tuple[str, str] = family.ARMS
CENSUS = kit.CENSUS
SEAM_RECORD = kit.SEAM_RECORD
MODES = kit.MODES

#: Every module whose bytes this gate's verdict depends on.
SOURCES: Tuple[str, ...] = (
    "meep_gpu/triton_kernels/cylindrical_real_fused_hd_pair.py",
    "meep_gpu/triton_kernels/fused_hd_pair.py",
    "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
    "meep_gpu/triton_kernels/cylindrical_triton.py",
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
    "parity/meep_gpu/gate_triton_cylindrical_real_fused_hd_pair.py",
)

#: The synthetic fixtures. z is METALLIC on every one -- the real family admits exactly
#: the ``("axis", "periodic", "metallic")`` triple -- and the shapes and Courant numbers
#: vary (two non-power-of-two Courants, one exact half). ``pml`` is the absorber
#: thickness in cells, on the high r face and both z faces. Every case is seeded MID-RUN
#: except the last, the run's first ``update_H`` (MEEP's start of H at step 1).
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("nr12_nz16", {"shape": (12, 16), "courant": 0.35, "pml": 3}),
    ("nr9_nz7_c05", {"shape": (9, 7), "courant": 0.5, "pml": 2}),
    ("nr20_nz5_c037", {"shape": (20, 5), "courant": 0.37, "pml": 1}),
    # FIVE PROGRAMS (1280 cells at BLOCK = 256), and a radial column the scan walks
    # across them.
    ("nr40_nz32", {"shape": (40, 32), "courant": 0.35, "pml": 4}),
    ("nr12_nz16_first_update",
     {"shape": (12, 16), "courant": 0.35, "pml": 3, "h_started": False}),
)
MUTATION_CASE = "nr12_nz16"
FIRST_UPDATE_CASE = "nr12_nz16_first_update"

#: The launch-1 kernel's function name -- the mutation seam ``kernel=`` reaches.
KERNEL_NAME = "cyl_real_update_H_scratch"

# ---------------------------------------------------------------------------
# The mutations, each with the case it is scored on and why it must (or must not) bite
# ---------------------------------------------------------------------------

#: The scratch history store of component 0 (Hx, whose own axis r carries the layer),
#: as launch 1 spells it. Spelled once because two rows rewrite it.
_FW_STORE_0 = ("        tl.store(wo0 + idx, tl.where(dw0, src0, tl.load(\n"
               "            wi0 + idx, mask=live & (dw0 == 0), other=0.0)), mask=live)\n")

MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m_H_written_in_place": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(ho1 + idx, own1, mask=live)\n",
        "new": "        tl.store(hi1 + idx, own1, mask=live)\n",
        "why": "the stepped Hy written into the PRE-LAUNCH buffer instead of the scratch: "
               "the scan and the curl read a scratch Hy nobody wrote this step, and the "
               "rotation publishes it",
    },
    "m_H_takes_B": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(ho0 + idx, own0, mask=live)\n",
        "new": "        tl.store(ho0 + idx, src0, mask=live)\n",
        "why": "the scratch Hx takes the flux density the constitutive read rather than "
               "the stepped field -- the mis-selection a nine-value return makes easy. "
               "Armed on Hx, whose own axis r carries the layer: Hy's (phi) never does, "
               "so its stepped field IS B on every cell and the same edit there is a "
               "null by construction",
    },
    "m_f_w_H_written_in_place": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(wo1 + idx, tl.where(dw1, src1, tl.load(\n",
        "new": "        tl.store(wi1 + idx, tl.where(dw1, src1, tl.load(\n",
        "why": "the split-field history written into the PRE-LAUNCH buffer: the rotation "
               "publishes an unwritten scratch history",
    },
    "b7_fw_scratch_stored_unmasked": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _FW_STORE_0,
        "new": "        tl.store(wo0 + idx, src0, mask=live)\n",
        "why": "the scratch f_w_Hx takes B at every cell, as the 0.9.2 kernels wrote it, "
               "instead of B on the dsigw cells (Hx's own axis r carries the layer) and "
               "the pre-launch word everywhere else; the rotation publishes the moved "
               "history on cells the auxiliary contract leaves at rest",
    },
    "b8_fw_scratch_stored_on_the_layer_only": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _FW_STORE_0,
        "new": "        tl.store(wo0 + idx, src0, mask=live & dw0)\n",
        "why": "the certified single's masked history store copied literally onto the "
               "scratch: the twin is rotated into the engine, so every plain cell of "
               "f_w_Hx is published as the twin's stale word instead of the kept word",
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
        "old": "            wi1 + idx, mask=live & (dw1 == 0), other=0.0)), mask=live)\n",
        "new": "            wi1 + idx, mask=live, other=0.0)), mask=live)\n",
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
               "field. The choreography is half the product",
    },
    "m_fp_fusion_on": {
        "target": "host", "host": "fusion_on", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "all three launches compiled with floating-point contraction enabled: the "
               "constitutive accumulation, the scan's increment and the curl's forms each "
               "contract a product into an fma, a different float32 number",
    },
    "m_scan_skipped": {
        "target": "host", "host": "scan_skipped", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "launch 2 not performed: the curl differences whatever the scan's output "
               "held (the zero buffer on step 1, the previous step's difference after), "
               "where MEEP sums this step's Hy",
    },
    "m_scan_reads_the_pre_launch_Hy": {
        "target": "host", "host": "scan_reads_pre_launch_Hy", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "launch 2 sums the PRE-LAUNCH Hy -- the previous step's field -- while the "
               "curl reads this step's scratch H. The purity ledger is the floor that "
               "makes this a claim about the product: update_H moves Hy on the rows the "
               "scan reads, so the stale sum must differ",
    },
    "m_curl_reads_the_pre_launch_H": {
        "target": "host", "host": "curl_reads_pre_launch_H", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "launches 2 and 3 run the certified plan built over the PRE-LAUNCH H: the "
               "scan and the curl consume the state the seam's first half has not yet "
               "produced -- the curl-first ordering, driven through the shipped launch "
               "code",
    },
    "m_ir0_zero_ladder": {
        "target": "host", "host": "ir0_zero_ladder", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "the scan's two invariant row vectors rebuilt at ir0 = 0.0 (the B side's "
               "ladder) where step_D sums Hp at 0.5 -- the same arithmetic on the wrong "
               "half-integer sites",
    },
    "m_constitutive_takes_the_half_integer_lattice": {
        "target": "host", "host": "half_integer", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the constitutive half bound the `_h` PML sub-lattice instead of the "
               "integer one. It compiles, launches and converges; what it produces is a "
               "half-cell-wrong absorber profile",
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
    "tag": "m_scratch_stores_reordered",
    "old": ("        tl.store(ho0 + idx, own0, mask=live)\n"
            "        tl.store(ho1 + idx, own1, mask=live)\n"),
    "new": ("        tl.store(ho1 + idx, own1, mask=live)\n"
            "        tl.store(ho0 + idx, own0, mask=live)\n"),
    "why": "the scratch Hx and Hy stored in the other order. PREDICTED NULL: they are "
           "different addresses of different volumes and nothing in launch 1 reads "
           "either, so only the instruction order may move -- the control that says a "
           "caught store mutation means something",
}


# ---------------------------------------------------------------------------
# The family adapter
# ---------------------------------------------------------------------------

class Spec(kit.FamilySpec):
    gate = GATE
    family = family
    complex_storage = False
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
        # The high r face (the low face is the axis) and both walled z faces.
        return {"x": {"high": thickness}, "z": thickness}

    def build_weld(self, driver: Any, kernel: Any = None, curl_kernel: Any = None,
                   forced: bool = False, scan_kernel: Any = None) -> Any:
        """The ENGINE route through the predicate, except on the one fixture the
        predicate refuses BY DESIGN -- the withdraw leg's standing integrated electric
        source -- where the gate's from-arrays route builds the same plan so the hoist
        and its two null controls can be driven."""
        fields, pml = driver.fields, driver.pml
        sources = tuple(getattr(driver, "_sources", ()))
        plan = family.plan_cylindrical_real_fused_hd_pair(
            fields, pml, sources, kernel=kernel, curl_kernel=curl_kernel,
            scan_kernel=scan_kernel)
        if plan is not None:
            return plan
        verdict = family.cylindrical_real_fused_hd_pair_coverage(fields, pml, sources)
        if not all("standing integrated" in reason for reason in verdict.reasons):
            raise RuntimeError("the product refused a fixture this gate expects it to "
                               f"admit: {verdict.reasons}")
        arrays = {name: getattr(fields, name)
                  for name in family.ROTATED + family.IN_PLACE + family.CONSTITUTIVE_SOURCES}
        for name, twin in twin_table(fields, family.ROTATED).items():
            arrays["scratch_" + name] = twin
        flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                for axis in "xyz" for stem in ("kms", "sinv", "kps")}
        # The same plan the engine route builds, so the same chunk record it binds.
        from gate_chunk_records import layer_record  # noqa: PLC0415

        plan = family.plan_cylindrical_real_fused_hd_pair_from_arrays(
            arrays, flat, driver.fields.grid.dt / driver.fields.grid.dx,
            float(driver.fields.grid.courant), fields, driver.xp,
            kernel=kernel, curl_kernel=curl_kernel, scan_kernel=scan_kernel,
            chunk_bounds=layer_record(fields.grid, pml),
            scan_starts=cyl.radial_scan_starts(fields, pml, "step_D", None))
        # The from-arrays route starts no H; this gate's fixtures say whether theirs
        # has, so the route takes the engine route's start-of-H state.
        plan._h_state = fields  # noqa: SLF001
        return plan

    def singles(self, driver: Any, forced: bool = False) -> Tuple[Any, Any]:
        constitutive = cyl.plan_cylindrical_constitutive(driver.fields, driver.pml, "H")
        curl = cyl.plan_cylindrical_curl(driver.fields, driver.pml, "step_D")
        return constitutive, curl

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        return family.cylindrical_real_fused_hd_pair_coverage(fields, pml, sources)

    def default_kernels(self, owner: Any) -> Dict[str, Any]:
        from meep_gpu.triton_kernels import kernels as tkernels  # noqa: PLC0415
        from meep_gpu.triton_kernels import (  # noqa: PLC0415
            cylindrical_real_fused_electric_pair as pair_d,
            cylindrical_real_fused_magnetic_pair as pair_b,
        )

        name = type(owner).__name__
        scan = {"scan._kernel": cyl.radial_scan_kernel()}
        if name == "CylindricalRealFusedHdPairPlan":
            # Launch 1, and BOTH certified step_D plans' scan and curl: the run picks
            # one, and the unpicked plan's counters must stay at zero.
            slots: Dict[str, Any] = {"_kernel": family.cyl_real_update_H_scratch_kernel()}
            for index in range(len(owner.step_d_plans)):
                slots[f"step_d_plans.{index}._kernel"] = cyl.cylindrical_curl_kernel()
                slots[f"step_d_plans.{index}.scan._kernel"] = cyl.radial_scan_kernel()
            return slots
        if name == "ConstitutivePlan":
            return {"_kernel": tkernels.constitutive_step}
        if name == "PmlCurlPlan":
            return {"_kernel": tkernels.pml_curl_step}
        if name == "CylindricalCurlPlan":
            return {"_kernel": cyl.cylindrical_curl_kernel(), **scan}
        if name == "CylindricalRealFusedMagneticPairPlan":
            return {"_kernel": pair_b.cyl_real_fused_curl_constitutive_B_kernel(), **scan}
        if name == "CylindricalRealFusedElectricPairPlan":
            return {"_kernel": pair_d.cyl_real_fused_curl_constitutive_D_kernel(), **scan}
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
            {"name": "m_is_one", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, grid=kit.GridView(grid, m=1)),
             "needle": "m = 0 ONLY"},
            {"name": "complex_storage", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, force_complex_fields=True),
             "needle": "complex64"},
            {"name": "inactive_absorber", "fields": fields, "pml": _Inactive(), "sources": (),
             "must_refuse": True, "first_half": "cylindrical constitutive half"},
        ]

    def half_integer_rebind(self, plan: Any, pml: Any) -> Any:
        from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

        plan._kps = tuple(CupyPointer(_flat(getattr(pml, f"kps_{axis}_h")))  # noqa: SLF001
                          for axis in "xyz")
        plan._kms = tuple(CupyPointer(_flat(getattr(pml, f"kms_{axis}_h")))  # noqa: SLF001
                          for axis in "xyz")
        return plan

    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        findings: List[str] = []
        measures: Dict[str, Any] = {}
        # THE CONSTITUTIVE HALF IS IMPORTED, NOT COPIED: its lift is the Cartesian
        # product's, re-checked here because this family launches it.
        certified = cartesian.certified_constitutive_tail()
        lifted = cartesian.lifted_constitutive_tail()
        measures["constitutive_chars"] = len(certified)
        if certified != lifted:
            findings.append("fused_hd_pair._h_cell is no longer kernels.constitutive_step's "
                            "own body with the declared edits")
        # LAUNCH 1 IS THE CARTESIAN WELD'S CONSTITUTIVE BLOCK, statement for statement.
        certified_block = family.certified_weld_block()
        lifted_block = family.lifted_weld_block()
        measures["launch1_statements"] = len(lifted_block)
        if certified_block != lifted_block:
            findings.append("launch 1 is not fused_hd_pair's constitutive block, statement "
                            "for statement")
        if len(lifted_block) < 10:
            findings.append(f"launch 1 is only {len(lifted_block)} statements; the cut "
                            f"anchors have crossed")
        # NO ARITHMETIC IN THIS MODULE'S KERNEL beyond the imported constitutive: no
        # divide, no multiply-add.
        text = Path(family.__file__).read_text(encoding="utf-8")
        for needle in ("tl.math.", " / "):
            if any(needle in line for line in lifted_block):
                findings.append(f"launch 1 spells {needle.strip()!r}")
        # THE RADIAL-SUM DECLARATION IS THE CERTIFIED FAMILY'S OWN TABLE, and the
        # scan and the curl are the certified plan's, built by the certified builder.
        spec = cyl.SUB_STEPS[family.CURL_SUB_STEP]
        if (family.PREFIX_COMPONENT, float(family.PREFIX_IR0)) != (
                spec["prefix_component"], float(spec["prefix_ir0"])):
            findings.append("the product's radial-sum declaration is not cylindrical_triton's")
        measures["radial_sum"] = {"component": family.PREFIX_COMPONENT,
                                  "ir0": family.PREFIX_IR0}
        for needle in ("_cyl.plan_cylindrical_curl_from_arrays(",
                       "_cyl.radial_scan_starts(fields, pml, CURL_SUB_STEP, None)"):
            if needle not in text:
                findings.append(f"the product no longer builds the certified step_D plan "
                                f"through {needle!r}")
        # DEFAULT_BLOCK is pinned against kernels.py's SOURCE.
        tree = ast.parse((API_ROOT / "meep_gpu" / "triton_kernels" / "kernels.py").read_text(
            encoding="utf-8"))
        certified_block = next(
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "DEFAULT_BLOCK")
        measures["default_block"] = family.DEFAULT_BLOCK
        if family.DEFAULT_BLOCK != certified_block:
            findings.append(f"DEFAULT_BLOCK {family.DEFAULT_BLOCK} is not kernels.py's "
                            f"{certified_block}")
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
        curl_text = Path(cyl.__file__).read_text(encoding="utf-8")
        product_text = Path(family.__file__).read_text(encoding="utf-8")
        # 1. THIS PRODUCT LIFTS NO CURL OR SCAN TEXT: launches 2 and 3 are the certified
        #    plan's, run by its own run().
        for needle in ("curl0 =", "curl1 =", "curl2 =", "def cyl_pml_curl_step",
                       "def cyl_radial_difference", "total + "):
            if needle in product_text:
                findings.append(f"the product module carries curl or scan text "
                                f"({needle!r}); launches 2 and 3 must be the certified "
                                f"plan's, not a copy")
        measures["launches_2_and_3_are_the_certified_plans_run"] = (
            "plan.run(guard)" in product_text)
        if not measures["launches_2_and_3_are_the_certified_plans_run"]:
            findings.append("the plan does not run the certified step_D plan")
        # 2. THE CURL READS THE SCAN'S DIFFERENCE AT ITS OWN CELL (no radial tap of its
        #    own), and Dz's curl is zeroed at the axis row by the ownership mask -- read
        #    off the certified text.
        guarded = "curl2 = dtdx * tl.load(rad + idx, mask=live, other=0.0)" in curl_text
        masked = "curl2 = tl.where(at_r, 0.0, curl2)" in curl_text
        measures["curl_reads_the_scans_difference_at_its_own_cell"] = guarded
        measures["target2_zeroed_at_the_axis_row_under_BACKWARD"] = masked
        if not guarded:
            findings.append("the certified curl no longer reads the scan's difference at "
                            "its own cell")
        if not masked:
            findings.append("the certified curl no longer zeroes Dz's curl at the axis row")
        measures["what_this_licenses"] = (
            "the r near ghost a backward radial tap would read at row 0 is masked and "
            "Dz's curl is zeroed there, so no edit to what that ghost serves can be "
            "byte-visible on the D side; the certified family records the same fact. "
            "This product changes none of it: it runs the certified plan over the "
            "scratch H")
        return findings, measures

    def arbitration_incumbents(self) -> Tuple[str, str]:
        return ("fused pair B (cylindrical)", "fused pair D (cylindrical)")

    def product_row(self) -> Dict[str, str]:
        return {"curl_slot": "update_H", "module": "cylindrical_real_fused_hd_pair",
                "coverage": "cylindrical_real_fused_hd_pair_coverage",
                "builder": "plan_cylindrical_real_fused_hd_pair",
                "label": "fused pair H->D (cylindrical)"}


SPEC = Spec()

# The template's names, so the laptop suite reads this gate the way it reads the others.
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
