#!/usr/bin/env python
"""DEVICE byte gate for the Triton Dcyl m = 0 REAL H->D product:
``meep_gpu/triton_kernels/cylindrical_real_fused_hd_pair.py``.

THE FOURTH SEAM ON A CYLINDRICAL ROW, on this backend, for the three real-storage rows
of the ``(cylindrical -> cylindrical PML)`` H->D cell. The product is TWO launches and
one array-module scan: launch 1 (slot ``update_H``) evaluates the certified ``update_H``
body at its own cell into write-only scratch and RECOMPUTES ``Hy`` at the one-cell
backward radial neighbour to form the pre-scan increment of ``cylindrical_rderiv_prefix``;
``xp.cumsum`` then runs UNTOUCHED on the array path's own pooled slot (it is the oracle:
a column-serial device scan is measured NOT to be ``cupy.cumsum``, by ``scan_order``
below); the plan rotates the six ``H``/``f_w_H`` references; launch 2 (slot ``step_D``)
is the certified cylindrical curl kernel, verbatim, reading the rotated ``H`` and the
scan's output. Three launches where the two certified singles and the array-path prefix
take seven.

THE CLAIM is per COMPLETE DRIVER STEP, as uint32 words over EVERY stored volume, against
the array path and three more reference engines driven in lockstep from one seed (the
template's four: array, the two certified singles, the composition the composer installs
today -- both released cylindrical pairs -- and the same slots unfused), over a stated
budget of steps, under BOTH float32 subnormal policies from empty caches, with every
zero beside a control that moves words. Every leg is the H->D template gate's
(``gate_triton_fused_hd_pair.py``), written once in ``triton_cylindrical_hd_gate_kit.py``,
plus the three the cylindrical seam owes: ``increment_stage`` (the FIRST measurement of
this arithmetic as compiled BY TRITON -- the design record's numbers are NVRTC's),
``scan_order`` and the cylindrical floors on every product row.

THE ARITHMETIC UNDER TEST is the real increment ``(Hy_new[i]*w[i] - Hy_new[i-1]*w[i-1])
/ d[i-1]`` with the divide spelled ``tl.math.div_rn`` (correctly rounded, the array
path's ``true_divide``) and ``enable_fp_fusion=False``. The armed alternatives, each
required to move words: Triton's ``/`` (``div.full.f32``), the reciprocal multiply, the
divide distributed over the subtraction, the same source with contraction on, the
reversed difference, a misaligned weight row, the axis row left un-zeroed, the halo read
from the STORED ``Hy`` (the previous step's field) and from the neighbour program's
scratch, the rotation skipped, the scan skipped, the ``ir0 = 0.0`` ladder, the curl bound
to the pre-launch ``H``, the constitutive half on the half-integer lattice, the withdraw
after ``step_D``. The swapped multiply operand order is a MEASURED equivalence (IEEE-754
multiplication commutes bitwise) and the byte-neutral edit reloads the own cell's
scratch instead of reading the register.

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
#: thickness in cells, on the high r face and both z faces.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("nr12_nz16", {"shape": (12, 16), "courant": 0.35, "pml": 3}),
    ("nr9_nz7_c05", {"shape": (9, 7), "courant": 0.5, "pml": 2}),
    ("nr20_nz5_c037", {"shape": (20, 5), "courant": 0.37, "pml": 1}),
    # FIVE PROGRAMS (1280 cells at BLOCK = 256): the backward halo recompute crosses
    # program boundaries here, as it does on every corpus row.
    ("nr40_nz32", {"shape": (40, 32), "courant": 0.35, "pml": 4}),
)
MUTATION_CASE = "nr12_nz16"

#: The launch-1 kernel's function name -- the mutation seam ``kernel=`` reaches.
KERNEL_NAME = "cyl_real_update_H_increment"

# ---------------------------------------------------------------------------
# The increment-stage variants: what launch 1 spells, and every armed alternative
# ---------------------------------------------------------------------------

_PRIMARY_MULTIPLY = "wi = f_i * w_i\nwim1 = f_im1 * w_im1\ndiff = wi - wim1"

INCREMENT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "primary": {
        "multiply": _PRIMARY_MULTIPLY,
        "divide": "value = tl.math.div_rn(diff, d_im1)",
        "expected": "IDENTICAL",
        # The same source with floating-point contraction ON: the subtract of two
        # products contracts one of them into an fma, a different float32 number.
        "also_fusion_on": True, "expected_fusion_on": "DIFFERS",
    },
    "slash_div_full": {
        "multiply": _PRIMARY_MULTIPLY,
        "divide": "value = diff / d_im1",
        "expected": "DIFFERS",
        "why": "Triton's fp32 `/` lowers to div.full.f32 (approximately 2 ulp), not the "
               "correctly rounded division cupy.true_divide performs",
    },
    "reciprocal_multiply": {
        "multiply": _PRIMARY_MULTIPLY,
        "divide": "value = diff * tl.math.div_rn(1.0, d_im1)",
        "expected": "DIFFERS",
        "why": "numpy's complex prescription carried onto the real side (the Metal "
               "verdict's inversion, wrong here): two roundings where the oracle has one",
    },
    "distributed_divide": {
        "multiply": _PRIMARY_MULTIPLY,
        "divide": "value = tl.math.div_rn(wi, d_im1) - tl.math.div_rn(wim1, d_im1)",
        "expected": "DIFFERS",
        "why": "the divide distributed over the subtraction: two quotients rounded "
               "before the difference",
    },
    "swapped_multiply_operands": {
        "multiply": "wi = w_i * f_i\nwim1 = w_im1 * f_im1\ndiff = wi - wim1",
        "divide": "value = tl.math.div_rn(diff, d_im1)",
        "expected": "IDENTICAL",
        "why": "IEEE-754 multiplication commutes bitwise; `field on the left` is a "
               "documentation fact for the real multiply, not a rounding one",
    },
}

MULTIPLY_VARIANTS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# The mutations, each with the case it is scored on and why it must (or must not) bite
# ---------------------------------------------------------------------------

_HALO_CALL = ("        halo = _h_tap(1, i - 1, j, k, inner, hi0, hi1, hi2, wi0, wi1, wi2, "
              "b0, b1, b2,\n"
              "                      kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)\n")

MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m_divide_spelled_with_slash": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        value = tl.math.div_rn(diff, d_im1)\n",
        "new": "        value = diff / d_im1\n",
        "why": "Triton's fp32 `/` is div.full.f32, approximately 2 ulp and not correctly "
               "rounded; the array path's in-place divide is cupy.true_divide, which is. "
               "The NVRTC design record was taken with C `/` under --fmad=false, an IEEE "
               "division, so `/` is the spelling a reader would copy and the one this "
               "backend must refuse",
    },
    "m_reciprocal_multiply": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        value = tl.math.div_rn(diff, d_im1)\n",
        "new": "        value = diff * tl.math.div_rn(1.0, d_im1)\n",
        "why": "the complex family's divide spelling (numpy's a * (1/d)) carried onto the "
               "real side: two roundings where the oracle has one (1,816,293 of 7,193,552 "
               "words in the NVRTC record)",
    },
    "m_divide_distributed": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        value = tl.math.div_rn(diff, d_im1)\n",
        "new": ("        value = tl.math.div_rn(weighted_here, d_im1) - "
                "tl.math.div_rn(weighted_below, d_im1)\n"),
        "why": "the divide distributed over the subtraction (2,114,294 of 7,193,552 words "
               "in the NVRTC record)",
    },
    "m_subtract_reversed": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        diff = weighted_here - weighted_below\n",
        "new": "        diff = weighted_below - weighted_here\n",
        "why": "the difference taken the wrong way round -- the B side's forward "
               "difference on the D side's backward one",
    },
    "m_weight_row_misaligned": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        w_im1 = tl.load(wgt + i - 1, mask=inner, other=0.0)\n",
        "new": "        w_im1 = tl.load(wgt + i, mask=inner, other=0.0)\n",
        "why": "the neighbour's field weighted by the OWN row's r/dr: the cylindrical "
               "derivative's whole content is that the two rows carry different weights",
    },
    "m_axis_row_not_zeroed": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        value = tl.where(inner, value, 0.0)\n",
        "new": "        value = value + 0.0\n",
        "why": "the array path assigns 0 to row 0 (the sum's start); left un-zeroed the "
               "axis row carries Hy[0]*w[0] into every prefix of its column",
    },
    "m_halo_reads_the_stored_Hy": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _HALO_CALL,
        "new": "        halo = tl.load(hi1 + idx - nyz, mask=inner, other=0.0)\n",
        "why": "the backward neighbour read from the STORED pre-launch Hy -- the previous "
               "step's field -- instead of being recomputed through the certified body. "
               "The purity ledger is the floor that makes this a claim about the kernel: "
               "update_H moves Hy at those cells, so the stale read must differ",
    },
    "m_halo_reads_the_neighbours_scratch": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "expected_override": "NULL",
        "old": _HALO_CALL,
        "new": "        halo = tl.load(ho1 + idx - nyz, mask=inner, other=0.0)\n",
        "why": "the backward neighbour read from the scratch this launch is WRITING. "
               "MEASURED NULL 2026-09-07 on this fixture (192 cells, ONE program of 256 "
               "at one warp): the store of the own cell precedes the load of the "
               "neighbour in program order and both lanes sit in the same warp, so the "
               "read sees the store. Across programs it is the block-schedule race the "
               "design refuses -- unobservable deterministically, which is why it is "
               "recorded rather than armed; the STORED-Hy halo is the deterministic "
               "control for `the halo must be a recompute`",
    },
    "m_f_w_H_written_in_place": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(wo1 + idx, src1, mask=live)\n",
        "new": "        tl.store(wi1 + idx, src1, mask=live)\n",
        "why": "the split-field history written into the PRE-LAUNCH buffer: the halo "
               "recompute of a neighbour then reads B where it needs B_prev, and the "
               "rotation publishes an unwritten scratch",
    },
    "m_swapped_multiply_operand_order": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "expected_override": "NULL",
        "old": "        weighted_here = own1 * w_i\n",
        "new": "        weighted_here = w_i * own1\n",
        "why": "MEASURED NULL (0 of 7,193,552 words under both policies in the NVRTC "
               "record) and PREDICTED NULL here: IEEE-754 multiplication commutes bitwise. "
               "`field on the left` is a documentation fact for the real multiply and "
               "this entry is what keeps that from being read as a rounding claim",
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
        "why": "both launches compiled with floating-point contraction enabled: the "
               "increment's subtract of two products contracts one into an fma, the "
               "constitutive accumulations contract, and the certified curl's guard is "
               "measured load-bearing on this family (0/32 identical unguarded)",
    },
    "m_scan_skipped": {
        "target": "host", "host": "scan_skipped", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the increment handed to launch 2 UNSCANNED (cumsum replaced by a copy): "
               "the curl then differences the increment where it must difference the "
               "prefix, and the prefix cannot be algebraically collapsed (the "
               "bz_flat_grouping class)",
    },
    "m_ir0_zero_ladder": {
        "target": "host", "host": "ir0_zero_ladder", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "the two invariant row vectors rebuilt at ir0 = 0.0 (the B side's ladder) "
               "where step_D prefixes Hp at 0.5 -- the same arithmetic on the wrong "
               "half-integer sites",
    },
    "m_curl_reads_the_pre_launch_H": {
        "target": "host", "host": "curl_reads_pre_launch_H", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "why": "launch 2's magnetic pointers (Hx, Hy, Hz for the raw curl and Hp for the "
               "axis add) redirected to the PRE-LAUNCH volumes: the curl consumes the "
               "state the seam's first half has not yet produced -- the curl-first "
               "ordering, driven through the shipped launch code",
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
        "why": "the seam's electric withdraw performed AFTER the launch rather than before "
               "it -- the array-path campaign's own after_step_D null control. Driven on a "
               "fixture carrying an integrated electric source, in the withdraw leg",
    },
}

#: The one edit required NOT to diverge.
BYTE_NEUTRAL: Dict[str, Any] = {
    "tag": "m_reload_own_Hy_from_the_scratch",
    "old": "        weighted_here = own1 * w_i\n",
    "new": "        weighted_here = tl.load(ho1 + idx, mask=live, other=0.0) * w_i\n",
    "why": "the increment's own-cell Hy re-loaded from the scratch this program just "
           "stored, instead of the register. PREDICTED NULL: one program's store and "
           "load of one address are ordered, so this is inert -- and confirming it is "
           "what shows the increment's correctness comes from the RECOMPUTED halo and not "
           "from the register",
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
    increment_variants = INCREMENT_VARIANTS
    multiply_variants = MULTIPLY_VARIANTS

    def driver_keywords(self, keywords: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(keywords)

    def pml_spec(self, keywords: Mapping[str, Any]) -> Any:
        thickness = int(keywords.get("pml", 3))
        # The high r face (the low face is the axis) and both walled z faces.
        return {"x": {"high": thickness}, "z": thickness}

    def build_weld(self, driver: Any, kernel: Any = None, curl_kernel: Any = None,
                   forced: bool = False) -> Any:
        """The ENGINE route through the predicate, except on the one fixture the
        predicate refuses BY DESIGN -- the withdraw leg's standing integrated electric
        source -- where the gate's from-arrays route builds the same plan so the hoist
        and its two null controls can be driven."""
        fields, pml = driver.fields, driver.pml
        sources = tuple(getattr(driver, "_sources", ()))
        plan = family.plan_cylindrical_real_fused_hd_pair(
            fields, pml, sources, kernel=kernel, curl_kernel=curl_kernel)
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
        return family.plan_cylindrical_real_fused_hd_pair_from_arrays(
            arrays, flat, driver.fields.grid.dt / driver.fields.grid.dx, fields, driver.xp,
            fields.scratch, kernel=kernel, curl_kernel=curl_kernel)

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
        if name == "CylindricalRealFusedHdPairPlan":
            return {"_kernel": family.cyl_real_update_H_increment_kernel(),
                    "_curl_kernel": cyl.cylindrical_curl_kernel()}
        if name == "ConstitutivePlan":
            return {"_kernel": tkernels.constitutive_step}
        if name == "PmlCurlPlan":
            return {"_kernel": tkernels.pml_curl_step}
        if name == "CylindricalCurlPlan":
            return {"_kernel": cyl.cylindrical_curl_kernel()}
        if name == "CylindricalRealFusedMagneticPairPlan":
            return {"_kernel": pair_b.cyl_real_fused_curl_constitutive_B_kernel()}
        if name == "CylindricalRealFusedElectricPairPlan":
            return {"_kernel": pair_d.cyl_real_fused_curl_constitutive_D_kernel()}
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
            {"name": "no_step_scratch", "pml": pml, "sources": (), "must_refuse": True,
             "fields": kit.FieldsView(fields, scratch=None), "needle": "StepScratch"},
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

    # -- launch 2's magnetic arguments ------------------------------------------
    # cyl_pml_curl_step(f0, f1, f2, u0, u1, u2, g0, g1, g2, pfx, hp, ...): the three
    # sources at 6..8 and the axis add's Hp (= Hy) at 10.
    def curl_prelaunch_positions(self) -> Dict[int, int]:
        return {6: 0, 7: 1, 8: 2, 10: 1}

    def default_curl_kernel(self) -> Any:
        return cyl.cylindrical_curl_kernel()

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
        statements = family.increment_statements()
        measures["launch1_statements"] = len(statements)
        for tag, needle in family.INCREMENT_SPELLING:
            hits = sum(1 for line in statements if needle in line)
            measures[f"spelling_{tag}"] = hits
            if hits != 1:
                findings.append(f"the increment statement {tag!r} appears {hits} times in "
                                f"the launch-1 kernel, not once")
        # NO `/` BELOW THE DECODE: the divide is div_rn and only div_rn.
        slashes = [line for line in statements
                   if " = " in line and not line.startswith(('"', "'"))
                   and " / " in line.split("#", 1)[0] and "//" not in line]
        measures["slash_divisions_in_launch1"] = len(slashes)
        if slashes:
            findings.append(f"launch 1 divides with `/` (div.full.f32): {slashes}")
        if "tl.math.div_rn(diff, d_im1)" not in "".join(statements):
            findings.append("the increment's divide is no longer tl.math.div_rn")
        # THE PREFIX DECLARATION IS THE CERTIFIED FAMILY'S OWN TABLE.
        spec = cyl.SUB_STEPS[family.CURL_SUB_STEP]
        if (family.PREFIX_COMPONENT, float(family.PREFIX_IR0)) != (
                spec["prefix_component"], float(spec["prefix_ir0"])):
            findings.append("the product's prefix declaration is not cylindrical_triton's")
        if spec["extend_wall_row"]:
            findings.append("the D side's prefix now extends a wall row; this product "
                            "recomputes one backward neighbour only")
        measures["prefix"] = {"component": family.PREFIX_COMPONENT, "ir0": family.PREFIX_IR0}
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
        if family.LAUNCHES_PER_RUN != family.KERNEL_LAUNCHES_PER_RUN + 1:
            findings.append("LAUNCHES_PER_RUN does not count exactly the two kernels and "
                            "the one scan")
        return findings, measures

    def ghost_observability(self) -> Tuple[List[str], Dict[str, Any]]:
        findings: List[str] = []
        measures: Dict[str, Any] = {}
        curl_text = Path(cyl.__file__).read_text(encoding="utf-8")
        product_text = Path(family.__file__).read_text(encoding="utf-8")
        # 1. THIS PRODUCT LIFTS NO CURL TEXT: launch 2 is the shipped kernel OBJECT.
        for needle in ("curl0 =", "curl1 =", "curl2 =", "def cyl_pml_curl_step"):
            if needle in product_text:
                findings.append(f"the product module carries curl text ({needle!r}); launch "
                                f"2 must be the certified kernel object, not a copy")
        measures["launch2_is_the_module_kernel_object"] = (
            "_cyl.cylindrical_curl_kernel()" in product_text)
        if not measures["launch2_is_the_module_kernel_object"]:
            findings.append("the plan does not resolve launch 2 through "
                            "cylindrical_triton.cylindrical_curl_kernel()")
        # 2. THE PREFIX'S BACKWARD RADIAL TAP IS GUARDED, and the target it feeds is
        #    zeroed at the axis row by the ownership mask -- read off the certified text.
        guarded = "p_down = tl.load(pfx + o_r, mask=vr, other=0.0)" in curl_text
        masked = "curl2 = tl.where(at_r, 0.0, curl2)" in curl_text
        measures["prefix_backward_tap_guarded"] = guarded
        measures["target2_zeroed_at_the_axis_row_under_BACKWARD"] = masked
        if not guarded:
            findings.append("the certified curl no longer guards the prefix's backward tap")
        if not masked:
            findings.append("the certified curl no longer zeroes Dz's curl at the axis row")
        measures["what_this_licenses"] = (
            "the r near ghost the prefix's backward tap would read at row 0 is masked and "
            "its target's curl is zeroed there, so no edit to what that ghost serves can "
            "be byte-visible on the D side; the certified family records the same fact "
            "as its axis_ghost_sign null. This product changes none of it: it passes the "
            "scan's output to the unchanged kernel")
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
