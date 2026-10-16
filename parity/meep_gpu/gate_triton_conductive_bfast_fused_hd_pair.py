#!/usr/bin/env python
"""DEVICE byte gate: the Triton H->D weld on the CONDUCTIVE and BFAST curl tails.

TWO BOARD CELLS, ONE FAMILY, TWO EMITTED KERNELS::

    H_to_D  (ordinary  -> conductive PML)   1 instance  tests:TestAdjointSolver.test_damping
    H_to_D  (BFAST run -> BFAST PML)        1 instance  tests:TestReflectanceAngular.test_reflectance_angular_2_35_7

Read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597).

WHAT IS BEING CERTIFIED. ``meep_gpu/triton_kernels/conductive_bfast_fused_hd_pair.py``
computes the certified ``update_H`` constitutive into launch-local SCRATCH, takes its
own cell's magnetic field from registers, RECOMPUTES every one of the curl half's six
foreign taps from pre-launch state through ``fused_hd_pair._h_cell``, steps
``D``/``fu_D`` (and the variant's own in-place state) in place, and rotates the
``H``/``f_w_H`` references afterwards. The claim is per COMPLETE DRIVER STEP as uint32
WORDS over every stored volume the engine allocates -- never ``allclose``, because
``-0.0 == 0.0`` lies.

THE LEG SET, THE REFERENCE COUNTING AND WHAT IS RECORDED-RATHER-THAN-ASSERTED are the
shared kit's; see :mod:`triton_hd_tail_gate_kit`. In particular the arbitration leg
does NOT assert that the composer refuses this product, because on this backend an
unwired family is never OFFERED to the composer and that clause would invert the
moment the wiring lands.

    CUDA_VISIBLE_DEVICES=<n> TRITON_LIBCUDA_PATH=$HOME/triton_libcuda_stub \\
      CUPY_CACHE_DIR=<fresh>/cupy_cache/ftz_stripped PYTHONPATH=. \\
      python -u parity/meep_gpu/gate_triton_conductive_bfast_fused_hd_pair.py \\
      --subnormal-policy keep --out <fresh>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import triton_hd_tail_gate_kit as kit  # noqa: E402

from meep_gpu.triton_kernels import bfast_curl as _bfast  # noqa: E402
from meep_gpu.triton_kernels import conductive_bfast_fused_hd_pair as family  # noqa: E402
from meep_gpu.triton_kernels import conductivity as _conductivity  # noqa: E402
from meep_gpu.triton_kernels import fused_hd_pair as _plain  # noqa: E402
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402

# `kernels` imports Triton at module scope, so it CANNOT be imported here: the host
# legs are the merge bar and they run on a laptop with no Triton. It is imported
# inside `default_kernel`, which only a device leg reaches.

GATE = "triton_conductive_bfast_fused_hd_pair"

#: The BFAST wavevector the fixtures carry. Nonzero on ONE axis, so the six
#: coefficients are not all the trivial pair and the ``HAS_BFAST`` arm is live.
BFAST_K: Tuple[float, float, float] = (0.3, 0.0, 0.0)

#: The uniform D conductivity the conductive fixtures install, in MEEP's sigma_D
#: units. Nonzero, so every branch of ``_conductive_component``'s case table that a
#: lossy component can reach is live rather than compiled out.
SIGMA_D = 0.6


class ConductiveBfastProduct(kit.Product):
    """The kit's descriptor for this family."""

    gate = GATE
    family = family
    complex_storage = False
    family_path = "meep_gpu/triton_kernels/conductive_bfast_fused_hd_pair.py"
    cell_arms = (family.ARMS,) + family.EXTRA_ARMS
    sources = (
        "meep_gpu/triton_kernels/conductive_bfast_fused_hd_pair.py",
        "meep_gpu/triton_kernels/fused_hd_pair.py",
        "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
        "meep_gpu/triton_kernels/conductivity.py",
        "meep_gpu/triton_kernels/bfast_curl.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/forms.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/withdraw_hoist.py",
        # THE DRIVER IS BOUND ON PURPOSE. This weld's whole REPLACES claim is about
        # WHICH passes run between update_H and step_D, and the reference this gate
        # compares against is that call order.
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/fastpath.py",
        "parity/meep_gpu/triton_hd_tail_gate_kit.py",
        "parity/meep_gpu/gate_triton_fused_hd_pair.py",
        "parity/meep_gpu/gate_triton_conductive_bfast_fused_hd_pair.py",
    )

    #: The fixtures. FOUR boundary triples per variant rather than eight, because the
    #: ghost rule is what specialisation changes and the four here already carry every
    #: axis both ways; the budget is what the whole-triple sweep costs on two kernels.
    fixtures = (
        ("cond_ppp", {"variant": "conductive", "boundaries": {}}),
        ("cond_mmm", {"variant": "conductive",
                      "boundaries": {axis: "metallic" for axis in "xyz"}}),
        ("cond_mixed", {"variant": "conductive", "boundaries": {"x": "metallic"},
                        "sigma_components": ("Dx", "Dz")}),
        ("bfast_ppp", {"variant": "bfast", "boundaries": {}}),
        ("bfast_mmm", {"variant": "bfast",
                       "boundaries": {axis: "metallic" for axis in "xyz"}}),
        # A CONDUCTIVITY ONLY SOME CHUNKS KEEP: sigma on the two low-x planes, which
        # lie inside the low-x layer's chunks, so every other chunk keeps none and the
        # certified curl's third test reads the conductive-chunk volume (CMASK 1) on
        # every conductive target.
        ("cond_partial", {"variant": "conductive",
                          "boundaries": {axis: "metallic" for axis in "xyz"},
                          "sigma_low_x_planes": 2}),
        # THE RUN'S FIRST update_H on the mutation fixture's grid: unseparated, so
        # every arrangement performs MEEP's start of H at step 1. Every fixture above
        # is seeded mid-run.
        ("cond_mmm_first_update", {"variant": "conductive",
                                   "boundaries": {axis: "metallic" for axis in "xyz"},
                                   "h_started": False}),
    )
    #: The case the mutations are armed on: every axis walled, so the metallic ghost
    #: rule is live on every axis and every line the shipped kernel can emit is
    #: present.
    mutation_fixture = "cond_mmm"

    #: The mutations. ``hits`` is 2 wherever the needle is in the lifted curl body,
    #: because this family emits TWO kernels from two lifted bodies that share that
    #: text -- both are edited and the fixture launches one. See the kit's
    #: ``mutated_family``.
    mutations: Dict[str, Dict[str, Any]] = {
        "m_own_cell_reads_the_pre_launch_H": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        a = own0\n"
                    "        b = own1\n"
                    "        c = own2\n"),
            "new": ("        a = tl.load(hi0 + idx, mask=live, other=0.0)\n"
                    "        b = tl.load(hi1 + idx, mask=live, other=0.0)\n"
                    "        c = tl.load(hi2 + idx, mask=live, other=0.0)\n"),
            "why": "the curl's OWN cell reads the pre-launch magnetic field instead "
                   "of the register the constitutive half just computed -- the weld "
                   "not welding at all. A null here would mean update_H moves nothing "
                   "on this fixture and every other leg is measuring a no-op",
        },
        "m_foreign_tap_reads_the_stale_value": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        a_y = _h_tap(0, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, "
                    "wi2, b0, b1, b2,\n"
                    "                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, "
                    "ny, nz)\n"),
            "new": ("        a_y = tl.load(hi0 + oy, mask=vy, other=0.0)\n"),
            "why": "one foreign tap reads the PRE-LAUNCH H instead of recomputing it "
                   "-- exactly the value an in-place weld would have read at a "
                   "neighbour the launch had not yet written. This is the defect the "
                   "whole design removes, so it MUST be caught",
        },
        "m_foreign_tap_lands_on_the_own_cell": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        b_x = _h_tap(1, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, "
                    "wi2, b0, b1, b2,\n"),
            "new": ("        b_x = _h_tap(1, i, j, k, vx, hi0, hi1, hi2, wi0, wi1, "
                    "wi2, b0, b1, b2,\n"),
            "why": "the backward-x tap recomputes the program's OWN cell instead of "
                   "its neighbour: the stencil collapses. The coordinates are PARSED "
                   "out of the certified body's own index lines, so this is the "
                   "failure mode a hand-written coordinate would have",
        },
        "m_foreign_tap_takes_the_wrong_component": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        c_y = _h_tap(2, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, "
                    "wi2, b0, b1, b2,\n"),
            "new": ("        c_y = _h_tap(0, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, "
                    "wi2, b0, b1, b2,\n"),
            "why": "a tap returns the wrong magnetic component -- the mis-selection a "
                   "six-value return makes easy, and one the curl consumes without "
                   "complaint",
        },
        "m_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        tl.store(ho0 + idx, own0, mask=live)\n",
            "new": "        tl.store(hi0 + idx, own0, mask=live)\n",
            "why": "the stepped magnetic field is written into the PRE-LAUNCH buffer, "
                   "so a foreign recompute at a neighbour reads a value another "
                   "program wrote -- the race the scratch output exists to remove. "
                   "It is caught here through the ROTATION (the plan then rotates a "
                   "stale twin in), which is what makes it deterministic rather than "
                   "schedule-dependent",
        },
        "m_f_w_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        tl.store(wo0 + idx, tl.where(dw0, src0, tl.load(\n",
            "new": "        tl.store(wi0 + idx, tl.where(dw0, src0, tl.load(\n",
            "why": "the split-field history is written into the PRE-LAUNCH buffer. On "
                   "this side the newly written f_w_H IS B exactly, so a foreign "
                   "recompute reads B where it needs B_prev -- the easily missed half "
                   "of the hazard",
        },
        "a1_pairwise_curl_grouping": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)\n",
            "new": "        curl0 = dtdx * ((c_y - c) + (b - b_z))\n",
            "why": "INVERTED (KERNELS.md 7.1): Dx's curl takes the PREVIOUS package "
                   "grouping, the pairwise `(g1s - g1) + (g2 - g2s)`, in place of MEEP's "
                   "left-to-right `((g1s - g1) + g2) - g2s` that forms.curl_stencil spells",
        },
        "b1_whole_grid_recurrence": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        du0 = record_in(cb, lattice + 8, k)\n",
            "new": "        du0 = idx >= 0\n",
            "why": "the select is removed for Dx: the split-field recurrence runs at EVERY "
                   "cell and fu_Dx is read and written everywhere, as the 0.9.2 kernels ran it",
        },
        "b3_membership_one_row_off": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        du0 = record_in(cb, lattice + 8, k)\n",
            "new": "        du0 = record_in(cb, lattice + 8, k + 1)\n",
            "why": "Dx's membership is read one row off along its dsigu axis (z), so the "
                   "boundary plane of each chunk run changes form",
        },
        "b6_fu_store_unmasked": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1, "case": "bfast_mmm",
            "old": "        tl.store(u0 + idx, n0, mask=live & du0)\n",
            "new": "        tl.store(u0 + idx, n0, mask=live)\n",
            "why": "fu_Dx is stored on a plain cell, which the auxiliary contract leaves "
                   "untouched; every fixture volume is seeded at every cell",
        },
        "b7_fw_scratch_stored_unmasked": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        tl.store(wo0 + idx, tl.where(dw0, src0, tl.load(\n"
                    "            wi0 + idx, mask=live & (dw0 == 0), other=0.0)), "
                    "mask=live)\n"),
            "new": "        tl.store(wo0 + idx, src0, mask=live)\n",
            "why": "the scratch f_w_Hx takes B at every cell, as the 0.9.2 kernels wrote it, "
                   "instead of B on the dsigw cells and the pre-launch word everywhere else; "
                   "the rotation publishes the moved history",
        },
        "b11_start_of_H_not_forwarded": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "case": "cond_mmm_first_update",
            "old": ("            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)\n"
                    "        tl.store(ho0 + idx, own0, mask=live)\n"),
            "new": ("            kp0, kp1, kp2, kmx, kmy, kmz, cb, 0, ny, nz)\n"
                    "        tl.store(ho0 + idx, own0, mask=live)\n"),
            "why": "the own cell's constitutive is handed 0 for MEEP's start of H, so the "
                   "run's first update_H accumulates onto the stored H at the program's own "
                   "cell while every foreign recompute starts it from B. Scored on the "
                   "first-update fixture",
        },
        "n1_fu_load_unmasked": {
            "target": "kernel", "expected": "NULL", "hits": 1, "case": "bfast_mmm",
            "old": "        p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)\n",
            "new": "        p0 = tl.load(u0 + idx, mask=live, other=0.0)\n",
            "why": "PREDICTED NULL (KERNELS.md 7.2, the unconditional auxiliary load): off "
                   "the dsigu chunk the selected form never consumes fu_Dx and its store "
                   "stays masked; the guard is pinned by the lift",
        },
        "c1_conductive_chunks_ignored": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "case": "cond_partial",
            "old": ("        _conductive_component(f0, u0, c0, cf0, ci0, m0, idx, live, "
                    "curl0,\n"
                    "                              km_y, si_y, km_z, si_z, ds0, du0, "
                    "COND0, CMASK0)\n"),
            "new": ("        _conductive_component(f0, u0, c0, cf0, ci0, m0, idx, live, "
                    "curl0,\n"
                    "                              km_y, si_y, km_z, si_z, ds0, du0, "
                    "COND0, 0)\n"),
            "why": "Dx's conductive forms ignore the conductive-chunk volume, so a "
                   "chunk that keeps no conductivity steps the conductive form there "
                   "(and moves f_cond_Dx) where MEEP steps the lossless one. Scored on "
                   "the fixture whose conductivity only the low-x chunks keep",
        },
        "f1_bfast_pass_dropped": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1, "case": "bfast_mmm",
            "old": "            v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)\n",
            "new": "",
            "why": "Dx's BFAST pass is dropped from the welded curl: MEEP runs it on each "
                   "component right after its curl (step_db.cpp:124-142)",
        },
        "m_rotation_skipped": {
            "target": "host", "expected": "CAUGHT",
            "why": "the launch happens and the engine's H / f_w_H references are NOT "
                   "moved onto the freshly written scratch, so the next step steps "
                   "from last step's state. The rotation is the plan's own "
                   "choreography and nothing in the kernel text can catch it",
        },
        "m_half_integer_rebind": {
            "target": "host", "expected": "CAUGHT",
            "why": "the coefficient group is rebound to the HALF-INTEGER sub-lattice. "
                   "Both halves sit on the integer one, so this compiles, launches "
                   "and converges: it is a half-cell error in the absorber profile, "
                   "smooth and entirely wrong. plan_* ASSERTS the two suffixes agree; "
                   "this arms the assertion",
        },
    }

    byte_neutral: Dict[str, Any] = {
        "hits": 2,
        "old": ("        a = own0\n"
                "        b = own1\n"
                "        c = own2\n"),
        "new": ("        a = tl.load(ho0 + idx, mask=live, other=0.0)\n"
                "        b = tl.load(ho1 + idx, mask=live, other=0.0)\n"
                "        c = tl.load(ho2 + idx, mask=live, other=0.0)\n"),
        "why": "the curl half re-loads its own cell's magnetic field from the scratch "
               "this program just wrote, instead of using the register. PREDICTED "
               "NULL: one program's store and load of one address are ordered, so "
               "this is inert -- and confirming it is what shows the correctness "
               "comes from the FOREIGN taps and not from the register",
    }

    # -- fixtures ------------------------------------------------------------
    def driver_kwargs(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        keywords: Dict[str, Any] = {"force_complex_fields": False, "dimensions": 3}
        if spec.get("variant") == "bfast":
            keywords["bfast_scaled_k"] = BFAST_K
        return keywords

    def pml_spec(self, spec: Mapping[str, Any]) -> Any:
        return 2

    def configure(self, driver: Any, spec: Mapping[str, Any]) -> None:
        """Install the D conductivity the conductive variant needs.

        THE BFAST FIXTURES INSTALL NONE, and that is forced rather than tidy:
        ``bfast_pml_curl_coverage`` refuses a conductivity on its own targets by name,
        so a bfast fixture carrying one would be refused before any byte is compared.
        """
        if spec.get("variant") != "conductive":
            return
        shape = tuple(int(n) for n in driver.grid.shape)
        xp = driver.xp
        host = np.full(shape, np.float32(SIGMA_D), np.float32)
        planes = int(spec.get("sigma_low_x_planes", 0))
        if planes:
            host[planes:, :, :] = np.float32(0.0)
        volume = xp.asarray(host)
        components = spec.get("sigma_components")
        if components:
            driver.fields.set_d_conductivity(
                {name: volume for name in components})
        else:
            driver.fields.set_d_conductivity(volume)

    # -- plans ---------------------------------------------------------------
    def variant(self, driver: Any) -> Optional[str]:
        resolved, _why = family.resolve_variant(driver.fields, driver.pml)
        return resolved

    def build_weld(self, driver: Any, kernel: Any = None, sources: Any = (),
                   module: Any = None) -> Any:
        owner = module if module is not None else family
        return owner.plan_conductive_bfast_fused_hd_pair(
            driver.fields, driver.pml, sources=sources, kernel=kernel)

    def singles(self, driver: Any) -> Dict[str, Any]:
        fields, pml = driver.fields, driver.pml
        resolved = self.variant(driver)
        if resolved == "bfast":
            return {
                "update_H": _bfast.plan_bfast_run_constitutive(fields, pml, "H"),
                "step_D": _bfast.plan_bfast_pml_curl(fields, pml, "step_D"),
            }
        return {
            "update_H": triton_launch.plan_constitutive(fields, pml, "H"),
            "step_D": _conductivity.plan_conductive_pml_curl(fields, pml, "step_D"),
        }

    def singles_arms(self, driver: Any) -> Dict[str, str]:
        arms = family.VARIANT_ARMS[self.variant(driver) or "conductive"]
        return {"update_H": arms[0], "step_D": arms[1]}

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        return family.conductive_bfast_fused_hd_pair_coverage(fields, pml, sources)

    def default_kernel(self, owner: Any, driver: Any) -> Any:
        name = type(owner).__name__
        if name == "ConductiveBfastFusedHdPairPlan":
            return family.fused_kernel_for(owner.variant)
        if name == "ConductivePmlCurlPlan":
            return _conductivity.conductive_pml_curl_step
        if name == "BfastPmlCurlPlan":
            return _bfast.bfast_pml_curl_step
        from meep_gpu.triton_kernels import kernels as _kernels  # noqa: PLC0415

        if name == "ConstitutivePlan":
            return _kernels.constitutive_step
        if name == "PmlCurlPlan":
            return _kernels.pml_curl_step
        return None

    def host_launcher(self, tag: str,
                      driver: Any) -> Optional[Callable[[Any], Any]]:
        if tag == "m_rotation_skipped":
            return kit.RotationSkipped
        if tag == "m_half_integer_rebind":
            return kit.half_integer_rebind(driver.pml)
        return None

    # -- host legs -----------------------------------------------------------
    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        findings: List[str] = []
        measures: Dict[str, Any] = {"variants": {}}
        for variant in family.VARIANTS:
            certified = family.certified_curl_tail(variant)
            lifted = family.lifted_curl_tail(variant)
            raw = family.raw_curl_tail(variant)
            taps = _plain.halo_taps(raw)
            offsets = _plain.offset_coordinates(raw)
            if certified != lifted:
                findings.append(
                    f"{variant}: the fused kernel's curl body is NOT the certified "
                    f"body with the declared redirects")
            for target in range(3):
                if f"g{target} +" in lifted:
                    findings.append(f"{variant}: g{target} survives below the seam")
            measures["variants"][variant] = {
                "certified_chars": len(certified),
                "certified_statements": certified.count("\n"),
                "lift_equal": certified == lifted,
                "own_load_edits": len(_plain.OWN_LOAD_EDITS),
                "halo_taps": {name: list(value) for name, value in taps.items()},
                "offsets": {name: list(value) for name, value in offsets.items()},
                "curl_source": str(family.CURL_PATHS[variant].name) + "::"
                               + family.CURL_FUNCTIONS[variant],
                "decode_anchor_indent": (
                    len(family.DECODE_END[variant])
                    - len(family.DECODE_END[variant].lstrip())),
            }
        # THE CONSTITUTIVE LIFT IS THE PLAIN PRODUCT'S, re-checked here rather than
        # inherited: this family IMPORTS `_h_cell`/`_h_tap`, so a drift there is a
        # drift in this family's arithmetic.
        constitutive_equal = (_plain.certified_constitutive_tail()
                              == _plain.lifted_constitutive_tail())
        if not constitutive_equal:
            findings.append(
                "the imported constitutive lift no longer equals the certified body")
        measures["imported_constitutive_lift_equal"] = constitutive_equal
        measures["arms"] = {"primary": list(family.ARMS),
                            "extra": [list(pair) for pair in family.EXTRA_ARMS]}
        measures["bfast_states_from_the_certified_table"] = list(family.BFAST_STATES)
        return findings, measures

    def refusal_cases(self, driver: Any) -> List[Dict[str, Any]]:
        """Every refusal this predicate must make, BY NAME and in both directions."""
        fields, pml = driver.fields, driver.pml

        class _Grid:
            def __init__(self, inner: Any, **overrides: Any) -> None:
                object.__setattr__(self, "_inner", inner)
                object.__setattr__(self, "_overrides", overrides)

            def __getattr__(self, name: str) -> Any:
                overrides = object.__getattribute__(self, "_overrides")
                if name in overrides:
                    return overrides[name]
                return getattr(object.__getattribute__(self, "_inner"), name)

        class _Fields:
            def __init__(self, inner: Any, **overrides: Any) -> None:
                object.__setattr__(self, "_inner", inner)
                object.__setattr__(self, "_overrides", overrides)

            def __getattr__(self, name: str) -> Any:
                overrides = object.__getattribute__(self, "_overrides")
                if name in overrides:
                    return overrides[name]
                return getattr(object.__getattribute__(self, "_inner"), name)

        class _IntegratedElectric:
            """A source whose withdraw DOES WORK, by ``withdraw_hoist``'s own three
            conditions (a callable ``withdraw``, ``is_integrated``, and at least one
            source point). The third is not decoration: a stand-in without it is
            ADMITTED, and the refusal leg would then be measuring nothing -- caught on
            this lane's first device smoke, 2026-09-08."""

            component = "Ez"
            is_integrated = True
            field_type = "D"
            _n_source_points = 4

            def withdraw(self, _fields: Any) -> None:
                return None

        folded = _Fields(fields, grid=_Grid(fields.grid,
                                            is_mirrored=lambda axis: axis == 1,
                                            has_symmetry=lambda: True))
        return [
            {"name": "admits_the_fixture", "fields": fields, "sources": (),
             "must_refuse": False},
            {"name": "undeclared_source_list", "fields": fields, "sources": None,
             "must_refuse": True, "needle": "the source set was not declared"},
            {"name": "standing_integrated_electric_withdraw", "fields": fields,
             "sources": (_IntegratedElectric(),), "must_refuse": True,
             "needle": "standing integrated"},
            {"name": "folded_grid", "fields": folded, "sources": (),
             "must_refuse": True, "needle": "is folded"},
            {"name": "no_absorber", "fields": fields, "pml": None, "sources": (),
             "must_refuse": True},
        ]

    def arbitration_incumbents(self) -> Dict[str, str]:
        return {"note": "recorded per fixture in composer_selected; this family is "
                        "not in CERTIFIED_FUSED_PRODUCTS, so the composer is never "
                        "offered it and no refusal text is expected"}


PRODUCT = ConductiveBfastProduct()


def main(argv: Optional[List[str]] = None) -> int:
    return kit.run(PRODUCT, Path(__file__).resolve(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
