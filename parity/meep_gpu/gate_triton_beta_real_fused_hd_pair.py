#!/usr/bin/env python
"""DEVICE byte gate: the Triton H->D weld on the two REAL beta cells.

TWO BOARD CELLS, ONE FAMILY, TWO EMITTED KERNELS::

    H_to_D  (real beta run   -> real beta PML)         1  examples:refl-angular-kz2d.py
    H_to_D  (folded beta run -> folded real beta PML)  1  tests:TestSpecialKz.test_eigsrc_kz_1_real_imag

Read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597).

WHAT IS BEING CERTIFIED. ``meep_gpu/triton_kernels/beta_real_fused_hd_pair.py``
computes the certified ``update_H`` constitutive into launch-local SCRATCH, takes its
own cell's magnetic field from registers, RECOMPUTES every one of the curl half's six
foreign taps from pre-launch state through ``fused_hd_pair._h_cell``, steps
``D``/``fu_D`` in place, and rotates the ``H``/``f_w_H`` references afterwards. Per
COMPLETE DRIVER STEP, as uint32 WORDS over every stored volume the engine allocates --
never ``allclose``.

WHAT THE FIXTURES MUST BE, and it is forced rather than chosen. A real beta run is
``dimensions=2``: ``grid`` refuses ``beta`` off an effective-2-D Cartesian grid, and
``PML.__init__`` refuses a layer on the translationally invariant z axis -- a scalar
``setup_pml(2)`` asks for all six faces and dies before the first step. On a FOLDED
axis the near face is the MIRROR PLANE, not a boundary, so the thickness is spelled
per face from the fixture's own mirror set.

THE LEG SET, THE REFERENCE COUNTING AND WHAT IS RECORDED-RATHER-THAN-ASSERTED are the
shared kit's; see :mod:`triton_hd_tail_gate_kit`.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import triton_hd_tail_gate_kit as kit  # noqa: E402

from meep_gpu.triton_kernels import beta_real_fused_hd_pair as family  # noqa: E402
from meep_gpu.triton_kernels import folded_complex as _folded  # noqa: E402
from meep_gpu.triton_kernels import fused_hd_pair as _plain  # noqa: E402
from meep_gpu.triton_kernels import special_kz as _kz  # noqa: E402

GATE = "triton_beta_real_fused_hd_pair"

#: The out-of-plane wavevector every fixture carries. Nonzero, so ``HAS_BETA`` is the
#: live arm and the two beta scalars are not the trivial pair.
BETA = 0.27


class BetaRealProduct(kit.Product):
    """The kit's descriptor for this family."""

    gate = GATE
    family = family
    complex_storage = False
    family_path = "meep_gpu/triton_kernels/beta_real_fused_hd_pair.py"
    cell_arms = (family.ARMS,) + family.EXTRA_ARMS
    sources = (
        "meep_gpu/triton_kernels/beta_real_fused_hd_pair.py",
        "meep_gpu/triton_kernels/fused_hd_pair.py",
        "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
        "meep_gpu/triton_kernels/special_kz.py",
        "meep_gpu/triton_kernels/folded_complex.py",
        "meep_gpu/triton_kernels/symmetry.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/withdraw_hoist.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/fastpath.py",
        "parity/meep_gpu/triton_hd_tail_gate_kit.py",
        "parity/meep_gpu/gate_triton_fused_hd_pair.py",
        "parity/meep_gpu/gate_triton_beta_real_fused_hd_pair.py",
    )

    fixtures = (
        ("beta_pp", {"variant": "beta", "boundaries": {}}),
        ("beta_mm", {"variant": "beta",
                     "boundaries": {"x": "metallic", "y": "metallic"}}),
        ("beta_mp", {"variant": "beta", "boundaries": {"x": "metallic"}}),
        # THE FOLDED FIXTURES. ``Mirror(Y)`` with a periodic termination gives
        # MIRROR_PERIODIC -- the code whose TOP-PLANE ownership mask is the delta the
        # unfolded kernel does not carry -- and with a metallic one, MIRROR_METALLIC.
        # Both are exercised, because which of the two an axis resolves to is the
        # single point of failure in the folded kernel.
        ("folded_beta_mirror_periodic",
         {"variant": "folded_beta", "boundaries": {"x": "metallic"},
          "symmetry": (("y", 1),)}),
        ("folded_beta_mirror_metallic",
         {"variant": "folded_beta",
          "boundaries": {"x": "metallic", "y": "metallic"},
          "symmetry": (("y", -1),)}),
    )
    mutation_fixture = "folded_beta_mirror_periodic"

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
                   "not welding at all",
        },
        "m_foreign_tap_reads_the_stale_value": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        a_y = _h_tap(0, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, "
                    "wi2, b0, b1, b2,\n"
                    "                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)\n"),
            "new": ("        a_y = tl.load(hi0 + oy, mask=vy, other=0.0)\n"),
            "why": "one foreign tap reads the PRE-LAUNCH H instead of recomputing it "
                   "-- exactly the value an in-place weld would have read at a "
                   "neighbour the launch had not yet written",
        },
        "m_foreign_tap_lands_on_the_own_cell": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        b_x = _h_tap(1, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, "
                    "wi2, b0, b1, b2,\n"),
            "new": ("        b_x = _h_tap(1, i, j, k, vx, hi0, hi1, hi2, wi0, wi1, "
                    "wi2, b0, b1, b2,\n"),
            "why": "the backward-x tap recomputes the program's OWN cell instead of "
                   "its neighbour: the stencil collapses",
        },
        "m_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        tl.store(ho0 + idx, own0, mask=live)\n",
            "new": "        tl.store(hi0 + idx, own0, mask=live)\n",
            "why": "the stepped magnetic field is written into the PRE-LAUNCH buffer "
                   "-- the race the scratch output exists to remove, caught here "
                   "deterministically through the rotation",
        },
        "m_f_w_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        tl.store(wo0 + idx, src0, mask=live)\n",
            "new": "        tl.store(wi0 + idx, src0, mask=live)\n",
            "why": "on this side the newly written f_w_H IS B exactly, so a foreign "
                   "recompute reads B where it needs B_prev",
        },
        "m_folded_codes_replaced_by_the_0_1_mapping": {
            "target": "host", "expected": "CAUGHT",
            "why": "the folded plan's boundary triple is replaced by the unfolded "
                   "0/1 mapping, which sends `mirror` to 0 = PERIODIC: the ghost "
                   "WRAPS to the far plane and neither ownership mask is emitted. "
                   "This compiles, launches and converges -- a half-grid of wrong "
                   "values rather than a crash -- and nothing in the kernel can catch "
                   "it, because 0 is a valid code",
        },
        "m_rotation_skipped": {
            "target": "host", "expected": "CAUGHT",
            "why": "the launch happens and the engine's H / f_w_H references are NOT "
                   "moved onto the freshly written scratch",
        },
        "m_half_integer_rebind": {
            "target": "host", "expected": "CAUGHT",
            "why": "the coefficient group is rebound to the HALF-INTEGER sub-lattice: "
                   "a half-cell error in the absorber profile, smooth and entirely "
                   "wrong",
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
               "this program just wrote. PREDICTED NULL: one program's store and load "
               "of one address are ordered, so this is inert -- and confirming it is "
               "what shows the correctness comes from the FOREIGN taps",
    }

    # -- fixtures ------------------------------------------------------------
    def driver_kwargs(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        from meep_gpu.grid import Mirror  # noqa: PLC0415

        return {
            "force_complex_fields": False,
            # EFFECTIVE 2-D IS FORCED: `grid` refuses beta off it.
            "dimensions": 2,
            "beta": BETA,
            "symmetry": tuple(Mirror(axis, int(phase))
                              for axis, phase in spec.get("symmetry") or ()),
        }

    #: THE Z EXTENT IS ZERO, and ``grid`` is what forces it: ``dimensions=2`` makes z
    #: translationally invariant, and an invariant axis is INFINITE -- a length there
    #: "describes nothing this run can represent" (grid.py:800). Measured on this
    #: lane's first device smoke, where a 1.2 z extent died before the first step.
    default_cell = (2.0, 2.1, 0.0)

    def pml_spec(self, spec: Mapping[str, Any]) -> Any:
        """The absorber, per face.

        NO LAYER ON Z: the axis is translationally invariant on a beta run and
        ``PML.__init__`` refuses one there by name. NO LAYER ON THE NEAR FACE OF A
        FOLDED AXIS: that face is the mirror plane, not a boundary.
        """
        folded = {axis.lower() for axis, _phase in (spec.get("symmetry") or ())}
        return {name: ((0.0, 2.0) if name in folded else 2.0) for name in "xy"}

    # -- plans ---------------------------------------------------------------
    def variant(self, driver: Any) -> Optional[str]:
        resolved, _why = family.resolve_variant(driver.fields, driver.pml)
        return resolved

    def build_weld(self, driver: Any, kernel: Any = None, sources: Any = (),
                   module: Any = None) -> Any:
        owner = module if module is not None else family
        return owner.plan_beta_real_fused_hd_pair(
            driver.fields, driver.pml, sources=sources, kernel=kernel)

    def singles(self, driver: Any) -> Dict[str, Any]:
        fields, pml = driver.fields, driver.pml
        if self.variant(driver) == "folded_beta":
            return {
                "update_H": _folded.plan_folded_beta_run_constitutive(
                    fields, pml, "H"),
                "step_D": _folded.plan_folded_beta_pml_curl(fields, pml, "step_D"),
            }
        return {
            "update_H": _kz.plan_beta_run_constitutive(fields, pml, "H"),
            "step_D": _kz.plan_beta_pml_curl(fields, pml, "step_D"),
        }

    def singles_arms(self, driver: Any) -> Dict[str, str]:
        arms = family.VARIANT_ARMS[self.variant(driver) or "beta"]
        return {"update_H": arms[0], "step_D": arms[1]}

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        return family.beta_real_fused_hd_pair_coverage(fields, pml, sources)

    def default_kernel(self, owner: Any, driver: Any) -> Any:
        name = type(owner).__name__
        if name == "BetaRealFusedHdPairPlan":
            return family.fused_kernel_for(owner.variant)
        if name == "BetaPmlCurlPlan":
            return _kz.beta_pml_curl_step
        if name == "FoldedBetaPmlCurlPlan":
            return _folded.folded_beta_pml_curl_step
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
        if tag == "m_folded_codes_replaced_by_the_0_1_mapping":
            def wrap(plan: Any) -> Any:
                # The unfolded mapping, spelled the way every unfolded plan spells it.
                plan.bc = tuple(0 if code in (_folded.CODE_PERIODIC,
                                              _folded.CODE_MIRROR_PERIODIC) else 1
                                for code in plan.bc)
                return plan
            return wrap
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
                "lift_equal": certified == lifted,
                "halo_taps": {name: list(value) for name, value in taps.items()},
                "offsets": {name: list(value) for name, value in offsets.items()},
                "curl_source": str(family.CURL_PATHS[variant].name) + "::"
                               + family.CURL_FUNCTIONS[variant],
            }
        constitutive_equal = (_plain.certified_constitutive_tail()
                              == _plain.lifted_constitutive_tail())
        if not constitutive_equal:
            findings.append(
                "the imported constitutive lift no longer equals the certified body")
        measures["imported_constitutive_lift_equal"] = constitutive_equal
        # THE FOUR GHOST CODES, pinned equal to the certified module's own integers.
        codes = {"PERIODIC": _folded.CODE_PERIODIC, "METALLIC": _folded.CODE_METALLIC,
                 "MIRROR_METALLIC": _folded.CODE_MIRROR_METALLIC,
                 "MIRROR_PERIODIC": _folded.CODE_MIRROR_PERIODIC}
        if codes != {"PERIODIC": 0, "METALLIC": 1, "MIRROR_METALLIC": 2,
                     "MIRROR_PERIODIC": 3}:
            findings.append(f"the folded ghost codes are {codes}")
        measures["ghost_codes"] = codes
        measures["arms"] = {"primary": list(family.ARMS),
                            "extra": [list(pair) for pair in family.EXTRA_ARMS]}
        return findings, measures

    def refusal_cases(self, driver: Any) -> List[Dict[str, Any]]:
        fields, pml = driver.fields, driver.pml

        class _View:
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

        beta_free = _View(fields, grid=_View(fields.grid, beta=0.0))
        complex_view = _View(fields, force_complex_fields=True)
        return [
            {"name": "admits_the_fixture", "fields": fields, "sources": (),
             "must_refuse": False},
            {"name": "undeclared_source_list", "fields": fields, "sources": None,
             "must_refuse": True, "needle": "the source set was not declared"},
            {"name": "standing_integrated_electric_withdraw", "fields": fields,
             "sources": (_IntegratedElectric(),), "must_refuse": True,
             "needle": "standing integrated"},
            {"name": "beta_zero", "fields": beta_free, "sources": (),
             "must_refuse": True, "needle": "beta"},
            {"name": "complex_storage", "fields": complex_view, "sources": (),
             "must_refuse": True, "needle": "complex"},
            {"name": "no_absorber", "fields": fields, "pml": None, "sources": (),
             "must_refuse": True},
        ]

    def arbitration_incumbents(self) -> Dict[str, str]:
        return {"note": "recorded per fixture in composer_selected; this family is "
                        "not in CERTIFIED_FUSED_PRODUCTS, so the composer is never "
                        "offered it"}


PRODUCT = BetaRealProduct()


def main(argv: Optional[List[str]] = None) -> int:
    return kit.run(PRODUCT, Path(__file__).resolve(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
