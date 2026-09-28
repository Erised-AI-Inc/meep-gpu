#!/usr/bin/env python
"""DEVICE byte gate: the Triton H->D weld on the two COMPLEX beta cells.

TWO BOARD CELLS, ONE FAMILY, TWO EMITTED KERNELS, FOUR SEAM-INSTANCES::

    H_to_D  (folded complex   -> folded complex beta PML)  3
    H_to_D  (complex beta run -> complex beta PML)         1

Read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597).

THE ASYMMETRY IN THE TWO CELL LABELS IS REAL. The folded cell's ``update_H`` arm is
``folded complex`` and NOT "folded complex beta"; the unfolded cell's is ``complex beta
run``. That is what the two shipped constitutive predicates ARE, and
``launch.CERTIFIED_FUSED_PAIR_ARMS`` already carries the same pairing for this cell's
B->H and D->E siblings. The family's module docstring states it; this gate records the
resolved variant per fixture so the record answers "which arm pair did this absorb".

WHAT THE FIXTURES MUST BE. Complex storage, ``dimensions=2`` (``grid`` refuses beta off
an effective-2-D Cartesian grid) and a nonzero ``beta``; the folded variant adds a
MIRROR axis, and the absorber is spelled per face because the near face of a folded
axis is the mirror plane rather than a boundary. No layer on z: that axis is
translationally invariant on a beta run and ``PML.__init__`` refuses one there.

THE BETA COEFFICIENT'S REAL WORD IS A SIGNED ZERO out of Python's complex arithmetic
(``special_kz.beta_curl_coefficients``, :771-772, one rounding at :784). It is passed
through rather than synthesized, and the mutation table arms the synthesis.

THE EXPANSION LICENCE IS POLICY-CONDITIONAL and it is the certified family's: every
complex arm here was certified under ``keep``, and the beta arms additionally need the
BETA expansion pattern, resolved by the VARIANT'S OWN resolver.

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

from meep_gpu.triton_kernels import beta_complex_fused_hd_pair as family  # noqa: E402
from meep_gpu.triton_kernels import complex_fused_hd_pair as _complex  # noqa: E402
from meep_gpu.triton_kernels import folded_complex as _folded  # noqa: E402
from meep_gpu.triton_kernels import special_kz as _kz  # noqa: E402

GATE = "triton_beta_complex_fused_hd_pair"

#: The out-of-plane wavevector every fixture carries.
BETA = 0.27


class BetaComplexProduct(kit.Product):
    """The kit's descriptor for this family."""

    gate = GATE
    family = family
    complex_storage = True
    family_path = "meep_gpu/triton_kernels/beta_complex_fused_hd_pair.py"
    cell_arms = (family.ARMS,) + family.EXTRA_ARMS
    sources = (
        "meep_gpu/triton_kernels/beta_complex_fused_hd_pair.py",
        "meep_gpu/triton_kernels/complex_fused_hd_pair.py",
        "meep_gpu/triton_kernels/fused_hd_pair.py",
        "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
        "meep_gpu/triton_kernels/folded_complex.py",
        "meep_gpu/triton_kernels/special_kz.py",
        "meep_gpu/triton_kernels/complex_fields.py",
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
        "parity/meep_gpu/gate_triton_beta_complex_fused_hd_pair.py",
    )

    fixtures = (
        ("folded_mirror_periodic",
         {"variant": "folded_beta_complex", "symmetry": (("y", 1),),
          "boundaries": {"x": "metallic"}}),
        ("folded_mirror_metallic",
         {"variant": "folded_beta_complex", "symmetry": (("y", -1),),
          "boundaries": {"x": "metallic", "y": "metallic"}}),
        ("unfolded_pp", {"variant": "beta_complex", "boundaries": {}}),
        ("unfolded_mm", {"variant": "beta_complex",
                         "boundaries": {"x": "metallic", "y": "metallic"}}),
    )
    mutation_fixture = "folded_mirror_periodic"

    #: TWO kernels emitted from two lifted bodies that share this text, so a needle in
    #: the curl body matches TWICE: both are edited and the fixture launches one.
    mutations: Dict[str, Dict[str, Any]] = {
        "m_own_cell_reads_the_pre_launch_H": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        a_re = own0_re\n"
                    "        a_im = own0_im\n"),
            "new": ("        a_re = tl.load(hi0 + 2 * idx, mask=live, other=0.0)\n"
                    "        a_im = tl.load(hi0 + 2 * idx + 1, mask=live, "
                    "other=0.0)\n"),
            "why": "the curl's OWN cell reads the pre-launch magnetic field instead "
                   "of the register pair the constitutive half just computed",
        },
        "m_foreign_tap_reads_the_stale_value": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        a_y_re, a_y_im = _h_tap_complex(0, i, sj, k, vy,\n"),
            "new": ("        a_y_re, a_y_im = tl.load(hi0 + 2 * oy, mask=vy, "
                    "other=0.0), tl.load(hi0 + 2 * oy + 1, mask=vy, other=0.0)\n"
                    "        _unused_tap = _h_tap_complex(0, i, sj, k, vy,\n"),
            "why": "one foreign tap reads the PRE-LAUNCH H word pair instead of "
                   "recomputing it -- exactly the value an in-place weld would have "
                   "read at a neighbour the launch had not yet written",
        },
        "m_foreign_tap_takes_the_wrong_word_plane": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": ("        c_y_re, c_y_im = _h_tap_complex(2, i, sj, k, vy,\n"),
            "new": ("        c_y_im, c_y_re = _h_tap_complex(2, i, sj, k, vy,\n"),
            "why": "a tap's two word planes are swapped -- the mis-binding a "
                   "word-pair return makes easy",
        },
        "m_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        tl.store(ho0 + 2 * idx, own0_re, mask=live)\n",
            "new": "        tl.store(hi0 + 2 * idx, own0_re, mask=live)\n",
            "why": "the stepped magnetic field is written into the PRE-LAUNCH buffer "
                   "-- the race the scratch output exists to remove",
        },
        "m_f_w_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 2,
            "old": "        tl.store(wo0 + 2 * idx, src0_re, mask=live)\n",
            "new": "        tl.store(wi0 + 2 * idx, src0_re, mask=live)\n",
            "why": "on this side the newly written f_w_H IS B exactly, so a foreign "
                   "recompute reads B where it needs B_prev",
        },
        "m_beta_real_word_synthesized": {
            "target": "host", "expected": "NULL",
            "why": "MEASURED NULL, 2026-09-08, and it is recorded rather than "
                   "dropped. The beta coefficient's REAL word is replaced by a "
                   "synthesized +0.0 instead of the SIGNED zero Python's complex "
                   "arithmetic produced (special_kz:771-772, one rounding at :784). "
                   "The mechanism: _mul_imag_coefficient_left multiplies that word "
                   "into the field and ADDS the product to the other term, and a "
                   "signed zero's sign survives an add only when the other addend is "
                   "itself a zero of the opposite sign -- which does not occur on "
                   "this fixture's value class. THE HOST-MUTATION ROUTE IS NOT "
                   "DISARMED, and that is what earns this null: m_rotation_skipped "
                   "and m_half_integer_rebind are host mutations on this SAME "
                   "fixture through the SAME launcher seam and both are CAUGHT. What "
                   "the null says is that this seam does not expose the sign, not "
                   "that the harness cannot see a host defect",
        },
        "m_rotation_skipped": {
            "target": "host", "expected": "CAUGHT",
            "why": "the launch happens and the engine's H / f_w_H references are NOT "
                   "moved onto the freshly written scratch",
        },
        "m_half_integer_rebind": {
            "target": "host", "expected": "CAUGHT",
            "why": "the coefficient group is rebound to the HALF-INTEGER sub-lattice: "
                   "a half-cell error in the absorber profile, smooth and wrong",
        },
    }

    byte_neutral: Dict[str, Any] = {
        "hits": 2,
        "old": ("        a_re = own0_re\n"
                "        a_im = own0_im\n"),
        "new": ("        a_re = tl.load(ho0 + 2 * idx, mask=live, other=0.0)\n"
                "        a_im = tl.load(ho0 + 2 * idx + 1, mask=live, other=0.0)\n"),
        "why": "the curl half re-loads its own cell's magnetic word pair from the "
               "scratch this program just wrote. PREDICTED NULL: one program's store "
               "and load of one address are ordered, so this is inert",
    }

    # -- fixtures ------------------------------------------------------------
    def driver_kwargs(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        from meep_gpu.grid import Mirror  # noqa: PLC0415

        return {
            "force_complex_fields": True,
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
        folded = {axis.lower() for axis, _phase in (spec.get("symmetry") or ())}
        return {name: ((0.0, 2.0) if name in folded else 2.0) for name in "xy"}

    # -- plans ---------------------------------------------------------------
    def variant(self, driver: Any) -> Optional[str]:
        resolved, _why = family.resolve_variant(driver.fields, driver.pml)
        return resolved

    def build_weld(self, driver: Any, kernel: Any = None, sources: Any = (),
                   module: Any = None) -> Any:
        owner = module if module is not None else family
        return owner.plan_beta_complex_fused_hd_pair(
            driver.fields, driver.pml, sources=sources, kernel=kernel)

    def singles(self, driver: Any) -> Dict[str, Any]:
        fields, pml = driver.fields, driver.pml
        if self.variant(driver) == "folded_beta_complex":
            return {
                "update_H": _folded.plan_folded_complex_constitutive(
                    fields, pml, "H"),
                "step_D": _folded.plan_folded_beta_bloch_pml_curl(
                    fields, pml, "step_D"),
            }
        return {
            "update_H": _kz.plan_beta_run_complex_constitutive(fields, pml, "H"),
            "step_D": _kz.plan_beta_bloch_pml_curl(fields, pml, "step_D"),
        }

    def singles_arms(self, driver: Any) -> Dict[str, str]:
        arms = family.VARIANT_ARMS[self.variant(driver) or "folded_beta_complex"]
        return {"update_H": arms[0], "step_D": arms[1]}

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        return family.beta_complex_fused_hd_pair_coverage(fields, pml, sources)

    def default_kernel(self, owner: Any, driver: Any) -> Any:
        name = type(owner).__name__
        if name == "BetaComplexFusedHdPairPlan":
            return family.fused_kernel_for(owner.variant)
        if name == "FoldedBetaBlochPmlCurlPlan":
            return _folded.folded_beta_bloch_pml_curl_step
        if name == "BetaBlochPmlCurlPlan":
            return _kz.beta_bloch_pml_curl_step
        from meep_gpu.triton_kernels import complex_fields as _cf  # noqa: PLC0415
        from meep_gpu.triton_kernels import kernels as _kernels  # noqa: PLC0415

        if name == "ComplexConstitutivePlan":
            return _cf.bloch_constitutive_step
        if name == "ComplexPmlCurlPlan":
            return _cf.bloch_pml_curl_step
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
        if tag == "m_beta_real_word_synthesized":
            def wrap(plan: Any) -> Any:
                (bp_re, bp_im), (bm_re, bm_im) = plan.beta_words
                plan.beta_words = ((0.0, bp_im), (0.0, bm_im))
                return plan
            return wrap
        return None

    # -- host legs -----------------------------------------------------------
    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        import re as _re  # noqa: PLC0415

        findings: List[str] = []
        measures: Dict[str, Any] = {"variants": {}}
        for variant in family.VARIANTS:
            certified = family.certified_curl_tail(variant)
            lifted = family.lifted_curl_tail(variant)
            raw = family.raw_curl_tail(variant)
            taps = _complex.halo_taps(raw)
            offsets = _complex.offset_coordinates(raw)
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
                "halo_taps": {name: {plane: list(value)
                                     for plane, value in planes.items()}
                              for name, planes in taps.items()},
                "offsets": {name: list(value) for name, value in offsets.items()},
                "curl_source": str(family.CURL_PATHS[variant].name) + "::"
                               + family.CURL_FUNCTIONS[variant],
            }
        constitutive_equal = (_complex.certified_constitutive_tail()
                              == _complex.lifted_constitutive_tail())
        if not constitutive_equal:
            findings.append(
                "the imported complex constitutive lift no longer equals the "
                "certified body")
        text = (API_ROOT / self.family_path).read_text(encoding="utf-8")
        called = set(_re.findall(r"\b(_(?:mul|rotate|div)_[A-Za-z0-9_]*)\s*\(", text))
        stray = sorted(called - set(family.MULTIPLY_HELPERS))
        if stray:
            findings.append(f"the kernels reach multiply helpers outside the "
                            f"declared set: {stray}")
        codes = {"PERIODIC": _folded.CODE_PERIODIC, "METALLIC": _folded.CODE_METALLIC,
                 "MIRROR_METALLIC": _folded.CODE_MIRROR_METALLIC,
                 "MIRROR_PERIODIC": _folded.CODE_MIRROR_PERIODIC}
        if codes != {"PERIODIC": 0, "METALLIC": 1, "MIRROR_METALLIC": 2,
                     "MIRROR_PERIODIC": 3}:
            findings.append(f"the folded ghost codes are {codes}")
        measures.update({
            "imported_constitutive_lift_equal": constitutive_equal,
            "multiply_helpers_declared": list(family.MULTIPLY_HELPERS),
            "multiply_helpers_reached": sorted(
                called & set(family.MULTIPLY_HELPERS)),
            "ghost_codes": codes,
            "arms": {"primary": list(family.ARMS),
                     "extra": [list(pair) for pair in family.EXTRA_ARMS]},
            "certified_under_subnormal_policy":
                family.CERTIFIED_UNDER_SUBNORMAL_POLICY,
            "the_update_H_arm_of_the_folded_cell": (
                "`folded complex`, not `folded complex beta`: the shipped "
                "constitutive predicate carries no beta clause, and "
                "launch.CERTIFIED_FUSED_PAIR_ARMS already pairs it that way for the "
                "B->H and D->E siblings of this cell"),
        })
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
            {"name": "no_absorber", "fields": fields, "pml": None, "sources": (),
             "must_refuse": True},
        ]

    def arbitration_incumbents(self) -> Dict[str, str]:
        return {"note": "recorded per fixture in composer_selected; this family is "
                        "not in CERTIFIED_FUSED_PRODUCTS, so the composer is never "
                        "offered it"}


PRODUCT = BetaComplexProduct()


def main(argv: Optional[List[str]] = None) -> int:
    return kit.run(PRODUCT, Path(__file__).resolve(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
