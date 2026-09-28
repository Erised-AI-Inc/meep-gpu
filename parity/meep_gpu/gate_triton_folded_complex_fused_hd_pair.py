#!/usr/bin/env python
"""DEVICE byte gate: the Triton H->D weld on a FOLDED COMPLEX grid.

TWO BOARD CELLS, ONE FAMILY, **ONE** EMITTED KERNEL, FIVE SEAM-INSTANCES::

    H_to_D  (folded complex              -> folded complex PML)              2
    H_to_D  (folded complex off-diagonal -> folded complex off-diagonal PML) 3

Read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597).

THE TWO CELLS ARE ONE LAUNCH, and that is the strongest form of the extra-arm claim
this package makes: the shipped off-diagonal builders say so in their own words --
"The kernel and the plan class are K1's, untouched; only the ADMISSION is new". This
gate therefore drives fixtures of BOTH admissions through the SAME kernel and records
which admission each resolved to.

WHAT THE FIXTURES MUST BE. Complex storage, at least one MIRROR axis, ``beta == 0``.
The near face of a folded axis is the mirror plane, not a boundary, so the absorber is
spelled per face. Both mirror codes are exercised -- ``MIRROR_PERIODIC`` (whose
TOP-PLANE ownership mask is a delta the unfolded kernel does not carry) and
``MIRROR_METALLIC`` -- because which of the two an axis resolves to is the single
point of failure in the folded kernel.

WHY THIS GATE COMPARES ``fu_D``. :mod:`.symmetry`'s own docstring says the two folded
masks are INVISIBLE at whole-step granularity, because the driver's fill passes
overwrite exactly the planes they protect. That statement is about the TARGET and is
NOT true of the AUXILIARY: ``curlN`` feeds both ``nN`` (stored to ``fu_D``) and ``vN``
(stored to ``D``), and the fills rewrite ``D`` at those planes and nothing of
``fu_D``. Every stored volume the engine allocates is compared, so both mask drops are
observable, and each is armed.

THE EXPANSION LICENCE IS POLICY-CONDITIONAL and it is the certified family's: every
complex arm in this package was certified under ``keep``, so under ``flush`` the
predicates this product conjoins refuse by that name and the record says so rather
than reporting a silent zero.

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

import numpy as np

HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import triton_hd_tail_gate_kit as kit  # noqa: E402

from meep_gpu.triton_kernels import complex_fused_hd_pair as _complex  # noqa: E402
from meep_gpu.triton_kernels import folded_complex as _folded  # noqa: E402
from meep_gpu.triton_kernels import folded_complex_fused_hd_pair as family  # noqa: E402

GATE = "triton_folded_complex_fused_hd_pair"


class FoldedComplexProduct(kit.Product):
    """The kit's descriptor for this family."""

    gate = GATE
    family = family
    complex_storage = True
    family_path = "meep_gpu/triton_kernels/folded_complex_fused_hd_pair.py"
    cell_arms = (family.ARMS,) + family.EXTRA_ARMS
    sources = (
        "meep_gpu/triton_kernels/folded_complex_fused_hd_pair.py",
        "meep_gpu/triton_kernels/complex_fused_hd_pair.py",
        "meep_gpu/triton_kernels/fused_hd_pair.py",
        "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
        "meep_gpu/triton_kernels/folded_complex.py",
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
        "parity/meep_gpu/gate_triton_folded_complex_fused_hd_pair.py",
    )

    fixtures = (
        ("mirror_periodic",
         {"symmetry": (("y", 1),), "boundaries": {"x": "metallic"}}),
        ("mirror_metallic",
         {"symmetry": (("y", -1),),
          "boundaries": {"x": "metallic", "y": "metallic"}}),
        ("mirror_periodic_bloch",
         {"symmetry": (("y", 1),), "boundaries": {},
          "k_point": (0.19, 0.0, 0.0)}),
        # THE OFF-DIAGONAL ADMISSION. The same launch; what changes is which
        # predicate pair admits the run, and the record says which.
        ("offdiag_mirror_periodic",
         {"symmetry": (("y", 1),), "boundaries": {"x": "metallic"},
          "offdiagonal": True}),
        ("offdiag_mirror_metallic",
         {"symmetry": (("y", -1),),
          "boundaries": {"x": "metallic", "y": "metallic"},
          "offdiagonal": True}),
    )
    mutation_fixture = "mirror_periodic"

    #: ONE kernel, so every needle in the lifted curl body matches ONCE.
    mutations: Dict[str, Dict[str, Any]] = {
        "m_own_cell_reads_the_pre_launch_H": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": ("        a_re = own0_re\n"
                    "        a_im = own0_im\n"),
            "new": ("        a_re = tl.load(hi0 + 2 * idx, mask=live, other=0.0)\n"
                    "        a_im = tl.load(hi0 + 2 * idx + 1, mask=live, "
                    "other=0.0)\n"),
            "why": "the curl's OWN cell reads the pre-launch magnetic field instead "
                   "of the register pair the constitutive half just computed",
        },
        "m_foreign_tap_reads_the_stale_value": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": ("        a_y_re, a_y_im = _h_tap_complex(0, i, sj, k, vy,\n"),
            "new": ("        a_y_re, a_y_im = tl.load(hi0 + 2 * oy, mask=vy, "
                    "other=0.0), tl.load(hi0 + 2 * oy + 1, mask=vy, other=0.0)\n"
                    "        _unused_tap = _h_tap_complex(0, i, sj, k, vy,\n"),
            "why": "one foreign tap reads the PRE-LAUNCH H word pair instead of "
                   "recomputing it -- exactly the value an in-place weld would have "
                   "read at a neighbour the launch had not yet written. This is the "
                   "defect the whole design removes",
        },
        "m_foreign_tap_takes_the_wrong_word_plane": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": ("        c_y_re, c_y_im = _h_tap_complex(2, i, sj, k, vy,\n"),
            "new": ("        c_y_im, c_y_re = _h_tap_complex(2, i, sj, k, vy,\n"),
            "why": "a tap's two word planes are swapped. Under complex storage this "
                   "is the mis-binding a word-pair return makes easy, and the Bloch "
                   "rotation the emitter applies afterwards consumes it without "
                   "complaint",
        },
        "m_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": "        tl.store(ho0 + 2 * idx, own0_re, mask=live)\n",
            "new": "        tl.store(hi0 + 2 * idx, own0_re, mask=live)\n",
            "why": "the stepped magnetic field is written into the PRE-LAUNCH buffer "
                   "-- the race the scratch output exists to remove",
        },
        "m_f_w_H_written_in_place": {
            "target": "kernel", "expected": "CAUGHT", "hits": 1,
            "old": "        tl.store(wo0 + 2 * idx, src0_re, mask=live)\n",
            "new": "        tl.store(wi0 + 2 * idx, src0_re, mask=live)\n",
            "why": "on this side the newly written f_w_H IS B exactly, so a foreign "
                   "recompute reads B where it needs B_prev",
        },
        "m_top_plane_mask_dropped": {
            "target": "kernel", "expected": "CAUGHT", "hits": 3,
            "old": "            if BCY == MIRROR_PERIODIC:\n",
            "new": "            if False:\n",
            "why": "the MIRROR_PERIODIC top-plane ownership mask is dropped ON THE Y "
                   "AXIS for all three curl targets -- the needle is the axis test, "
                   "which the certified body writes once per target. THE Y AXIS IS "
                   "THE ONE THIS GATE'S MUTATION FIXTURE FOLDS (`mirror_periodic` "
                   "carries symmetry ('y', 1) and terminates x with a metallic wall), "
                   "and pointing the same edit at BCX instead measured NULL on this "
                   "lane's first full campaign -- a dead branch, not a harmless mask; "
                   "the X-axis form is kept below as a declared null with that "
                   "reason. The Metal sibling measured 264 words on this defect after "
                   "COMPLETE driver steps, EVERY one of them in fu_D and none in D -- "
                   "the fills rewrite D at those planes and rewrite nothing of fu_D. "
                   "A gate that compared only the primaries would score this a pass",
        },
        "m_top_plane_mask_dropped_on_the_unfolded_axis": {
            "target": "kernel", "expected": "NULL", "hits": 3,
            "old": "            if BCX == MIRROR_PERIODIC:\n",
            "new": "            if False:\n",
            "why": "THE SAME EDIT ON THE AXIS THIS FIXTURE DOES NOT FOLD, declared "
                   "NULL and kept because the null is the evidence. `mirror_periodic` "
                   "folds y and walls x, so BCX never carries MIRROR_PERIODIC here "
                   "and all three X arms of the top-plane block are unreachable: the "
                   "mutation compiles, launches, and moves nothing. That is a "
                   "STRUCTURAL null earned by THIS fixture -- it says nothing about "
                   "whether the X arms are needed, and the armed Y-axis row above is "
                   "what establishes that the block bites at all. Measured as CAUGHT "
                   "here it would mean the fixture's fold is not the one it declares",
        },
        "m_cell_zero_mask_narrowed_to_metallic": {
            "target": "kernel", "expected": "CAUGHT", "hits": 3,
            "old": "            if BCY != PERIODIC:\n",
            "new": "            if BCY == METALLIC:\n",
            "why": "the cell-0 ownership mask is narrowed ON THE Y AXIS (three "
                   "occurrences, one per curl target) from `!= PERIODIC` back to "
                   "the unfolded `== METALLIC`, so a MIRROR axis stops masking. That "
                   "widening is one of the three deltas the folded curl carries and "
                   "the Metal sibling measured 540 words on it, again all in fu_D",
        },
        "m_folded_codes_replaced_by_the_0_1_mapping": {
            "target": "host", "expected": "CAUGHT",
            "why": "the plan's boundary triple is replaced by the unfolded 0/1 "
                   "mapping, which sends `mirror` to 0 = PERIODIC: the ghost WRAPS to "
                   "the far plane and neither ownership mask is emitted at all. This "
                   "compiles, launches and converges -- nothing in the kernel can "
                   "catch it, because 0 is a valid code",
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
        "hits": 1,
        "old": ("        a_re = own0_re\n"
                "        a_im = own0_im\n"),
        "new": ("        a_re = tl.load(ho0 + 2 * idx, mask=live, other=0.0)\n"
                "        a_im = tl.load(ho0 + 2 * idx + 1, mask=live, other=0.0)\n"),
        "why": "the curl half re-loads its own cell's magnetic word pair from the "
               "scratch this program just wrote. PREDICTED NULL: one program's store "
               "and load of one address are ordered, so this is inert -- and "
               "confirming it is what shows the correctness comes from the FOREIGN "
               "taps and not from the register",
    }

    # -- fixtures ------------------------------------------------------------
    def driver_kwargs(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        from meep_gpu.grid import Mirror  # noqa: PLC0415

        return {
            "force_complex_fields": True,
            "dimensions": 3,
            "k_point": tuple(spec.get("k_point") or (0.0, 0.0, 0.0)),
            "symmetry": tuple(Mirror(axis, int(phase))
                              for axis, phase in spec.get("symmetry") or ()),
        }

    def pml_spec(self, spec: Mapping[str, Any]) -> Any:
        """The absorber, per face: NOT on the near face of a folded axis.

        That face is the MIRROR PLANE, not a boundary -- ``PML.__post_init__`` refuses
        a layer there and ``folded_axis_kinds`` refuses a grid whose layer it cannot
        resolve.
        """
        folded = {axis.lower() for axis, _phase in (spec.get("symmetry") or ())}
        return {name: ((0.0, 2.0) if name in folded else 2.0) for name in "xyz"}

    def configure(self, driver: Any, spec: Mapping[str, Any]) -> None:
        """Install an off-diagonal chi1inv row where the fixture asks for one.

        THE ROW IS WHAT MOVES THE ADMISSION, not the launch: the off-diagonal
        predicate is K1's clause set with the off-diagonal clause INVERTED, and both
        admissions build the same plan around the same kernel.
        """
        if not spec.get("offdiagonal"):
            return
        shape = tuple(int(n) for n in driver.grid.shape)
        xp = driver.xp
        row = xp.asarray(np.full(shape, np.float32(0.08), np.float32))
        # THE ROWS RIDE WITH THE DIAGONAL, because that is the engine's only door:
        # `Fields.set_epsilon_volumes` takes `chi1inv_offdiagonal` and validates it
        # (fields.py:1210-1310), and a row installed any other way would not be the
        # row the array path reads.
        driver.fields.set_epsilon_volumes(
            {name: xp.asarray(np.full(shape, np.float32(value), np.float32))
             for name, value in self.epsilon.items()},
            {name: xp.asarray(np.full(shape, np.float32(1.0 / value), np.float32))
             for name, value in self.epsilon.items()},
            chi1inv_offdiagonal={"Ex": {"Ey": row}, "Ey": {"Ex": row}})
        if not driver.fields.has_offdiagonal_epsilon:
            # NAMED, not silently skipped: a fixture whose row did not survive
            # installation is a fixture whose admission this gate did not measure.
            raise RuntimeError(
                "the off-diagonal chi1inv row did not survive installation; the "
                "off-diagonal admission cannot be built and must not be reported as "
                "measured")

    # -- plans ---------------------------------------------------------------
    def variant(self, driver: Any) -> Optional[str]:
        admission, _why = family.resolve_admission(driver.fields, driver.pml)
        return admission

    def build_weld(self, driver: Any, kernel: Any = None, sources: Any = (),
                   module: Any = None) -> Any:
        owner = module if module is not None else family
        return owner.plan_folded_complex_fused_hd_pair(
            driver.fields, driver.pml, sources=sources, kernel=kernel)

    def singles(self, driver: Any) -> Dict[str, Any]:
        fields, pml = driver.fields, driver.pml
        if self.variant(driver) == "offdiag":
            return {
                "update_H": _folded.plan_folded_complex_offdiag_constitutive(
                    fields, pml, "H"),
                "step_D": _folded.plan_folded_complex_offdiag_pml_curl(
                    fields, pml, "step_D"),
            }
        return {
            "update_H": _folded.plan_folded_complex_constitutive(fields, pml, "H"),
            "step_D": _folded.plan_folded_complex_pml_curl(fields, pml, "step_D"),
        }

    def singles_arms(self, driver: Any) -> Dict[str, str]:
        arms = family.ADMISSION_ARMS[self.variant(driver) or "plain"]
        return {"update_H": arms[0], "step_D": arms[1]}

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        return family.folded_complex_fused_hd_pair_coverage(fields, pml, sources)

    def default_kernel(self, owner: Any, driver: Any) -> Any:
        name = type(owner).__name__
        if name == "FoldedComplexFusedHdPairPlan":
            return family.fused_folded_complex_constitutive_curl_H_to_D_kernel()
        if name == "FoldedComplexPmlCurlPlan":
            return _folded.folded_bloch_pml_curl_step
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
        if tag == "m_folded_codes_replaced_by_the_0_1_mapping":
            def wrap(plan: Any) -> Any:
                plan.bc = tuple(0 if code in (_folded.CODE_PERIODIC,
                                              _folded.CODE_MIRROR_PERIODIC) else 1
                                for code in plan.bc)
                return plan
            return wrap
        return None

    # -- host legs -----------------------------------------------------------
    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        findings: List[str] = []
        certified = family.certified_curl_tail()
        lifted = family.lifted_curl_tail()
        raw = family.raw_curl_tail()
        taps = _complex.halo_taps(raw)
        offsets = _complex.offset_coordinates(raw)
        if certified != lifted:
            findings.append(
                "the fused kernel's curl body is NOT the certified body with the "
                "declared redirects")
        for target in range(3):
            if f"g{target} +" in lifted:
                findings.append(f"g{target} survives below the seam")
        # EVERY SHIFTED OPERAND IS A PAIR whose two word planes agree on component,
        # cell AND guard -- `halo_taps` raises otherwise, and this records what it
        # parsed so a reader sees the six operands rather than trusting the count.
        constitutive_equal = (_complex.certified_constitutive_tail()
                              == _complex.lifted_constitutive_tail())
        if not constitutive_equal:
            findings.append(
                "the imported complex constitutive lift no longer equals the "
                "certified body")
        # THE MULTIPLY HELPERS THIS KERNEL MAY REACH, and no other: a helper outside
        # the declared set is an orientation neither half already launches.
        import re as _re  # noqa: PLC0415

        text = (API_ROOT / self.family_path).read_text(encoding="utf-8")
        # CALL SITES, parsed, never "any line that mentions the name": the import
        # statement at the top of the Triton guard names all three helpers on one
        # line and a substring scan reads that as three stray calls.
        called = set(_re.findall(r"\b(_(?:mul|rotate|div)_[A-Za-z0-9_]*)\s*\(",
                                 text))
        reached = sorted(called & set(family.MULTIPLY_HELPERS))
        stray = sorted(called - set(family.MULTIPLY_HELPERS))
        if stray:
            findings.append(f"the kernel reaches multiply helpers outside the "
                            f"declared set: {stray}")
        codes = {"PERIODIC": _folded.CODE_PERIODIC, "METALLIC": _folded.CODE_METALLIC,
                 "MIRROR_METALLIC": _folded.CODE_MIRROR_METALLIC,
                 "MIRROR_PERIODIC": _folded.CODE_MIRROR_PERIODIC}
        if codes != {"PERIODIC": 0, "METALLIC": 1, "MIRROR_METALLIC": 2,
                     "MIRROR_PERIODIC": 3}:
            findings.append(f"the folded ghost codes are {codes}")
        measures = {
            "certified_chars": len(certified),
            "lift_equal": certified == lifted,
            "halo_taps": {name: {plane: list(value)
                                 for plane, value in planes.items()}
                          for name, planes in taps.items()},
            "offsets": {name: list(value) for name, value in offsets.items()},
            "curl_source": str(family.CURL_PATH.name) + "::" + family.CURL_FUNCTION,
            "imported_constitutive_lift_equal": constitutive_equal,
            "multiply_helpers_declared": list(family.MULTIPLY_HELPERS),
            "multiply_helpers_reached": reached,
            "ghost_codes": codes,
            "arms": {"primary": list(family.ARMS),
                     "extra": [list(pair) for pair in family.EXTRA_ARMS]},
            "certified_under_subnormal_policy":
                family.CERTIFIED_UNDER_SUBNORMAL_POLICY,
            "one_kernel_two_cells": (
                "the off-diagonal cell is served by the SAME launch: the shipped "
                "plan_folded_complex_offdiag_pml_curl says 'The kernel and the plan "
                "class are K1's, untouched; only the ADMISSION is new'"),
        }
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

        unfolded = _View(fields, grid=_View(fields.grid,
                                            is_mirrored=lambda axis: False,
                                            has_symmetry=lambda: False))
        return [
            {"name": "admits_the_fixture", "fields": fields, "sources": (),
             "must_refuse": False},
            {"name": "undeclared_source_list", "fields": fields, "sources": None,
             "must_refuse": True, "needle": "the source set was not declared"},
            {"name": "standing_integrated_electric_withdraw", "fields": fields,
             "sources": (_IntegratedElectric(),), "must_refuse": True,
             "needle": "standing integrated"},
            {"name": "unfolded_grid_names_the_plain_complex_product",
             "fields": unfolded, "sources": (), "must_refuse": True,
             "needle": "complex_fused_hd_pair"},
            {"name": "no_absorber", "fields": fields, "pml": None, "sources": (),
             "must_refuse": True},
        ]

    def arbitration_incumbents(self) -> Dict[str, str]:
        return {"note": "recorded per fixture in composer_selected; this family is "
                        "not in CERTIFIED_FUSED_PRODUCTS, so the composer is never "
                        "offered it"}


PRODUCT = FoldedComplexProduct()


def main(argv: Optional[List[str]] = None) -> int:
    return kit.run(PRODUCT, Path(__file__).resolve(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
