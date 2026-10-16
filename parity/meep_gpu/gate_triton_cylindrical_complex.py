"""Bit-identity gate for the Triton COMPLEX cylindrical (Dcyl, complex64) kernels.

The complex family of :mod:`gate_triton_cylindrical`, whose machinery it imports: twin real
``Grid``/``Fields``/``PML`` triples on one array module, the same seeded full-volume state,
:mod:`meep_gpu.stepping` (the aligned oracle) on one and the Triton plan on the other,
every stored word compared as ``uint32``. Nothing is transcribed here.

WHAT IS MEASURED. :mod:`meep_gpu.triton_kernels.cylindrical_complex`'s curl kernel with the
shared radial scan (:func:`cylindrical_triton.cyl_radial_difference`) at every m class --
m = 0 under complex storage, |m| = 1 (the axis-row increments, the Dz axis zero), |m| >= 2
(the near-axis zeroing, on the default and the accurate branch) -- on both z terminations
and on layers with one and with two r chunks; and the complex constitutive plan
(:class:`complex_fields.ComplexConstitutivePlan` as
:func:`cylindrical_complex.plan_cylindrical_complex_constitutive` binds it) against
``stepping.update_H`` / ``update_E`` through MEEP's start of H.

THE MUTATIONS name the forms they guard: the Appendix A inversion (the previous pairwise
curl grouping), the section 7.2 battery (whole-grid recurrence, membership a row off,
lattices swapped, ``fu`` stored on every cell, the predicate on coefficient identity), the
i*m/r pass (dropped, its dsig scale, its inside form, its ``fu`` stage, its quarter-turn
sign, its unowned mask), the |m| = 1 axis increments (in float32, not negated, the
partner's sign), the per-|m| axis rules, and two declared nulls (an unconditional ``fu``
load, the near ghost put back). Each runs on the case its form needs (``mutation_case``).

Usage (the GPU host, one clear device)::

    CUDA_VISIBLE_DEVICES=N python -u gate_triton_cylindrical_complex.py \\
        --out <dir>/gate.json

One flushed line per row; the JSON artifact is rewritten after every row. The verdict is
``verdict.pass``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gate_triton_cylindrical as common  # noqa: E402

SEEDINGS = common.SEEDINGS

#: The complex family's cases: every m class, both z terminations, one and two r chunks,
#: Courant numbers that are not float32 words, the accurate near-axis branch.
COMPLEX_CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "c_m0_rz", "m": 0, "complex": True, "cell": (2.0, 0.0, 4.0),
     "thickness": {"x": (0, 5), "z": 5}, "courant": 0.37},
    {"name": "c_m1_rz", "m": 1, "cell": (2.0, 0.0, 4.0),
     "thickness": {"x": (0, 5), "z": 5}, "courant": 0.37},
    {"name": "c_mm1_rz_thin", "m": -1, "cell": (2.0, 0.0, 2.4),
     "thickness": {"x": (0, 3), "z": 2}, "courant": 0.5},
    {"name": "c_m1_zper", "m": 1, "cell": (2.0, 0.0, 3.0), "z": "periodic",
     "thickness": {"x": (0, 4)}, "courant": 0.3141592653589793},
    {"name": "c_m2_rz", "m": 2, "cell": (2.0, 0.0, 4.0),
     "thickness": {"x": (0, 5), "z": 5}, "courant": 0.37},
    {"name": "c_m3_accurate", "m": 3, "accurate": True, "cell": (2.0, 0.0, 3.0),
     "thickness": {"x": (0, 4), "z": 3}, "courant": 0.27},
    {"name": "c_mm5_zper", "m": -5, "cell": (2.0, 0.0, 3.0), "z": "periodic",
     "thickness": {"x": (0, 4)}, "courant": 0.3},
    {"name": "c_m2_z_only", "m": 2, "cell": (2.0, 0.0, 4.0),
     "thickness": {"z": 4}, "courant": 0.37},
)

def complex_curl_plan(fields: Any, pml: Any, sub_step: str, kernel: Any = None,
                      scan_kernel: Any = None) -> Any:
    """:func:`cylindrical_complex.plan_cylindrical_complex_curl_from_arrays` on ``fields``."""
    from meep_gpu import stepping
    from meep_gpu.triton_kernels import cylindrical_complex as cc
    from meep_gpu.triton_kernels import cylindrical_triton as cyl
    from meep_gpu.triton_kernels.launch import SUB_STEPS, chunk_bounds_for

    spec = SUB_STEPS[sub_step]
    arrays = {n: getattr(fields, n) for n in tuple(spec["targets"]) + tuple(spec["sources"])}
    arrays.update({"fu_" + n: getattr(fields, "fu_" + n) for n in spec["targets"]})
    flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{spec['suffix']}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}
    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, pml)
    return cc.plan_cylindrical_complex_curl_from_arrays(
        sub_step, arrays, flat, grid.dt / grid.dx, float(grid.courant), int(grid.m),
        bool(grid.accurate_fields_near_cylorigin), 1 if kinds[2] == "metallic" else 0, grid.xp, chunk_bounds=chunk_bounds_for(fields, pml, None),
        scan_starts=cyl.radial_scan_starts(fields, pml, sub_step, None),
        kernel=kernel, scan_kernel=scan_kernel)


def complex_constitutive_plan(fields: Any, pml: Any, side: str,
                              kernel: Any = None) -> Any:
    """The complex constitutive plan, bound as the cylindrical engine route binds it."""
    from meep_gpu.triton_kernels import coverage
    from meep_gpu.triton_kernels.complex_fields import ComplexConstitutivePlan
    from meep_gpu.triton_kernels.launch import chunk_bounds_for

    spec = coverage.CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ComplexConstitutivePlan(
        side, fields.grid.shape, 256,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["aux"]],
        [getattr(fields, n) for n in spec["sources"]],
        ([fields.inverse_epsilon_for(n) for n in spec["targets"]] if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1)
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, chunk_bounds=chunk_bounds_for(fields, pml, None), h_state=fields)


#: (name, kernel, edits, expected, seedings): "curl" is the complex curl kernel, "scan" the
#: shared radial scan. ``mutation_case`` picks the case each form needs.
COMPLEX_MUTATIONS: Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...], str, Tuple[str, ...]], ...] = (
    ("a1_pairwise_curl_grouping", "curl",
     (("curl1_re = curl_stencil(a_z_re, a_re, c_re, c_r_re, dtdx, pz, pr)",
       "curl1_re = dtdx * ((a_z_re - a_re) + (c_re - c_r_re))"),), "caught", SEEDINGS),
    ("a1_phi_term_kept", "curl",
     (("curl0_re = curl_stencil(c_p_re, c_re, b_re, b_z_re, dtdx, pp, pz)",
       "curl0_re = dtdx * (((c_p_re - c_re) + b_re) - b_z_re)"),), "caught", ("lattice",)),
    ("b1_whole_grid_recurrence", "curl",
     (("du0 = record_in(cb, lattice + 8, k)", "du0 = k >= 0"),), "caught", SEEDINGS),
    ("b2_membership_one_row_off", "curl",
     (("du1 = record_in(cb, lattice + 0, i)", "du1 = record_in(cb, lattice + 0, i + 1)"),),
     "caught", SEEDINGS),
    ("b3_lattices_swapped", "curl",
     (("        lattice = RECORD_HALF_INTEGER\n", "        lattice = 0\n"),), "caught",
     SEEDINGS),
    ("b4_fu_stored_on_every_cell", "curl",
     (("tl.store(u1 + 2 * idx, n1_re, mask=um1)", "tl.store(u1 + 2 * idx, n1_re, mask=live)"),),
     "caught", SEEDINGS),
    ("b5_predicate_on_coefficient_identity", "curl",
     (("du0 = record_in(cb, lattice + 8, k)", "du0 = km_z != 1.0"),), "caught", SEEDINGS),
    ("n1_fu_load_unconditional", "curl",
     (("p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)",
       "p_re = tl.load(u1 + 2 * idx, mask=live, other=0.0)"),), "null", SEEDINGS),
    ("i1_imr_dropped", "curl",
     (("    if M_CLASS != M_ZERO:\n        r0 =", "    if M_CLASS == 7:\n        r0 ="),),
     "caught", SEEDINGS),
    ("i2_imr_dsig_scale_dropped", "curl",
     (("d_re = si_x * i2_re", "d_re = i2_re"),), "caught", SEEDINGS),
    ("i3_imr_inside_form_dropped", "curl",
     (("v0_re = tl.where(du0, v0_re - (si_z * d_re), v0_re - d_re)", "v0_re = v0_re - d_re"),),
     "caught", SEEDINGS),
    ("i4_imr_fu_stage_dropped", "curl",
     (("n0_re = n0_re - d_re", "n0_re = n0_re"),), "caught", SEEDINGS),
    ("i5_imr_quarter_turn_sign", "curl",
     (("i0_re = (r0 * c_im) * -1.0", "i0_re = r0 * c_im"),), "caught", SEEDINGS),
    ("i6_imr_unowned_mask_dropped", "curl",
     (("            i0_re = tl.where(at_r, 0.0, i0_re)\n", "\n"),), "caught", SEEDINGS),
    ("x1_axis_increment_in_float32", "curl",
     (("inc_re = tl.cast(tl.cast(s_re, tl.float64) * factor, tl.float32)",
       "inc_re = s_re * tl.cast(factor, tl.float32)"),), "caught", SEEDINGS),
    ("x2_axis_increment_not_negated", "curl",
     (("curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)",
       "curl0_re = tl.where(at_r, inc_re, curl0_re)"),), "caught", SEEDINGS),
    ("x3_axis_partner_sign", "curl",
     (("partner_im = (mf * e1_re) * -1.0", "partner_im = mf * e1_re"),), "caught", SEEDINGS),
    ("x4_dz_axis_zero_dropped", "curl",
     (("            v2_re = tl.where(at_r, 0.0, v2_re)\n", "\n"),), "caught", SEEDINGS),
    ("z1_near_rows_fu_kept_outside", "curl",
     (("um2 = live & (du2 | near)", "um2 = live & du2"),), "caught", SEEDINGS),
    ("z2_near_rows_field_only", "curl",
     (("n0_re = tl.where(near, 0.0, n0_re)", "n0_re = n0_re"),), "caught", SEEDINGS),
    ("m0_dz_axis_add_in_float32", "curl",
     (("h_re = tl.cast(tl.cast(b_re, tl.float64) * four, tl.float32)",
       "h_re = b_re * tl.cast(four, tl.float32)"),), "caught", SEEDINGS),
    ("m0_dp_axis_zero_dropped", "curl",
     (("v1_re = tl.where(at_r, 0.0, v1_re)", "v1_re = v1_re"),), "caught", SEEDINGS),
    ("g1_near_ghost_restored", "curl",
     (("vr = live & (si >= 0) & (si < nx)", "vr = live & (si < nx)"),
      ("o_r = si * nyz + j * nz + k", "o_r = tl.where(si < 0, 0, si) * nyz + j * nz + k")),
     "null", SEEDINGS),
    ("s1_one_segment", "scan",
     (("for t in range(0, n_segs):", "for t in range(0, 1):"),), "caught", SEEDINGS),
    ("s3_first_row_not_from_zero", "scan",
     (("after = total + tl.load(rinv + (g - 1)) * (here - prev)",
       "after = tl.where(g == s + 1, tl.load(rinv + (g - 1)) * (here - prev), "
       "total + tl.load(rinv + (g - 1)) * (here - prev))"),), "caught", ("lattice",)),
    ("s4_no_wall_row", "scan",
     (("mask=live & (g < rows_stored), other=0.0)",
       "mask=live & (g < rows_stored), other=1.0)"),), "caught", SEEDINGS),
)


#: Mutations measured at m = 0 under complex storage: the m = 0 axis pair, and the dropped
#: phi addend, whose signed zero the i*m/r pass's nonzero increment launders on target 0
#: at every m != 0 (a curl that differs only in the sign of a zero, then ``v - df`` with
#: ``df`` nonzero, is one word).
M_ZERO_MUTATIONS: Tuple[str, ...] = ("a1_phi_term_kept", "m0_dz_axis_add_in_float32",
                                     "m0_dp_axis_zero_dropped")


def mutation_case(name: str) -> Dict[str, Any]:
    """The case a mutation's form needs: the near-axis rows at |m| = 2, m = 0 under complex
    storage for :data:`M_ZERO_MUTATIONS`, |m| = 1 for the rest (its axis increments and the
    i*m/r pass)."""
    by_name = {case["name"]: case for case in COMPLEX_CASES}
    if name.startswith("z"):
        return by_name["c_m2_rz"]
    if name in M_ZERO_MUTATIONS:
        return by_name["c_m0_rz"]
    return by_name["c_m1_rz"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="JSON artifact path")
    parser.add_argument("--steps", type=int, default=common.DEFAULT_STEPS)
    parser.add_argument("--legs",
                        default="synthetic,guard,multi_step,engine,constitutive,mutations")
    args = parser.parse_args(argv)
    legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
    xp = common.device_xp()
    from meep_gpu.triton_kernels import cylindrical_complex as cc  # noqa: PLC0415
    from meep_gpu.triton_kernels import cylindrical_triton as cyl  # noqa: PLC0415

    def curl(fields, pml, sub_step, kernel=None, scan_kernel=None):
        return complex_curl_plan(fields, pml, sub_step, kernel=kernel,
                                 scan_kernel=scan_kernel)

    def constitutive(fields, pml, side, kernel=None):
        return complex_constitutive_plan(fields, pml, side, kernel=kernel)

    results: Dict[str, Any] = {
        "what": "bit-identity gate for the Triton complex cylindrical kernels against the "
                "array path",
        "environment": common.environment(), "legs": legs, "steps": args.steps,
        "cases": [case["name"] for case in COMPLEX_CASES],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    common.save(results, args.out)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(results)
    if "synthetic" in legs:
        common.log("leg: synthetic")
        common.run_synthetic(COMPLEX_CASES, xp, curl, results, args.out)
    if "guard" in legs:
        common.log("leg: guard (contraction enabled; must diverge somewhere)")
        common.run_synthetic(COMPLEX_CASES, xp, curl, results, args.out, key="guard",
                             guard=True)
    if "multi_step" in legs:
        common.log("leg: multi_step")
        common.run_multi_step(COMPLEX_CASES, xp, curl, args.steps, results, args.out)
    if "engine" in legs:
        common.log("leg: engine")
        common.run_engine(
            COMPLEX_CASES, xp,
            lambda f, p, s: cc.plan_cylindrical_complex_curl(f, p, s),
            lambda f, p, s: cc.cylindrical_complex_curl_coverage(f, p, s).reasons,
            results, args.out)
    if "constitutive" in legs:
        common.log("leg: constitutive")
        common.run_constitutive(COMPLEX_CASES[:3], xp, constitutive, results, args.out)
    if "mutations" in legs:
        common.log("leg: mutations")
        common.run_mutations(COMPLEX_CASES, xp, curl, COMPLEX_MUTATIONS,
                             {"scan": (cyl.cyl_radial_difference, cyl),
                              "curl": (cc.cyl_complex_pml_curl_step, cc)},
                             common.compile_mutant, results, args.out,
                             case_for=mutation_case)
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["subnormal_policy_end"] = common.policy_end_stamp()
    results["verdict"] = common.verdict(results)
    common.save(results, args.out)
    _stamp_provenance(results)
    common.log(f"VERDICT: {json.dumps(results['verdict'], default=str)}")
    return 0 if results["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
