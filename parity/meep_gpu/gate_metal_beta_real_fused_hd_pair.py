#!/usr/bin/env python3
"""Native MPS byte gate for the REAL BETA Metal H->D weld, BOTH variants.

ONE PRODUCT, TWO CELLS, measured from one descriptor:

    (special_kz real beta -> special_kz real beta)   1 corpus row
        examples:refl-angular-kz2d.py
    (folded beta real     -> folded beta real)       1 corpus row
        tests:TestSpecialKz.test_eigsrc_kz_1_real_imag

both ``buildable_not_built`` on ``results/fusion_matrix_metal_2026-09-07_wired`` and
neither carrying a standing in-seam withdraw.

TWO CORPUS ROWS IS A SMALL DENOMINATOR AND THE ARTIFACT SAYS SO RATHER THAN HIDING IT.
The lift leg's strength is bounded by the cell it covers, which is why the synthetic
matrix here is EIGHT fixtures spanning both variants, both beta signs, both mirror
terminations, a live non-folded wall and an odd full count -- the levers a two-row cell
cannot exercise on its own.

``meep_gpu/metal_kernels/beta_real_fused_hd_pair.py`` computes the CERTIFIED
``update_H`` into launch-local SCRATCH, takes its own cell's magnetic field from
registers, RECOMPUTES every one of the beta curl's six foreign taps from pre-launch
state through the same ``h_cell``, steps ``D``/``fu_D`` in place, and rotates the
``H``/``f_w_H`` bindings afterwards. ONE transform emits two device strings --
``special_kz.beta_curl_source`` for ``plain`` and
``folded_beta.folded_beta_curl_source`` for ``folded`` -- and ``resolve_variant`` reads
the fold off the grid.

THE CLAIM IS PER COMPLETE DRIVER STEP, as uint32 WORDS over every stored volume,
never ``allclose``, against the array path, the two CERTIFIED beta singles for the
resolved variant, ``plan_step(fuse=True)`` and ``plan_step(fuse=False)``.

WHAT SEPARATES THIS GATE FROM ITS COMPLEX TWIN, and it is more than the dtype:

* **the packed record is the only UNPADDED one of the four H->D welds** -- seven 4-byte
  members at 4-byte alignment, 28 bytes, no tail padding. The probe launches it and
  reads ``sizeof(Params)`` back rather than computing it;
* **there is no expansion arm and no phase block**, so the two mutations that carry
  them on the complex products are structurally absent here rather than skipped, and
  the artifact records that;
* **the beta term is two real multiplies** rather than two complex ones, and its
  partners are the same own-cell registers the redirect replaces -- so the same two
  beta mutations exercise a different arithmetic path with the same argument.

WRITTEN FOR THE UNWIRED TREE: every clause that depends on the wiring tables is
RECORDED, never asserted.

Rule 7: one flushed timestamped line per unit of work; every leg appended and fsynced
as it lands; the lift writes one JSON per corpus row as it lands.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# THE POLICY, SET BEFORE ANY meep_gpu MODULE IS REACHED.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import metal_composition_matrix as matrix  # noqa: E402

PROBES = matrix.prepare_environment()

import hd_tail_gate_common as common  # noqa: E402

from meep_gpu.metal_kernels import (  # noqa: E402
    beta_real_fused_hd_pair as family,
)
from meep_gpu.metal_kernels import folded_beta, special_kz, symmetry  # noqa: E402
from meep_gpu.metal_kernels.device import Residency  # noqa: E402

SUBJECT_PACKAGE = "metal_kernels"

CENSUS = "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = "h_to_d_seam_2026-09-04"

#: The board's ``(update_H arm, step_D arm)`` pair per variant -- READ FROM THE
#: FAMILY, never re-typed here. The cell a gate measures and the cell the family
#: claims have to be the same cell, and two spellings of one join key is how they
#: stop being.
CELL_ARMS: Dict[str, Tuple[str, str]] = dict(family.VARIANT_CELLS)

LIFT_PREFIX = "MEEP_GPU_BETA_REAL_HD_GATE"

#: EIGHT FIXTURES, deliberately wider than the two-row cell they certify: both
#: variants, both beta signs, both mirror terminations, a live non-folded wall, an odd
#: full count. The corpus rows are the ground truth for the CELL; the matrix is what
#: keeps the levers a two-row cell cannot pull from being untested.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("plain_2d", dict(cell=(2.0, 2.1, 0.0), beta=0.33)),
    ("plain_2d_negative_beta", dict(cell=(2.0, 2.1, 0.0), beta=-0.685)),
    ("plain_2d_metallic",
     dict(cell=(2.0, 2.1, 0.0), beta=0.33, boundaries={"x": "metallic"})),
    ("plain_2d_metallic_both",
     dict(cell=(2.0, 2.1, 0.0), beta=0.33,
          boundaries={"x": "metallic", "y": "metallic"})),
    ("folded_y_periodic_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), beta=0.3)),
    ("folded_y_periodic_odd_phase_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", -1),), beta=0.3)),
    ("folded_y_periodic_odd_count_2d",
     dict(cell=(1.6, 2.1, 0.0), folds=(("Y", 1),), beta=0.3)),
    ("folded_y_metallic_x_wall_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), beta=0.3,
          boundaries={"x": "metallic", "y": "metallic"})),
)

#: ONE OF EACH VARIANT plus the MIRROR_METALLIC fold, for the reason the complex gate
#: gives: two variants are two device strings, and a metallic fold is the only
#: specialisation on which a folded top plane is STEPPED rather than masked.
MUTATION_CASES: Tuple[str, ...] = ("plain_2d", "folded_y_periodic_2d",
                                   "folded_y_metallic_x_wall_2d")

MASK_CASE = "folded_y_periodic_2d"


def emit(variant: str, codes: Sequence[int], contract: str) -> str:
    return family.beta_real_fused_hd_pair_source(variant, codes, contract)


def parent_curl(variant: str, codes: Sequence[int], phased: Sequence[int],
                arm: Any) -> str:
    """The CERTIFIED real beta curl this weld lifts, unwelded, for this variant.

    ``phased`` and ``arm`` are accepted and IGNORED: real storage carries neither a
    Bloch phase nor an expansion arm, and the shared harness passes the same five
    arguments to every product so the three gates cannot drift on the signature.
    """
    del phased, arm
    return family.beta_real_curl_source(variant, codes)


def singles(driver: Any, residency: Residency) -> Tuple[Any, Any]:
    """The two CERTIFIED parent plans for the RESOLVED variant."""
    variant = family.resolve_variant(driver.grid)
    if variant == "plain":
        return (special_kz.plan_beta_run_constitutive(
                    driver.fields, driver.pml, "H", residency),
                special_kz.plan_beta_pml_curl(
                    driver.fields, driver.pml, "step_D", residency))
    return (folded_beta.plan_folded_beta_constitutive(
                driver.fields, driver.pml, "H", residency),
            folded_beta.plan_folded_beta_pml_curl(
                driver.fields, driver.pml, "step_D", residency))


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _shipped(variant, codes, phased, arm):  # noqa: ANN001
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    del phased, arm
    return emit(variant, codes, shaders.CONTRACT_OFF)


def _own_load_from_the_input(variant, codes, phased, arm):  # noqa: ANN001
    """The own-cell magnetic load taken from the PRE-launch ``H``.

    A double defect on a beta curl: the curl's own-cell operand AND the beta term's
    centre partner are the same register.
    """
    return common.needle(_shipped(variant, codes, phased, arm),
                         "    float a   = own.a0;\n",
                         "    float a   = hi0[ii];\n")


def _foreign_tap_component_swapped(variant, codes, phased, arm):  # noqa: ANN001
    source = _shipped(variant, codes, phased, arm)
    start = source.index("h_cell(", source.index("h_cell_result own"))
    tap = source.index("h_cell(", start + 1)
    hit = source.index(".a0", tap)
    return source[:hit] + ".a1" + source[hit + 3:]


def _scratch_history_store_dropped(variant, codes, phased, arm):  # noqa: ANN001
    return common.needle(
        _shipped(variant, codes, phased, arm),
        "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n", "")


def _constitutive_bound_to_the_curl_sinv(variant, codes, phased, arm):  # noqa: ANN001
    """THE SHARED COEFFICIENT GROUP, BROKEN -- the premise of the 30-pointer shape."""
    return _shipped(variant, codes, phased, arm).replace(
        "kmx, kmy, kmz)", "sinvx, sinvy, sinvz)")


def _dtdx_dropped(variant, codes, phased, arm):  # noqa: ANN001
    return common.needle(_shipped(variant, codes, phased, arm),
                         "    float curl0 = dtdx * ((c_y - c) + (b - b_z));\n",
                         "    float curl0 = ((c_y - c) + (b - b_z));\n")


def _beta_term_dropped(variant, codes, phased, arm):  # noqa: ANN001
    source = _shipped(variant, codes, phased, arm)
    source = common.needle(source, "    curl0 = curl0 - (beta_plus * b);\n", "")
    return common.needle(source, "    curl1 = curl1 - (beta_minus * a);\n", "")


def _beta_partners_swapped(variant, codes, phased, arm):  # noqa: ANN001
    """The beta term's two UNSHIFTED CENTRE partners swapped.

    ``step_db.cpp:148-176`` runs ``cc`` over ``d_c`` in ``{X, Y}`` only: target 0 takes
    the SECOND source's centre at sign +1 and target 1 the FIRST source's at sign -1.
    """
    source = _shipped(variant, codes, phased, arm)
    source = common.needle(source, "    curl0 = curl0 - (beta_plus * b);\n",
                           "    curl0 = curl0 - (beta_plus * a);\n")
    return common.needle(source, "    curl1 = curl1 - (beta_minus * a);\n",
                         "    curl1 = curl1 - (beta_minus * b);\n")


def _beta_sign_flipped(variant, codes, phased, arm):  # noqa: ANN001
    """``curl - (c * g)`` respelled as ``curl + (c * g)``.

    IEEE-754 defines subtraction as the addition of the negation, so the array path's
    ``curl + (-(c*g))`` and the kernel's ``curl - (c*g)`` are the same bits; the sign
    is therefore the one thing about this line that a spelling change CAN move, and it
    is armed rather than argued.
    """
    source = _shipped(variant, codes, phased, arm)
    source = common.needle(source, "    curl0 = curl0 - (beta_plus * b);\n",
                           "    curl0 = curl0 + (beta_plus * b);\n")
    return common.needle(source, "    curl1 = curl1 - (beta_minus * a);\n",
                         "    curl1 = curl1 + (beta_minus * a);\n")


def _top_plane_mask_dropped(source: str) -> str:
    return common.needle(source, "    curl1 = last_y ? 0.0f : curl1;\n", "")


def _cell0_mask_dropped(source: str) -> str:
    return common.needle(
        common.needle(source, "    curl0 = at_y ? 0.0f : curl0;\n", ""),
        "    curl2 = at_y ? 0.0f : curl2;\n", "")


def _host_codes_from_boundary_kinds(product, case_name, spec, steps):  # noqa: ANN001
    driver = common.build_driver(spec, 7, scale_bits=0)
    return common.run_case(product, case_name, spec, steps, with_composer=False,
                           codes=common.boundary_kinds_codes(driver), scale_bits=0)


def _host_rotation_skipped(product, case_name, spec, steps):  # noqa: ANN001
    return common.run_case(product, case_name, spec, steps, with_composer=False,
                           rotation=False, scale_bits=0)


def _host_written_in_place(product, case_name, spec, steps):  # noqa: ANN001
    return common.run_case(product, case_name, spec, steps, with_composer=False,
                           in_place=True, scale_bits=0)


def _host_beta_words_magnetic(product, case_name, spec, steps):  # noqa: ANN001
    """THE BETA COEFFICIENT FROM THE MAGNETIC CONVENTION.

    Under REAL storage the two conventions differ only in the ``sign``: the ``+/-1j``
    factor is complex and does not apply, so ``beta_curl_coefficients(magnetic=True)``
    returns the same two words. THIS MUTATION IS THEREFORE ARMED WITH THE PAIR
    REVERSED instead -- ``(minus, plus)`` where the plan passes ``(plus, minus)`` --
    which is the same class of defect (the coefficient the kernel multiplies by is not
    the one the array path rounded for that target) and is the one the real path can
    actually make. The complex twin arms the ``+1j``/``-1j`` convention itself, and the
    difference between the two gates is recorded rather than papered over.
    """
    driver = common.build_driver(spec, 7, scale_bits=0)
    plus, minus = special_kz.beta_curl_coefficients(
        driver.grid.beta, driver.grid.dt, magnetic=False, complex_storage=False)
    return common.run_case(product, case_name, spec, steps, with_composer=False,
                           beta_words=(minus, plus), scale_bits=0)


SHADER_MUTATIONS: Dict[str, Tuple[Any, ...]] = {
    "own_cell_load_from_the_pre_launch_H": (
        _own_load_from_the_input,
        "the curl's own cell -- and the beta term's centre partner -- read the "
        "PRE-launch H instead of the register the constitutive half produced", True),
    "foreign_tap_component_swapped": (
        _foreign_tap_component_swapped,
        "one foreign recompute returns the wrong component of h_cell's struct", True),
    "f_w_H_scratch_store_dropped": (
        _scratch_history_store_dropped,
        "the split-field history never reaches the scratch, so the rotation publishes "
        "a zero plane", True),
    "constitutive_bound_to_the_curl_sinv_group": (
        _constitutive_bound_to_the_curl_sinv,
        "the shared coefficient group broken: h_cell handed the curl's sinv instead "
        "of the kms both halves share", True),
    "dtdx_dropped_from_the_curl": (
        _dtdx_dropped, "the curl's dtdx scale dropped on target 0", True),
    "beta_term_dropped": (
        _beta_term_dropped,
        "the special-kz beta term compiled out of the fused curl", True),
    "beta_partners_swapped": (
        _beta_partners_swapped,
        "the beta term's two unshifted centre partners swapped", True),
    "beta_sign_flipped": (
        _beta_sign_flipped,
        "the beta term ADDED rather than subtracted; IEEE-754 makes the subtraction "
        "and the array path's add-of-a-negation identical, so the sign is the one "
        "thing this line's spelling can move", True),
}

HOST_MUTATIONS: Dict[str, Tuple[Any, ...]] = {
    "codes_from_boundary_kinds": (
        _host_codes_from_boundary_kinds,
        "the PERIODIC/METALLIC triple the plain variant builds, on a FOLDED grid: the "
        "ghost wraps and the cell-0 mask is not widened. ARMED ON THE FOLDED FIXTURES "
        "ONLY, because on an unfolded one this expression IS the one the plan uses",
        ("folded_y_periodic_2d", "folded_y_metallic_x_wall_2d")),
    "rotation_skipped": (
        _host_rotation_skipped,
        "the post-launch rotation dropped: the constitutive half's output is invisible "
        "under the engine's own name"),
    "H_and_f_w_H_written_in_place": (
        _host_written_in_place,
        "the scratch bound to the pre-launch buffers, which is the read-write hazard "
        "the scratch-output shape removes"),
    "beta_words_reversed": (
        _host_beta_words_magnetic,
        "the beta coefficient PAIR reversed: the kernel multiplies target 0 by the "
        "minus-sign word and target 1 by the plus-sign one"),
}

MASK_MUTATIONS: Dict[str, Tuple[Any, str]] = {
    "top_plane_mask_dropped": (
        _top_plane_mask_dropped,
        "the folded PERIODIC top-plane ownership mask removed"),
    "cell0_mask_dropped": (
        _cell0_mask_dropped,
        "the widened cell-0 ownership mask removed on the folded axis"),
}


def _byte_neutral_own_from_scratch(variant, codes, phased, arm):  # noqa: ANN001
    return common.needle(_shipped(variant, codes, phased, arm),
                         "    float a   = own.a0;\n",
                         "    float a   = ho0[ii];\n")


def _byte_neutral_store_order(variant, codes, phased, arm):  # noqa: ANN001
    source = _shipped(variant, codes, phased, arm)
    first = "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n"
    second = "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n"
    return common.needle(common.needle(source, first, "__FIRST__\n"),
                         second, first).replace("__FIRST__\n", second)


BYTE_NEUTRAL: Tuple[Dict[str, Any], ...] = (
    {"name": "own_cell_load_read_back_from_its_own_scratch",
     "build": _byte_neutral_own_from_scratch, "case": "folded_y_periodic_2d",
     "why": "same thread, same cell, program order: the scratch store must BE the "
            "register the curl and the beta term consume"},
    {"name": "scratch_stores_reordered", "build": _byte_neutral_store_order,
     "case": "plain_2d",
     "why": "the H and f_w_H scratch groups are disjoint volumes written by one thread "
            "at one cell; their order cannot matter and the leg says so"},
)


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def _sibling_plain_real(fields: Any, pml: Any) -> Any:
    from meep_gpu.metal_kernels import fused_hd_pair  # noqa: PLC0415

    return fused_hd_pair.metal_fused_hd_pair_coverage(fields, pml, (), Residency())


def _sibling_beta_complex(fields: Any, pml: Any) -> Any:
    from meep_gpu.metal_kernels import beta_complex_fused_hd_pair  # noqa: PLC0415

    return beta_complex_fused_hd_pair.metal_beta_complex_fused_hd_pair_coverage(
        fields, pml, (), Residency())


REFUSALS: Tuple[Dict[str, Any], ...] = (
    {"name": "beta_free_real",
     "spec": dict(cell=(2.0, 2.1, 0.0)),
     "must_name": "beta",
     "sibling": _sibling_plain_real, "sibling_expected": True},
    {"name": "beta_complex_storage",
     "spec": dict(cell=(2.0, 2.1, 0.0), beta=0.33, complex_storage=True),
     "must_name": "complex",
     "sibling": _sibling_beta_complex, "sibling_expected": True},
    {"name": "folded_beta_free_real",
     "spec": dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),)),
     "must_name": "beta"},
    {"name": "beta_real_inactive_absorber",
     "pair": lambda: matrix.folded_no_pml(),
     "must_name": "no active PML layer"},
)


# ---------------------------------------------------------------------------
# The Params probe — launched, every field read back
# ---------------------------------------------------------------------------

def params_probe() -> Dict[str, Any]:
    """The 28-byte packed record BOUND TO A LAUNCH, with every field read back.

    THE ONLY UNPADDED RECORD OF THE FOUR H->D WELDS: seven 4-byte members at 4-byte
    alignment. That is a fact about this struct rather than about the family, so it is
    measured here -- ``sizeof(Params)`` is read off the device, not computed.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    source = """
#include <metal_stdlib>
using namespace metal;
struct Params {
    uint nx; uint ny; uint nz; uint n_elem;
    float dtdx; float beta_plus; float beta_minus;
};
kernel void params_probe(device float* out [[buffer(0)]],
                         constant Params& prm [[buffer(1)]],
                         uint idx [[thread_position_in_grid]])
{
    if (idx != 0) { return; }
    out[0] = float(prm.nx); out[1] = float(prm.ny); out[2] = float(prm.nz);
    out[3] = float(prm.n_elem);
    out[4] = prm.dtdx; out[5] = prm.beta_plus; out[6] = prm.beta_minus;
    out[7] = float(sizeof(Params));
}
"""
    record = np.zeros(1, dtype=family.params_record_dtype())
    record["nx"], record["ny"], record["nz"], record["n_elem"] = 11, 12, 13, 1716
    record["dtdx"] = np.float32(0.25)
    record["beta_plus"] = np.float32(1.5)
    record["beta_minus"] = np.float32(-1.5)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    tensor = torch.from_numpy(words).to("mps")
    out = torch.zeros(8, dtype=torch.float32, device="mps")
    getattr(compile_source(source), "params_probe")(out, tensor, threads=1)
    torch.mps.synchronize()
    read = [float(value) for value in out.cpu().numpy()]
    expected = [11.0, 12.0, 13.0, 1716.0, 0.25, 1.5, -1.5]
    return {
        "passed": (read[:7] == expected
                   and int(read[7]) == family.PARAMS_ITEMSIZE),
        "read_back": read[:7], "expected": expected,
        "device_sizeof_Params": int(read[7]),
        "host_record_itemsize": family.PARAMS_ITEMSIZE,
        "why": ("seven 4-byte members at 4-byte alignment: 28 bytes, no padding, and "
                "the only unpadded Params record among the four H->D welds"),
    }


# ---------------------------------------------------------------------------
# The product descriptor
# ---------------------------------------------------------------------------

PRODUCT = common.Product(
    module=family,
    coverage=family.metal_beta_real_fused_hd_pair_coverage,
    planner=family.plan_metal_beta_real_fused_hd_pair,
    source=emit,
    parent_curl_source=parent_curl,
    singles=singles,
    # NO EXPANSION ARM. Real storage carries no complex multiply, so there is no
    # platform fact to probe and nothing to default -- which is why this is None
    # rather than a function returning a placeholder.
    expansion=None,
    complex_storage=False,
    cases=CASES,
    cell_arms=CELL_ARMS,
    mutation_cases=MUTATION_CASES,
    params_probe=params_probe,
    refuted_signatures={
        "one_more_pointer": (family.refuted_one_more_pointer_source,
                             family.ONE_MORE_POINTER_BINDINGS),
        "unshared_kms": (family.refuted_unshared_kms_source,
                         family.UNSHARED_KMS_BINDINGS),
        "separate_scalars": (family.refuted_separate_scalar_source,
                             family.SEPARATE_SCALAR_BINDINGS),
    },
    shader_mutations=SHADER_MUTATIONS,
    host_mutations=HOST_MUTATIONS,
    census=CENSUS, seam_record=SEAM_RECORD,
    lift_environment_prefix=LIFT_PREFIX,
    gate_stem=Path(__file__).stem,
)


def runtime_reasons() -> List[str]:
    return common.runtime_reasons()


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:  # noqa: ARG001
    """The census driver's battery hook: the identity claim on ONE lifted corpus row."""
    steps = int(os.environ.get(f"{LIFT_PREFIX}_STEPS", 12))
    block = common.evaluate_row(PRODUCT, driver, steps)
    path = os.environ.get(f"{LIFT_PREFIX}_PROGRESS")
    label = os.environ.get(f"{LIFT_PREFIX}_LABEL", "?")
    line = (f"{label} variant={block.get('variant')} "
            f"driven={block.get('driven')} identical={block.get('identical')} "
            f"steps={block.get('steps_compared')}")
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
    return {f"{family.FAMILY}_gate": block}


def _drivers() -> List[Any]:
    return [common.build_driver(spec, 2, scale_bits=0) for _name, spec in CASES]


def _leg_transcription_both_variants() -> Dict[str, Any]:
    """The transcription leg on ONE fixture PER VARIANT, run ONCE each."""
    rows = [common.leg_transcription(
                PRODUCT, common.build_driver(dict(CASES)[case], 1, scale_bits=0))
            for case in ("plain_2d", "folded_y_periodic_2d")]
    return {"passed": all(row["passed"] for row in rows), "variants": rows}


BUILDERS: Dict[str, Any] = {
    "host": lambda args: {
        **common.host_block(PRODUCT, __file__),
        "probes": PROBES,
        "expansion_arm": None,
        "why_no_expansion_arm": (
            "real float32 storage performs no complex multiply, so there is no "
            "expansion arm to probe and none to default; the two mutations that carry "
            "one on the complex siblings are structurally absent here"),
        "cases": [name for name, _spec in CASES],
        "variants": list(family.VARIANTS),
    },
    "policy": lambda args: common.leg_policy(PRODUCT,
                                            dict(CASES)["plain_2d"]),
    "driver_order": lambda args: common.leg_driver_order(PRODUCT),
    "transcription": lambda args: _leg_transcription_both_variants(),
    "refusal": lambda args: common.leg_refusal(PRODUCT, REFUSALS),
    "withdraw": lambda args: common.leg_withdraw(PRODUCT, dict(CASES)["plain_2d"]),
    "binding_ceiling": lambda args: common.leg_binding_ceiling(PRODUCT, _drivers()),
    "product": lambda args: common.leg_product(PRODUCT, args.steps),
    "seed_scale": lambda args: common.leg_seed_scale(PRODUCT, min(args.steps, 12)),
    "purity": lambda args: common.leg_purity(PRODUCT, min(args.steps, 8)),
    "ghost_observability": lambda args: common.leg_ghost_observability(
        PRODUCT, min(args.steps, 8)),
    "masks": lambda args: common.leg_masks(PRODUCT, min(args.steps, 8),
                                           MASK_MUTATIONS, MASK_CASE),
    "launch_structure": lambda args: common.leg_launch_structure(
        PRODUCT, min(args.steps, 12)),
    "sync": lambda args: common.leg_sync(PRODUCT, min(args.steps, 12), (3, 7)),
    "arbitration": lambda args: common.leg_arbitration(PRODUCT),
    "lift": lambda args: common.leg_lift(
        PRODUCT, args.out.parent, args.lift_steps, args.lift_max_cells,
        args.lift_timeout, args.lift_resume,
        args.lift_only.split(",") if args.lift_only else None),
    "byte_neutral": lambda args: common.leg_byte_neutral(
        PRODUCT, min(args.steps, 8), BYTE_NEUTRAL),
    "mutation": lambda args: common.leg_mutation(PRODUCT, min(args.steps, 8)),
    "disarm": lambda args: common.leg_disarm(PRODUCT, min(args.steps, 8)),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    return common.run_gate(PRODUCT, __file__, BUILDERS, argv)


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
