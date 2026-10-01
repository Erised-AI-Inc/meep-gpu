#!/usr/bin/env python3
"""Native MPS byte gate for the COMPLEX BLOCH-BETA Metal H->D weld, BOTH variants.

ONE PRODUCT, TWO CELLS, and this gate measures both from one descriptor:

    (special_kz complex beta -> special_kz complex beta)   1 corpus row
    (folded beta complex     -> folded beta complex)       3 corpus rows

all four ``buildable_not_built`` on ``results/fusion_matrix_metal_2026-09-07_wired``
and none of them carrying a standing in-seam withdraw.

``meep_gpu/metal_kernels/beta_complex_fused_hd_pair.py`` computes the CERTIFIED complex
``update_H`` into launch-local SCRATCH, takes its own cell's magnetic field from
registers, RECOMPUTES every one of the beta curl's six foreign taps from pre-launch
state through the same ``h_cell``, steps ``D``/``fu_D`` in place, and rotates the
``H``/``f_w_H`` bindings afterwards. ONE transform emits two device strings --
``special_kz.beta_bloch_curl_source`` for ``plain`` and
``folded_beta.folded_beta_bloch_curl_source`` for ``folded``, which is that same parent
template with one slot inserted -- and ``resolve_variant`` reads the fold off the grid,
so a row never chooses between them.

THE CLAIM IS PER COMPLETE DRIVER STEP, as uint32 WORDS over every stored volume the
engine allocates, never ``allclose``, against the array path, the two CERTIFIED beta
singles for the resolved variant, ``plan_step(fuse=True)`` and ``plan_step(fuse=False)``.

WHAT THIS GATE ADDS OVER ITS TWO SIBLINGS, because it is what beta adds:

* **the beta term is armed twice** -- compiled out, and with its two centre partners
  swapped -- on both variants. Both must fire;
* **the beta COEFFICIENT's convention is armed as a host defect.** This seam's curl is
  ``step_D``, so ``beta_curl_coefficients`` is asked with ``magnetic=False``; the
  magnetic convention multiplies by ``+1j`` instead of ``-1j`` and is a plausible,
  smooth, entirely wrong answer;
* **the constitutive lift is asserted BETA-FREE.** ``certified_h_cell`` re-derives the
  imported ``update_H`` body and requires it to mention none of ``bpr``/``bpi``/
  ``bmr``/``bmi``, which is what makes "beta is a curl-only term" a property of the
  emission rather than of a docstring, and leg ``transcription`` re-reads it;
* **the packed record is wider and is LAUNCHED.** Five ``float2`` members then five
  scalars is 60 bytes of payload and ``sizeof(Params)`` is 64; the probe binds a host
  record and reads every field back, because a struct four bytes short of what the
  shader may address is a plausible number rather than a crash.

WRITTEN FOR THE UNWIRED TREE: every clause that depends on the wiring tables is
RECORDED, never asserted. What is asserted is refused BY NAME, the refusal naming
``INSTALLABLE`` False, the composer's selection unchanged, and the released neighbours
keeping their slots.

Progress reporting: one flushed timestamped line per unit of work; every leg appended and fsynced
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
    beta_complex_fused_hd_pair as family,
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

LIFT_PREFIX = "MEEP_GPU_BETA_COMPLEX_HD_GATE"

#: EVERY LEVER BOTH VARIANTS HAVE, and the two that only one of them has.
#:
#:   VARIANT      plain and folded, at least three fixtures each, because the two
#:                emitters are the whole of what the variant means;
#:   BETA SIGN    a negative beta is a different coefficient pair and a different
#:                rounding, and the corpus carries both signs;
#:   TERMINATION  MIRROR_METALLIC vs MIRROR_PERIODIC on the folded variant: the
#:                top-plane mask exists on one and not the other;
#:   WALL         a live non-folded metallic wall, without which every wall-scoped
#:                question is structurally absent;
#:   BLOCH        a phased PERIODIC axis, which is what makes the phase block's
#:                survival through the lift measurable -- and the corpus cell carries
#:                it on both variants;
#:   FULL COUNT   an odd extent, where ``_far_reflect_rows`` is a whole cell different.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("plain_2d", dict(cell=(2.0, 2.1, 0.0), beta=0.33, complex_storage=True)),
    ("plain_2d_negative_beta",
     dict(cell=(2.0, 2.1, 0.0), beta=-0.685, complex_storage=True)),
    ("plain_2d_metallic",
     dict(cell=(2.0, 2.1, 0.0), beta=0.33, complex_storage=True,
          boundaries={"x": "metallic"})),
    ("plain_2d_bloch",
     dict(cell=(2.0, 2.1, 0.0), beta=0.33, complex_storage=True,
          k_point=(0.3, 0.0, 0.0))),
    ("folded_y_periodic_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), beta=0.3, complex_storage=True)),
    ("folded_y_periodic_odd_phase_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", -1),), beta=0.3, complex_storage=True)),
    ("folded_y_periodic_odd_count_2d",
     dict(cell=(1.6, 2.1, 0.0), folds=(("Y", 1),), beta=0.3, complex_storage=True)),
    ("folded_y_metallic_x_wall_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), beta=0.3, complex_storage=True,
          boundaries={"x": "metallic", "y": "metallic"})),
)

#: The fixtures the shader mutations are armed on: ONE OF EACH VARIANT, because the two
#: variants are two device strings and a defect caught in one is not caught in the
#: other, plus the MIRROR_METALLIC fold, which is the only specialisation on which a
#: folded top plane is STEPPED rather than masked.
MUTATION_CASES: Tuple[str, ...] = ("plain_2d", "folded_y_periodic_2d",
                                   "folded_y_metallic_x_wall_2d")

#: The fixture the mask leg scores on: it must carry a folded PERIODIC axis, because
#: ``folded_top_plane_mask`` emits a comment and no line on a MIRROR_METALLIC one.
MASK_CASE = "folded_y_periodic_2d"


def expansion() -> str:
    """The PROBE-MEASURED complex-multiply arm -- the BETA probe's, never a default.

    The beta artifact classifies one more call-site orientation than the plain complex
    one, so it licenses everything that licenses and one pattern more; both
    :mod:`.special_kz` and :mod:`.folded_beta` bind it this way and this product
    follows.
    """
    arm = special_kz.beta_expansion_from_probe(special_kz.load_expansion_probe())
    if arm is None:
        raise SystemExit(
            "the beta expansion probe is absent or non-discriminating on this host; "
            "the arm is a PLATFORM FACT and this gate refuses to default it")
    return arm


def emit(variant: str, codes: Sequence[int], phased: Sequence[int],
         arm: str, contract: str) -> str:
    return family.beta_complex_fused_hd_pair_source(variant, codes, phased, arm,
                                                    contract)


def parent_curl(variant: str, codes: Sequence[int], phased: Sequence[int],
                arm: str) -> str:
    """The CERTIFIED beta complex curl this weld lifts, unwelded, for this variant."""
    return family.beta_complex_curl_source(variant, codes, phased, arm)


def singles(driver: Any, residency: Residency) -> Tuple[Any, Any]:
    """The two CERTIFIED parent plans for the RESOLVED variant.

    Two different pairs, because the two cells have two different arm pairs: the plain
    variant's constitutive and curl are :mod:`.special_kz`'s and the folded variant's
    are :mod:`.folded_beta`'s. Both build the same certified complex constitutive body
    under different admissions, which is the fact the whole product rests on.
    """
    variant = family.resolve_variant(driver.grid)
    if variant == "plain":
        return (special_kz.plan_beta_run_complex_constitutive(
                    driver.fields, driver.pml, "H", residency),
                special_kz.plan_beta_bloch_pml_curl(
                    driver.fields, driver.pml, "step_D", residency))
    return (folded_beta.plan_folded_beta_complex_constitutive(
                driver.fields, driver.pml, "H", residency),
            folded_beta.plan_folded_beta_bloch_pml_curl(
                driver.fields, driver.pml, "step_D", residency))


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _shipped(variant, codes, phased, arm):  # noqa: ANN001
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    return emit(variant, codes, phased, arm, shaders.CONTRACT_OFF)


def _own_load_from_the_input(variant, codes, phased, arm):  # noqa: ANN001
    """The own-cell magnetic load taken from the PRE-launch ``H``.

    On a beta curl this is a double defect and the leg records it as one: the curl's
    own-cell operand AND the beta term's centre partner are the same registers, so a
    stale own-cell load moves both the curl and the beta term.
    """
    return common.needle(_shipped(variant, codes, phased, arm),
                         "    float2 a   = own.a0;\n",
                         "    float2 a   = hi0[ii];\n")


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
                         "    float2 curl0 = c_mul_coefficient_left(dtdx, t0);\n",
                         "    float2 curl0 = t0;\n")


def _beta_term_dropped(variant, codes, phased, arm):  # noqa: ANN001
    """The special-kz beta term compiled out of the fused curl."""
    source = _shipped(variant, codes, phased, arm)
    source = common.needle(
        source, "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);\n", "")
    return common.needle(
        source, "    curl1 = curl1 - c_mul(float2(bmr, bmi), a);\n", "")


def _beta_partners_swapped(variant, codes, phased, arm):  # noqa: ANN001
    """The beta term's two UNSHIFTED CENTRE partners swapped.

    ``step_db.cpp:148-176`` runs ``cc`` over ``d_c`` in ``{X, Y}`` only: target 0 takes
    the SECOND source's centre at sign +1 and target 1 the FIRST source's at sign -1.
    Swapping them compiles, converges and is wrong.
    """
    source = _shipped(variant, codes, phased, arm)
    source = common.needle(
        source, "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);\n",
        "    curl0 = curl0 - c_mul(float2(bpr, bpi), a);\n")
    return common.needle(
        source, "    curl1 = curl1 - c_mul(float2(bmr, bmi), a);\n",
        "    curl1 = curl1 - c_mul(float2(bmr, bmi), b);\n")


def _phase_orientation_swapped(variant, codes, phased, arm):  # noqa: ANN001
    """The Bloch rotation applied with the COEFFICIENT on the left.

    ``special_kz._phase_lines`` emits ``b_x = wx ? c_mul(b_x, float2(pxr, pxi)) : b_x;``
    -- FIELD on the left, which is the array path's orientation (stepping.py:1909) and
    the one the expansion arm was PROBED for. Complex multiplication is commutative in
    exact arithmetic and is not in this expansion.

    Generic over the emitted line rather than a fixed needle, so a template that
    renamed an operand arms ZERO sites and is reported as a no-op rather than as a
    caught defect.
    """
    source = _shipped(variant, codes, phased, arm)
    out: List[str] = []
    armed = 0
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if " ? c_mul(" in stripped and stripped.startswith(("a_", "b_", "c_")):
            head, rest = line.split(" ? c_mul(", 1)
            inner, tail = rest.rsplit(")", 1)
            operand, word = inner.split(", ", 1)
            out.append(f"{head} ? c_mul({word}, {operand}){tail}")
            armed += 1
        else:
            out.append(line)
    if not armed:
        raise AssertionError(
            "no Bloch phase site was armed; this fixture emits no phase block")
    return "".join(out)


def _top_plane_mask_dropped(source: str) -> str:
    return common.needle(source, "    curl1 = last_y ? float2(0.0f, 0.0f) : curl1;\n",
                         "")


def _cell0_mask_dropped(source: str) -> str:
    return common.needle(
        common.needle(source, "    curl0 = at_y ? float2(0.0f, 0.0f) : curl0;\n", ""),
        "    curl2 = at_y ? float2(0.0f, 0.0f) : curl2;\n", "")


def _host_codes_from_boundary_kinds(product, case_name, spec, steps):  # noqa: ANN001
    """The PERIODIC/METALLIC triple on a folded grid -- a wrap and a missing mask."""
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

    ``beta_curl_coefficients`` multiplies by ``+1j`` for a magnetic sub-step and
    ``-1j`` for an electric one (stepping.py:798-799). This seam's curl is ``step_D``,
    so the electric convention is the right one; the magnetic one compiles, converges
    and is a smooth wrong answer -- exactly the class of defect a byte gate exists for.
    """
    driver = common.build_driver(spec, 7, scale_bits=0)
    wrong = special_kz.beta_curl_coefficients(
        driver.grid.beta, driver.grid.dt, magnetic=True, complex_storage=True)
    return common.run_case(product, case_name, spec, steps, with_composer=False,
                           beta_words=wrong, scale_bits=0)


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
    "bloch_phase_orientation_swapped": (
        _phase_orientation_swapped,
        "the Bloch rotation applied coefficient-left; the array path is field-left "
        "and that orientation is what the expansion arm was probed for", True,
        ("plain_2d_bloch",)),
}

HOST_MUTATIONS: Dict[str, Tuple[Any, str]] = {
    "codes_from_boundary_kinds": (
        _host_codes_from_boundary_kinds,
        "the PERIODIC/METALLIC triple the plain variant builds, on a FOLDED grid: the "
        "ghost wraps and the cell-0 mask is not widened. ARMED ON THE FOLDED FIXTURES "
        "ONLY, because on an unfolded one this expression IS the one the plan uses and "
        "the mutation is a no-op -- an unarmable defect scored as uncaught",
        ("folded_y_periodic_2d", "folded_y_metallic_x_wall_2d"),
    ),
    "rotation_skipped": (
        _host_rotation_skipped,
        "the post-launch rotation dropped: the constitutive half's output is invisible "
        "under the engine's own name"),
    "H_and_f_w_H_written_in_place": (
        _host_written_in_place,
        "the scratch bound to the pre-launch buffers, which is the read-write hazard "
        "the scratch-output shape removes"),
    "beta_words_from_the_magnetic_convention": (
        _host_beta_words_magnetic,
        "the beta coefficient rounded through the MAGNETIC convention (+1j) when this "
        "seam's curl is step_D and takes the electric one (-1j)"),
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
                         "    float2 a   = own.a0;\n",
                         "    float2 a   = ho0[ii];\n")


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

def _sibling_plain_complex(fields: Any, pml: Any) -> Any:
    from meep_gpu.metal_kernels import complex_fused_hd_pair  # noqa: PLC0415

    return complex_fused_hd_pair.metal_complex_fused_hd_pair_coverage(
        fields, pml, (), Residency())


def _sibling_beta_real(fields: Any, pml: Any) -> Any:
    from meep_gpu.metal_kernels import beta_real_fused_hd_pair  # noqa: PLC0415

    return beta_real_fused_hd_pair.metal_beta_real_fused_hd_pair_coverage(
        fields, pml, (), Residency())


REFUSALS: Tuple[Dict[str, Any], ...] = (
    {"name": "beta_free_complex",
     "spec": dict(cell=(2.0, 2.1, 0.0), complex_storage=True),
     "must_name": "beta",
     "sibling": _sibling_plain_complex, "sibling_expected": True},
    {"name": "beta_real_storage",
     "spec": dict(cell=(2.0, 2.1, 0.0), beta=0.33),
     "must_name": "complex",
     "sibling": _sibling_beta_real, "sibling_expected": True},
    {"name": "folded_beta_free_complex",
     "spec": dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), complex_storage=True),
     "must_name": "beta"},
    {"name": "beta_complex_inactive_absorber",
     # ``metal_composition_matrix``'s OWN no-PML builder: ``setup_pml`` refuses a layer
     # that absorbs nowhere BY NAME, so a driver cannot be built in this configuration.
     "pair": lambda: matrix.folded_no_pml(complex_storage=True),
     "must_name": "no active PML layer"},
)


# ---------------------------------------------------------------------------
# The Params probe — launched, every field read back
# ---------------------------------------------------------------------------

def params_probe() -> Dict[str, Any]:
    """The 64-byte packed record BOUND TO A LAUNCH, with every field read back.

    Five ``float2`` members then five scalars is 60 bytes of payload and Metal rounds
    the struct up to its 8-byte alignment. A host record of 60 is four bytes short of
    what the shader may address, and a struct read one word early is a plausible
    complex number rather than a crash -- so the layout is measured on the device.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    source = """
#include <metal_stdlib>
using namespace metal;
struct Params {
    float2 px; float2 py; float2 pz; float2 bp; float2 bm;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
};
kernel void params_probe(device float* out [[buffer(0)]],
                         constant Params& prm [[buffer(1)]],
                         uint idx [[thread_position_in_grid]])
{
    if (idx != 0) { return; }
    out[0] = prm.px.x; out[1] = prm.px.y;
    out[2] = prm.py.x; out[3] = prm.py.y;
    out[4] = prm.pz.x; out[5] = prm.pz.y;
    out[6] = prm.bp.x; out[7] = prm.bp.y;
    out[8] = prm.bm.x; out[9] = prm.bm.y;
    out[10] = float(prm.nx); out[11] = float(prm.ny); out[12] = float(prm.nz);
    out[13] = float(prm.n_elem); out[14] = prm.dtdx;
    out[15] = float(sizeof(Params));
}
"""
    record = np.zeros(1, dtype=family.params_record_dtype())
    record["px"] = (1.5, 2.5)
    record["py"] = (3.5, 4.5)
    record["pz"] = (5.5, 6.5)
    record["bp"] = (7.5, 8.5)
    record["bm"] = (9.5, 10.5)
    record["nx"], record["ny"], record["nz"], record["n_elem"] = 11, 12, 13, 1716
    record["dtdx"] = np.float32(0.25)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    tensor = torch.from_numpy(words).to("mps")
    out = torch.zeros(16, dtype=torch.float32, device="mps")
    getattr(compile_source(source), "params_probe")(out, tensor, threads=1)
    torch.mps.synchronize()
    read = [float(value) for value in out.cpu().numpy()]
    expected = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
                11.0, 12.0, 13.0, 1716.0, 0.25]
    return {
        "passed": (read[:15] == expected
                   and int(read[15]) == family.PARAMS_ITEMSIZE),
        "read_back": read[:15], "expected": expected,
        "device_sizeof_Params": int(read[15]),
        "host_record_itemsize": family.PARAMS_ITEMSIZE,
        "why": ("the five float2 members come FIRST and the record declares itemsize "
                "64 explicitly: the payload is 60 bytes and Metal's struct is 64, so a "
                "natural record is four bytes short of what the shader may address"),
    }


# ---------------------------------------------------------------------------
# The product descriptor
# ---------------------------------------------------------------------------

PRODUCT = common.Product(
    module=family,
    coverage=family.metal_beta_complex_fused_hd_pair_coverage,
    planner=family.plan_metal_beta_complex_fused_hd_pair,
    source=emit,
    parent_curl_source=parent_curl,
    singles=singles,
    expansion=expansion,
    complex_storage=True,
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
        "split_re_im_planes": (family.split_plane_pair_signature,
                               family.SPLIT_PLANE_BINDINGS),
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


BUILDERS: Dict[str, Any] = {
    "host": lambda args: {
        **common.host_block(PRODUCT, __file__),
        "probes": PROBES,
        "expansion_arm": expansion(),
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


def _leg_transcription_both_variants() -> Dict[str, Any]:
    """The transcription leg on ONE fixture PER VARIANT, run ONCE each.

    BOTH VARIANTS, because they are two device strings: a lift that resolved on the
    plain emitter says nothing about the folded one, whose template carries an extra
    block the redirect must survive. Run once and reported together, so the leg's
    verdict and its evidence cannot disagree.
    """
    rows = [common.leg_transcription(
                PRODUCT, common.build_driver(dict(CASES)[case], 1, scale_bits=0))
            for case in ("plain_2d", "folded_y_periodic_2d")]
    return {"passed": all(row["passed"] for row in rows), "variants": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    return common.run_gate(PRODUCT, __file__, BUILDERS, argv)


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
