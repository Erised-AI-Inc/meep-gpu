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



def emit(variant: str, codes: Sequence[int], phased: Sequence[int],
         contract: str) -> str:
    return family.beta_complex_fused_hd_pair_source(variant, codes, phased,
                                                    contract)


def parent_curl(variant: str, codes: Sequence[int], phased: Sequence[int]) -> str:
    """The CERTIFIED beta complex curl this weld lifts, unwelded, for this variant."""
    return family.beta_complex_curl_source(variant, codes, phased)


def singles(driver: Any, residency: Residency) -> Tuple[Any, Any]:
    """The two CERTIFIED parent plans for the RESOLVED variant.

    Two different pairs, because the two cells have two different arm pairs: the plain
    variant's constitutive and curl are :mod:`.special_kz`'s and the folded variant's
    are :mod:`.folded_beta`'s. Both build the same certified complex constitutive body
    under different admissions, which is the fact the whole product rests on.
    """
    variant = family.resolve_variant(driver.grid)
    record = driver.chunk_bounds_record()
    if variant == "plain":
        return (special_kz.plan_beta_run_complex_constitutive(
                    driver.fields, driver.pml, "H", residency, chunk_bounds=record),
                special_kz.plan_beta_bloch_pml_curl(
                    driver.fields, driver.pml, "step_D", residency,
                    chunk_bounds=record))
    return (folded_beta.plan_folded_beta_complex_constitutive(
                driver.fields, driver.pml, "H", residency, chunk_bounds=record),
            folded_beta.plan_folded_beta_bloch_pml_curl(
                driver.fields, driver.pml, "step_D", residency,
                chunk_bounds=record))


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _shipped(variant, codes, phased):  # noqa: ANN001
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    return emit(variant, codes, phased, shaders.CONTRACT_OFF)


def _own_load_from_the_input(variant, codes, phased):  # noqa: ANN001
    """The own-cell magnetic load taken from the PRE-launch ``H``.

    On a beta curl this is a double defect and the leg records it as one: the curl's
    own-cell operand AND the beta term's centre partner are the same registers, so a
    stale own-cell load moves both the curl and the beta term.
    """
    return common.needle(_shipped(variant, codes, phased),
                         "    float2 a   = own.a0;\n",
                         "    float2 a   = hi0[ii];\n")


def _foreign_tap_component_swapped(variant, codes, phased):  # noqa: ANN001
    source = _shipped(variant, codes, phased)
    start = source.index("h_cell(", source.index("h_cell_result own"))
    tap = source.index("h_cell(", start + 1)
    hit = source.index(".a0", tap)
    return source[:hit] + ".a1" + source[hit + 3:]


def _scratch_history_store_dropped(variant, codes, phased):  # noqa: ANN001
    return common.needle(
        _shipped(variant, codes, phased),
        "    wo0[ii] = own.wn0; wo1[ii] = own.wn1; wo2[ii] = own.wn2;\n", "")


def _constitutive_bound_to_the_curl_sinv(variant, codes, phased):  # noqa: ANN001
    """THE SHARED COEFFICIENT GROUP, BROKEN -- the premise of the 30-pointer shape."""
    source = _shipped(variant, codes, phased)
    if "kmx, kmy, kmz, cb)" not in source:
        raise AssertionError("no h_cell call ends with the shared kms group")
    return source.replace("kmx, kmy, kmz, cb)", "sinvx, sinvy, sinvz, cb)")


def _dtdx_dropped(variant, codes, phased):  # noqa: ANN001
    """The curl's dtdx scale dropped on target 0, on both live-partner branches."""
    source = _shipped(variant, codes, phased)
    source = common.needle(
        source, "c_mul_coefficient_left(dtdx, ((c_y - c_dy) + b_dz) - b_z)",
        "(((c_y - c_dy) + b_dz) - b_z)")
    return common.needle(source, "c_mul_coefficient_left(dtdx, c_y - c_dy)",
                         "(c_y - c_dy)")


def _beta_term_dropped(variant, codes, phased):  # noqa: ANN001
    """The special-kz beta pass's increment compiled out of the fused curl."""
    source = _shipped(variant, codes, phased)
    source = common.needle(source, "    float2 bi0 = -bq0;\n",
                           "    float2 bi0 = float2(0.0f, 0.0f);\n")
    return common.needle(source, "    float2 bi1 = -bq1;\n",
                         "    float2 bi1 = float2(0.0f, 0.0f);\n")


def _beta_partners_swapped(variant, codes, phased):  # noqa: ANN001
    """The beta pass's two UNSHIFTED CENTRE partners swapped.

    ``step_db.cpp:161-175`` runs ``cc`` over ``d_c`` in ``{X, Y}`` only: target 0 takes
    the SECOND source's centre at sign +1 and target 1 the FIRST source's at sign -1.
    Swapping them compiles, converges and is wrong.
    """
    source = _shipped(variant, codes, phased)
    source = common.needle(
        source, "float2(-(beta_plus * b.y), beta_plus * b.x)",
        "float2(-(beta_plus * a.y), beta_plus * a.x)")
    return common.needle(
        source, "float2(-(beta_minus * a.y), beta_minus * a.x)",
        "float2(-(beta_minus * b.y), beta_minus * b.x)")


def _beta_increment_not_negated(variant, codes, phased):  # noqa: ANN001
    """The pass's increment taken as the quarter turn itself, not its negation."""
    return common.needle(_shipped(variant, codes, phased),
                         "    float2 bi0 = -bq0;\n", "    float2 bi0 = bq0;\n")


def _beta_ladder_ignores_dsigu(variant, codes, phased):  # noqa: ANN001
    """The additive ladder's dsigu branch dropped: ``f -= df`` on every cell, where
    MEEP's dsigu chunk takes ``f -= sinv_dsigu * df`` (stepping._apply_folded_increment).
    Armed on target 1, whose dsigu axis is x: a 2-D fixture has no z layer, so target
    0's dsigu branch is never taken there.
    """
    return common.needle(
        _shipped(variant, codes, phased),
        "    v1 = du1 ? (v1 - float2(si_x * bd1.x, si_x * bd1.y)) : (v1 - bd1);\n",
        "    v1 = v1 - bd1;\n")


def _previous_fused_connection(variant, codes, phased):  # noqa: ANN001
    """The Bloch connection's product fused, NumPy's complex64 multiply.

    INVERTED (KERNELS.md §7.1): the connection is MEEP's unfused product
    (``c_mul``, the array path's ``stepping._bloch_ghost_operand``); this plants the
    package's previous fused spelling in the one helper every connection calls, so a
    fixture with a phased axis must diverge.
    """
    return common.needle(
        _shipped(variant, codes, phased),
        "    return float2((z.x * p.x) - (z.y * p.y),\n"
        "                  (z.x * p.y) + (z.y * p.x));",
        "    return float2(fma(z.x, p.x, -(z.y * p.y)),\n"
        "                  fma(z.x, p.y,  (z.y * p.x)));")


def _top_plane_mask_dropped(source: str) -> str:
    return common.needle(source, "    curl1 = last_y ? float2(0.0f, 0.0f) : curl1;\n",
                         "")


def _pass_top_plane_mask_dropped(source: str) -> str:
    """The beta pass's increment left unmasked on the folded top plane."""
    return common.needle(source, "    bi1 = last_y ? float2(0.0f, 0.0f) : bi1;\n", "")


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

    Under complex storage ``beta_curl_coefficients`` negates the word for an electric
    sub-step (the quarter turn by ``-coefficient``, stepping._special_kz_beta_term).
    This seam's curl is ``step_D``, so the electric convention is the right one; the
    magnetic one compiles, converges and is a smooth wrong answer -- exactly the class
    of defect a byte gate exists for.
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
        "the special-kz beta pass's increment compiled out of the fused curl", True),
    "beta_partners_swapped": (
        _beta_partners_swapped,
        "the beta pass's two unshifted centre partners swapped", True),
    "beta_increment_not_negated": (
        _beta_increment_not_negated,
        "the beta pass's increment taken as the quarter turn, not its negation", True),
    "beta_ladder_ignores_dsigu": (
        _beta_ladder_ignores_dsigu,
        "the additive ladder's dsigu branch dropped on target 1", True),
    "bloch_previous_fused_connection": (
        _previous_fused_connection,
        "the Bloch connection's product fused, where the array path's is MEEP's "
        "unfused one", True, ("plain_2d_bloch",)),
}

HOST_MUTATIONS: Dict[str, Tuple[Any, str]] = {
    "codes_from_boundary_kinds": (
        _host_codes_from_boundary_kinds,
        "the PERIODIC/METALLIC triple the plain variant builds, on a FOLDED grid: the "
        "ghost wraps and the cell-0 mask is not widened. ARMED ON THE FOLDED FIXTURES "
        "ONLY, because on an unfolded one this expression IS the one the plan uses and "
        "the mutation is a no-op -- an unarmable defect scored as uncaught -- and on the "
        "folded PERIODIC one only: on a 2-D MIRROR_METALLIC fold the fills restore D at "
        "the mirror plane and, under the per-chunk forms, no fu_D word sits there (a "
        "beta run is 2-D, so no fixture of this product meets a dsigu layer at cell 0)",
        ("folded_y_periodic_2d",),
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
        "the beta word taken from the MAGNETIC convention (not negated) when this "
        "seam's curl is step_D and takes the electric one (negated)"),
}

MASK_MUTATIONS: Dict[str, Tuple[Any, str]] = {
    "top_plane_mask_dropped": (
        _top_plane_mask_dropped,
        "the folded PERIODIC top-plane ownership mask removed"),
    "pass_top_plane_mask_dropped": (
        _pass_top_plane_mask_dropped,
        "the beta pass's increment left unmasked on the folded PERIODIC top plane"),
}


def _byte_neutral_own_from_scratch(variant, codes, phased):  # noqa: ANN001
    return common.needle(_shipped(variant, codes, phased),
                         "    float2 a   = own.a0;\n",
                         "    float2 a   = ho0[ii];\n")


def _byte_neutral_store_order(variant, codes, phased):  # noqa: ANN001
    source = _shipped(variant, codes, phased)
    first = "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n"
    second = "    wo0[ii] = own.wn0; wo1[ii] = own.wn1; wo2[ii] = own.wn2;\n"
    return common.needle(common.needle(source, first, "__FIRST__\n"),
                         second, first).replace("__FIRST__\n", second)


def _cell0_mask_dropped_shipped(variant, codes, phased):  # noqa: ANN001
    return _cell0_mask_dropped(_shipped(variant, codes, phased))


BYTE_NEUTRAL: Tuple[Dict[str, Any], ...] = (
    {"name": "cell0_mask_dropped_is_unobservable_on_a_2d_fold",
     "build": _cell0_mask_dropped_shipped, "case": "folded_y_periodic_2d",
     "why": "DECLARED NULL under the per-chunk forms: on a 2-D fold the cell-0 targets' "
            "dsigu axes are the invariant z and the folded axis, whose mirror plane "
            "carries no layer, so no fu_D word is written at cell 0, and the fills "
            "restore D there. A beta run is 2-D, so no fixture of this product reaches "
            "it; the top-plane mask on the same case is the control that fires"},
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
    """The 152-byte packed record BOUND TO A LAUNCH, with every field read back.

    Three ``float2`` members, five scalars, two beta words and the chunk record, which
    Metal rounds up to its 8-byte alignment. A host record that is short of the
    struct, or a member read one word early, is a plausible number rather than a
    crash -- so the layout is measured on the device, through the shipped struct
    source (the chunk record's own ``struct_source``).
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from meep_gpu.metal_kernels import chunk_bounds  # noqa: PLC0415
    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    source = f"""
#include <metal_stdlib>
using namespace metal;
{chunk_bounds.struct_source()}
struct Params {{
    float2 px; float2 py; float2 pz;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
    float beta_plus; float beta_minus;
{chunk_bounds.params_member()}
}};
kernel void params_probe(device float* out [[buffer(0)]],
                         constant Params& prm [[buffer(1)]],
                         uint idx [[thread_position_in_grid]])
{{
    if (idx != 0) {{ return; }}
    out[0] = prm.px.x; out[1] = prm.px.y;
    out[2] = prm.py.x; out[3] = prm.py.y;
    out[4] = prm.pz.x; out[5] = prm.pz.y;
    out[6] = float(prm.nx); out[7] = float(prm.ny); out[8] = float(prm.nz);
    out[9] = float(prm.n_elem); out[10] = prm.dtdx;
    out[11] = prm.beta_plus; out[12] = prm.beta_minus;
    out[13] = float(sizeof(Params));
}}
"""
    record = np.zeros(1, dtype=family.params_record_dtype())
    record["px"] = (1.5, 2.5)
    record["py"] = (3.5, 4.5)
    record["pz"] = (5.5, 6.5)
    record["nx"], record["ny"], record["nz"], record["n_elem"] = 11, 12, 13, 1716
    record["dtdx"] = np.float32(0.25)
    record["beta_plus"] = np.float32(7.5)
    record["beta_minus"] = np.float32(-9.5)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    tensor = torch.from_numpy(words).to("mps")
    out = torch.zeros(14, dtype=torch.float32, device="mps")
    getattr(compile_source(source), "params_probe")(out, tensor, threads=1)
    torch.mps.synchronize()
    read = [float(value) for value in out.cpu().numpy()]
    expected = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 11.0, 12.0, 13.0, 1716.0, 0.25,
                7.5, -9.5]
    return {
        "passed": (read[:13] == expected
                   and int(read[13]) == family.PARAMS_ITEMSIZE),
        "read_back": read[:13], "expected": expected,
        "device_sizeof_Params": int(read[13]),
        "host_record_itemsize": family.PARAMS_ITEMSIZE,
        "why": ("the three float2 members come FIRST and the record declares its "
                "itemsize explicitly: Metal pads the struct to its 8-byte alignment, "
                "so a natural record can be short of what the shader may address"),
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


def evaluate(driver: Any) -> Dict[str, Any]:
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
