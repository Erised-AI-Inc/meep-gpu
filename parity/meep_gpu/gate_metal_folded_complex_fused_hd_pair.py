#!/usr/bin/env python3
"""Native MPS byte gate for the FOLDED COMPLEX Metal H->D weld.

THE FIFTH PRODUCT ON THE FOURTH SEAM, on the cell both of its parents refuse by name:
``(folded complex -> folded complex)``, **5 corpus rows**, all five
``buildable_not_built`` on ``results/fusion_matrix_metal_2026-09-07_wired`` and none of
them carrying a standing in-seam withdraw.

``meep_gpu/metal_kernels/folded_complex_fused_hd_pair.py`` computes the certified
complex ``update_H`` into launch-local SCRATCH, takes its own cell's magnetic field
from registers, RECOMPUTES every one of the folded complex curl's six foreign taps from
pre-launch state through the same ``h_cell``, steps ``D``/``fu_D`` in place, and
rotates the ``H``/``f_w_H`` bindings afterwards.

THE CLAIM IS PER COMPLETE DRIVER STEP -- ``FdtdDriver.step``'s own consult order, with
the MIRROR FILLS, the wall clear and the FAR GHOST PASSES exactly where the driver runs
them -- as uint32 WORDS over every stored volume the engine allocates, never
``allclose`` (``-0.0 == 0.0`` lies), against four reference engines from one seed: the
array path, the two CERTIFIED folded-complex singles, ``plan_step(fuse=True)`` and
``plan_step(fuse=False)``.

WHAT THIS GATE OWNS AND WHAT IT IMPORTS. The driver walk, the capture/restore/compare
mechanism, the dispatch shim, the two launch counters, the rotation settling and every
leg body live in ``hd_tail_gate_common``, which three gates share so a defect repaired
in one is repaired in all three; the shared harness in turn imports
``gate_metal_fused_hd_pair``'s certified machinery rather than copying it. What THIS
file owns is the fixtures, the parent emitters, the refusal table, the mutation table
and the ``evaluate`` hook the census driver calls (the census names the GATE module as
its ``--battery``, so that hook cannot be shared).

WRITTEN FOR THE UNWIRED TREE. This product is not in ``registry.FAMILY_MODULES`` and
holds no ``launch.FUSED_PAIR_ARMS`` absorb row. Every clause that depends on that state
is RECORDED, never asserted, because the wiring diff this round writes inverts exactly
those; what is asserted is the substantive half -- refused BY NAME, the refusal naming
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# THE POLICY, SET BEFORE ANY meep_gpu MODULE IS REACHED. `flush` is the only value the
# MPS executor can honour; a resolved `keep` refuses every predicate BY NAME, which is
# itself a measurement and is what the keep leg of a two-policy campaign records.
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

from meep_gpu.metal_kernels import complex_fields, folded_complex  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    folded_complex_fused_hd_pair as family,
)
from meep_gpu.metal_kernels import fused_hd_pair, symmetry  # noqa: E402
from meep_gpu.metal_kernels.device import Residency  # noqa: E402

#: ``measure_predicate_coverage`` digests this package into every lifted row's
#: ``subject_manifest_sha256``. Declared, as every battery declares it.
SUBJECT_PACKAGE = "metal_kernels"

#: The census and seam records the cell is cut from -- the board's own defaults.
CENSUS = "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = "h_to_d_seam_2026-09-04"

#: The board's ``(update_H arm, step_D arm)`` pair per variant -- READ FROM THE
#: FAMILY, never re-typed here. The cell a gate measures and the cell the family
#: claims have to be the same cell, and two spellings of one join key is how they
#: stop being.
CELL_ARMS: Dict[str, Tuple[str, str]] = dict(family.VARIANT_CELLS)

#: The environment prefix the lift's child processes read.
LIFT_PREFIX = "MEEP_GPU_FOLDED_COMPLEX_HD_GATE"

P = symmetry.CODE_PERIODIC
M = symmetry.CODE_METALLIC
MM = symmetry.CODE_MIRROR_METALLIC
MP = symmetry.CODE_MIRROR_PERIODIC

#: EVERY LEVER THE FOLD HAS UNDER COMPLEX STORAGE, AT BOTH VALUES.
#:
#:   TERMINATION  MIRROR_METALLIC vs MIRROR_PERIODIC is the folded families' single
#:                point of failure: the far ghost pass and the top-plane mask exist on
#:                one and not the other, so a defect caught on one is not caught on the
#:                other;
#:   PHASE        the folded complex FILL's parity is a full complex multiply (that is
#:                this composition's one arithmetic delta) and it lives OUTSIDE this
#:                seam -- so this product must be blind to it, which only an odd-plane
#:                fixture can measure;
#:   WALL         ``zero_metal_B`` emits work only for an axis that is metallic and NOT
#:                mirrored, so a matrix without a live non-folded wall would report
#:                every wall-scoped question as structurally absent;
#:   FOLDED AXES  one exercises no composition; two make a corner unowned on both
#:                planes;
#:   BLOCH        a phased PERIODIC axis beside a folded one is what makes the phase
#:                block's survival through the lift measurable at all -- and the corpus
#:                cell carries it (``TestModeDecomposition.test_triangular_lattice_
#:                oblique``, ``TestHoleyWvgBands.test_fields_at_kx``);
#:   FULL COUNT   ``_far_reflect_rows`` is ``stored - 2`` at an even full count and
#:                ``stored - 3`` at an odd one, so ``2.1`` is where a wrong formula is
#:                a whole cell wrong;
#:   DIMENSION    the corpus cell is 2-D and 3-D both.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("y_periodic_even_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), complex_storage=True)),
    ("y_periodic_odd_phase_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", -1),), complex_storage=True)),
    ("y_periodic_odd_count_2d",
     dict(cell=(1.6, 2.1, 0.0), folds=(("Y", 1),), complex_storage=True)),
    ("y_metallic_x_wall_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), complex_storage=True,
          boundaries={"x": "metallic", "y": "metallic"})),
    ("xy_metallic_2d",
     dict(cell=(2.0, 2.0, 0.0), folds=(("X", 1), ("Y", -1)), complex_storage=True,
          boundaries={"x": "metallic", "y": "metallic"})),
    ("y_periodic_x_bloch_2d",
     dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), complex_storage=True,
          k_point=(0.3, 0.0, 0.0))),
    ("y_periodic_3d",
     dict(cell=(1.6, 2.0, 1.2), folds=(("Y", 1),), complex_storage=True,
          dimensions=3)),
    ("yz_metallic_3d",
     dict(cell=(1.6, 2.0, 1.2), folds=(("Y", 1), ("Z", 1)), complex_storage=True,
          dimensions=3,
          boundaries={"x": "metallic", "y": "metallic", "z": "metallic"})),
)

#: The fixtures the shader mutations are armed on. TWO, and not a duplicate pair: the
#: MIRROR_PERIODIC one is the only specialisation that emits a top-plane mask line at
#: all, and the MIRROR_METALLIC one with a live non-folded wall is the only one on
#: which a metallic top plane is STEPPED rather than masked. A mutation run on one
#: alone would report the other's defects as uncaught for the uninteresting reason that
#: no line is emitted there.
MUTATION_CASES: Tuple[str, ...] = ("y_periodic_even_2d", "y_metallic_x_wall_2d")

#: The fixture the mask leg scores on: it must carry a folded PERIODIC axis, because
#: ``folded_top_plane_mask`` emits a comment and no line on a MIRROR_METALLIC one.
MASK_CASE = "y_periodic_even_2d"


def expansion() -> str:
    """The PROBE-MEASURED complex-multiply arm. Never a default."""
    arm = folded_complex.expansion_from_probe(folded_complex.load_expansion_probe())
    if arm is None:
        raise SystemExit(
            "the folded-complex expansion probe is absent or non-discriminating on "
            "this host; the arm is a PLATFORM FACT and this gate refuses to default it")
    return arm


def emit(variant: str, codes: Sequence[int], phased: Sequence[int],
         arm: str, contract: str) -> str:
    return family.folded_complex_fused_hd_pair_source(codes, phased, arm, contract)


def parent_curl(variant: str, codes: Sequence[int], phased: Sequence[int],
                arm: str) -> str:
    """The CERTIFIED folded complex curl this weld lifts, unwelded."""
    return folded_complex.folded_bloch_curl_source(codes, True, phased, arm)


def singles(driver: Any, residency: Residency) -> Tuple[Any, Any]:
    """The two CERTIFIED folded-complex parent plans at the seam's two slots."""
    return (folded_complex.plan_folded_complex_constitutive(
                driver.fields, driver.pml, "H", residency),
            folded_complex.plan_folded_complex_pml_curl(
                driver.fields, driver.pml, "step_D", residency))


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _shipped(variant: str, codes: Sequence[int], phased: Sequence[int],
             arm: str) -> str:
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    return emit(variant, codes, phased, arm, shaders.CONTRACT_OFF)


def _own_load_from_the_input(variant, codes, phased, arm):  # noqa: ANN001
    """The own-cell magnetic load taken from the PRE-launch ``H``.

    The whole weld rests on the curl's own cell reading the register the constitutive
    half just produced, which is the post-``update_H`` value the array path loads.
    """
    return common.needle(_shipped(variant, codes, phased, arm),
                         "    float2 a   = own.a0;\n",
                         "    float2 a   = hi0[ii];\n")


def _foreign_tap_component_swapped(variant, codes, phased, arm):  # noqa: ANN001
    """One foreign recompute returns the wrong component of ``h_cell``'s struct."""
    source = _shipped(variant, codes, phased, arm)
    start = source.index("h_cell(", source.index("h_cell_result own"))
    tap = source.index("h_cell(", start + 1)
    hit = source.index(".a0", tap)
    return source[:hit] + ".a1" + source[hit + 3:]


def _scratch_history_store_dropped(variant, codes, phased, arm):  # noqa: ANN001
    """The split-field history never reaches the scratch, so the rotation publishes 0."""
    return common.needle(
        _shipped(variant, codes, phased, arm),
        "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n", "")


def _constitutive_bound_to_the_curl_sinv(variant, codes, phased, arm):  # noqa: ANN001
    """THE SHARED COEFFICIENT GROUP, BROKEN.

    Both halves sit on the same Yee sub-lattice, so the curl's ``kms`` and the
    constitutive's ``km`` are the same three volumes and are bound once -- the fact
    that makes 30 pointers possible. This hands ``h_cell`` the curl's OTHER group
    instead, which compiles and is a smooth wrong answer.
    """
    return _shipped(variant, codes, phased, arm).replace(
        "kmx, kmy, kmz)", "sinvx, sinvy, sinvz)")


def _dtdx_dropped(variant, codes, phased, arm):  # noqa: ANN001
    """The curl's ``dtdx`` scale dropped on target 0."""
    return common.needle(_shipped(variant, codes, phased, arm),
                         "    float2 curl0 = c_mul_coefficient_left(dtdx, t0);\n",
                         "    float2 curl0 = t0;\n")


def _phase_orientation_swapped(variant, codes, phased, arm):  # noqa: ANN001
    """The Bloch rotation applied with the COEFFICIENT on the left.

    ``complex_fields._phase_block`` emits ``b_x = wx ? c_mul(b_x, px) : b_x;`` --
    FIELD on the left -- which is the array path's orientation (stepping.py:1909) and
    the one the expansion arm was PROBED for. Complex multiplication is commutative in
    exact arithmetic and is not in this expansion, so the swap is a must-catch defect
    here (unlike the folded fill's real +/-1 coefficient, for which the same swap is a
    measured equivalence).

    The rewrite is generic over the emitted line rather than a fixed needle: every
    ``c_mul(<operand>, p?)`` inside a phase ternary is inverted, so a template that
    renamed an operand cannot silently disarm this leg -- it would arm zero sites, and
    a zero-site arm is reported as a no-op rather than as a caught defect.
    """
    source = _shipped(variant, codes, phased, arm)
    out: List[str] = []
    armed = 0
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("b_", "c_", "a_")) and " ? c_mul(" in stripped:
            head, rest = line.split(" ? c_mul(", 1)
            inner, tail = rest.split(")", 1)
            operand, word = (piece.strip() for piece in inner.split(",", 1))
            out.append(f"{head} ? c_mul({word}, {operand}){tail}")
            armed += 1
        else:
            out.append(line)
    if not armed:
        raise AssertionError(
            "no Bloch phase site was armed; this fixture emits no phase block and the "
            "mutation must not be scored on it")
    return "".join(out)


def _top_plane_mask_dropped(source: str) -> str:
    return common.needle(source, "    curl1 = last_y ? float2(0.0f, 0.0f) : curl1;\n",
                         "")


def _cell0_mask_dropped(source: str) -> str:
    return common.needle(
        common.needle(source, "    curl0 = at_y ? float2(0.0f, 0.0f) : curl0;\n", ""),
        "    curl2 = at_y ? float2(0.0f, 0.0f) : curl2;\n", "")


def _host_codes_from_boundary_kinds(product, case_name, spec, steps):  # noqa: ANN001
    """THE ONE STRUCTURAL DEFECT THIS FAMILY CAN MAKE SILENTLY.

    The unfolded complex product builds ``1 if kind == "metallic" else 0`` over
    ``_boundary_kinds``; on a folded grid that reports ``"mirror"``, which the
    expression maps to 0 = PERIODIC -- the backward ghost becomes a WRAP to the far
    plane and the cell-0 mask is not widened. Both codes are valid, so neither the
    emitter nor the compiler can catch it.
    """
    driver = common.build_driver(spec, 7, scale_bits=0)
    return common.run_case(product, case_name, spec, steps, with_composer=False,
                           codes=common.boundary_kinds_codes(driver), scale_bits=0)


def _host_rotation_skipped(product, case_name, spec, steps):  # noqa: ANN001
    """The post-launch rotation dropped: the engine keeps naming the pre-launch pair."""
    return common.run_case(product, case_name, spec, steps, with_composer=False,
                           rotation=False, scale_bits=0)


def _host_written_in_place(product, case_name, spec, steps):  # noqa: ANN001
    """``H`` and ``f_w_H`` bound IN PLACE -- the premise the scratch output removes."""
    return common.run_case(product, case_name, spec, steps, with_composer=False,
                           in_place=True, scale_bits=0)


SHADER_MUTATIONS: Dict[str, Tuple[Any, ...]] = {
    "own_cell_load_from_the_pre_launch_H": (
        _own_load_from_the_input,
        "the curl's own cell reads the PRE-launch H instead of the register the "
        "constitutive half produced", True),
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
    "bloch_phase_orientation_swapped": (
        _phase_orientation_swapped,
        "the Bloch rotation applied coefficient-left; the array path is field-left "
        "(stepping.py:1909) and that orientation is what the expansion arm was probed "
        "for", True, ("y_periodic_x_bloch_2d",)),
}

HOST_MUTATIONS: Dict[str, Tuple[Any, str]] = {
    "codes_from_boundary_kinds": (
        _host_codes_from_boundary_kinds,
        "the PERIODIC/METALLIC triple the unfolded complex product builds, on a folded "
        "grid: the ghost wraps and the cell-0 mask is not widened"),
    "rotation_skipped": (
        _host_rotation_skipped,
        "the post-launch rotation dropped: the constitutive half's output is invisible "
        "under the engine's own name"),
    "H_and_f_w_H_written_in_place": (
        _host_written_in_place,
        "the scratch bound to the pre-launch buffers, which is exactly the read-write "
        "hazard the scratch-output shape removes"),
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
    """The thread's own cell read back from the scratch IT just wrote.

    Same thread, same cell, program order -- so this is required NOT to diverge, and
    what it measures is that the scratch store is EXACTLY the register the curl uses:
    a store that rounded, widened or reordered would show here.
    """
    return common.needle(_shipped(variant, codes, phased, arm),
                         "    float2 a   = own.a0;\n",
                         "    float2 a   = ho0[ii];\n")


def _byte_neutral_store_order(variant, codes, phased, arm):  # noqa: ANN001
    """The six scratch stores reordered: two disjoint volume groups, one thread."""
    source = _shipped(variant, codes, phased, arm)
    first = "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n"
    second = "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n"
    return common.needle(common.needle(source, first, "__FIRST__\n"),
                         second, first).replace("__FIRST__\n", second)


BYTE_NEUTRAL: Tuple[Dict[str, Any], ...] = (
    {"name": "own_cell_load_read_back_from_its_own_scratch",
     "build": _byte_neutral_own_from_scratch,
     "why": "same thread, same cell, program order: the scratch store must BE the "
            "register the curl consumes"},
    {"name": "scratch_stores_reordered", "build": _byte_neutral_store_order,
     "why": "the H and f_w_H scratch groups are disjoint volumes written by one "
            "thread at one cell; their order cannot matter and the leg says so"},
)


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def _sibling_plain_complex(fields: Any, pml: Any) -> Any:
    from meep_gpu.metal_kernels import complex_fused_hd_pair  # noqa: PLC0415

    return complex_fused_hd_pair.metal_complex_fused_hd_pair_coverage(
        fields, pml, (), Residency())


def _sibling_folded_real(fields: Any, pml: Any) -> Any:
    from meep_gpu.metal_kernels import folded_fused_hd_pair  # noqa: PLC0415

    return folded_fused_hd_pair.metal_folded_fused_hd_pair_coverage(
        fields, pml, (), Residency())


REFUSALS: Tuple[Dict[str, Any], ...] = (
    {"name": "unfolded_complex",
     "spec": dict(cell=(2.0, 2.1, 0.0), complex_storage=True),
     "must_name": "complex_fused_hd_pair",
     "sibling": _sibling_plain_complex, "sibling_expected": True},
    {"name": "folded_real_storage",
     "spec": dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),)),
     "must_name": "complex",
     "sibling": _sibling_folded_real, "sibling_expected": True},
    {"name": "folded_complex_with_beta",
     "spec": dict(cell=(1.6, 2.0, 0.0), folds=(("Y", 1),), complex_storage=True,
                  beta=0.3),
     "must_name": "beta"},
    {"name": "folded_complex_inactive_absorber",
     # ``metal_composition_matrix.folded_no_pml``'s OWN fixture: ``setup_pml`` refuses
     # a layer that absorbs nowhere BY NAME, so a driver cannot be built in this
     # configuration and the certified builder is the only route to it.
     "pair": lambda: matrix.folded_no_pml(complex_storage=True),
     "must_name": "no active PML layer"},
)


# ---------------------------------------------------------------------------
# The Params probe — launched, every field read back
# ---------------------------------------------------------------------------

def params_probe() -> Dict[str, Any]:
    """The packed record BOUND TO A LAUNCH, with every field read back.

    A struct whose host record is a byte short is a plausible number rather than a
    crash, so the layout is measured on the device rather than argued from a dtype.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    source = """
#include <metal_stdlib>
using namespace metal;
struct Params {
    float2 px; float2 py; float2 pz;
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
    out[6] = float(prm.nx); out[7] = float(prm.ny); out[8] = float(prm.nz);
    out[9] = float(prm.n_elem); out[10] = prm.dtdx;
    out[11] = float(sizeof(Params));
}
"""
    record = np.zeros(1, dtype=family.params_record_dtype())
    record["px"] = (1.5, 2.5)
    record["py"] = (3.5, 4.5)
    record["pz"] = (5.5, 6.5)
    record["nx"], record["ny"], record["nz"], record["n_elem"] = 11, 12, 13, 1716
    record["dtdx"] = np.float32(0.25)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    tensor = torch.from_numpy(words).to("mps")
    out = torch.zeros(12, dtype=torch.float32, device="mps")
    getattr(compile_source(source), "params_probe")(out, tensor, threads=1)
    torch.mps.synchronize()
    read = [float(value) for value in out.cpu().numpy()]
    expected = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 11.0, 12.0, 13.0, 1716.0, 0.25]
    return {
        "passed": (read[:11] == expected
                   and int(read[11]) == family.PARAMS_ITEMSIZE),
        "read_back": read[:11], "expected": expected,
        "device_sizeof_Params": int(read[11]),
        "host_record_itemsize": family.PARAMS_ITEMSIZE,
        "why": ("the three float2 members come FIRST: with the scalars first the "
                "natural host record puts every phase one word early and reads back "
                "as a plausible complex number rather than as garbage"),
    }


# ---------------------------------------------------------------------------
# The product descriptor
# ---------------------------------------------------------------------------

PRODUCT = common.Product(
    module=family,
    coverage=family.metal_folded_complex_fused_hd_pair_coverage,
    planner=family.plan_metal_folded_complex_fused_hd_pair,
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


# ---------------------------------------------------------------------------
# The census battery hook
# ---------------------------------------------------------------------------

def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate this battery at all, or ``[]``."""
    return common.runtime_reasons()


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:  # noqa: ARG001
    """The census driver's battery hook: the identity claim on ONE lifted corpus row."""
    steps = int(os.environ.get(f"{LIFT_PREFIX}_STEPS", 12))
    block = common.evaluate_row(PRODUCT, driver, steps)
    path = os.environ.get(f"{LIFT_PREFIX}_PROGRESS")
    label = os.environ.get(f"{LIFT_PREFIX}_LABEL", "?")
    line = (f"{label} driven={block.get('driven')} "
            f"identical={block.get('identical')} "
            f"steps={block.get('steps_compared')}")
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
    return {f"{family.FAMILY}_gate": block}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _drivers() -> List[Any]:
    return [common.build_driver(spec, 2, scale_bits=0) for _name, spec in CASES]


BUILDERS: Dict[str, Any] = {
    "host": lambda args: {
        **common.host_block(PRODUCT, __file__),
        "probes": PROBES,
        "expansion_arm": expansion(),
        "cases": [name for name, _spec in CASES],
    },
    "policy": lambda args: common.leg_policy(PRODUCT,
                                            dict(CASES)[CASES[0][0]]),
    "driver_order": lambda args: common.leg_driver_order(PRODUCT),
    "transcription": lambda args: common.leg_transcription(
        PRODUCT, common.build_driver(dict(CASES)[MUTATION_CASES[0]], 1, scale_bits=0)),
    "refusal": lambda args: common.leg_refusal(PRODUCT, REFUSALS),
    "withdraw": lambda args: common.leg_withdraw(PRODUCT, dict(CASES)[CASES[0][0]]),
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
