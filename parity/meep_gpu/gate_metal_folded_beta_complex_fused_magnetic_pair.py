#!/usr/bin/env python3
"""Native MPS byte gate for the FOLDED BETA COMPLEX fused magnetic kernel.

``step_B`` -> ``fill_symmetry_bc_B`` -> ``zero_metal_B`` -> ``fill_folded_far_ghosts_B``
-> ``update_H``, in ONE dispatch, on a folded complex grid carrying MEEP's out-of-plane
2-D wavevector ``beta``.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans FIVE driver
passes, so a per-sub-step comparison could not see it at all: the whole claim is about
the SEAM between them, and about two mirror fills carried INSIDE one dispatch whose
source cell a different thread computes, under COMPLEX storage where the parity is a
complex product rather than a sign, with the BETA insert riding in the curl. The
comparison is therefore per COMPLETE DRIVER STEP over a stated budget, against an
array-path oracle running the identical live pass list, with the first divergent step
reported rather than a final pass/fail.

**EVERY FIXTURE IN THIS FILE IS TWO-DIMENSIONAL, AND THAT IS A REFUSAL RATHER THAN A
CONVENIENCE.** ``grid._resolve_beta`` (grid.py:690) refuses a nonzero beta on a 3-D
grid, naming MEEP's own abort (fields.cpp:546-547). A gate that carried a 3-D folded
beta row would be certifying a configuration the engine cannot build, so leg
``refusal`` ASKS for one and requires the refusal by name instead.

THE ELEVEN THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that was
   never launched is byte-identical to the oracle BY CONSTRUCTION, because the oracle
   is the array path this walk would otherwise take. Every case asserts the exact
   LAUNCH COUNT (``launches_per_run`` x steps) and a nonzero ``runs``.
2. **A vacuous comparison.** The movement floor is SPLIT BY ROLE and measured on the
   DISPATCH ALONE (leg ``movement_role``): the four volume groups this seam WRITES
   must every one move, and the four it only READS must not move by a single word.
   A whole-step floor cannot see either half, because ``step_D``/``update_E`` move the
   read-only groups on the host every step.
3. **A hollow pass.** Twenty-six shader mutations, three host mutations, four
   PREDICTED NULLS and two NULL-EVERYWHERE edits are armed. Each is scored CAUGHT or NULL CONFIRMED — never merely
   "permitted" — and every declared null is paired with a COMPLEMENTARY FIXTURE on
   which the same needle must be caught, so a null is a measured property of the
   fixture and not of the kernel.
4. **A DEAD-BRANCH MUTATION.** A needle can rewrite a real line the scored case never
   reaches. Every mutation declares the CASE it is armed on, and ``main`` asserts that
   case's specialisation carries the feature the needle depends on — the right fold
   termination, a non-identity parity word, a live wall, a phased axis, a reflect row
   that is not the fixed ``n - 2``, an absorber that reaches stored cell 2 — before a
   single edit is applied. FOUR mutation cases exist because no single folded beta
   complex grid can carry all of it: a folded axis may not be phased, a 2-D grid has
   no third axis to wall when both x and y are folded, and the deep absorber that
   makes the moved coefficient observable also makes the reflect row the fixed one.
5. **A LINE NOTHING CAN SEE.** The imaged near ghost's coefficient pair is read at
   stored index 0 rather than reused from the source thread at index 2. On a folded
   axis the mirror plane carries no absorber, so at the shared matrix's 2-cell layer
   ``kps[0] == kps[2]`` EXACTLY and the mutation that reuses the wrong entry is
   UNOBSERVABLE. Leg ``moved_coefficient`` censuses the coefficients, PREDICTS
   observability from them, and requires ``caught == observable`` on both fixtures.
6. **A construction claimed rather than measured.** The curl is
   ``folded_beta.folded_beta_bloch_curl_source``'s own text and every carry below the
   cut is ``folded_complex_fused_magnetic_pair``'s own emitter, CALLED. Leg
   ``transcription`` re-runs those emitters in BOTH contraction modes and requires the
   lifted head to be a verbatim prefix, the FIVE recurrence statements to be present
   line for line, the BETA INSERT's two statements to be present, the constitutive
   half to reproduce ``complex_fields.bloch_constitutive_source('H')``, and both fill
   transcriptions to be identical.
7. **An unmeasured platform assumption.** This family is ONE dispatch for all three
   components because eleven non-pointer arguments are packed into a
   ``constant Params&``. Leg ``binding_ceiling`` compiles the 42-binding
   separate-scalar signature and requires the FAILURE, then compiles AND LAUNCHES the
   packed 28-binding one with THIS FAMILY'S OWN 120-byte record and requires every
   struct field to read back correctly.
8. **A refusal that is really an omission.** Leg ``refusal`` requires: an undeclared
   source list refused; a real MAGNETIC source refused by name; a real ELECTRIC source
   ADMITTED; an UNFOLDED beta grid refused (that is ``beta_fused_magnetic_pair``'s
   scope); a folded grid with beta = 0 refused (that is
   ``folded_complex_fused_magnetic_pair``'s scope); a REAL-storage folded beta grid
   refused; and a 3-D beta grid refused by the engine itself.
9. **A composition claimed rather than measured.** Leg ``separate_control`` steps the
   same seam with ``folded_beta``'s two ALREADY CERTIFIED products as separate
   dispatches, with both fills and the wall clear left on the HOST between them, and
   requires three-way byte agreement plus the dispatch and host-pass counts the fusion
   removes.
10. **A vacuous subnormal precondition and a single value class.** Leg
   ``value_classes`` walks gaussian, uniform and a signed-zero lattice and requires
   byte identity on each, then walks a SUBNORMAL_BAND seed and requires the census to
   FIRE. The band is a REFUSAL with the numbers behind it, never a comparison: MPS
   flushes float32 subnormals natively with no lever and NumPy keeps, so a subnormal
   in the reference state IS a divergence and no kernel can be right about it. Leg
   ``policy`` records the resolved policy and requires ``keep`` REFUSED BY NAME.
11. **AN ARM TABLE THAT MOVED.** This weld registers ``wired=False``. Leg
   ``arm_table`` requires exactly one row for this family, ``wired`` False,
   ``is_weld`` True, ``replaces`` equal to ``REPLACES``, and ``arms_for('step_B', …)``
   NOT offering it — so ``plan_step`` cannot select it and no existing arm's selection
   changes.

LEGS
  0  policy            the resolved policy, and ``keep`` refused by name
  1  expansion         BOTH probes bind the arm; missing / ambiguous / wrong-backend /
                       disagreeing probes must each REFUSE
  2  binding_ceiling   42 separate bindings must FAIL; 27 pointers + one packed
                       Params& must COMPILE, LAUNCH and read back every field
  3  transcription     the curl head, the five recurrence statements, the beta insert,
                       the constitutive half and both fills are the certified text
  4  parity_chain_order  the composed parity is a CHAIN; its ORDER moves bytes and so
                       does folding it into one word
  5  product           complete steps, per-step byte compare, launch counters, the
                       signed-zero seeding floor, the subnormal census
  6  movement_role     the DISPATCH ALONE: what must move, what must not
  7  separate_control  the certified folded beta products as two dispatches
  8  refusal           the source seam, both sibling scopes, real storage, 3-D beta
  9  value_classes     three classes identical; the band seed FIRES the census
 10  moved_coefficient predicted observability vs measured catch, on two fixtures
 11  byte_neutral_control  the register-vs-reload edit must NOT diverge
 12  mutation          twenty-six shader defects + three host defects, each CAUGHT
 13  null_pairs        four PREDICTED nulls, each confirmed null here and CAUGHT on
                       its complementary fixture
 13b null_everywhere   the TWO chain collapses, null on every fixture this family can
                       build, scored on NOT diverging with their ±1 precondition
                       measured on every product row
 14  arm_table         registered unwired, is_weld, and unreachable from arms_for
 15  disarm            the same harness, shipped bytes, must not diverge

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms as metal_arms, complex_fields, folded_beta, folded_complex,
    folded_beta_complex_fused_magnetic_pair as family, launch as metal_launch,
    shaders, special_kz, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402
from meep_gpu.metal_kernels.symmetry import MIRROR_CODES  # noqa: E402
from meep_gpu.triton_kernels.symmetry import (  # noqa: E402
    CODE_MIRROR_PERIODIC, folded_axis_kinds,
)

#: The budget every case runs. Twelve, matching every other fused pair's and the
#: whole-step arbiter's, and for their reason: the classes this gate exists for
#: COMPOUND. ``fu_B`` and ``f_w_H`` are state carried between steps, and a ghost plane
#: imaged one row over is a defect that needs several steps to reach the low bits of
#: the interior. The comparison is per COMPLETE STEP.
STEPS = 12

#: THE BASE SEED. Every case's actual seed is a DIGEST of the case's identity added to
#: this, never ``hash()``: Python salts ``hash()`` of a string with PYTHONHASHSEED, so
#: a hash-seeded gate draws a different fixture every process and a failing case cannot
#: be replayed.
SEED = 9_300_000


def seed_for(*parts: str) -> int:
    """A stable per-case seed from a sha256 digest of the case's identity."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big")


#: Which array-path function each live pass is. BOTH fill passes are in this table:
#: they are the difference between an unfolded seam and a folded one, and this weld
#: carries them inline.
ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
    "update_P": lambda f, p: stepping.update_P(f, p),
}

#: Every stored volume a folded complete step can touch. The D/E half is in this list
#: even though this seam does not touch it, because a fused pair that corrupted B would
#: reach E through ``step_D`` on the very next step, and a comparison blind to that
#: would be reporting on half the engine.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: THE ROLE SPLIT, and the movement floor is split on exactly this line. The kernel
#: binds three B volumes, three ``fu_B``, three E (READ ONLY — they are the curl's
#: sources), three H and three ``f_w_H``. Everything else is bound nowhere and must not
#: move by a single word when the dispatch runs alone.
WRITTEN_NAMES: Tuple[str, ...] = tuple(
    [f"B{axis}" for axis in "xyz"] + [f"fu_B{axis}" for axis in "xyz"]
    + [f"H{axis}" for axis in "xyz"] + [f"f_w_H{axis}" for axis in "xyz"])
READ_ONLY_NAMES: Tuple[str, ...] = tuple(
    name for name in STATE_NAMES if name not in WRITTEN_NAMES)

#: The value classes leg's seeds, and what each is for.
VALUE_CLASSES: Tuple[str, ...] = ("gaussian", "uniform", "signed_zero_lattice")
SUBNORMAL_BAND = "subnormal_band"

#: name -> (``metal_composition_matrix.folded`` keywords, explicit PML faces or None).
#:
#: EVERY ROW IS ``complex_storage=True`` AND CARRIES A NONZERO ``beta``, because either
#: one missing is a different family: beta = 0 is
#: :mod:`.folded_complex_fused_magnetic_pair`'s shipped weld and real storage is
#: :mod:`.folded_beta`'s real arm. What the rows carry between them:
#:
#:   FOLD TERMINATION   MIRROR_METALLIC (the top plane is owned and stepped, no far
#:                      ghost; ``fill_folded_far_ghosts_B`` is not even live) and
#:                      MIRROR_PERIODIC (the stored array carries the not-owned slot
#:                      past ``big_corner`` and the far fill images it). A mutation
#:                      caught on one is not caught on the other;
#:   PARITY             even and odd, and MIXED on a two-axis fold. The parity is a
#:                      RUNTIME complex word here, so two rows differing only in parity
#:                      compile to the SAME kernel and differ only in the Params record
#:                      — which is exactly why both must be walked. It is also what
#:                      makes the near and far words each non-identity on SOME row:
#:                      ``mirror_parity`` is ``phase * (1 - 2*iyee)``, so on one folded
#:                      axis the near and far words are always OPPOSITE and only a
#:                      two-axis mixed-parity row carries a non-identity of each;
#:   FULL-COUNT PARITY  ``extent`` decides whether ``_far_reflect_rows``' row is
#:                      ``stored - 2`` (even count) or ``stored - 3`` (odd). At the
#:                      default 2.0 the WRONG fixed ``n - 2`` happens to be right;
#:   WALL               ``zero_metal_B`` only emits a line for an axis that is metallic
#:                      and NOT mirrored. On a 2-D grid the z axis cannot be walled at
#:                      all (matrix.folded refuses it by name), so a walled row must
#:                      leave x or y unfolded;
#:   BLOCH PHASE        ``k = 0`` is the REDUCTION (no rotation emitted at all);
#:                      ``k = 0.3`` is a generic interior point; ``k = 0.5`` is the
#:                      BRILLOUIN EDGE where the phase is exactly ``-1+0j`` and a
#:                      plane-wise collapse would be invisible on random data. A FOLDED
#:                      axis may never be phased, so every phased row phases x beside a
#:                      fold on y;
#:   BETA SIGN          both signs, because the two curl coefficients are ``+i*beta``
#:                      and ``-i*beta`` and a swap between them is a mutation this file
#:                      arms; a single-sign matrix would leave that armed at one word;
#:   ABSORBER DEPTH     the matrix gives a folded axis a 2-cell high-face layer, on
#:                      which ``kps[0] == kps[2]`` to the bit. Two rows replace it with
#:                      an explicit 4-cell layer on a 6-cell axis, which is the only
#:                      shape that reaches stored cell 2 and the only one on which this
#:                      family's moved NEAR coefficient index is observable at all;
#:   OFF-DIAGONAL eps   ``update_H`` reads B and mu and NEVER an inverse epsilon, which
#:                      is why this signature drops the certified complex
#:                      constitutive's three inverse-epsilon pointers and binds 27
#:                      rather than 30. If that were wrong, an off-diagonal chi1inv row
#:                      is where it would show, and the walk runs ``update_E`` on the
#:                      array path every step.
CASES: Tuple[Tuple[str, Dict[str, Any], Optional[Any]], ...] = (
    ("y_periodic_even_2d", dict(complex_storage=True, beta=0.3), None),
    ("y_periodic_odd_count_2d",
     dict(complex_storage=True, beta=0.3, extent=2.1), None),
    ("y_periodic_odd_parity_2d",
     dict(complex_storage=True, beta=-0.41, phase=-1, extent=2.1), None),
    ("y_metallic_even_2d",
     dict(complex_storage=True, beta=-0.41, boundaries={"y": "metallic"}), None),
    ("y_metallic_wall_x_2d",
     dict(complex_storage=True, beta=0.3,
          boundaries={"y": "metallic", "x": "metallic"}), None),
    # THE WALL MUTATION CASE. A folded PERIODIC y at ODD parity beside a walled x, and
    # a 4-cell high-face absorber so the near coefficient index is observable too.
    ("y_periodic_wall_x_deep_2d",
     dict(complex_storage=True, beta=0.3, phase=-1, extent=0.8,
          boundaries={"x": "metallic"}), ((0, 4), (0, 4), (0, 0))),
    ("xy_mixed_2d",
     dict(complex_storage=True, beta=0.3, axis="XY", phase=(1, -1)), None),
    # THE FOLD MUTATION CASE. Two folded PERIODIC axes at MIXED parities, so the NEAR
    # word on y and the FAR word on x are both non-identity, a component owns a FAR/FAR
    # corner (a two-parity CHAIN), and the deep absorber reaches stored cell 2.
    ("xy_mixed_deep_2d",
     dict(complex_storage=True, beta=0.3, axis="XY", phase=(1, -1), extent=0.8),
     ((0, 4), (0, 4), (0, 0))),
    # THE REFLECT MUTATION CASE. Same fold, ODD full count, so the reflect row is
    # ``stored - 3`` and the fixed ``n - 2`` is a whole cell wrong.
    ("xy_mixed_odd_count_2d",
     dict(complex_storage=True, beta=0.3, axis="XY", phase=(1, -1), extent=1.1),
     ((0, 4), (0, 4), (0, 0))),
    ("xy_mixed_negative_beta_2d",
     dict(complex_storage=True, beta=-0.41, axis="XY", phase=(-1, 1)), None),
    ("y_bloch_kx_2d",
     dict(complex_storage=True, beta=0.3, k_point=(0.3, 0.0, 0.0)), None),
    ("y_bloch_edge_kx_2d",
     dict(complex_storage=True, beta=0.3, k_point=(0.5, 0.0, 0.0)), None),
    # THE PHASE MUTATION CASE. A phased x beside a folded PERIODIC y at ODD parity, so
    # both the rotation block and the fold's two ghost carries are present at once.
    ("y_odd_bloch_kx_2d",
     dict(complex_storage=True, beta=0.3, phase=-1, k_point=(0.3, 0.0, 0.0)), None),
    ("y_metallic_bloch_kx_2d",
     dict(complex_storage=True, beta=-0.41, boundaries={"y": "metallic"},
          k_point=(0.3, 0.0, 0.0)), None),
    ("y_periodic_offdiag_2d",
     dict(complex_storage=True, beta=0.3,
          rows={"Ex": ("Ey",), "Ey": ("Ex",)}), None),
    # The SHALLOW twin of the deep row: identical fold, identical parities, only the
    # absorber differs. It is the fixture on which the moved coefficient index is
    # PREDICTED unobservable, and leg ``moved_coefficient`` requires the prediction.
    ("xy_mixed_shallow_2d",
     dict(complex_storage=True, beta=0.3, axis="XY", phase=(1, -1), extent=0.8), None),
)

FOLD_CASE = "xy_mixed_deep_2d"
WALL_CASE = "y_periodic_wall_x_deep_2d"
PHASE_CASE = "y_odd_bloch_kx_2d"
REFLECT_CASE = "xy_mixed_odd_count_2d"

#: The fixture pair leg ``moved_coefficient`` measures on, and the axis it asks about.
#: Same fold, same parities, only the absorber differs.
SHALLOW_CASE = "xy_mixed_shallow_2d"
COEFFICIENT_AXIS = 1

#: The parity words the two parity needles rewrite, named as data so ``main`` can
#: assert the SAME word the needle edits is non-identity. An identity word is not a
#: defect to remove: the mutation would be arithmetically a no-op and would score
#: against nothing.
NEAR_PARITY_AXIS = 1
FAR_PARITY_AXIS = 0

#: Bound by leg ``expansion`` and read by every later leg. Deliberately NOT defaulted:
#: a leg that ran before the probes would bind a guess.
EXPANSION: Optional[str] = None
PARITY_PROBE: Any = None
BETA_PROBE: Any = None


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose.

    A complex64 array's word view is its two planes interleaved, which is exactly what
    the kernel writes, so this is the same comparison a real gate makes with no
    complex-specific relaxation anywhere.
    """
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def complex_from_planes(real: Any, imag: Any) -> Any:
    """Build a complex64 array from two float32 planes THROUGH THE WORD VIEW.

    ``real + 1j*imag`` DESTROYS THE SIGN OF ZEROS in ``imag``: ``1j`` is a complex128
    scalar, so the product is a full complex multiply whose imaginary part comes out
    ``-0.0`` regardless of what ``imag`` held. Every complex construction in this file
    goes through the word view, for that reason.
    """
    out = np.empty(np.shape(real), dtype=np.complex64)
    view = out.view(np.float32)
    view[..., 0::2] = np.asarray(real, dtype=np.float32)
    view[..., 1::2] = np.asarray(imag, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

def _planes(rng: Any, shape: Sequence[int], value_class: str
            ) -> Tuple[np.ndarray, np.ndarray]:
    """One (real, imaginary) plane pair in ONE value class."""
    if value_class == "gaussian":
        real = (rng.standard_normal(shape) * 0.37).astype(np.float32)
        imag = (rng.standard_normal(shape) * 0.29).astype(np.float32)
        # SIGNED ZEROS SEEDED INTO BOTH PLANES. They are the class the complex helpers'
        # literal zero cross terms exist for, and the class the parity CHAIN's order
        # and grouping are observable on at all: folding a two-parity chain into one
        # word is exact on every ordinary float and moves words only where a zero's
        # sign rides through the cross term. Random seeding alone would arm the two
        # chain mutations at nothing.
        real.reshape(-1)[::17] = np.float32(-0.0)
        real.reshape(-1)[7::23] = np.float32(0.0)
        imag.reshape(-1)[3::19] = np.float32(-0.0)
        imag.reshape(-1)[11::29] = np.float32(0.0)
        return real, imag
    if value_class == "uniform":
        return (rng.uniform(-1.0, 1.0, size=shape).astype(np.float32),
                rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
    if value_class == "signed_zero_lattice":
        # +0.0 and -0.0 on a checkerboard, in BOTH planes. The two words differ
        # (0x00000000 vs 0x80000000) and a byte gate SEES the difference, so this is
        # what exercises every ``flag ? float2(0,0) : v``, every subtraction's sign
        # convention, and the parity and beta products against an exact zero.
        real = np.where(rng.integers(0, 2, size=shape) == 0, 0.0, -0.0)
        imag = np.where(rng.integers(0, 2, size=shape) == 0, 0.0, -0.0)
        return real.astype(np.float32), imag.astype(np.float32)
    if value_class == SUBNORMAL_BAND:
        return ((rng.standard_normal(shape) * 1e-38).astype(np.float32),
                (rng.standard_normal(shape) * 1e-38).astype(np.float32))
    raise ValueError(f"unknown value class {value_class!r}")


def build(keywords: Mapping[str, Any], seed: int, pml_faces: Optional[Any] = None,
          value_class: str = "gaussian") -> Tuple[Any, Any]:
    """One seeded COMPLEX folded BETA engine. Called twice (or three times) per case.

    The GRID and the MATERIAL come from the shared composition matrix, which is the
    same builder both halves' own gates use, so the fixture this certifies is the
    fixture they were certified on. Only the field state is reseeded here, and
    identically on every side — the epsilon volume must stay bit-equal or the
    comparison measures the material rather than the kernel.

    ``pml_faces`` REPLACES the matrix's absorber with an explicitly requested one. Two
    cases use it and the reason is measured rather than aesthetic: the matrix gives a
    folded axis a 2-cell high-face layer, and on such a layer ``kps[0] == kps[2]`` to
    the bit, which makes this family's moved coefficient index unobservable.
    """
    fields, pml = matrix.folded(**dict(keywords))
    if pml_faces is not None:
        pml = PML(grid=fields.grid, thickness=pml_faces)
    rng = np.random.default_rng(seed)
    shape = tuple(fields.grid.shape)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        real, imag = _planes(rng, shape, value_class)
        array[...] = complex_from_planes(real, imag)
    return fields, pml


def state_of(fields: Any) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Any, right: Any) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


def plan_for(fields: Any, pml: Any, residency: Residency,
             functions: Optional[Mapping[str, Any]] = None) -> Any:
    """The shipped plan, with BOTH probe artifacts threaded rather than re-read."""
    return family.plan_metal_folded_beta_complex_fused_magnetic_pair(
        fields, pml, sources=(), residency=residency, functions=functions,
        probe=PARITY_PROBE, beta_probe=BETA_PROBE)


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    """The composer's OWN live set, never a second model of what a step is.

    ``None`` is a refusal rather than an empty tuple: a gate that stepped nothing would
    compare a no-op with a no-op and pass.
    """
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    return tuple(live)


def array_step(fields: Any, pml: Any, live: Sequence[str]) -> None:
    for name in live:
        ARRAY_PATH[name](fields, pml)


def metal_step(fields: Any, pml: Any, dispatch: Mapping[str, Any],
               owned: Sequence[str], residency: Residency,
               live: Sequence[str]) -> None:
    """One complete step, on the device where a plan owns the pass.

    A pass no plan owns runs on the array path and is bracketed with an explicit
    ``sync_out`` / ``sync_in``. That bracket is the residency clause made operational:
    without it the next device launch would read the mirror's stale bytes, which is
    smooth, plausible and wrong.

    ``owned`` and ``dispatch`` are SEPARATE because this plan owns FIVE passes and
    dispatches at one of them. Deriving the skip set from the dispatch keys would
    silently leave both fills and the wall clear running on the host on top of the
    copies the kernel already carried — which, for the fills, is IDEMPOTENT and would
    hide a dropped carry entirely.
    """
    skip = set(owned)
    for name in live:
        if name in skip:
            plan = dispatch.get(name)
            if plan is not None:
                plan.run()
            continue
        residency.sync_out()
        ARRAY_PATH[name](fields, pml)
        residency.sync_in()


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(keywords: Mapping[str, Any], seed: int, steps: int,
             pml_faces: Optional[Any] = None,
             functions: Optional[Mapping[str, Any]] = None,
             patch: Optional[Callable[[Any, Any, Residency], None]] = None,
             value_class: str = "gaussian",
             expect_subnormal_free: bool = True,
             ) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    reference, reference_pml = build(keywords, seed, pml_faces, value_class)
    actual, actual_pml = build(keywords, seed, pml_faces, value_class)

    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    plan = plan_for(actual, actual_pml, residency, functions)
    if plan is None:
        reasons = family.metal_folded_beta_complex_fused_magnetic_pair_coverage(
            actual, actual_pml, (), residency, PARITY_PROBE, BETA_PROBE).reasons
        return {"passed": False,
                "reason": "the folded beta complex fused magnetic pair was refused",
                "refusals": list(reasons)}
    if patch is not None:
        patch(plan, actual_pml, residency)

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    before = frozen(actual)
    # THE SIGNED-ZERO SEEDING FLOOR. A census of zero would mean the seeding never
    # built the class the literal zero cross terms and the parity chain's grouping are
    # observable on, and several mutations below would be armed at nothing. Not
    # required on the band seed, which has no zeros by construction.
    zeros = subnormal.signed_zero_census(
        np.concatenate([words(value).view(np.float32) for value in before.values()]))
    residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(actual, actual_pml, {family.SLOT: plan},
                   plan.replaces_sub_steps, residency, live)
        residency.sync_out()
        difference = compare(reference, actual)
        census = sum(subnormal.census(value)
                     for value in state_of(reference).values())
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": int(census)})
        if difference or (census and expect_subnormal_free):
            break

    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    clean = all(row["reference_subnormals"] == 0 for row in per_step)
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step))
    seeded_zeros = (int(zeros.get("negative_zero", 0))
                    + int(zeros.get("positive_zero", 0)))
    # The WHOLE-STEP floor: every compared array must move, because every live pass
    # runs. The role split is a DIFFERENT and stronger measurement and it lives in leg
    # ``movement_role``, which runs the dispatch alone.
    return {
        "passed": bool(identical and (clean or not expect_subnormal_free)
                       and launches_ok and not still),
        "bit_identical": identical,
        "subnormal_free": clean,
        "seeded_signed_zeros": dict(zeros),
        "seeded_signed_zero_count": seeded_zeros,
        "value_class": value_class,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "reference_subnormals": per_step[-1]["reference_subnormals"],
        "arrays_compared": len(before),
        "arrays_that_never_moved": still,
        "moved_words": int(sum(moved.values())),
        "runs": plan.runs, "launches": plan.launches,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
        "boundary_codes": list(plan.codes), "phased": list(plan.phased),
        "phases": list(plan.phases),
        "parity_values": [list(value) for value in plan.parity_values],
        "far_parity_values": [list(value) for value in plan.far_parity_values],
        "beta_values": [list(value) for value in plan.beta_values],
        "reflect_rows": list(plan.reflect),
        "zero_metal": list(plan.zero_metal),
        "carried_ghost_axes": [list(axes) for axes in plan.carried_axes],
        "carried_far_ghost_axes": [list(axes) for axes in plan.far_carried_axes],
        "expansion": plan.expansion,
        "has_offdiagonal_epsilon": bool(
            getattr(actual, "has_offdiagonal_epsilon", False)),
        "mirrors": len(residency.names),
        "shape": list(plan.shape),
    }


# ---------------------------------------------------------------------------
# The movement floor, SPLIT BY ROLE, on the dispatch alone
# ---------------------------------------------------------------------------

def leg_movement_role(name: str, keywords: Mapping[str, Any], seed: int,
                      pml_faces: Optional[Any] = None) -> Dict[str, Any]:
    """ONE dispatch, nothing else, and the floor split on what the seam WRITES.

    THE WHOLE-STEP FLOOR CANNOT SEE THIS. Every product row runs the complete live pass
    list, so ``step_D`` and ``update_E`` move D, E, ``fu_D`` and ``f_w_E`` on the host
    at every step and a floor over all arrays would report them as moved no matter what
    the kernel did. Here the engine is seeded, mirrored in, the fused plan is launched
    ONCE, and the state is read back:

    * the four groups the signature binds as OUTPUTS — B, ``fu_B``, H, ``f_w_H`` —
      must every one MOVE. A group that never moved would mean a whole store was
      compiled out and the byte comparison agreed because the array path had not
      reached that volume yet either;
    * everything else must not move BY A SINGLE WORD. The E volumes are bound and are
      READ ONLY; D, ``fu_D`` and ``f_w_E`` are bound nowhere at all, so a word moving
      there is an out-of-range write, which is the one defect a byte comparison against
      an oracle running the same passes can be blind to.
    """
    fields, pml = build(keywords, seed, pml_faces)
    residency = Residency()
    plan = plan_for(fields, pml, residency)
    if plan is None:
        return {"passed": False, "label": name,
                "reason": "the plan was refused on the movement-floor case"}
    before = frozen(fields)
    residency.sync_in()
    plan.run()
    residency.sync_out()
    after = state_of(fields)
    moved = {key: differing(before[key], after[key]) for key in before}
    written_still = sorted(key for key in WRITTEN_NAMES
                           if key in moved and moved[key] == 0)
    read_disturbed = sorted(key for key in READ_ONLY_NAMES
                            if key in moved and moved[key])
    return {
        "passed": bool(not written_still and not read_disturbed
                       and plan.launches == plan.launches_per_run),
        "label": name,
        "launches": plan.launches,
        "written_arrays": [key for key in WRITTEN_NAMES if key in moved],
        "written_arrays_that_never_moved": written_still,
        "read_only_arrays": [key for key in READ_ONLY_NAMES if key in moved],
        "read_only_arrays_the_dispatch_disturbed": read_disturbed,
        "moved_words_by_array": {key: moved[key] for key in sorted(moved)},
    }


# ---------------------------------------------------------------------------
# The separate control: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def run_separate_control(keywords: Mapping[str, Any], seed: int, steps: int,
                         pml_faces: Optional[Any] = None) -> Dict[str, Any]:
    """The SEPARATE certified folded beta products beside the fused one, same state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three engines
    from one seed: the array path; ``folded_beta``'s two ALREADY CERTIFIED products —
    the complex Bloch beta curl on ``step_B`` and the complex constitutive on ``H`` —
    stepping the same seam as separate dispatches with BOTH mirror fills and the wall
    clear left on the HOST between them; and the fused product. All three must agree
    word for word at every complete step, and the DISPATCH COUNTS and the surviving
    in-seam host passes are recorded on both sides — which is the only place the
    difference between the compositions shows up at all, since a correct fusion is
    byte-neutral by construction.

    The separate side is not a strawman: those two plans are exactly what ``plan_step``
    composes today for this configuration, which is also why this weld registers
    unwired (it would contend with the curl arm on ``step_B`` and leave the slot
    unselected).
    """
    reference, reference_pml = build(keywords, seed, pml_faces)
    separate, separate_pml = build(keywords, seed, pml_faces)
    fused, fused_pml = build(keywords, seed, pml_faces)

    separate_residency = Residency()
    curl = folded_beta.plan_folded_beta_bloch_pml_curl(
        separate, separate_pml, "step_B", separate_residency, probe=BETA_PROBE)
    magnetic = folded_beta.plan_folded_beta_complex_constitutive(
        separate, separate_pml, "H", separate_residency, probe=BETA_PROBE)
    fused_residency = Residency()
    plan = plan_for(fused, fused_pml, fused_residency)
    if curl is None or magnetic is None or plan is None:
        return {"passed": False, "reason": "a declared product was refused",
                "curl": curl is not None, "constitutive": magnetic is not None,
                "fused": plan is not None}

    live = live_passes(fused, fused_pml)
    separate_residency.sync_in()
    fused_residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, reference_pml, live)
        metal_step(separate, separate_pml, {"step_B": curl, "update_H": magnetic},
                   ("step_B", "update_H"), separate_residency, live)
        metal_step(fused, fused_pml, {family.SLOT: plan},
                   plan.replaces_sub_steps, fused_residency, live)
        separate_residency.sync_out()
        fused_residency.sync_out()
        per_step.append({
            "step": step,
            "separate_vs_array": sum(compare(reference, separate).values()),
            "fused_vs_array": sum(compare(reference, fused).values()),
            "fused_vs_separate": sum(compare(separate, fused).values()),
        })
        if any(value for key, value in per_step[-1].items() if key != "step"):
            break

    # The host passes each composition leaves on the array path INSIDE the seam. The
    # fused side carries both fills and the wall clear in the kernel; the separate side
    # does not, so on every run it pays host round trips the fused side does not.
    seam = ("fill_B", "zero_metal_B", "fill_folded_far_ghosts_B")
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    return {
        "passed": bool(identical and plan.launches and curl.launches
                       and magnetic.launches),
        "per_step": per_step,
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": (curl.launches_per_run
                                         + magnetic.launches_per_run),
        "fused_dispatches_per_step": plan.launches_per_run,
        "separate_launches": curl.launches + magnetic.launches,
        "fused_launches": plan.launches,
        "seam_host_passes_for_separate": [n for n in live if n in seam],
        "seam_host_passes_for_fused": [],
        "driver_passes_the_fusion_removes":
            len(plan.replaces_sub_steps) - plan.launches_per_run,
        "live_passes": list(live),
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def _specialisation(fields: Any, pml: Any) -> Dict[str, Any]:
    """The (codes, Bloch flags, walls, parities) the SHIPPED plan compiles from."""
    residency = Residency()
    plan = plan_for(fields, pml, residency)
    assert plan is not None, (
        "the shipped plan was refused for a case the gate arms mutations on; every "
        "needle below would then be applied to a source nothing launches")
    return {"codes": plan.codes, "phased": plan.phased, "walls": plan.zero_metal,
            "phases": plan.phases, "expansion": plan.expansion,
            "near": plan.parity_values, "far": plan.far_parity_values,
            "reflect": plan.reflect, "shape": plan.shape,
            "carried": plan.carried_axes, "far_carried": plan.far_carried_axes}


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``count`` occurrences and REFUSE a no-op edit.

    A mutation that changed nothing would launch the shipped kernel and report the
    defect as uncaught, which is the one failure mode a mutation leg cannot see from
    its own result.
    """
    if old not in source:
        raise AssertionError(f"mutation needle is absent from the source: {old!r}")
    mutated = source.replace(old, new, count)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


#: ``name -> (case, old, new, count, why)``. The CASE is declared with the needle so
#: ``main`` can check that case's specialisation carries the feature the line depends
#: on before the edit is applied — the dead-branch guard.
SHADER_EDITS: Dict[str, Tuple[str, str, str, int, str]] = {
    # ---- THE BETA INSERT: the whole reason this family is not its twin -----------
    "beta_signs_swapped": (
        FOLD_CASE,
        "float bpr = prm.bp.x, bpi = prm.bp.y;\n    "
        "float bmr = prm.bm.x, bmi = prm.bm.y;",
        "float bpr = prm.bm.x, bpi = prm.bm.y;\n    "
        "float bmr = prm.bp.x, bmi = prm.bp.y;", 1,
        "the two coefficients are +i*beta and -i*beta; swapping them is the single "
        "most likely slip in binding the pair and is not a crash"),
    "beta_term_dropped": (
        FOLD_CASE, "curl0 = curl0 - c_mul(float2(bpr, bpi), b);",
        "// beta term dropped", 1,
        "a has_beta=0 build IS the shipped folded complex weld, so this must diverge "
        "or the two products would be the same kernel"),
    "beta_term_sign_flipped": (
        FOLD_CASE, "curl0 = curl0 - c_mul(float2(bpr, bpi), b);",
        "curl0 = curl0 + c_mul(float2(bpr, bpi), b);", 1,
        "the array path adds the NEGATED product (stepping._special_kz_beta_term)"),
    "beta_partner_takes_the_shifted_operand": (
        FOLD_CASE, "curl1 = curl1 - c_mul(float2(bmr, bmi), a);",
        "curl1 = curl1 - c_mul(float2(bmr, bmi), a_y);", 1,
        "the beta partners are the unrotated CENTERS; reaching for the shifted "
        "operand is the slip the lifted head's comment exists to prevent"),
    # ---- THE PARITY WORDS -------------------------------------------------------
    "near_parity_forced_to_plus_one": (
        FOLD_CASE, "c_mul(mp1,", "c_mul(float2(1.0f, 0.0f),", -1,
        "the near mirror parity on the odd axis is -1; dropping it images the wrong "
        "sign into the ghost plane"),
    "far_parity_forced_to_plus_one": (
        FOLD_CASE, "c_mul(fp0,", "c_mul(float2(1.0f, 0.0f),", -1,
        "the far parity on the even axis is -1 (mirror_parity is phase*(1-2*iyee), so "
        "near and far are opposite on one axis)"),
    "far_parity_takes_the_near_word": (
        FOLD_CASE, "c_mul(fp0,", "c_mul(mp0,", -1,
        "the far word is MINUS the near one; a struct member read one slot over is a "
        "plausible complex number rather than garbage"),
    # ---- THE GHOST GEOMETRY -----------------------------------------------------
    "near_source_index_moved": (
        FOLD_CASE, "int g1_n_i = ii - 2 * nzi;", "int g1_n_i = ii - 1 * nzi;", 1,
        "the near fill images stored cell 0 from cell 2; one row over is a plane of "
        "smooth wrong values"),
    "far_carry_index_zeroed": (
        FOLD_CASE, "int g2_j_i = ii + ((nyi - 1) - reflect_y) * nzi;",
        "int g2_j_i = ii + 0 * ((nyi - 1) - reflect_y) * nzi;", 1,
        "a zeroed far offset writes the ghost onto the source cell itself"),
    "ghost_coefficient_reused_from_source": (
        FOLD_CASE, "float g1_n_kp = kp1[0], g1_n_km = km1[0];",
        "float g1_n_kp = kp_1, g1_n_km = km_1;", 1,
        "the DESTINATION's own coefficient entry, not the source thread's at index 2; "
        "observable only where the absorber reaches stored cell 2"),
    # ---- THE MASKS --------------------------------------------------------------
    "ownership_mask_dropped": (
        FOLD_CASE, "    curl0 = at_x ? float2(0.0f, 0.0f) : curl0;\n", "", 1,
        "stepping._mask_non_owned_cells zeroes the component's own cell 0"),
    "top_plane_mask_dropped": (
        FOLD_CASE, "    curl0 = last_y ? float2(0.0f, 0.0f) : curl0;\n", "", 1,
        "the top plane of a folded PERIODIC axis sits past big_corner; the FILL writes "
        "it and the curl must not"),
    # ---- THE CURL ---------------------------------------------------------------
    "curl_direction_reversed": (
        FOLD_CASE, "int si = i + 1, sj = j + 1, sk = k + 1;",
        "int si = i - 1, sj = j - 1, sk = k - 1;", 1,
        "step_B is FORWARD; step_D's negated strides are a different product"),
    "curl_parens_flattened": (
        FOLD_CASE, "float2 t0 = ((c_y - c) + (b - b_z));",
        "float2 t0 = (c_y - c + b - b_z);", 1,
        "shaders.py rule 2: flattening these parens is a different float32 number"),
    "recurrence_axis_pair_swapped": (
        FOLD_CASE, "float2 q0 = c_mul_field_left(p0, km_y) - curl0;",
        "float2 q0 = c_mul_field_left(p0, km_z) - curl0;", 1,
        "the recurrence pairs are vec.hpp's cycle_direction: target 0 takes (y, z)"),
    "sinv_scaling_dropped": (
        FOLD_CASE, "float2 n0 = c_mul_field_left(q0, si_y);", "float2 n0 = q0;", 1,
        "the split-field recurrence's second factor"),
    "fu_store_dropped": (
        FOLD_CASE, "u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;",
        "u1[ii] = n1; u2[ii] = n2;", 1,
        "the PML auxiliary is STATE; a dropped store diverges only once it accumulates"),
    "flux_store_dropped": (
        FOLD_CASE, "        f0[ii] = v0;\n", "        // flux store dropped\n", 1,
        "the stepped displacement itself"),
    "fw_store_dropped": (
        FOLD_CASE, "        w0[ii] = o0_src;", "        // stale f_w_Hx", 1,
        "the constitutive workspace is STATE and feeds the next step's prev term"),
    # ---- THE SEAM ---------------------------------------------------------------
    "seam_takes_the_pre_store_value": (
        FOLD_CASE, "float2 o0_src = v0;", "float2 o0_src = n0;", 1,
        "the seam takes the STEPPED displacement, not the split-field auxiliary"),
    "seam_takes_the_wrong_component": (
        FOLD_CASE, "float2 o2_src = v2;", "float2 o2_src = n2;", 1,
        "the same slip on the third component, where the recurrence pairs differ"),
    # ---- THE CONSTITUTIVE HALF --------------------------------------------------
    "constitutive_coefficient_index_moved": (
        FOLD_CASE, "float kp_0 = kp0[i], km_0 = km0[i];",
        "float kp_0 = kp0[j], km_0 = km0[j];", 1,
        "the coefficient is indexed on the component's OWN axis (stepping.py:226-227)"),  # stepping.py live lines for the frozen device-text citation(s) in this string: 226-227->227-228
    "constitutive_accumulations_reversed": (
        FOLD_CASE,
        "o0_acc = o0_acc + c_mul_coefficient_left(kp_0, o0_src);\n        "
        "o0_acc = o0_acc - c_mul_coefficient_left(km_0, o0_prev);",
        "o0_acc = o0_acc - c_mul_coefficient_left(km_0, o0_src);\n        "
        "o0_acc = o0_acc + c_mul_coefficient_left(kp_0, o0_prev);", 1,
        "kps multiplies the NEW source and kms the previous one"),
    # ---- THE WALL, on the only case that emits one ------------------------------
    "wall_clear_dropped": (
        WALL_CASE, "v0 = at_x ? float2(0.0f, 0.0f) : v0;", "v0 = at_x ? v0 : v0;", 1,
        "zero_metal_B clears the component's own cell 0 on its own metallic axis"),
    "wall_clear_uses_the_d_side_table": (
        WALL_CASE, "v0 = at_x ? float2(0.0f, 0.0f) : v0;",
        "v0 = at_y ? float2(0.0f, 0.0f) : v0;", 1,
        "the B wall table is the DIAGONAL and D's is the off-diagonal; reusing the "
        "sibling's table is the most likely port slip"),
    # ---- THE BLOCH ROTATION, on the only case that emits one --------------------
    "bloch_phase_conjugated": (
        PHASE_CASE, "c_mul(b_x, float2(pxr, pxi))",
        "c_mul(b_x, float2(pxr, -pxi))", -1,
        "step_B and step_D take opposite conjugations; the host resolves it and the "
        "kernel cannot tell"),
    "bloch_rotation_applied_to_the_center": (
        PHASE_CASE, "b_x = wx ? c_mul(b_x, float2(pxr, pxi)) : b_x;",
        "b = wx ? c_mul(b, float2(pxr, pxi)) : b;", 1,
        "SHIFTED operands only; rotating the center would also rotate the beta "
        "partner, which the array path never does"),
    "bloch_rotation_dropped": (
        PHASE_CASE, "    b_x = wx ? c_mul(b_x, float2(pxr, pxi)) : b_x;\n", "", 1,
        "the wrapped face carries the phase (stepping:1862)"),
    # ---- THE REFLECT ROW, on the only case where it is not the fixed n - 2 -------
    "reflect_row_is_the_fixed_n_minus_two": (
        REFLECT_CASE, "int g2_j_i = ii + ((nyi - 1) - reflect_y) * nzi;",
        "int g2_j_i = ii + ((nyi - 1) - (nyi - 2)) * nzi;", 1,
        "_far_reflect_rows is n_full - stored + 2, which is stored - 3 at an ODD full "
        "count; the fixed n - 2 is right only at an even one"),
}

#: The PREDICTED NULLS. Each is the SAME needle as a scored mutation above, run on a
#: fixture whose specialisation makes it arithmetically a no-op. A null scored as
#: "uncaught" would be a defect; here each one is required to be null HERE and CAUGHT on
#: its complementary fixture, which is what makes the null a property of the fixture.
#: ``name -> (null_case, caught_case, needle_name, why)``.
NULL_PAIRS: Dict[str, Tuple[str, str, str, str]] = {
    "near_parity_null_at_even_parity": (
        "y_periodic_even_2d", WALL_CASE, "near_parity_forced_to_plus_one",
        "mirror_parity is phase*(1-2*iyee); at phase=+1 the NEAR word IS (1,0), so "
        "substituting the literal changes no value. Caught at phase=-1"),
    "far_parity_null_at_odd_parity": (
        "y_periodic_odd_parity_2d", "y_periodic_even_2d",
        "far_parity_forced_to_plus_one_axis1",
        "the FAR word is MINUS the plane's phase, so an ODD plane hands the far pass "
        "the identity. Caught on the EVEN row, where the far word is -1"),
    "ghost_coefficient_null_on_a_shallow_layer": (
        SHALLOW_CASE, FOLD_CASE, "ghost_coefficient_reused_from_source",
        "a folded axis carries no absorber at the mirror plane, so on the matrix's "
        "2-cell high-face layer kps[0] == kps[2] to the bit and the two reads are the "
        "same word. Caught with an explicit 4-cell layer on a 6-cell axis"),
    "reflect_row_null_at_an_even_full_count": (
        FOLD_CASE, REFLECT_CASE, "reflect_row_is_the_fixed_n_minus_two",
        "at an EVEN full count _far_reflect_rows IS stored - 2, so the substituted "
        "constant is the value it replaces. Caught at an odd full count"),
}

#: THE TWO CHAIN DEFECTS: NULL ON EVERY FIXTURE THIS FAMILY CAN BUILD, and null by
#: MEASUREMENT rather than by preference. They were armed here as CAUGHT-required first
#: and both reported 0 differing words with the plan launched, on the FAR/FAR corner of
#: :data:`FOLD_CASE`; the reason is that ``mirror_parity`` is ``phase * (1 - 2*iyee)``
#: with ``phase`` in {+1, -1}, so every parity word is exactly ``±1 + 0j`` and ±1
#: composes EXACTLY in float32 — the cross terms are products with an exact +0.0 and
#: cannot move a normal operand's bits under EITHER order or grouping. The shipped
#: folded complex weld's own gate reached the same verdict independently and records it
#: the same way (``gate_metal_folded_complex_fused_magnetic_pair.NULL_EDITS``, entries
#: ``parity_chain_folded_into_one_word`` / ``parity_chain_order_reversed``, measured at
#: ``results/metal_folded_complex_far_carry_2026-08-21/`` legs 62 and 63).
#:
#: THEY ARE NOT DROPPED, AND THE DISTINCTION MATTERS. The ORDER is still TRANSCRIBED
#: rather than chosen, leg ``parity_chain_order`` still measures the residual as real on
#: an engineered signed-zero/subnormal table, and the null carries a PRECONDITION this
#: gate checks on every product row: every near and far parity word must be exactly
#: ``(±1, 0)`` or the unused ``(0, 0)``. A collapse that started diverging on ordinary
#: field values would mean the parity words had stopped being ±1, which is a finding
#: either way — so both are armed, scored on NOT diverging, and their precondition is
#: measured rather than asserted.
NULL_EVERYWHERE: Dict[str, Tuple[str, str, str, int, str]] = {
    "parity_chain_order_reversed": (
        FOLD_CASE,
        "float2 g2_ij_v = c_mul(fp0, v2);\n            "
        "f2[g2_ij_i] = c_mul(fp1, g2_ij_v);",
        "float2 g2_ij_v = c_mul(fp1, v2);\n            "
        "f2[g2_ij_i] = c_mul(fp0, g2_ij_v);", 1,
        "the array path applies the far passes in ASCENDING axis order "
        "(stepping._fill_folded_far_ghosts) and the emitter transcribes that order, "
        "but two exact ±1 multiplies commute bitwise on normal operands"),
    "parity_chain_folded_into_one_word": (
        FOLD_CASE,
        "float2 g2_ij_v = c_mul(fp0, v2);\n            "
        "f2[g2_ij_i] = c_mul(fp1, g2_ij_v);",
        "float2 g2_ij_v = c_mul(c_mul(fp1, fp0), v2);\n            "
        "f2[g2_ij_i] = g2_ij_v;", 1,
        "the real board's far carry folds a composed parity into ONE compile-time "
        "sign; with every word exactly ±1 the collapse is exact here too, and leg "
        "parity_chain_order is where the residual is measured directly"),
}

#: The needle each null pair applies, spelled where it differs from the scored table's
#: (the far-parity null asks about axis 1 rather than axis 0, because a single folded
#: axis has only one far word).
NULL_NEEDLES: Dict[str, Tuple[str, str, int]] = {
    "far_parity_forced_to_plus_one_axis1":
        ("c_mul(fp1,", "c_mul(float2(1.0f, 0.0f),", -1),
}


def byte_neutral_source(spec: Mapping[str, Any]) -> str:
    """The seam replaced by a RELOAD of the words the curl half just stored.

    Not a defect. ``f0[ii] = v0;`` executes two lines above ``float2 o0_src = v0;``, so
    ``f0[ii]`` and ``v0`` hold the same float32 pair, and a float32 stored to a
    ``device float2*`` and reloaded is bit-identical to the register. This edit is
    therefore the fusion's central claim written as a program, and leg
    ``byte_neutral_control`` requires it NOT to diverge — the one armed edit in this
    file scored on being UNCAUGHT.
    """
    source = family.folded_beta_complex_fused_magnetic_pair_source(
        spec["codes"], spec["phased"], spec["walls"], spec["expansion"])
    for target in range(3):
        source = needle(source, f"float2 o{target}_src = v{target};",
                        f"float2 o{target}_src = f{target}[ii];")
    return source


#: Where each binding group starts in the plan's argument tuple. Spelled once, so the
#: host mutations and the kernel signature cannot drift apart. 15 volumes, then six
#: curl coefficient vectors, then six constitutive ones, then the packed struct.
CURL_COEFFICIENT_SLOTS = tuple(range(15, 21))
CONSTITUTIVE_COEFFICIENT_SLOTS = tuple(range(21, 27))
PARAMS_SLOT = 27


def _rebind(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    args = list(plan._args)
    for slot, tensor in zip(slots, tensors):
        args[slot] = tensor
    plan._args = tuple(args)


def swap_curl_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: hand the B curl the INTEGER split-field coefficients.

    ``step_B`` reads HALF-INTEGER positions (``SUB_STEPS['step_B']['suffix'] == '_h'``)
    and ``step_D`` integer ones. This is the OPPOSITE polarity to the D/E fused pair's
    equivalent mutation, and the kernel cannot tell: it is a half-cell error in the
    absorber profile, not a crash, so no shader mutation can reach it.
    """
    _rebind(plan, CURL_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}",
                              getattr(pml, f"{stem}_{axis}"), constant=True)
             for axis in "xyz" for stem in ("kms", "sinv")])


def swap_h_lattice(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: bind the HALF-INTEGER constitutive coefficients to the H half.

    ``update_H`` takes ``kps_a``/``kms_a`` and ``update_E`` takes ``kps_a_h``
    (stepping.py:948 vs :1015). The kernel takes six pointers and never asks which
    lattice they came from.
    """
    _rebind(plan, CONSTITUTIVE_COEFFICIENT_SLOTS,
            [residency.mirror(f"mutation:{stem}_{axis}_h",
                              getattr(pml, f"{stem}_{axis}_h"), constant=True)
             for axis in "xyz" for stem in ("kps", "kms")])


def swap_beta_pair(plan: Any, pml: Any, residency: Residency) -> None:
    """HOST DEFECT: pack the beta pair the other way round.

    THE PACKED STRUCT IS ITSELF A BINDING, and nothing else in this file mutates it. The
    two coefficients ride as ``bp`` and ``bm`` inside ``Params``; a builder that filled
    them in the wrong order produces a kernel whose SOURCE is correct, so every shader
    mutation and the whole transcription leg would still pass.
    """
    plan.params = family._params_tensor(
        plan.shape, plan.dtdx, plan.phase_values, plan.parity_values,
        plan.far_parity_values, tuple(reversed(plan.beta_values)), plan.reflect,
        plan.residency.device)
    _rebind(plan, (PARAMS_SLOT,), (plan.params,))


HOST_MUTATIONS: Dict[str, Callable[[Any, Any, Residency], None]] = {
    "curl_takes_the_integer_lattice": swap_curl_lattice,
    "h_half_takes_the_half_integer_lattice": swap_h_lattice,
    "params_beta_pair_packed_reversed": swap_beta_pair,
}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_policy() -> Dict[str, Any]:
    """The resolved float32 subnormal policy, and ``keep`` REFUSED BY NAME.

    Metal has ONE attainable policy. That is not a gap in this gate: it is a
    measurement the executor's own arm carries (``subnormal.ATTAINABLE``), and a gate
    that quietly ran under whatever the environment happened to hold would be
    certifying bytes under an unknown precondition.
    """
    report = subnormal.mps_policy_report()
    keep = subnormal.mps_policy_report(subnormal.KEEP)
    keep_named = [reason for reason in keep["reasons"]
                  if "cannot honour it" in reason and "Metal flushes" in reason]
    return {
        "passed": bool(report["admitted"] and report["resolved"] == subnormal.FLUSH
                       and not keep["admitted"] and keep_named),
        "report": report,
        "attainable": list(subnormal.ATTAINABLE),
        "keep_admitted": keep["admitted"],
        "keep_named_refusals": keep_named,
        "note": "both float32 subnormal policies are ASKED here; this executor has "
                "one, and the other's refusal is the measurement",
    }


def leg_expansion() -> Dict[str, Any]:
    """The complex-multiply arm is bound by TWO PROBE ARTIFACTS, never defaulted.

    This kernel carries complex products from BOTH parents' call sites — the mirror
    fill's ±1 parity orientation and the beta insert's purely-imaginary coefficient —
    and ONE ``templates.complex_helpers(expansion)`` emission serves the whole kernel.
    So both artifacts are required, each licensed by its own parent's rule, and the two
    verdicts must be the SAME WORD. Five things are required:

    1. both artifacts CONFIGURED and classifying this host to one arm;
    2. a MISSING probe on either side must refuse — never a default arm;
    3. an AMBIGUOUS parity probe must refuse, because a reference matching both arms
       licenses neither;
    4. a WRONG-BACKEND record on either side must refuse;
    5. two records that DISAGREE must refuse — inheriting one verdict for both would be
       exactly the guess the probe mechanism exists to prevent.
    """
    global EXPANSION, PARITY_PROBE, BETA_PROBE
    parity_path = os.environ.get(folded_complex.PROBE_PATH_ENVIRONMENT)
    beta_path = os.environ.get(special_kz.PROBE_PATH_ENVIRONMENT)
    parity = folded_complex.load_expansion_probe()
    beta = special_kz.load_expansion_probe()
    arm = family.expansion_from_probes(parity, beta)
    EXPANSION, PARITY_PROBE, BETA_PROBE = arm, parity, beta

    missing_parity = family.expansion_from_probes(None, beta) is None
    missing_beta = family.expansion_from_probes(parity, None) is None
    ambiguous = {"backend": folded_complex.PROBE_BACKEND,
                 "patterns": {name: complex_fields.AMBIGUOUS_BOTH
                              for name in folded_complex.PARITY_PROBE_PATTERNS}}
    assert folded_complex.PARITY_PROBE_PATTERN in folded_complex.PARITY_PROBE_PATTERNS
    ambiguous_refused = family.expansion_from_probes(ambiguous, beta) is None
    wrong_parity = dict(parity or {}, backend="not-a-backend")
    wrong_beta = dict(beta or {}, backend="not-a-backend")
    backend_refused = (family.expansion_from_probes(wrong_parity, beta) is None
                       and family.expansion_from_probes(parity, wrong_beta) is None)
    # THE DISAGREEMENT. A record whose patterns all name the OTHER arm, so the two
    # verdicts are definite and different rather than one of them absent.
    other = "FMA_V1" if arm != "FMA_V1" else "MUL_ADD_V1"
    disagreeing = {"backend": special_kz.PROBE_BACKEND,
                   "patterns": {name: other
                                for name in special_kz.BETA_PROBE_PATTERNS}}
    disagree_verdict = special_kz.beta_expansion_from_probe(disagreeing)
    disagree_refused = (disagree_verdict != arm
                        and family.expansion_from_probes(parity, disagreeing) is None)
    reasons = family._expansion_reasons(parity, disagreeing)
    disagree_named = [reason for reason in reasons if "disagree" in reason] or (
        [reason for reason in reasons if "beta probe" in reason])

    def digest(path: Optional[str]) -> Optional[str]:
        return (hashlib.sha256(Path(path).read_bytes()).hexdigest()
                if path and Path(path).exists() else None)

    return {
        "passed": bool(arm and parity_path and beta_path and missing_parity
                       and missing_beta and ambiguous_refused and backend_refused
                       and disagree_refused and disagree_named),
        "expansion": arm,
        "parity_probe_artifact": parity_path,
        "parity_probe_sha256": digest(parity_path),
        "parity_probe_backend": (parity or {}).get("backend"),
        "beta_probe_artifact": beta_path,
        "beta_probe_sha256": digest(beta_path),
        "beta_probe_backend": (beta or {}).get("backend"),
        "missing_parity_probe_refused": missing_parity,
        "missing_beta_probe_refused": missing_beta,
        "ambiguous_parity_probe_refused": ambiguous_refused,
        "wrong_backend_probe_refused": backend_refused,
        "disagreeing_probes_refused": disagree_refused,
        "disagreement_named": disagree_named,
    }


def leg_binding_ceiling() -> Dict[str, Any]:
    """42 separate bindings must FAIL; the packed 28 must COMPILE, LAUNCH and read.

    This is what makes "one dispatch for all three components" a measurement rather
    than a preference. The refuted signature is spelled by THIS family (the beta parent
    binds six ``constant float&`` phase words where the plain complex parent binds three
    ``constant float2&``, plus four beta words), and the packed half is this family's
    own 120-byte record — ELEVEN ``float2`` members, four uints, one float and three
    ints. The padding question that record raises is answered by LAUNCHING it and
    reading every field back, not by arithmetic.
    """
    import torch  # noqa: PLC0415

    row: Dict[str, Any] = {
        "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
        "packed_bindings": family.PACKED_BINDINGS,
        "params_itemsize": family.PARAMS_ITEMSIZE,
        "params_record_itemsize": int(family.params_record_dtype().itemsize),
        "ceiling": 31}
    try:
        compile_source(family.refuted_separate_scalar_source())
        row.update(separate_compiled=True, separate_error="", passed=False,
                   note="the separate-scalar signature COMPILED; this family's shape "
                        "rests on a ceiling this host does not have")
        return row
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        row["separate_compiled"] = False
        row["separate_error"] = message.splitlines()[0] if message else ""
        row["separate_refused_for_the_right_reason"] = (
            "out of bounds" in message and "buffer" in message)

    pointers = "\n".join(f"    device float*       b{n:<6}[[buffer({n})]],"
                         for n in range(family.PACKED_BINDINGS - 1))
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params {",
        "    float2 px; float2 py; float2 pz;",
        "    float2 m0; float2 m1; float2 m2;",
        "    float2 d0; float2 d1; float2 d2;",
        "    float2 bp; float2 bm;",
        "    uint nx; uint ny; uint nz; uint n_elem; float dtdx;",
        "    int rx; int ry; int rz;",
        "};", "",
        "kernel void packed_probe(", pointers,
        f"    constant Params&    prm     [[buffer({family.PACKED_BINDINGS - 1})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        "    b0[idx] = float(prm.nx);", "    b1[idx] = float(prm.ny);",
        "    b2[idx] = float(prm.nz);", "    b3[idx] = float(prm.n_elem);",
        "    b4[idx] = prm.dtdx;",
        "    b5[idx] = prm.px.x;", "    b6[idx] = prm.px.y;",
        "    b7[idx] = prm.pz.x;", "    b8[idx] = prm.m0.x;",
        "    b9[idx] = prm.m1.x;", "    b10[idx] = prm.m2.y;",
        "    b11[idx] = prm.d0.x;", "    b12[idx] = prm.d1.x;",
        "    b13[idx] = prm.d2.y;",
        "    b14[idx] = prm.bp.x;", "    b15[idx] = prm.bp.y;",
        "    b16[idx] = prm.bm.x;", "    b17[idx] = prm.bm.y;",
        "    b18[idx] = float(prm.rx);", "    b19[idx] = float(prm.ry);",
        "    b20[idx] = float(prm.rz);",
        f"    b21[idx] = b{family.PACKED_BINDINGS - 2}[idx] + 1.0f;", "}", ""))
    try:
        function = compile_source(source).packed_probe
    except Exception as exc:  # noqa: BLE001
        row.update(packed_compiled=False, packed_error=str(exc).splitlines()[0],
                   passed=False)
        return row
    row["packed_compiled"] = True
    count = 8
    buffers = [torch.zeros(count, dtype=torch.float32, device="mps")
               for _ in range(family.PACKED_BINDINGS - 1)]
    buffers[-1] = torch.full((count,), 7.0, dtype=torch.float32, device="mps")
    shape, dtdx = (3, 4, 5), 0.125
    phases = ((0.25, -0.5), (0.75, 0.125), (-0.375, 0.625))
    parities = ((1.0, 0.0), (-1.0, 0.0), (0.0, 0.5))
    # DISTINCT WORDS FROM THE NEAR TRIPLE AND FROM THE PHASES, deliberately: a struct
    # whose far members aliased the near ones, or whose beta pair sat one slot early,
    # would read back "correctly" while the kernel multiplied by the wrong word.
    far_parities = ((0.5, -0.25), (-0.75, 0.0), (0.0, -0.875))
    beta = ((0.0, 0.15625), (-0.0, -0.15625))
    reflect = (9, -1, 4)
    params = family._params_tensor(shape, dtdx, phases, parities, far_parities,
                                   beta, reflect, "mps")
    function(*buffers, params)
    torch.mps.synchronize()
    read = [buffers[index].cpu().numpy() for index in range(22)]
    expected = (float(shape[0]), float(shape[1]), float(shape[2]),
                float(shape[0] * shape[1] * shape[2]), float(np.float32(dtdx)),
                phases[0][0], phases[0][1], phases[2][0],
                parities[0][0], parities[1][0], parities[2][1],
                far_parities[0][0], far_parities[1][0], far_parities[2][1],
                beta[0][0], beta[0][1], beta[1][0], beta[1][1],
                float(reflect[0]), float(reflect[1]), float(reflect[2]), 8.0)
    fields_ok = all(bool(np.all(read[index] == expected[index]))
                    for index in range(22))
    row.update(packed_launched=True,
               packed_fields_read_back=[float(value[0]) for value in read],
               packed_fields_expected=[float(value) for value in expected],
               packed_fields_correct=fields_ok,
               itemsize_agrees=bool(row["params_itemsize"]
                                    == row["params_record_itemsize"]),
               passed=bool(row["separate_refused_for_the_right_reason"] and fields_ok
                           and row["params_itemsize"]
                           == row["params_record_itemsize"]))
    return row


def leg_transcription(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Every piece must be the CERTIFIED emitters' own text, not a similar one.

    A construction is a hypothesis until something compares the strings, and this
    module's whole claim is that it is the shipped folded complex weld with ONE emitter
    swapped. Six comparisons, in BOTH contraction modes:

    * the lifted curl head must be a VERBATIM PREFIX of
      ``folded_beta.folded_beta_bloch_curl_source(codes, backward=False, phased,
      expansion, has_beta=True)``'s body;
    * the BETA INSERT's own statements must be present in that head — a lift that
      silently produced the ``has_beta=0`` arm would be the shipped weld under a second
      name, and every device leg would still pass because beta = 0 IS that arithmetic;
    * all FIVE recurrence statements per target must appear in the certified body. The
      shipped weld anchors THREE because ``complex_fields`` nests the intermediates;
      the beta template NAMES them (``q``, ``r``), so anchoring three would silently
      drop two;
    * the constitutive statements must reproduce
      ``complex_fields.bloch_constitutive_source('H')``;
    * both fill transcriptions must be identical to
      ``folded_complex.folded_mirror_fill_complex_source``'s own lines.
    """
    rows: Dict[str, Any] = {"modes": {}}
    ok = True
    codes, phased = spec["codes"], spec["phased"]
    expansion = spec["expansion"]
    folded = [axis for axis, code in enumerate(codes) if int(code) in MIRROR_CODES]
    beta_statements = [line.strip() for line in
                       special_kz._COMPLEX_BETA_INSERT.splitlines() if line.strip()
                       and not line.strip().startswith("//")]
    for mode in shaders.CONTRACT_MODES:
        head = family.certified_curl_head(codes, phased, expansion, mode)
        certified = folded_beta.folded_beta_bloch_curl_source(
            codes, family.BACKWARD, phased, expansion, family.HAS_BETA, mode)
        body = family._body_of(certified, "folded beta complex curl")
        head_ok = body.startswith(head) and bool(head.strip())
        source = family.folded_beta_complex_fused_magnetic_pair_source(
            codes, phased, spec["walls"], expansion, mode)
        beta_ok = bool(beta_statements) and all(
            statement in head and statement in source
            for statement in beta_statements)
        statements = family.certified_curl_statements(codes, phased, expansion, mode)
        recurrence_ok = all(len(block) == 5 and all(line in body for line in block)
                            for block in statements["recurrence"])
        transcription = family.constitutive_transcription(expansion, mode)
        fills = {axis: family.near_fill_transcription(axis, expansion, mode)
                 for axis in folded}
        fill_ok = bool(fills) and all(r["identical"] for r in fills.values())
        far_fills = {axis: family.far_fill_transcription(axis, expansion, mode)
                     for axis in range(3)}
        far_ok = all(r["identical"] for r in far_fills.values())
        rows["modes"][mode] = {
            "curl_head_is_a_verbatim_prefix": head_ok,
            "curl_head_lines": len(head.splitlines()),
            "beta_insert_statements": beta_statements,
            "beta_insert_present": beta_ok,
            "recurrence_statements_are_lifted": recurrence_ok,
            "recurrence_statements_per_target":
                [len(block) for block in statements["recurrence"]],
            "constitutive_identical": transcription["identical"],
            "near_fill_identical": fill_ok,
            "far_fill_identical": far_ok,
        }
        ok = (ok and head_ok and beta_ok and recurrence_ok
              and transcription["identical"] and fill_ok and far_ok)
    rows["folded_axes"] = folded
    rows["passed"] = bool(ok)
    return rows


def leg_parity_chain_order() -> Dict[str, Any]:
    """The composed parity is a CHAIN, its ORDER is normative, and both are measured.

    THE ONE PLACE THIS FAMILY DIVERGES FROM THE REAL BOARD'S FAR CARRY.
    ``folded_fused_magnetic_pair.carried_destinations`` folds a composed parity into ONE
    compile-time sign and states the result is independent of the order the axes are
    applied in. Under COMPLEX storage the parity is a ``float2`` and the application is
    a full complex multiply, so both collapses have to be re-established or refused.
    This leg refuses them over an EXHAUSTIVE signed-zero/subnormal table rather than
    random data, which is exactly why the gaussian class seeds signed zeros:

    1. the array path's ``phase * complex64_array`` must be the FULL complex product and
       NOT a plane-wise real scaling — everything else rests on which one it is;
    2. re-ordering a two-parity chain must MOVE WORDS;
    3. folding a chain into a single word must MOVE WORDS, at every sign combination
       and for three parities as well as two.

    The two source mutations ``parity_chain_reversed`` and
    ``parity_chain_folded_into_one_word`` arm exactly those two collapses on the kernel;
    this leg is what says they are arming something real.
    """
    table = np.array(
        [complex(float(np.float32(re)), float(np.float32(im)))
         for re in (0.0, -0.0, 1.0, -1.0, 1e-42, -1e-42, 3.5, -2.25)
         for im in (0.0, -0.0, 1.0, -1.0, 1e-42, -1e-42, 3.5, -2.25)],
        dtype=np.complex64)
    total = int(words(table).size)

    reference: List[Dict[str, Any]] = []
    for phase in (1, -1):
        got = phase * table
        full = np.empty_like(table)
        coefficient = np.complex64(phase)
        full.real = (np.float32(coefficient.real) * table.real
                     - np.float32(coefficient.imag) * table.imag)
        full.imag = (np.float32(coefficient.real) * table.imag
                     + np.float32(coefficient.imag) * table.real)
        scaled = np.empty_like(table)
        scaled.real = np.float32(phase) * table.real
        scaled.imag = np.float32(phase) * table.imag
        reference.append({"phase": phase,
                          "vs_full_complex_product": differing(got, full),
                          "vs_real_scaling": differing(got, scaled)})
    is_full = all(row["vs_full_complex_product"] == 0 for row in reference)
    not_scaling = all(row["vs_real_scaling"] > 0 for row in reference)

    pairs: List[Dict[str, Any]] = []
    for first, second in itertools.product((1, -1), repeat=2):
        sequential = second * (first * table)
        pairs.append({
            "first": first, "second": second,
            "chain_vs_reversed_chain": differing(sequential, first * (second * table)),
            "chain_vs_folded_product": differing(sequential,
                                                 (first * second) * table)})
    triples: List[Dict[str, Any]] = []
    for first, second, third in itertools.product((1, -1), repeat=3):
        sequential = third * (second * (first * table))
        triples.append({
            "parities": [first, second, third],
            "chain_vs_folded_product": differing(
                sequential, (first * second * third) * table)})
    order_observable = any(row["chain_vs_reversed_chain"] for row in pairs)
    folding_observable = (all(row["chain_vs_folded_product"] for row in pairs)
                          and all(row["chain_vs_folded_product"] for row in triples))
    return {
        "passed": bool(is_full and not_scaling and order_observable
                       and folding_observable),
        "table_words": total,
        "parity_is_the_full_complex_product": is_full,
        "parity_is_not_a_real_scaling": not_scaling,
        "reference": reference,
        "pairs": pairs,
        "triples": triples,
        "order_observable": order_observable,
        "folding_observable": folding_observable,
    }


class _Unindexed:
    """A magnetic source that declares its field type and NOTHING ELSE.

    The smallest thing that is still in the B seam and still unrepairable: it names no
    cell, so ``deposit_repair`` has nothing to save before the launch or restore after
    it. Kept as a stub rather than built from ``sources`` because every real source
    this engine constructs publishes an index — the case exists precisely to hold the
    predicate honest about one it cannot get from the shipped constructors.
    """

    field_type = "B"


def _volume_source(fields: Any, component: str) -> Any:
    """A REAL engine source on ``component``, so ``field_type`` is the engine's.

    A stub with a hand-set ``field_type`` would let this leg pass while the engine
    classified the same component the other way; ``sources.VolumeSource`` resolves the
    slot itself through ``_field_type_for``.
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                        envelope=ContinuousEnvelope(frequency=1.0))


def leg_refusal() -> Dict[str, Any]:
    """The source seam, BOTH sibling scopes, real storage, and 3-D beta.

    THE MOST IMPORTANT LEG IN THIS FILE, because this product exists in the gap between
    two shipped welds and a clause nothing tests is a claim. Seven questions:

    1. an UNDECLARED source list must be refused (ignorance is not an empty set);
    2. a real MAGNETIC ``VolumeSource`` must now be ADMITTED and its plan BUILT. That
       is the 2026-08-30 flip: this family gained a ``launch.FUSED_PAIR_ARMS`` row, so
       ``_install_fused_pair`` brackets its launch with
       ``LeadingRepairPlan``/``TrailingRepairPlan`` and the deposit is saved and
       restored across it. Before the row existed nothing bracketed anything here and
       this clause required the opposite;
    2b. AND THE REFUSAL THAT SURVIVES IT: a magnetic source that publishes no deposit
       index must still be refused BY NAME. Without this half, clause 2 alone would be
       satisfied by a predicate that admitted every in-seam deposit including the ones
       no closure can restore, which is the exact over-covering the flag exists to
       bound;
    3. a real ELECTRIC ``VolumeSource`` must NOT be refused — it is injected in the D/E
       half, outside this seam;
    4. an UNFOLDED beta grid must be refused. That configuration is
       ``beta_fused_magnetic_pair``'s, and admitting it would put two welds on one cell;
    5. a folded grid with ``beta = 0`` must be refused. That is
       ``folded_complex_fused_magnetic_pair``'s shipped weld;
    6. a REAL-STORAGE folded beta grid must be refused;
    7. a 3-D beta grid must be refused BY THE ENGINE, naming MEEP's own abort — which
       is why every fixture in this file is two-dimensional.
    """
    fields, pml = build(dict(CASES[0][1]), seed_for("refusal"))
    residency = Residency()

    def coverage(f: Any, p: Any, sources: Any) -> Any:
        return family.metal_folded_beta_complex_fused_magnetic_pair_coverage(
            f, p, sources, Residency(), PARITY_PROBE, BETA_PROBE)

    undeclared = coverage(fields, pml, None)
    undeclared_named = [r for r in undeclared.reasons if "was not declared" in r]

    # THE ADMISSION, since 2026-08-30, and the refusal that survives it. Until this
    # family gained its `launch.FUSED_PAIR_ARMS` row nothing could bracket its launch,
    # so `CARRIES_DEPOSIT_REPAIR` was False and the clause below barred EVERY in-seam
    # magnetic source by name. The row landed and the flag went True, so the question
    # this half asks changed with it: not "is a magnetic source barred" but "is the
    # deposit one the repair can carry". BOTH DIRECTIONS ARE MEASURED, because a
    # predicate that answered one of them wrongly would pass a leg that asked only the
    # other -- admitting what nothing can restore, or refusing everything and leaving
    # the flag inert.
    magnetic = _volume_source(fields, "Hy")
    magnetic_coverage = coverage(fields, pml, (magnetic,))
    magnetic_plan = family.plan_metal_folded_beta_complex_fused_magnetic_pair(
        fields, pml, sources=(magnetic,), residency=residency,
        probe=PARITY_PROBE, beta_probe=BETA_PROBE)
    # THE REFUSAL THAT SURVIVES. `_Unindexed` declares a field type and nothing else,
    # so it names no cell for the repair to save; a closure cannot carry what a source
    # will not name, and `deposit_repair.seam_source_reasons` refuses it BY NAME rather
    # than admitting it on the strength of the flag.
    unrepairable_coverage = coverage(fields, pml, (_Unindexed(),))
    unrepairable_named = [r for r in unrepairable_coverage.reasons
                          if "does not publish the index it writes" in r]
    unrepairable_plan = family.plan_metal_folded_beta_complex_fused_magnetic_pair(
        fields, pml, sources=(_Unindexed(),), residency=residency,
        probe=PARITY_PROBE, beta_probe=BETA_PROBE)

    electric = _volume_source(fields, "Ez")
    electric_coverage = coverage(fields, pml, (electric,))

    # The cell's z EXTENT must be zero at dimensions=2: an invariant axis is infinite
    # and uniform, and Grid._resolve_dimensions refuses a length there by name.
    unfolded_fields, unfolded_pml = matrix.cart(
        cell=(2.0, 2.1, 0.0), pml=((2, 2), (2, 2), (0, 0)), complex_storage=True,
        beta=0.3, dimensions=2)
    unfolded = coverage(unfolded_fields, unfolded_pml, ())
    unfolded_named = [r for r in unfolded.reasons if "fold" in r.lower()]

    nobeta_fields, nobeta_pml = matrix.folded(complex_storage=True)
    nobeta = coverage(nobeta_fields, nobeta_pml, ())
    nobeta_named = [r for r in nobeta.reasons if "beta" in r.lower()]

    real_fields, real_pml = matrix.folded(beta=0.3)
    real = coverage(real_fields, real_pml, ())
    real_named = [r for r in real.reasons
                  if "complex" in r.lower() or "storage" in r.lower()]

    three_d_error = ""
    try:
        matrix.folded(complex_storage=True, beta=0.3, depth=1.2)
        three_d_refused = False
    except Exception as exc:  # noqa: BLE001 - the refusal IS the measurement
        three_d_refused = True
        three_d_error = str(exc).splitlines()[0]
    three_d_named = ("dimensions other than 2" in three_d_error
                     or "dimensions=3" in three_d_error)

    return {
        "passed": bool(not undeclared.covered and undeclared_named
                       and magnetic_coverage.covered and magnetic_plan is not None
                       and not unrepairable_coverage.covered and unrepairable_named
                       and unrepairable_plan is None
                       and electric_coverage.covered
                       and not unfolded.covered and unfolded_named
                       and not nobeta.covered and nobeta_named
                       and not real.covered and real_named
                       and three_d_refused and three_d_named),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": undeclared_named,
        "magnetic_source_field_type": str(magnetic.field_type),
        "magnetic_source_admitted": magnetic_coverage.covered,
        "magnetic_refusals_if_any": list(magnetic_coverage.reasons),
        "magnetic_plan_built": magnetic_plan is not None,
        "unrepairable_source_refused": not unrepairable_coverage.covered,
        "unrepairable_named": unrepairable_named,
        "unrepairable_plan_is_none": unrepairable_plan is None,
        "electric_source_field_type": str(electric.field_type),
        "electric_source_admitted": electric_coverage.covered,
        "electric_reasons": list(electric_coverage.reasons),
        "unfolded_beta_refused": not unfolded.covered,
        "unfolded_named": unfolded_named[:3],
        "folded_zero_beta_refused": not nobeta.covered,
        "zero_beta_named": nobeta_named[:3],
        "real_storage_refused": not real.covered,
        "real_storage_named": real_named[:3],
        "three_d_beta_refused_by_the_engine": three_d_refused,
        "three_d_error": three_d_error,
        "three_d_named_meeps_abort": three_d_named,
    }


def leg_value_classes(name: str, keywords: Mapping[str, Any],
                      pml_faces: Optional[Any], steps: int) -> Dict[str, Any]:
    """Three classes byte-identical; the band seed FIRES the census.

    Every product row reports ``reference_subnormals: 0`` under policy ``flush``. Read
    alone that is a SILENCE: it cannot distinguish "the flush policy was exercised and
    changed nothing" from "the band was never entered". So a SUBNORMAL_BAND seed is run
    and the census is required to be NONZERO — the detector shown to work — while the
    band itself is a REFUSAL rather than a comparison, for the reason recorded in
    ``band_refusal``: MPS flushes float32 subnormals natively with no lever and NumPy
    keeps, so a subnormal in the reference state IS a divergence and no kernel can be
    right about it.
    """
    rows: List[Dict[str, Any]] = []
    for value_class in VALUE_CLASSES:
        result = run_case(keywords, seed_for("class", name, value_class), steps,
                          pml_faces, value_class=value_class)
        rows.append({"value_class": value_class,
                     "bit_identical": result.get("bit_identical"),
                     "differing_words": result.get("differing_words"),
                     "launches": result.get("launches"),
                     "seeded_signed_zero_count":
                         result.get("seeded_signed_zero_count"),
                     "passed": bool(result.get("passed"))})
        log(f"    class {value_class} identical={result.get('bit_identical')} "
            f"diff={result.get('differing_words')}")
    band = run_case(keywords, seed_for("class", name, SUBNORMAL_BAND), 2, pml_faces,
                    value_class=SUBNORMAL_BAND, expect_subnormal_free=False)
    fired = int(band.get("reference_subnormals") or 0)
    log(f"    class {SUBNORMAL_BAND} census={fired} (refused, never compared)")
    # The gaussian class must actually carry signed zeros, or the chain mutations and
    # the literal zero cross terms are armed at nothing.
    zeros_ok = all(row["seeded_signed_zero_count"] for row in rows
                   if row["value_class"] == "gaussian")
    return {
        "passed": bool(all(row["passed"] for row in rows) and fired > 0 and zeros_ok),
        "rows": rows,
        "signed_zeros_were_seeded": zeros_ok,
        "band_census_words": fired,
        "band_census_fired": fired > 0,
        "band_was_byte_compared": False,
        "band_refusal":
            "MPS flushes float32 subnormals natively and exposes no lever; the array "
            "path keeps them. A subnormal in the reference state is therefore a "
            "divergence no kernel can be right about, so byte identity is claimed "
            "under a CHECKED subnormal-free precondition and this row is the check "
            "being shown to fire.",
    }


def leg_moved_coefficient(deep: Tuple[str, Dict[str, Any], Optional[Any]],
                          shallow: Tuple[str, Dict[str, Any], Optional[Any]],
                          steps: int) -> Dict[str, Any]:
    """PREDICTED observability of the moved NEAR coefficient index vs measured catch.

    THE LINE NOTHING CAN SEE, turned into a measurement. The imaged near ghost reads its
    coefficient pair at stored index 0 — the DESTINATION's own entry — rather than
    reusing the source thread's at index 2. On a FOLDED axis the mirror plane carries no
    absorber (the layer is on the high face only), so at the shared matrix's 2-cell
    layer ``kps[0] == kps[2]`` to the bit and the mutation that reuses the wrong entry
    is UNOBSERVABLE.

    This leg censuses the coefficients on both fixtures, PREDICTS observability from
    that census alone, runs the same needle on both, and requires ``caught ==
    observable`` on each. A null discovered rather than predicted would be a defect in
    this gate; a null that disappeared would mean the fixture drifted.
    """
    rows: List[Dict[str, Any]] = []
    ok = True
    axis_name = "xyz"[COEFFICIENT_AXIS]
    for label, (name, keywords, faces) in (("deep", deep), ("shallow", shallow)):
        fields, pml = build(keywords, seed_for("coefficient", name), faces)
        kps = np.asarray(getattr(pml, f"kps_{axis_name}")).ravel()
        kms = np.asarray(getattr(pml, f"kms_{axis_name}")).ravel()
        observable = bool(kps.size > 2 and (kps[0] != kps[2] or kms[0] != kms[2]))
        spec = _specialisation(fields, pml)
        base = family.folded_beta_complex_fused_magnetic_pair_source(
            spec["codes"], spec["phased"], spec["walls"], spec["expansion"])
        old, new, count = ("float g1_n_kp = kp1[0], g1_n_km = km1[0];",
                           "float g1_n_kp = kp_1, g1_n_km = km_1;", 1)
        mutated = needle(base, old, new, count)
        functions = {shaders.CONTRACT_OFF: compile_source(
            mutated).folded_beta_complex_fused_magnetic_pair_step}
        result = run_case(keywords, seed_for("coefficient", name), steps, faces,
                          functions=functions)
        caught = not result.get("bit_identical", False)
        agrees = caught == observable
        ok = ok and agrees and bool(result.get("launches"))
        rows.append({"fixture": label, "case": name,
                     "kps_stored_0": float(kps[0]),
                     "kps_stored_2": float(kps[2]) if kps.size > 2 else None,
                     "kms_stored_0": float(kms[0]),
                     "kms_stored_2": float(kms[2]) if kms.size > 2 else None,
                     "stored_cells_on_the_axis": int(kps.size),
                     "predicted_observable": observable,
                     "caught": caught,
                     "differing_words": result.get("differing_words"),
                     "launches": result.get("launches"),
                     "agrees": agrees})
        log(f"    moved_coefficient {label} observable={observable} caught={caught}")
    return {"passed": bool(ok and len(rows) == 2
                           and rows[0]["predicted_observable"]
                           and not rows[1]["predicted_observable"]),
            "axis": COEFFICIENT_AXIS, "rows": rows}


def leg_arm_table() -> Dict[str, Any]:
    """Registered UNWIRED, marked a weld, and unreachable from ``arms_for``.

    ``plan_step`` assigns at most one arm per slot and this product spans five, so there
    is no slot it could claim without a composition rule nothing has measured. It would
    additionally contend on ``step_B`` with ``folded_beta``'s WIRED complex curl arm,
    which admits exactly the same configurations, and ``_select_slot`` would then leave
    the slot UNSELECTED — the certified curl would come off the device as well. So the
    registration must be enumerable (for the disjointness sweeps) and unselectable, and
    both halves are measured here rather than read off a docstring.
    """
    rows = [arm for arm in metal_arms.registered() if arm.family == family.FAMILY]
    one = len(rows) == 1
    arm = rows[0] if one else None
    fields, pml = build(dict(CASES[0][1]), seed_for("arm_table"))
    residency = Residency()

    class _Context:
        pass

    context = _Context()
    context.fields, context.pml = fields, pml
    context.sources, context.residency = (), residency
    context.contract_variants = (shaders.CONTRACT_OFF,)
    # ``arms_for`` hands back the shared ``_Arm``, which carries the LABEL rather than
    # the family name — so the question is asked in the vocabulary the composer uses.
    label = getattr(arm, "label", None)
    offered = [getattr(candidate, "label", repr(candidate)) for candidate in
               metal_arms.arms_for(family.SLOT, context)]
    return {
        "passed": bool(one and arm is not None and not arm.wired and arm.is_weld
                       and tuple(arm.replaces) == family.REPLACES
                       and arm.slot == family.SLOT
                       and label is not None and label not in offered),
        "arm_label": label,
        "rows_for_this_family": len(rows),
        "slot": getattr(arm, "slot", None),
        "wired": getattr(arm, "wired", None),
        "is_weld": getattr(arm, "is_weld", None),
        "replaces": list(getattr(arm, "replaces", ()) or ()),
        "replaces_expected": list(family.REPLACES),
        "arms_for_step_B_offers": offered,
        "registered_total": len(metal_arms.registered()),
    }


def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['leg']}] {row['label']}: "
        f"passed={row.get('passed')} diff={row.get('differing_words', '-')} "
        f"launches={row.get('launches', '-')}")


def compile_edit(source: str) -> Dict[str, Any]:
    return {shaders.CONTRACT_OFF:
            compile_source(source).folded_beta_complex_fused_magnetic_pair_step}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")

    matrix.prepare_environment()
    cases = {name: (keywords, faces) for name, keywords, faces in CASES}

    # THE EXPANSION ARM FIRST. Every later leg emits a source, and a source emitted
    # before the arm is bound would carry a guessed complex multiply.
    expansion_row = leg_expansion()
    if not expansion_row["passed"]:
        log("the expansion arm is unbound; the gate cannot certify")

    # THE MUTATION CASES' SPECIALISATIONS, CHECKED BEFORE ANY NEEDLE IS APPLIED. This
    # is the dead-branch guard: a needle whose line sits under a feature the case does
    # not carry would report UNCAUGHT while measuring nothing.
    specs: Dict[str, Dict[str, Any]] = {}
    for name in (FOLD_CASE, WALL_CASE, PHASE_CASE, REFLECT_CASE):
        keywords, faces = cases[name]
        fields, pml = build(keywords, seed_for("arm", name), faces)
        specs[name] = _specialisation(fields, pml)

    fold = specs[FOLD_CASE]
    folded_axes = [axis for axis, code in enumerate(fold["codes"])
                   if int(code) == CODE_MIRROR_PERIODIC]
    assert len(folded_axes) >= 2, (
        f"{FOLD_CASE} folds fewer than two PERIODIC axes ({fold['codes']}); the far "
        f"carry, the parity chain and the near/far composite would all be absent")
    assert set(fold["phases"][axis] for axis in folded_axes) == {1, -1}, (
        f"{FOLD_CASE} does not carry BOTH parities: {fold['phases']}")
    near_word = fold["near"][NEAR_PARITY_AXIS]
    assert (float(near_word[0]), float(near_word[1])) != (1.0, 0.0), (
        f"{FOLD_CASE}'s NEAR parity word on axis {NEAR_PARITY_AXIS} is the identity "
        f"{near_word}; near_parity_forced_to_plus_one would be a no-op")
    far_word = fold["far"][FAR_PARITY_AXIS]
    assert (float(far_word[0]), float(far_word[1])) != (1.0, 0.0), (
        f"{FOLD_CASE}'s FAR parity word on axis {FAR_PARITY_AXIS} is the identity "
        f"{far_word}; far_parity_forced_to_plus_one would be a no-op")
    assert any(len(axes) == 2 for axes in fold["far_carried"]), (
        f"{FOLD_CASE} gives no component two far axes; the FAR/FAR composite and its "
        f"two-parity chain would be absent and both chain needles would fail to apply")
    assert not any(fold["phased"]), (
        f"{FOLD_CASE} carries a phased axis; the phase mutations belong on "
        f"{PHASE_CASE} and this assertion is what keeps the split honest")

    wall = specs[WALL_CASE]
    assert any(wall["walls"]), (
        f"{WALL_CASE} has no walled axis; both wall mutations would be vacuous")
    assert wall["walls"][0] and not wall["walls"][1], (
        f"{WALL_CASE} walls {wall['walls']}; the d-side-table needle substitutes at_y "
        f"for at_x and needs x walled and y not")

    phase = specs[PHASE_CASE]
    assert any(phase["phased"]), (
        f"{PHASE_CASE} phases no axis: {phase['phased']}; the rotation mutations would "
        f"rewrite absent text")
    assert any(int(code) in MIRROR_CODES for code in phase["codes"]), (
        f"{PHASE_CASE} is not folded; it would not be this family at all")

    reflect_spec = specs[REFLECT_CASE]
    row_y = reflect_spec["reflect"][1]
    fixed = int(reflect_spec["shape"][1]) - 2
    assert row_y is not None and int(row_y) != fixed, (
        f"{REFLECT_CASE} has reflect row {row_y} on axis 1 of shape "
        f"{reflect_spec['shape']}, which IS the fixed n - 2 = {fixed} the "
        f"reflect_row_is_the_fixed_n_minus_two needle substitutes; the mutation would "
        f"rewrite the plane to itself")

    bases = {name: family.folded_beta_complex_fused_magnetic_pair_source(
        spec["codes"], spec["phased"], spec["walls"], spec["expansion"])
        for name, spec in specs.items()}
    mutants: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for label, (case, old, new, count, _why) in SHADER_EDITS.items():
        mutants[label] = (case, compile_edit(
            needle(bases[case], old, new, count)))
    neutral = compile_edit(byte_neutral_source(fold))

    controls = (FOLD_CASE, WALL_CASE, "y_metallic_even_2d")
    movement_rows = (FOLD_CASE, WALL_CASE, "y_metallic_even_2d")
    total = (1 + 1 + 1 + 1 + 1 + len(CASES) + len(movement_rows) + len(controls)
             + 1 + 1 + 1 + 1 + len(mutants) + len(HOST_MUTATIONS)
             + len(NULL_PAIRS) + len(NULL_EVERYWHERE) + 1 + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        def record(leg: str, label: str, payload: Dict[str, Any]) -> None:
            nonlocal index
            index += 1
            row = {"index": index, "total": total, "leg": leg, "label": label}
            row.update(payload)
            rows.append(row)
            emit(handle, row)

        record("policy", "flush_resolved_and_keep_refused_by_name", leg_policy())
        record("expansion", "both_probes_bind_one_arm", expansion_row)
        record("binding_ceiling", "twenty_seven_pointers_plus_one_packed_struct",
               leg_binding_ceiling())
        record("transcription", "every_piece_is_the_certified_text",
               leg_transcription(fold))
        record("parity_chain_order",
               "the_chain_is_ordered_and_neither_collapse_is_free",
               leg_parity_chain_order())

        for name, keywords, faces in CASES:
            record("product", name,
                   {"steps": args.steps,
                    **run_case(keywords, seed_for("product", name), args.steps,
                               faces)})

        for name in movement_rows:
            keywords, faces = cases[name]
            record("movement_role", name,
                   leg_movement_role(name, keywords, seed_for("movement", name),
                                     faces))

        for name in controls:
            keywords, faces = cases[name]
            record("separate_control", name,
                   {"steps": args.steps,
                    **run_separate_control(keywords, seed_for("separate", name),
                                           args.steps, faces)})

        record("refusal", "the_source_seam_both_sibling_scopes_and_three_d_beta",
               leg_refusal())

        keywords, faces = cases[FOLD_CASE]
        record("value_classes", f"{FOLD_CASE}:three_classes_and_the_band",
               leg_value_classes(FOLD_CASE, keywords, faces, args.steps))

        deep = (FOLD_CASE, *cases[FOLD_CASE])
        shallow = (SHALLOW_CASE, *cases[SHALLOW_CASE])
        record("moved_coefficient", "predicted_observability_vs_measured_catch",
               leg_moved_coefficient(deep, shallow, args.steps))

        # THE ONE ARMED EDIT REQUIRED TO BE UNCAUGHT. Scored on NOT diverging.
        keywords, faces = cases[FOLD_CASE]
        result = run_case(keywords, seed_for("neutral"), args.steps, faces,
                          functions=neutral)
        record("byte_neutral_control",
               "register_replaced_by_a_reload_of_the_same_word",
               {"diverged": not result.get("bit_identical", False),
                "passed": bool(result.get("passed")),
                "differing_words": result.get("differing_words"),
                "launches": result.get("launches")})

        for label, (case, functions) in mutants.items():
            keywords, faces = cases[case]
            result = run_case(keywords, seed_for("mutation", label), args.steps,
                              faces, functions=functions)
            caught = not result.get("bit_identical", False)
            record("mutation", label,
                   {"case": case, "caught": caught,
                    "why": SHADER_EDITS[label][4],
                    "passed": bool(caught and result.get("launches")),
                    "first_divergence": result.get("first_divergence"),
                    "differing_words": result.get("differing_words"),
                    "differing_arrays": result.get("differing_arrays"),
                    "launches": result.get("launches")})

        for label, patch in HOST_MUTATIONS.items():
            keywords, faces = cases[FOLD_CASE]
            result = run_case(keywords, seed_for("host", label), args.steps, faces,
                              patch=patch)
            caught = not result.get("bit_identical", False)
            record("mutation", label,
                   {"case": FOLD_CASE, "host_defect": True, "caught": caught,
                    "passed": bool(caught and result.get("launches")),
                    "first_divergence": result.get("first_divergence"),
                    "differing_words": result.get("differing_words"),
                    "differing_arrays": result.get("differing_arrays"),
                    "launches": result.get("launches")})

        # THE PREDICTED NULLS. Each must be NULL on its own fixture and CAUGHT on the
        # complementary one. A null scored as merely "uncaught" is a defect; a null
        # with no complementary catch is an untested line wearing an excuse.
        for label, (null_case, caught_case, needle_name, why) in NULL_PAIRS.items():
            if needle_name in SHADER_EDITS:
                _case, old, new, count, _w = SHADER_EDITS[needle_name]
            else:
                old, new, count = NULL_NEEDLES[needle_name]
            pair_rows = []
            for role, case in (("null", null_case), ("caught", caught_case)):
                keywords, faces = cases[case]
                fields, pml = build(keywords, seed_for("null", label, case), faces)
                spec = _specialisation(fields, pml)
                base = family.folded_beta_complex_fused_magnetic_pair_source(
                    spec["codes"], spec["phased"], spec["walls"], spec["expansion"])
                functions = compile_edit(needle(base, old, new, count))
                result = run_case(keywords, seed_for("null", label, case),
                                  args.steps, faces, functions=functions)
                pair_rows.append({
                    "role": role, "case": case,
                    "caught": not result.get("bit_identical", False),
                    "differing_words": result.get("differing_words"),
                    "launches": result.get("launches"),
                    "near_parity": [list(v) for v in spec["near"]],
                    "far_parity": [list(v) for v in spec["far"]],
                    "reflect": list(spec["reflect"]),
                    "shape": list(spec["shape"])})
            null_ok = (not pair_rows[0]["caught"]) and pair_rows[1]["caught"]
            record("null_pairs", label,
                   {"passed": bool(null_ok
                                   and all(r["launches"] for r in pair_rows)),
                    "needle": needle_name, "why_null": why, "rows": pair_rows})

        # THE NULLS THAT ARE NULL EVERYWHERE. Scored on NOT diverging, with the
        # PRECONDITION that makes them null measured off the product rows that already
        # ran: every near and far parity word on every case must be exactly (±1, 0) or
        # the unused (0, 0). If a case ever carried a genuinely complex parity, ±1 would
        # no longer compose exactly and these two would have to be caught instead.
        product_rows = [row for row in rows if row["leg"] == "product"]
        offending: List[Dict[str, Any]] = []
        for row in product_rows:
            for kind in ("parity_values", "far_parity_values"):
                for axis, value in enumerate(row.get(kind) or ()):
                    pair = (float(value[0]), float(value[1]))
                    if pair not in ((1.0, 0.0), (-1.0, 0.0), (0.0, 0.0),
                                    (-0.0, 0.0), (1.0, -0.0), (-1.0, -0.0),
                                    (0.0, -0.0), (-0.0, -0.0)):
                        offending.append({"case": row["label"], "kind": kind,
                                          "axis": axis, "word": list(pair)})
        parities_are_pm_one = not offending
        for label, (case, old, new, count, why) in NULL_EVERYWHERE.items():
            keywords, faces = cases[case]
            functions = compile_edit(needle(bases[case], old, new, count))
            result = run_case(keywords, seed_for("null_everywhere", label),
                              args.steps, faces, functions=functions)
            diverged = not result.get("bit_identical", False)
            record("null_everywhere", label,
                   {"case": case, "diverged": diverged, "why_null": why,
                    "scored_on": "NOT diverging",
                    "armed_as_caught_required_first": True,
                    "precondition_every_parity_word_is_pm_one": parities_are_pm_one,
                    "parity_words_that_are_not_pm_one": offending,
                    "cases_checked": len(product_rows),
                    "passed": bool(not diverged and result.get("launches")
                                   and parities_are_pm_one),
                    "differing_words": result.get("differing_words"),
                    "launches": result.get("launches")})

        record("arm_table", "registered_unwired_and_unselectable", leg_arm_table())

        # THE DISARM CHECK. The identical harness, the identical case, the SHIPPED
        # bytes and no host patch. A nonzero here would mean every "caught" above is a
        # harness that diverges on its own.
        keywords, faces = cases[FOLD_CASE]
        result = run_case(keywords, seed_for("disarm"), args.steps, faces)
        record("disarm", "shipped_bytes_on_the_fold_mutation_case",
               {"differing_words": result.get("differing_words"),
                "launches": result.get("launches"),
                "arrays_that_never_moved": result.get("arrays_that_never_moved"),
                "passed": bool(result.get("passed"))})

    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "counts": {"product": len(CASES), "movement_role": len(movement_rows),
                   "separate_controls": len(controls),
                   "shader_mutations": len(mutants),
                   "host_mutations": len(HOST_MUTATIONS),
                   "predicted_null_pairs": len(NULL_PAIRS),
                   "nulls_null_everywhere": len(NULL_EVERYWHERE),
                   "byte_neutral_controls": 1,
                   "value_classes": len(VALUE_CLASSES)},
        "mutation_cases": {
            "fold": FOLD_CASE, "wall": WALL_CASE, "phase": PHASE_CASE,
            "reflect": REFLECT_CASE},
        "mutation_specialisations": {
            name: {"codes": list(spec["codes"]), "phased": list(spec["phased"]),
                   "walls": list(spec["walls"]), "phases": list(spec["phases"]),
                   "near": [list(v) for v in spec["near"]],
                   "far": [list(v) for v in spec["far"]],
                   "reflect": list(spec["reflect"]),
                   "shape": list(spec["shape"])}
            for name, spec in specs.items()},
        "expansion": EXPANSION,
        "dimensionality_note":
            "every fixture is 2-D because grid._resolve_beta (grid.py:690) refuses a "
            "nonzero beta on a 3-D grid, naming MEEP's own abort "
            "(fields.cpp:546-547). Leg refusal ASKS for a 3-D beta grid and requires "
            "that refusal rather than leaving the dimension untested.",
        "corpus_rows": {
            "cell": "B_to_H (folded beta complex, folded beta complex)",
            "driven_by": ["tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2",
                          "tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7",
                          "tests:TestSpecialKz.test_eigsrc_kz_0_complex"],
            "unreachable": "tests:TestSpecialKz.test_eigsrc_kz_0_complex declares a "
                           "MAGNETIC source, which driver.step deposits BETWEEN "
                           "step_B and update_H (driver.py:3292-3293); no launch can "
                           "straddle a deposit. Leg refusal measures that clause.",
            "note": "THIS GATE ASSERTS NO CELL MOVE. The board is a separate "
                    "measurement (build_fusion_matrix.py); what is certified here is "
                    "the predicate, the binding arithmetic and the bytes.",
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    failed = [row["label"] for row in rows if not row["passed"]]
    log(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
        f"{len(failed)} failing rows {failed[:8]}; artifact {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
