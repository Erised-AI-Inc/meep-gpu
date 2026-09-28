"""The Dcyl COMPLEX hand-CUDA fused pair: ``step_B`` -> ``zero_metal_B`` -> ``update_H``.

THE BOARD'S RANK-1 UNSERVED CELL, and the largest one on this seam that no product
occupies: ``B_to_H (cuda_cyl_complex/cylindrical complex, cuda_complex/complex)``,
16 seam-instances of demand and 16 of them clearing the source seam
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-08-30_regate/fusion_matrix_cuda.json``,
``cells_ranked_by_demand[1]``). 13 of the 16 drive ``zero_metal_B`` inside the seam,
which is the most of any cell on this board bar the served one.

IT IS ALSO THE CELL METAL CANNOT TAKE. The Metal board's 21
``cannot_be_bound_on_metal`` instances are a pointer-count verdict against
``MAX_BUFFER_BINDINGS = 31``, and the cylindrical complex curl alone binds eleven
volumes before the constitutive half adds six more. CUDA HAS NO SUCH CEILING: the
kernel parameter space is 32,764 bytes on sm_70+ and this signature spends 296 of
them, so the fusion is refused here on nothing.

=============================================================================
WHAT ONE LAUNCH PERFORMS
=============================================================================

:data:`REPLACES` is ``step_B`` (driver.py:3291) -> ``zero_metal_B`` (:3295) ->
``update_H`` (:3298), declared rather than inferred from the two slots.

THE TWO MIRROR FILLS ARE ABSENT BY REFUSAL. ``fill_symmetry_bc_B`` (:3294) and
``fill_folded_far_ghosts_B`` (:3296) do nothing on an unfolded grid, and this pair
refuses every folded one -- twice over, once through
``covers_pml_cylindrical_complex_curl``'s own clause ("a mirror plane is active:
symmetry folding is not carried") and once in this file, so the guarantee that makes
:data:`REPLACES` honest does not rest on a clause in another module. Grid refuses a
mirror on a Dcyl cell in the first place; the refusal is written anyway, because a
fused product's REPLACES is a claim about what the launch did and a claim may not
depend on a fact nobody restated.

``zero_metal_B`` IS carried, as a register clear beside a store, so the constitutive
half reads the wiped value inside the same launch.

THE RADIAL PREFIX IS NOT PART OF THE WELD AND IS NOT MEANT TO BE.
``cylindrical_prefix`` ends in ``xp.cumsum`` on the array path, deliberately: CuPy's
float32 cumsum is not a sequential accumulation, so a hand-written column-serial scan
would be a DIFFERENT float32 number and this family's whole claim is bytewise
identity. It is computed before the launch exactly as the certified stand-alone
``cyl_step_B_pml_complex`` computes it, and the fusion neither adds nor removes it.

=============================================================================
THE AXIS TAIL IS PART OF THE SEAM, WHICH IS THE ONE THING THIS PAIR ADDS
=============================================================================

The certified cylindrical curl does not end at ``pml_apply``. At |m| >= 2 it runs
``stepping._cylindrical_axis_zero_B`` (:648-671) AFTER the recurrence, zeroing all
three B components and their three ``fu`` volumes on rows ``[0:zero_rows]``. A weld
that captured the register at ``pml_apply`` and stopped would hand ``update_H`` the
PRE-ZEROING displacement on every one of those rows -- a converged, smooth, entirely
wrong field near the axis, on exactly the rows a Dcyl run cares most about.

So the carry follows the certified store rather than the certified recurrence: each
of the three ``cf_store(B*, idx, cf_zero())`` lines becomes a register clear beside
the same store, anchored on the certified line so a change to that block is a named
failure rather than a silent divergence.

=============================================================================
THE ALIASING HAZARD
=============================================================================

The curl half writes B in place and the constitutive half reads it. Binding that one
allocation twice -- once writable for the curl, once ``const __restrict__`` for the
constitutive source -- is undefined behaviour NVRTC miscompiles without a diagnostic.
So B appears EXACTLY ONCE in :data:`_SIGNATURE` and the constitutive half has no
source pointers at all.

=============================================================================
THE ONE NAME COLLISION THE SPLICE MUST RESOLVE
=============================================================================

The two certified bodies do not collide on their volume names -- the cylindrical curl
spells its own ``Bx/By/Bz`` and ``Ex/Ey/Ez`` while the complex constitutive template
uses ``f*``/``g*``/``w*`` -- so those are lifted untouched. They DO collide on
``kms_x/kms_y/kms_z``, which is the HALF-INTEGER sub-lattice for the B curl
(``stepping._curl_coefficients``, :2418) and the INTEGER one for ``update_H``
(:2428 through stepping.py:948). That pair compiles perfectly and differs by half a
cell in the absorber profile, which is why the constitutive group is renamed
``kms_int_*`` in the signature and in the spliced body both.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch.
"""

from __future__ import annotations

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

# THE CERTIFIED TEXT, and the predicates. ``complex_emitter``,
# ``cylindrical_coverage``, ``coverage``, ``cylindrical_prefix`` and
# ``in_seam_coverage`` are all CuPy-free, so the emitter and the predicate run at the
# merge bar. ``cylindrical_complex_kernels`` imports CuPy at module scope, so its
# device text is reached through a defensive import and the emitter refuses BY NAME
# where it is absent (:func:`_certified_source`).
try:
    from . import complex_emitter
    from .coverage import covers_real_pml_complex_constitutive
    from .cylindrical_coverage import covers_pml_cylindrical_complex_curl
    from .cylindrical_prefix import cylindrical_prefix
    from .in_seam_coverage import zero_metal_axes
    from . import compile_cache
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    def _load(stem):
        here = _os.path.dirname(_os.path.abspath(__file__))
        spec = _importlib_util.spec_from_file_location(
            f"cuda_kernels_{stem}", _os.path.join(here, f"{stem}.py"))
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    complex_emitter = _load("complex_emitter")
    covers_real_pml_complex_constitutive = _load(
        "coverage").covers_real_pml_complex_constitutive
    covers_pml_cylindrical_complex_curl = _load(
        "cylindrical_coverage").covers_pml_cylindrical_complex_curl
    cylindrical_prefix = _load("cylindrical_prefix").cylindrical_prefix
    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")

try:
    from . import cylindrical_complex_kernels
except ImportError:  # a host with no CuPy: only the emitter is unavailable
    cylindrical_complex_kernels = None

try:
    from . import complex_pml_kernels
except ImportError:  # a host with no CuPy: only the launcher is unavailable
    complex_pml_kernels = None

# The source-seam clause, shared with every other fused-pair predicate on all three
# tracks. Imported rather than re-spelled: writing it out per product produced a real
# bug once.
try:
    from .. import deposit_repair as _deposit_repair
except ImportError:  # loaded by path, outside the package
    import importlib.util as _importlib_util
    import os as _os

    _spec = _importlib_util.spec_from_file_location(
        "meep_gpu_deposit_repair",
        _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "deposit_repair.py"))
    _deposit_repair = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_deposit_repair)


# =============================================================================
# THE PARTITION — PLAIN ASSIGNMENTS, NO ANNOTATION
# =============================================================================
#
# The partition readers walk the syntax tree WITHOUT importing the module (they have
# to — these modules import cupy) and an annotated assignment is an ``ast.AnnAssign``
# those readers do not match.

#: Byte-identical to the array path with a gate verdict AND a ``certification.json``
#: record behind it. A fused product's bit-identity is a claim about the WELD, which
#: no verdict on either half establishes.
#: RELEASED 2026-09-04 under BOTH float32 subnormal policies, at EVERY m class --
#: m = 0 (the third arm, landed the same day), |m| = 1 and |m| >= 2 -- by
#: ``parity/meep_gpu/gate_cuda_fused_complex_pairs.py --family cylindrical``
#: on one RTX A6000: byte-identical per COMPLETE driver step over 60 steps against
#: BOTH the array path and the separately certified halves on every fixture in
#: CYLINDRICAL_SPECS (including the corpus row's own (150, 1, 300) m = 0 shape),
#: the launch counted by two instruments, the deposit legs carrying a real in-seam
#: source with the unrepaired null control diverging, and every mutation leg --
#: the m = 0 tail's own included -- scored as declared. The block is
#: ``cuda_cylindrical_fused_magnetic_pair_2026-09-04`` in ``certification.json``; the flip was made BEFORE
#: the run so the gate measured the bytes that ship.
CERTIFIED_KERNELS = (
    "fused_magnetic_pair_pml_cyl_complex",
)
#: EMPTY since 2026-09-04; what emptied it was the run above.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its fused launch with the deposit repair? TRUE, wired
#: through ``fused_pairs._install_fused_pair`` -- the flag and the wiring change
#: together or not at all (``deposit_repair.py:221-226``).
#:
#: THE FOLD CLAUSES IN ``deposit_repair._folded_seam_reasons`` ARE VACUOUS ON EVERY
#: ROW THIS ADMITS, including the one that names the cylindrical r = 0 axis: that
#: clause fires only on a FOLDED grid, and this predicate refuses every folded grid
#: outright. What the repair inverts here is ``(f + kps*fw_fresh) - kms*fw_prev`` over
#: complex64 storage, which is word for word what ``constitutive_apply`` computes.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_cylindrical_fused_magnetic_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_magnetic_pair_pml_cyl_complex"

#: The driver passes ONE launch of this kernel performs, in driver order.
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: The sub-step slot the planner holds this on.
SLOT = "step_B"

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ cf pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the helper "
            "names and returns the word pair it already computed. The expression, "
            "its parenthesisation and the store to f are unchanged."},
    {"line": "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), "
             "sinv_u));",
     "became": "    cf value = mul_field_left(cf_sub(cf_add(a, fu_new), fprev), "
               "sinv_u);\n    cf_store(f, idx, value);\n    return value;",
     "why": "same right-hand side, same tree, same store; the result is "
            "additionally named so it can be returned."},
    {"line": "        pml_apply(Bx, fu_Bx, idx, curl, ...);",
     "became": "        b0 = pml_apply_reg(Bx, fu_Bx, idx, curl, ...);",
     "why": "capture the register. Three lines, one per component; the argument "
            "lists are untouched. Three 'cf b* = cf_zero();' declarations are "
            "hoisted above the certified braced blocks because a value declared "
            "inside one does not outlive it."},
    {"line": "        cf_store(Bx, idx, cf_zero());   "
             "(stepping._cylindrical_axis_zero_B)",
     "became": "        b0 = cf_zero(); cf_store(Bx, idx, b0);",
     "why": "THE AXIS TAIL, CARRIED. The |m| >= 2 rule runs AFTER the recurrence "
            "and zeroes B on rows [0:zero_rows]; a weld that captured at pml_apply "
            "and stopped would hand update_H the pre-zeroing displacement on those "
            "rows. The guard, the row test and the six certified stores are "
            "untouched -- the register is cleared beside the three B stores it "
            "already makes."},
    {"line": "    cf s0 = cf_load(g0, idx);",
     "became": "    cf s0 = b0;",
     "why": "THE SEAM. The certified constitutive body opens each component with a "
            "reload of the flux density the curl just stored; this reads the "
            "register instead, which is also what lets B be bound exactly once."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i]);",
     "became": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_int_x[i]);",
     "why": "THE SUB-LATTICE RENAME AND NOTHING ELSE. Both halves ship a vector "
            "spelled kms_a on DIFFERENT Yee sub-lattices -- half-integer for the B "
            "curl, integer for the H constitutive -- and letting one shadow the "
            "other is a half-cell error in the absorber profile, not a compile "
            "failure. The two bodies do NOT collide on f*/g*/w*, so those keep "
            "their certified spelling."},
    {"line": "        array[_face(axis, 0)] = 0   (stepping._zero_metal, :2206)",
     "became": "    if (wall_x && i == 0) { b0 = cf_zero(); cf_store(Bx, idx, b0); }",
     "why": "THE ONE RE-SPELLING, and it is launch geometry rather than "
            "arithmetic. The array path assigns the integer 0 to a plane; this "
            "launch walks the VOLUME and already holds i/j/k. Same diagonal, same "
            "stored cell 0, same (+0.0f, +0.0f) word pair."},
    {"line": "    if (m_class == 0 && i == 0) { cf_store(Bx, idx, cf_zero()); }   "
             "(stepping._cylindrical_axis_zero_B, the m = 0 branch, 2026-09-04)",
     "became": "    if (m_class == 0 && i == 0) { b0 = cf_zero(); cf_store(Bx, idx, b0); }",
     "why": "THE m = 0 AXIS TAIL, CARRIED. It runs AFTER the recurrence and zeroes "
            "Br -- THE FIELD ONLY, never fu_Bx (stepping.py:686-688) -- on the axis "
            "row. A weld that captured at pml_apply and stopped would hand update_H "
            "the pre-zeroing Bx there. The guard, the row test and the store are "
            "untouched; the register is cleared beside the store."},
    {"line": "    cf b2 = cf_zero();",
     "became": "    cf b2 = cf_zero();\n"
               "    cf pre_w_0 = cf_load(w0, idx);   // ... twelve lines, 24 words",
     "why": "THE OWN-CELL HOIST. Every word this thread reads at its own idx -- "
            "w*, f* (H), fu_B*, B* -- is issued before the kernel's first store. "
            "UNGUARDED, because the certified loads are: this family refuses every "
            "folded grid, so no fill images a cell and no other block writes any "
            "word at this thread's idx in this launch. The only stores this thread "
            "makes to those words before their certified load points are none: the "
            "axis tails and the wall clears store B/fu_B AFTER pml_apply_reg_pre "
            "has consumed them. hoisted_loads()."},
    {"line": "__device__ __forceinline__ cf pml_apply_reg(\n    ...)",
     "became": "__device__ __forceinline__ cf pml_apply_reg_pre(\n"
               "    ...,\n    cf fprev, cf fcur\n)",
     "why": "DERIVED, not typed: pml_apply_reg_pre_source() rewrites the helper's "
            "two own-cell loads into parameters by anchored replacement and "
            "refuses unless its statements equal the certified helper's under "
            "fprev -> cf_load(fu, idx), fcur -> cf_load(f, idx). pml_apply_reg "
            "itself is RETIRED from the prelude: no call is left, and a dead copy "
            "is where a first-site mutation needle lands vacuously."},
    {"line": "__device__ __forceinline__ void constitutive_apply(\n    ...)",
     "became": "__device__ __forceinline__ void constitutive_apply_pre(\n"
               "    ..., cf src,\n    cf prev, cf fcur, float kps, float kms\n)",
     "why": "DERIVED from complex_emitter's constitutive_apply the same way (prev -> "
            "cf_load(fw, idx), fcur -> cf_load(f, idx)). The certified helper is "
            "RETIRED from this product's prelude, unlike fused_magnetic_pair's: this "
            "family carries no ghost cell, so all three of its calls take the "
            "derived twin and a kept copy would be dead text."},
    {"line": "        b0 = pml_apply_reg(Bx, fu_Bx, idx, curl, ...);",
     "became": "        b0 = pml_apply_reg_pre(Bx, fu_Bx, idx, curl, ..., "
               "pre_fu_0, pre_b_0);",
     "why": "the three own-cell curl calls take their preloaded words; the "
            "argument list is otherwise the certified one."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_int_x[i]);",
     "became": "    constitutive_apply_pre(f0, w0, idx, s0, pre_w_0, pre_h_0, "
               "kps_x[i], kms_int_x[i]);",
     "why": "the three constitutive statements take their preloaded words, in the "
            "composer (hoisted_constitutive_body), not in "
            "certified_constitutive_body(), which stays the certified lift."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body",
    "certified_curl_body", "constitutive_apply_pre_source",
    "covers_cylindrical_fused_magnetic_pair",
    "cylindrical_fused_magnetic_pair_prelude",
    "cylindrical_fused_magnetic_pair_source", "device_sources",
    "hoisted_constitutive_body", "hoisted_loads", "pml_apply_reg_pre_source",
    "launch_cylindrical_fused_magnetic_pair",
    "run_cylindrical_fused_magnetic_pair", "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================

_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"

_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ cf pml_apply_reg(\n"
_PML_APPLY_STORE = (
    "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u));\n")
_PML_APPLY_STORE_REG = (
    "    // THE ONE EDIT TO THIS HELPER: the right-hand side, its\n"
    "    // parenthesisation and the store are the certified ones; the value is\n"
    "    // additionally NAMED so the constitutive half can read it from a\n"
    "    // register instead of reloading the word pair from global memory.\n"
    "    cf value = mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u);\n"
    "    cf_store(f, idx, value);\n"
    "    return value;\n")

#: The three carried registers, and the certified curl's own name for each target.
_CARRIED: Tuple[str, ...] = ("0", "1", "2")
_TARGETS: Tuple[str, ...] = ("Bx", "By", "Bz")

#: ``stepping._zero_metal``'s diagonal for the B side, as (register, target, flag,
#: coordinate). SPELLED HERE rather than imported from anything D-shaped: the D
#: family's table is the off-diagonal complement, and a pair that reused it would
#: clear the wrong two components on every walled run. ``fields.IYEE_SHIFTS`` gives
#: Bx (0,1,1), By (1,0,1), Bz (1,1,0) whatever the coordinate system, so each B
#: component is wiped on its OWN axis and only there -- on a Dcyl grid that is r for
#: Br, phi for Bp and z for Bz.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("b0", "Bx", "wall_x", "i"),
    ("b1", "By", "wall_y", "j"),
    ("b2", "Bz", "wall_z", "k"),
)

#: ``stepping._cylindrical_axis_zero_B``'s m = 0 branch (2026-09-04), exactly as the
#: certified curl emits it, and its carried rewrite. Anchored as a WHOLE BLOCK because
#: its one store also appears in the |m| >= 2 branch.
_AXIS_TAIL_M0 = (
    "    if (m_class == 0 && i == 0) {\n"
    "        cf_store(Bx, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M0_CARRIED = (
    "    // THE m = 0 AXIS TAIL, CARRIED (2026-09-04). stepping._cylindrical_axis_zero_B\n"
    "    // (:657-659) zeroes Br on the axis row -- THE FIELD ONLY, never fu_Bx -- after\n"
    "    // the recurrence, so the register has to follow the store or update_H would\n"
    "    // read the pre-zeroing Bx there.\n"
    "    if (m_class == 0 && i == 0) {\n"
    "        b0 = cf_zero(); cf_store(Bx, idx, b0);\n"
    "    }\n")

#: The fused entry point. The ONLY hand-written device text in this module, and it is
#: a signature. B appears EXACTLY ONCE -- see the module docstring's aliasing section.
_SIGNATURE = r'''
extern "C" __global__ void fused_magnetic_pair_pml_cyl_complex(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 B.
    // The curl half writes it and the constitutive half reads it from a register;
    // binding it a second time as a const __restrict__ source would be two
    // restrict pointers to one allocation, which is UB NVRTC miscompiles silently.
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    // The radial prefix (nx + 1 rows) and the two bound i*m/r coefficient rows.
    const float* __restrict__ pfx,
    const float* __restrict__ imr0, const float* __restrict__ imr2,
    // H and its constitutive history. The certified constitutive template's own
    // names: the two bodies do not collide on these.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx,
    float minus_dtdx, float inc_re, float inc_im,
    // The B curl's HALF-INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The H constitutive's INTEGER coefficients. Its kms is renamed kms_int_*:
    // the two sub-lattices collide on the bare name in one scope, and a shadow
    // there is a half-cell error in the absorber profile, not a compile failure.
    const float* __restrict__ kps_x, const float* __restrict__ kms_int_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_int_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_int_z,
    int bc_x, int bc_y, int bc_z,
    // This family carries NO Bloch phase -- the predicate refuses one -- so every
    // flag is 0 and cshift_* never multiplies. The trio is passed anyway because
    // it is the certified helper's own signature.
    int ph_x, int ph_y, int ph_z,
    int m_class, int zero_rows,
    // zero_metal_B's three walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z
) {
'''


def _split_body(source: str, prelude: str, name: str) -> str:
    """One emitted kernel string, reduced to its body.

    Raises rather than returning a truncated body: a splice that silently dropped a
    component would produce a kernel that steps two fields and reports success.
    """
    if not source.startswith(prelude):
        raise AssertionError(
            f"{name} no longer begins with the prelude this module lifts "
            f"separately; the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, "
            f"not 1; the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure."""
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this family LIFTS that line rather than "
            f"retyping it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the "
            f"lift of {what} would take an arbitrary one")
    return matches[0]


def _certified_source() -> Any:
    """Refuse the emitter BY NAME on a host where the cylindrical text is unreachable.

    THE SPLICE IS THE LIFT, so a missing half is not a degraded emit -- there is
    nothing to emit. Raising here keeps that fact one frame from the caller instead
    of surfacing as ``'NoneType' object has no attribute`` inside a string operation.
    :func:`covers_cylindrical_fused_magnetic_pair` needs none of it and still answers.
    """
    if cylindrical_complex_kernels is None:
        raise RuntimeError(
            "cylindrical_complex_kernels is not importable on this host (it imports "
            "CuPy at module scope), so there is no certified cylindrical text to "
            "splice. covers_cylindrical_fused_magnetic_pair needs none of it and "
            "still answers")
    return cylindrical_complex_kernels


def _curl_prelude(arm) -> str:
    """The prelude ``cylindrical_complex_source`` builds, before its own template."""
    module = _certified_source()
    code = complex_emitter.normalized_expansion(arm)
    return (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[code]
            + complex_emitter._TAIL + module._CYLINDRICAL_HELPERS)


def _constitutive_prelude(arm) -> str:
    """The prelude ``complex_source`` builds -- the curl's, minus the cyl helpers."""
    code = complex_emitter.normalized_expansion(arm)
    return (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[code]
            + complex_emitter._TAIL)


# -----------------------------------------------------------------------------
# THE OWN-CELL HOIST (S3d, ported from fused_magnetic_pair to the cf word pair). The
# fused kernel issues every own-cell word it reads -- fu_B*, B*, w*, f* -- before its
# first store, and the two helpers that would have loaded them take the words as
# arguments. Both helpers are DERIVED from the certified text by anchored replacement,
# and their statements are asserted equal to the certified helper's under the
# argument-to-load map, so a certified edit that moves the arithmetic refuses here by
# name.
# -----------------------------------------------------------------------------

#: The certified complex helpers' signatures, as complex_emitter._TAIL spells them.
_CONSTITUTIVE_SIGNATURE = "__device__ __forceinline__ void constitutive_apply(\n"
_CONSTITUTIVE_PARAMETERS = (
    "    float* f, float* fw, int idx, cf src, float kps, float kms\n)")
_PML_APPLY_PARAMETERS = (
    "    float kms, float sinv, float kms_u, float sinv_u\n)")


def _hoist_function_text(prelude: str, signature: str, what: str) -> str:
    if prelude.count(signature) != 1:
        raise AssertionError(
            f"the prelude declares {what} {prelude.count(signature)} times, not once; "
            f"the hoisted helper has nothing to be derived from")
    text = prelude[prelude.index(signature):]
    end = text.find("\n}\n")
    if end < 0:
        raise AssertionError(f"{what} has no closing brace in the prelude")
    return text[: end + len("\n}\n")]


def _hoist_statements(text: str, loads: Dict[str, str], what: str) -> List[str]:
    """The helper's statements with every PARAMETERISED name replaced by its load.

    Only a ``cf NAME = cf_load(P, idx);`` line whose NAME the derivation turns into a
    parameter is a binding, and it must bind exactly the load the map names; every
    other line is a statement and stays in the stream.
    """
    import re  # noqa: PLC0415
    if text.count(") {\n") != 1:
        raise AssertionError(f"{what}: the parameter list no longer closes on one line")
    body = text.split(") {\n", 1)[1][: -len("}\n")]
    names: Dict[str, str] = {}
    out: List[str] = []
    for line in body.splitlines():
        code = line.split("//", 1)[0].strip()
        if not code:
            continue
        bound = re.fullmatch(r"cf (\w+) = (cf_load\(\w+, idx\));", code)
        if bound and bound.group(1) in loads:
            if bound.group(2) != loads[bound.group(1)]:
                raise AssertionError(
                    f"{what}: {bound.group(1)} is bound to {bound.group(2)}, not to "
                    f"{loads[bound.group(1)]}; the load map names the wrong word")
            names[bound.group(1)] = bound.group(2)
            continue
        for name, load in loads.items():
            code = re.sub(rf"\b{name}\b", load, code)
        out.append(code)
    return out


def _hoist_derive(helper: str, edits, loads: Dict[str, str], what: str) -> str:
    derived = helper
    for old, new, anchor in edits:
        if derived.count(old) != 1:
            raise AssertionError(
                f"the certified {what} no longer carries {anchor} ({old!r}) exactly once "
                f"(found {derived.count(old)}); the hoisted helper cannot be derived from it")
        derived = derived.replace(old, new, 1)
    certified = _hoist_statements(helper, loads, what)
    hoisted = _hoist_statements(derived, loads, what + " (hoisted)")
    if certified != hoisted:
        raise AssertionError(
            f"the hoisted {what} does not carry the certified arithmetic: "
            f"{certified!r} != {hoisted!r}")
    return derived


def constitutive_apply_pre_source(arm) -> str:
    """``complex_emitter``'s ``constitutive_apply`` taking ``prev = cf_load(fw, idx)``
    and ``fcur = cf_load(f, idx)`` as arguments -- derived from the certified text."""
    helper = _hoist_function_text(
        _constitutive_prelude(arm), _CONSTITUTIVE_SIGNATURE, "constitutive_apply")
    return _hoist_derive(helper, (
        ("void constitutive_apply(\n", "void constitutive_apply_pre(\n", "its signature"),
        (_CONSTITUTIVE_PARAMETERS,
         "    float* f, float* fw, int idx, cf src,\n"
         "    cf prev, cf fcur, float kps, float kms\n)", "its parameter list"),
        ("    cf prev = cf_load(fw, idx);\n", "", "its load of fw"),
        ("    cf a = cf_load(f, idx);\n", "    cf a = fcur;\n", "its load of f"),
    ), {"prev": "cf_load(fw, idx)", "fcur": "cf_load(f, idx)"}, "constitutive_apply")


def pml_apply_reg_pre_source(prelude: str) -> str:
    """The rewritten ``pml_apply_reg`` taking ``fprev = cf_load(fu, idx)`` and
    ``fcur = cf_load(f, idx)`` as arguments -- derived from the rewritten prelude."""
    helper = _hoist_function_text(prelude, _PML_APPLY_SIGNATURE_REG, "pml_apply_reg")
    return _hoist_derive(helper, (
        (_PML_APPLY_SIGNATURE_REG,
         _PML_APPLY_SIGNATURE_REG.replace("pml_apply_reg(", "pml_apply_reg_pre("),
         "its signature"),
        (_PML_APPLY_PARAMETERS,
         "    float kms, float sinv, float kms_u, float sinv_u,\n"
         "    cf fprev, cf fcur\n)", "its parameter list"),
        ("    cf fprev = cf_load(fu, idx);\n", "", "its load of fu"),
        ("    cf a = mul_field_left(cf_load(f, idx), kms_u);\n",
         "    cf a = mul_field_left(fcur, kms_u);\n", "its load of f"),
    ), {"fprev": "cf_load(fu, idx)", "fcur": "cf_load(f, idx)"}, "pml_apply_reg")


def _retire(prelude: str, signature: str, what: str) -> str:
    """Remove one helper's function text from ``prelude``, exactly once."""
    helper = _hoist_function_text(prelude, signature, what)
    if prelude.count(helper) != 1:
        raise AssertionError(
            f"the prelude carries {what} {prelude.count(helper)} times, not once; the "
            f"retired helper has no single anchor")
    return prelude.replace(helper, "", 1)


def hoisted_loads() -> str:
    """The twelve own-cell loads (24 words), unguarded as the certified loads are."""
    lines = [
        "    // --- the own-cell hoist: every word this thread reads at idx, issued",
        "    // before the kernel's first store. Unguarded, as the certified loads",
        "    // are: this family refuses every folded grid, so no other block writes",
        "    // a word at this idx in this launch. ---",
    ]
    for register in _CARRIED:
        lines.append(f"    cf pre_w_{register} = cf_load(w{register}, idx);")
        lines.append(f"    cf pre_h_{register} = cf_load(f{register}, idx);")
    for register, target in zip(_CARRIED, _TARGETS):
        lines.append(f"    cf pre_fu_{register} = cf_load(fu_{target}, idx);")
        lines.append(f"    cf pre_b_{register} = cf_load({target}, idx);")
    return "\n".join(lines) + "\n"


def cylindrical_fused_magnetic_pair_prelude(arm) -> str:
    """The certified prelude with ``pml_apply`` turned into a value.

    ONE prelude for both halves, and that is a fact about the two families rather
    than a convenience: ``cylindrical_complex_source`` builds its source from the
    CERTIFIED complex family's ``_HEAD``/``_ARM_SOURCE``/``_TAIL`` plus its own
    ``_CYLINDRICAL_HELPERS``, so the constitutive half's prelude is a strict prefix
    of the curl half's. ``mul_complex_left`` is an ALIAS of ``rotate_field_left``
    and adds no arithmetic.
    """
    prelude = _curl_prelude(arm)
    if prelude.count(_PML_APPLY_SIGNATURE) != 1:
        raise AssertionError(
            f"the certified prelude declares pml_apply "
            f"{prelude.count(_PML_APPLY_SIGNATURE)} times, not once; the value "
            f"rewrite has no anchor")
    if prelude.count(_PML_APPLY_STORE) != 1:
        raise AssertionError(
            "the certified pml_apply no longer closes with the store this module "
            "turns into a named value; the arithmetic may have moved")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_PML_APPLY_STORE, _PML_APPLY_STORE_REG, 1)
    # THE HOIST RETIRES BOTH HELPERS it derives from: every call takes the derived
    # twin, so a kept copy would be dead text -- and a dead copy of a helper is where a
    # first-site needle lands vacuously. Derived first, then removed, exactly once.
    hoisted = pml_apply_reg_pre_source(prelude)
    prelude = _retire(prelude, _PML_APPLY_SIGNATURE_REG, "pml_apply_reg")
    prelude = _retire(prelude, _CONSTITUTIVE_SIGNATURE, "constitutive_apply")
    return prelude + constitutive_apply_pre_source(arm) + hoisted


def certified_curl_body(arm) -> str:
    """``cyl_step_B_pml_complex``'s body, lifted, with the three registers captured.

    TWO CAPTURES PER COMPONENT, not one, and the second is the whole reason this
    family needs a paragraph the Cartesian one does not: the recurrence's value is
    taken at ``pml_apply``, and the |m| >= 2 axis tail then OVERWRITES B on rows
    ``[0:zero_rows]`` after it. Both are followed -- and since 2026-09-04 so is the
    m = 0 tail, which zeroes ``Bx`` alone on the axis row (stepping.py:686-688).
    """
    module = _certified_source()
    body = _split_body(module.cylindrical_complex_source("step_B", arm),
                       _curl_prelude(arm), "cylindrical_complex_source('step_B')")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the "
            "carried registers have nowhere to be declared")
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END,
        "\n    // The three carried registers. Declared HERE because each\n"
        "    // certified component below is a braced scope, and a value declared\n"
        "    // inside one does not outlive it.\n"
        "    cf b0 = cf_zero();\n"
        "    cf b1 = cf_zero();\n"
        "    cf b2 = cf_zero();\n"
        "\n",
        hoisted_loads(),
        tail,
    ))
    for register, target in zip(_CARRIED, _TARGETS):
        old = f"pml_apply({target}, fu_{target}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} "
                f"{body.count(old)} times, not once; the capture has no anchor")
        body = body.replace(
            old, f"b{register} = pml_apply_reg({target}, fu_{target}, ", 1)
        prefix = f"        b{register} = pml_apply_reg({target}, fu_{target}, "
        call = _line_starting(body, prefix,
                              f"target {target}'s captured pml_apply")
        if not call.endswith(");"):
            raise AssertionError(
                f"the certified curl body no longer closes {target}'s pml_apply on "
                f"one line ({call!r})")
        # THE HOIST: the same statement, taking its two preloaded own-cell words.
        body = body.replace(call, call.replace(
            f"b{register} = pml_apply_reg(", f"b{register} = pml_apply_reg_pre(", 1)[:-2]
            + f", pre_fu_{register}, pre_b_{register});", 1)
    if "pml_apply(" in body or "pml_apply_reg(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be "
            "written to global memory and never read into the seam")

    # THE m = 0 AXIS TAIL FIRST, anchored as a WHOLE BLOCK: ``cf_store(Bx, idx,
    # cf_zero());`` appears in BOTH the m = 0 block and the |m| >= 2 block, so the
    # per-line anchor below could not name which one it had rewritten until this
    # block's store has been rewritten out from under it.
    if body.count(_AXIS_TAIL_M0) != 1:
        raise AssertionError(
            f"the certified curl body carries the m = 0 branch of "
            f"stepping._cylindrical_axis_zero_B {body.count(_AXIS_TAIL_M0)} times, not "
            f"once; the register clear has no anchor and update_H would read the "
            f"pre-zeroing Bx on the axis row")
    body = body.replace(_AXIS_TAIL_M0, _AXIS_TAIL_M0_CARRIED, 1)

    # THE |m| >= 2 AXIS TAIL. The guard and the six stores are certified; the
    # register is cleared beside each of the three B stores, anchored on the
    # certified line.
    for register, target in zip(_CARRIED, _TARGETS):
        old = f"        cf_store({target}, idx, cf_zero());\n"
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body's |m| >= 2 axis tail stores zero into "
                f"{target} {body.count(old)} times, not once; the register clear "
                f"has no anchor and update_H would read the pre-zeroing "
                f"displacement on rows [0:zero_rows]")
        body = body.replace(
            old,
            f"        b{register} = cf_zero(); cf_store({target}, idx, "
            f"b{register});   // THE AXIS TAIL, CARRIED\n", 1)
    return body


def zero_metal_carry() -> str:
    """``zero_metal_B`` (driver.py:3295), carried between the two halves.

    ``stepping._zero_metal`` writes the integer 0 into stored cell 0 of every
    component whose Yee shift on a walled axis is 0 -- for B that is the DIAGONAL.
    The register is cleared BESIDE the store, because the constitutive half below
    reads the register and not the volume.

    IT RUNS AFTER THE AXIS TAIL, which is the driver's order and not a choice: the
    axis tail is part of ``step_B`` (stepping.py:403-404, inside the sub-step) and
    ``zero_metal_B`` is a separate driver pass at :3295. A wall on the r axis and the
    |m| >= 2 axis rows both write the same cells to zero, so the composition is
    order-independent there; stating the order anyway is what keeps that a fact
    rather than an accident.
    """
    lines = [
        "\n    // stepping._zero_metal (:2206-2245) / in_seam_passes.zero_metal_B,\n"
        "    // CARRIED, and AFTER the certified axis tail above -- which is the\n"
        "    // driver's order (the tail is inside step_B, this is the pass at\n"
        "    // driver.py:3295). The diagonal only: a B component sits ON the wall\n"
        "    // of the axis whose Yee shift is 0, which is its own.\n"
    ]
    for register, target, flag, coordinate in _ZERO_METAL_ROWS:
        lines.append(
            f"    if ({flag} && {coordinate} == 0) {{ {register} = cf_zero(); "
            f"cf_store({target}, idx, {register}); }}\n")
    return "".join(lines)


def certified_constitutive_body(arm) -> str:
    """``update_H_pml_complex_bloch``'s body, lifted, reading the registers.

    Drops the second index decomposition (the curl body above already declares
    ``idx``, ``i``, ``j`` and ``k`` and its bounds guard has already returned), turns
    the three source reloads into the seam, and renames the constitutive
    coefficients' sub-lattice.
    """
    body = _split_body(complex_emitter.complex_source("update_H", arm),
                       _constitutive_prelude(arm), "complex_source('update_H')")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own line; "
            "the duplicate decomposition cannot be identified")
    tail = body.split(_DECODE_END, 1)[1]

    for register in _CARRIED:
        old = f"    cf s{register} = cf_load(g{register}, idx);\n"
        if tail.count(old) != 1:
            raise AssertionError(
                f"the certified constitutive body loads s{register} "
                f"{tail.count(old)} times, not once; the seam has no anchor")
        tail = tail.replace(
            old,
            f"    cf s{register} = b{register};   // THE SEAM: the register the "
            f"curl half just stored\n", 1)
    if "cf_load(g" in tail:
        raise AssertionError(
            "a source reload survived the seam rewrite; the constitutive half would "
            "need B bound a second time, which is the aliasing hazard this "
            "signature exists to avoid")

    for register, axis in zip(_CARRIED, ("x", "y", "z")):
        prefix = f"    constitutive_apply(f{register}, w{register}, idx, s{register}, "
        call = _line_starting(tail, prefix,
                              f"target {register}'s constitutive_apply")
        coordinate = "ijk"[int(register)]
        expected = (f"{prefix}kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);")
        if call != expected:
            raise AssertionError(
                f"target {register}'s constitutive_apply is {call!r}, not the "
                f"certified {expected!r}; the sub-lattice rename would be applied "
                f"to an argument list this module has not read")
        tail = tail.replace(
            call, call.replace(f"kms_{axis}[", f"kms_int_{axis}[", 1), 1)
    return tail


def hoisted_constitutive_body(arm) -> str:
    """:func:`certified_constitutive_body` with the three statements taking their
    preloaded words (:func:`hoisted_loads`). The certified function is unchanged."""
    body = certified_constitutive_body(arm)
    for register in _CARRIED:
        prefix = f"    constitutive_apply(f{register}, w{register}, idx, s{register}, "
        call = _line_starting(body, prefix, f"target {register}'s constitutive_apply")
        body = body.replace(call, call.replace(
            prefix, f"    constitutive_apply_pre(f{register}, w{register}, idx, "
                    f"s{register}, pre_w_{register}, pre_h_{register}, ", 1), 1)
    if "constitutive_apply(" in body:
        raise AssertionError(
            "a constitutive_apply call survived the hoist rewrite; the retired helper "
            "would be called with no declaration")
    return body


def cylindrical_fused_magnetic_pair_source(arm) -> str:
    """The whole fused kernel for one expansion arm: prelude, signature, body."""
    source = "".join((
        cylindrical_fused_magnetic_pair_prelude(arm),
        _SIGNATURE,
        certified_curl_body(arm),
        zero_metal_carry(),
        hoisted_constitutive_body(arm),
        "}\n",
    ))
    # PURE ASCII IS A COMPILE REQUIREMENT on this family, not a style rule; checked
    # here so a mutation leg that inserts a non-ASCII character is refused at
    # emission with the reason instead of at NVRTC three frames away.
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by arm name -- for a digest."""
    return {name: cylindrical_fused_magnetic_pair_source(name)
            for name in sorted(complex_emitter.EXPANSIONS)}


#: NVRTC compile options -- CORRECTNESS, not performance, and identical to both
#: certified halves'.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm, name: str = KERNEL_NAME):
    """Compile the fused kernel under one arm, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING: the source is part of
    the memo key, and a gate mutates this family by monkeypatching the emitter. A
    source memoized at first call would hand back the pre-mutation string forever.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be "
            "compiled; the predicate and the emitter need no device and still run")
    code = cylindrical_fused_magnetic_pair_source(arm)
    key = compile_cache.kernel_cache_key(
        f"{name}_arm{complex_emitter.normalized_expansion(arm)}", True,
        _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_cylindrical_fused_magnetic_pair(fields: Any, pml: Any, grid: Any,
                                           sources: Any = None,
                                           license: Any = None,
                                           subnormal_policy: Any = None
                                           ) -> Tuple[bool, str]:
    """May ONE launch span ``step_B`` -> ``zero_metal_B`` -> ``update_H``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own
    certified predicate refuses is refused here with that half's reason, prefixed so
    a reader can tell which side said it. What this predicate ADDS is the seam
    clauses: the source slot, the two UNCARRIED fills, and the grid's ability to say
    which axes are walled.
    """
    covered, reason = covers_pml_cylindrical_complex_curl(
        fields, pml, grid, "step_B", license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "H", license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE SOURCE SEAM, CARRIED RATHER THAN REFUSED. A MAGNETIC source is injected
    # BETWEEN the two halves (driver.py:3293). IGNORANCE IS NEVER AN EMPTY SET:
    # `Fields` does not hold the source list, so a predicate that inferred "no
    # sources" from not being told would be the over-covering this clause prevents.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an "
            "empty magnetic source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the driver "
            f"injects it BETWEEN step_B and update_H (driver.py:3293)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

    # THE TWO FILLS ARE REFUSED, NOT CARRIED. The curl half already refuses every
    # folded grid and Grid refuses a mirror on a Dcyl cell before that; the clause is
    # stated HERE anyway, because REPLACES is a claim about what one launch performed
    # and it may not rest on a clause in another module that a future device verdict
    # could licence away.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_B and "
                       "fill_folded_far_ghosts_B run inside it")
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry or any(folded):
        return False, ("a mirror plane is active: fill_symmetry_bc_B "
                       "(driver.py:3294) and fill_folded_far_ghosts_B (:3296) run "
                       "inside this seam and this pair carries neither")

    # zero_metal_B IS CARRIED, so the grid must be able to say which axes are walled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_B cannot be "
                           f"carried in registers")
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE STORED SHAPE THE KERNEL INDEXES MUST BE THE ONE THE WALL PLANE AND THE
    # PREFIX ARE DERIVED FROM. `nx, ny, nz` come from `fields.Bx.shape`; the wall is
    # stored cell 0 of a `grid.stored_cells` axis and the prefix has `nx + 1` rows.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Bx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks "
                       f"Bx.shape {extents}; the wall plane and the radial prefix "
                       f"are derived from the first and indexed into the second")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The fifteen complex volumes bound from ``Fields``, in SIGNATURE ORDER. The prefix
#: and the two i*m/r rows are derived per launch and per configuration respectively,
#: so they are bound separately.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Bx", "By", "Bz",
    "fu_Bx", "fu_By", "fu_Bz",
    "Ex", "Ey", "Ez",
)
_CONSTITUTIVE_BINDINGS: Tuple[str, ...] = (
    "Hx", "Hy", "Hz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def cylindrical_fused_magnetic_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads.

    THE SUB-LATTICE IS DECIDED HERE, not by the caller: while it was an argument,
    every call site was one more place the pairing could be got backwards, and
    backwards is a half-cell error rather than a crash.
    """
    if cylindrical_complex_kernels is None or complex_pml_kernels is None:
        raise RuntimeError(
            "the CuPy-backed table builders are not importable on this host")
    return {"curl": cylindrical_complex_kernels.cylindrical_complex_curl_tables(
                pml, True),
            "constitutive": complex_pml_kernels.complex_constitutive_tables(
                pml, False)}


def assert_disjoint_bindings(fields: "Fields", tables: Dict[str, Dict[str, Any]],
                             prefix: Any, imr_rows: Sequence[Any]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes.

    ONCE PER FROZEN CONFIGURATION, OFF THE LAUNCH PATH. The shared flux density is
    bound once by construction -- the signature has no second B group -- so what is
    left to check is that no OTHER two arguments are the same allocation. THE PREFIX
    AND THE TWO i*m/r ROWS ARE INCLUDED, and they are the reason this check takes
    arguments the Cartesian sibling's does not: the prefix is derived from Ep and a
    scratch buffer aliasing a field volume would be UB reached through a route no
    signature inspection would find.

    Returns the number of distinct allocations checked.
    """
    bound: Dict[int, str] = {}
    collisions = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in _FIELD_BINDINGS + _CONSTITUTIVE_BINDINGS:
        visit(name, getattr(fields, name))
    visit("prefix", prefix)
    for index, row in enumerate(imr_rows):
        visit(f"imr{index}", row)
    for key in _CURL_TABLE_KEYS:
        visit(f"curl:{key}", tables["curl"][key])
    for key in _CONSTITUTIVE_TABLE_KEYS:
        visit(f"constitutive:{key}", tables["constitutive"][key])
    if collisions:
        raise ValueError(
            "the cylindrical fused magnetic pair binds every argument "
            "__restrict__, and these arguments alias, which is undefined behaviour "
            "NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_cylindrical_fused_magnetic_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]], prefix: Any,
        imr_rows: Sequence[Any], boundary_codes: Sequence[Any],
        increment_scalars: Sequence[Any], m_class_code: int, zero_rows_count: int,
        walls: Sequence[int], dtdx: float, arm: Any,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional: a gate compiles a deliberately
    broken copy of the shipped source and hands it here. A launcher that could not be
    handed its own kernel could not arm a single mutation.

    Returns the launch geometry rather than ``None`` so a gate can assert that
    something was actually launched.
    """
    word_view = _certified_source().word_view
    nx, ny, nz = (int(n) for n in fields.Bx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    minus_dtdx, (inc_re, inc_im) = increment_scalars
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(
        word_view(getattr(fields, name)) for name in _FIELD_BINDINGS
    ) + (word_view(prefix),) + tuple(word_view(row) for row in imr_rows) + tuple(
        word_view(getattr(fields, name)) for name in _CONSTITUTIVE_BINDINGS
    ) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
        np.float32(minus_dtdx), np.float32(inc_re), np.float32(inc_im),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + tuple(np.int32(code) for code in boundary_codes) + (
        # NO BLOCH PHASE: this family refuses one, so all three flags are 0 and the
        # certified helper never reaches its rotation.
        np.int32(0), np.int32(0), np.int32(0),
        np.int32(int(m_class_code)), np.int32(int(zero_rows_count)),
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3))
    (kernel or _get_kernel(arm))((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "m_class": int(m_class_code), "zero_rows": int(zero_rows_count),
            "arm": complex_emitter.normalized_expansion(arm)}


def run_cylindrical_fused_magnetic_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, arm: Any, *,
        sources: Any = None, license: Any = None, subnormal_policy: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        imr_rows: Optional[Sequence[Any]] = None, prefix: Any = None,
        scratch: Any = None, kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path.

    ``tables``, ``imr_rows``, ``prefix`` and ``kernel`` are the gate's doors,
    keyword-only, and each is a mutation this family's gate arms.
    """
    module = _certified_source()
    covered, reason = covers_cylindrical_fused_magnetic_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = cylindrical_fused_magnetic_pair_tables(pml)
    m = int(grid.m)
    if imr_rows is None:
        imr_rows = module.imr_rows_for("step_B", grid.xp, m, dtdx,
                                       int(fields.Bx.shape[0]), fields.Bx.dtype)
    if prefix is None:
        prefix = cylindrical_prefix(fields, "step_B", scratch=scratch)
    assert_disjoint_bindings(fields, tables, prefix, imr_rows)
    return launch_cylindrical_fused_magnetic_pair(
        fields, tables, prefix, imr_rows,
        module.cylindrical_complex_boundary_codes(grid),
        module.axis_increment_scalars(m, dtdx), module.m_class(m),
        module.zero_rows(m, bool(grid.accurate_fields_near_cylorigin)),
        zero_metal_axes(grid), dtdx, arm, kernel)
