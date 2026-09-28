"""The Dcyl COMPLEX hand-CUDA fused pair: ``step_D`` -> ``zero_metal_D`` -> ``update_E``.

THE BOARD'S LARGEST UNREACHED CELL, and the one whose whole demand is blocked by the
injection rather than by the weld: ``D_to_E (cuda_cyl_complex/cylindrical complex,
cuda_complex/complex)``, 16 seam-instances of demand and ZERO of them clearing the
source seam today
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-08-30_electric/board.log``). Every
one of the 16 carries an electric deposit between the halves, so the board prices the
whole cell on the driver fact and the fitness verdict -- POINTWISE-BUILDABLE -- buys
nothing until a product on the cell carries the repair. THAT BRACKET IS WHAT THIS
MODULE IS FOR; the launch count is the smaller half of it.

13 of the 16 drive ``zero_metal_D`` inside the seam, which is the most of any D cell on
this board.

IT IS ALSO A CELL METAL CANNOT TAKE. The Metal board's 21 ``cannot_be_bound_on_metal``
instances are a pointer-count verdict against ``MAX_BUFFER_BINDINGS = 31``
(``metal_kernels/device.py:72``), and the cylindrical complex curl alone binds eleven
volumes before the constitutive half adds nine more (six E-side plus three
inverse-permittivity). CUDA HAS NO SUCH CEILING: the kernel parameter space is 32,764
bytes on sm_70+ and this signature spends well under 400 of them, so the fusion is
refused here on nothing.

=============================================================================
WHAT ONE LAUNCH PERFORMS
=============================================================================

:data:`REPLACES` is ``step_D`` (driver.py:3302) -> ``zero_metal_D`` (:3310) ->
``update_E`` (:3313), declared rather than inferred from the two slots.

THE TWO MIRROR FILLS ARE ABSENT BY REFUSAL. ``fill_symmetry_bc_D`` (:3309) and
``fill_folded_far_ghosts_D`` (:3311) do nothing on an unfolded grid, and this pair
refuses every folded one -- twice over, once through
``covers_pml_cylindrical_complex_curl``'s own clause and once in this file, so the
guarantee that makes :data:`REPLACES` honest does not rest on a clause in another
module. ``Grid`` refuses a mirror on a Dcyl cell in the first place; the refusal is
written anyway, because a fused product's REPLACES is a claim about what the launch did
and a claim may not depend on a fact nobody restated.

``zero_metal_D`` IS carried, as a register clear beside a store, so the constitutive
half reads the wiped value inside the same launch. IT IS THE OFF-DIAGONAL TABLE -- two
components per walled axis, the complement of the B side's diagonal
(:data:`_ZERO_METAL_ROWS`).

THE RADIAL PREFIX IS NOT PART OF THE WELD AND IS NOT MEANT TO BE.
``cylindrical_prefix`` ends in ``xp.cumsum`` on the array path, deliberately: CuPy's
float32 cumsum is not a sequential accumulation, so a hand-written column-serial scan
would be a DIFFERENT float32 number and this family's whole claim is bytewise identity.
It is computed before the launch exactly as the certified stand-alone
``cyl_step_D_pml_complex`` computes it, and the fusion neither adds nor removes it.

=============================================================================
THE AXIS TAIL IS PART OF THE SEAM, AND ON THE D SIDE IT HAS TWO BRANCHES
=============================================================================

The certified cylindrical curl does not end at ``pml_apply``. ``stepping.
_cylindrical_axis_zero_D`` (:560-598) runs AFTER the recurrence, and unlike the B
side's single branch it has two:

  * ``|m| = 1``: Dz on the axis row is zeroed -- THE FIELD ONLY, never ``fu_Dz``
    (:588). The B side does nothing at all here, which is a fact read off the array
    path rather than inferred from the D side being nearby.
  * ``|m| >= 2``: all three components AND their ``fu``, on rows ``[0:zero_rows]``.

A weld that captured the register at ``pml_apply`` and stopped would hand ``update_E``
the PRE-ZEROING displacement on every one of those rows -- a converged, smooth, entirely
wrong field near the axis, on exactly the rows a Dcyl run cares most about. So the carry
follows the certified STORES rather than the certified recurrence, and it follows BOTH
BRANCHES: each is anchored as a whole block (:data:`_AXIS_TAIL_M1`,
:data:`_AXIS_TAIL_M2`) so a change to either is a named failure rather than a silent
divergence, and so the three ``fu_D*`` stores in the second block are visibly NOT given
a register clear -- there is no ``fu`` register, and the constitutive half never reads
one.

=============================================================================
THE ALIASING HAZARD
=============================================================================

The curl half writes D in place and the constitutive half reads it. Binding that one
allocation twice -- once writable for the curl, once ``const __restrict__`` for the
constitutive source -- is undefined behaviour NVRTC miscompiles without a diagnostic. So
D appears EXACTLY ONCE in :data:`_SIGNATURE` and the constitutive half has no
flux-density source pointers at all.

THE THREE ``inv_eps`` POINTERS ARE DELIBERATELY NOT ``__restrict__``, lifted from the
certified ``update_E``: an isotropic run hands one allocation three times
(``fields.py:1323-1325``), and ``restrict`` on mutually aliasing arguments is a promise
the caller cannot keep.

=============================================================================
THE ONE NAME COLLISION THE SPLICE MUST RESOLVE
=============================================================================

The two certified bodies do not collide on their volume names -- the cylindrical curl
spells its own ``Dx/Dy/Dz`` and ``Hx/Hy/Hz`` while the complex constitutive template
uses ``f*``/``g*``/``w*`` -- so those are lifted untouched. They DO collide on
``kms_x/kms_y/kms_z``, which is the INTEGER sub-lattice for the D curl
(``complex_emitter.HALF_INTEGER['step_D']`` is False) and the HALF-INTEGER one for
``update_E`` (``HALF_INTEGER['E']`` is True, stepping.py:1015). THE PAIRING IS THE MIRROR
IMAGE OF THE B/H SEAM'S. That pair compiles perfectly and differs by half a cell in the
absorber profile, which is why the constitutive group is renamed ``kms_half_*`` in the
signature and in the spliced body both.

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

# THE CERTIFIED TEXT, and the predicates. ``complex_emitter``, ``cylindrical_coverage``,
# ``coverage``, ``cylindrical_prefix`` and ``in_seam_coverage`` are all CuPy-free, so
# the emitter and the predicate run at the merge bar. ``cylindrical_complex_kernels``
# imports CuPy at module scope, so its device text is reached through a defensive import
# and the emitter refuses BY NAME where it is absent (:func:`_certified_source`).
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
# The partition readers walk the syntax tree WITHOUT importing the module (they have to
# — these modules import cupy) and an annotated assignment is an ``ast.AnnAssign`` those
# readers do not match.

#: Byte-identical to the array path with a gate verdict AND a ``certification.json``
#: record behind it. A fused product's bit-identity is a claim about the WELD, which no
#: verdict on either half establishes.
#: RELEASED 2026-09-04 under BOTH float32 subnormal policies, at EVERY m class --
#: m = 0 (the third arm, landed the same day), |m| = 1 and |m| >= 2 -- by
#: ``parity/meep_gpu/gate_cuda_fused_complex_pairs.py --family cylindrical_electric``
#: on one RTX A6000: byte-identical per COMPLETE driver step over 60 steps against
#: BOTH the array path and the separately certified halves on every fixture in
#: CYLINDRICAL_SPECS (including the corpus row's own (150, 1, 300) m = 0 shape),
#: the launch counted by two instruments, the deposit legs carrying a real in-seam
#: source with the unrepaired null control diverging, and every mutation leg --
#: the m = 0 tail's own included -- scored as declared. The block is
#: ``cuda_cylindrical_fused_electric_pair_2026-09-04`` in ``certification.json``; the flip was made BEFORE
#: the run so the gate measured the bytes that ship.
CERTIFIED_KERNELS = (
    "fused_electric_pair_pml_cyl_complex",
)
#: EMPTY since 2026-09-04; what emptied it was the run above.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its fused launch with the deposit repair? TRUE, wired
#: through ``fused_pairs._install_fused_pair`` -- the flag and the wiring change together
#: or not at all (``deposit_repair.py:221-226``).
#:
#: THE FOLD CLAUSES IN ``deposit_repair._folded_seam_reasons`` ARE VACUOUS ON EVERY ROW
#: THIS ADMITS, including the one that names the cylindrical r = 0 axis: that clause
#: fires only on a FOLDED grid, and this predicate refuses every folded grid outright.
#: The two D-only clauses in ``deposit_repair.repairable`` (an off-diagonal chi1inv row,
#: an instantaneous chi2/chi3) are refused by the constitutive half's own predicate
#: before this predicate reaches the seam clause. What the repair inverts here is
#: ``(f + kps*fw_fresh) - kms*fw_prev`` with ``fw_fresh`` being ``D * inv_eps``
#: (``deposit_repair.SEAMS["D"]``) over complex64 storage, which is word for word what
#: ``constitutive_apply`` computes.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_cylindrical_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_electric_pair_pml_cyl_complex"

#: The driver passes ONE launch of this kernel performs, in driver order.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ cf pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the helper names "
            "and returns the word pair it already computed. The expression, its "
            "parenthesisation and the store to f are unchanged."},
    {"line": "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), "
             "sinv_u));",
     "became": "    cf value = mul_field_left(cf_sub(cf_add(a, fu_new), fprev), "
               "sinv_u);\n    cf_store(f, idx, value);\n    return value;",
     "why": "same right-hand side, same tree, same store; the result is additionally "
            "named so it can be returned."},
    {"line": "        pml_apply(Dx, fu_Dx, idx, curl, ...);",
     "became": "        d0 = pml_apply_reg(Dx, fu_Dx, idx, curl, ...);",
     "why": "capture the register. Three lines, one per component; the argument lists "
            "are untouched. Three 'cf d* = cf_zero();' declarations are hoisted above "
            "the certified braced blocks because a value declared inside one does not "
            "outlive it."},
    {"line": "    if (m_class == 1 && i == 0) { cf_store(Dz, idx, cf_zero()); }   "
             "(stepping._cylindrical_axis_zero_D, the |m| = 1 branch)",
     "became": "    if (m_class == 1 && i == 0) { d2 = cf_zero(); cf_store(Dz, idx, "
               "d2); }",
     "why": "THE |m| = 1 AXIS TAIL, CARRIED. It runs AFTER the recurrence and zeroes "
            "Dz -- THE FIELD ONLY, never fu_Dz (stepping.py:619) -- on the axis row. A "
            "weld that captured at pml_apply and stopped would hand update_E the "
            "pre-zeroing displacement there. The guard, the row test and the store are "
            "untouched; the register is cleared beside the store."},
    {"line": "    if (m_class == 2 && i < zero_rows) { six cf_store(..., cf_zero()); }"
             "   (stepping._cylindrical_axis_zero_D, the |m| >= 2 branch)",
     "became": "the same block with 'd* = cf_zero();' beside each of the THREE D "
               "stores; the three fu_D stores are untouched",
     "why": "THE |m| >= 2 AXIS TAIL, CARRIED. Same reason as the |m| = 1 branch. The "
            "three fu_D stores get NO register clear because there is no fu register "
            "and the constitutive half never reads one -- writing one would be device "
            "text with nothing behind it."},
    {"line": "    if (m_class == 0 && i == 0) { cf_store(Dz, idx, cf_add(cf_load(Dz, idx), "
             "mul_coefficient_left(axis_coef, cf_load(Hy, idx)))); cf_store(Dy, idx, "
             "cf_zero()); }   (stepping._cylindrical_axis_zero_D, the m = 0 branch, "
             "2026-09-04)",
     "became": "    if (m_class == 0 && i == 0) { d2 = cf_add(cf_load(Dz, idx), "
               "mul_coefficient_left(axis_coef, cf_load(Hy, idx))); cf_store(Dz, idx, "
               "d2); d1 = cf_zero(); cf_store(Dy, idx, d1); }",
     "why": "THE m = 0 AXIS TAIL, CARRIED. It runs AFTER the recurrence: the on-axis "
            "Dz POST-ADD of the host-rounded 4*Courant times the raw stored Hp "
            "(stepping.py:585) and Dp = 0 (:586). A weld that captured at pml_apply "
            "and stopped would hand update_E the pre-add Dz and the pre-zeroing Dy on "
            "the axis row. The load of Dz reads the word pml_apply_reg stored one "
            "statement earlier in the SAME lane (a float32 round trip is the "
            "identity), the add, the multiply and its operand order are the certified "
            "line's; the two registers follow the two stores."},
    {"line": "    cf s0 = cf_load(g0, idx);",
     "became": "    cf s0 = d0;",
     "why": "THE SEAM. The certified constitutive body opens each component with a "
            "reload of the flux density the curl just stored; this reads the register "
            "instead, which is also what lets D be bound exactly once. The "
            "inverse-permittivity multiply on the next line and its operand order are "
            "untouched."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i]);",
     "became": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_half_x[i]);",
     "why": "THE SUB-LATTICE RENAME AND NOTHING ELSE. Both halves ship a vector spelled "
            "kms_a on DIFFERENT Yee sub-lattices -- INTEGER for the D curl, "
            "HALF-INTEGER for the E constitutive, the mirror image of the B/H seam's "
            "pairing -- and letting one shadow the other is a half-cell error in the "
            "absorber profile, not a compile failure. The two bodies do NOT collide on "
            "f*/g*/w*, so those keep their certified spelling."},
    {"line": "        Dy[base] = 0.0f; Dz[base] = 0.0f;   "
             "(in_seam_passes._zero_metal_D_kernel_code)",
     "became": "    if (wall_x && i == 0) { d1 = cf_zero(); cf_store(Dy, idx, d1); "
               "d2 = cf_zero(); cf_store(Dz, idx, d2); }",
     "why": "THE ONE RE-SPELLING, and it is launch geometry rather than arithmetic. The "
            "certified pass walks a FACE with face_geometry(); this launch walks the "
            "VOLUME and already holds i/j/k. Same OFF-DIAGONAL table (two components "
            "per wall), same (+0.0f, +0.0f) word pair."},
    {"line": "    cf d2 = cf_zero();",
     "became": "    cf d2 = cf_zero();\n"
               "    cf pre_w_0 = cf_load(w0, idx);   // ... twelve lines, 24 words",
     "why": "THE OWN-CELL HOIST. Every word this thread reads at its own idx -- "
            "w*, f* (E), fu_D*, D* -- is issued before the kernel's first store. "
            "UNGUARDED, because the certified loads are: this family refuses every "
            "folded grid, so no fill images a cell and no other block writes any "
            "word at this thread's idx in this launch. The only stores this thread "
            "makes to those words before their certified load points are none: the "
            "axis tails and the wall clears store D/fu_D AFTER pml_apply_reg_pre "
            "has consumed them; the m = 0 tail's certified reload of Dz after that "
            "store is left as it is. hoisted_loads()."},
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
    {"line": "        d0 = pml_apply_reg(Dx, fu_Dx, idx, curl, ...);",
     "became": "        d0 = pml_apply_reg_pre(Dx, fu_Dx, idx, curl, ..., "
               "pre_fu_0, pre_d_0);",
     "why": "the three own-cell curl calls take their preloaded words; the "
            "argument list is otherwise the certified one."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_half_x[i]);",
     "became": "    constitutive_apply_pre(f0, w0, idx, s0, pre_w_0, pre_e_0, "
               "kps_x[i], kms_half_x[i]);",
     "why": "the three constitutive statements take their preloaded words, in the "
            "composer (hoisted_constitutive_body), not in "
            "certified_constitutive_body(), which stays the certified lift."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body", "certified_curl_body",
    "constitutive_apply_pre_source", "covers_cylindrical_fused_electric_pair",
    "hoisted_constitutive_body", "hoisted_loads", "pml_apply_reg_pre_source",
    "cylindrical_fused_electric_pair_prelude",
    "cylindrical_fused_electric_pair_source",
    "cylindrical_fused_electric_pair_tables", "device_sources",
    "inverse_epsilon_bindings", "launch_cylindrical_fused_electric_pair",
    "run_cylindrical_fused_electric_pair", "zero_metal_carry",
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
_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")

#: ``stepping._zero_metal``'s OFF-DIAGONAL table for the D side, as (flag, coordinate,
#: the two component indices wiped there). SPELLED HERE rather than imported from
#: anything B-shaped: the B family's table is the DIAGONAL, and a pair that reused it
#: would clear one wrong component and leave two right ones standing on every walled
#: run. ``fields.IYEE_SHIFTS`` gives Dx (1,0,0), Dy (0,1,0), Dz (0,0,1) whatever the
#: coordinate system, so a D component has Yee shift 0 on the OTHER TWO axes -- on a
#: Dcyl grid that is (phi, z) for Dr, (r, z) for Dp and (r, phi) for Dz.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("wall_x", "i", ("1", "2")),
    ("wall_y", "j", ("0", "2")),
    ("wall_z", "k", ("0", "1")),
)

#: ``stepping._cylindrical_axis_zero_D``'s TWO branches, exactly as the certified curl
#: emits them, and their carried rewrites. Anchored as WHOLE BLOCKS rather than by line:
#: ``cf_store(Dz, idx, cf_zero());`` appears in BOTH branches, so a per-line anchor
#: could not name which one it had rewritten, and the ``fu_D*`` stores in the second
#: branch must visibly survive untouched.
_AXIS_TAIL_M1 = (
    "    if (m_class == 1 && i == 0) {\n"
    "        cf_store(Dz, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M1_CARRIED = (
    "    // THE |m| = 1 AXIS TAIL, CARRIED. stepping._cylindrical_axis_zero_D (:588)\n"
    "    // zeroes Dz on the axis row -- THE FIELD ONLY, never fu_Dz -- after the\n"
    "    // recurrence, so the register has to follow the store or update_E would\n"
    "    // read the pre-zeroing displacement there.\n"
    "    if (m_class == 1 && i == 0) {\n"
    "        d2 = cf_zero(); cf_store(Dz, idx, d2);\n"
    "    }\n")
_AXIS_TAIL_M2 = (
    "    if (m_class == 2 && i < zero_rows) {\n"
    "        cf_store(Dx, idx, cf_zero());\n"
    "        cf_store(Dy, idx, cf_zero());\n"
    "        cf_store(Dz, idx, cf_zero());\n"
    "        cf_store(fu_Dx, idx, cf_zero());\n"
    "        cf_store(fu_Dy, idx, cf_zero());\n"
    "        cf_store(fu_Dz, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M0 = (
    "    if (m_class == 0 && i == 0) {\n"
    "        cf_store(Dz, idx, cf_add(cf_load(Dz, idx),\n"
    "                                 mul_coefficient_left(axis_coef, cf_load(Hy, idx))));\n"
    "        cf_store(Dy, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M0_CARRIED = (
    "    // THE m = 0 AXIS TAIL, CARRIED (2026-09-04). stepping._cylindrical_axis_zero_D\n"
    "    // (:583-587): the on-axis Dz POST-ADD of the host-rounded 4*Courant times the\n"
    "    // raw stored Hp, then Dp = 0 -- both AFTER the recurrence, so both registers\n"
    "    // follow their stores or update_E would read the pre-add Dz and the\n"
    "    // pre-zeroing Dy on the axis row. The Dz load is the word this lane stored\n"
    "    // one statement earlier (a float32 round trip is the identity).\n"
    "    if (m_class == 0 && i == 0) {\n"
    "        d2 = cf_add(cf_load(Dz, idx),\n"
    "                    mul_coefficient_left(axis_coef, cf_load(Hy, idx)));\n"
    "        cf_store(Dz, idx, d2);\n"
    "        d1 = cf_zero(); cf_store(Dy, idx, d1);\n"
    "    }\n")
_AXIS_TAIL_M2_CARRIED = (
    "    // THE |m| >= 2 AXIS TAIL, CARRIED. The three fu_D stores keep the certified\n"
    "    // spelling and get NO register clear: there is no fu register and the\n"
    "    // constitutive half never reads one.\n"
    "    if (m_class == 2 && i < zero_rows) {\n"
    "        d0 = cf_zero(); cf_store(Dx, idx, d0);\n"
    "        d1 = cf_zero(); cf_store(Dy, idx, d1);\n"
    "        d2 = cf_zero(); cf_store(Dz, idx, d2);\n"
    "        cf_store(fu_Dx, idx, cf_zero());\n"
    "        cf_store(fu_Dy, idx, cf_zero());\n"
    "        cf_store(fu_Dz, idx, cf_zero());\n"
    "    }\n")

#: The fused entry point. The ONLY hand-written device text in this module, and it is a
#: signature. D appears EXACTLY ONCE -- see the module docstring's aliasing section.
_SIGNATURE = r'''
extern "C" __global__ void fused_electric_pair_pml_cyl_complex(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 D.
    // The curl half writes it and the constitutive half reads it from a register;
    // binding it a second time as a const __restrict__ source would be two restrict
    // pointers to one allocation, which is UB NVRTC miscompiles silently.
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    // The radial prefix (nx + 1 rows) and the two bound i*m/r coefficient rows.
    const float* __restrict__ pfx,
    const float* __restrict__ imr0, const float* __restrict__ imr2,
    // E and its constitutive history. The certified constitutive template's own
    // names: the two bodies do not collide on these.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    // NOT __restrict__, and lifted from the certified update_E rather than decided
    // here: an isotropic run hands the same device pointer three times
    // (fields.py:1323-1325). They stay FLOAT32 under complex storage and are indexed
    // by the COMPLEX CELL index, never word-doubled (stepping.py:41-50).
    const float* inv_eps_0, const float* inv_eps_1, const float* inv_eps_2,
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx,
    float minus_dtdx, float inc_re, float inc_im,
    // The m = 0 on-axis Dz multiplicand, host-rounded float32(4.0 * dtdx)
    // (cylindrical_complex_kernels.axis_coefficient); read under m_class == 0 only.
    float axis_coef,
    // The D curl's INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The E constitutive's HALF-INTEGER coefficients. Its kms is renamed kms_half_*:
    // the two sub-lattices collide on the bare name in one scope, and a shadow there
    // is a half-cell error in the absorber profile, not a compile failure. THE
    // PAIRING IS THE MIRROR IMAGE OF THE B/H SEAM'S.
    const float* __restrict__ kps_x, const float* __restrict__ kms_half_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_half_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_half_z,
    int bc_x, int bc_y, int bc_z,
    // This family carries NO Bloch phase -- the predicate refuses one -- so every
    // flag is 0 and cshift_* never multiplies. The trio is passed anyway because
    // it is the certified helper's own signature.
    int ph_x, int ph_y, int ph_z,
    int m_class, int zero_rows,
    // zero_metal_D's three walled-axis flags, from in_seam_coverage.zero_metal_axes.
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
            f"{name} no longer begins with the prelude this module lifts separately; "
            f"the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, not 1; "
            f"the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure."""
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line starting "
            f"{prefix!r}); this family LIFTS that line rather than retyping it and "
            f"cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the lift "
            f"of {what} would take an arbitrary one")
    return matches[0]


def _certified_source() -> Any:
    """Refuse the emitter BY NAME on a host where the cylindrical text is unreachable.

    THE SPLICE IS THE LIFT, so a missing half is not a degraded emit -- there is nothing
    to emit. Raising here keeps that fact one frame from the caller instead of surfacing
    as ``'NoneType' object has no attribute`` inside a string operation.
    :func:`covers_cylindrical_fused_electric_pair` needs none of it and still answers.
    """
    if cylindrical_complex_kernels is None:
        raise RuntimeError(
            "cylindrical_complex_kernels is not importable on this host (it imports "
            "CuPy at module scope), so there is no certified cylindrical text to "
            "splice. covers_cylindrical_fused_electric_pair needs none of it and still "
            "answers")
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
# fused kernel issues every own-cell word it reads -- fu_D*, D*, w*, f* -- before its
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
        lines.append(f"    cf pre_e_{register} = cf_load(f{register}, idx);")
    for register, target in zip(_CARRIED, _TARGETS):
        lines.append(f"    cf pre_fu_{register} = cf_load(fu_{target}, idx);")
        lines.append(f"    cf pre_d_{register} = cf_load({target}, idx);")
    return "\n".join(lines) + "\n"


def cylindrical_fused_electric_pair_prelude(arm) -> str:
    """The certified prelude with ``pml_apply`` turned into a value.

    ONE prelude for both halves, and that is a fact about the two families rather than a
    convenience: ``cylindrical_complex_source`` builds its source from the CERTIFIED
    complex family's ``_HEAD``/``_ARM_SOURCE``/``_TAIL`` plus its own
    ``_CYLINDRICAL_HELPERS``, so the constitutive half's prelude is a strict prefix of
    the curl half's.
    """
    prelude = _curl_prelude(arm)
    if prelude.count(_PML_APPLY_SIGNATURE) != 1:
        raise AssertionError(
            f"the certified prelude declares pml_apply "
            f"{prelude.count(_PML_APPLY_SIGNATURE)} times, not once; the value rewrite "
            f"has no anchor")
    if prelude.count(_PML_APPLY_STORE) != 1:
        raise AssertionError(
            "the certified pml_apply no longer closes with the store this module turns "
            "into a named value; the arithmetic may have moved")
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
    """``cyl_step_D_pml_complex``'s body, lifted, with the three registers captured.

    FOUR CAPTURE SITES PER RUN, not one: the recurrence's value is taken at
    ``pml_apply``, and every branch of the per-m axis tail then OVERWRITES D after it
    -- Dz alone at ``|m| = 1``, all three on rows ``[0:zero_rows]`` at ``|m| >= 2``, and
    (since 2026-09-04) Dz post-added and Dy zeroed at ``m = 0``. All are followed.
    """
    module = _certified_source()
    body = _split_body(module.cylindrical_complex_source("step_D", arm),
                       _curl_prelude(arm), "cylindrical_complex_source('step_D')")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the carried "
            "registers have nowhere to be declared")
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END,
        "\n    // The three carried registers. Declared HERE because each\n"
        "    // certified component below is a braced scope, and a value declared\n"
        "    // inside one does not outlive it.\n"
        "    cf d0 = cf_zero();\n"
        "    cf d1 = cf_zero();\n"
        "    cf d2 = cf_zero();\n"
        "\n",
        hoisted_loads(),
        tail,
    ))
    for register, target in zip(_CARRIED, _TARGETS):
        old = f"pml_apply({target}, fu_{target}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} {body.count(old)} "
                f"times, not once; the capture has no anchor")
        body = body.replace(
            old, f"d{register} = pml_apply_reg({target}, fu_{target}, ", 1)
        prefix = f"        d{register} = pml_apply_reg({target}, fu_{target}, "
        call = _line_starting(body, prefix, f"target {target}'s captured pml_apply")
        if not call.endswith(");"):
            raise AssertionError(
                f"the certified curl body no longer closes {target}'s pml_apply on one "
                f"line ({call!r})")
        # THE HOIST: the same statement, taking its two preloaded own-cell words.
        body = body.replace(call, call.replace(
            f"d{register} = pml_apply_reg(", f"d{register} = pml_apply_reg_pre(", 1)[:-2]
            + f", pre_fu_{register}, pre_d_{register});", 1)
    if "pml_apply(" in body or "pml_apply_reg(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be written "
            "to global memory and never read into the seam")

    # THE AXIS TAIL, BOTH BRANCHES, each anchored as a whole block. A per-line anchor
    # cannot be used here: `cf_store(Dz, idx, cf_zero());` appears in BOTH branches, so
    # a line match would rewrite an arbitrary one and report success.
    for name, certified, carried in (
            ("the |m| = 1 branch", _AXIS_TAIL_M1, _AXIS_TAIL_M1_CARRIED),
            ("the |m| >= 2 branch", _AXIS_TAIL_M2, _AXIS_TAIL_M2_CARRIED),
            ("the m = 0 branch", _AXIS_TAIL_M0, _AXIS_TAIL_M0_CARRIED)):
        if body.count(certified) != 1:
            raise AssertionError(
                f"the certified curl body carries {name} of "
                f"stepping._cylindrical_axis_zero_D {body.count(certified)} times, not "
                f"once; the register clear has no anchor and update_E would read the "
                f"pre-zeroing displacement on the rows it writes")
        body = body.replace(certified, carried, 1)
    return body


def zero_metal_carry() -> str:
    """``zero_metal_D`` (driver.py:3310), carried between the two halves.

    ``stepping._zero_metal`` writes the integer 0 into stored cell 0 of every component
    whose Yee shift on a walled axis is 0 -- for D that is the OFF-DIAGONAL, TWO
    components per wall. The register is cleared BESIDE the store, because the
    constitutive half below reads the register and not the volume.

    IT RUNS AFTER THE AXIS TAIL, which is the driver's order and not a choice: the axis
    tail is part of ``step_D`` (stepping.py:485-486, inside the sub-step) and
    ``zero_metal_D`` is a separate driver pass at :3310. A wall on the r axis and the
    ``|m|`` axis rows both write some of the same cells to zero, so the composition is
    order-independent there; stating the order anyway is what keeps that a fact rather
    than an accident.
    """
    lines = [
        "\n    // stepping._zero_metal (:2206-2245) / in_seam_passes.zero_metal_D,\n"
        "    // CARRIED, and AFTER the certified axis tail above -- which is the\n"
        "    // driver's order (the tail is inside step_D, this is the pass at\n"
        "    // driver.py:3310). The OFF-DIAGONAL: a D component sits ON the wall of\n"
        "    // each axis whose Yee shift is 0, which for D is the other two.\n"
    ]
    for flag, coordinate, registers in _ZERO_METAL_ROWS:
        stores = " ".join(
            f"d{register} = cf_zero(); "
            f"cf_store({_TARGETS[int(register)]}, idx, d{register});"
            for register in registers)
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {stores} }}\n")
    return "".join(lines)


def certified_constitutive_body(arm) -> str:
    """``update_E_pml_complex_bloch``'s body, lifted, reading the registers.

    Drops the second index decomposition (the curl body above already declares ``idx``,
    ``i``, ``j`` and ``k`` and its bounds guard has already returned), turns the three
    source reloads into the seam, and renames the constitutive coefficients'
    sub-lattice.
    """
    body = _split_body(complex_emitter.complex_source("update_E", arm),
                       _constitutive_prelude(arm), "complex_source('update_E')")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own line; the "
            "duplicate decomposition cannot be identified")
    tail = body.split(_DECODE_END, 1)[1]

    for register in _CARRIED:
        old = f"    cf s{register} = cf_load(g{register}, idx);\n"
        if tail.count(old) != 1:
            raise AssertionError(
                f"the certified constitutive body loads s{register} {tail.count(old)} "
                f"times, not once; the seam has no anchor")
        tail = tail.replace(
            old,
            f"    cf s{register} = d{register};   // THE SEAM: the register the curl "
            f"half just stored\n", 1)
    if "cf_load(g" in tail:
        raise AssertionError(
            "a source reload survived the seam rewrite; the constitutive half would "
            "need D bound a second time, which is the aliasing hazard this signature "
            "exists to avoid")

    # THE INVERSE-PERMITTIVITY PRODUCTS ARE ASSERTED, NOT EDITED. They are the one part
    # of this body that has no counterpart on the H seam, so a change to them would be
    # exactly the kind of drift that compiles: the line is READ and required to be the
    # certified one, and nothing is substituted into it.
    for register in _CARRIED:
        expected = (f"    s{register} = mul_field_left(s{register}, "
                    f"inv_eps_{register}[idx]);")
        line = _line_starting(tail, f"    s{register} = mul_field_left(",
                              f"target {register}'s inverse-permittivity product")
        if line != expected:
            raise AssertionError(
                f"target {register}'s inverse-permittivity product is {line!r}, not the "
                f"certified {expected!r}; this family binds inv_eps_* on the strength "
                f"of that line and does not edit it")

    for register, axis in zip(_CARRIED, ("x", "y", "z")):
        prefix = f"    constitutive_apply(f{register}, w{register}, idx, s{register}, "
        call = _line_starting(tail, prefix, f"target {register}'s constitutive_apply")
        coordinate = "ijk"[int(register)]
        expected = (f"{prefix}kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);")
        if call != expected:
            raise AssertionError(
                f"target {register}'s constitutive_apply is {call!r}, not the certified "
                f"{expected!r}; the sub-lattice rename would be applied to an argument "
                f"list this module has not read")
        tail = tail.replace(
            call, call.replace(f"kms_{axis}[", f"kms_half_{axis}[", 1), 1)
    for register, axis in zip(_CARRIED, ("x", "y", "z")):
        coordinate = "ijk"[int(register)]
        if f"kms_{axis}[{coordinate}])" in tail:
            raise AssertionError(
                f"kms_{axis} survived the sub-lattice rename; it is the INTEGER vector "
                f"in the curl body and the HALF-INTEGER one here, and the two would "
                f"collide on the bare name")
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
                    f"s{register}, pre_w_{register}, pre_e_{register}, ", 1), 1)
    if "constitutive_apply(" in body:
        raise AssertionError(
            "a constitutive_apply call survived the hoist rewrite; the retired helper "
            "would be called with no declaration")
    return body


def cylindrical_fused_electric_pair_source(arm) -> str:
    """The whole fused kernel for one expansion arm: prelude, signature, body."""
    source = "".join((
        cylindrical_fused_electric_pair_prelude(arm),
        _SIGNATURE,
        certified_curl_body(arm),
        zero_metal_carry(),
        hoisted_constitutive_body(arm),
        "}\n",
    ))
    # PURE ASCII IS A COMPILE REQUIREMENT on this family, not a style rule; checked here
    # so a mutation leg that inserts a non-ASCII character is refused at emission with
    # the reason instead of at NVRTC three frames away.
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by arm name -- for a digest."""
    return {name: cylindrical_fused_electric_pair_source(name)
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

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING: the source is part of the
    memo key, and a gate mutates this family by monkeypatching the emitter. A source
    memoized at first call would hand back the pre-mutation string forever.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    code = cylindrical_fused_electric_pair_source(arm)
    key = compile_cache.kernel_cache_key(
        f"{name}_arm{complex_emitter.normalized_expansion(arm)}", True,
        _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_cylindrical_fused_electric_pair(fields: Any, pml: Any, grid: Any,
                                           sources: Any = None, license: Any = None,
                                           subnormal_policy: Any = None
                                           ) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``zero_metal_D`` -> ``update_E``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own certified
    predicate refuses is refused here with that half's reason, prefixed so a reader can
    tell which side said it. What this predicate ADDS is the seam clauses: the source
    slot, the injection ROUTE, the two UNCARRIED fills, and the grid's ability to say
    which axes are walled.
    """
    covered, reason = covers_pml_cylindrical_complex_curl(
        fields, pml, grid, "step_D", license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "E", license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE TWO FILLS ARE REFUSED, NOT CARRIED. The curl half already refuses every folded
    # grid and Grid refuses a mirror on a Dcyl cell before that; the clause is stated
    # HERE anyway, because REPLACES is a claim about what one launch performed and it
    # may not rest on a clause in another module that a future device verdict could
    # licence away.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_D and "
                       "fill_folded_far_ghosts_D run inside it")
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry or any(folded):
        return False, ("a mirror plane is active: fill_symmetry_bc_D "
                       "(driver.py:3309) and fill_folded_far_ghosts_D (:3311) run "
                       "inside this seam and this pair carries neither")

    # THE INJECTION ROUTE, REFUSED BY NAME, and it is a D-seam clause with no B-seam
    # counterpart. `_inject_electric_through_conductivity` (driver.py:3305) snapshots the
    # D components, injects, and rescales only the increment by `condinv` (MEEP
    # step.cpp:294-317). BOTH COMPLEX HALVES ADMIT A CONDUCTIVITY BY NAME -- it is read
    # in `stepping._apply_curl` and nowhere else -- so unlike the real electric pair
    # this refusal is not inherited from a half and has to be made here. The deposit
    # repair has never been measured against the scaled route.
    if getattr(fields, "has_conductivity", False):
        return False, ("the engine carries a conductivity, so the driver deposits "
                       "this seam's electric sources through "
                       "_inject_electric_through_conductivity (driver.py:3305), which "
                       "rescales the increment by condinv; the deposit repair has no "
                       "verdict on that route")

    # THE SOURCE SEAM, CARRIED RATHER THAN REFUSED. An ELECTRIC source is injected
    # BETWEEN the two halves (driver.py:3305/:3308). IGNORANCE IS NEVER AN EMPTY SET:
    # `Fields` does not hold the source list, so a predicate that inferred "no sources"
    # from not being told would be the over-covering this clause prevents.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3308)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

    # zero_metal_D IS CARRIED, so the grid must be able to say which axes are walled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_D cannot be "
                           f"carried in registers")
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE STORED SHAPE THE KERNEL INDEXES MUST BE THE ONE THE WALL PLANE AND THE PREFIX
    # ARE DERIVED FROM. `nx, ny, nz` come from `fields.Dx.shape`; the wall is stored cell
    # 0 of a `grid.stored_cells` axis and the prefix has `nx + 1` rows.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Dx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks Dx.shape "
                       f"{extents}; the wall plane and the radial prefix are derived "
                       f"from the first and indexed into the second")

    # THE THREE INVERSE-PERMITTIVITY VOLUMES MUST BE ASKABLE AND MUST MATCH THE SHAPE
    # THIS LAUNCH WALKS. The constitutive half's own predicate already checks them
    # against `_grid_facts`' shape; this asks the same question against the extents THIS
    # LAUNCH walks (`fields.Dx.shape`), which is the array the flat idx indexes.
    accessor = getattr(fields, "inverse_epsilon_for", None)
    if not callable(accessor):
        return False, ("fields does not expose inverse_epsilon_for; update_E's three "
                       "inverse-permittivity volumes cannot be bound")
    for component in _INVERSE_EPSILON_COMPONENTS:
        try:
            volume = accessor(component)
        except Exception as exc:  # noqa: BLE001
            return False, (f"fields.inverse_epsilon_for({component!r}) raised: "
                           f"{type(exc).__name__}: {exc}")
        if volume is None:
            return False, f"fields.inverse_epsilon_for({component!r}) is None"
        shape = tuple(int(n) for n in getattr(volume, "shape", ()))
        if shape != extents:
            return False, (f"inverse_epsilon_for({component!r}) has shape {shape} but "
                           f"the launch walks {extents}; the kernel indexes it with "
                           f"this launch's flat index")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The nine complex volumes the curl half binds from ``Fields``, in SIGNATURE ORDER. The
#: prefix and the two i*m/r rows are derived per launch and per configuration
#: respectively, so they are bound separately.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz",
    "fu_Dx", "fu_Dy", "fu_Dz",
    "Hx", "Hy", "Hz",
)
_CONSTITUTIVE_BINDINGS: Tuple[str, ...] = (
    "Ex", "Ey", "Ez",
    "f_w_Ex", "f_w_Ey", "f_w_Ez",
)

#: The three E components whose inverse permittivity ``update_E`` multiplies by, in
#: signature order.
_INVERSE_EPSILON_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def cylindrical_fused_electric_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads.

    THE SUB-LATTICE IS DECIDED HERE, not by the caller: while it was an argument, every
    call site was one more place the pairing could be got backwards, and backwards is a
    half-cell error rather than a crash. The D curl reads the INTEGER positions and
    ``update_E`` the HALF-INTEGER ones -- the opposite pairing to
    :mod:`.cylindrical_fused_magnetic_pair`'s.
    """
    if cylindrical_complex_kernels is None or complex_pml_kernels is None:
        raise RuntimeError(
            "the CuPy-backed table builders are not importable on this host")
    return {"curl": cylindrical_complex_kernels.cylindrical_complex_curl_tables(
                pml, False),
            "constitutive": complex_pml_kernels.complex_constitutive_tables(
                pml, True)}


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three per-component inverse-permittivity volumes, in signature order.

    ``Fields.inverse_epsilon_for`` and never ``fields.inv_eps``, which is the Ez view
    (fields.py:1259-1260).
    """
    return tuple(  # type: ignore[return-value]
        fields.inverse_epsilon_for(component)
        for component in _INVERSE_EPSILON_COMPONENTS)


def assert_disjoint_bindings(fields: "Fields", tables: Dict[str, Dict[str, Any]],
                             prefix: Any, imr_rows: Sequence[Any]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes.

    ONCE PER FROZEN CONFIGURATION, OFF THE LAUNCH PATH. The shared flux density is bound
    once by construction -- the signature has no second D group -- so what is left to
    check is that no OTHER two restrict arguments are the same allocation. THE PREFIX
    AND THE TWO i*m/r ROWS ARE INCLUDED, and they are the reason this check takes
    arguments the Cartesian sibling's does not: the prefix is derived from Hp and a
    scratch buffer aliasing a field volume would be UB reached through a route no
    signature inspection would find.

    THE THREE inv_eps VOLUMES ARE CHECKED DIFFERENTLY AND DELIBERATELY. They are not
    ``__restrict__``, so they may alias EACH OTHER -- an isotropic run hands one pointer
    three times. What they may NOT do is alias a restrict-qualified argument.

    Returns the number of distinct allocations checked.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []

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
    for component, volume in zip(_INVERSE_EPSILON_COMPONENTS,
                                 inverse_epsilon_bindings(fields)):
        pointer = int(volume.data.ptr)
        if pointer in bound:
            collisions.append(
                f"inv_eps_{component} and {bound[pointer]} are the same allocation, "
                f"and the second is bound __restrict__")
    if collisions:
        raise ValueError(
            "the cylindrical fused electric pair binds every field and table argument "
            "__restrict__, and these arguments alias, which is undefined behaviour "
            "NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_cylindrical_fused_electric_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]], prefix: Any,
        imr_rows: Sequence[Any], boundary_codes: Sequence[Any],
        increment_scalars: Sequence[Any], m_class_code: int, zero_rows_count: int,
        walls: Sequence[int], dtdx: float, arm: Any,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional: a gate compiles a deliberately
    broken copy of the shipped source and hands it here. A launcher that could not be
    handed its own kernel could not arm a single mutation.

    Returns the launch geometry rather than ``None`` so a gate can assert that something
    was actually launched.
    """
    word_view = _certified_source().word_view
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    minus_dtdx, (inc_re, inc_im) = increment_scalars
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(
        word_view(getattr(fields, name)) for name in _FIELD_BINDINGS
    ) + (word_view(prefix),) + tuple(word_view(row) for row in imr_rows) + tuple(
        word_view(getattr(fields, name)) for name in _CONSTITUTIVE_BINDINGS
    ) + tuple(inverse_epsilon_bindings(fields)) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
        np.float32(minus_dtdx), np.float32(inc_re), np.float32(inc_im),
        # THE m = 0 ON-AXIS Dz MULTIPLICAND, derived from the SAME dtdx the kernel is
        # handed (cylindrical_complex_kernels.axis_coefficient) and bound at every m;
        # the kernel reads it only under m_class == 0.
        np.float32(_certified_source().axis_coefficient(dtdx)),
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


def run_cylindrical_fused_electric_pair(
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

    THE PREFIX AND THE i*m/r ROWS ARE THE ``step_D`` ONES. ``cylindrical_prefix(fields,
    "step_D")`` sums Hp and ``imr_rows_for("step_D", ...)`` binds the D-side coefficient
    rows; handing this the ``step_B`` pair compiles, runs, and is a different engine.
    """
    module = _certified_source()
    covered, reason = covers_cylindrical_fused_electric_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = cylindrical_fused_electric_pair_tables(pml)
    m = int(grid.m)
    if imr_rows is None:
        imr_rows = module.imr_rows_for("step_D", grid.xp, m, dtdx,
                                       int(fields.Dx.shape[0]), fields.Dx.dtype)
    if prefix is None:
        prefix = cylindrical_prefix(fields, "step_D", scratch=scratch)
    assert_disjoint_bindings(fields, tables, prefix, imr_rows)
    return launch_cylindrical_fused_electric_pair(
        fields, tables, prefix, imr_rows,
        module.cylindrical_complex_boundary_codes(grid),
        module.axis_increment_scalars(m, dtdx), module.m_class(m),
        module.zero_rows(m, bool(grid.accurate_fields_near_cylorigin)),
        zero_metal_axes(grid), dtdx, arm, kernel)
