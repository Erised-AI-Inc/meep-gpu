"""The REAL hand-CUDA fused pair on the D/E seam: ``step_D`` -> ``zero_metal_D`` -> ``update_E``.

THE FIRST PRODUCT THIS TRACK HAS EVER PUT ON THE ELECTRIC SEAM, and it is what the
board says is worth building next. The three products that shipped before it all sit
on ``B_to_H``; ``D_to_E`` carries 186 seam-instances of which the board prices only 29
as reachable, because 157 are BLOCKED BY THE INJECTION -- the driver deposits the
electric sources between ``step_D`` (driver.py:3302) and ``update_E`` (:3313), and one
launch cannot be on both sides of a deposit
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-08-30_armsrows/fusion_matrix_cuda.json``,
``source_seam_ceiling``). That ceiling is not a property of the seam: it is a property
of there being NO D-side product that brackets its launch. ``source_seam_ceiling``
lists three ``deposit_carrying_products`` and every one is a ``B_to_H`` row.

This module is the D-side entry in that list.

=============================================================================
THE CELL, AND WHY IT IS THE ONE TO TAKE
=============================================================================

``D_to_E (cuda_curl/PML, cuda_constitutive/ordinary)`` -- the same two arms the
shipped :mod:`.fused_magnetic_pair` implements one seam earlier -- carries 79 corpus
rows, the largest D-side cell on the board by a factor of four, and the board prices
it POINTWISE-BUILDABLE from ``constitutive_kernels.py``'s own device text: every read
``update_E_pml_real`` makes of the flux density lands on the cell the thread just
wrote, so the weld is a register hand-off and needs no barrier.

Of those 79, four clear the injection on the driver fact alone and 75 do not. What
this product adds is not a wider predicate: it is the BRACKET
(:data:`CARRIES_DEPOSIT_REPAIR`), through which the deposit points are saved
immediately before the launch and recomputed after the driver has injected, filled
symmetry and cleared walls.

ALL 79 SINCE 2026-08-31. Until then 53 of them were FOLDED and this pair refused every
folded grid by name, serving 26 -- the largest single gap the hand-CUDA board carried,
and a gap that was one clause wide: every one of the 53 was refused by the fold clause
and by nothing else (measured against the board's own census). The OWNERSHIP INVERSION
below is what closed it.

=============================================================================
WHAT ONE LAUNCH PERFORMS, AND WHAT IT DOES NOT
=============================================================================

:data:`REPLACES` is ``step_D`` -> ``zero_metal_D`` -> ``update_E``, DECLARED rather
than inferred from the two slots -- ``zero_metal_D`` is a third driver pass
(driver.py:3310) that no slot names, and a composition reporting only the slots would
say it ran on the array path when the kernel performed it.

THE TWO MIRROR FILLS ARE CARRIED. ``fill_symmetry_bc_D`` (driver.py:3309) and
``fill_folded_far_ghosts_D`` (:3311) do nothing at all on an unfolded grid; on a folded
one they are performed by the OWNERSHIP INVERSION ported from
:mod:`.fused_magnetic_pair` -- the SOURCE thread writes each imaged ghost from a live
register, so no thread ever reads a word another block writes in the same launch and no
grid-wide barrier is needed. What :func:`covers_fused_electric_pair` refuses now is what
the two fills' OWN predicates refuse (``in_seam_coverage.covers_fill_symmetry`` /
``covers_fill_folded_far``) plus three clauses the carry adds, and nothing else.

THE GEOMETRY IS THE MIRROR IMAGE OF THE B FAMILY'S, WHICH IS WHY THIS IS A PORT AND NOT
A COPY. ``IYEE_SHIFTS`` gives Dx (1,0,0) against Bx (0,1,1): a D component's Yee shift
is 1 on its OWN axis and 0 on the other two, so it takes ONE far destination plane and
TWO near ones, where a B component takes two far and one near. Three consequences, each
carried in a named place below:

* the FAR fill is the one that moves the axis ``update_E`` indexes the component on, so
  the far ghost takes the DESTINATION's coefficient pair at ``n - 1`` and the near ghost
  takes this thread's unchanged -- the opposite assignment to the B family's;
* a D component's NEAR axes are exactly the two axes ``zero_metal_D`` clears it on, so
  it can be folded on one and walled on the other at once, and its near ghost then lands
  in a plane the clear owns. The array path leaves ``+0.0f`` there; a carry that
  multiplied the POST-clear register by an ODD plane's parity leaves ``-0.0f``. The near
  carry therefore reads a PRE-clear register and the clear is applied to the result,
  which is the driver's own order between :3309 and :3310. MEASURED, not argued:
  ``parity/meep_gpu/probe_cuda_electric_fill_carry_order.py`` records 0 differing words
  for the shipped ordering on twelve configurations and 9 and 11 for the naive one on
  the two where the cross term exists;
* the B family has no such row at all -- its one near axis IS the only axis its wall
  clears the component on, and a folded axis is never walled -- so this ordering is
  invisible on that seam and a straight port would have shipped it wrong.

``zero_metal_D`` IS carried, and it is the D family's OFF-DIAGONAL table rather than
the B family's diagonal: ``fields.IYEE_SHIFTS`` gives Dx (1,0,0), Dy (0,1,0),
Dz (0,0,1), so a D component has Yee shift 0 on the OTHER TWO axes and TWO components
sit on each wall -- x wall: Dy and Dz; y wall: Dx and Dz; z wall: Dx and Dy. A pair
that reused the B diagonal would clear the wrong component on every walled run, which
is why the table is spelled out below rather than imported from anything B-shaped.
23 of the 26 rows this product serves drive that pass.

=============================================================================
THE ALIASING HAZARD, AND THE ONE PLACE IT IS DELIBERATELY NOT PROMISED
=============================================================================

The curl half writes D in place and the constitutive half reads it. Binding that one
allocation twice -- once writable for the curl, once ``const __restrict__`` for the
constitutive source -- is undefined behaviour that NVRTC miscompiles without a
diagnostic. So D appears EXACTLY ONCE in :data:`_SIGNATURE` and the constitutive half
has no flux-density source pointers at all: its three ``D*[idx]`` reads are replaced
by the three registers the curl already holds.

That substitution is exact rather than approximate: the certified
``update_E_pml_real`` opens each component with a reload of the very word
``step_D_pml_real`` just stored, and a float32 stored to global and reloaded is the
identity on the bits.

THE THREE INVERSE-PERMITTIVITY POINTERS ARE NOT ``__restrict__``, and that is lifted
rather than decided: the certified ``update_E_pml_real`` declares them as plain
``const float*`` because an isotropic run hands the same device pointer three times
(``fields.py:1323-1325``), and ``restrict`` on mutually aliasing arguments is a promise
the caller cannot keep. :func:`assert_disjoint_bindings` therefore checks the
restrict-qualified arguments for collisions and checks SEPARATELY that no inv_eps
volume is one of them -- an inv_eps that aliased a written volume would be UB for a
different reason than the one this signature is shaped around.

=============================================================================
THE ONE NAME COLLISION THE SPLICE MUST RESOLVE, AND WHY IT MATTERS
=============================================================================

The two certified bodies were written against the same helper vocabulary, so spliced
verbatim into one scope they collide on ``kms_x/kms_y/kms_z``. That name is the
INTEGER sub-lattice for the D curl (``step_curl_kernels.real_pml_curl_tables(pml,
half_integer=False)``, and ``_step_D_fused_pml_real``'s own docstring) and the
HALF-INTEGER one for ``update_E`` (``coverage.CONSTITUTIVE_SIDES["E"]``,
``half_integer: True``, stepping.py:1015). THE PAIRING IS THE MIRROR IMAGE OF THE B/H
SEAM'S, where the curl reads half-integer and the constitutive integer.

Those two compile perfectly and differ by half a cell in the absorber profile --
converged, smooth and entirely wrong. So the constitutive group is renamed
``kms_half_*`` in the signature and in the spliced body both, and :func:`_line_starting`
matches the certified call as a WHOLE LINE so a moved argument list is a named failure
rather than a partial substitution.

The volume names do NOT collide and are lifted untouched: the curl spells
``Dx/Dy/Dz``, ``fu_D*`` and ``Hx/Hy/Hz``; the constitutive spells ``Ex/Ey/Ez``,
``f_w_E*``, ``inv_eps_E*`` and -- only in the three reads the seam removes --
``Dx/Dy/Dz``.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch. This
module ships a predicate, an emitter and a launcher; ``fused_pairs`` can plan it
opt-in (``arms.plan_step(..., fuse=True)``), and no shipped code path launches it.
"""

from __future__ import annotations

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None
import itertools
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

# THE PREDICATES AND THE IN-SEAM HELPERS. ``coverage`` and ``in_seam_coverage`` are
# stdlib-only by construction, so the predicate below answers on a laptop -- which is
# what lets the backend-free census ask it on all 186 rows.
try:
    from .coverage import covers_real_pml_constitutive, covers_real_pml_curl
    from .in_seam_coverage import (IYEE_SHIFTS, MIRROR_SOURCE_INDEX,
                                   covers_fill_folded_far, covers_fill_symmetry,
                                   folded_far_rows, mirror_fill_phases,
                                   zero_metal_axes)
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

    _coverage = _load("coverage")
    covers_real_pml_constitutive = _coverage.covers_real_pml_constitutive
    covers_real_pml_curl = _coverage.covers_real_pml_curl
    _in_seam_coverage = _load("in_seam_coverage")
    IYEE_SHIFTS = _in_seam_coverage.IYEE_SHIFTS
    MIRROR_SOURCE_INDEX = _in_seam_coverage.MIRROR_SOURCE_INDEX
    covers_fill_folded_far = _in_seam_coverage.covers_fill_folded_far
    covers_fill_symmetry = _in_seam_coverage.covers_fill_symmetry
    folded_far_rows = _in_seam_coverage.folded_far_rows
    mirror_fill_phases = _in_seam_coverage.mirror_fill_phases
    zero_metal_axes = _in_seam_coverage.zero_metal_axes
    compile_cache = _load("compile_cache")

# THE CERTIFIED DEVICE TEXT. Both modules import CuPy at module scope, so they are
# taken DEFENSIVELY -- exactly as :mod:`.fused_magnetic_pair` takes them, and for the
# same reason: the predicate is the part whose failure mode is a silent wrong answer,
# and it must be askable on a host with no device. The emitter refuses BY NAME where
# they are absent (:func:`_certified_halves`).
from . import own_cell_hoist

try:
    from . import constitutive_kernels
except ImportError:  # a host with no CuPy
    constitutive_kernels = None
try:
    from . import step_curl_kernels
except ImportError:  # a host with no CuPy
    step_curl_kernels = None
try:
    from . import in_seam_passes
except ImportError:  # a host with no CuPy
    in_seam_passes = None

# The source-seam clause, shared with every other fused-pair predicate on all three
# tracks. Imported rather than re-spelled: writing it out per product produced a real
# bug once (a Metal pair asked about the wrong seam and would have refused a magnetic
# source while ADMITTING an electric one -- which is precisely this seam).
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
# THE PARTITION — EMPTY ON THE CERTIFIED SIDE UNTIL A GATE MOVES IT
# =============================================================================
#
# Both are PLAIN assignments with NO type annotation, for the reason every sibling in
# this directory spells them that way: the partition readers walk the syntax tree
# WITHOUT importing the module (they have to — these modules import cupy) and an
# annotated assignment is an ``ast.AnnAssign`` those readers do not match.

#: Byte-identical to the array path with a gate verdict AND a ``certification.json``
#: record behind it. A fused product's bit-identity is a claim about the WELD, which
#: no verdict on either half establishes, so this stays empty until the fused gate has
#: run on a device.
#: RELEASED ON DEVICE 2026-09-11 by the campaign ``certification.json:cuda_fused_electric_pair_2026-08-31_fillcarry_r3``
#: (parity/meep_gpu/gate_cuda_fused_electric_pair.py, keep and flush legs, 34/34 cases identical over 60 complete steps).
#: Moved here from ``UNCERTIFIED_KERNELS`` by that verdict and by nothing
#: else: the gate ran BEFORE this edit, on the bytes that ship, and the
#: weld in ``cuda_kernels/fingerprints.json`` binds them.
CERTIFIED_KERNELS = (
    "fused_electric_pair_pml_real",
)
#: Shipped without a gate verdict. The kernel is here to BE gated.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its fused launch with the deposit repair? TRUE, AND THE
#: WIRING IS ``fused_pairs._install_fused_pair``'s -- reached because this family has
#: a row in ``fused_pairs.FUSED_PRODUCTS`` and because ``fused_pairs.FUSED_PAIR_SEAMS``
#: carries the ``step_D`` row this product landed with. The flag and the wiring change
#: together or not at all (``deposit_repair.py:221-226``); a product that declared it
#: without the two slots would compute ``update_E`` against a pre-injection D and
#: report success.
#:
#: THIS IS THE WHOLE POINT OF THE PRODUCT. On the B/H seam the flag widened a cell
#: from 7 rows to 123; here 75 of the cell's 79 rows carry an electric deposit, so
#: without the bracket this module would serve four.
#:
#: WHAT THE REPAIR HAS TO INVERT HERE. ``deposit_repair.apply`` recomputes
#: ``(f + kps*fw_fresh) - kms*fw_prev`` with ``xp`` array ops, which is word for word
#: what ``constitutive_apply`` computes -- with ``fw_fresh`` being ``D * inv_eps`` on
#: this side rather than ``B`` (``deposit_repair.SEAMS["D"]``). The two D-only clauses
#: in ``deposit_repair.repairable`` (an off-diagonal chi1inv row, an instantaneous
#: chi2/chi3) are refused by the constitutive half's own predicate before this
#: predicate reaches the seam clause.
#:
#: THE FOLD CLAUSES IN ``_folded_seam_reasons`` ARE LIVE SINCE 2026-08-31 and were
#: vacuous before it, because this product refused a folded grid outright. They are
#: what decides whether the repair can reconstruct a deposit whose MIRROR IMAGES the
#: two fills write after the injection: ``deposit_repair.repair_cells`` hands
#: ``save``/``apply`` the CLOSURE of those images, and a point-only repair is
#: measurably wrong on a folded seam. ``gate_cuda_fused_electric_pair``'s
#: ``leg_deposit_image_closure`` is the three-leg proof -- full closure, point-only,
#: no repair -- and it FAILS the run rather than passing it if the point-only control
#: does not diverge.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_electric_pair_pml_real"

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3302, :3309, :3310, :3311, :3313). Declared rather than inferred from
#: the two slots.
#:
#: THE TWO FILLS JOINED THE LIST ON 2026-08-31, and they are the whole of this round:
#: until then they were REFUSED BY NAME and every folded row on this seam sat outside
#: the predicate -- 53 of the cell's 79, measured, which is the largest single gap the
#: hand-CUDA board carried. They are carried by the OWNERSHIP INVERSION ported from
#: :mod:`.fused_magnetic_pair` -- the SOURCE thread writes the destination from a live
#: register instead of the destination thread reading a word another block wrote -- so
#: no grid-wide barrier is needed and none is used.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ float pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the helper "
            "names and returns the float it already computed. The expression, its "
            "parenthesisation and the store to f are unchanged, so the global state "
            "it leaves is the certified one."},
    {"line": "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;",
     "became": "    float value = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
               "    f[idx] = value;\n    return value;",
     "why": "same right-hand side, same tree, same store; the result is "
            "additionally named so it can be returned. A float32 stored to global "
            "and reloaded is the identity on the bits."},
    {"line": "        pml_apply(Dx, fu_Dx, idx, curl, ...);",
     "became": "        d_x = pml_apply_reg(Dx, fu_Dx, idx, curl, ...);",
     "why": "capture the register. Three lines, one per component; the argument "
            "lists are untouched. Three 'float d_* = 0.0f;' declarations are hoisted "
            "above the certified braced blocks because a value declared inside one "
            "does not outlive it."},
    {"line": "    float src_x = Dx[idx] * inv_eps_Ex[idx];   // stepping.py:982: "  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011
             "source * inv_eps",
     "became": "    float src_x = d_x * inv_eps_Ex[idx];   // stepping.py:982: "  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011
               "source * inv_eps",
     "why": "THE SEAM. The certified constitutive body opens each component by "
            "reloading the flux density the curl just stored; this reads the "
            "register instead. It also removes the constitutive half's only use of "
            "Dx/Dy/Dz, which is what lets D be bound exactly once. The multiply, its "
            "operand ORDER (D on the left, stepping.py:1011) and the fact that all "
            "three products are formed before any store are untouched."},
    {"line": "    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);",
     "became": "    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], "
               "kms_half_x[i]);",
     "why": "ONE RENAME, NO ARITHMETIC. kms_x is the INTEGER sub-lattice for the D "
            "curl and the HALF-INTEGER one for update_E -- the mirror image of the "
            "B/H seam's pairing -- and letting one shadow the other is a half-cell "
            "error in the absorber profile rather than a compile failure."},
    {"line": "        Dy[base] = 0.0f; Dz[base] = 0.0f;   "
             "(in_seam_passes._zero_metal_D_kernel_code)",
     "became": "    int clr_y = 0; ... if (wall_x && i == 0) { clr_y = 1; clr_z = 1; }"
               "\n    if (own_y && clr_y) { d_y = 0.0f; Dy[idx] = d_y; }",
     "why": "THE ONE RE-SPELLING, and it is launch geometry rather than arithmetic. "
            "The certified pass walks a FACE with face_geometry(); this launch walks "
            "the VOLUME and already holds i/j/k, and stored cell 0 of an axis is "
            "exactly that face. Same off-diagonal table (two components per wall), "
            "same +0.0f. The register is cleared beside the store so update_E reads "
            "the wiped value. The clear is NAMED as a per-component flag because the "
            "near fill carry reads it (a near ghost of a cleared cell carries +0.0f, "
            "not the parity-weighted one), and it is GUARDED on own_* because a cell "
            "a fill images is written by its source thread and a second writer would "
            "be a race."},
    # ----- THE FILL CARRY, ADDED 2026-08-31 --------------------------------------
    {"line": "    float f[idx] = ... (pml_apply's store)",
     "became": "    if (!owned) return 0.0f;  before the store",
     "why": "THE OWNERSHIP GUARD, ported from fused_magnetic_pair. On a cell one of "
            "the two fills images, the SOURCE thread writes D -- so this thread must "
            "not read D[idx] either, because that word is written by another block "
            "in this same launch and an ordinary CUDA launch has no grid-wide "
            "barrier. The fu recurrence stays ABOVE the guard: step_D writes fu at "
            "every cell and neither fill touches it (stepping.py:1451 writes field, "
            "never fu_field)."},
    {"line": "    field[_face(axis, 0)] = phase * field[_face(axis, 2)]   "
             "(stepping._write_mirror_ghost:1450-1451)",
     "became": "the SOURCE thread at stored 2 writes stored 0 from its register",
     "why": "THE OWNERSHIP INVERSION. The destination thread would have to read a "
            "word another block writes in this launch; the source thread already "
            "holds it. Same parity, same destination cell, same order relative to "
            "the wall clear -- what changes is which thread performs the write."},
    {"line": "    array[_face(axis, -1)] = parity * array[_face(axis, reflect)]   "
             "(stepping._fill_folded_far_ghosts:1524-1534)",
     "became": "the SOURCE thread at the reflect row writes the top plane",
     "why": "The same inversion for the far fill. Its parity is -phase (Yee shift 1 "
            "on a far axis) against the near fill's +phase, and it runs AFTER "
            "zero_metal_D (driver.py:3310 then :3311), which is why the far carry "
            "reads the POST-clear register and the near carry the pre-clear one."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "NEAR_SOURCE_INDEX", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "carried_destinations", "certified_constitutive_body",
    "certified_curl_body", "covers_fused_electric_pair", "device_sources",
    "far_fill_axes", "fill_carry_blocks", "fused_electric_pair_fills",
    "fused_electric_pair_prelude", "fused_electric_pair_source",
    "fused_electric_pair_tables", "inverse_epsilon_bindings",
    "launch_fused_electric_pair", "near_fill_axes", "ownership_declarations",
    "pre_clear_registers", "run_fused_electric_pair", "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================
#
# Every anchor below is an exact line of certified device text. If one stops matching,
# the certified string changed, and the splice raises rather than emitting a kernel
# that is quietly missing a mask, a store or a seam.

#: Separates a certified kernel's signature from its body. Both certified strings
#: close their parameter list on its own line, so ONE anchor lifts either body.
_BODY_ANCHOR = "\n) {\n"

#: The last line of the index decomposition, emitted by BOTH certified bodies. The
#: curl half keeps it; the constitutive half's copy (and the bounds guard above it) is
#: dropped, because splicing both would redeclare i/j/k and the kernel would not
#: compile.
_DECODE_END = "    int i = idx / (ny * nz);\n"

#: The certified curl helper, and the two edits that turn it into a value the seam
#: can carry. ``owned`` joined on 2026-08-31 with the fill carry: it is 0 exactly on a
#: cell one of the two fills images, and the array path's own behaviour licenses the
#: skip -- ``step_D`` writes ``fu`` EVERYWHERE and the fills touch ``D`` only
#: (stepping.py:1451 writes ``field``, never ``fu_field``), so the ``fu`` recurrence
#: stays above the guard and the displacement the fill is about to overwrite is simply
#: never formed.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ float pml_apply_reg(\n"
#: The certified helper's parameter list, and the one parameter the carry adds.
_PML_APPLY_PARAMETERS = (
    "    float kms, float sinv, float kms_u, float sinv_u\n)")
_PML_APPLY_PARAMETERS_OWNED = (
    "    float kms, float sinv, float kms_u, float sinv_u, int owned\n)")
_PML_APPLY_STORE = "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
_PML_APPLY_STORE_REG = (
    "    // THE OWNERSHIP GUARD, AND IT IS ABOUT MEMORY TRAFFIC RATHER THAN\n"
    "    // ABOUT A DEAD REGISTER. On a cell a fill images, the SOURCE thread\n"
    "    // writes f -- so this thread must not read f[idx] either: that word is\n"
    "    // written by another block in this same launch and an ordinary CUDA\n"
    "    // launch has no grid-wide barrier at which the load would be defined.\n"
    "    // fu above is UNGUARDED: step_D writes it at every cell and neither\n"
    "    // fill touches it (stepping.py:1451).\n"
    "    if (!owned) return 0.0f;\n"
    "    // THE ONE EDIT TO THIS HELPER: the right-hand side, its\n"
    "    // parenthesisation and the store are the certified ones; the value is\n"
    "    // additionally NAMED so the constitutive half can read it from a\n"
    "    // register instead of reloading it from global memory.\n"
    "    float value = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
    "    f[idx] = value;\n"
    "    return value;\n")

#: The three carried registers, the certified curl's own name for each target volume,
#: and the axis letter each carries in the constitutive body's ``src_*`` names.
_CARRIED: Tuple[str, ...] = ("x", "y", "z")
_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")

#: ``in_seam_passes._zero_metal_D_kernel_code``'s component table, restated as
#: (walled-axis flag, that axis's coordinate, the two components wiped there).
#:
#: THE OFF-DIAGONAL COMPLEMENT, and it is spelled here rather than imported from
#: anything B-shaped: ``fields.IYEE_SHIFTS`` gives Dx (1,0,0), Dy (0,1,0), Dz (0,0,1),
#: so a D component has Yee shift 0 on the OTHER TWO axes and TWO components sit on
#: each wall. A pair that reused the B family's diagonal would clear ONE wrong
#: component and leave two right ones standing, on every walled run.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("wall_x", "i", ("y", "z")),
    ("wall_y", "j", ("x", "z")),
    ("wall_z", "k", ("x", "y")),
)

#: The lifted curl body's own spellings for each axis's coordinate, stride and
#: extent. Spelled once: a second decode here would be a second place to get the
#: layout wrong, and every name is a variable the CERTIFIED curl body already
#: declares (``int k = idx % nz`` and the three ``const int s*`` above it).
_AXIS_NAME: Tuple[str, ...] = ("x", "y", "z")
_COORDINATE: Tuple[str, ...] = ("i", "j", "k")
_STRIDE: Tuple[str, ...] = ("sx", "sy", "sz")
_EXTENT: Tuple[str, ...] = ("nx", "ny", "nz")

#: The three fill scalars' runtime names, per axis, in :data:`_SIGNATURE`'s own
#: spelling. ``_FAR_ACTIVE`` is a TEST rather than a flag because the far plan is
#: carried as a reflect row with ``-1`` standing for "this pass does not run on this
#: axis" -- a sentinel, so a source that read it anyway would index outside the volume
#: and be caught rather than silently imaging row 0.
_NEAR_FLAG: Tuple[str, ...] = ("near_x", "near_y", "near_z")
_REFLECT: Tuple[str, ...] = ("reflect_x", "reflect_y", "reflect_z")
_FAR_ACTIVE: Tuple[str, ...] = ("reflect_x >= 0", "reflect_y >= 0", "reflect_z >= 0")
_PHASE: Tuple[str, ...] = ("phase_x", "phase_y", "phase_z")

#: The per-component name of the wall-clear flag the carry reads. Emitted by
#: :func:`zero_metal_carry` and consumed by :func:`fill_carry_blocks`; see that
#: function for why the clear has to be a NAMED flag on this family and is not one on
#: the B side.
_CLEARED: Tuple[str, ...] = ("clr_x", "clr_y", "clr_z")

#: ``stepping.MIRROR_SOURCE_INDEX`` under this family's own name, re-exported so a
#: reader does not have to chase :mod:`.in_seam_coverage` for the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX

def near_fill_axes(target: int) -> Tuple[int, ...]:
    """Which axes ``fill_symmetry_bc_D`` can image for ONE D component.

    ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1440-1447) writes axis ``a``
    for component ``m`` exactly when the plane's phase is declared and
    ``iyee[m][a] == 0``. THE YEE TEST IS STRUCTURAL and is what this returns; the
    phase is a RUNTIME flag on this track, because the axis is a runtime argument here
    rather than a source specialisation (``in_seam_passes``' own reason: one device
    string per family, so one digest covers every fold orientation).

    ``IYEE_SHIFTS`` gives Dx (1,0,0), Dy (0,1,0), Dz (0,0,1), so the answer is the TWO
    axes that are NOT the component's own -- THE EXACT COMPLEMENT of the B family's,
    where the near set is the component's own axis alone. That inversion is why this
    is read off the table rather than written down: a hand-typed constant here is the
    one place it is plausible and wrong, and wrong is a plane of mirrored values on
    the wrong two faces.
    """
    shifts = IYEE_SHIFTS[_TARGETS[target]]
    return tuple(axis for axis in range(3) if shifts[axis] == 0)


def far_fill_axes(target: int) -> Tuple[int, ...]:
    """Which axes ``fill_folded_far_ghosts_D`` can image for ONE D component.

    ``stepping._fill_folded_far_ghosts`` (stepping.py:1524-1534) writes axis ``a`` for
    component ``m`` exactly when ``_stored_past_owned(grid, a)`` -- a folded PERIODIC
    axis, at either full-count parity -- and ``iyee[m][a] == 1``. Again the Yee test
    is structural and the fold test is the runtime reflect row.

    THE EXACT COMPLEMENT OF :func:`near_fill_axes` ON THIS FAMILY, and the whole
    geometric content of the D carry is that the complement runs the other way from
    the B family's: a D component's Yee shift is 1 on its OWN axis and 0 on the other
    two, so it takes ONE far destination plane against TWO near ones, where a B
    component takes two far and one near. A cell at stored 0 of both near axes carries
    the PRODUCT of the two near parities.
    """
    shifts = IYEE_SHIFTS[_TARGETS[target]]
    return tuple(axis for axis in range(3) if shifts[axis] == 1)


def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """Every ghost cell ONE source thread owns, as ``(far axes at top, near axes at 0)``.

    THE OWNERSHIP RULE, PORTED FROM :mod:`.fused_magnetic_pair` AND GENERALISED IN
    EXACTLY ONE PLACE: there the near set is a single axis and is carried as a
    boolean; here it is a SUBSET, because a D component has two near axes. The far
    set was already a subset there and is a single axis here, so the two families are
    the same function with the roles exchanged and this signature covers both.

    The driver runs the near fill, the wall clear and the far fill in that order
    (driver.py:3309-3311), and each fill is applied axis by axis in place, so a cell
    the fills leave at the top of several planes is written more than once. Composing
    the passes gives one closed form: for component ``m`` the seam leaves, at the cell
    that sits at ``last`` on the far-axis set ``T`` and at stored 0 on the near-axis
    set ``S``,

        (prod over a in T of -phase_a)
            * cleared_at_that_thread(
                  (prod over a in S of phase_a) * v_m(the fully back-substituted cell))

    -- independent of the order the axes are applied in, which is why nothing here has
    to model either fill's X/Y/Z loop. The WALL CLEAR SITS BETWEEN THE TWO PRODUCTS
    and that placement is not cosmetic: see :func:`fill_carry_blocks`.

    So ONE thread, the one at the fully back-substituted cell, computes the
    displacement every one of those ghosts carries, and it writes them all. Every
    destination word is written by exactly one thread and no thread reads a word
    another writes, which is what makes the carry legal inside ONE ordinary CUDA
    launch with no grid-wide barrier.

    Returned smallest-subset-first for a stable emission order; the order is
    unobservable (the cells are distinct) and a stable one keeps a source diff
    readable.
    """
    near = tuple(int(axis) for axis in near)
    far = tuple(int(axis) for axis in far)
    combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for far_size in range(len(far) + 1):
        for far_subset in itertools.combinations(far, far_size):
            for near_size in range(len(near) + 1):
                for near_subset in itertools.combinations(near, near_size):
                    if not far_subset and not near_subset:
                        continue  # the thread's OWN cell, not a ghost
                    combos.append((far_subset, near_subset))
    return tuple(sorted(combos, key=lambda item: (len(item[0]) + len(item[1]),
                                                  item[0], item[1])))


#: The fused entry point. The ONLY hand-written device text in this module, and it is
#: a signature: no arithmetic lives here. Dx/Dy/Dz appear EXACTLY ONCE -- see the
#: module docstring's aliasing section, which is the reason the constitutive half has
#: no flux-density source pointers at all.
_SIGNATURE = r'''
extern "C" __global__ void fused_electric_pair_pml_real(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the constitutive
    // half reads it from a register; binding it a second time as a const
    // __restrict__ source would be two restrict pointers to one allocation, which
    // is UB and which NVRTC miscompiles without a diagnostic.
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    // H, the curl's operands.
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    // E and its constitutive history.
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    // NOT __restrict__, and lifted from the certified update_E rather than decided
    // here: an isotropic run hands the same device pointer three times
    // (fields.py:1323-1325), and restrict on mutually aliasing arguments is a
    // promise the caller cannot keep. They are read-only, so nothing is lost but
    // the promise.
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    int nx, int ny, int nz, float dtdx,
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
    // zero_metal_D's three walled-axis flags, from in_seam_coverage.zero_metal_axes.
    // A folded metallic axis is EXCLUDED there (stepping.py:2237-2239), and the
    // predicate refuses an axis reported both folded and walled, so no cell is ever
    // both a fill destination and a wall plane for one component.
    int wall_x, int wall_y, int wall_z,
    // THE TWO FILLS' RUNTIME PLAN, from in_seam_coverage.mirror_fill_phases and
    // .folded_far_rows. `near_a` is 1 on a MIRROR-folded axis (either termination),
    // where fill_symmetry_bc_D images stored 0 from stored 2. `reflect_a` is
    // fill_folded_far_ghosts_D's image row on a folded PERIODIC axis and -1 where
    // that pass does not run -- a SENTINEL rather than 0, so a source that read it
    // anyway would index outside the volume and be caught rather than silently
    // imaging row 0. `phase_a` is the plane's declared mirror phase; the near fill
    // weights by +phase (the imaged components have Yee shift 0 there) and the far
    // fill by -phase (theirs is 1), which is the certified in_seam_passes kernels'
    // own spelling of fields.mirror_parity.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float phase_x, float phase_y, float phase_z
) {
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 2237-2239->2284-2286


def _certified_halves() -> None:
    """Refuse the emitter BY NAME on a host where the certified text is unreachable.

    THE SPLICE IS THE LIFT, so a missing half is not a degraded emit -- there is
    nothing to emit. Raising here keeps that fact one frame from the caller instead of
    surfacing as ``'NoneType' object has no attribute '_REAL_PML_PRELUDE'`` inside a
    string operation. :func:`covers_fused_electric_pair` needs none of them and still
    answers.
    """
    missing = [name for name, module in (
        ("step_curl_kernels", step_curl_kernels),
        ("constitutive_kernels", constitutive_kernels)) if module is None]
    if missing:
        raise RuntimeError(
            f"the certified halves {missing} are not importable on this host (they "
            f"import CuPy at module scope), so there is no certified text to splice. "
            f"covers_fused_electric_pair needs neither and still answers")


def _split_body(source: str, prelude: str, name: str) -> str:
    """One certified kernel string, reduced to its body.

    Strips the prelude it was built from and the signature that follows it, and the
    closing brace. Raises rather than returning a truncated body: a splice that
    silently dropped a component would produce a kernel that steps two fields and
    reports success.
    """
    if not source.startswith(prelude):
        raise AssertionError(
            f"{name} no longer begins with the prelude this module lifts separately; "
            f"the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, not "
            f"1; the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    A missing anchor is certified text that has changed under this family, and
    splicing around it would produce a kernel that compiles and is quietly not the
    certified arithmetic; TWO matches would mean the anchor no longer identifies a
    single statement.
    """
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this family LIFTS that line rather than retyping "
            f"it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the lift "
            f"of {what} would take an arbitrary one")
    return matches[0]


def fused_electric_pair_prelude() -> str:
    """Both certified preludes, with ``pml_apply`` turned into a value.

    ``constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE`` is emitted UNTOUCHED --
    ``constitutive_apply`` needs no edit at all, because the seam changes where its
    ``src`` argument comes from and not what it does with it.
    """
    _certified_halves()
    prelude = step_curl_kernels._REAL_PML_PRELUDE
    if prelude.count(_PML_APPLY_SIGNATURE) != 1:
        raise AssertionError(
            f"the certified curl prelude declares pml_apply "
            f"{prelude.count(_PML_APPLY_SIGNATURE)} times, not once; the value "
            f"rewrite has no anchor")
    if prelude.count(_PML_APPLY_STORE) != 1:
        raise AssertionError(
            "the certified pml_apply no longer closes with the store this module "
            "turns into a named value; the arithmetic may have moved")
    if prelude.count(_PML_APPLY_PARAMETERS) != 1:
        raise AssertionError(
            f"the certified pml_apply's parameter list appears "
            f"{prelude.count(_PML_APPLY_PARAMETERS)} times, not once; the ownership "
            f"guard has no anchor to take its argument on")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_PML_APPLY_PARAMETERS,
                              _PML_APPLY_PARAMETERS_OWNED, 1)
    prelude = prelude.replace(_PML_APPLY_STORE, _PML_APPLY_STORE_REG, 1)
    return prelude + constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE


def ownership_declarations() -> str:
    """``own_x``/``own_y``/``own_z`` -- is this thread's own cell imaged by a fill?

    THE ONE PLACE THE INVERSION IS DECIDED, and every other emitter below reads these
    three flags rather than restating the test.

    A cell either fill writes is OWNED BY ITS SOURCE THREAD: the thread standing on it
    computes the curl and the split-field recurrence and stores ``fu`` (the array
    path's ``step_D`` writes ``fu`` at every cell and neither fill touches it), and
    then STOPS. It never loads ``D`` at its own cell, never stores ``D`` there, and
    never touches ``E`` or ``f_w_E`` there -- so no load it issues can land on a word
    another block writes in this launch, which is what makes the carry legal with no
    grid-wide barrier rather than merely convenient.

    THE FLAGS ARE RUNTIME, not baked. That is this track's rule, not a compromise:
    ``in_seam_passes`` takes its axis as a runtime argument for exactly this reason --
    one device string per family, so one certification digest covers every fold
    orientation and no per-axis variant can end up pinned that nobody ran.
    """
    lines = [
        "\n    // --- the ownership inversion (fill_symmetry_bc_D, "
        "fill_folded_far_ghosts_D) ---",
        "    // A cell a fill images is written by its SOURCE thread, from a live",
        "    // register, below. The thread standing ON it stops after fu: forming",
        "    // v would read a D word another block writes in this same launch, and",
        "    // an ordinary CUDA launch has no grid-wide barrier at which that load",
        "    // is defined. The array path discards that displacement too -- the",
        "    // fill overwrites it (driver.py:3309, :3311).",
    ]
    for target, name in enumerate(_TARGETS):
        tests = [f"({_NEAR_FLAG[axis]} && {_COORDINATE[axis]} == 0)"
                 for axis in near_fill_axes(target)]
        tests += [f"({_FAR_ACTIVE[axis]} && "
                  f"{_COORDINATE[axis]} == {_EXTENT[axis]} - 1)"
                  for axis in far_fill_axes(target)]
        if not tests:
            raise AssertionError(
                f"{name} is imaged by neither fill on any axis; IYEE_SHIFTS gives "
                f"every D component one shift-1 axis and two shift-0 axes, so an "
                f"empty test means the Yee table has drifted")
        near = "".join(_AXIS_NAME[axis] for axis in near_fill_axes(target))
        far = "".join(_AXIS_NAME[axis] for axis in far_fill_axes(target))
        lines.append(f"    // {name} {IYEE_SHIFTS[name]}: near fill on {near}, "
                     f"far fill on {far}.")
        lines.append(f"    int own_{_CARRIED[target]} = !({' || '.join(tests)});")
    return "\n".join(lines) + "\n"


def certified_curl_body() -> str:
    """``step_D_pml_real``'s body, lifted, with the three registers captured.

    Not a transcription: this is ``step_curl_kernels._step_D_pml_real_kernel_code``
    itself, minus its prelude and signature. ``step_B``'s forward shifts are a
    different product and never appear here -- the backward ``shift_dn`` and the
    top-plane mirror masks below are what make this the D side.
    """
    _certified_halves()
    body = _split_body(step_curl_kernels._step_D_pml_real_kernel_code,
                       step_curl_kernels._REAL_PML_PRELUDE,
                       "_step_D_pml_real_kernel_code")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the "
            "carried registers have nowhere to be declared")
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END,
        "\n    // The three carried registers. Declared HERE because each certified\n"
        "    // component below is a braced scope, and a value declared inside one\n"
        "    // does not outlive it.\n"
        "    float d_x = 0.0f;\n"
        "    float d_y = 0.0f;\n"
        "    float d_z = 0.0f;\n"
        "\n",
        ownership_declarations(),
        tail,
    ))
    for axis, target in zip(_CARRIED, _TARGETS):
        old = f"pml_apply({target}, fu_{target}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} {body.count(old)} "
                f"times, not once; the capture has no anchor")
        body = body.replace(old, f"d_{axis} = pml_apply_reg({target}, fu_{target}, ", 1)
        # Matched as a WHOLE LINE rather than by a `);` tail: a tail anchor would
        # attach to whichever statement happened to end first if the certified body
        # ever moved this call onto two lines -- and would then append the ownership
        # flag to a statement this module has not read.
        prefix = f"        d_{axis} = pml_apply_reg({target}, fu_{target}, "
        call = _line_starting(body, prefix, f"{target}'s captured pml_apply")
        if not call.endswith(");"):
            raise AssertionError(
                f"the certified curl body no longer closes {target}'s pml_apply on "
                f"one line ({call!r})")
        body = body.replace(call, f"{call[:-2]}, own_{axis});", 1)
    if "pml_apply(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be "
            "written to global memory and never read into the seam")
    return body


def pre_clear_registers() -> str:
    """``pre_x/pre_y/pre_z`` -- the displacement as ``fill_symmetry_bc_D`` sees it.

    THE DRIVER'S ORDER, MADE INTO A REGISTER. ``fill_symmetry_bc_D`` (driver.py:3309)
    runs one pass BEFORE ``zero_metal_D`` (:3310), so a near ghost images the
    UNCLEARED displacement and is only then cleared at its own cell. Keeping the
    pre-clear value is the whole of that ordering: without it the near carry would
    multiply a cleared ``0.0f`` by the plane's parity and leave ``-0.0f`` on an odd
    plane where the array path leaves ``+0.0f`` -- a different word, on a whole plane,
    with no numerical difference to notice.

    Emitted unconditionally rather than under a fold test: this track's rule is one
    device string per family with the fold as a runtime argument, and three dead
    copies on an unfolded run cost nothing the compiler does not remove.
    """
    return (
        "\n    // The PRE-CLEAR displacement, for the near fill carry. "
        "fill_symmetry_bc_D\n"
        "    // (driver.py:3309) runs BEFORE zero_metal_D (:3310), so a near ghost "
        "images\n"
        "    // what the curl left, and the clear then applies at the ghost's own "
        "cell.\n"
        + "".join(f"    float pre_{axis} = d_{axis};\n" for axis in _CARRIED))


def zero_metal_carry() -> str:
    """``zero_metal_D`` (driver.py:3310), carried between the two halves.

    ``stepping._zero_metal`` writes the integer 0 into stored cell 0 of every
    component whose Yee shift on a walled axis is 0 -- for D that is the OFF-DIAGONAL,
    TWO components per wall (:data:`_ZERO_METAL_ROWS`). The register is cleared BESIDE
    the store, because the constitutive half below reads the register and not the
    volume.

    STORED CELL 0 IS THE LOW WALL and the high wall is the zero ghost ``shift_dn``
    already supplies, which is the certified pass's own note. That pass walks a FACE
    through ``face_geometry``; this launch walks the volume and already holds i/j/k,
    and ``face_geometry(axis)``'s ``base`` set is exactly the cells with that axis's
    coordinate at 0 -- axis 0 gives ``base = t`` over ``ny*nz`` cells (i = 0), axis 1
    gives ``(t/nz)*(ny*nz) + t%nz`` (j = 0), axis 2 gives ``(t/ny)*(ny*nz) +
    (t%ny)*nz`` (k = 0).

    A FOLDED METALLIC AXIS CARRIES NO WALL (``stepping._zero_metal``'s ``is_metallic
    and not is_mirrored``, :2237-2239), and :func:`covers_fused_electric_pair` refuses
    an axis reported both folded and walled, so ``zero_metal_axes``, the boundary
    codes and the fill plan cannot disagree on any admitted row.

    THE CLEAR IS A NAMED FLAG, AND ON THIS FAMILY IT HAS TO BE. The B side can write
    the wipe straight into the register and be done, because a B component's ONE near
    axis is its own axis and is the ONLY axis ``zero_metal_B`` clears it on -- folded
    and walled are mutually exclusive there, so a B near ghost never lands in a wall
    plane. A D component's near axes are the OTHER TWO, which are exactly the two axes
    ``zero_metal_D`` clears it on, so it can be folded on one and walled on the other
    at the same time and its near ghost then lands in a cleared plane. The array path
    leaves ``+0.0f`` there (the near fill at :3309, the clear at :3310); a carry that
    multiplied the post-clear register by the parity would leave ``-0.0f`` on an odd
    plane -- a different word. :func:`fill_carry_blocks` reads these flags to place the
    clear between the two parity products, which is where the driver puts it.

    GUARDED ON own_*: a cell one of the fills images belongs to its SOURCE thread, and
    the array path agrees -- ``zero_metal_D`` (:3310) is overwritten there by
    ``fill_folded_far_ghosts_D`` (:3311), or was written before it by
    ``fill_symmetry_bc_D`` (:3309). Clearing it here would be a second thread writing
    the same word.
    """
    lines = [
        "\n    // stepping._zero_metal (:2206-2245) / in_seam_passes.zero_metal_D,\n"
        "    // CARRIED. The OFF-DIAGONAL: a D component sits ON the wall of each\n"
        "    // axis whose Yee shift is 0, which for D is the other two -- so two\n"
        "    // components are wiped per walled axis. Stored cell 0 is the LOW wall;\n"
        "    // the high wall is the zero ghost shift_dn already supplies.\n"
        "    // The flag is NAMED because the near fill carry below reads it: a near\n"
        "    // ghost of a cleared cell carries +0.0f, not the parity-weighted one.\n"
    ]
    lines.append("    " + " ".join(f"int {name} = 0;" for name in _CLEARED) + "\n")
    for flag, coordinate, axes in _ZERO_METAL_ROWS:
        sets = " ".join(f"{_CLEARED[_CARRIED.index(axis)]} = 1;" for axis in axes)
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {sets} }}\n")
    for axis, cleared in zip(_CARRIED, _CLEARED):
        lines.append(
            f"    if (own_{axis} && {cleared}) "
            f"{{ d_{axis} = 0.0f; D{axis}[idx] = d_{axis}; }}\n")
    return "".join(lines)


def fill_carry_blocks(target: int, indent: str = "        ") -> List[str]:
    """Every imaged ghost this thread owns, then ``update_E`` at each of them.

    ONE BLOCK PER DESTINATION IN :func:`carried_destinations`, guarded on the flags
    that say THIS thread is that destination's source -- stored
    :data:`NEAR_SOURCE_INDEX` on each near axis, the runtime reflect row on the far
    axis.

    EVERY BLOCK IS NESTED INSIDE THE COMPONENT'S OWNERSHIP GUARD, and that nesting is
    not tidiness. It is a defect the sibling track's device gate found: with two fills
    live a thread can be the SOURCE of one and the DESTINATION of the other -- at
    ``j == 2`` on a folded y while standing on ``i == nx - 1`` of a folded periodic x
    -- and unguarded it would write a ghost from a ``d_*`` that was never formed. The
    destination it would have written is already owned by the fully back-substituted
    thread at ``(i == reflect_x, j == 2)``, which is the closed form's whole point.
    :func:`certified_constitutive_body` is the only caller and emits these strictly
    inside ``if (own_*) {`` ... ``}``.

    THE THREE VALUES, IN DRIVER ORDER, AND THE ORDER IS THE ARITHMETIC:

    1. the NEAR product, formed from ``pre_*`` -- the PRE-clear displacement -- because
       ``fill_symmetry_bc_D`` (driver.py:3309) runs BEFORE ``zero_metal_D`` (:3310);
    2. the WALL CLEAR at the destination, which on this family is the same test as the
       source thread's own (a near ghost moves the component to stored 0 of an axis
       that is folded, hence never walled, so every walled coordinate is unchanged
       between the two cells) -- so the flag :func:`zero_metal_carry` already computed
       IS the destination's, and it is applied here rather than folded into the
       register;
    3. the FAR product, ``-phase`` on the component's own axis, applied last because
       ``fill_folded_far_ghosts_D`` (:3311) runs after the clear and reads what the
       clear left. A far-written cell is never itself cleared afterwards, so no clear
       follows step 3.

    A pure far ghost takes steps 2 and 3 only, which is exactly "read the register
    after the clear" -- the B family's rule, recovered rather than special-cased.

    ``update_E`` AT THE GHOST reads ``inv_eps`` AT THE GHOST INDEX. That volume is
    read-only in this launch, so a load off this thread's own cell is defined; ``E``
    and ``f_w_E`` at the ghost are written only here, by this thread, because the
    thread standing on the ghost is outside its own ``own_*`` guard.

    THE COEFFICIENT PAIR IS THE DESTINATION'S. ``update_E`` indexes ``E{m}`` on the
    component's OWN axis (``constitutive_kernels._update_E_pml_real_kernel_code``:
    ``kps_x[i]``, ``kps_y[j]``, ``kps_z[k]``), and the FAR fill is the one that moves
    that axis on this family -- the mirror image of the B side, where the NEAR fill
    moved it. So a far ghost takes ``kps[n - 1]`` and a pure near ghost takes this
    thread's own index unchanged.
    """
    near = near_fill_axes(target)
    far = far_fill_axes(target)
    name = _TARGETS[target]
    if far != (target,):
        raise AssertionError(
            f"{name} is a FAR-fill destination on axes {far}; for the D family that "
            f"set is exactly the component's own axis (fields.IYEE_SHIFTS is 1 there "
            f"and 0 on the other two), and this carry images one far ghost plane per "
            f"component")
    if len(near) != 2 or target in near:
        raise AssertionError(
            f"{name} is a NEAR-fill destination on axes {near} against a far axis "
            f"{far}; on the D family the two sets are COMPLEMENTARY, so near holds "
            f"exactly the two axes that are not the component's own")
    walls = tuple(_COORDINATE.index(coordinate)
                  for _flag, coordinate, axes in _ZERO_METAL_ROWS
                  if _CARRIED[target] in axes)
    if set(walls) & set(far):
        raise AssertionError(
            f"{name} images a FAR ghost along {far} that zero_metal_D also clears on "
            f"{sorted(set(walls) & set(far))}; _ZERO_METAL_ROWS is the D "
            f"OFF-DIAGONAL and far_fill_axes is the component's own axis, so a far "
            f"ghost sits at the SAME coordinate on every axis that could clear it as "
            f"the thread that owns it and inherits that thread's clear")
    if set(walls) != set(near):
        raise AssertionError(
            f"{name} is cleared on axes {sorted(walls)} and near-imaged on "
            f"{sorted(near)}; on the D family those two sets are THE SAME (both are "
            f"the shift-0 axes), which is the fact that makes the wall clear belong "
            f"BETWEEN the near and far products rather than inside the register")
    register = f"d_{_CARRIED[target]}"
    pre = f"pre_{_CARRIED[target]}"
    cleared = _CLEARED[target]
    inner = indent + "    "
    lines: List[str] = []
    for far_subset, near_subset in carried_destinations(near, far):
        tag = ("g" + _CARRIED[target] + "_"
               + "".join(_AXIS_NAME[axis] for axis in far_subset)
               + ("n" if near_subset else "")
               + "".join(_AXIS_NAME[axis] for axis in near_subset))
        guards: List[str] = []
        terms: List[str] = []
        near_weights: List[str] = []
        far_weights: List[str] = []
        described: List[str] = []
        for axis in far_subset:
            guards.append(f"{_FAR_ACTIVE[axis]} && "
                          f"{_COORDINATE[axis]} == {_REFLECT[axis]}")
            terms.append(f"+ ({_EXTENT[axis]} - 1 - {_REFLECT[axis]}) "
                         f"* {_STRIDE[axis]}")
            # Yee shift 1 on a far axis -> mirror_parity = phase * (1 - 2*1).
            # Spelled as a negation of the SCALAR, which is
            # in_seam_passes._fill_folded_far_D_kernel_code's own line.
            far_weights.append(f"(-{_PHASE[axis]})")
            described.append(
                f"the far fill on {_AXIS_NAME[axis]}: {_COORDINATE[axis]} = "
                f"{_EXTENT[axis]} - 1 imaged from {_REFLECT[axis]}, weight "
                f"-{_PHASE[axis]} (stepping._fill_folded_far_ghosts:1524-1534)")
        for axis in near_subset:
            guards.append(f"{_NEAR_FLAG[axis]} && "
                          f"{_COORDINATE[axis]} == {NEAR_SOURCE_INDEX}")
            terms.append(f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}")
            # Yee shift 0 on a near axis -> mirror_parity = phase * (1 - 2*0), which
            # is in_seam_passes._fill_symmetry_D_kernel_code's bare `phase`.
            near_weights.append(_PHASE[axis])
            described.append(
                f"the near fill on {_AXIS_NAME[axis]}: {_COORDINATE[axis]} = 0 "
                f"imaged from {NEAR_SOURCE_INDEX}, weight {_PHASE[axis]} "
                f"(stepping._fill_symmetry_ghost_cells:1440-1447, "
                f"_write_mirror_ghost:1450-1451)")
        coefficient = (f"{_EXTENT[target]} - 1" if far_subset
                       else _COORDINATE[target])
        lines.append(f"{indent}if ({' && '.join(guards)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(f"{inner}int {tag}_i = idx {' '.join(terms)};")
        if near_weights:
            lines.append(
                f"{inner}// fill_symmetry_bc_D (:3309) reads the PRE-clear "
                f"displacement:")
            lines.append(
                f"{inner}// the driver clears at :3310, one pass later.")
            lines.append(f"{inner}float {tag}_v = {' * '.join(near_weights)} * {pre};")
            lines.append(
                f"{inner}// zero_metal_D (:3310) at the DESTINATION. Its walled "
                f"coordinates are")
            lines.append(
                f"{inner}// this thread's -- a near ghost only moves a FOLDED axis, "
                f"and a folded")
            lines.append(
                f"{inner}// axis is never walled -- so the source thread's own flag "
                f"IS the test.")
            lines.append(f"{inner}if ({cleared}) {tag}_v = 0.0f;")
        else:
            lines.append(
                f"{inner}// No near fill on this destination: the register is "
                f"already the")
            lines.append(
                f"{inner}// post-clear value, which is what the far fill reads at "
                f":3311.")
            lines.append(f"{inner}float {tag}_v = {register};")
        if far_weights:
            lines.append(
                f"{inner}// fill_folded_far_ghosts_D (:3311), applied LAST because "
                f"the driver")
            lines.append(
                f"{inner}// runs it last; a far-written cell is not cleared "
                f"afterwards.")
            lines.append(f"{inner}{tag}_v = {' * '.join(far_weights)} * {tag}_v;")
            lines.extend([
                f"{inner}// The DESTINATION's own coefficient entry: stored index "
                f"{_EXTENT[target]} - 1 on",
                f"{inner}// {_AXIS_NAME[target]}, NOT this thread's at "
                f"{_REFLECT[target]}. update_E indexes {name[-1]} on",
                f"{inner}// {_COORDINATE[target]} "
                f"(constitutive_kernels._update_E_pml_real_kernel_code) and the far",
                f"{inner}// fill images along that same axis, so reusing the "
                f"source's pair would",
                f"{inner}// apply the reflect row's absorber profile to the top "
                f"plane.",
            ])
        else:
            lines.append(
                f"{inner}// A near ghost does not move {_COORDINATE[target]}, the "
                f"axis update_E indexes this")
            lines.append(
                f"{inner}// component on, so it takes this thread's own pair "
                f"unchanged.")
        # THE STORE TO D IS NOT OPTIONAL, and leaving it out is a defect that costs
        # exactly one plane per component: the flux density at a ghost is what the NEXT
        # timestep's curl differences, and update_E alone would leave it holding the
        # un-imaged value the curl wrote. Measured on the GPU host's first device run at 3
        # planes x 99 cells on a 9x7x11 single-fold row.
        lines.append(f"{inner}{name}[{tag}_i] = {tag}_v;")
        lines.append(
            f"{inner}float {tag}_s = {tag}_v * inv_eps_E{_CARRIED[target]}"
            f"[{tag}_i];")
        lines.append(
            f"{inner}constitutive_apply(E{_CARRIED[target]}, "
            f"f_w_E{_CARRIED[target]}, {tag}_i, {tag}_s, "
            f"kps_{_CARRIED[target]}[{coefficient}], "
            f"kms_half_{_CARRIED[target]}[{coefficient}]);")
        lines.append(f"{indent}}}")
    return lines


def certified_constitutive_body() -> str:
    """``update_E_pml_real``'s body, lifted, reading the registers.

    Drops the second index decomposition (the curl body above already declares
    ``idx``, ``i``, ``j`` and ``k`` and its bounds guard has already returned), turns
    the three flux-density reads into the seam, and resolves the one name collision the
    module docstring names.
    """
    _certified_halves()
    body = _split_body(own_cell_hoist.unhoisted_kernel_code(
                           constitutive_kernels._update_E_pml_real_kernel_code,
                           "update_E_pml_real", "_update_E_pml_real_kernel_code"),
                       constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE,
                       "_update_E_pml_real_kernel_code")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own line; "
            "the duplicate decomposition cannot be identified")
    tail = body.split(_DECODE_END, 1)[1]

    # THE SEAM. Each certified read becomes the register the curl already holds, which
    # is also what removes this half's only use of Dx/Dy/Dz. ONE TOKEN PER LINE is
    # substituted and the rest of the certified statement -- the operand order, the
    # inv_eps index and the line's own annotation -- is left exactly as it stands.
    for axis, target in zip(_CARRIED, _TARGETS):
        anchor = f"    float src_{axis} = {target}[idx] * inv_eps_E{axis}[idx];"
        line = _line_starting(tail, anchor, f"{target}'s inverse-permittivity product")
        if line.count(f"{target}[idx]") != 1:
            raise AssertionError(
                f"{target}[idx] appears {line.count(f'{target}[idx]')} times in "
                f"{line!r}; the seam substitution has no unambiguous token")
        tail = tail.replace(line, line.replace(f"{target}[idx]", f"d_{axis}", 1), 1)
    first = _line_starting(tail, "    float src_x = d_x * inv_eps_Ex[idx];",
                           "the rewritten first inverse-permittivity product")
    tail = tail.replace(
        first,
        "    // THE SEAM. The certified body reads D[idx] here -- the very word the\n"
        "    // curl half stored a few lines above -- and this reads the register\n"
        "    // instead. A float32 stored to global and reloaded is the identity on\n"
        "    // the bits, so the substitution is exact rather than close; it is also\n"
        "    // what removes this half's only use of Dx/Dy/Dz, which is what lets D\n"
        "    // be bound exactly once.\n" + first, 1)
    for target in _TARGETS:
        if f"{target}[idx]" in tail:
            raise AssertionError(
                f"a read of {target} survived the seam rewrite; the constitutive half "
                f"would need D bound a second time, which is the aliasing hazard this "
                f"signature exists to avoid")

    # THE ONE RENAME, THE OWNERSHIP GUARD AND THE CARRY, on one line each. The
    # certified statement is unchanged inside the brace: what is new is that a thread
    # standing on a cell a fill images does not run it at all (its E and f_w_E belong
    # to that cell's source thread), and that a source thread runs it again at every
    # ghost it owns. Matched as a whole line so a moved argument list is a named
    # failure rather than a partial substitution.
    for target, (axis, coordinate) in enumerate(zip(_CARRIED, _COORDINATE)):
        prefix = (f"    constitutive_apply(E{axis}, f_w_E{axis}, idx, src_{axis}, ")
        call = _line_starting(tail, prefix, f"E{axis}'s constitutive_apply")
        expected = f"{prefix}kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);"
        if call != expected:
            raise AssertionError(
                f"E{axis}'s constitutive_apply is {call!r}, not the certified "
                f"{expected!r}; the sub-lattice rename would be applied to an "
                f"argument list this module has not read")
        renamed = call.replace(f"kms_{axis}[", f"kms_half_{axis}[", 1)
        tail = tail.replace(call, "\n".join(
            [f"    if (own_{axis}) {{", f"        {renamed.strip()}"]
            + fill_carry_blocks(target)
            + ["    }"]), 1)
    for axis, coordinate in zip(_CARRIED, _COORDINATE):
        if f"kms_{axis}[{coordinate}])" in tail:
            raise AssertionError(
                f"kms_{axis} survived the sub-lattice rename; it is the INTEGER "
                f"vector in the curl body and the HALF-INTEGER one here, and the two "
                f"would collide on the bare name")
    return tail


def fused_electric_pair_source() -> str:
    """The whole fused kernel: two lifted preludes, two lifted bodies, one seam.

    PURE ASCII, and that is a COMPILE REQUIREMENT rather than a style rule:
    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source with a bare
    ``open(..., 'w')``, so the bytes go through the interpreter's locale encoding --
    ASCII under C/POSIX, which is what a non-interactive shell on the validation host
    gets. Two em-dashes in a comment once killed a kernel at its first launch with
    ``UnicodeEncodeError``.
    """
    source = "".join((
        fused_electric_pair_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        pre_clear_registers(),
        zero_metal_carry(),
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml:2065) --\n"
        "    // Its three sources are the registers above, not a reload of D, and\n"
        "    // each runs again at every ghost cell this thread owns.\n",
        certified_constitutive_body(),
        "}\n",
    ))
    try:
        source.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AssertionError(
            f"the fused device source is not pure ASCII ({exc}); NVRTC's source file "
            f"is written through the interpreter's locale encoding and this would "
            f"fail at first launch, not at import") from exc
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by kernel name -- for a digest.

    ONE ENTRY: this family has no arm and no specialisation. It is a dict rather than
    a string so a digest walker treats it exactly as it treats its complex siblings'.
    """
    return {KERNEL_NAME: fused_electric_pair_source()}


#: NVRTC compile options -- CORRECTNESS, not performance, and identical to both
#: certified halves'. Spelled here rather than imported so that loading this file by
#: path (which the bit-identity probe does) cannot pick up a different tuple than the
#: one the gate compiled.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one cell per lane -- what every real sibling landed on.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str = KERNEL_NAME):
    """Compile the fused kernel, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING, for the reason every
    sibling rebuilds its code map per call: the source is part of the memo key, so it
    has to be built before there is a key to miss on, and a gate mutates this family
    by monkeypatching the certified halves. A source memoized at first call would hand
    back the pre-mutation string forever -- a leg reporting a pass for a mutation it
    never applied.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    code = fused_electric_pair_source()
    key = compile_cache.kernel_cache_key(name, True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_fused_electric_pair(fields: Any, pml: Any, grid: Any,
                               sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``zero_metal_D`` -> ``update_E``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal, which is
    this directory's convention.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own
    certified predicate refuses is refused here with that half's reason, prefixed so a
    reader can tell which side said it. What this predicate ADDS is the seam clauses:
    the source slot, the two UNCARRIED fills, the injection route, and the grid's
    ability to say which axes are walled.
    """
    covered, reason = covers_real_pml_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_constitutive(fields, pml, grid, "E")
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE TWO FILLS, CARRIED SINCE 2026-08-31 -- and this is where what the carry does
    # NOT reach is refused BY NAME. Until this round both were refused outright, on a
    # reason that was true of the NAIVE carry and only of it: the destination thread
    # reading row 2 (near) or `reflect_row` (far) reads a word another block writes in
    # the same launch, and an ordinary CUDA launch has no grid-wide barrier. THE
    # OWNERSHIP INVERSION REMOVES THAT READ ENTIRELY rather than synchronising it --
    # the SOURCE thread writes the destination from a live register -- so the clause
    # below is no longer about the fold, it is about whether the two fills' own
    # arithmetic is one this launch implements. 53 of this cell's 79 corpus rows sat
    # outside the predicate on the old clause, measured, and they are the whole of
    # this round.
    #
    # DELEGATED, NOT RESTATED. `in_seam_coverage` is the single place each fill's
    # clauses live (it is the predicate for the certified stand-alone passes this
    # carry lifts), so a clause added there reaches this seam without a second edit.
    # What is asked of the D family only: the B side's fills sit in the other seam and
    # are not this pair's problem.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_D and "
                       "fill_folded_far_ghosts_D run inside it")
    covered, reason = covers_fill_symmetry(fields, grid, "D")
    if not covered:
        return False, (f"fill_symmetry_bc_D runs inside this seam (driver.py:3309) "
                       f"and this carry cannot serve it: {reason}")
    covered, reason = covers_fill_folded_far(fields, grid, "D")
    if not covered:
        return False, (f"fill_folded_far_ghosts_D runs inside this seam "
                       f"(driver.py:3311) and this carry cannot serve it: {reason}")
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry != any(folded):
        # Fail closed, mirroring the curl predicate's own clause: a grid that
        # disagrees with itself about whether it is folded is a grid whose ghost rule
        # nothing here has resolved, and the carry is decided PER AXIS.
        return False, ("the grid reports has_symmetry() and no mirrored axis, or the "
                       "reverse; this carry is decided per axis and cannot be read "
                       "off a grid that disagrees with itself")

    # THE INJECTION ROUTE, REFUSED BY NAME. The driver has TWO ways to deposit an
    # electric source in this seam: the plain `source.inject` loop (driver.py:3308)
    # and `_inject_electric_through_conductivity` (:3305), which snapshots the D
    # components, injects, and then rescales only the increment by `condinv`
    # (MEEP step.cpp:294-317). The curl half above refuses a conductivity on a D
    # TARGET, which is a different question: `fields.has_conductivity` is the whole
    # engine's, so a magnetic-side sigma alone would route the electric deposit
    # through the scaled path while leaving step_D admissible. The deposit repair has
    # never been measured against that route, and a repair that reconstructed the
    # unscaled increment would be a silently wrong field at every deposit point.
    if getattr(fields, "has_conductivity", False):
        return False, ("the engine carries a conductivity, so the driver deposits "
                       "this seam's electric sources through "
                       "_inject_electric_through_conductivity (driver.py:3305), "
                       "which rescales the increment by condinv; the deposit repair "
                       "has no verdict on that route")

    # THE SOURCE SEAM, CARRIED RATHER THAN REFUSED. An ELECTRIC source is injected
    # BETWEEN the two halves (driver.py:3305/:3308), so a fused pair computes update_E
    # against a pre-injection D; `fused_pairs._install_fused_pair` brackets the launch
    # with LeadingRepairPlan/TrailingRepairPlan, CARRIES_DEPOSIT_REPAIR says so, and
    # the shared clause then asks what the repair can actually reconstruct. IGNORANCE
    # IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a predicate
    # that inferred "no sources" from not being told would be exactly the
    # over-covering this clause exists to prevent. A MAGNETIC source is injected in
    # the B/H half and does not disqualify this pair.
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
    # A grid that cannot answer would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_D cannot be "
                           f"carried in registers")
    try:
        walls = zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE THREE CLAUSES THE CARRY ADDS OVER THE TWO FILLS' OWN PREDICATES. Each is a
    # fact the stand-alone passes never had to ask, because each launched over a plane
    # of its own AFTER the curl had finished.
    #
    # 1. NO AXIS MAY BE BOTH FOLDED AND WALLED. `zero_metal_axes` is `is_metallic and
    #    not is_mirrored` (stepping.py:2284-2286), so this is unreachable from a real
    #    Grid -- and the carry RESTS on it TWICE on this family. A D component's near
    #    axes ARE the axes zero_metal_D clears it on, so if one were both folded and
    #    walled the near ghost would land on a cell the clear also owns, at stored 0 of
    #    a folded axis, and the emitted kernel writes no wall line at a ghost. It is
    #    also what makes the source thread's own clear flag the DESTINATION's test in
    #    `fill_carry_blocks`.
    for axis in range(3):
        if folded[axis] and bool(walls[axis]):
            return False, (
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction and the "
                f"near carry writes no wall line at the ghost it images")
    # 2. THE STORED SHAPE THE KERNEL INDEXES MUST BE THE ONE THE WALL PLANE AND THE
    #    FILL ROWS ARE DERIVED FROM. `nx, ny, nz` come from `fields.Dx.shape`; the wall
    #    is stored cell 0 of a `grid.stored_cells` axis and `folded_far_rows` is
    #    derived from the same reading. If the two ever disagree the wipe and the
    #    reflect row are right for an array this launch is not walking.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Dx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks "
                       f"Dx.shape {extents}; the wall plane and the two fills' rows "
                       f"are derived from the first and indexed into the second")
    # 3. A FOLDED AXIS MUST BE LONG ENOUGH TO HOLD THE NEAR FILL'S SOURCE ROW.
    #    `stepping._mirror_source` raises on a shorter one (stepping.py:1584-1588);
    #    this kernel would read outside the volume instead, which is why the clause is
    #    stated here rather than left to the array path's own raise.
    for axis in range(3):
        if folded[axis] and extents[axis] <= NEAR_SOURCE_INDEX:
            return False, (
                f"folded axis {axis} stores {extents[axis]} cells, so the near fill's "
                f"source row {NEAR_SOURCE_INDEX} does not exist "
                f"(stepping._mirror_source raises on it)")

    # THE THREE INVERSE-PERMITTIVITY VOLUMES MUST BE ASKABLE AND MUST MATCH THE SHAPE
    # THIS LAUNCH WALKS. `Fields.inverse_epsilon_for` is what the certified update_E
    # binds (constitutive_kernels._update_E_fused_pml_real); a component whose volume
    # is a different extent would be indexed with this launch's flat idx.
    accessor = getattr(fields, "inverse_epsilon_for", None)
    if not callable(accessor):
        return False, ("fields does not expose inverse_epsilon_for; update_E's three "
                       "inverse-permittivity volumes cannot be bound")
    for component in ("Ex", "Ey", "Ez"):
        try:
            volume = accessor(component)
        except Exception as exc:  # noqa: BLE001
            return False, (f"fields.inverse_epsilon_for({component!r}) raised: "
                           f"{type(exc).__name__}: {exc}")
        if volume is None:
            return False, f"fields.inverse_epsilon_for({component!r}) is None"
        shape = tuple(int(n) for n in getattr(volume, "shape", ()))
        if shape != extents:
            return False, (f"inverse_epsilon_for({component!r}) has shape {shape} "
                           f"but the launch walks {extents}; the kernel indexes it "
                           f"with this launch's flat index")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The fifteen ``__restrict__`` volumes, in SIGNATURE ORDER. Nothing else keeps the
#: launch and the kernel in step, so the order is spelled once and both the binding
#: check and the launch read it. The three inv_eps pointers are NOT here: they are
#: bound separately because they may legitimately alias each other.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz",
    "fu_Dx", "fu_Dy", "fu_Dz",
    "Hx", "Hy", "Hz",
    "Ex", "Ey", "Ez",
    "f_w_Ex", "f_w_Ey", "f_w_Ez",
)

#: The three E components whose inverse permittivity ``update_E`` multiplies by, in
#: signature order.
_INVERSE_EPSILON_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: The curl's INTEGER coefficient keys and the constitutive's HALF-INTEGER ones, in
#: signature order. THE PAIRING IS THE OPPOSITE OF THE B/H SEAM'S and the kernel
#: cannot tell -- a swap on either group is a half-cell error in the absorber profile,
#: converged, smooth and wrong.
_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def fused_electric_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads.

    THE SUB-LATTICE IS DECIDED HERE, not by the caller: while it was an argument,
    every call site was one more place the pairing could be got backwards, and
    backwards is a half-cell error rather than a crash. The D curl reads the INTEGER
    positions (``real_pml_curl_tables(pml, False)``) and ``update_E`` the HALF-INTEGER
    ones (``real_constitutive_tables(pml, True)``, stepping.py:1015) -- the opposite
    pairing to :mod:`.fused_magnetic_pair`'s.
    """
    _certified_halves()
    return {
        "curl": step_curl_kernels.real_pml_curl_tables(pml, False),
        "constitutive": constitutive_kernels.real_constitutive_tables(pml, True),
    }


def fused_electric_pair_fills(grid: "Grid") -> Dict[str, Tuple[Any, ...]]:
    """The two fills' launch plan: near flags, far reflect rows, mirror phases.

    THE SAME THREE READINGS ``in_seam_coverage.plan`` GIVES THE STAND-ALONE PASSES,
    packed as the flat per-axis triples the kernel takes. Read from that module rather
    than from the grid directly, so the carry and the certified passes cannot disagree
    about which axes each fill visits.

    ``near[a]`` is 1 on a MIRROR-folded axis of either termination -- the near fill's
    own condition is a declared mirror phase (stepping.py:1484-1485), which a folded
    METALLIC axis has just as a folded PERIODIC one does.

    ``reflect[a]`` is ``-1`` where ``fill_folded_far_ghosts_D`` does not run, and that
    sentinel is deliberate: it is a value the kernel never reads (the guard that would
    read it is ``reflect_a >= 0``), and a source that read it anyway would index
    outside the volume and be caught rather than silently imaging row 0.

    RAISES RATHER THAN RETURNING A PLAN IT CANNOT STAND BEHIND. A launcher handed a
    grid the predicate would have refused must not quietly build a plan for it: the
    failure mode is a plane of wrong values, not an exception, so the two facts the
    carry rests on are asserted here as well as refused there.
    """
    phases = mirror_fill_phases(grid)
    rows = folded_far_rows(grid)
    walls = zero_metal_axes(grid)
    for axis in range(3):
        if phases[axis] is not None and int(phases[axis]) not in (1, -1):
            raise ValueError(
                f"axis {axis} is folded but its mirror phase is {phases[axis]!r}; a "
                f"plane's parity is +1 or -1 and the even-mirror default standing in "
                f"for a plane that declared otherwise is a run wrong by twice the "
                f"field wherever the parity mattered")
        if phases[axis] is not None and bool(walls[axis]):
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction "
                f"(stepping.py:2284-2286) and this kernel's ghost carry relies on the "
                f"two sets being disjoint")
        if rows[axis] is not None and phases[axis] is None:
            raise ValueError(
                f"axis {axis} carries a far reflect row {rows[axis]!r} and no mirror "
                f"phase; stepping._fill_folded_far_ghosts weights that image with the "
                f"plane's parity and cannot run without one")
    return {
        "near": tuple(int(phases[axis] is not None) for axis in range(3)),
        "reflect": tuple(-1 if rows[axis] is None else int(rows[axis])
                         for axis in range(3)),
        "phase": tuple(0.0 if phases[axis] is None else float(phases[axis])
                       for axis in range(3)),
    }


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three per-component inverse-permittivity volumes, in signature order.

    ``Fields.inverse_epsilon_for`` and never ``fields.inv_eps``, which is the Ez view
    (fields.py:1259-1260) -- binding it for all three is the defect the certified
    ``update_E_pml_real`` docstring names by name.
    """
    return tuple(  # type: ignore[return-value]
        fields.inverse_epsilon_for(component)
        for component in _INVERSE_EPSILON_COMPONENTS)


def assert_disjoint_bindings(fields: "Fields",
                             tables: Dict[str, Dict[str, Any]]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes.

    ONCE PER FROZEN CONFIGURATION, OFF THE LAUNCH PATH. The shared flux density is
    bound once by construction -- the signature has no second D group to hand it to --
    so what is left to check is that no OTHER two restrict arguments are the same
    allocation. Two ``__restrict__`` pointers to one object is UB whatever the route
    to it, and NVRTC reorders across it without a diagnostic.

    THE THREE inv_eps VOLUMES ARE CHECKED DIFFERENTLY AND DELIBERATELY. They are not
    ``__restrict__`` in the signature, so they may alias EACH OTHER -- an isotropic run
    hands one pointer three times and the certified kernel is built for it. What they
    may NOT do is alias a restrict-qualified argument, which would break that
    argument's promise from the other side, so that is the check they get.

    Returns the number of distinct allocations checked, so a caller can assert that
    something was actually inspected; a check that examined nothing and reported
    success is the vacuity this whole track guards against.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in _FIELD_BINDINGS:
        visit(name, getattr(fields, name))
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
            "the fused electric pair binds every field and table argument "
            "__restrict__, and these arguments alias, which is undefined behaviour "
            "NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_fused_electric_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], walls: Sequence[int],
        fills: Dict[str, Tuple[Any, ...]], dtdx: float,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch.

    ``boundary_codes`` are ``step_curl_kernels.real_curl_boundary_codes(grid)``;
    ``walls`` are ``in_seam_coverage.zero_metal_axes(grid)``; ``fills`` is
    :func:`fused_electric_pair_fills`. THE THREE READINGS ANSWER DIFFERENT QUESTIONS
    and confusing them is a plane of wrong values rather than a crash: the codes are
    ``_boundary_kinds``' ghost-rule resolution (where a folded METALLIC axis is
    indistinguishable from a plain wall), the walls are ``_zero_metal``'s own
    declaration (with a folded metallic axis EXCLUDED), and the fills are the two
    mirror passes' (where a folded metallic axis DOES carry the near fill and does NOT
    carry the far one).

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is: a gate
    compiles a deliberately broken copy of the shipped source and hands it here. A
    launcher that could not be handed its own kernel could not arm a single mutation,
    and every mutation leg would silently launch the shipped one and report the defect
    as uncaught.

    Returns the launch geometry rather than ``None`` so a gate can assert that
    something was actually launched.
    """
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + tuple(
        inverse_epsilon_bindings(fields)
    ) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + (
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3)) + tuple(
        np.int32(int(fills["near"][axis])) for axis in range(3)
    ) + tuple(
        np.int32(int(fills["reflect"][axis])) for axis in range(3)
    ) + tuple(
        np.float32(float(fills["phase"][axis])) for axis in range(3))
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "fills": {key: tuple(fills[key]) for key in ("near", "reflect", "phase")}}


def run_fused_electric_pair(fields: "Fields", grid: "Grid", pml: "PML", dtdx: float,
                            *, sources: Any = None,
                            tables: Optional[Dict[str, Dict[str, Any]]] = None,
                            kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``tables`` and ``kernel`` are the gate's doors, keyword-only. Passing ``tables`` is
    how a swapped-sub-lattice mutation is armed: a launcher that always derived its own
    could not be handed mis-paired ones.
    """
    covered, reason = covers_fused_electric_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = fused_electric_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    return launch_fused_electric_pair(
        fields, tables, step_curl_kernels.real_curl_boundary_codes(grid),
        zero_metal_axes(grid), fused_electric_pair_fills(grid), dtdx, kernel)


# The in-seam module is imported for one reason and it is worth naming: this file
# CARRIES ``zero_metal_D``, and a reader checking that claim should find the certified
# text one attribute away rather than in another directory. Nothing below reads it at
# runtime.
_IN_SEAM_REFERENCE = getattr(in_seam_passes, "KERNEL_FOR", None)
