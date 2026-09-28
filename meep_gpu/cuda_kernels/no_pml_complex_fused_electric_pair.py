"""The COMPLEX NO-ABSORBER hand-CUDA fused pair: ``step_D`` -> ``update_E``.

THE BOARD'S CELL. ``D_to_E (cuda_complex_no_pml/complex no-PML curl,
cuda_complex_no_pml/complex no-PML stored E)``, 4 seam-instances of demand of which
ALL FOUR clear the source seam on the driver fact alone AND carry NOTHING in the seam
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-08-31_fillcarry/board.log``, rank 9:
"reach 4 (of 4 rows) ... carries nothing in the seam on 4 of those; the rest need
nothing"). That is the whole shape of this product and it is why it is the SIMPLEST
weld on the board: no deposit to repair, no wall to clear, no fill to carry, no
split-field auxiliary, no coefficient vector, no sub-lattice pairing.

=============================================================================
WHAT THE FOUR ROWS ACTUALLY ARE, MEASURED RATHER THAN ASSUMED
=============================================================================

Read out of the census this product is priced against
(``cuda_predicate_coverage_2026-08-31_fillcarry3``), all four rows are
``tests:TestLoadDump.test_load_dump_*_3d``:

* ``source_field_types == ['B']`` on 4 of 4 -- a MAGNETIC source only, injected in the
  B/H seam and not in this one. Hence :data:`CARRIES_DEPOSIT_REPAIR` is ``False``, and
  it is a MEASUREMENT rather than a default; see that constant's own note.
* ``plan_step.live`` carries ``fill_B`` and no ``fill_D``, no ``zero_metal_D`` and no
  ``fill_folded_far_ghosts_D`` on 4 of 4 -- the D seam is empty. Hence
  :data:`REPLACES` is a CONTIGUOUS run of driver order and carries no in-seam pass;
  see that constant's own note.
* ``has_conductivity == True`` and the composer's own recorded ``arm`` is
  ``"conductive"`` on 4 of 4, at ``step_B`` and at ``step_D`` both.

THAT LAST FACT DECIDES THE SHAPE OF THIS MODULE. The certified family emits TWO D-side
curls -- ``step_D_no_pml_complex`` and ``step_D_no_pml_complex_conductive``, both in
``complex_no_pml_kernels.CERTIFIED_KERNELS`` -- and every row this cell reaches takes
the SECOND one. A product welding only the plain curl would compile, gate green on
every leg it ran, and serve 0 of the 4 rows; a product welding the plain curl behind a
predicate that admitted a conductive run would be a silently wrong simulation, because
the conductive tail's ``field *= condfac; field -= curl; field *= condinv`` differs
from the plain ``field -= curl`` on essentially every word. So BOTH ARMS ARE BUILT and
:func:`no_pml_complex_curl_arm` -- the certified family's own classifier -- decides
which, per run, from ``fields.condfac_for``.

=============================================================================
WHY IT IS A SEPARATE MODULE AND NOT AN ARM OF THE COMPLEX PML PAIR
=============================================================================

:mod:`.complex_fused_electric_pair` welds the SPLIT-FIELD ``step_D`` to the split-field
``update_E`` under an ACTIVE absorber. This welds the no-absorber pair. They share the
``complex_emitter`` prelude and not a byte of spliced text: that one carries ``fu``,
two coefficient vector groups per half, a sub-lattice pairing that is the mirror image
of the B/H seam's, an ownership mask and a ghost rule; this one carries none of them,
because without a layer ``stepping._apply_curl`` takes its FOURTH tail (stepping.py:539)
and ``stepping.update_E`` takes its no-PML store (:1013-1022). What this one carries that
the PML pair does not is the POLE BANK: ``update_E``'s source is
``displacement_minus_polarization``, D minus every driving P in registration order.

THE TWO OCCUPY THE SAME SEAM AND ARE STILL DISJOINT, BY THE ABSORBER. Every other
``step_D`` product's predicate refuses a run with no active layer BY NAME -- the real
pair through ``coverage.covers_real_pml_curl`` (coverage.py:1140-1141, "no active PML
layer"), the complex pair through ``coverage._complex_grid_refusal`` (:2785, the same
words), the Dcyl pair through ``cylindrical_coverage.covers_pml_cylindrical_complex_curl``
(its own copy of the clause) -- and :func:`covers_no_pml_complex_fused_electric_pair`
REQUIRES the mirror image of it, refusing every run that HAS one. So the four
candidates on this seam partition on a single boolean and no configuration can admit
two. That matters rather than being tidy: ``fused_pairs.install_fused_pairs`` leaves a
seam UNFUSED naming both when two products claim it, so an overlap here would cost the
slots this module exists to buy.

=============================================================================
WHAT ONE LAUNCH PERFORMS, AND WHAT IT DOES NOT
=============================================================================

:data:`REPLACES` is ``step_D`` (driver.py:3302) -> ``update_E`` (:3313), declared
rather than inferred from the two slots, and it is a CONTIGUOUS RUN of driver order
because every pass the driver puts between those two consults is refused by name:

* the electric injections (:3305, :3308) -- refused by the seam-source clause, since
  :data:`CARRIES_DEPOSIT_REPAIR` is ``False``. That refusal covers BOTH routes,
  including ``_inject_electric_through_conductivity`` (:3305), which the driver takes
  only when there IS an electric source (``if electric and
  self.fields.has_conductivity``);
* ``fill_symmetry_bc_D`` (:3309) and ``fill_folded_far_ghosts_D`` (:3311) -- refused
  with the fold and symmetry clauses, inherited from ``_complex_grid_refusal`` and
  restated in this file so the guarantee does not rest on a clause another module
  could licence away;
* ``zero_metal_D`` (:3310) -- refused by a WALL clause this file adds. See
  :data:`REPLACES`, which states the decision and its cost.

=============================================================================
THE ALIASING HAZARD, AND THE TWO ANSWERS TO IT
=============================================================================

The curl half writes D in place and the constitutive half reads it. Binding that one
allocation twice -- once writable for the curl, once as a source for the constitutive
half -- is undefined behaviour NVRTC miscompiles without a diagnostic. So D appears
EXACTLY ONCE in the emitted signature, and there are TWO places it could have crept
back in:

1. The constitutive half's ``minus_poles(g0, ...)`` reads D through ``g``. The seam
   replaces the whole ``g`` parameter with the REGISTER the curl already holds, which
   removes this half's only use of ``g0/g1/g2`` outright.
2. The certified ``update_E`` launcher binds every UNUSED pole slot to the SOURCE
   pointer -- which is D -- and never dereferences it. Doing that here would bind D a
   second time. This launcher binds the unused slots to ``inv_eps_{index}`` instead:
   also never dereferenced (the runtime ``np`` guard is false for them), also already
   in the signature, and NOT ``__restrict__``, so no promise is made about it that a
   second binding could break.

:func:`assert_disjoint_bindings` checks both, off the launch path, and returns the
number of allocations it inspected so a caller can assert that something was looked at.

=============================================================================
THE NAME COLLISIONS THE SPLICE MUST RESOLVE
=============================================================================

The two certified bodies come from ONE template family, so they use the same local
names for different things. Spliced verbatim into one scope:

* ``f0/f1/f2`` is D in the curl body and E in the constitutive body. A collision here
  is a kernel that does not compile, which is the harmless half; it is renamed anyway,
  as a whole line, so a moved store is a named failure rather than a partial edit.
* ``g0/g1/g2`` is B in the curl body (``Fields.get_H`` returns the B arrays themselves
  without an absorber, fields.py:1164-1187) and D in the constitutive one. THIS IS THE
  SEAM and it is resolved by removing the second use entirely, per (1) above.
* ``n_elem`` is the constitutive guard's bound and ``nx * ny * nz`` is the curl's. The
  constitutive index decomposition and its guard are DROPPED -- the curl's have already
  run -- so ``n_elem`` is not a parameter of this kernel at all. The predicate checks
  that the two bounds would have agreed, by requiring ``Ex.shape``, ``Dx.shape`` and
  ``grid.stored_cells`` to be one tuple.

WHAT IS *NOT* HERE, and each is absent because the certified constitutive half says so
in its own words: NO ghost rule, NO ownership mask, NO phase and NO ``f_w``. "This
branch writes every cell of the volume, reads no neighbour, and without an absorber
there is no auxiliary to keep." Anyone porting the split-field pair into this one will
reach for the last of them.

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
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid

# THE CERTIFIED TEXT. ``complex_emitter`` imports nothing and runs at the merge bar,
# and ``complex_no_pml_kernels`` imports CuPy only inside its launchers, so both
# imports are unconditional and the emitter below works on a laptop.
try:
    from . import complex_emitter
    from . import complex_no_pml_kernels as certified
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
    certified = _load("complex_no_pml_kernels")
    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")

# The source-seam clause, shared with every other fused-pair predicate on all three
# tracks. Imported rather than re-spelled: writing it out per product produced a real
# bug once (a Metal pair asked about the wrong seam and would have refused a magnetic
# source while ADMITTING an electric one).
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
#: record behind it. A fused product's bit-identity is a claim about the WELD, which no
#: verdict on either half establishes, so this stays empty until the fused gate has run
#: on a device.
#: RELEASED ON DEVICE 2026-09-11 by the campaign ``certification.json:cuda_no_pml_complex_fused_electric_pair_2026-09-11_regate``
#: (parity/meep_gpu/gate_cuda_fused_complex_pairs.py --family no_pml_complex_electric, keep and flush legs, 14/14 cases identical, 17/17 mutations caught).
#: Moved here from ``UNCERTIFIED_KERNELS`` by that verdict and by nothing
#: else: the gate ran BEFORE this edit, on the bytes that ship, and the
#: weld in ``cuda_kernels/fingerprints.json`` binds them.
CERTIFIED_KERNELS = (
    "fused_electric_pair_no_pml_complex",
    "fused_electric_pair_no_pml_complex_conductive",
)
#: Shipped without a gate verdict. The kernels are here to BE gated, and there are TWO
#: of them because the certified curl has two arms and every corpus row on this cell
#: takes the conductive one.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its fused launch with the deposit repair? FALSE, AND IT IS
#: A MEASUREMENT.
#:
#: WHAT WAS MEASURED. All four rows of this product's board cell declare
#: ``source_field_types == ['B']`` -- a magnetic source and nothing else. The driver
#: injects a magnetic source between ``step_B`` and ``update_H`` (driver.py:3293), one
#: seam earlier; the D seam it would have to carry is EMPTY on 4 of 4. So the flag at
#: ``True`` would buy exactly ZERO rows here, while requiring
#: ``LeadingRepairPlan``/``TrailingRepairPlan`` machinery around every launch that
#: nothing on this cell exercises.
#:
#: THE SIBLING IS TRUE FOR THE OPPOSITE REASON and the contrast is the point:
#: ``complex_fused_electric_pair`` carries the repair because 15 of its 16 rows declare
#: an ELECTRIC source inside the seam, so on THAT cell the flag is the product. Here it
#: is not, and copying the sibling's ``True`` would have been the defect
#: ``deposit_repair`` exists to make impossible -- the flag and the wiring change
#: together or not at all (``deposit_repair.py:221-226``), and this module installs no
#: wiring.
#:
#: WHAT THE FLAG AT FALSE COSTS: every in-seam ELECTRIC deposit is refused by name, so
#: a run that grew one would fall back to the array path rather than be served wrongly.
#:
#: WHAT IS CHECKED, AND WHERE. The 0-of-4 is a census number and the census is a
#: gitignored artifact, so no merge-bar test may read it -- what the host suite pins is
#: the CONSEQUENCE, which is checkable without one:
#: ``test_the_flag_at_False_refuses_every_deposit_the_installer_could_have_bracketed``
#: requires the predicate's verdict and ``deposit_repair.in_seam_sources`` to agree in
#: BOTH directions over a source matrix (admitted implies the seam is empty; a
#: non-empty seam implies refused), which is exactly the property that makes the
#: ``False`` sound. ``test_the_flag_is_load_bearing_and_flipping_it_changes_the_answer``
#: is its null control. The gate's ``leg_source_refusal`` adds the device half: the
#: refusal is measured, and so is the DIVERGENCE a launch spanning the deposit produces.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_no_pml_complex_fused_electric_pair"

#: The kernel LABEL, in the bracket spelling this family already uses for its two-armed
#: curl (``registry._TABLE``'s ``fused_{slot}_no_pml_complex[_conductive]``). It is a
#: label and NOT a symbol: the two symbols are :data:`KERNEL_KEYS`' values and
#: :func:`kernel_name` is the one resolver, so nothing can hand this string to NVRTC by
#: accident and get a "kernel not found" three frames from the cause.
KERNEL_NAME = "fused_electric_pair_no_pml_complex[_conductive]"

#: The ``extern "C"`` symbol each curl arm's weld is emitted under. Keyed by the arm
#: name ``complex_no_pml_kernels.complex_no_pml_curl_arm`` returns, so the classifier
#: and this table cannot disagree about which of the two a run takes.
KERNEL_KEYS: Dict[str, str] = {
    "plain": "fused_electric_pair_no_pml_complex",
    "conductive": "fused_electric_pair_no_pml_complex_conductive",
}

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3302, :3313). Declared rather than inferred from the two slots.
#:
#: ``zero_metal_D`` IS NOT HERE, AND THAT IS A DECISION RATHER THAN AN OMISSION.
#: ``covers_no_pml_complex_fused_electric_pair`` REFUSES every walled run by name, so
#: on every row this product admits ``zero_metal_D`` (driver.py:3310) writes nothing at
#: all and the two consults it sits between are CONTIGUOUS.
#:
#: WHAT IT COST, MEASURED: 0 slots. The board's cell reports
#: ``reachable_rows_needing_each_in_seam_pass == {}`` -- not one of the four rows runs
#: the pass -- and all four record ``has_metallic == False`` with
#: ``metallic == [false, false, false]``. Carrying it would therefore have bought
#: nothing and cost a branch of device text that NO corpus row exercises, whose
#: correctness rests on the OFF-DIAGONAL table (``fields.IYEE_SHIFTS`` gives Dx
#: (1,0,0), so a D component sits on the walls of the OTHER TWO axes, two components
#: per wall -- ``in_seam_passes._zero_metal_D_kernel_code``:352-363). Reusing the B
#: family's DIAGONAL table there clears one wrong component and leaves two right ones
#: standing on every walled run: converged, smooth and wrong. A refusal cannot be
#: wrong that way, and a refusal is what a run that grows a wall gets.
#:
#: THE OTHER TWO IN-SEAM PASSES are absent for the same kind of reason and a stronger
#: clause: ``fill_symmetry_bc_D`` (:3309) and ``fill_folded_far_ghosts_D`` (:3311) do
#: nothing on an unfolded grid, and this predicate refuses every folded one -- twice
#: over, once through the curl half's inherited ``_complex_grid_refusal`` clause and
#: once in this file, so the guarantee that makes this constant honest does not rest on
#: a clause another module could licence away.
REPLACES: Tuple[str, ...] = ("step_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void no_pml_apply(float* f, int idx, "
             "cf curl) {",
     "became": "__device__ __forceinline__ cf no_pml_apply_reg(float* f, int idx, "
               "cf curl) {",
     "why": "the constitutive half consumes the value in a register; the helper names "
            "and returns the word pair it already computed. The expression, its "
            "parenthesisation and the store to f are unchanged, so the global state it "
            "leaves is the certified one."},
    {"line": "    cf_store(f, idx, cf_sub(cf_load(f, idx), curl));",
     "became": "    cf value = cf_sub(cf_load(f, idx), curl);\n"
               "    cf_store(f, idx, value);\n    return value;",
     "why": "same right-hand side, same tree, same store; the result is additionally "
            "named so it can be returned. A float32 word pair stored to global and "
            "reloaded is the identity on the bits, which is what makes the register "
            "hand-off exact rather than approximate."},
    {"line": "__device__ __forceinline__ void conductive_apply(",
     "became": "__device__ __forceinline__ cf conductive_apply_reg(",
     "why": "the conductive arm's tail, given the same treatment as the plain one. "
            "BOTH arms are built because every corpus row on this cell takes the "
            "conductive curl, and a product that welded only the plain one would "
            "serve none of them."},
    {"line": "    cf_store(f, idx, mul_field_left(t, condinv));",
     "became": "    cf value = mul_field_left(t, condinv);\n"
               "    cf_store(f, idx, value);\n    return value;",
     "why": "same right-hand side, same store, additionally named. THE ORDER OF THE "
            "THREE OPERATIONS IS UNTOUCHED: reassociating them differs on 400000 of "
            "400000 float32 words (measured, test_complex_no_pml.py), and this edit "
            "touches only the last statement's binding."},
    {"line": "        no_pml_apply(f0, idx, curl);",
     "became": "        d0 = no_pml_apply_reg(f0, idx, curl);",
     "why": "capture the register. Three lines, one per component; the argument lists "
            "are untouched. Three 'cf d* = cf_zero();' declarations are hoisted above "
            "the certified braced blocks because a value declared inside one does not "
            "outlive it."},
    {"line": "        conductive_apply(f0, idx, curl, condfac_0[idx], "
             "condinv_0[idx]);",
     "became": "        d0 = conductive_apply_reg(f0, idx, curl, condfac_0[idx], "
               "condinv_0[idx]);",
     "why": "the conductive arm's three capture sites, same edit as the plain arm's. "
            "The two conductivity volume reads and their order are untouched."},
    {"line": "__device__ __forceinline__ cf minus_poles(\n    const float* g,",
     "became": "__device__ __forceinline__ cf minus_poles_reg(\n    cf s,",
     "why": "THE SEAM. The certified helper opens by RELOADING the flux density from "
            "g; this variant takes it as the register the curl half already holds. It "
            "is also what removes the constitutive half's only use of g0/g1/g2, which "
            "is what lets D be bound EXACTLY ONCE and answers the __restrict__ "
            "aliasing hazard. Every one of the eight guarded subtractions below it, "
            "their LEFT-TO-RIGHT order and the return are untouched -- the "
            "registration order is bit-load-bearing (measured: pre-summing two poles "
            "differs on 50279 of 400000 words)."},
    {"line": "    cf s = cf_load(g, idx);",
     "became": "",
     "why": "removed, because it IS the new parameter. The value the certified line "
            "computed is the value the curl half just stored, and a float32 word pair "
            "stored and reloaded is the identity on the bits."},
    {"line": "        minus_poles(g0, a0, a1, a2, a3, a4, a5, a6, a7, idx, np0),",
     "became": "        minus_poles_reg(d0, a0, a1, a2, a3, a4, a5, a6, a7, idx, np0),",
     "why": "the three call sites of the seam, one per component. The "
            "inverse-permittivity multiply on the NEXT line, its operand order (the "
            "source on the left, stepping.py:1013-1014) and the enclosing cf_store are "
            "untouched by this edit."},
    {"line": "    cf_store(f0, idx, mul_field_left(",
     "became": "    cf_store(h0, idx, mul_field_left(",
     "why": "ONE RENAME, NO ARITHMETIC. f0 is D in the curl body and E here; letting "
            "one shadow the other is a kernel that does not compile, which is the "
            "harmless half of a name collision -- it is renamed as a whole line "
            "anyway, so a moved store is a named failure rather than a partial edit."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= n_elem) return;",
     "became": "",
     "why": "the constitutive half's index decomposition and bounds guard are DROPPED: "
            "the curl body above already declares idx and its own guard has already "
            "returned, and splicing both would redeclare idx. n_elem is therefore not "
            "a parameter of this kernel at all, and the predicate requires Ex.shape, "
            "Dx.shape and grid.stored_cells to be one tuple so the two bounds the "
            "certified kernels would have used are the same number."},
    {"line": "(the LAUNCHER, not device text) the certified update_E binds every "
             "UNUSED pole slot to the SOURCE pointer, which is D",
     "became": "this launcher binds an unused slot to inv_eps_{index}",
     "why": "binding D there would bind it a SECOND time, which is exactly the "
            "aliasing hazard the emitted signature exists to avoid. inv_eps is "
            "already in the signature, is NOT __restrict__ (an isotropic run "
            "legitimately hands one allocation three times) and is never dereferenced "
            "through the pole slot, because the runtime np guard is false for it. The "
            "device text is unchanged by this edit."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_KEYS",
    "KERNEL_NAME", "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body", "certified_curl_body",
    "covers_no_pml_complex_fused_electric_pair", "device_sources",
    "base_address", "inverse_epsilon_bindings", "kernel_name",
    "launch_no_pml_complex_fused_electric_pair", "no_pml_complex_curl_arm",
    "no_pml_complex_fused_electric_pair_prelude",
    "no_pml_complex_fused_electric_pair_source", "pole_bank_bindings",
    "run_no_pml_complex_fused_electric_pair",
]


# =============================================================================
# THE LIFT
# =============================================================================
#
# Every anchor below is an exact line of certified device text. If one stops matching,
# the certified string changed, and the splice raises rather than emitting a kernel
# that is quietly missing a store, a subtraction or the seam.

#: Separates a certified kernel's signature from its body. Both emitted strings close
#: their parameter list on its own line, so ONE anchor lifts either body.
_BODY_ANCHOR = "\n) {\n"

#: The last line of the CURL body's index decomposition. The curl half keeps the whole
#: decomposition; the three carried registers are declared immediately after it.
_DECODE_END = "    int i = idx / (ny * nz);\n"

#: The constitutive half's index decomposition and bounds guard, dropped WHOLE. Its
#: bound is ``n_elem`` and the curl's is ``nx * ny * nz``; the predicate is what makes
#: those the same number on an admitted run.
_CONSTITUTIVE_HEAD = ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                      "    if (idx >= n_elem) return;\n")

#: The certified plain tail helper, and the one edit that turns it into a value.
_PLAIN_APPLY_SIGNATURE = (
    "__device__ __forceinline__ void no_pml_apply(float* f, int idx, cf curl) {\n")
_PLAIN_APPLY_SIGNATURE_REG = (
    "__device__ __forceinline__ cf no_pml_apply_reg(float* f, int idx, cf curl) {\n")
_PLAIN_APPLY_STORE = "    cf_store(f, idx, cf_sub(cf_load(f, idx), curl));\n"
_PLAIN_APPLY_STORE_REG = (
    "    // THE ONE EDIT TO THIS HELPER: the right-hand side, its\n"
    "    // parenthesisation and the store are the certified ones; the value is\n"
    "    // additionally NAMED so the constitutive half can read it from a\n"
    "    // register instead of reloading the word pair from global memory.\n"
    "    cf value = cf_sub(cf_load(f, idx), curl);\n"
    "    cf_store(f, idx, value);\n"
    "    return value;\n")

#: The certified conductive tail helper, and the same edit. THE THREE OPERATIONS AND
#: THEIR ORDER ARE UNTOUCHED -- only the last statement's result is named.
_CONDUCTIVE_APPLY_SIGNATURE = "__device__ __forceinline__ void conductive_apply(\n"
_CONDUCTIVE_APPLY_SIGNATURE_REG = "__device__ __forceinline__ cf conductive_apply_reg(\n"
_CONDUCTIVE_APPLY_STORE = "    cf_store(f, idx, mul_field_left(t, condinv));\n"
_CONDUCTIVE_APPLY_STORE_REG = (
    "    // THE ONE EDIT TO THIS HELPER: the three operations, their ORDER and\n"
    "    // their operand orientations are the certified ones -- reassociating\n"
    "    // them differs on 400000 of 400000 float32 words. Only the last\n"
    "    // statement's result is additionally NAMED, so it can be returned.\n"
    "    cf value = mul_field_left(t, condinv);\n"
    "    cf_store(f, idx, value);\n"
    "    return value;\n")

#: THE SEAM ITSELF. The certified pole-subtraction helper opens by reloading the flux
#: density through ``g``; this variant takes it as a register instead, which removes
#: the constitutive half's only use of ``g0/g1/g2``.
_MINUS_POLES_SIGNATURE = ("__device__ __forceinline__ cf minus_poles(\n"
                          "    const float* g,\n")
_MINUS_POLES_SIGNATURE_REG = (
    "// THE SEAM. Character for character the certified minus_poles below the\n"
    "// signature: the eight guarded subtractions, their LEFT-TO-RIGHT order and\n"
    "// the return. What changed is where the first value comes from -- the\n"
    "// register the curl half just stored, rather than a reload of the word pair\n"
    "// it stored it into. A float32 word pair stored to global and reloaded is\n"
    "// the identity on the bits, and removing the g parameter is what lets D be\n"
    "// bound EXACTLY ONCE in the fused signature.\n"
    "__device__ __forceinline__ cf minus_poles_reg(\n"
    "    cf s,\n")
_MINUS_POLES_LOAD = "    cf s = cf_load(g, idx);\n"

#: The three component registers the seam carries, in target order.
_CARRIED: Tuple[str, ...] = ("0", "1", "2")

#: The pole bank letter each E component's slots are spelled with in the certified
#: ``update_E`` signature, in target order.
_POLE_STEMS: Tuple[str, ...] = ("a", "b", "c")

#: The three E components, in ``stepping.E_CONSTITUTIVE_TERMS`` order — the certified
#: family's own tuple, re-read rather than retyped.
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: This sub-step's displacement source per E component (``fields.py:1095``).
_DISPLACEMENT: Dict[str, str] = {"Ex": "Dx", "Ey": "Dy", "Ez": "Dz"}

#: The curl's source arrays. SPELLED AS B, because that is the array the kernel
#: receives: ``Fields.get_H`` returns the B array itself without an absorber
#: (fields.py:1164-1187) and ``stepping.step_D`` differences exactly it.
_CURL_SOURCES: Tuple[str, ...] = ("Bx", "By", "Bz")

#: The conductive arm's two extra parameter triples, lifted from the certified curl's
#: own ``_CURL_CONDUCTIVE_PARAMS``. NOT ``__restrict__``, and deliberately: MEEP
#: allocates a conductivity per (component, direction) and a run whose three targets
#: share one profile hands the same device pointer three times.
_CONDUCTIVE_PARAMS = (
    "    // NOT __restrict__, and lifted from the certified conductive curl rather\n"
    "    // than decided here: a run whose three targets share one profile hands the\n"
    "    // same device pointer three times, and restrict on mutually aliasing\n"
    "    // arguments is a promise the caller cannot keep. float32 VOLUMES indexed by\n"
    "    // the COMPLEX CELL index, never word-doubled.\n"
    "    const float* condfac_0, const float* condfac_1,\n"
    "    const float* condfac_2,\n"
    "    const float* condinv_0, const float* condinv_1,\n"
    "    const float* condinv_2,\n")

#: The fused entry point. The ONLY hand-written device text in this module, and it is a
#: signature: no arithmetic lives here. D (f0/f1/f2) appears EXACTLY ONCE — see the
#: module docstring's aliasing section, which is the reason the constitutive half has
#: no flux-density source pointers at all.
_SIGNATURE = r'''
extern "C" __global__ void __NAME__(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 D.
    // The curl half writes it and the constitutive half reads it from a register;
    // binding it a second time as a source would be two pointers to one allocation
    // with a __restrict__ promise on one of them, which is UB and which NVRTC
    // miscompiles without a diagnostic.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    // B, the curl's operands. Fields.get_H returns the B arrays themselves without
    // an absorber (fields.py:1164-1187), so this is spelled as what it receives.
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
__CONDUCTIVE__    // E, the constitutive target. There is no f_w beside it: without a layer the
    // stored E IS the constitutive product (fields.py:1150).
    float* __restrict__ h0, float* __restrict__ h1, float* __restrict__ h2,
    // NOT __restrict__, and lifted from the certified update_E rather than decided
    // here: an isotropic run hands the same device pointer three times
    // (fields.py:1323-1325). They stay FLOAT32 under complex storage and are indexed
    // by the COMPLEX CELL index, never word-doubled (stepping.py:41-50).
    const float* inv_eps_0, const float* inv_eps_1, const float* inv_eps_2,
    // The pole banks, MAX_POLES wide per component, in registration order. NOT
    // __restrict__ either, and for a sharper reason than inv_eps: the UNUSED slots
    // are bound to inv_eps itself and never dereferenced (the runtime np guard is
    // false for them), so two of these parameters may legitimately be one pointer.
__POLE_PARAMS__    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling. There is
    // no n_elem: the constitutive half's own guard is dropped and this one's bound is
    // the curl's, which the predicate requires to be the same number.
    int nx, int ny, int nz, float dtdx,
    int bc_x, int bc_y, int bc_z,
    // The Bloch table: a flag per axis and the rounded complex64 factor split into
    // words. THE IMAGINARY PART IS THE CONJUGATE ONE -- this is the BACKWARD sub-step
    // and bloch_phase_arguments(grid, True) negates it (stepping._shift_down,
    // :1818-1822). ph = 0 SKIPS the multiply entirely rather than doing it against
    // 1+0j, which is what keeps k = 0 bit-identical to the plain complex engine.
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,
    // The per-component pole COUNTS, which are the runtime guards on the banks above.
    int np0, int np1, int np2
) {
'''


def kernel_name(arm: str) -> str:
    """The ``extern "C"`` symbol NVRTC is asked for, by curl arm.

    A REFUSAL RATHER THAN A DEFAULT. The two arms are different arithmetic, both
    compile and both run, so an arm name this table does not carry is a wrong answer
    waiting to happen rather than a missing key.
    """
    if arm not in KERNEL_KEYS:
        raise ValueError(
            f"arm must be one of {sorted(KERNEL_KEYS)}, got {arm!r}; the plain and "
            f"conductive tails are different arithmetic and neither is a default")
    return KERNEL_KEYS[arm]


def no_pml_complex_curl_arm(fields: Any, sub_step: str = "step_D") -> str:
    """Which curl arm this run takes at ``step_D``. THE CERTIFIED CLASSIFIER, delegated.

    ``complex_no_pml_kernels.complex_no_pml_curl_arm`` reads
    ``fields.condfac_for(target)`` per TARGET, exactly as ``stepping._apply_curl``
    does (stepping.py:508). It is delegated rather than re-derived so the classifier
    the certified family ships and the arm this product compiles cannot disagree; a
    sub-step whose three targets DISAGREE raises there, and the predicate below refuses
    it by name before anything asks.
    """
    return certified.complex_no_pml_curl_arm(fields, sub_step)


def _split_body(source: str, prelude: str, name: str) -> str:
    """One emitted kernel string, reduced to its body.

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
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, not 1; "
            f"the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    A missing anchor is certified text that has changed under this family, and splicing
    around it would produce a kernel that compiles and is quietly not the certified
    arithmetic; TWO matches would mean the anchor no longer identifies a single
    statement.
    """
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


def _code_lines(body: str) -> List[str]:
    """``body``'s lines with whole-line ``//`` comments removed.

    Several checks below ask whether an IDENTIFIER survived a splice, and the comments
    this module splices in NAME the identifiers they explain -- a reader has to be told
    which parameter went and why. A substring test over the raw text would fire on that
    prose, so the checks that are about what COMPILES are made on what compiles.
    """
    return [line for line in body.splitlines() if not line.lstrip().startswith("//")]


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    """Substitute ``old`` exactly once, or raise naming what was being lifted."""
    if text.count(old) != 1:
        raise AssertionError(
            f"the certified text carries {text.count(old)} occurrences of {what}, not "
            f"1; this family LIFTS it rather than retyping it and cannot splice "
            f"around an anchor that no longer identifies a single site")
    return text.replace(old, new, 1)


def _prelude_for(arm) -> str:
    """The certified no-absorber prelude for one arm, exactly as this family builds it."""
    return certified._prelude(complex_emitter.normalized_expansion(arm))


def no_pml_complex_fused_electric_pair_prelude(arm) -> str:
    """The certified prelude with the three helpers the seam needs as values.

    ALL THREE ARE REWRITTEN ON BOTH ARMS, not only the one the arm calls: the prelude
    is then a function of the EXPANSION arm alone, which is one fewer thing for a
    digest, a memo key or a mutation to have to be right about. The unused one is dead
    ``__device__`` code, exactly as ``pml_apply`` and ``constitutive_apply`` already are
    in this certified prelude.
    """
    prelude = _prelude_for(arm)
    prelude = _replace_once(prelude, _PLAIN_APPLY_SIGNATURE,
                            _PLAIN_APPLY_SIGNATURE_REG, "no_pml_apply's signature")
    prelude = _replace_once(prelude, _PLAIN_APPLY_STORE, _PLAIN_APPLY_STORE_REG,
                            "no_pml_apply's store")
    prelude = _replace_once(prelude, _CONDUCTIVE_APPLY_SIGNATURE,
                            _CONDUCTIVE_APPLY_SIGNATURE_REG,
                            "conductive_apply's signature")
    prelude = _replace_once(prelude, _CONDUCTIVE_APPLY_STORE,
                            _CONDUCTIVE_APPLY_STORE_REG, "conductive_apply's store")
    prelude = _replace_once(prelude, _MINUS_POLES_SIGNATURE,
                            _MINUS_POLES_SIGNATURE_REG,
                            "minus_poles' signature (THE SEAM)")
    prelude = _replace_once(prelude, _MINUS_POLES_LOAD, "",
                            "minus_poles' reload of the flux density (THE SEAM)")
    # THE SEAM, CHECKED FROM THE OTHER SIDE. If a call to the certified helper survived
    # anywhere in the prelude, the constitutive half would still need D bound a second
    # time -- which is the aliasing hazard this whole shape exists to avoid.
    code = "\n".join(_code_lines(prelude))
    for name in ("no_pml_apply(", "conductive_apply(", "minus_poles("):
        if name in code.replace(f"{name[:-1]}_reg(", ""):
            raise AssertionError(
                f"a call to the certified {name[:-1]} survived the value rewrite; the "
                f"fused kernel would either write a value it never reads or need D "
                f"bound a second time")
    return prelude


def _pole_parameters() -> str:
    """The three ``MAX_POLES``-wide pointer banks, in component order.

    THE CERTIFIED EMITTER'S OWN FUNCTION, called rather than retyped: the bank width is
    ``complex_no_pml_kernels.MAX_POLES`` and a copy here would be a second number to
    keep in step with the launcher that fills them.
    """
    return certified._pole_parameters()


def _signature(arm_name: str, conductive: bool) -> str:
    """The fused signature for one curl arm."""
    source = _SIGNATURE.replace("__NAME__", arm_name)
    source = source.replace("__CONDUCTIVE__", _CONDUCTIVE_PARAMS if conductive else "")
    source = source.replace("__POLE_PARAMS__", _pole_parameters())
    if "__" in source.replace("__restrict__", "").replace("__global__", ""):
        raise AssertionError("an unsubstituted placeholder survived the signature")
    return source


def certified_curl_body(arm, conductive: bool) -> str:
    """``step_D_no_pml_complex[_conductive]``'s body, with the three registers captured.

    Not a transcription: this is what ``complex_no_pml_kernels.kernel_source`` emits,
    minus its prelude and signature. ``step_B``'s FORWARD shifts are a different product
    and never appear here -- the backward ``cshift_dn`` and the conjugated phase are what
    make this the D side.
    """
    key = "step_D_conductive" if conductive else "step_D"
    body = _split_body(certified.kernel_source(key, arm), _prelude_for(arm),
                       f"complex_no_pml_kernels.kernel_source({key!r})")
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
        tail,
    ))
    for target in _CARRIED:
        if conductive:
            old = (f"        conductive_apply(f{target}, idx, curl, "
                   f"condfac_{target}[idx], condinv_{target}[idx]);")
            new = (f"        d{target} = conductive_apply_reg(f{target}, idx, curl, "
                   f"condfac_{target}[idx], condinv_{target}[idx]);")
            what = f"target {target}'s conductive tail"
        else:
            old = f"        no_pml_apply(f{target}, idx, curl);"
            new = f"        d{target} = no_pml_apply_reg(f{target}, idx, curl);"
            what = f"target {target}'s plain tail"
        body = _replace_once(body, old, new, what)
        # Matched as a WHOLE LINE rather than by a `);` tail: a tail anchor would attach
        # to whichever statement happened to end first if the certified body ever moved
        # this call onto two lines.
        call = _line_starting(body, f"        d{target} = ", f"captured {what}")
        if not call.endswith(");"):
            raise AssertionError(
                f"the certified curl body no longer closes {what} on one line "
                f"({call!r})")
    for stem in ("no_pml_apply(", "conductive_apply("):
        if stem in body.replace(f"{stem[:-1]}_reg(", ""):
            raise AssertionError(
                f"a call to {stem[:-1]} survived the capture rewrite; its value would "
                f"be written to global memory and never read into the seam")
    # THE STENCIL IS ASSERTED, NOT EDITED. It is the arithmetic this family reuses
    # unchanged, so its survival is a property of the build rather than a claim.
    stencil = ("        cf curl = mul_coefficient_left(dtdx, "
               "cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));")
    if body.count(stencil) != 3:
        raise AssertionError(
            f"the certified curl body carries {body.count(stencil)} of the three curl "
            f"stencil lines; this family exists to reuse them unchanged")
    return body


def certified_constitutive_body(arm) -> str:
    """``update_E_no_pml_complex_stored``'s body, lifted, reading the registers.

    Drops the second index decomposition and bounds guard (the curl body above already
    declares ``idx`` and its guard has already returned), turns the three pole-bank
    sources into the seam, and renames the store target from D's ``f`` to E's ``h``.
    """
    body = _split_body(certified.kernel_source("update_E", arm), _prelude_for(arm),
                       "complex_no_pml_kernels.kernel_source('update_E')")
    body = _replace_once(
        body, _CONSTITUTIVE_HEAD,
        "    // The index decomposition and the bounds guard are the CURL half's,\n"
        "    // above: this half's own copy would redeclare idx, and its bound\n"
        "    // (n_elem) is the same number the curl's (nx * ny * nz) is on every\n"
        "    // run the predicate admits.\n",
        "the constitutive index decomposition and bounds guard")

    # THE SEAM. Each certified pole-bank call takes the flux density from the register
    # the curl already holds, which is also what removes this half's only use of
    # g0/g1/g2. The inverse-permittivity multiply on the following line and the
    # enclosing cf_store are untouched by this edit.
    for target, stem in zip(_CARRIED, _POLE_STEMS):
        slots = ", ".join(f"{stem}{slot}" for slot in range(certified.MAX_POLES))
        old = f"        minus_poles(g{target}, {slots}, idx, np{target}),"
        new = (f"        minus_poles_reg(d{target}, {slots}, idx, np{target}),"
               f"   // THE SEAM")
        body = _replace_once(body, old, new, f"target {target}'s pole-bank source")
    code = "\n".join(_code_lines(body))
    for target in _CARRIED:
        if f"g{target}" in code:
            raise AssertionError(
                f"g{target} survived the seam rewrite; the constitutive half would "
                f"need D bound a second time, which is the aliasing hazard this "
                f"signature exists to avoid")

    # THE INVERSE-PERMITTIVITY PRODUCTS ARE ASSERTED, NOT EDITED. They are the one part
    # of this body with no counterpart on the H seam, so a change to them would be
    # exactly the kind of drift that compiles: the line is READ and required to be the
    # certified one, and nothing is substituted into it.
    for target in _CARRIED:
        expected = f"        inv_eps_{target}[idx]));"
        line = _line_starting(body, f"        inv_eps_{target}[",
                              f"target {target}'s inverse-permittivity product")
        if line != expected:
            raise AssertionError(
                f"target {target}'s inverse-permittivity product is {line!r}, not the "
                f"certified {expected!r}; this family binds inv_eps_* on the strength "
                f"of that line and does not edit it")

    # THE ONE RENAME. Matched as a whole line so a moved store is a named failure
    # rather than a partial substitution.
    for target in _CARRIED:
        old = f"    cf_store(f{target}, idx, mul_field_left("
        line = _line_starting(body, old, f"target {target}'s constitutive store")
        if line != old:
            raise AssertionError(
                f"target {target}'s constitutive store is {line!r}, not the certified "
                f"{old!r}; the target rename would be applied to a statement this "
                f"module has not read")
        body = _replace_once(body, old, f"    cf_store(h{target}, idx, mul_field_left(",
                             f"target {target}'s constitutive store")
    for target in _CARRIED:
        if f"(f{target}," in "\n".join(_code_lines(body)):
            raise AssertionError(
                f"f{target} survived the target rename; it is D in the curl body and E "
                f"here, and the two would collide")
    # THE DROPPED BOUND, CHECKED ON CODE RATHER THAN ON PROSE. The comment spliced in
    # above NAMES ``n_elem`` on purpose -- a reader has to be told which bound went --
    # so a substring test over the whole body would fire on this module's own
    # explanation. Comment lines are stripped and the check is made on what compiles.
    survived = [line for line in _code_lines(body) if "n_elem" in line]
    if survived:
        raise AssertionError(
            f"n_elem survived in {survived!r}; the constitutive bound is dropped in "
            f"favour of the curl's and is not a parameter of this kernel")
    return body


def no_pml_complex_fused_electric_pair_source(arm, conductive: bool) -> str:
    """The whole fused kernel for one expansion arm and one curl arm."""
    if not isinstance(conductive, bool):
        raise TypeError(
            f"conductive must be a bool, got {type(conductive).__name__}; the two "
            f"tails are different arithmetic and a truthy value is not a choice")
    source = "".join((
        no_pml_complex_fused_electric_pair_prelude(arm),
        _signature(kernel_name("conductive" if conductive else "plain"), conductive),
        certified_curl_body(arm, conductive),
        certified_constitutive_body(arm),
        "}\n",
    ))
    # PURE ASCII IS A COMPILE REQUIREMENT, not a style rule; checked here so a mutation
    # leg that inserts a non-ASCII character is refused at emission with the reason
    # instead of at NVRTC three frames away.
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by ``arm/curl-arm`` — for a digest."""
    return {f"{name}/{key}": no_pml_complex_fused_electric_pair_source(
                name, key == "conductive")
            for name in sorted(complex_emitter.EXPANSIONS)
            for key in sorted(KERNEL_KEYS)}


#: NVRTC compile options — CORRECTNESS, not performance, and identical to both
#: certified halves'. Spelled here rather than imported so that loading this file by
#: path (which the bit-identity probe does) cannot pick up a different tuple than the
#: one the gate compiled.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane — what every complex sibling landed on.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm, conductive: bool):
    """Compile the fused kernel under one arm, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING, for the reason every
    sibling rebuilds its code map per call: the source is part of the memo key, so it
    has to be built before there is a key to miss on, and a gate mutates this family by
    monkeypatching the emitter. A source memoized at first call would hand back the
    pre-mutation string forever -- a leg reporting a pass for a mutation it never
    applied.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    name = kernel_name("conductive" if conductive else "plain")
    code = no_pml_complex_fused_electric_pair_source(arm, conductive)
    key = compile_cache.kernel_cache_key(
        f"{name}_arm{complex_emitter.normalized_expansion(arm)}", True,
        _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_no_pml_complex_fused_electric_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None, license: Any = None,
        subnormal_policy: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``update_E`` with no absorber?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal, which is
    this directory's convention.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own certified
    predicate refuses is refused here with that half's reason, prefixed so a reader can
    tell which side said it. What this predicate ADDS is the seam clauses: the source
    slot, the three UNCARRIED in-seam passes, and the shape agreement the dropped
    ``n_elem`` guard rests on.

    ``license`` and ``subnormal_policy`` are the complex family's arm arbitration and
    are passed straight through to both halves. They are REQUIRED in practice: the arm
    is compiled into the binary at this seam and there is no later rung to check it at,
    and a wrong arm is a wrong answer rather than a crash.
    """
    covered, reason = certified.covers_complex_no_pml_curl(
        fields, pml, grid, "step_D", license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = certified.covers_complex_no_pml_stored_e(
        fields, pml, grid, license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE ARM MUST BE CLASSIFIABLE, and it is asked HERE rather than at the launch
    # because the arm is compiled into the binary. The curl half already refuses a
    # sub-step whose three targets disagree; this is the same question asked of the
    # object this launch will actually compile against, and a raise is a refusal.
    try:
        arm = no_pml_complex_curl_arm(fields, "step_D")
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"the step_D curl arm could not be classified: "
                       f"{type(exc).__name__}: {exc}")
    if arm not in KERNEL_KEYS:
        return False, (f"the certified classifier returned arm {arm!r}, which names "
                       f"no kernel this product emits")

    # THE TWO MIRROR FILLS ARE REFUSED, NOT CARRIED, and the refusal is stated HERE
    # rather than inherited. The curl half above already refuses every folded grid, so
    # this clause is unreachable today -- and that is precisely why it is written: the
    # guarantee that `fill_symmetry_bc_D` (driver.py:3309) and
    # `fill_folded_far_ghosts_D` (:3311) do nothing inside this seam is what makes
    # REPLACES honest, and it may not depend on a clause in another module that a
    # future device verdict could licence away.
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

    # zero_metal_D IS REFUSED RATHER THAN CARRIED, and this is the clause that makes
    # REPLACES a contiguous run of driver order. See REPLACES for the decision and its
    # measured cost (0 slots: not one row of this product's board cell runs the pass).
    # An UNANSWERABLE grid is refused rather than read as unwalled: treating a missing
    # accessor as "no wall" would put a launch where the driver clears a plane.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; whether zero_metal_D "
                           f"(driver.py:3310) runs inside this seam cannot be decided")
    try:
        walls = zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")
    if any(walls):
        return False, (f"a metallic wall is active on axes "
                       f"{[axis for axis in range(3) if walls[axis]]}: zero_metal_D "
                       f"(driver.py:3310) runs inside this seam and this pair does "
                       f"not carry it -- its OFF-DIAGONAL table (two D components per "
                       f"walled axis) is device text no corpus row on this cell "
                       f"exercises, and a refusal cannot be silently wrong the way a "
                       f"wrong table would be")

    # THE SOURCE SEAM, REFUSED RATHER THAN CARRIED. An ELECTRIC source is injected
    # BETWEEN the two halves (driver.py:3305/:3308) by one of two routes -- the plain
    # `source.inject` loop and `_inject_electric_through_conductivity`, which the
    # driver takes only when there IS an electric source -- so a fused pair would
    # compute update_E against a pre-injection D. CARRIES_DEPOSIT_REPAIR is False and
    # nothing brackets this launch, so the shared clause refuses every one of them.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be exactly the
    # over-covering this clause exists to prevent. A MAGNETIC source is injected in the
    # B/H half and does not disqualify this pair -- which is what makes this product
    # serve its cell at all, since all four of its rows declare exactly one.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3305/:3308) and this "
            f"pair declares no deposit repair"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

    # THE STORED SHAPE THE KERNEL INDEXES MUST BE ONE TUPLE, on three readings. The
    # launch walks `Dx.shape`; the certified curl's guard is `nx * ny * nz` and the
    # certified constitutive's is `n_elem` from `Ex.shape`; and this kernel DROPS the
    # second guard. If those ever disagreed the fused launch would run the constitutive
    # half over a range its own certified kernel would not have.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        curl_extents = tuple(int(n) for n in fields.Dx.shape)
        constitutive_extents = tuple(int(n) for n in fields.Ex.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid or fields could not state the stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if not stored == curl_extents == constitutive_extents:
        return False, (f"grid.stored_cells is {stored}, Dx.shape is {curl_extents} "
                       f"and Ex.shape is {constitutive_extents}; this launch drops "
                       f"update_E's own bounds guard and may only do so where the "
                       f"three are one number")

    # THE THREE INVERSE-PERMITTIVITY VOLUMES MUST BE ASKABLE AND MUST MATCH THE SHAPE
    # THIS LAUNCH WALKS. The constitutive half's own predicate already checks them
    # against `_grid_facts`' shape; this asks the same question against the extents THIS
    # LAUNCH walks (`fields.Dx.shape`), which is the array the flat idx indexes -- and
    # this launch binds them a SECOND time, in the unused pole slots, so an unreadable
    # one is not merely a missing coefficient.
    accessor = getattr(fields, "inverse_epsilon_for", None)
    if not callable(accessor):
        return False, ("fields does not expose inverse_epsilon_for; update_E's three "
                       "inverse-permittivity volumes cannot be bound")
    for component in _ELECTRIC:
        try:
            volume = accessor(component)
        except Exception as exc:  # noqa: BLE001
            return False, (f"fields.inverse_epsilon_for({component!r}) raised: "
                           f"{type(exc).__name__}: {exc}")
        if volume is None:
            return False, f"fields.inverse_epsilon_for({component!r}) is None"
        shape = tuple(int(n) for n in getattr(volume, "shape", ()))
        if shape != curl_extents:
            return False, (f"inverse_epsilon_for({component!r}) has shape {shape} but "
                           f"the launch walks {curl_extents}; the kernel indexes it "
                           f"with this launch's flat index")

    # THE CONDUCTIVITY VOLUMES, ON THE ARM THAT BINDS THEM. The curl half's own
    # predicate checks them against `_grid_facts`' shape; this asks against the extents
    # THIS LAUNCH walks, for the same reason inv_eps is asked twice.
    if arm == "conductive":
        for reader_name in ("condfac_for", "condinv_for"):
            reader = getattr(fields, reader_name, None)
            if not callable(reader):
                return False, (f"fields does not expose {reader_name}; the conductive "
                               f"arm binds three of each")
            for component in _DISPLACEMENT.values():
                try:
                    volume = reader(component)
                except Exception as exc:  # noqa: BLE001
                    return False, (f"fields.{reader_name}({component!r}) raised: "
                                   f"{type(exc).__name__}: {exc}")
                if volume is None:
                    return False, f"fields.{reader_name}({component!r}) is None"
                shape = tuple(int(n) for n in getattr(volume, "shape", ()))
                if shape != curl_extents:
                    return False, (f"{reader_name}({component!r}) has shape {shape} "
                                   f"but the launch walks {curl_extents}")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The nine complex volumes bound ``__restrict__``, in SIGNATURE ORDER. Nothing else
#: keeps the launch and the kernel in step, so the order is spelled once and both the
#: binding check and the launch read it. There is NO ``fu_*`` and NO ``f_w_*`` group:
#: without a layer this family writes no auxiliary, and the constitutive half's own
#: predicate refuses a run where one is nevertheless allocated.
#: DERIVED FROM THE THREE COMPONENT TABLES rather than retyped, so the order the
#: launch binds and the order the signature declares come from one place: the flux
#: densities are :data:`_DISPLACEMENT`'s values (fields.py:1095), the curl's operands
#: are :data:`_CURL_SOURCES` (what ``get_H`` returns without an absorber), and the
#: constitutive targets are :data:`_ELECTRIC`.
_FIELD_BINDINGS: Tuple[str, ...] = (
    tuple(_DISPLACEMENT[component] for component in _ELECTRIC)
    + _CURL_SOURCES + _ELECTRIC)


def base_address(array: Any) -> int:
    """One array's base address, on either array module.

    ``.data.ptr`` is CuPy's and ``__array_interface__`` is NumPy's, and BOTH are needed
    rather than only the first: the binding check below is the one part of this module
    whose failure is silent on a device, so it has to be runnable on the host that
    composes the plan. A view shares its base's pointer, which is why the check reads
    the complex volumes as stored rather than word-viewing them.
    """
    pointer = getattr(getattr(array, "data", None), "ptr", None)
    if pointer is not None:
        return int(pointer)
    interface = getattr(array, "__array_interface__", None)
    if isinstance(interface, dict) and interface.get("data") is not None:
        return int(interface["data"][0])
    raise ValueError(
        f"{type(array).__name__} exposes no readable base address; an unverifiable "
        f"binding is not accepted, because the alias it might carry is undefined "
        f"behaviour NVRTC does not diagnose")


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three per-component inverse-permittivity volumes, in signature order.

    ``Fields.inverse_epsilon_for`` and never ``fields.inv_eps``, which is the Ez view
    (fields.py:1259-1260); binding it for all three is the defect the twelve
    uncertified complex kernels in ``step_curl_kernels`` carry.
    """
    return tuple(  # type: ignore[return-value]
        fields.inverse_epsilon_for(component) for component in _ELECTRIC)


def pole_bank_bindings(fields: "Fields",
                       poles: Optional[Dict[str, Sequence[Any]]] = None
                       ) -> Tuple[Tuple[Any, ...], Tuple[int, int, int]]:
    """The three pole banks and their counts, resolved AT LAUNCH. NOT word-viewed.

    THE ARRAYS COME BACK AS THEY ARE STORED and the float32 word view is taken at the
    bind, in :func:`launch_no_pml_complex_fused_electric_pair`. That split is the one
    thing that lets the resolution -- registration order, the ceiling, and which slot
    gets which pointer -- be checked on a host with no CuPy, which is where a slot bound
    to the wrong allocation is cheapest to catch. The two kinds are NOT
    interchangeable: a used slot holds a complex64 P volume and a spare one holds a
    float32 ``inv_eps``, and ``counts`` is what says which is which.

    THE POLE POINTERS ARE RESOLVED HERE AND NEVER CACHED BY A PLAN, exactly as the
    certified ``update_E_complex_no_pml_stored`` resolves them and for its reason:
    ``PolarizationState.update`` rotates ``P``/``P_prev``/``_scratch`` on every
    ``update_P``, so a launcher that snapshotted ``state.P[component]`` would be
    reading a retired history from the second step onward -- stale in a way that still
    computes. REGISTRATION ORDER IS BIT-LOAD-BEARING and comes from the certified
    family's own ``poles_per_component``.

    THE UNUSED SLOTS ARE BOUND TO ``inv_eps`` AND NOT TO THE SOURCE, which is the one
    place this launcher deviates from the certified one and the reason is the whole
    shape of this module: the certified ``update_E`` binds them to D, and D is bound
    exactly once here. They are never dereferenced -- the runtime ``np`` guard is false
    for them -- and ``inv_eps`` is not ``__restrict__``, so no promise is broken.

    ``poles`` is the gate's door: a mutation leg binds a deliberately reordered or
    truncated bank so the registration-order clause has something to be measured
    against.
    """
    if poles is None:
        banks_of_arrays = {component: tuple(state.P[component] for state in states)
                           for component, states
                           in certified.poles_per_component(fields).items()}
    else:
        banks_of_arrays = {component: tuple(poles[component])
                           for component in _ELECTRIC}
    inverse = inverse_epsilon_bindings(fields)
    bound: List[Any] = []
    counts: List[int] = []
    for index, component in enumerate(_ELECTRIC):
        entries = tuple(banks_of_arrays[component])
        if len(entries) > certified.MAX_POLES:
            raise ValueError(
                f"{component} is driven by {len(entries)} poles; MAX_POLES="
                f"{certified.MAX_POLES}")
        counts.append(len(entries))
        bound.extend(entries)
        bound.extend([inverse[index]] * (certified.MAX_POLES - len(entries)))
    return tuple(bound), (counts[0], counts[1], counts[2])


def assert_disjoint_bindings(fields: "Fields", arm: str,
                             poles: Optional[Dict[str, Sequence[Any]]] = None) -> int:
    """Check the promise every ``__restrict__`` in the signature makes.

    ONCE PER FROZEN CONFIGURATION, OFF THE LAUNCH PATH. The shared flux density is
    bound once by construction -- the signature has no second D group to hand it to --
    so what is left to check is that no OTHER two restrict arguments are the same
    allocation, and that nothing NON-restrict aliases one of them. Two ``__restrict__``
    pointers to one object is UB whatever the route to it, and NVRTC reorders across it
    without a diagnostic.

    THE NON-RESTRICT GROUPS ARE CHECKED DIFFERENTLY AND DELIBERATELY. ``inv_eps``, the
    pole banks and the conductivity volumes may all alias EACH OTHER -- an isotropic run
    hands one inv_eps pointer three times, a shared conductivity profile does the same,
    and this launcher deliberately binds every unused pole slot to ``inv_eps``. What
    they may NOT do is alias a restrict-qualified argument, which would break that
    argument's promise from the other side.

    ALSO CHECKED: THAT D IS BOUND EXACTLY ONCE. Counted over EVERY argument this launch
    passes rather than reasoned from the signature, because the signature is emitted
    text and the count is the claim the whole module rests on.

    Returns the number of distinct allocations checked, so a caller can assert that
    something was actually inspected; a check that examined nothing and reported success
    is the vacuity this whole track guards against.
    """
    restrict: Dict[int, str] = {}
    collisions: List[str] = []

    # THE COMPLEX VOLUME'S ADDRESS IS ITS WORD VIEW'S ADDRESS -- ``view`` shares the
    # base pointer -- so the check reads the arrays as stored rather than word-viewing
    # them. That is not a shortcut: word-viewing needs CuPy, and a binding collision is
    # cheapest to catch on the host that composes the plan.
    for name in _FIELD_BINDINGS:
        address = base_address(getattr(fields, name))
        if address in restrict:
            collisions.append(f"{name} and {restrict[address]} are the same allocation")
            continue
        restrict[address] = name

    loose: List[Tuple[str, Any]] = [
        (f"inv_eps_{component}", volume)
        for component, volume in zip(_ELECTRIC, inverse_epsilon_bindings(fields))]
    bank, counts = pole_bank_bindings(fields, poles)
    loose.extend((f"pole_slot_{index}", array) for index, array in enumerate(bank))
    if arm == "conductive":
        for reader_name in ("condfac_for", "condinv_for"):
            reader = getattr(fields, reader_name)
            loose.extend((f"{reader_name}({component})", reader(component))
                         for component in _DISPLACEMENT.values())
    # D BOUND EXACTLY ONCE, COUNTED. The three D bindings are the only writable
    # flux-density arguments in the signature; anything else pointing at one of them is
    # a second binding, whatever parameter it arrived through. Reported FIRST and
    # separately from the general restrict collision, because it is the one hazard this
    # module's whole shape exists to answer and a reader is entitled to see it named.
    displacement = {base_address(getattr(fields, name)): name
                    for name in ("Dx", "Dy", "Dz")}
    for label, array in loose:
        address = base_address(array)
        if address in displacement:
            collisions.append(
                f"{label} is {displacement[address]}: the shared flux density would be "
                f"bound a second time, which is the aliasing hazard this signature "
                f"exists to avoid")
        elif address in restrict:
            collisions.append(
                f"{label} and {restrict[address]} are the same allocation, and the "
                f"second is bound __restrict__")
    if collisions:
        raise ValueError(
            "the complex no-absorber fused electric pair binds every field argument "
            "__restrict__, and these arguments alias, which is undefined behaviour "
            "NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(restrict) + len({base_address(array) for _label, array in loose})


def launch_no_pml_complex_fused_electric_pair(
        fields: "Fields", boundary_codes: Sequence[Any], phase_flags: Sequence[Any],
        phase_values: Sequence[Any], dtdx: float, arm: str, expansion: Any,
        kernel: Optional[Any] = None,
        poles: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """Both of :data:`REPLACES` in ONE launch.

    ``boundary_codes`` are ``complex_pml_kernels.complex_boundary_codes(grid)``;
    ``phase_flags``/``phase_values`` are ``bloch_phase_arguments(grid, True)`` -- THE
    BACKWARD TABLE, whose imaginary parts are the conjugates of a step_B launch's, and
    handing this launch the forward one is a wrong answer on every phased row rather
    than a crash.

    ``arm`` is ``"plain"`` or ``"conductive"`` and is REQUIRED, never derived here:
    which tail a run takes is ``fields.condfac_for``'s answer and the classifier that
    reads it is :func:`no_pml_complex_curl_arm`, asked once, in the predicate.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is: a gate
    compiles a deliberately broken copy of the shipped source and hands it here. A
    launcher that could not be handed its own kernel could not arm a single mutation,
    and every mutation leg would silently launch the shipped one and report the defect
    as uncaught. ``poles`` is the same door on the pole banks.

    Returns the launch geometry rather than ``None`` so a gate can assert that
    something was actually launched.
    """
    if arm not in KERNEL_KEYS:
        raise ValueError(
            f"arm must be one of {sorted(KERNEL_KEYS)}, got {arm!r}")
    conductive = arm == "conductive"
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    bank, counts = pole_bank_bindings(fields, poles)
    # THE WORD VIEW IS TAKEN HERE AND ONLY ON THE USED SLOTS. A used slot holds a
    # complex64 P volume, whose float32 word view is the pointer the kernel wants; a
    # SPARE slot holds ``inv_eps``, which is float32 already and must not be re-viewed.
    # ``counts`` is what says which is which, and it is the same tuple bound as np0/1/2
    # below, so the guard the kernel branches on and the pointers it was handed cannot
    # disagree.
    slots = tuple(
        certified._word_view(array)
        if slot < counts[component] else array
        for component in range(3)
        for slot, array in enumerate(
            bank[component * certified.MAX_POLES:
                 (component + 1) * certified.MAX_POLES]))
    arguments = tuple(
        certified._word_view(getattr(fields, name)) for name in _FIELD_BINDINGS[:6])
    if conductive:
        arguments += tuple(
            fields.condfac_for(name) for name in _DISPLACEMENT.values())
        arguments += tuple(
            fields.condinv_for(name) for name in _DISPLACEMENT.values())
    arguments += tuple(
        certified._word_view(getattr(fields, name)) for name in _FIELD_BINDINGS[6:])
    arguments += tuple(inverse_epsilon_bindings(fields)) + slots + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(np.int32(code) for code in boundary_codes) + tuple(
        np.int32(flag) for flag in phase_flags
    ) + tuple(np.float32(value) for value in phase_values) + tuple(
        np.int32(count) for count in counts)
    (kernel or _get_kernel(expansion, conductive))(
        (blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES, "arm": arm,
            "kernel": kernel_name(arm), "poles": counts,
            "expansion": complex_emitter.normalized_expansion(expansion)}


def run_no_pml_complex_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: Any, dtdx: float, expansion: Any, *,
        sources: Any = None, license: Any = None, subnormal_policy: Any = None,
        arm: Optional[str] = None, kernel: Optional[Any] = None,
        poles: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``arm``, ``kernel`` and ``poles`` are the gate's doors, keyword-only. Passing
    ``arm`` is how a WRONG-TAIL mutation is armed: a launcher that always derived its
    own could not be handed the plain kernel on a conductive run, which is a converged,
    smooth and entirely wrong field rather than a crash.
    """
    from .complex_pml_kernels import (bloch_phase_arguments,  # noqa: PLC0415
                                      complex_boundary_codes)

    covered, reason = covers_no_pml_complex_fused_electric_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if arm is None:
        arm = no_pml_complex_curl_arm(fields, "step_D")
    assert_disjoint_bindings(fields, arm, poles)
    # THE BACKWARD PHASE TABLE. `True` is `complex_emitter.KERNELS['step_D'][1]`, read
    # off the same table the certified `step_complex_no_pml_curl` reads it from, and it
    # is the one argument of this call a copy of a step_B launcher would get silently
    # wrong.
    flags, values = bloch_phase_arguments(grid, True)
    return launch_no_pml_complex_fused_electric_pair(
        fields, complex_boundary_codes(grid), flags, values, dtdx, arm, expansion,
        kernel, poles)
