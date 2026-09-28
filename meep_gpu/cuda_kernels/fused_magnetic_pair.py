"""The FIRST fused hand-CUDA product: ``step_B`` welded into ``update_H``.

The unfolded real-field PML magnetic pair. One launch performs three driver
passes -- ``step_B``, ``zero_metal_B``, ``update_H`` (driver.py:3291, :3295,
:3298; the same three again in the second stepper at :4275, :4280, :4283) -- and
the flux density never leaves a register between the curl that computes it and
the constitutive relation that consumes it.

Until this file the hand-CUDA track had NO fused product at all: the twenty
certified kernels are twenty single sub-steps, and ``certification.json``'s
``dispatch`` block says ``wired=false`` for every one of them. What existed was
THE PARTS -- the certified curl (``step_curl_kernels.step_B_pml_real``), the
certified constitutive (``constitutive_kernels.update_H_pml_real``) and the three
certified in-seam passes (``in_seam_passes``) that sit between them. This module
is the weld, and nothing more: it introduces no new arithmetic.

=============================================================================
EVERYTHING IS LIFTED. NOTHING IS RE-SPELLED THAT COULD BE LIFTED.
=============================================================================

The Metal track's rule -- "lift, do not re-spell" -- is followed here by
CONSTRUCTION rather than by discipline. :func:`fused_magnetic_pair_source` reads
the sibling modules' own module-level device strings and splices them; it does
not carry a second copy of either half. A reader can check the claim by running
:func:`certified_curl_body` and diffing it against
``step_curl_kernels._step_B_pml_real_kernel_code``, and every anchor the splice
depends on raises :class:`AssertionError` if the certified text stops carrying
it, so a silent partial lift is not a reachable state.

WHAT WAS LIFTED VERBATIM, byte for byte:

* ``step_curl_kernels._REAL_PML_PRELUDE``'s three boundary ``#define``s and both
  ghost gathers, ``shift_up`` / ``shift_dn``;
* the whole body of ``step_curl_kernels._step_B_pml_real_kernel_code`` -- the
  index decomposition, all three components' operand gathers, the parenthesised
  stencil ``dtdx * ((sf - f1) + (f2 - ss))``, the metallic curl mask and the
  folded top-plane masks;
* ``constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE``'s ``constitutive_apply``,
  untouched -- the two separate left-to-right accumulations, and ``prev`` loaded
  before the store;
* the three ``constitutive_apply`` calls of
  ``constitutive_kernels._update_H_pml_real_kernel_code``, apart from the two
  substitutions named below.

WHAT HAD TO CHANGE, and why -- the full list is :data:`LIFT_EDITS`, which is data
so a gate can assert against it rather than against this prose:

1. ``pml_apply`` -> ``pml_apply_reg``. The certified helper returns ``void`` and
   its last statement stores the new field value. Fused, the constitutive half
   needs that value IN A REGISTER, so the helper names the expression and returns
   it. THE EXPRESSION, ITS PARENTHESISATION AND THE STORE ARE UNCHANGED: the
   global state this helper leaves is byte-for-byte the certified helper's, and
   the returned register is the same bits as a reload would be -- a float32
   stored to global and loaded back is the identity on the bits, which the
   sibling track measured as its own null mutation (``reload_fu_from_memory``,
   60/60 identical, ``constitutive_kernels.py``'s prelude comment).
2. ``pml_apply(Bx, fu_Bx, ...)`` -> ``b_x = pml_apply_reg(Bx, fu_Bx, ...)``, on
   each of the three component blocks, plus the three ``float b_* = 0.0f;``
   declarations hoisted above them (the certified blocks are braced scopes and a
   value declared inside one does not outlive it).
3. ``constitutive_apply(Hx, f_w_Hx, idx, Bx[idx], ...)`` ->
   ``constitutive_apply(Hx, f_w_Hx, idx, b_x, ...)``. THE SEAM, and it is exactly
   three characters of substitution in three places: the certified H kernel opens
   each component with a RELOAD of the flux density; fused, that is the live
   register.
4. ``kms_x`` -> ``kms_int_x`` on the constitutive half's three calls only. Both
   halves ship a coefficient vector spelled ``kms_a``, ON DIFFERENT YEE
   SUB-LATTICES: the B curl reads the HALF-INTEGER set (``kms_x_h``,
   ``real_pml_curl_tables(pml, half_integer=True)``) and the H constitutive the
   INTEGER one (``real_constitutive_tables(pml, half_integer=False)``,
   stepping.py:948). In one scope the two names collide, and resolving the
   collision by letting one shadow the other is precisely the half-cell error
   ``swap_constitutive_sublattice`` plants on the constitutive gate -- converged,
   smooth and wrong. The curl's six keep their certified names; only the
   constitutive's three ``kms`` are renamed, on lines that had to change for the
   seam anyway.
5. ``zero_metal_B``, carried in registers. This is the ONE re-spelling in the
   file and it is a LAUNCH-GEOMETRY re-spelling, not an arithmetic one -- see the
   next section.
6. THE OWN-CELL HOIST (2026-09-24), :data:`LIFT_EDITS` entries 8-12. Every word a
   thread reads at its own ``idx`` -- ``fu_B*``, ``B*``, ``f_w_H*``, ``H*`` -- is
   loaded by :func:`hoisted_loads` right after the ownership flags, behind the
   certified load's own guard, and the six own-cell calls take the loaded words as
   arguments: ``pml_apply_reg_pre`` and ``constitutive_apply_pre``, each DERIVED
   from the certified helper by anchored replacement and refused unless its
   statements equal the certified ones under the load map. The ARITHMETIC, its
   parenthesisation, the stores and their order are unchanged; what moves is when
   the loads issue. Shipped, each thread issued its twelve plain loads one at a
   time, each after the dependent store before it (the PTX already carried that
   order; NVVM did not hoist it), so at one element per thread the fused kernel
   kept fewer bytes in flight per SM than the two singles did across two launches
   and ran 0.88-0.90x of them. Hoisted, measured on ``pml_3d`` through
   ``bench_fused_products`` (RTX A6000, monitors attached, bit-identical to the
   array path): 1.3771 -> 1.0043 ms/step at 2.1M cells, 2.5987 -> 1.9423 at 4.1M,
   4.4744 -> 3.3403 at 7.1M (``results/round_2026-09-24_night_drivers/cuda_hoist/``;
   the spelling was chosen in ``results/cuda_hoist_design_2026-09-24/DECISION.md``).
   ``pml_apply_reg`` is RETIRED from the prelude -- no call is left, and a dead copy
   is where a first-site mutation needle lands vacuously. ``constitutive_apply``
   stays: the 21 ghost carries still call it at cells that are not this thread's
   own. The cost is 48 registers (5 blocks per SM against 6), measured costless.

=============================================================================
THE SEAM: WHICH IN-SEAM PASSES ARE CARRIED, AND WHICH ARE REFUSED BY NAME
=============================================================================

Between ``step_B`` and ``update_H`` the driver runs four things
(driver.py:3292-3296)::

    step_B -> magnetic source.inject -> fill_symmetry_bc_B
                                     -> zero_metal_B
                                     -> fill_folded_far_ghosts_B -> update_H

* **the magnetic source injection -- CARRIED, since 2026-08-28.** ``driver.step``
  withdraws each magnetic source's standing offset before ``step_B``
  (driver.py:3289) and injects the fresh dipole at :3293, between the halves, so
  a fused pair spanning that slot computes ``update_H`` against a PRE-INJECTION
  B. That is a property of the PRODUCT'S SHAPE, not of the seam: the deposit is a
  sparse scatter, the constitutive half has no neighbour reads, and the two slots
  this pair owns are two separate driver consults. So the launch runs over the
  whole grid against the uninjected field and the deposit points are RECOMPUTED
  at the second consult, where the injected field is already final. The clause is
  delegated to ``deposit_repair.seam_source_reasons`` with ``pair='B'`` -- the
  shared implementation the Metal and Triton fused pairs use -- and
  :data:`CARRIES_DEPOSIT_REPAIR` is ``True``, which is this module's declaration
  that :mod:`.fused_pairs` brackets its launch with the leading/trailing repair
  plans. The clause still REFUSES BY NAME what the repair cannot reconstruct: a
  source that does not publish the index it writes, an unallocated ``f_w_H*``,
  and (on the D/E seam, not this one) an off-diagonal or nonlinear constitutive
  half. An ELECTRIC source is injected in the D/E half and does not disqualify
  this pair. IGNORANCE IS NOT AN EMPTY SET: ``Fields`` does not hold the source
  list, so a caller that does not pass ``sources`` is refused rather than assumed
  clean -- and that refusal is UNCHANGED by the flip, because a repair cannot
  save what it has not been told about.

* **``zero_metal_B`` -- CARRIED IN REGISTERS.** The one pass whose whole content
  is a store of the constant ``+0.0f`` into a plane. Its component/axis table is
  THE DIAGONAL, lifted from ``in_seam_passes._zero_metal_B_kernel_code``:
  ``IYEE_SHIFTS`` (fields.py:214-219) gives Bx (0,1,1), By (1,0,1), Bz (1,1,0),
  so a B component has Yee shift 0 on its OWN axis and nowhere else -- Bx clears
  on an x wall, By on a y wall, Bz on a z wall. That is the EXACT COMPLEMENT of
  the D family's off-diagonal table (Dy/Dz on x, Dx/Dz on y, Dx/Dy on z), and a
  pair that reused the D table here would clear the wrong two components on every
  walled run. ``fu_B`` is deliberately NOT masked: ``zero_metal_B`` passes
  ``B_COMPONENTS`` only (stepping.py:2250).

  WHAT WAS RE-SPELLED, AND EXACTLY WHAT WAS NOT. The certified pass launches over
  a PLANE and selects its cells with ``face_geometry``; this kernel launches over
  the VOLUME and already holds ``i``, ``j``, ``k``. So ``Bx[base] = 0.0f`` on the
  x face becomes ``if (wall_x && i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }``. The
  face selected, the components selected and the value stored are identical; the
  addressing is not, and it CANNOT be, because ``face_geometry``'s ``base`` is a
  function of a plane-shaped thread index this launch does not have. The register
  is cleared alongside the store so ``update_H`` reads the wiped value, which is
  the whole point; the extra store costs one write on at most three planes of the
  volume and keeps the global B this kernel leaves byte-for-byte what the unfused
  sequence leaves.

* **``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` -- CARRIED SINCE
  2026-08-28, BY THE OWNERSHIP INVERSION.** Until this round both were refused,
  and the refusal's reason was true of the NAIVE carry and only of it: the near
  fill computes ``B[face(a, 0)] = phase * B[face(a, 2)]`` and the far fill
  ``B[face(a, -1)] = -phase * B[face(a, reflect_row)]``, so a thread standing on
  the destination would READ a post-curl B produced by a thread in another block,
  and an ordinary CUDA launch has no grid-wide barrier at which that read is
  defined.

  THE INVERSION REMOVES THE READ INSTEAD OF SYNCHRONISING IT, and it is a PORT
  rather than an invention: ``metal_kernels/folded_fused_magnetic_pair`` and
  ``triton_kernels/folded_fused_magnetic_pair`` both ship it, and Metal has
  strictly LESS synchronisation available than CUDA does. THE SOURCE THREAD WRITES
  THE DESTINATION, from the register it already holds:

  * a thread whose own cell is imaged by either fill is a DESTINATION. It computes
    the curl and the split-field recurrence and stores ``fu`` -- the array path's
    ``step_B`` writes ``fu`` at every cell and neither fill touches it
    (stepping.py:1451 writes ``field``, never ``fu_field``) -- and then STOPS. It
    does not form the displacement, does not read or write ``B`` at its own cell,
    and does not touch ``H`` or ``f_w_H`` there. Its displacement is discarded by
    the array path too: the fill overwrites it;
  * the thread at stored :data:`NEAR_SOURCE_INDEX` on the near axis and at the
    reflect row on each far axis is the SOURCE. It does its own cell in full and
    then writes every destination that back-substitutes to it -- up to SEVEN, when
    all three axes are folded -- each at its composed parity, followed by the
    certified constitutive statement at that cell. A near image moves the very
    axis ``update_H`` indexes, so it takes the DESTINATION's coefficient pair at
    stored index 0; a far image does not move it and takes this thread's own pair
    unchanged.

  Every destination word is written by exactly one thread, no thread reads a word
  another thread writes, and no barrier is needed. :func:`carried_destinations` is
  the closed form and :func:`fill_carry_blocks` emits it;
  ``results/metal_folded_far_carry_2026-08-20/ownership.json`` is where that form
  was MEASURED against the array path's own three passes (9 configurations, 27
  rows, zero differing words).

  WHAT :func:`covers_fused_magnetic_pair` STILL REFUSES, BY NAME: everything
  ``in_seam_coverage``'s own two fill predicates refuse (a cylindrical radial axis,
  complex64 storage, a mirrored axis with no declared +/-1 phase, a fold whose two
  termination routes disagree, an out-of-bounds reflect row), plus the three facts
  this carry adds and the stand-alone passes never had to ask -- an axis reported
  both folded and walled, a ``grid.stored_cells`` that disagrees with the array
  the launch walks, and a folded axis too short to hold the near fill's source row.

  THE FOLD BRANCHES IN THE CERTIFIED CURL TEXT ARE NOW REACHED.
  ``BC_MIRROR_PERIODIC`` and its four top-plane masks were lifted verbatim while
  unreachable; the fold admission they were gated for
  (``coverage.CURL_FOLD_ADMISSION``, 2026-08-19/20) is what this seam now stands on
  alongside ``CONSTITUTIVE_FOLD_ADMISSION``.

=============================================================================
THE SHARED VOLUME IS BOUND EXACTLY ONCE, AND THAT IS A CORRECTNESS RULE
=============================================================================

The curl half WRITES Bx/By/Bz; the constitutive half READS them. Spliced naively,
the fused signature would carry both parameter groups -- ``float* __restrict__
Bx`` from the curl and ``const float* __restrict__ Bx`` from the constitutive --
and a caller would hand the SAME allocation to both. Two ``__restrict__``
pointers to one object is undefined behaviour, and NVRTC does not diagnose it: it
reorders the load across the store and MISCOMPILES SILENTLY, which is a wrong
answer at full speed rather than a launch failure.

So the fused signature binds ``Bx``, ``By`` and ``Bz`` ONCE each, as writable
``float* __restrict__``, and the constitutive half has NO source pointers at all
-- it reads the registers. There is nothing left for a caller to double-bind.
:func:`assert_disjoint_bindings` checks the rest of the promise the qualifier
makes (twenty-seven distinct allocations) once per frozen configuration, off the
launch path.

=============================================================================
THE SIGNATURE -- 27 pointers and 10 scalars, and the ceiling does not bind
=============================================================================

    3 B  +  3 fu_B  +  3 E  +  3 H  +  3 f_w_H                     = 15
  + 6 curl coefficients (kms/sinv per axis, HALF-INTEGER)          =  6
  + 6 constitutive coefficients (kps/kms per axis, INTEGER)        =  6
                                                                     ---
                                                                      27 pointers
  + nx, ny, nz, dtdx, bc_x/y/z, wall_x/y/z                         = 10 scalars

which is 27*8 + 10*4 = 256 bytes against CUDA's 4 KB parameter space. THE METAL
TRACK'S BINDING CEILING HAS NO ANALOGUE HERE and this file does not pretend to
one: that platform refuses a 32nd buffer binding and its fused pair packs five
scalars into a ``constant Params&`` to land at 28. CUDA's constraint is not a
count, it is POINTWISE vs STENCIL -- and this pair is pointwise on both halves,
which is why it is the first product rather than one of the hard ones. There is
no inverse-mu volume, because mu = 1 is baked into the array path too
(``stepping.update_H`` passes ``fields.Bx`` directly, stepping.py:949).

=============================================================================
CERTIFIED, COMPOSABLE, AND STILL NOT DISPATCHED
=============================================================================

:data:`CERTIFIED_KERNELS` names the one kernel this file ships and
:data:`UNCERTIFIED_KERNELS` is EMPTY. The construction argument -- both halves
lifted, one seam substitution, one re-spelled pass -- was a HYPOTHESIS about
bit-identity until 2026-08-27, when the gate measured it: both legs RELEASED under
both float32 subnormal policies, recorded as ``cuda_fused_magnetic_pair_2026-08-27``
in ``certification.json``.

THAT RECORD IS DRIFTED AS OF THIS EDIT, and saying so is the point of the ledger.
``fingerprints.json:cuda_fused_magnetic_pair_2026-08-27`` already carried this file
in its ``source_drift`` list before today; the deposit carry adds to that drift. The
device text is UNCHANGED -- the flip, the defensive CuPy import and the prose touch
no spliced string and no entry of :data:`LIFT_EDITS` -- but the byte-identity verdict
is welded to a source digest, not to an argument about which lines matter, so the
gate must be re-cut on a device before this product's identity claim binds again.

STILL NOT A SLOT ARM, AND NOW REACHABLE ANYWAY. This product is not registered in
``registry._TABLE``: one launch spans ``step_B -> zero_metal_B -> update_H``, so it
has no single slot to bid at, and admitting it into a slot table would fire the
composer's ambiguity refusal on every row it covers. :mod:`.fused_pairs` is how it is
reached instead -- an opt-in fusion block that runs AFTER the arm table has selected,
may absorb the two slots only where the table already gave them to the arms this
kernel implements, and is the sole constructor of the two repair plans on this track.
``arms.plan_step(..., fuse=True)`` is the door; ``fuse=False`` (the default) leaves
the composition exactly what the census measured.

A VERDICT ON THE WELD IS STILL NOT A DISPATCH. ``fastpath.plan_fast_path`` returns
``None`` on every branch and no dispatch table names this kernel.
``test_package_boundary.py`` pins the absence of any ``meep_gpu`` -> ``cuda_kernels``
module-level import in both directions, and that stays true: :mod:`.fused_pairs`
lives INSIDE this directory and imports this module lazily, from a function body.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

# CUPY IS IMPORTED DEFENSIVELY, and that is a statement about WHICH HALF OF THIS
# FILE IS THE DANGEROUS ONE. The emitter, :data:`LIFT_EDITS` and
# :func:`covers_fused_magnetic_pair` touch no CuPy symbol -- they splice strings and
# read a grid -- and the predicate is the part whose failure mode is a SILENT WRONG
# ANSWER rather than a crash (``in_seam_coverage.py:26-30`` gives the same reason for
# keeping that module import-free). While this was a hard import the predicate could
# not be asked at all on the backend-free host that cuts this track's coverage census
# (``measure_predicate_coverage.py:127`` lifts with ``prefer_gpu=False``) and
# ``test_arms.py``'s NumPy-wearing-CuPy's-name stand-in could not reach it, so the
# seam clause below shipped with no laptop test. ``cp is None`` is refused BY NAME in
# :func:`_get_kernel`, which is the only function here that needs a device.
try:
    import cupy as cp
except ImportError:  # a host with no CuPy: the emitter and the predicate still run
    cp = None
import itertools
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

# The certified halves. THESE IMPORTS ARE THE LIFT: the device text below is
# spliced from these modules' own module-level strings, so there is exactly one
# copy of each half in the package and this file cannot drift from the gated
# bytes without an anchor assertion firing.
#
# DEFENSIVE FOR THE SAME REASON AS ``cupy`` ABOVE, AND NO FURTHER. Those three
# modules import CuPy at their own module scope, so a hard import here would take
# the whole file down on a CuPy-free host and with it
# :func:`covers_fused_magnetic_pair`, which touches none of them. Absent, the
# EMITTER cannot run -- there is nothing to splice -- and that is refused by name in
# :func:`_certified_halves` rather than surfacing as an ``AttributeError`` on
# ``None`` inside a string operation. The predicate does not consult them at all.
try:
    from . import constitutive_kernels, step_curl_kernels
    from . import in_seam_passes
except ImportError:  # a host with no CuPy: only the emitter is unavailable
    constitutive_kernels = step_curl_kernels = in_seam_passes = None
from . import own_cell_hoist

# The predicates and the compile memo are CuPy-free siblings, imported here with
# the same by-path fallback every module in this directory carries, so a probe
# that loads these files OUTSIDE the package still resolves them.
try:
    from .coverage import (covers_real_pml_constitutive, covers_real_pml_curl,
                           real_curl_boundary_codes as _real_curl_boundary_codes)
    from .in_seam_coverage import (IYEE_SHIFTS, MIRROR_SOURCE_INDEX,
                                   covers_fill_folded_far, covers_fill_symmetry,
                                   folded_far_rows, mirror_fill_phases,
                                   zero_metal_axes)
    from .nonlinear_constitutive import covers_real_pml_nonlinear_constitutive
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
    _real_curl_boundary_codes = _coverage.real_curl_boundary_codes
    covers_real_pml_nonlinear_constitutive = \
        _load("nonlinear_constitutive").covers_real_pml_nonlinear_constitutive
    _in_seam_coverage = _load("in_seam_coverage")
    IYEE_SHIFTS = _in_seam_coverage.IYEE_SHIFTS
    MIRROR_SOURCE_INDEX = _in_seam_coverage.MIRROR_SOURCE_INDEX
    covers_fill_folded_far = _in_seam_coverage.covers_fill_folded_far
    covers_fill_symmetry = _in_seam_coverage.covers_fill_symmetry
    folded_far_rows = _in_seam_coverage.folded_far_rows
    mirror_fill_phases = _in_seam_coverage.mirror_fill_phases
    zero_metal_axes = _in_seam_coverage.zero_metal_axes
    compile_cache = _load("compile_cache")

# The source-seam clause, shared with the eleven Triton and ten Metal fused-pair
# predicates. It is imported rather than written out here because it was written
# out here-shaped once per product and the duplication produced a real bug: one
# Metal pair asked about the wrong seam and would have refused on a magnetic
# source while ADMITTING an electric one (``test_fused_pair_deposit_wiring.py``,
# ``test_each_predicate_asks_about_the_seam_its_own_refusal_names``).
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
# THE PARTITION — EMPTY ON THE CERTIFIED SIDE, AND THAT IS THE POINT
# =============================================================================
#
# Both are spelled as PLAIN assignments with NO type annotation. The partition
# readers in this directory walk the syntax tree WITHOUT importing the module —
# they have to, these modules import cupy — and an annotated assignment is an
# ``ast.AnnAssign``, which those readers do not match. An annotation here would
# make this file's kernel invisible to every partition check that exists.

#: RECORDED 2026-08-27 as ``cuda_fused_magnetic_pair_2026-08-27`` in ``certification.json``; the
#: two halves landed together as the partition test requires. Both legs RELEASED
#: under BOTH float32 subnormal policies, and the SUBJECT module was checked
#: unchanged since those runs before the record was landed.
CERTIFIED_KERNELS = (
    "fused_magnetic_pair_pml_real",
)
#: Shipped without a gate verdict. EMPTY since 2026-08-27, and what emptied it was
#: a run rather than an argument: a fused product's bit-identity is a claim about
#: the WELD, which no verdict on either half establishes. The halves were certified
#: individually (``step_B_pml_real`` 2026-08-09/10, ``update_H_pml_real`` on the
#: same round, the three in-seam passes on 2026-08-21) and none of those is a
#: verdict about the three composed in one launch; the fused gate is.
UNCERTIFIED_KERNELS = {}
#: Does this product bracket its fused launch with the deposit repair? TRUE SINCE
#: 2026-08-28, AND THE WIRING LANDED IN THE SAME EDIT, which is the only way this
#: constant is allowed to move (``deposit_repair.py:221-226``). What it was waiting
#: for was a plan: this track had no fusion block at all, so there was no leading
#: slot that saved and no trailing slot that restored. :mod:`.fused_pairs` is that
#: block -- ``fused_pairs._install_fused_pair`` puts a
#: ``deposit_repair.LeadingRepairPlan`` in ``step_B`` (it saves ``Hx/Hy/Hz`` and
#: ``f_w_Hx/f_w_Hy/f_w_Hz`` at every deposit point, then launches) and a
#: ``deposit_repair.TrailingRepairPlan`` in ``update_H`` (it recomputes them after
#: the driver has injected, filled symmetry and cleared walls). Those are the two
#: consults ``driver.py:3292``/``:3303`` already are; no driver change was needed.
#:
#: THE FOLD WAS A SEPARATE QUESTION AND IT IS NOW ANSWERED IN THE SHARED MODULE. A
#: folded seam also runs ``fill_symmetry_bc_B``/``fill_folded_far_ghosts_B`` AFTER
#: the injection, and a repair that wrote only the deposit INDEX would leave every
#: MIRROR IMAGE of that index describing the pre-injection field. That is why this
#: note read "this product refuses every mirrored axis outright, so it never
#: reaches that question" until 2026-08-28. It no longer refuses one, and it does
#: not have to: ``deposit_repair.repair_cells`` hands ``save``/``apply`` the CLOSURE
#: of the cells the two fills image each deposit point into -- the inverse of the
#: forward carry :func:`carried_destinations` implements -- and
#: ``deposit_repair.repairable`` refuses BY NAME (``_folded_seam_reasons``) a fold
#: whose fill map it cannot read: the cylindrical r = 0 axis, and a folded axis too
#: short to hold the near fill's source row.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_fused_magnetic_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_magnetic_pair_pml_real"

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3291, :3294, :3295, :3296, :3298). Declared rather than inferred.
#:
#: THE TWO FILLS JOINED THE LIST ON 2026-08-28, and they are the whole of this
#: round: until then they were REFUSED BY NAME and every folded row on this seam
#: sat outside the predicate. They are carried by the OWNERSHIP INVERSION ported
#: from ``metal_kernels/folded_fused_magnetic_pair`` and
#: ``triton_kernels/folded_fused_magnetic_pair`` -- the SOURCE thread writes the
#: destination from a live register instead of the destination thread reading a
#: word another block wrote -- so no grid-wide barrier is needed and none is used.
REPLACES: Tuple[str, ...] = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: The sub-step slot a future planner would hold this on. Declared so a refusal
#: can be NAMED on the slot the fusion starts at; nothing reads it today, because
#: nothing registers this product.
SLOT = "step_B"

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring
#: -- and so that adding a silent sixth edit is a visible diff.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ float pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the helper "
            "names and returns the expression it already computed. The "
            "expression, its parenthesisation and the store to f[idx] are "
            "unchanged, so the global state it leaves is the certified one."},
    {"line": "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;",
     "became": "    float value = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
               "    f[idx] = value;\n    return value;",
     "why": "same right-hand side, same tree, same store; the result is "
            "additionally named so it can be returned. A float32 stored to "
            "global and reloaded is the identity on the bits, measured on this "
            "track as reload_fu_from_memory (60/60 identical)."},
    {"line": "        pml_apply(Bx, fu_Bx, idx, curl, ...);",
     "became": "        b_x = pml_apply_reg(Bx, fu_Bx, idx, curl, ...);",
     "why": "capture the register. Three lines, one per component; the argument "
            "lists are untouched. Three 'float b_* = 0.0f;' declarations are "
            "hoisted above the certified braced blocks because a value declared "
            "inside one does not outlive it."},
    {"line": "    constitutive_apply(Hx, f_w_Hx, idx, Bx[idx], kps_x[i], kms_x[i]);",
     "became": "    constitutive_apply(Hx, f_w_Hx, idx, b_x, kps_x[i], kms_int_x[i]);",
     "why": "THE SEAM (Bx[idx] -> b_x: the certified H kernel opens each "
            "component with a reload of the flux density the curl just stored), "
            "plus the sub-lattice rename. Both halves ship a vector spelled "
            "kms_a on DIFFERENT Yee sub-lattices -- half-integer for the B curl, "
            "integer for the H constitutive -- and letting one shadow the other "
            "is the half-cell error swap_constitutive_sublattice plants."},
    {"line": "        if (t < plane) Bx[base] = 0.0f;   (in_seam_passes.zero_metal_B)",
     "became": "    if (wall_x && i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }",
     "why": "THE ONE RE-SPELLING, and it is launch geometry, not arithmetic. The "
            "certified pass launches over a PLANE and addresses it through "
            "face_geometry(); this kernel launches over the VOLUME and already "
            "holds i/j/k, and face_geometry's `base` is a function of a "
            "plane-shaped thread index this launch does not have. Same face, "
            "same components (the B diagonal), same +0.0f. The register is "
            "cleared beside the store so update_H reads the wiped value."},
    {"line": "__device__ __forceinline__ float pml_apply_reg(\n"
             "    ... float kms_u, float sinv_u\n)",
     "became": "__device__ __forceinline__ float pml_apply_reg(\n"
               "    ... float kms_u, float sinv_u, int owned\n)\n"
               "    ...\n    if (!owned) return 0.0f;",
     "why": "THE OWNERSHIP INVERSION, half one. On a cell one of the two fills "
            "images, the SOURCE thread writes B -- so this thread must not READ "
            "f[idx] either, because that word is written by another block in this "
            "same launch and an ordinary CUDA launch has no grid-wide barrier at "
            "which the load would be defined. The fu recurrence stays ABOVE the "
            "guard: step_B writes fu at every cell and neither fill touches it "
            "(stepping.py:1451 writes field, never fu_field). The expression, its "
            "parenthesisation and the store below the guard are the certified "
            "ones."},
    {"line": "    constitutive_apply(Hx, f_w_Hx, idx, b_x, kps_x[i], kms_int_x[i]);",
     "became": "    if (own_x) {\n"
               "        constitutive_apply(Hx, f_w_Hx, idx, b_x, ...);\n"
               "        if (near_x && i == 2) { ... }   // and six more\n"
               "    }",
     "why": "THE OWNERSHIP INVERSION, half two -- fill_symmetry_bc_B "
            "(driver.py:3294) and fill_folded_far_ghosts_B (:3296), CARRIED. The "
            "certified statement is UNCHANGED inside the brace. What is new is "
            "that a thread standing on a cell a fill images does not run it at all "
            "(that cell's H and f_w_H belong to its source thread), and that a "
            "source thread runs it AGAIN at every ghost it owns, at the composed "
            "parity and -- for a near image, which moves the very axis update_H "
            "indexes -- at the DESTINATION's coefficient entry. The block list is "
            "fill_carry_blocks(); the rule is carried_destinations()."},
    {"line": "    int own_z = !(...);",
     "became": "    int own_z = !(...);\n"
               "    float pre_w_x = own_x ? f_w_Hx[idx] : 0.0f;   // ... twelve lines",
     "why": "THE OWN-CELL HOIST. Every word this thread reads at its own idx -- "
            "f_w_H*, H*, fu_B*, B* -- is issued before the kernel's first store, "
            "behind the certified load's own guard (fu unguarded, the rest behind "
            "own_*). No store in this thread reaches those words before the "
            "certified load point, and a word another block writes in this launch "
            "sits under own_* == 0, where the load is not issued. hoisted_loads()."},
    {"line": "__device__ __forceinline__ float pml_apply_reg(\n    ...)",
     "became": "__device__ __forceinline__ float pml_apply_reg_pre(\n"
               "    ..., int owned,\n    float fprev, float fcur\n)",
     "why": "DERIVED, not typed: pml_apply_reg_pre_source() rewrites the helper's "
            "two own-cell loads into parameters by anchored replacement and "
            "refuses unless its statements equal the certified helper's under "
            "fprev -> fu[idx], fcur -> f[idx]. pml_apply_reg itself is RETIRED from "
            "the prelude: no call is left, and a dead copy is where a first-site "
            "mutation needle lands vacuously."},
    {"line": "(nothing: appended after _REAL_CONSTITUTIVE_PRELUDE)",
     "became": "__device__ __forceinline__ void constitutive_apply_pre(\n"
               "    ..., float src,\n    float prev, float fcur, float kps, float kms\n)",
     "why": "DERIVED from constitutive_kernels.constitutive_apply the same way "
            "(prev -> fw[idx], fcur -> f[idx]). The certified helper stays, "
            "untouched: the 21 ghost carries still call it at cells that are not "
            "this thread's own."},
    {"line": "        b_x = pml_apply_reg(Bx, fu_Bx, idx, curl, ..., own_x);",
     "became": "        b_x = pml_apply_reg_pre(Bx, fu_Bx, idx, curl, ..., own_x, "
               "pre_fu_x, pre_b_x);",
     "why": "the three own-cell curl calls take their preloaded words; the "
            "argument list is otherwise the certified one."},
    {"line": "        constitutive_apply(Hx, f_w_Hx, idx, b_x, kps_x[i], kms_int_x[i]);",
     "became": "        constitutive_apply_pre(Hx, f_w_Hx, idx, b_x, pre_w_x, pre_h_x, "
               "kps_x[i], kms_int_x[i]);",
     "why": "the three OWNED constitutive statements take their preloaded words, "
            "in the composer (hoisted_constitutive_body), NOT in "
            "certified_constitutive_body(), which special_kz_fused_magnetic_pair "
            "lifts verbatim. The 21 ghost-carry statements are untouched."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "NEAR_SOURCE_INDEX", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "carried_destinations",
    "certified_constitutive_body", "certified_curl_body",
    "covers_fused_magnetic_pair", "device_sources", "far_fill_axes",
    "fill_carry_blocks", "fused_magnetic_pair_fills",
    "fused_magnetic_pair_prelude", "fused_magnetic_pair_source",
    "fused_magnetic_pair_tables", "launch_fused_magnetic_pair",
    "near_fill_axes", "ownership_declarations", "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================
#
# Every anchor below is an exact line of certified device text. If one stops
# matching, the certified string changed, and the splice raises rather than
# emitting a kernel that is quietly missing a mask, a store or a seam.

#: Separates a certified kernel's signature from its body. Both certified strings
#: close their parameter list on its own line, so ONE anchor lifts either body.
_BODY_ANCHOR = "\n) {\n"

#: The last line of the index decomposition, emitted by BOTH certified bodies.
#: The curl half keeps it; the constitutive half's copy (and the bounds guard
#: above it) is dropped, because splicing both would redeclare i/j/k and the
#: kernel would not compile.
_DECODE_END = "    int i = idx / (ny * nz);\n"

#: The certified curl helper, and the one edit that turns it into a value.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ float pml_apply_reg(\n"
#: The certified helper's parameter list, and the one parameter the carry adds.
#: ``owned`` is 0 exactly on a cell one of the two fills images, and the array
#: path's own behaviour is what licenses the skip: ``step_B`` writes ``fu``
#: EVERYWHERE and the fills touch ``B`` only (stepping.py:1451 writes ``field``,
#: never ``fu_field``), so the ``fu`` recurrence stays above the guard and the
#: displacement the fill is about to overwrite is simply never formed.
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
    "    // fu above is UNGUARDED: step_B writes it at every cell and neither\n"
    "    // fill touches it (stepping.py:1451).\n"
    "    if (!owned) return 0.0f;\n"
    "    // THE ONE EDIT TO THIS HELPER: the right-hand side, its\n"
    "    // parenthesisation and the store are the certified ones; the value is\n"
    "    // additionally NAMED so the constitutive half can read it from a\n"
    "    // register instead of reloading it from global memory.\n"
    "    float value = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
    "    f[idx] = value;\n"
    "    return value;\n")

#: The three component registers the seam carries, in x/y/z order.
_CARRIED = ("x", "y", "z")

#: ``in_seam_passes._zero_metal_B_kernel_code``'s component/axis table, restated
#: as (component, axis, flag, coordinate). THE DIAGONAL, and it is spelled here
#: rather than imported from anything D-shaped for the reason the Metal sibling
#: gives: the D family's table is the off-diagonal complement (Dy/Dz on an x wall,
#: not Dx), and a pair that reused it would clear the wrong two components on
#: every walled run.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("Bx", "wall_x", "i"),
    ("By", "wall_y", "j"),
    ("Bz", "wall_z", "k"),
)

#: The three B targets in x/y/z order, and the lifted curl body's own spellings
#: for the coordinate, the stride and the extent of each axis. Spelled once: a
#: second decode here would be a second place to get the layout wrong, and every
#: name below is a variable the CERTIFIED curl body already declares
#: (``int k = idx % nz`` and the three ``const int s*`` above it).
_TARGETS: Tuple[str, ...] = ("Bx", "By", "Bz")
_AXIS_NAME: Tuple[str, ...] = ("x", "y", "z")
_COORDINATE: Tuple[str, ...] = ("i", "j", "k")
_STRIDE: Tuple[str, ...] = ("sx", "sy", "sz")
_EXTENT: Tuple[str, ...] = ("nx", "ny", "nz")

#: The three fill scalars' runtime names, per axis, in :data:`_SIGNATURE`'s own
#: spelling. ``_FAR_ACTIVE`` is a TEST rather than a flag because the far plan is
#: carried as a reflect row with ``-1`` standing for "this pass does not run on
#: this axis" -- a sentinel, so a source that read it anyway would index outside
#: the volume and be caught rather than silently imaging row 0.
_NEAR_FLAG: Tuple[str, ...] = ("near_x", "near_y", "near_z")
_REFLECT: Tuple[str, ...] = ("reflect_x", "reflect_y", "reflect_z")
_FAR_ACTIVE: Tuple[str, ...] = ("reflect_x >= 0", "reflect_y >= 0", "reflect_z >= 0")
_PHASE: Tuple[str, ...] = ("phase_x", "phase_y", "phase_z")

#: ``stepping.MIRROR_SOURCE_INDEX`` under this family's own name, re-exported so a
#: reader does not have to chase :mod:`.in_seam_coverage` for the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX


def near_fill_axes(target: int) -> Tuple[int, ...]:
    """Which axes ``fill_symmetry_bc_B`` can image for ONE B component.

    ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1440-1447) writes axis
    ``a`` for component ``m`` exactly when the plane's phase is declared and
    ``iyee[m][a] == 0``. THE YEE TEST IS STRUCTURAL and is what this returns; the
    phase is a RUNTIME flag on this track, because the axis is a runtime argument
    here rather than a source specialisation (``in_seam_passes``' own reason: one
    device string per family, so one digest covers every fold orientation).

    ``IYEE_SHIFTS`` gives Bx (0,1,1), By (1,0,1), Bz (1,1,0), so the answer is the
    component's OWN axis and only it -- at most ONE near destination plane per
    component. Read off the table rather than returned as ``(target,)``, because
    the D family's answer is the exact complement and a hand-written constant here
    is the one place that inversion is plausible and wrong.
    """
    shifts = IYEE_SHIFTS[_TARGETS[target]]
    return tuple(axis for axis in range(3) if shifts[axis] == 0)


def far_fill_axes(target: int) -> Tuple[int, ...]:
    """Which axes ``fill_folded_far_ghosts_B`` can image for ONE B component.

    ``stepping._fill_folded_far_ghosts`` (stepping.py:1524-1534) writes axis ``a``
    for component ``m`` exactly when ``_stored_past_owned(grid, a)`` -- a folded
    PERIODIC axis, at either full-count parity -- and ``iyee[m][a] == 1``. Again
    the Yee test is structural and the fold test is the runtime reflect row.

    THE EXACT COMPLEMENT OF :func:`near_fill_axes` ON THIS FAMILY, and that is the
    whole geometric content of the far carry: a B component's Yee shift is 0 on
    its own axis and 1 on the other two, so the far fill images it on up to TWO
    destination planes against the near fill's one, and a cell at the top of both
    carries the PRODUCT of the two parities.
    """
    shifts = IYEE_SHIFTS[_TARGETS[target]]
    return tuple(axis for axis in range(3) if shifts[axis] == 1)


def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], bool], ...]:
    """Every ghost cell ONE source thread owns, as ``(far axes at top, near?)``.

    THE OWNERSHIP RULE, PORTED RATHER THAN INVENTED, and it is MEASURED on the
    sibling track rather than argued
    (``parity/meep_gpu/results/metal_folded_far_carry_2026-08-20/ownership.json``:
    9 configurations, 27 rows, ``worst_differing_words`` 0, against the ARRAY
    PATH's own three passes in driver order). The driver runs the near fill, the
    wall clear and the far fill in that order (driver.py:3294-3296) and the far
    fill is applied axis by axis, so a cell the fills leave at the top of SEVERAL
    folded periodic axes is written more than once. Composing the passes gives one
    closed form: for component ``m`` the seam leaves, at the cell that sits at
    ``last`` on the far-axis set ``T`` and at stored 0 on the near axis when ``z``,

        (prod over a in T of mirror_parity(m, a, phase_a))
            * (z ? mirror_parity(m, near, phase_near) : 1)
            * v_m(that cell with every a in T moved to reflect_a, near moved to 2)

    -- independent of the order the axes are applied in, which is why nothing here
    has to model either fill's X/Y/Z loop. So ONE thread, the one at the fully
    back-substituted cell, computes the displacement every one of those ghosts
    carries, and it writes them all. Every destination word is written by exactly
    one thread and no thread reads a word another writes, which is what makes the
    carry legal inside ONE ordinary CUDA launch with no grid-wide barrier.

    Returned smallest-subset-first for a stable emission order; the order is
    unobservable (the cells are distinct) and a stable one keeps a source diff
    readable.
    """
    far = tuple(int(axis) for axis in far)
    combos: List[Tuple[Tuple[int, ...], bool]] = []
    for size in range(len(far) + 1):
        for subset in itertools.combinations(far, size):
            for carries_near in ((False, True) if near else (False,)):
                if not subset and not carries_near:
                    continue  # the thread's OWN cell, not a ghost
                combos.append((subset, carries_near))
    return tuple(sorted(combos, key=lambda item: (len(item[0]), item[0], item[1])))

#: The fused entry point. With :func:`hoisted_loads` (twelve guarded load
#: statements, no arithmetic) the ONLY hand-written device text in this module, and
#: it is a signature: no arithmetic lives here. Bx/By/Bz appear EXACTLY ONCE --
#: see the module docstring's aliasing section, which is the reason the
#: constitutive half has no source pointers at all.
_SIGNATURE = r'''
extern "C" __global__ void fused_magnetic_pair_pml_real(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the
    // constitutive half reads it; binding it a second time as a const
    // __restrict__ source would be two restrict pointers to one allocation,
    // which is UB and which NVRTC miscompiles without a diagnostic.
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
    float* __restrict__ f_w_Hx, float* __restrict__ f_w_Hy,
    float* __restrict__ f_w_Hz,
    int nx, int ny, int nz, float dtdx,
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
    // zero_metal_B's three walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z,
    // THE TWO FILLS' RUNTIME PLAN, from in_seam_coverage.mirror_fill_phases and
    // .folded_far_rows. `near_a` is 1 on a MIRROR-folded axis (either
    // termination), where fill_symmetry_bc_B images stored 0 from stored 2.
    // `reflect_a` is fill_folded_far_ghosts_B's image row on a folded PERIODIC
    // axis and -1 where that pass does not run -- a SENTINEL rather than 0, so a
    // source that read it anyway would index outside the volume and be caught
    // rather than silently imaging row 0. `phase_a` is the plane's declared
    // mirror phase; the near fill weights by +phase (the imaged components have
    // Yee shift 0 there) and the far fill by -phase (theirs is 1), which is the
    // certified in_seam_passes kernels' own spelling of fields.mirror_parity.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float phase_x, float phase_y, float phase_z
) {
'''


def _split_body(source: str, prelude: str, name: str) -> str:
    """One certified kernel string, reduced to its body.

    Strips the prelude it was built from and the signature that follows it, and
    the closing brace. Raises rather than returning a truncated body: a splice
    that silently dropped a component would produce a kernel that steps two
    fields and reports success.
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


def _certified_halves() -> None:
    """Refuse the emitter BY NAME on a host where the certified text is unreachable.

    THE SPLICE IS THE LIFT (this file's opening claim), so a missing half is not a
    degraded emit -- there is nothing to emit. Raising here keeps that fact one
    frame from the caller instead of surfacing as ``'NoneType' object has no
    attribute '_REAL_PML_PRELUDE'`` inside a string operation.
    """
    missing = [name for name, module in (
        ("step_curl_kernels", step_curl_kernels),
        ("constitutive_kernels", constitutive_kernels),
        ("in_seam_passes", in_seam_passes)) if module is None]
    if missing:
        raise RuntimeError(
            f"the certified halves {missing} are not importable on this host (they "
            f"import CuPy at module scope), so there is no certified text to splice. "
            f"covers_fused_magnetic_pair needs none of them and still answers")


# -----------------------------------------------------------------------------
# THE OWN-CELL HOIST (candidate spelling S3). The fused kernel issues every own-cell
# word it reads -- fu_B*, B*, f_w_H*, H* -- before its first store, and the two
# helpers that would have loaded them take the words as arguments. Both helpers are
# DERIVED from the certified text by anchored replacement, and their statements are
# asserted equal to the certified helper's under the argument-to-load map, so a
# certified edit that moves the arithmetic refuses here by name.
# -----------------------------------------------------------------------------

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
    import re  # noqa: PLC0415
    if text.count(") {\n") != 1:
        raise AssertionError(f"{what}: the parameter list no longer closes on one line")
    body = text.split(") {\n", 1)[1][: -len("}\n")]
    names = dict(loads)
    out: List[str] = []
    for line in body.splitlines():
        code = line.split("//", 1)[0].strip()
        if not code:
            continue
        bound = re.fullmatch(r"float (\w+) = (\w+\[idx\]);", code)
        if bound:
            names[bound.group(1)] = bound.group(2)
            continue
        for name, load in names.items():
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
    certified = _hoist_statements(helper, {}, what)
    hoisted = _hoist_statements(derived, loads, what + " (hoisted)")
    if certified != hoisted:
        raise AssertionError(
            f"the hoisted {what} does not carry the certified arithmetic: "
            f"{certified!r} != {hoisted!r}")
    return derived


def constitutive_apply_pre_source() -> str:
    """``constitutive_kernels.constitutive_apply`` taking ``prev = fw[idx]`` and
    ``fcur = f[idx]`` as arguments -- derived from the certified text."""
    helper = _hoist_function_text(
        constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE,
        "__device__ __forceinline__ void constitutive_apply(\n", "constitutive_apply")
    return _hoist_derive(helper, (
        ("void constitutive_apply(\n", "void constitutive_apply_pre(\n", "its signature"),
        ("float src,\n    float kps, float kms\n)",
         "float src,\n    float prev, float fcur, float kps, float kms\n)",
         "its parameter list"),
        ("    float prev = fw[idx];\n", "", "its load of fw[idx]"),
        ("    float a = f[idx] + kps * src;\n", "    float a = fcur + kps * src;\n",
         "its load of f[idx]"),
    ), {"prev": "fw[idx]", "fcur": "f[idx]"}, "constitutive_apply")


def pml_apply_reg_pre_source(prelude: str) -> str:
    """The rewritten ``pml_apply_reg`` taking ``fprev = fu[idx]`` and ``fcur = f[idx]``
    as arguments -- derived from the text :func:`fused_magnetic_pair_prelude` emits."""
    helper = _hoist_function_text(prelude, _PML_APPLY_SIGNATURE_REG, "pml_apply_reg")
    return _hoist_derive(helper, (
        (_PML_APPLY_SIGNATURE_REG,
         _PML_APPLY_SIGNATURE_REG.replace("pml_apply_reg(", "pml_apply_reg_pre("),
         "its signature"),
        (_PML_APPLY_PARAMETERS_OWNED,
         "    float kms, float sinv, float kms_u, float sinv_u, int owned,\n"
         "    float fprev, float fcur\n)", "its parameter list"),
        ("    float fprev = fu[idx];\n", "", "its load of fu[idx]"),
        ("    float value = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n",
         "    float value = (((fcur * kms_u) + fu_new) - fprev) * sinv_u;\n",
         "its load of f[idx]"),
    ), {"fprev": "fu[idx]", "fcur": "f[idx]"}, "pml_apply_reg")


def hoisted_loads() -> str:
    """The twelve own-cell loads, each behind the certified load's own guard."""
    lines = [
        "    // --- the own-cell hoist: every word this thread reads at idx, issued",
        "    // before the kernel's first store, behind the certified load's guard",
        "    // (fu unguarded; B, H and f_w_H behind own_*). ---",
    ]
    for axis in _CARRIED:
        lines.append(f"    float pre_w_{axis} = own_{axis} ? f_w_H{axis}[idx] : 0.0f;")
        lines.append(f"    float pre_h_{axis} = own_{axis} ? H{axis}[idx] : 0.0f;")
    for axis in _CARRIED:
        lines.append(f"    float pre_fu_{axis} = fu_B{axis}[idx];")
        lines.append(f"    float pre_b_{axis} = own_{axis} ? B{axis}[idx] : 0.0f;")
    return "\n".join(lines) + "\n"


def _hoisted_constitutive_call(statement: str, axis: str) -> str:
    """The owned ``update_H`` statement, taking its two preloaded words."""
    old = f"constitutive_apply(H{axis}, f_w_H{axis}, idx, b_{axis}, "
    if statement.count(old) != 1:
        raise AssertionError(
            f"the lifted H statement no longer reads {old!r} once ({statement!r}); the "
            f"hoisted call has no anchor")
    return statement.replace(
        old, f"constitutive_apply_pre(H{axis}, f_w_H{axis}, idx, b_{axis}, "
             f"pre_w_{axis}, pre_h_{axis}, ", 1)


def fused_magnetic_pair_prelude() -> str:
    """Both certified preludes, with ``pml_apply`` turned into a value and hoisted.

    ``constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE`` is emitted UNTOUCHED --
    ``constitutive_apply`` needs no edit at all, because the seam changes where
    its ``src`` argument comes from and not what it does with it; the ghost
    carries still call it. Appended after it: its DERIVED twin
    ``constitutive_apply_pre`` and ``pml_apply_reg_pre``, which take the
    own-cell words :func:`hoisted_loads` issued. ``pml_apply_reg`` itself is
    retired (no call is left). See :data:`LIFT_EDITS` entries 8-12.
    """
    _certified_halves()
    prelude = step_curl_kernels._REAL_PML_PRELUDE
    if _PML_APPLY_SIGNATURE not in prelude:
        raise AssertionError(
            "the certified curl prelude no longer declares pml_apply with the "
            "signature this module rewrites")
    if _PML_APPLY_STORE not in prelude:
        raise AssertionError(
            "the certified pml_apply no longer closes with the store this module "
            "turns into a named value; the arithmetic may have moved")
    if prelude.count(_PML_APPLY_PARAMETERS) != 1:
        raise AssertionError(
            f"the certified pml_apply's parameter list appears "
            f"{prelude.count(_PML_APPLY_PARAMETERS)} times, not once; the "
            f"ownership guard has no anchor to take its argument on")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_PML_APPLY_PARAMETERS,
                              _PML_APPLY_PARAMETERS_OWNED, 1)
    prelude = prelude.replace(_PML_APPLY_STORE, _PML_APPLY_STORE_REG, 1)
    # THE HOIST RETIRES pml_apply_reg: every own-cell call takes the derived
    # pml_apply_reg_pre, so the value-returning helper would be dead text -- and a
    # dead copy of a helper is where a first-site needle lands vacuously. Derived
    # first (from the rewritten helper), then removed, exactly once.
    hoisted = pml_apply_reg_pre_source(prelude)
    helper = _hoist_function_text(prelude, _PML_APPLY_SIGNATURE_REG, "pml_apply_reg")
    if prelude.count(helper) != 1:
        raise AssertionError(
            f"the rewritten curl prelude carries pml_apply_reg {prelude.count(helper)} "
            f"times, not once; the retired helper has no single anchor")
    prelude = prelude.replace(helper, "", 1)
    return (prelude + constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE
            + constitutive_apply_pre_source()
            + hoisted)


def certified_curl_body() -> str:
    """``step_B_pml_real``'s body, lifted, with the three registers captured.

    Not a transcription: this is ``step_curl_kernels._step_B_pml_real_kernel_code``
    itself, minus its prelude and signature. ``backward`` is not a parameter and
    never will be -- ``step_D``'s negated strides are a different product.
    """
    _certified_halves()
    body = _split_body(step_curl_kernels._step_B_pml_real_kernel_code,
                       step_curl_kernels._REAL_PML_PRELUDE,
                       "_step_B_pml_real_kernel_code")
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
        "    float b_x = 0.0f;\n"
        "    float b_y = 0.0f;\n"
        "    float b_z = 0.0f;\n"
        "\n",
        ownership_declarations(),
        hoisted_loads(),
        tail,
    ))
    for target, axis in enumerate(_CARRIED):
        old = f"pml_apply(B{axis}, fu_B{axis}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} "
                f"{body.count(old)} times, not once; the capture has no anchor")
        body = body.replace(old, f"b_{axis} = pml_apply_reg(B{axis}, fu_B{axis}, ", 1)
        # The SAME line, now taking the ownership flag. Matched whole rather than
        # by its tail: a `);` anchor would happily attach the flag to whichever
        # statement happened to end first if the certified body ever moved this
        # call onto two lines.
        prefix = f"        b_{axis} = pml_apply_reg(B{axis}, fu_B{axis}, "
        call = _line_starting(body, prefix, f"target {target}'s captured pml_apply")
        if not call.endswith(");"):
            raise AssertionError(
                f"the certified curl body no longer closes target {target}'s "
                f"pml_apply on one line ({call!r}); the ownership flag has no "
                f"anchor to be appended at")
        body = body.replace(call, f"{call[:-2]}, own_{axis});", 1)
        # THE HOIST: the same statement, taking its two preloaded own-cell words.
        owned = _line_starting(body, prefix, f"target {target}'s owned pml_apply")
        body = body.replace(owned, owned.replace(
            f"b_{axis} = pml_apply_reg(", f"b_{axis} = pml_apply_reg_pre(", 1)[:-2]
            + f", pre_fu_{axis}, pre_b_{axis});", 1)
    if "pml_apply(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be "
            "written to global memory and never read into the seam")
    return body


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    The lift's safety property, in :mod:`.folded_fused_magnetic_pair`'s words: a
    missing anchor is certified text that has changed under this family, and
    splicing around it would produce a kernel that compiles and is quietly not the
    certified arithmetic; TWO matches would mean the anchor no longer identifies a
    single statement.
    """
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


def ownership_declarations() -> str:
    """``own_x``/``own_y``/``own_z`` -- is this thread's own cell imaged by a fill?

    THE ONE PLACE THE INVERSION IS DECIDED, and every other emitter below reads
    these three flags rather than restating the test.

    A cell either fill writes is OWNED BY ITS SOURCE THREAD: the thread standing on
    it computes the curl and the split-field recurrence and stores ``fu`` (the
    array path's ``step_B`` writes ``fu`` at every cell and neither fill touches
    it), and then STOPS. It never loads ``B`` at its own cell, never stores ``B``
    there, and never touches ``H`` or ``f_w_H`` there -- so no load it issues can
    land on a word another block writes in this launch, which is what makes the
    carry legal with no grid-wide barrier rather than merely convenient.

    THE FLAGS ARE RUNTIME, not baked. That is this track's rule, not a compromise:
    ``in_seam_passes`` takes its axis as a runtime argument for exactly this
    reason -- one device string per family, so one certification digest covers
    every fold orientation and no per-axis variant can end up pinned that nobody
    ran.
    """
    lines = [
        "    // --- the ownership inversion (fill_symmetry_bc_B, "
        "fill_folded_far_ghosts_B) ---",
        "    // A cell a fill images is written by its SOURCE thread, from a live",
        "    // register, below. The thread standing ON it stops after fu: forming",
        "    // v would read a B word another block writes in this same launch, and",
        "    // an ordinary CUDA launch has no grid-wide barrier at which that load",
        "    // is defined. The array path discards that displacement too -- the",
        "    // fill overwrites it (driver.py:3294, :3296).",
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
                f"every B component one shift-0 axis and two shift-1 axes, so an "
                f"empty test means the Yee table has drifted")
        near = "".join(_AXIS_NAME[axis] for axis in near_fill_axes(target))
        far = "".join(_AXIS_NAME[axis] for axis in far_fill_axes(target))
        lines.append(f"    // {name} {IYEE_SHIFTS[name]}: near fill on {near}, "
                     f"far fill on {far}.")
        lines.append(f"    int own_{_CARRIED[target]} = !({' || '.join(tests)});")
    return "\n".join(lines) + "\n"


def zero_metal_carry() -> str:
    """``stepping.zero_metal_B`` for all three targets, carried on the registers.

    THE ARITHMETIC IS LIFTED AND THE ADDRESSING IS NOT, which is the whole content
    of :data:`LIFT_EDITS` entry 5. The face, the component table and the ``+0.0f``
    are ``in_seam_passes._zero_metal_B_kernel_code``'s; the plane-shaped
    ``face_geometry`` addressing cannot be, because this launch has no
    plane-shaped thread index.

    THE STORE IS NOT OPTIONAL. Clearing only the register would leave global B
    holding the un-wiped wall plane, which is a plane of wrong values the next
    timestep's curl differences. Clearing only global memory would leave
    ``update_H`` reading the un-wiped value out of the register, which is the same
    error one sub-step earlier. Both, in that order, is what the unfused sequence
    leaves.

    ``fu_B`` IS NOT MASKED. ``zero_metal_B`` passes ``B_COMPONENTS``
    (stepping.py:2250); the split-field auxiliary is not in it, and masking it
    would damp the recurrence on the wall plane.
    """
    lines = [
        "    // --- zero_metal_B, carried in registers ---------------------------",
        "    // stepping._zero_metal (stepping.py:2206-2247), the B DIAGONAL from",
        "    // IYEE_SHIFTS: Bx (0,1,1) clears on an x wall, By (1,0,1) on y,",
        "    // Bz (1,1,0) on z. The D family's table is the complement and reusing",
        "    // it here would clear the wrong two components on every walled run.",
        "    // The walled set already excludes a folded metallic axis, on the host",
        "    // (in_seam_coverage.zero_metal_axes); a wall written across a fold",
        "    // destroys the fold rather than imposing a boundary.",
        "    // GUARDED ON own_*: a cell one of the fills images belongs to its",
        "    // SOURCE thread, and the array path agrees -- zero_metal_B (:3295) is",
        "    // overwritten there by fill_folded_far_ghosts_B (:3296). Clearing it",
        "    // here would be a second thread writing the same word.",
    ]
    for target, flag, coordinate in _ZERO_METAL_ROWS:
        register = f"b_{target[-1]}"
        lines.append(
            f"    if (own_{target[-1]} && {flag} && {coordinate} == 0) "
            f"{{ {register} = 0.0f; {target}[idx] = 0.0f; }}")
    return "\n".join(lines) + "\n"


def fill_carry_blocks(target: int, indent: str = "        ") -> List[str]:
    """Every imaged ghost this thread owns, then ``update_H`` at each of them.

    ONE BLOCK PER DESTINATION IN :func:`carried_destinations`, guarded on the flags
    that say THIS thread is that destination's source -- stored
    :data:`NEAR_SOURCE_INDEX` on the near axis, the runtime reflect row on each far
    axis.

    EVERY BLOCK IS NESTED INSIDE THE COMPONENT'S OWNERSHIP GUARD, and that nesting
    is not tidiness. It is a defect the sibling track's device gate found
    (``triton_kernels/folded_fused_magnetic_pair``'s ownership section, where lanes
    are vectors and the same fact has to be spelled as an AND of masks). With two
    fills live a thread can be the SOURCE of one and the DESTINATION of the other
    -- at ``i == 2`` on a folded x while standing on ``j == ny - 1`` of a folded
    periodic y -- and unguarded it would write a ghost from a ``b_*`` that was
    never formed. The destination it would have written is already owned by the
    fully back-substituted thread at ``(i == 2, j == reflect_y)``, which is the
    closed form's whole point. :func:`certified_constitutive_body` is the only
    caller and emits these strictly inside ``if (own_*) {`` ... ``}``.

    ``b_{m}`` IS READ AFTER THE WALL CLEAR, deliberately, and the driver order is
    why: ``zero_metal_B`` (driver.py:3295) runs BEFORE
    ``fill_folded_far_ghosts_B`` (:3296), so a far image of a cleared row must
    carry the cleared value -- including its SIGN, since the far weight is
    ``-phase`` and ``-1.0f * +0.0f`` is ``-0.0f`` on both paths. The near fill runs
    BEFORE the clear (:3294), but its destination and the clear's are disjoint per
    component, so reading the same post-clear register for both is exact.

    NO WALL CLEAR IS EMITTED AT A GHOST, and the two fills reach that by DIFFERENT
    routes:

    * the NEAR ghost moves the component to stored 0 of its OWN axis -- the exact
      cell ``_zero_metal`` clears -- but that axis is FOLDED wherever the near
      carry fires, and ``_zero_metal`` skips a folded axis (stepping.py:2284-2286).
      That is a RUNTIME fact on this track, where the walls and the fold flags are
      launch arguments rather than source specialisations, so it is asserted on the
      host instead: :func:`fused_magnetic_pair_fills` refuses an axis reported both
      folded and walled, and :func:`covers_fused_magnetic_pair` refuses it by name;
    * the FAR ghost moves the component along an axis that is NOT its own, and
      :data:`_ZERO_METAL_ROWS` is the B DIAGONAL, so the ghost sits at the SAME
      coordinate on the only axis that could clear it as the thread that owns it.
      Its clear status is that thread's, INHERITED by reading ``b_{m}`` after the
      clear lines rather than re-emitted. That one IS structural, and it is
      asserted here.
    """
    near = near_fill_axes(target)
    far = far_fill_axes(target)
    name = _TARGETS[target]
    if near != (target,):
        raise AssertionError(
            f"{name} is a NEAR-fill destination on axes {near}; for the B family "
            f"that set is exactly the component's own axis (fields.IYEE_SHIFTS is "
            f"0 there and 1 on the other two), and this carry images one near "
            f"ghost cell per component")
    if len(far) != 2 or target in far:
        raise AssertionError(
            f"{name} is a FAR-fill destination on axes {far} against a near axis "
            f"{near}; on the B family the two sets are COMPLEMENTARY, so far holds "
            f"exactly the two axes that are not the component's own")
    walls = tuple(_COORDINATE.index(coordinate)
                  for row_target, _flag, coordinate in _ZERO_METAL_ROWS
                  if row_target == name)
    if set(walls) & set(far):
        raise AssertionError(
            f"{name} images a FAR ghost along {far} that zero_metal_B also clears "
            f"on {sorted(set(walls) & set(far))}; _ZERO_METAL_ROWS is the B "
            f"diagonal and far_fill_axes excludes the component's own axis, so the "
            f"ghost would no longer inherit its source thread's clear")
    register = f"b_{_CARRIED[target]}"
    inner = indent + "    "
    lines: List[str] = []
    for subset, carries_near in carried_destinations(near, far):
        tag = ("g" + _CARRIED[target] + "_"
               + "".join(_AXIS_NAME[axis] for axis in subset)
               + ("n" if carries_near else ""))
        guards: List[str] = []
        terms: List[str] = []
        weights: List[str] = []
        described: List[str] = []
        for axis in subset:
            guards.append(f"{_FAR_ACTIVE[axis]} && "
                          f"{_COORDINATE[axis]} == {_REFLECT[axis]}")
            terms.append(f"+ ({_EXTENT[axis]} - 1 - {_REFLECT[axis]}) "
                         f"* {_STRIDE[axis]}")
            # Yee shift 1 on a far axis -> mirror_parity = phase * (1 - 2*1).
            # Spelled as a negation of the SCALAR, which is
            # in_seam_passes._fill_folded_far_B_kernel_code's own line.
            weights.append(f"(-{_PHASE[axis]})")
            described.append(
                f"the far fill on {_AXIS_NAME[axis]}: {_COORDINATE[axis]} = "
                f"{_EXTENT[axis]} - 1 imaged from {_REFLECT[axis]}, weight "
                f"-{_PHASE[axis]} (stepping._fill_folded_far_ghosts:1524-1534)")
        if carries_near:
            axis = near[0]
            guards.append(f"{_NEAR_FLAG[axis]} && "
                          f"{_COORDINATE[axis]} == {NEAR_SOURCE_INDEX}")
            terms.append(f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}")
            # Yee shift 0 on the near axis -> mirror_parity = phase * (1 - 2*0),
            # which is in_seam_passes._fill_symmetry_B_kernel_code's bare `phase`.
            weights.append(_PHASE[axis])
            described.append(
                f"the near fill on {_AXIS_NAME[axis]}: {_COORDINATE[axis]} = 0 "
                f"imaged from {NEAR_SOURCE_INDEX}, weight {_PHASE[axis]} "
                f"(stepping._fill_symmetry_ghost_cells:1440-1447, "
                f"_write_mirror_ghost:1450-1451)")
        coefficient = "0" if carries_near else _COORDINATE[target]
        lines.append(f"{indent}if ({' && '.join(guards)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(f"{inner}int {tag}_i = idx {' '.join(terms)};")
        lines.append(f"{inner}float {tag}_p = {' * '.join(weights)};")
        lines.append(f"{inner}float {tag}_v = {tag}_p * {register};")
        lines.append(f"{inner}{name}[{tag}_i] = {tag}_v;")
        if carries_near:
            lines.extend([
                f"{inner}// The DESTINATION's own coefficient entry: stored index 0 "
                f"on {_AXIS_NAME[target]},",
                f"{inner}// NOT this thread's at {NEAR_SOURCE_INDEX}. update_H "
                f"indexes {name[-1]} on {_COORDINATE[target]} "
                f"(stepping.H_CONSTITUTIVE_TERMS:226)",
                f"{inner}// and the near fill images along that same axis, so "
                f"reusing the source's pair",
                f"{inner}// would apply stored cell {NEAR_SOURCE_INDEX}'s absorber "
                f"profile to stored cell 0.",
            ])
        else:
            lines.append(
                f"{inner}// A far ghost does not move {_COORDINATE[target]}, the "
                f"axis update_H indexes this")
            lines.append(
                f"{inner}// component on, so it takes this thread's own pair "
                f"unchanged.")
        lines.append(
            f"{inner}constitutive_apply(H{_CARRIED[target]}, "
            f"f_w_H{_CARRIED[target]}, {tag}_i, {tag}_v, "
            f"kps_{_CARRIED[target]}[{coefficient}], "
            f"kms_int_{_CARRIED[target]}[{coefficient}]);")
        lines.append(f"{indent}}}")
    return lines


def certified_constitutive_body() -> str:
    """``update_H_pml_real``'s body, lifted, with the seam and the rename applied.

    Two kinds of edit and no third: the decode prologue is dropped (the curl half
    already emitted it, and a second copy would redeclare i/j/k), and each
    component's call takes the carried register instead of a reload and the
    renamed integer-lattice ``kms``.

    THE ARGUMENT ORDER, THE TARGETS, THE AUXILIARIES AND ``kps`` ARE UNTOUCHED,
    and so is ``constitutive_apply`` itself -- the two separate left-to-right
    accumulations, and ``prev`` loaded before the store, are the certified
    helper's, character for character.
    """
    _certified_halves()
    body = _split_body(own_cell_hoist.unhoisted_kernel_code(
                           constitutive_kernels._update_H_pml_real_kernel_code,
                           "update_H_pml_real", "_update_H_pml_real_kernel_code"),
                       constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE,
                       "_update_H_pml_real_kernel_code")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer carries its decode "
            "prologue; the lift would redeclare the curl half's indices")
    body = body.split(_DECODE_END, 1)[1]
    for target, (axis, coordinate) in enumerate(zip(_CARRIED, _COORDINATE)):
        source = f"B{axis}[idx]"
        if body.count(source) != 1:
            raise AssertionError(
                f"the certified H body reads {source} {body.count(source)} times, "
                f"not once; the seam has no anchor")
        body = body.replace(source, f"b_{axis}", 1)
        lattice = f"kms_{axis}[{coordinate}]"
        if body.count(lattice) != 1:
            raise AssertionError(
                f"the certified H body indexes {lattice} {body.count(lattice)} "
                f"times, not once; the sub-lattice rename has no anchor")
        body = body.replace(lattice, f"kms_int_{axis}[{coordinate}]", 1)
        # THE OWNERSHIP GUARD AND THE CARRY, on the SAME line the seam already
        # rewrote. The certified statement is unchanged inside the brace: what is
        # new is that a thread standing on a cell a fill images does not run it at
        # all (its H and f_w_H belong to that cell's source thread), and that a
        # source thread runs it again at every ghost it owns.
        call = _line_starting(body, f"    constitutive_apply(H{axis}, ",
                              f"target {target}'s constitutive statement")
        body = body.replace(call, "\n".join(
            [f"    if (own_{axis}) {{", f"        {call.strip()}"]
            + fill_carry_blocks(target)
            + ["    }"]), 1)
    return body


def hoisted_constitutive_body() -> str:
    """The lifted ``update_H`` body with each OWNED statement taking its preloaded words.

    Applied by the composer, not inside :func:`certified_constitutive_body`, which a
    sibling family lifts as-is. The ghost-carry statements are untouched."""
    body = certified_constitutive_body()
    for axis in _CARRIED:
        statement = _line_starting(body, f"        constitutive_apply(H{axis}, f_w_H{axis}, idx, b_{axis}, ",
                                   f"the owned H{axis} statement")
        body = body.replace(statement, "        " + _hoisted_constitutive_call(statement.strip(), axis), 1)
    return body


def fused_magnetic_pair_source() -> str:
    """The whole fused kernel: two lifted preludes, two lifted bodies, one seam.

    PURE ASCII, and that is a COMPILE REQUIREMENT rather than a style rule:
    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source with a bare
    ``open(..., 'w')``, so the bytes go through the interpreter's locale encoding
    -- ASCII under C/POSIX, which is what a non-interactive shell on the
    validation host gets. Two em-dashes in a comment once killed a kernel at its
    first launch with ``UnicodeEncodeError``.
    """
    source = "".join((
        fused_magnetic_pair_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        "\n",
        zero_metal_carry(),
        "\n    // --- update_H (stepping.update_H / _apply_constitutive_pml:2065) --\n"
        "    // Its three sources are the registers above, not a reload of B.\n",
        hoisted_constitutive_body(),
        "}\n",
    ))
    try:
        source.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AssertionError(
            f"the fused device source is not pure ASCII ({exc}); NVRTC's source "
            f"file is written through the interpreter's locale encoding and this "
            f"would fail at first launch, not at import") from exc
    return source


def device_sources() -> Dict[str, str]:
    """The shipped device text by kernel name -- what a record block would pin."""
    return {KERNEL_NAME: fused_magnetic_pair_source()}


# =============================================================================
# COMPILATION
# =============================================================================

#: NVRTC compile options -- CORRECTNESS, not performance, and identical to both
#: halves' (``step_curl_kernels._COMPILE_OPTIONS``,
#: ``constitutive_kernels._COMPILE_OPTIONS``, ``in_seam_passes._COMPILE_OPTIONS``).
#: ``--fmad=false`` is what makes the curl's stencil and the PML recurrence round
#: where ``stepping`` rounds: both ``(fu*kms) - curl`` and ``(f*kms_u) + fu_new``
#: are FMA candidates the array path rounds twice, and the certified pair reaches
#: bit-identity ONLY under this guard (120/120 identical with it, 0/120 without).
#: Spelled here rather than imported so that loading this file by path cannot pick
#: up a different tuple than the one a gate compiled.
_COMPILE_OPTIONS = ('--fmad=false',)

#: Lanes per block, matching every other kernel on this track
#: (``step_curl_kernels._REAL_PML_THREADS``,
#: ``constitutive_kernels._CONSTITUTIVE_THREADS``,
#: ``in_seam_passes._IN_SEAM_THREADS``). One element per lane, flat 1-D grid.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    Shared memo with the sibling modules -- the cache is keyed on the source
    string, so two modules cannot collide. A gate drives this between guard sets,
    where the compiler ITSELF is substituted and the option tuple is overridden
    from outside, a change no memo key can see.
    """
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str = KERNEL_NAME):
    """Compile on first use, memoized on (name, options, policy, source).

    THE SOURCE IS REBUILT PER CALL AND THAT IS LOAD-BEARING, for the reason both
    siblings give and one this module adds. The siblings' reason: the source is
    part of the memo key, so it has to be read before a key exists to miss on, and
    a bit-identity probe mutates kernels by assigning over module-level source
    strings. This module's own: its source is SPLICED FROM the siblings' strings,
    so a probe that mutates ``step_curl_kernels._step_B_pml_real_kernel_code``
    must see that mutation reach the fused kernel too -- a source memoized here
    would report a pass for a mutation the fused product never carried, which is
    exactly the defect the memo key exists to remove.
    """
    if name != KERNEL_NAME:
        raise ValueError(f"this module ships one kernel, {KERNEL_NAME!r}, "
                         f"not {name!r}")
    if cp is None:
        # The ONE function in this file that needs the device library. Refused by
        # name rather than by ``AttributeError: 'NoneType' object has no attribute
        # 'RawKernel'`` three frames away, which is the diagnostic the defensive
        # import at the top of this file would otherwise have bought.
        raise RuntimeError(
            f"{KERNEL_NAME} cannot be compiled here: CuPy is not importable on this "
            f"host. The emitter and covers_fused_magnetic_pair need no device and "
            f"still run; a LAUNCH does")
    code = fused_magnetic_pair_source()
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_fused_magnetic_pair(fields: Any, pml: Any, grid: Any,
                               sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_B`` -> ``zero_metal_B`` -> ``update_H``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal, which
    is this directory's convention (``covers_real_pml_curl``,
    ``covers_real_pml_constitutive``, ``in_seam_coverage.covers``).

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own
    certified predicate refuses is refused here with that half's reason, prefixed
    so a reader can tell which side said it. What this predicate ADDS is the three
    seam clauses -- the source slot, the two uncarried fills, and the grid's
    ability to answer which axes are walled.
    """
    covered, reason = covers_real_pml_curl(fields, pml, grid, "step_B")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_constitutive(fields, pml, grid, "H")
    if not covered:
        # THE NONLINEAR WIDENING, 2026-09-01, and it admits NO NEW BYTES: on a
        # run carrying an instantaneous chi2/chi3 the ordinary predicate refuses
        # BOTH sides by name while ``stepping.update_H`` (:907-923) reads nothing
        # nonlinear -- the engine's chi lives entirely inside ``update_E``
        # (stepping.py:26-28). ``cuda_nonlinear``'s H arm is that fact as a
        # predicate: it admits the SAME certified ``update_H_pml_real`` -- the
        # very kernel this pair's constitutive half splices -- on a nonlinear
        # run, and ``gate_cuda_nonlinear.py`` measured the claim on device
        # (``cuda_nonlinear_2026-08-21``) with the refusal's own premise armed
        # as a defect (``pade_scale_the_H_source``, CAUGHT). What THAT verdict
        # does not establish is the WELD on a nonlinear run, so
        # ``gate_cuda_fused_magnetic_pair.py`` carries nonlinear fixtures of its
        # own; this clause is the census's door to them. The composer selects
        # (``PML``, ``nonlinear``) on exactly these rows, and
        # ``fused_pairs.FUSED_PAIR_EXTRA_ARMS`` is the absorb declaration that
        # matches. Two predicates are consulted and BOTH reasons are reported on
        # a double refusal, so a nonlinear row refused for an unrelated clause
        # (a fold, a bad chi operand) still names it.
        nonlinear_covered, nonlinear_reason = \
            covers_real_pml_nonlinear_constitutive(fields, pml, grid, "H")
        if not nonlinear_covered:
            return False, (f"constitutive half: {reason}; and the nonlinear H "
                           f"widening also refuses: {nonlinear_reason}")

    # THE SOURCE SEAM, AND IT IS NOW CARRIED RATHER THAN REFUSED. A MAGNETIC source
    # is injected BETWEEN the two halves (driver.py:3293, and again at :4277), so a
    # fused pair computes update_H against a pre-injection B; `fused_pairs` brackets
    # the launch with the two repair plans, `CARRIES_DEPOSIT_REPAIR` says so, and
    # the shared clause then asks what the repair can actually reconstruct instead
    # of refusing on sight. WHAT IT STILL REFUSES, BY NAME: a source that does not
    # publish `_point_ix/_point_iy/_point_iz` (there is nothing to save and
    # restore), and an unallocated `f_w_H*` (there is no state to save). IGNORANCE
    # IS NEVER AN EMPTY SET and the flip does not touch that: ``Fields`` does not
    # hold the source list, so a predicate that inferred "no sources" from not being
    # told would be the over-covering this clause exists to prevent -- and a repair
    # that was never handed a source saves nothing and restores nothing, which is
    # the same wrong answer with a plan wrapped around it. An ELECTRIC source is
    # injected in the D/E half and does NOT disqualify this pair.
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

    # THE TWO FILLS, CARRIED SINCE 2026-08-28 -- and this is where what the carry
    # does NOT reach is refused BY NAME. Until this round both were refused
    # outright, on a reason that was true of the NAIVE carry and only of it: the
    # destination thread reading row 2 (near) or `reflect_row` (far) reads a word
    # another block writes in the same launch, and an ordinary CUDA launch has no
    # grid-wide barrier. THE OWNERSHIP INVERSION REMOVES THAT READ ENTIRELY rather
    # than synchronising it -- the SOURCE thread writes the destination from a live
    # register -- so the clause below is no longer about the fold, it is about
    # whether the two fills' own arithmetic is one this launch implements.
    #
    # DELEGATED, NOT RESTATED. `in_seam_coverage` is the single place each fill's
    # clauses live (it is the predicate for the certified stand-alone passes this
    # carry lifts), so a clause added there reaches this seam without a second
    # edit. What is asked of the B family only: the D side's fills sit in the other
    # seam and are not this pair's problem.
    mirrored = getattr(grid, "is_mirrored", None)
    if not callable(mirrored):
        return False, ("grid does not expose is_mirrored; this seam cannot tell "
                       "whether fill_symmetry_bc_B and fill_folded_far_ghosts_B "
                       "run inside it")
    covered, reason = covers_fill_symmetry(fields, grid, "B")
    if not covered:
        return False, (f"fill_symmetry_bc_B runs inside this seam "
                       f"(driver.py:3294) and this carry cannot serve it: {reason}")
    covered, reason = covers_fill_folded_far(fields, grid, "B")
    if not covered:
        return False, (f"fill_folded_far_ghosts_B runs inside this seam "
                       f"(driver.py:3296) and this carry cannot serve it: {reason}")

    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(has_symmetry):
        return False, "grid does not expose has_symmetry; the seam cannot be resolved"
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry != any(folded):
        # Fail closed, mirroring the curl predicate's own clause: a grid that
        # disagrees with itself about whether it is folded is a grid whose ghost
        # rule nothing here has resolved, and the carry is decided PER AXIS.
        return False, ("the grid reports has_symmetry() and no mirrored axis, or "
                       "the reverse; this carry is decided per axis and cannot be "
                       "read off a grid that disagrees with itself")

    # zero_metal_B IS CARRIED TOO, so the grid must be able to say which axes are
    # walled. A grid that cannot answer would silently be treated as unwalled,
    # which is a plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_B cannot be "
                           f"carried in registers")
    try:
        walls = zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE THREE CLAUSES THE CARRY ADDS OVER THE TWO FILLS' OWN PREDICATES. Each is
    # a fact the stand-alone passes never had to ask, because each launched over a
    # plane of its own AFTER the curl had finished.
    #
    # 1. NO AXIS MAY BE BOTH FOLDED AND WALLED. `zero_metal_axes` is
    #    `is_metallic and not is_mirrored` (stepping.py:2284-2286), so this is
    #    unreachable from a real Grid -- and the carry RESTS on it: the near ghost
    #    moves a component to stored 0 of its OWN axis, which is exactly the cell
    #    `_zero_metal` clears, and the emitted kernel writes no wall line there.
    for axis in range(3):
        if folded[axis] and bool(walls[axis]):
            return False, (
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction and "
                f"the near carry writes no wall line at the ghost it images")
    # 2. THE STORED SHAPE THE KERNEL INDEXES MUST BE THE ONE THE FILL ROWS WERE
    #    DERIVED FROM. `nx, ny, nz` come from `fields.Bx.shape`; `folded_far_rows`
    #    and the near source row are derived from `grid.stored_cells`. If those two
    #    ever disagree the reflect row is right for an array this launch is not
    #    walking -- a whole plane written at the wrong offset, not a crash.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Bx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks "
                       f"Bx.shape {extents}; the two fills' rows are derived from "
                       f"the first and indexed into the second")
    # 3. A FOLDED AXIS MUST BE LONG ENOUGH TO HOLD THE NEAR FILL'S SOURCE ROW.
    #    `stepping._mirror_source` raises on a shorter one (stepping.py:1584-1588);
    #    this kernel would read outside the volume instead, which is why the clause
    #    is stated here rather than left to the array path's own raise.
    for axis in range(3):
        if folded[axis] and extents[axis] <= NEAR_SOURCE_INDEX:
            return False, (
                f"folded axis {axis} stores {extents[axis]} cells, so the near "
                f"fill's source row {NEAR_SOURCE_INDEX} does not exist "
                f"(stepping._mirror_source raises on it)")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The twenty-seven pointer arguments, in SIGNATURE ORDER. Nothing else keeps the
#: launch and the kernel in step, so the order is spelled once and both the
#: binding check and the launch read it.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Bx", "By", "Bz",
    "fu_Bx", "fu_By", "fu_Bz",
    "Ex", "Ey", "Ez",
    "Hx", "Hy", "Hz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: The curl's HALF-INTEGER coefficient keys and the constitutive's INTEGER ones,
#: in signature order. THE PAIRING IS THE OPPOSITE OF THE D/E SEAM'S and the
#: kernel cannot tell -- a swap on either group is a half-cell error in the
#: absorber profile, converged, smooth and wrong.
_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def fused_magnetic_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads.

    THE SUB-LATTICE IS DECIDED HERE, not by the caller, for the reason
    ``constitutive_kernels.constitutive_tables_for`` gives: while it was an
    argument, every call site was one more place the pairing could be got
    backwards, and backwards is a half-cell error rather than a crash. The B curl
    reads the HALF-INTEGER positions (``real_pml_curl_tables(pml, True)``) and the
    H constitutive the INTEGER ones (``real_constitutive_tables(pml, False)``,
    stepping.py:948) -- the opposite pairing to the D/E pair.

    Called ONCE per frozen configuration, not per launch: both helpers return
    views over tables that never change.
    """
    return {
        "curl": step_curl_kernels.real_pml_curl_tables(pml, True),
        "constitutive": constitutive_kernels.real_constitutive_tables(pml, False),
    }


def fused_magnetic_pair_fills(grid: "Grid") -> Dict[str, Tuple[Any, ...]]:
    """The two fills' launch plan: near flags, far reflect rows, mirror phases.

    THE SAME THREE READINGS ``in_seam_coverage.plan`` GIVES THE STAND-ALONE
    PASSES, packed as the flat per-axis triples the kernel takes. Read from that
    module rather than from the grid directly, so the carry and the certified
    passes cannot disagree about which axes each fill visits.

    ``near[a]`` is 1 on a MIRROR-folded axis of either termination -- the near
    fill's own condition is a declared mirror phase (stepping.py:1484-1485), which
    a folded METALLIC axis has just as a folded PERIODIC one does.

    ``reflect[a]`` is ``-1`` where ``fill_folded_far_ghosts_B`` does not run, and
    that sentinel is deliberate for :mod:`.folded_fused_magnetic_pair`'s reason: it
    is a value the kernel never reads (the guard that would read it is
    ``reflect_a >= 0``), and a source that read it anyway would index outside the
    volume and be caught rather than silently imaging row 0.

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
                f"axis {axis} is folded but its mirror phase is {phases[axis]!r}; "
                f"a plane's parity is +1 or -1 and the even-mirror default standing "
                f"in for a plane that declared otherwise is a run wrong by twice "
                f"the field wherever the parity mattered")
        if phases[axis] is not None and bool(walls[axis]):
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction "
                f"(stepping.py:2284-2286) and this kernel's ghost carry relies on "
                f"the two sets being disjoint")
        if rows[axis] is not None and phases[axis] is None:
            raise ValueError(
                f"axis {axis} carries a far reflect row {rows[axis]!r} and no "
                f"mirror phase; stepping._fill_folded_far_ghosts weights that "
                f"image with the plane's parity and cannot run without one")
    return {
        "near": tuple(int(phases[axis] is not None) for axis in range(3)),
        "reflect": tuple(-1 if rows[axis] is None else int(rows[axis])
                         for axis in range(3)),
        "phase": tuple(0.0 if phases[axis] is None else float(phases[axis])
                       for axis in range(3)),
    }


def assert_disjoint_bindings(fields: "Fields",
                            tables: Dict[str, Dict[str, Any]]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes.

    ONCE PER FROZEN CONFIGURATION, OFF THE LAUNCH PATH. The shared flux density is
    bound once by construction -- the signature has no second B group to hand it
    to -- so what is left to check is that no OTHER two arguments are the same
    allocation. Two ``__restrict__`` pointers to one object is UB whatever the
    route to it, and NVRTC reorders across it without a diagnostic.

    Returns the number of distinct allocations checked, so a caller can assert
    that something was actually inspected; a check that examined nothing and
    reported success is the vacuity this whole track guards against.
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
    if collisions:
        raise ValueError(
            "the fused magnetic pair binds every argument __restrict__, and "
            "these arguments alias, which is undefined behaviour NVRTC "
            "miscompiles silently rather than diagnosing: " + "; ".join(collisions))
    return len(bound)


def launch_fused_magnetic_pair(fields: "Fields", tables: Dict[str, Dict[str, Any]],
                               boundary_codes: Sequence[Any],
                               walls: Sequence[int],
                               fills: Dict[str, Sequence[Any]], dtdx: float,
                               kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch.

    ``boundary_codes`` are ``step_curl_kernels.real_curl_boundary_codes(grid)``;
    ``walls`` are ``in_seam_coverage.zero_metal_axes(grid)``; ``fills`` is
    :func:`fused_magnetic_pair_fills`. THE THREE ANSWER DIFFERENT QUESTIONS and
    confusing any two is a plane of wrong values rather than a crash: the codes are
    ``_boundary_kinds``' ghost-rule resolution (where a folded METALLIC axis is
    indistinguishable from a plain wall), the walls are ``_zero_metal``'s own
    question (the grid's declaration, with a folded metallic axis EXCLUDED), and
    the fills are the two mirror passes' own (where a folded metallic axis DOES
    carry the near fill and does NOT carry the far one). No one of the three can
    be derived from another, which is why all three are arguments.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is: a
    gate compiles a deliberately broken copy of the shipped source and hands it
    here. A launcher that could not be handed its own kernel could not arm a
    single mutation, and every mutation leg would silently launch the shipped one
    and report the defect as uncaught.

    Returns the launch geometry rather than ``None`` so a gate can assert that
    something was actually launched.
    """
    nx, ny, nz = (int(n) for n in fields.Bx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + (
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
        np.int32(int(bool(walls[0]))), np.int32(int(bool(walls[1]))),
        np.int32(int(bool(walls[2]))),
    ) + tuple(np.int32(int(fills["near"][axis])) for axis in range(3)) + tuple(
        np.int32(int(fills["reflect"][axis])) for axis in range(3)
    ) + tuple(np.float32(float(fills["phase"][axis])) for axis in range(3))
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "near": tuple(int(v) for v in fills["near"]),
            "reflect": tuple(int(v) for v in fills["reflect"]),
            "phase": tuple(float(v) for v in fills["phase"])}


def run_fused_magnetic_pair(fields: "Fields", grid: "Grid", pml: "PML",
                            dtdx: float, *, sources: Any = None,
                            tables: Optional[Dict[str, Dict[str, Any]]] = None,
                            kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because
    the caller's correct response to a configuration this product does not carry
    is the array path -- never an exception into a stepper that would otherwise
    have stepped correctly.

    ``tables`` and ``kernel`` are the gate's doors, keyword-only. Passing
    ``tables`` is how ``swap_constitutive_sublattice``-style mutations are armed:
    a launcher that always derived its own could not be handed mis-paired ones.
    """
    covered, reason = covers_fused_magnetic_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = fused_magnetic_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    return launch_fused_magnetic_pair(
        fields, tables, step_curl_kernels.real_curl_boundary_codes(grid),
        zero_metal_axes(grid), fused_magnetic_pair_fills(grid), dtdx, kernel)


# The in-seam module is imported for one reason and it is worth naming: this file
# CARRIES all three of its B passes, and a reader checking that claim should find
# the certified text one attribute away rather than in another directory. Nothing
# below reads it at runtime.
_IN_SEAM_REFERENCE = getattr(in_seam_passes, "KERNEL_FOR", None)
