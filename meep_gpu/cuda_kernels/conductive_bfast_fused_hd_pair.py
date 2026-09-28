"""The H->D weld on the two curl tails the Cartesian product does not reach.

ONE PRODUCT, TWO VARIANTS, TWO BOARD CELLS. :mod:`.fused_hd_pair` welds the certified
``update_H`` into the certified LOSSLESS real-PML ``step_D``. Two seam-instances on the
CUDA fusion board sit on the same seam with the same ``update_H`` half and a DIFFERENT
``step_D`` tail, and until this module they were the board's last two
``buildable_not_built`` H->D instances outside the beta cluster::

    H_to_D (cuda_constitutive/ordinary -> cuda_conductive/conductive)   1 instance
        tests:TestAdjointSolver.test_damping
    H_to_D (cuda_bfast/BFAST          -> cuda_bfast/BFAST)              1 instance
        tests:TestReflectanceAngular.test_reflectance_angular_2_35_7

THE PAIRING IS REAL AND IT WAS CHECKED AGAINST THE CERTIFIED ARMS, not against the
board's cell label. ``parity/meep_gpu/results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl``
records, per row, the ENTRY POINT each half resolves to on this backend::

    test_damping                     update_H_kernel  fused_update_H_pml_real
                                     step_D_kernel    fused_step_D_(no_)pml_conductive
    test_reflectance_angular_2_35_7  update_H_kernel  fused_update_H_pml_real
                                     step_D_kernel    fused_step_D_bfast_real

So the first cell's two halves really do live in two different modules -- the
``update_H`` half is :mod:`.constitutive_kernels`' ``update_H_pml_real`` and the
``step_D`` half is :mod:`.conductive_kernels`' ``step_D_pml_conductive`` -- and the
board label is not a mislabel. What the same record ALSO shows, and what decided this
module's shape, is that the BFAST cell's ``update_H`` half is that SAME certified
body: ``bfast_curl`` ships no constitutive kernel at all and
``covers_bfast_constitutive`` admits the shipped one with the BFAST clause inverted
(that module's own docstring). The two cells therefore differ in the CURL HALF AND
NOTHING ELSE, which is exactly the shape ``certification.json``'s
``cuda_complex_fused_hd_pair_2026-09-07`` block states for the complex H->D product --
one transform emitting two device strings, with ``resolve`` reading the variant off
the run rather than taking it from a caller.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE, UNCHANGED
=============================================================================

Both certified ``step_D`` tails read ``H`` at the thread's own cell AND at three
BACKWARD neighbours through the SAME certified ``shift_dn`` -- both are built by
transforming ``step_curl_kernels``' own ``_step_D_pml_real_kernel_code`` (the
conductive one at :mod:`.conductive_kernels`'s ``_pml_template``) or by splicing the
BFAST insert into a copy of it (:mod:`.bfast_curl`'s ``_step_D_bfast_real_kernel_code``),
so the stencil, the ghost rule, the ownership mask and the operand grouping are the
certified pair's in both. MEASURED on the emitted text rather than read off that
history: six own-cell magnetic loads and six ``shift_dn`` taps in each, asserted per
emit (:data:`OWN_LOAD_EDITS`, :data:`HALO_TAPS`).

The hazard and its removal are therefore :mod:`.fused_hd_pair`'s, verbatim:

* **the constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go to
  LAUNCH-LOCAL SCRATCH, so ``H``, ``f_w_H`` and ``B`` are ``const`` for the whole
  launch and no thread can observe another thread's store;
* **the foreign read is a RECOMPUTE**, through the sibling's own
  ``raw_update_H_cell`` -- the certified ``update_H_pml_real`` body, whole, evaluated
  at an arbitrary cell. A pure function of unwritten memory has no schedule to depend
  on;
* the launcher then ROTATES the ``H``/``f_w_H`` bindings against the scratch.

WHAT EACH VARIANT ADDS, AND WHY IT IS SAFE IN PLACE. Both tails write a THIRD per-cell
history beside ``D`` and ``fu_D``, and both write it at the THREAD'S OWN CELL ONLY:

* ``conductive`` -- ``cond_pml_apply`` (conductive_kernels.py's
  ``_COND_PML_APPLY_PRELUDE``) reads ``f[idx]``, ``u[idx]``, ``c[idx]``, ``cf[idx]``,
  ``ci[idx]`` and writes ``f[idx]``, ``u[idx]``, ``c[idx]``. No neighbour, either way;
* ``bfast`` -- ``fb_D*[idx]`` is read into ``bprev`` and written back at the same
  index, and nothing else touches it.

Neither history is read by ``update_H``, which reads ``H``, ``f_w_H``, ``B`` and the
two coefficient groups and nothing else (stepping.py:907-923). So the weld's premise
-- *nothing this launch reads is a word this launch wrote* -- holds on both variants
for the same reason it holds on the lossless one, and the launcher's
:func:`assert_bindings_are_disjoint` refuses a binding that would break it.

=============================================================================
THE VARIANT IS READ OFF THE RUN, AND THE TWO ARE DISJOINT BY MEASUREMENT
=============================================================================

:func:`variant_for` asks the two questions the array path asks -- ``grid.bfast_active``
(stepping.py:392/:475, the only switch the BFAST term is behind) and
``fields.condfac_for`` over this sub-step's three targets (stepping.py:508, the only
place a conductivity is read) -- and returns ``"bfast"``, ``"conductive"``, or a
REFUSAL. It never returns a default: a run that is both, or neither, is a run this
product must not bind, and a reader that guessed would compile the wrong tail.

THE TWO ARE DISJOINT AND THAT IS A PROPERTY OF THE SHIPPED PREDICATES, not of this
file: ``covers_conductive_curl`` delegates every non-conductivity clause to
``coverage.covers_real_pml_curl`` on a LOSSLESS proxy, which refuses a BFAST grid by
name; ``covers_bfast_curl`` delegates every non-BFAST clause to the same predicate on
a BFAST-FREE proxy with the REAL fields, which refuses a conductive target by name. So
a run carrying both is refused by BOTH halves and there is no cell for it. This
module's gate drives that as a leg (``refusal``: ``conductive_and_bfast_together``)
rather than asserting it here.

=============================================================================
WHAT SITS IN THE SEAM, AND WHAT THIS PRODUCT DOES WITH IT
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT rather than a decision:
nothing is INJECTED between the ``update_H`` consult (driver.py:3311) and the
``step_D`` consult (:3315). The electric injection is one seam later and the magnetic
one is one seam earlier, so :mod:`..deposit_repair` has nothing to say about this
seam.

:data:`HOISTS_THE_WITHDRAW` is False, for :mod:`.fused_hd_pair`'s reason and with the
same wiring consequence: the only wiring that performs the hoist is
``fused_pairs._install_fused_pair``'s :data:`..withdraw_hoist.SEAM` branch, which
``_declared_uninstallable`` makes unreachable for a product declaring
:data:`INSTALLABLE` False. True here would be a claim about wiring that cannot fire,
so the predicate refuses BY NAME every row with a standing integrated electric
withdraw.

WHAT THAT COSTS ON THESE TWO CELLS IS ZERO, MEASURED rather than assumed. The seam
record's own columns for the two rows read ``withdraw_in_seam: false`` and
``n_electric_withdraws_that_do_work: 0`` (``h_to_d_seam_2026-09-04``), so this clause
refuses nothing either cell contains.

=============================================================================
IT IS NOT INSTALLED, AND THE ARBITRATION IS MEASURED THROUGH THE COMPOSER
=============================================================================

:data:`INSTALLABLE` is False. The Cartesian algebra :mod:`.fused_hd_pair` prices
applies unchanged here -- over ``step_B - update_H - step_D - update_E`` launches are
``4 - (installed pairs)`` and a span taking one slot from each neighbour can only tie
or lose -- and on BOTH of this product's cells the two neighbours EXIST as certified
products on this backend: :mod:`.conductive_fused_electric_pair` holds
``step_D``/``update_E`` on a conductive row, and :mod:`.bfast_fused_magnetic_pair` and
:mod:`.bfast_fused_electric_pair` hold the two seams either side on a BFAST row. The
flag ships False FIRST and the arbitration is then MEASURED through the shipped
composer (``arms.plan_step(..., fuse=True)``), which is the gate's ``arbitration``
leg: it reports which product won each slot, that this one is refused BY NAME, that
the refusal names ``INSTALLABLE`` False, and that the released neighbours keep the
slots they had.

=============================================================================
WHAT THIS FAMILY DOES NOT CARRY
=============================================================================

Every other configuration is refused through the two halves' OWN certified predicates,
which this one conjoins without weakening: complex64 storage, a Bloch phase,
``grid.beta``, special_kz, chi2/chi3, a Dcyl grid, an inactive absorber on the
conductive variant (``step_D_no_pml_conductive`` is a different kernel with a different
tail and no ``f_cond`` history -- see :data:`ACTIVE_LAYER_ONLY`), a mirror fold on the
BFAST variant (``bfast_curl``'s clause 4, refused by name because BFAST SUMS the
ghosts the certified fold verdict DIFFERENCES), and any grid whose two halves resolve
different boundary triples.

THE FOLD IS ADMITTED ON THE CONDUCTIVE VARIANT and by the two halves' own device
verdicts rather than by argument: ``coverage.CONSTITUTIVE_FOLD_ADMISSION`` for the
constitutive half and ``BC_MIRROR_PERIODIC`` for the curl, whose mask split was
measured by ``gate_cuda_folded_curl.py``. The seam is fill-FREE on a folded grid --
``fill_B``/``fill_folded_far_ghosts_B`` close one seam earlier (driver.py:3305-3310)
and ``fill_D`` opens one seam later (:3324-3330) -- so no ghost plane is rewritten
between the two consults this launch spans.

=============================================================================
NOTHING DISPATCHES THIS
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` builds ``triton_kernels`` plans and never names
this package. This module ships a predicate, an emitter and a launcher; once wired,
``fused_pairs`` can plan it opt-in (``arms.plan_step(..., fuse=True)``) and refuses to
INSTALL it on the declaration above, so no shipped code path launches it.

LAPTOP-EVALUABLE BY CONSTRUCTION. Every certified string this file splices is READ
with :mod:`ast` out of the sibling that owns it, the technique
``bfast_curl._sibling_prelude`` and ``conductive_kernels._pml_sibling_source`` already
use, so the whole emitter -- not just the predicate -- answers on a host with no CuPy
and the transcription leg can run anywhere. Where the sibling IS importable,
:func:`sibling_agreement` asserts this file's four lifted pieces are byte-equal to
:mod:`.fused_hd_pair`'s own, so "the two products cannot disagree about the
constitutive half" is CHECKED rather than asserted.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import bfast_curl as _bfast
from . import conductive_kernels as _conductive
from . import coverage as _coverage
from . import own_cell_hoist
from . import fused_hd_pair as _hd
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import constitutive_sub_lattice, covers_real_pml_constitutive

from .. import withdraw_hoist as _withdraw_hoist

FAMILY = "cuda_conductive_bfast_fused_hd_pair"

#: The two variants and the kernel each emits. The names are spelled as LITERALS here
#: and in :func:`signature` because ``test_kernel_partition.py`` reads shipped kernel
#: names off this file's own text with a regex that requires a C identifier.
KERNEL_NAMES: Dict[str, str] = {
    "conductive": "fused_hd_pair_pml_conductive",
    "bfast": "fused_hd_pair_bfast_real",
}

VARIANTS: Tuple[str, ...] = ("conductive", "bfast")

#: The default the tooling that expects a single ``KERNEL_NAME`` reads. Every source
#: function and every launcher takes the variant explicitly and none consults this.
KERNEL_NAME: str = KERNEL_NAMES["conductive"]

#: The certified ``step_D`` kernel each variant welds, in that family's own spelling.
CERTIFIED_CURL: Dict[str, str] = {
    "conductive": "step_D_pml_conductive",
    "bfast": "step_D_bfast_real",
}

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes ONE launch performs, in driver order (driver.py:3311, :3315).
#: Nothing between them is carried -- the electric withdraw is the only statement
#: there and :data:`HOISTS_THE_WITHDRAW` declines it in this round.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam ``fused_pairs.FUSED_PAIR_SEAMS`` files this row under, spelled through
#: :mod:`..withdraw_hoist` rather than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The six volumes the rotation swaps and the three magnetic components, taken from
#: :mod:`.fused_hd_pair` so the products cannot disagree about what a rotation covers.
SCRATCH_VOLUMES: Tuple[str, ...] = _hd.SCRATCH_VOLUMES
H_TARGETS: Tuple[str, str, str] = _hd.H_TARGETS

#: The three electric targets ``step_D`` writes, in the certified bodies' order.
D_TARGETS: Tuple[str, str, str] = ("Dx", "Dy", "Dz")

#: How many shifted magnetic taps and own-cell magnetic loads each certified
#: ``step_D`` makes. ASSERTED against the lifted text on every emit and for every
#: variant: a tap left standing would read a word another block is writing, and an own
#: load left standing would read the PRE-launch H.
HALO_TAPS = _hd.HALO_TAPS
OWN_LOAD_EDITS = _hd.OWN_LOAD_EDITS

#: The conductive variant binds ``f_cond`` and takes the FOUR-CASE PML recurrence, so
#: it is the ACTIVE-LAYER kernel and only that. Without an absorber
#: ``Fields._ensure_conductive_pml_storage`` returns at its first line
#: (fields.py:731-733), ``f_cond_*`` is never allocated and the array path takes a
#: DIFFERENT tail (``_apply_conductive_update``, one expression) served by a DIFFERENT
#: certified kernel (``step_D_no_pml_conductive``). Welding that one is a separate
#: product with its own signature and its own gate; this predicate refuses the
#: no-absorber row BY NAME rather than binding a kernel whose history does not exist.
ACTIVE_LAYER_ONLY: bool = True

#: Shipped but not gated. The two names live here until a device campaign releases
#: them; the partition (``test_kernel_partition.py``) requires every shipped kernel to
#: be in exactly one of the two sets, and a name in :data:`CERTIFIED_KERNELS` with no
#: ``certification.json`` block fails there. Spelled as a dict WITHOUT a type
#: annotation for the reason every sibling records: the partition readers walk the
#: syntax tree so they run where there is no CuPy, and an annotated assignment is an
#: ``ast.AnnAssign`` the plain-assignment readers do not match.
UNCERTIFIED_KERNELS = {}

#: BYTE-IDENTICAL TO FOUR INDEPENDENT ARRANGEMENTS OF THE SEAM, WITH A DEVICE VERDICT
#: BEHIND IT AND A RECORD BEHIND THE VERDICT. ``certification.json``'s
#: ``cuda_conductive_bfast_fused_hd_pair_2026-09-07`` block names the two artifacts
#: these lines rest on (``parity/meep_gpu/results/
#: cuda_conductive_bfast_fused_hd_pair_2026-09-07/{keep,flush}/gate.json``, both
#: released on all sixteen clauses), and ``test_kernel_partition.py`` refuses a name
#: here that no record block claims.
#:
#: THE CAMPAIGN WAS CUT AGAINST THE BYTES WITH THESE NAMES STILL UNCERTIFIED, which the
#: record states in its ``post_campaign_source_edits`` table: the partition sets are
#: read off the syntax tree and are not part of any emitted device string, so the
#: digests the record binds at NVRTC are unchanged by this move -- but the module's own
#: subject pin is not, and the record says so rather than absorbing it.
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "fused_hd_pair_pml_conductive",
    "fused_hd_pair_bfast_real",
)

#: Nothing is injected between the two consults, so there is no deposit to bracket.
CARRIES_DEPOSIT_REPAIR = False

#: Not in this round -- see the module docstring. The predicate refuses every row with
#: a standing integrated electric withdraw BY NAME, which on this product's two cells
#: costs zero rows.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration.
#: ``fused_pairs._declared_uninstallable`` reads this and reports
#: :data:`INSTALLABLE_REASON` on every run.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "slot arbitration, and on THESE cells the Cartesian algebra applies unchanged. "
    "Over the driver's step_B - update_H - step_D - update_E slot path launches are "
    "4 - (installed pairs), and this product takes one slot from EACH neighbouring "
    "seam. On the conductive cell the D->E neighbour is the released "
    "cuda_conductive_fused_electric_pair and the B->H neighbour is the released "
    "cuda_fused_magnetic_pair (a D conductivity leaves step_B lossless); on the BFAST "
    "cell they are cuda_bfast_fused_magnetic_pair and cuda_bfast_fused_electric_pair. "
    "A span taking one slot from each can only TIE (one neighbour installs: 3 "
    "launches / 1 seam either way) or LOSE (both install: 2 pairs -> 1, one more "
    "launch and one fewer seam served). The only strictly additive shape on this path "
    "is a four-slot step_B -> update_H -> step_D -> update_E weld, which is not built "
    "here. THE FLAG IS NOT THE MEASUREMENT: this product ships INSTALLABLE False "
    "first and its gate's arbitration leg then asks the SHIPPED composer what it "
    "selects, records that this product is refused by name with this reason, and "
    "records that the released neighbours keep their slots")

#: What a released verdict on this family would and would not license. Said once,
#: here, so every artifact that quotes this module quotes the same sentence.
WHAT_A_RELEASE_DOES_NOT_LICENSE = (
    "SERVED on the fusion board is PREDICATE ADMISSION by the board's own definition, "
    "so a released product credits its admitted seam-instances while executing "
    "NOWHERE: nothing under meep_gpu/ imports cuda_kernels outside tests, this family "
    "declares INSTALLABLE = False so the composer refuses to install it on every "
    "configuration once wired, and NO TIMING of any kind has been taken of this "
    "shape. The fused route does MORE memory traffic than the two singles for the "
    "same step-level launch count -- up to six extra pointwise constitutive "
    "evaluations per thread -- and whether the saved launch and the saved H "
    "write-then-read round trip pay for them is a HYPOTHESIS. Every launch figure in "
    "this module is a COUNT.")

#: Lanes per block. Both certified halves' own -- ``_REAL_PML_THREADS``,
#: ``conductive_kernels._THREADS``, ``bfast_curl._THREADS`` and
#: ``_CONSTITUTIVE_THREADS`` are all 256, and this kernel is one element per lane on a
#: flat 1-D grid like all of them.
_FUSED_THREADS = 256

#: NVRTC options, restated rather than imported for the reason
#: ``fused_hd_pair`` restates them: loading this file by path (which the probes do)
#: must not pick up a different tuple than the one the gate compiled. Both certified
#: curl families and the constitutive family ship exactly this tuple, and
#: ``--fmad=false`` is CORRECTNESS on every one of them -- ``a - kms * prev``,
#: ``f + kps * src``, ``((f * cf) - curl) * ci``, ``k1 * (sf + f1) - k2 * (ss + f2)``
#: and ``total - 2.0f * bprev`` are all FMA candidates the array path rounds twice.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "ACTIVE_LAYER_ONLY", "ADDED_PARAMETERS", "CARRIES_DEPOSIT_REPAIR",
    "CERTIFIED_CURL", "CERTIFIED_KERNELS", "CURL_LIFT_EDITS", "D_TARGETS", "FAMILY",
    "HALO_TAPS", "HOISTS_THE_WITHDRAW", "H_TARGETS", "INSTALLABLE",
    "INSTALLABLE_REASON", "KERNEL_NAME", "KERNEL_NAMES", "OWN_LOAD_EDITS", "REPLACES",
    "SCRATCH_VOLUMES", "SEAM", "SLOT", "UNCERTIFIED_KERNELS", "VARIANTS",
    "WHAT_A_RELEASE_DOES_NOT_LICENSE", "assert_bindings_are_disjoint",
    "certified_curl_text", "conductive_mask", "constitutive_prelude",
    "covers_conductive_bfast_fused_hd_pair", "current_source", "device_sources",
    "kernel_name", "kernel_source", "launch_conductive_bfast_fused_hd_pair",
    "raw_update_H_cell_source", "reset_kernel_sources", "resolution_source", "resolve",
    "rotate_into_fields", "run_conductive_bfast_fused_hd_pair", "set_kernel_source",
    "shift_dn_recompute_source", "sibling_agreement", "signature", "source_digest",
    "variant_for", "weld_args_construction", "weld_args_struct", "welded_curl_pieces",
]


# ---------------------------------------------------------------------------
# The certified text, READ rather than imported
# ---------------------------------------------------------------------------

def _literal(module: str, name: str) -> str:
    """A string literal assigned in a sibling module, parsed out of its source.

    ``bfast_curl._sibling_prelude``'s technique, generalised to the two spellings the
    certified modules use: a bare ``NAME = r'''...'''`` and the
    ``NAME = <prelude> + r'''...'''`` concatenation every kernel-code literal on this
    track is written as. Importing the owner would pull ``cupy`` in and make this file
    -- emitter and all -- unimportable where the coverage census runs; copying the
    text would fork it silently. Parsing has neither problem: the string IS the
    sibling's bytes by construction.
    """
    source = (pathlib.Path(__file__).with_name(module)
              .read_text(encoding="utf-8"))
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == name):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
                and isinstance(value.left, ast.Name)
                and isinstance(value.right, ast.Constant)
                and isinstance(value.right.value, str)):
            return _literal(module, value.left.id) + value.right.value
    raise RuntimeError(
        f"{module} no longer assigns a string literal (or <name> + <literal>) to "
        f"{name}; this weld lifts that text and cannot be built without it")


def _needle(source: str, old: str, new: str, what: str) -> str:
    """Replace ``old`` exactly once, or raise naming what stopped matching.

    :mod:`.fused_hd_pair`'s rule, restated for the same reason: a missing anchor is
    certified text that changed under this family, and splicing around it would emit a
    kernel that compiles and is quietly not the certified arithmetic.
    """
    count = source.count(old)
    if count != 1:
        raise AssertionError(
            f"the certified text carries {count} copies of {what}, not one; this weld "
            f"LIFTS that text rather than retyping it and cannot splice around its "
            f"absence")
    return source.replace(old, new, 1)


def _the_line_starting(body: str, prefix: str, what: str) -> str:
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise AssertionError(
            f"the certified body carries {len(matches)} lines starting {prefix!r}; "
            f"this weld lifts exactly one line for {what}")
    return matches[0]


# ---------------------------------------------------------------------------
# The declared edits — DATA, so a gate and the host suite can assert the list
# ---------------------------------------------------------------------------

#: The parameters this weld ADDS to each certified ``step_D`` signature, in emission
#: order and by group. DATA, so the transcription leg can assert the emitted parameter
#: list is exactly the certified one plus these and in no other respect changed.
#:
#: THE CERTIFIED LIST IS CARRIED ACROSS VERBATIM, which is this product's own
#: strengthening of the sibling's hand-written signature: ``Hx``/``Hy``/``Hz`` are
#: already ``const float* __restrict__`` in both certified curl signatures and are
#: exactly the pre-launch magnetic field the recompute needs, so they are not moved,
#: not retyped and not renamed.
ADDED_PARAMETERS: Dict[str, Tuple[str, ...]] = {
    "scratch_out": ("Hx_out", "Hy_out", "Hz_out",
                    "f_w_Hx_out", "f_w_Hy_out", "f_w_Hz_out"),
    "constitutive_in": ("f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz"),
    "constitutive_tables": ("kps_x", "kps_y", "kps_z"),
}

#: Every line of certified CURL text this weld does not lift verbatim, with the reason.
#: The same five entries :mod:`.fused_hd_pair` declares, because the transform is the
#: same one applied to a different certified body -- and that sameness is the point.
CURL_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "        float f1 = H*[idx];   /  float f2 = H*[idx];",
     "became": "        float f1 = own_h[n];  /  float f2 = own_h[n];",
     "why": "the curl's OWN-cell magnetic loads. The thread has already computed its "
            "own H into a register and that register is the post-update_H value the "
            "array path would have loaded. Leaving the load standing would read the "
            "PRE-launch H, which is the armed mutation `own_cell_reads_stale`."},
    {"line": "        float sf = shift_dn(H*, idx, <axis>, <extent>, <stride>, <bc>);",
     "became": "        float sf = shift_dn_recompute(<tag>, idx, <axis>, <extent>, "
               "<stride>, <bc>, weld);",
     "why": "the curl's six SHIFTED magnetic loads. Only the first argument changes: "
            "the pointer becomes its component tag, read off the certified line "
            "rather than tabulated here, and every other argument -- the axis "
            "coordinate, the extent, the stride and the boundary code -- is carried "
            "across character for character."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x; ... "
             "int i = idx / (ny * nz);",
     "became": "(kept, verbatim, as the fused kernel's own preamble)",
     "why": "NOT AN EDIT. The fused kernel keeps the CURL's preamble and decode as "
            "its own -- one bounds guard, one decode -- and the constitutive half's "
            "copy is dropped instead. Splicing both would redeclare idx, i, j and k "
            "and the kernel would not compile."},
    {"line": "        cond_pml_apply(D*, fu_D*, fc*, cf*, ci*, idx, curl, ...);",
     "became": "(kept, verbatim -- the conductive variant's tail)",
     "why": "NOT AN EDIT, and listed because a reader will look for one. The "
            "four-case conductive PML recurrence, its exact `!= 1.0f` partition and "
            "its three store predicates are conductive_kernels' own and reach this "
            "kernel unread and unrewritten. It touches f[idx], u[idx] and c[idx] and "
            "no neighbour, which is what makes it safe IN PLACE inside a launch whose "
            "magnetic half writes only scratch."},
    {"line": "        fb_D*[idx] = bprev + advance;  /  curl = curl - advance;",
     "became": "(kept, verbatim -- the BFAST variant's tail)",
     "why": "NOT AN EDIT either. The BFAST insert, its DOUBLE mask (on `advance` "
            "before the state absorbs it, then on the `curl` it was added to) and the "
            "IIR store are bfast_curl's own. The state is read and written at idx and "
            "nowhere else."},
)


# ---------------------------------------------------------------------------
# The lift: the constitutive half, shared with the sibling
# ---------------------------------------------------------------------------

def constitutive_prelude() -> str:
    """``constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE`` with the helper made PURE.

    :func:`.fused_hd_pair.constitutive_prelude`'s four anchored edits, applied to the
    certified text READ rather than imported and through THAT MODULE'S OWN anchor
    constants -- so a change to either side's spelling is a named failure here rather
    than two preludes that quietly differ. :func:`sibling_agreement` checks the result
    against the sibling's own on any host where the sibling can emit.
    """
    prelude = _literal("constitutive_kernels.py", "_REAL_CONSTITUTIVE_PRELUDE")
    for old, new, what in (
            (_hd._APPLY_SIGNATURE, _hd._APPLY_SIGNATURE_PURE,  # noqa: SLF001
             "constitutive_apply's declaration"),
            (_hd._APPLY_PARAMETERS, _hd._APPLY_PARAMETERS_PURE,  # noqa: SLF001
             "constitutive_apply's parameter list"),
            (_hd._APPLY_FW_STORE, _hd._APPLY_FW_STORE_PURE,  # noqa: SLF001
             "constitutive_apply's split-field store"),
            (_hd._APPLY_F_STORE, _hd._APPLY_F_STORE_PURE,  # noqa: SLF001
             "constitutive_apply's field store")):
        prelude = _needle(prelude, old, new, what)
    return prelude


def weld_args_struct() -> str:
    """The argument pack, declared. The sibling's, byte for byte."""
    return _hd.weld_args_struct()


def weld_args_construction(indent: str = "    ") -> str:
    """The struct, built from the kernel's own parameters. The sibling's."""
    return _hd.weld_args_construction(indent)


def _split_body(source: str, prelude: str, name: str) -> str:
    """One certified kernel string, reduced to its body."""
    if not source.startswith(prelude):
        raise AssertionError(
            f"{name} no longer begins with the prelude this module lifts separately; "
            f"the splice would emit it twice")
    tail = source[len(prelude):]
    anchor = _hd._BODY_ANCHOR  # noqa: SLF001
    if tail.count(anchor) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(anchor)} signature terminators, not 1; the "
            f"body anchor no longer identifies the signature")
    body = tail.split(anchor, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def raw_update_H_cell_source() -> str:
    """The certified ``update_H_pml_real`` body as a ``__device__`` function of a CELL.

    :func:`.fused_hd_pair.raw_update_H_cell_source`, over text read rather than
    imported. THE ENTIRE BODY IS LIFTED: only the thread preamble and the three
    ``constitutive_apply`` call lines are edited, each through an exact anchor. The
    decode, the component-to-axis pairing and the ``B*[idx]`` sources arrive character
    for character.
    """
    body = _split_body(own_cell_hoist.unhoisted_kernel_code(
                           _literal("constitutive_kernels.py",
                                    "_update_H_pml_real_kernel_code"),
                           "update_H_pml_real", "_update_H_pml_real_kernel_code"),
                       _literal("constitutive_kernels.py",
                                "_REAL_CONSTITUTIVE_PRELUDE"),
                       "_update_H_pml_real_kernel_code")
    body = _needle(
        body, _hd._THREAD_PREAMBLE,  # noqa: SLF001
        "    // THE CELL ARRIVES AS A FLAT INDEX. The bounds guard left with the\n"
        "    // thread index: every caller is either the kernel's own guarded thread\n"
        "    // or an index the certified shift_dn composed, which that helper's own\n"
        "    // branch keeps inside the volume. The DECODE BELOW STAYS -- it is what\n"
        "    // indexes the absorber profile at the recomputed cell rather than at\n"
        "    // the thread's own.\n",
        "the constitutive body's thread preamble")
    decode = _hd._INDEX_DECODE  # noqa: SLF001
    if body.count(decode) != 1:
        raise AssertionError(
            f"the certified update_H body carries {body.count(decode)} copies of the "
            f"index decode, not one; this weld keeps that block verbatim and cannot "
            f"vouch for a changed one")
    for component, target in enumerate(H_TARGETS):
        prefix = f"    constitutive_apply({target}, f_w_{target}, idx, "
        line = _the_line_starting(body, prefix,
                                  f"{target}'s certified constitutive_apply call")
        if not line.endswith(");"):
            raise AssertionError(
                f"{target}'s certified constitutive_apply call does not close on its "
                f"own line ({line!r}); the capture has no suffix to take")
        captured = (f"    h_out[{component}] = constitutive_apply_pure("
                    + line[len("    constitutive_apply("):-len(");")]
                    + f", &w_out[{component}]);")
        body = body.replace(line, captured, 1)

    signature_text = (
        "\n// The certified update_H_pml_real, evaluated at ONE ARBITRARY CELL and\n"
        "// storing nothing. This is the function the curl half calls for every\n"
        "// foreign tap it needs: a pure function of H, f_w_H and B, none of which\n"
        "// this launch writes, so a foreign evaluation cannot depend on which block\n"
        "// ran first.\n"
        "__device__ __forceinline__ void raw_update_H_cell(\n"
        "    int idx, const WeldArgs& weld, float* h_out, float* w_out\n"
        ") {\n")
    unpack = ["    // The certified body's own names, bound to the pack.",
              "    const int nx = weld.nx; const int ny = weld.ny;",
              "    const int nz = weld.nz;",
              "    (void) nx;   // the removed bounds guard's operand"]
    for _kind, name in _hd.WELD_ARGS_FIELDS:
        if name.startswith(("H", "f_w_H", "B", "kps_", "kms_")):
            unpack.append(f"    const float* __restrict__ {name} = weld.{name};")
    return signature_text + "\n".join(unpack) + "\n" + body + "}\n"


def resolution_source() -> str:
    """``resolve_H`` -- ``update_H``'s result for ONE component at any cell."""
    return _hd.resolution_source()


def shift_dn_recompute_source() -> str:
    """The certified ``shift_dn`` with its two leaf loads RESOLVED.

    :func:`.fused_hd_pair.shift_dn_recompute_source`, over the certified prelude read
    rather than imported. THE BRANCHES AND THE INDEX EXPRESSIONS ARE THE CERTIFIED
    HELPER'S OWN, parsed out of its text: everything left of ``g[`` is carried across
    verbatim, the expression inside the brackets is taken as written, and the exact
    ``0.0f`` the metallic and mirror-periodic ghosts read is untouched.
    """
    prelude = _literal("step_curl_kernels.py", "_REAL_PML_PRELUDE")
    declaration = _hd._SHIFT_DN_SIGNATURE  # noqa: SLF001
    if prelude.count(declaration) != 1:
        raise AssertionError(
            f"the certified curl prelude carries {prelude.count(declaration)} copies "
            f"of shift_dn's declaration, not one; the recompute has no anchor")
    body = prelude.split(declaration, 1)[1]
    if not body.startswith(" {\n"):
        raise AssertionError(
            "the certified shift_dn no longer opens its body on the declaration's own "
            "line; this weld lifts that body and cannot find it")
    body = body[len(" {\n"):]
    end = body.find("}\n")
    if end < 0:
        raise AssertionError("the certified shift_dn does not close")
    body, remainder = body[:end], body[end:]
    if remainder.lstrip("}\n").lstrip().startswith("float"):
        raise AssertionError(
            "the certified shift_dn's body appears to extend past the first closing "
            "brace; this weld lifts a two-statement helper")
    loads = [line for line in body.splitlines(keepends=True) if "g[" in line]
    if len(loads) != 2:
        raise AssertionError(
            f"the certified shift_dn makes {len(loads)} loads of g, not 2; this weld "
            f"resolves exactly the near-neighbour load and the periodic wrap")
    for line in loads:
        head, _, rest = line.partition("g[")
        expression, closing, tail = rest.partition("]")
        if not closing:
            raise AssertionError(
                f"the certified shift_dn line {line!r} opens a load of g and does not "
                f"close it; the index expression this weld carries across cannot be "
                f"read off it")
        body = body.replace(
            line, f"{head}resolve_H(comp, {expression}, weld){tail}", 1)
    if "g[" in body or " g," in body:
        raise AssertionError(
            "the resolved shift_dn still reads the pointer g; in this helper there is "
            "no such parameter and every magnetic read must be a recompute")
    return ("\n// stepping._shift_down's certified device helper, with the leaf loads\n"
            "// RESOLVED: the value the curl needs at a backward neighbour is not a\n"
            "// stored word but update_H's result there, recomputed from pre-launch\n"
            "// state. The branch, the wrap arithmetic and the exact 0.0f ghost are\n"
            "// the certified helper's own.\n"
            + _hd._SHIFT_DN_SIGNATURE_RECOMPUTE + " {\n"  # noqa: SLF001
            + body + "}\n")


def sibling_agreement() -> Dict[str, Any]:
    """This file's four lifted pieces against :mod:`.fused_hd_pair`'s own.

    THE WELD BETWEEN THE TWO PRODUCTS, CHECKED RATHER THAN ASSERTED. The sibling reads
    the certified text by IMPORT and this file by :mod:`ast`, so the two routes are
    independent; where the sibling can emit, every piece must agree byte for byte.
    Returns ``{"available": False, ...}`` on a host where it cannot, which is a
    RECORDED fact and not a pass.
    """
    try:
        theirs = {"constitutive_prelude": _hd.constitutive_prelude(),
                  "raw_update_H_cell": _hd.raw_update_H_cell_source(),
                  "resolve_H": _hd.resolution_source(),
                  "shift_dn_recompute": _hd.shift_dn_recompute_source()}
    except Exception as error:  # noqa: BLE001 - no CuPy on this host
        return {"available": False, "why": f"{type(error).__name__}: {error}",
                "agree": None, "disagreements": []}
    ours = {"constitutive_prelude": constitutive_prelude(),
            "raw_update_H_cell": raw_update_H_cell_source(),
            "resolve_H": resolution_source(),
            "shift_dn_recompute": shift_dn_recompute_source()}
    disagreements = sorted(name for name in theirs if theirs[name] != ours[name])
    return {"available": True, "why": None, "compared": sorted(theirs),
            "agree": not disagreements, "disagreements": disagreements}


# ---------------------------------------------------------------------------
# The lift: the curl half, per variant
# ---------------------------------------------------------------------------

def _check_variant(variant: str) -> str:
    if variant not in KERNEL_NAMES:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return variant


def kernel_name(variant: str) -> str:
    """The entry-point symbol this variant emits."""
    return KERNEL_NAMES[_check_variant(variant)]


def conductive_mask(fields: Any) -> Tuple[bool, bool, bool]:
    """``(COND0, COND1, COND2)`` for ``step_D`` -- the certified family's own reader.

    Asked through ``conductive_kernels.conductive_targets`` and never re-derived, which
    is what keeps this weld's compile-time choice and that family's admission from
    drifting apart.
    """
    return _conductive.conductive_targets(fields, "step_D")


def certified_curl_text(variant: str,
                        cond: Optional[Tuple[bool, bool, bool]] = None) -> str:
    """The certified ``step_D`` device text this variant welds, unmodified.

    Taken from the OWNING family's own public emitter, so a template edit or a mask
    reaches this weld the way it reaches the certified launch. ``cond`` is required on
    the conductive variant and refused on the BFAST one: the mask is three ``#define``
    lines the conductive family prepends, and the BFAST text has no such switch.
    """
    _check_variant(variant)
    if variant == "conductive":
        if cond is None:
            raise ValueError(
                "the conductive variant's device text is per-component: pass the "
                "(COND0, COND1, COND2) mask conductive_kernels.conductive_targets "
                "reads off the run, which is what its predicate reads too")
        return _conductive.kernel_source(CERTIFIED_CURL["conductive"],
                                         tuple(bool(flag) for flag in cond))
    if cond is not None:
        raise ValueError(
            "the BFAST variant carries no conductivity mask; its certified text has "
            "no COND switch and passing one would name a compile-time choice this "
            "kernel does not have")
    return _bfast.kernel_source(CERTIFIED_CURL["bfast"])


def _split_certified(text: str, name: str) -> Tuple[str, str, str]:
    """``(everything above the entry point, its parameter list, its body)``.

    The split is at the certified entry point's own declaration line, so the PRELUDE
    part -- the shared real-PML prelude, plus whatever helper the owning family put in
    front of its kernel -- is carried across whole and unread.
    """
    marker = f'extern "C" __global__ void {name}(\n'
    if text.count(marker) != 1:
        raise AssertionError(
            f"the certified text carries {text.count(marker)} declarations of "
            f"{name}, not one; this weld splits on that line")
    prelude, rest = text.split(marker, 1)
    anchor = _hd._BODY_ANCHOR  # noqa: SLF001
    if rest.count(anchor) != 1:
        raise AssertionError(
            f"{name} carries {rest.count(anchor)} signature terminators, not 1; the "
            f"body anchor no longer identifies the signature")
    parameters, body = rest.split(anchor, 1)
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return prelude, parameters, body[: -len("}\n")]


def welded_curl_pieces(variant: str,
                       cond: Optional[Tuple[bool, bool, bool]] = None
                       ) -> Dict[str, str]:
    """The certified curl, split and with every magnetic read redirected.

    Returns ``{"prelude", "parameters", "head", "tail"}``. ``head`` is the certified
    preamble, the stride constants and the index decode, lifted VERBATIM and kept as
    the fused kernel's own; ``tail`` is everything below it with the six own-cell
    magnetic loads taken from registers and the six shifted ones recomputed.
    """
    name = CERTIFIED_CURL[_check_variant(variant)]
    prelude, parameters, body = _split_certified(
        certified_curl_text(variant, cond), name)
    preamble = _hd._THREAD_PREAMBLE  # noqa: SLF001
    decode = _hd._INDEX_DECODE  # noqa: SLF001
    if body.count(preamble) != 1 or body.count(decode) != 1:
        raise AssertionError(
            f"the certified {name} body no longer carries exactly one thread preamble "
            f"and one index decode; this weld keeps both verbatim and splices its "
            f"constitutive half directly below the decode")
    cut = body.index(decode) + len(decode)
    head, tail = body[:cut], body[cut:]

    own = 0
    for component, target in enumerate(H_TARGETS):
        for variable in ("f1", "f2"):
            old = f"        float {variable} = {target}[idx];\n"
            if old not in tail:
                continue
            tail = _needle(tail, old,
                           f"        float {variable} = own_h[{component}];\n",
                           f"{target}'s own-cell load into {variable}")
            own += 1
    if own != OWN_LOAD_EDITS:
        raise AssertionError(
            f"the certified {name} body makes {own} own-cell magnetic loads of the "
            f"shape this weld redirects, not {OWN_LOAD_EDITS}; a load left standing "
            f"would read the PRE-launch H")

    taps = 0
    for line in [line + "\n" for line in tail.splitlines() if "shift_dn(" in line]:
        head_text, _, rest = line.partition("shift_dn(")
        pointer, _, arguments = rest.partition(", ")
        if pointer not in H_TARGETS:
            raise AssertionError(
                f"the certified {name} body reads {pointer!r} through shift_dn; this "
                f"weld resolves exactly the three magnetic components {H_TARGETS}")
        if not arguments.rstrip("\n").endswith(");"):
            raise AssertionError(
                f"the certified shift_dn call {line!r} does not close on its own "
                f"line; the weld appends its argument pack to that closing line")
        component = H_TARGETS.index(pointer)
        carried = arguments.rstrip("\n")[: -len(");")]
        tail = _needle(
            tail, line,
            f"{head_text}shift_dn_recompute({component}, {carried}, weld);\n",
            f"the shifted magnetic load {line.strip()!r}")
        taps += 1
    if taps != HALO_TAPS:
        raise AssertionError(
            f"the certified {name} body makes {taps} shifted magnetic loads, not "
            f"{HALO_TAPS}; this weld redirects every one of them and a missed tap "
            f"would read a word another block is writing")
    for target in H_TARGETS:
        if f"{target}[" in tail:
            raise AssertionError(
                f"the welded curl half still reads {target}; in this signature that "
                f"pointer is the PRE-LAUNCH magnetic field and every read must be the "
                f"register or a recompute")
    if "shift_dn(" in tail:
        raise AssertionError(
            "the welded curl half still calls the certified shift_dn, which loads a "
            "stored H; every shifted tap must go through shift_dn_recompute")
    return {"prelude": prelude, "parameters": parameters, "head": head, "tail": tail}


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

def signature(variant: str, parameters: str) -> str:
    """The kernel's parameter list: the certified curl's, carried across VERBATIM.

    THE CERTIFIED LIST IS NOT RETYPED. ``parameters`` arrives from
    :func:`welded_curl_pieces` as the exact characters between the certified entry
    point's declaration and its body anchor, and this function only puts three groups
    around it: the six scratch outputs in front, the six pre-launch constitutive
    inputs behind them, and ``update_H``'s three ``kps`` volumes at the end.

    THE ``kms`` GROUP IS BOUND ONCE FOR BOTH HALVES because both read the INTEGER Yee
    sub-lattice -- and that is ASSERTED at plan time (:func:`resolve`) rather than
    assumed, because binding the half-integer set instead compiles and is a half-cell
    error in the absorber profile rather than a failure.

    ``Hx``/``Hy``/``Hz`` are already ``const float* __restrict__`` in the certified
    list and stay exactly where they are: they ARE the pre-launch magnetic field the
    recompute reads, and moving them would be an edit this weld does not need.
    """
    name = kernel_name(variant)
    return (f'''
extern "C" __global__ void {name}(
    // THE SCRATCH OUTPUTS. A DIFFERENT ALLOCATION from every pre-launch pointer below
    // -- assert_bindings_are_disjoint checks it by base address before every launch --
    // and that separation is the whole design: nothing this launch reads is ever a
    // word this launch wrote, so the curl half's foreign taps are a pure function of
    // unwritten memory rather than a race with another block.
    float* __restrict__ Hx_out, float* __restrict__ Hy_out,
    float* __restrict__ Hz_out,
    float* __restrict__ f_w_Hx_out, float* __restrict__ f_w_Hy_out,
    float* __restrict__ f_w_Hz_out,
    // update_H's OTHER pre-launch inputs. H itself is already in the certified curl's
    // own list below, as const, and is bound there.
    const float* __restrict__ f_w_Hx, const float* __restrict__ f_w_Hy,
    const float* __restrict__ f_w_Hz,
    const float* __restrict__ Bx, const float* __restrict__ By,
    const float* __restrict__ Bz,
    // ---- THE CERTIFIED {CERTIFIED_CURL[variant]} PARAMETER LIST, VERBATIM ----
'''
            + parameters
            + ''',
    // update_H's kps. Its kms IS the curl's kms above and is bound once: step_D reads
    // the INTEGER sub-lattice and update_H reads the INTEGER one too
    // (constitutive_sub_lattice("H") is False), so the two groups are the same three
    // volumes. There is no kms_half_* here and there must not be.
    const float* __restrict__ kps_x, const float* __restrict__ kps_y,
    const float* __restrict__ kps_z
) {
''')


def kernel_source(variant: str,
                  cond: Optional[Tuple[bool, bool, bool]] = None) -> str:
    """The whole device source for one variant (and, on the conductive one, one mask)."""
    pieces = welded_curl_pieces(variant, cond)
    weld = "\n".join([
        "",
        weld_args_construction().rstrip("\n"),
        "",
        "    // --- THE WELD: update_H, computed into registers and stored to SCRATCH",
        "    // Nothing written here is read by this launch. The curl half below takes",
        "    // its OWN cell's magnetic field from these registers and recomputes",
        "    // every foreign tap from PRE-LAUNCH state through the same",
        "    // raw_update_H_cell, so no thread observes another thread's store; the",
        "    // launcher rotates H/f_w_H against the scratch afterwards.",
        "    float own_h[3];",
        "    float own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    Hx_out[idx] = own_h[0]; Hy_out[idx] = own_h[1]; Hz_out[idx] = own_h[2];",
        "    f_w_Hx_out[idx] = own_w[0]; f_w_Hy_out[idx] = own_w[1];",
        "    f_w_Hz_out[idx] = own_w[2];",
        "",
        f"    // --- the certified {CERTIFIED_CURL[variant]} body from its own stride",
        "    // constants down, with the twelve magnetic reads redirected and NOTHING",
        "    // else touched.",
        "",
    ])
    return (pieces["prelude"] + constitutive_prelude() + weld_args_struct()
            + raw_update_H_cell_source() + resolution_source()
            + shift_dn_recompute_source()
            + signature(variant, pieces["parameters"])
            + pieces["head"] + weld + pieces["tail"] + "}\n")


#: The gate's mutation seam: ``(variant, mask) -> source``. A MUTABLE MODULE GLOBAL on
#: purpose, and read on EVERY call, so a rewritten body is a memo miss and reaches
#: NVRTC. A map memoized at first call would hand back the pre-mutation bytes forever
#: -- "a leg reporting a pass for a mutation it never applied", the defect this track
#: has hit three times.
_OVERRIDES: Dict[Tuple[str, Optional[Tuple[bool, bool, bool]]], str] = {}


def _key(variant: str,
         cond: Optional[Tuple[bool, bool, bool]]
         ) -> Tuple[str, Optional[Tuple[bool, bool, bool]]]:
    return (_check_variant(variant),
            None if cond is None else tuple(bool(flag) for flag in cond))


def current_source(variant: str,
                   cond: Optional[Tuple[bool, bool, bool]] = None) -> str:
    """The source a launch would compile now: the override if one stands, else shipped."""
    return _OVERRIDES.get(_key(variant, cond)) or kernel_source(variant, cond)


def set_kernel_source(variant: str, cond: Optional[Tuple[bool, bool, bool]],
                      source: str) -> None:
    """Install one deliberately altered body -- the gate's door, and only that."""
    _OVERRIDES[_key(variant, cond)] = source


def reset_kernel_sources() -> int:
    """Drop every override; returns how many went."""
    count = len(_OVERRIDES)
    _OVERRIDES.clear()
    return count


def device_sources() -> Dict[str, str]:
    """``{kernel name: source}`` -- the shape every family on this track publishes.

    The conductive entry is emitted at the ALL-CONDUCTIVE mask, which is the widest
    text: every ``#if COND*`` branch present, so a digest over it observes every line
    a narrower mask can produce. The per-mask texts are pinned separately by the gate.
    """
    return {KERNEL_NAMES["conductive"]: kernel_source("conductive",
                                                      (True, True, True)),
            KERNEL_NAMES["bfast"]: kernel_source("bfast")}


def source_digest() -> str:
    """One sha256 over the family's two device strings, in :data:`VARIANTS` order."""
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for variant in VARIANTS:
        digest.update(KERNEL_NAMES[variant].encode("utf-8"))
        digest.update(b"\0")
        digest.update(device_sources()[KERNEL_NAMES[variant]].encode("utf-8"))
    return digest.hexdigest()


def _get_kernel(variant: str, cond: Optional[Tuple[bool, bool, bool]] = None,
                source: Optional[str] = None):
    """Compile on first use, memoized on (name, options, policy, source).

    THE SOURCE IS IN THE KEY, so a mutation harness that handed in a rewritten string
    cannot be served the shipped binary and a late policy install cannot be served an
    earlier one. ``cupy`` is imported HERE, never at scope.
    """
    import cupy as cp  # noqa: PLC0415 - device-only, in a laptop-importable module

    code = current_source(variant, cond) if source is None else source
    name = kernel_name(variant)
    key = _kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# ---------------------------------------------------------------------------
# The variant, resolved from the run
# ---------------------------------------------------------------------------

def variant_for(fields: Any, grid: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(variant, refusal)`` -- which curl tail this run takes, or why neither.

    READ FROM THE RUN, never chosen by a caller, and through the two readers the ARRAY
    PATH itself uses: ``grid.bfast_active`` is the only switch stepping.py:392/:475
    reads for the BFAST term, and ``fields.condfac_for`` is the only place
    ``_apply_curl`` reads a conductivity (stepping.py:508).

    NEITHER IS A DEFAULT. A run that is both is refused -- and is a run BOTH certified
    predicates already refuse, so there is no cell for it -- and a run that is neither
    belongs to :mod:`.fused_hd_pair`, whose lossless tail this product must not
    displace.
    """
    try:
        bfast = bool(getattr(grid, "bfast_active", False))
    except Exception as error:  # noqa: BLE001 - a raise is a refusal here
        return None, (f"the grid could not be asked whether BFAST is active: "
                      f"{type(error).__name__}: {error}")
    try:
        cond = conductive_mask(fields)
    except Exception as error:  # noqa: BLE001
        return None, (f"fields could not be asked for a conductivity: "
                      f"{type(error).__name__}: {error}")
    conductive = any(cond)
    if bfast and conductive:
        return None, (
            "this run is BOTH a BFAST run and a conductive-step_D run. There is no "
            "such cell: covers_conductive_curl delegates to the certified curl "
            "predicate, which refuses a BFAST grid by name, and covers_bfast_curl "
            "delegates to the same predicate with the REAL fields, which refuses a "
            "conductive target by name -- so both halves refuse it and this product "
            "must not pick a tail for it")
    if bfast:
        return "bfast", None
    if conductive:
        return "conductive", None
    return None, (
        "neither a BFAST grid nor a conductivity on any step_D target: this run's "
        "step_D is the certified LOSSLESS real-PML curl, which fused_hd_pair welds. "
        "Admitting it here would be an overlap on a cell that product already serves")


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def covers_conductive_bfast_fused_hd_pair(fields: Any, pml: Any, grid: Any,
                                          sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal, this
    directory's convention.

    A CONJUNCTION OF THE TWO ARMS THE CENSUS RECORDS ON THESE CELLS, and nothing is
    weakened. The variant is resolved first, because it decides WHICH pair of
    predicates answers:

    * ``conductive`` -- ``coverage.covers_real_pml_constitutive(..., "H")`` (the census
      column ``cuda_constitutive/ordinary``) and
      ``conductive_kernels.covers_conductive_curl(..., "step_D")`` (the column
      ``cuda_conductive/conductive``);
    * ``bfast`` -- ``bfast_curl.covers_bfast_constitutive(..., "H")`` and
      ``bfast_curl.covers_bfast_curl(..., "step_D")``, which are the two arms the
      ``cuda_bfast`` family registers.

    THE ORDER IS THE DRIVER'S: the constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.

    WHAT THIS ADDS to the two halves: the seam's own withdraw clause, the shared
    boundary-triple check, the active-layer requirement the conductive variant's
    signature rests on, and the rotation's requirement that the six magnetic volumes
    be allocated attributes this weld may rebind.
    """
    variant, refusal = variant_for(fields, grid)
    if variant is None:
        return False, f"variant: {refusal}"

    if variant == "conductive":
        covered, reason = covers_real_pml_constitutive(fields, pml, grid, "H")
    else:
        covered, reason = _bfast.covers_bfast_constitutive(fields, pml, grid, "H")
    if not covered:
        return False, f"constitutive half ({variant}): {reason}"

    if variant == "conductive":
        # THE ACTIVE LAYER IS THIS PRODUCT'S OWN CLAUSE and it is asked BEFORE the
        # curl predicate, which admits both layers. Without an absorber the array path
        # takes a different tail and the certified family a different kernel -- one
        # with no f_cond history and no split-field pair -- so this signature has
        # nothing to bind. See ACTIVE_LAYER_ONLY.
        layer, _arm = _conductive.conductive_arm(fields, pml, "step_D")
        if layer != "active":
            return False, (
                "no active absorber layer: with none, step_D's conductive tail is "
                "stepping._apply_conductive_update -- one expression, no f_cond "
                "history and no split-field pair -- served by the certified "
                "step_D_no_pml_conductive, whose signature this weld does not carry. "
                "That configuration is a separate product (ACTIVE_LAYER_ONLY)")
        covered, reason = _conductive.covers_conductive_curl(
            fields, pml, grid, "step_D")
    else:
        covered, reason = _bfast.covers_bfast_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half ({variant}): {reason}"

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all. What IS between them is the electric
    # integrated-source withdraw (driver.py:3313-3314), and `withdraw_hoist` owns it.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no withdraw stands" from not being told would be the
    # over-covering refusal this clause exists to prevent.
    seam_reasons = _withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False because it declares INSTALLABLE = False, so "
            f"fused_pairs._install_fused_pair's withdraw-hoist branch is unreachable "
            f"for it and nothing would perform the withdraw before the launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES)
    if seam_reasons:
        return False, seam_reasons[0]

    # ONE BOUNDARY TRIPLE FOR BOTH HALVES, CHECKED. Both certified curl launchers
    # resolve it through `coverage.real_curl_boundary_codes`, which fails closed and
    # returns the refusal rather than raising.
    codes, refusal = _coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        return False, (f"the curl's own boundary resolution refuses this grid: "
                       f"{refusal}")
    if len(tuple(codes)) != 3:
        return False, (f"the curl's boundary resolution returned {codes!r}, which is "
                       f"not the per-axis triple this launch binds")

    # THE SUB-LATTICE, ASSERTED RATHER THAN ASSUMED. This kernel binds ONE kms group
    # for both halves, which is only correct while step_D's curl and update_H read the
    # SAME Yee sub-lattice. Both read the integer positions today. If either moved,
    # sharing the group would bind one half's coefficients to the other half's lattice
    # -- a converged, smooth, half-cell-wrong absorber profile rather than a failure.
    if constitutive_sub_lattice("H"):
        return False, ("update_H now reads the HALF-INTEGER PML sub-lattice while "
                       "step_D's curl reads the integer one; this weld binds ONE kms "
                       "group for both halves and may only do so while they agree")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The launcher swaps the ENGINE's
    # references for the six volumes in SCRATCH_VOLUMES after every launch, so those
    # attributes must exist and be allocated.
    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this weld rotates it against a "
                           f"launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def resolve(fields: Any, grid: Any, pml: Any, *,
            variant: Optional[str] = None,
            tables: Optional[Dict[str, Any]] = None,
            boundary_codes: Optional[Sequence[int]] = None,
            coefficients: Optional[Sequence[Sequence[float]]] = None,
            scratch: Optional[Dict[str, Any]] = None,
            dtdx: Optional[float] = None) -> Dict[str, Any]:
    """Everything the launch needs, derived once per frozen configuration.

    DERIVED HERE BY THE SAME FUNCTIONS THE CERTIFIED LAUNCHERS ASK, which is what makes
    "the weld and the singles cannot disagree about which sub-lattice, which boundary
    code or which BFAST coefficient this seam reads" an enforced property rather than
    a convention:

    * the curl's ``kms``/``sinv`` from ``step_curl_kernels.real_pml_curl_tables(pml,
      False)`` -- ``half_integer`` False is ``step_D``'s, and getting it backwards is a
      half-cell error rather than a crash;
    * ``update_H``'s ``kps`` from ``constitutive_kernels.real_constitutive_tables``;
    * the boundary triple from ``coverage.real_curl_boundary_codes``;
    * on the BFAST variant, the three ``(k1, k2)`` pairs from
      ``bfast_curl.bfast_curl_coefficients(grid, "step_D")``, whose host-side
      invariance gates are the transcription's whole point.

    THE SHARED ``kms`` IS CHECKED BY POINTER, not assumed: the two groups must be the
    same allocations, or this signature's single binding is one half's coefficients on
    the other half's lattice.

    THE OVERRIDES ARE THE GATE'S DOOR and stay, keyword-only and named for what they
    are: a gate feeds deliberately wrong tables, wrong boundary codes and wrong BFAST
    coefficients and requires each to diverge.
    """
    from . import constitutive_kernels, step_curl_kernels  # noqa: PLC0415

    if variant is None:
        variant, refusal = variant_for(fields, grid)
        if variant is None:
            raise ValueError(
                f"this run has no variant for this product: {refusal}. Ask "
                f"covers_conductive_bfast_fused_hd_pair first; it refuses rather "
                f"than raising")
    _check_variant(variant)
    cond = conductive_mask(fields) if variant == "conductive" else None

    if tables is None:
        curl = step_curl_kernels.real_pml_curl_tables(pml, False)
        constitutive = constitutive_kernels.real_constitutive_tables(
            pml, constitutive_sub_lattice("H"))
        for axis in ("x", "y", "z"):
            left, right = curl[f"kms_{axis}"], constitutive[f"kms_{axis}"]
            if left.data.ptr != right.data.ptr:
                raise ValueError(
                    f"step_D's kms_{axis} and update_H's kms_{axis} are different "
                    f"allocations; this weld binds ONE kms group for both halves and "
                    f"may only do so while both read the INTEGER Yee sub-lattice")
        tables = {"kms": {axis: curl[f"kms_{axis}"] for axis in "xyz"},
                  "sinv": {axis: curl[f"sinv_{axis}"] for axis in "xyz"},
                  "kps": {axis: constitutive[f"kps_{axis}"] for axis in "xyz"}}

    if boundary_codes is None:
        codes, refusal = _coverage.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise ValueError(
                f"this grid has no boundary-code triple for the real-field PML curl: "
                f"{refusal}. Ask covers_conductive_bfast_fused_hd_pair first")
        boundary_codes = tuple(int(code) for code in codes)

    if variant == "bfast" and coefficients is None:
        coefficients = _bfast.bfast_curl_coefficients(grid, "step_D")

    if scratch is None:
        import cupy as cp  # noqa: PLC0415 - device-only

        scratch = {}
        for name in SCRATCH_VOLUMES:
            source = getattr(fields, name, None)
            if source is None:
                raise ValueError(
                    f"fields.{name} is not allocated; this weld rotates a scratch "
                    f"shaped like it and has nothing to shape one from")
            # UNINITIALIZED IS CORRECT. Every cell of all six is written by every
            # launch, so there is no cell whose prior contents a reader could observe.
            scratch[name] = cp.empty_like(source)

    return {"variant": variant, "cond": cond, "tables": tables,
            "boundary_codes": tuple(int(code) for code in boundary_codes),
            "coefficients": (None if coefficients is None else
                             tuple(tuple(float(v) for v in pair)
                                   for pair in coefficients)),
            "scratch": scratch,
            "dtdx": float(grid.dt / grid.dx) if dtdx is None else float(dtdx)}


def assert_bindings_are_disjoint(fields: Any, state: Dict[str, Any]) -> int:
    """THE CHECK THE WHOLE DESIGN RESTS ON, plus the ordinary restrict promise.

    Two things, and the first is not a convention:

    1. **no scratch volume is any bound input.** If ``Hx_out`` were ``Hx`` the launch
       would be the IN-PLACE weld the board refused -- every foreign recompute would
       read words other blocks had already overwritten, and the answer would be a
       schedule. This is the gate's ``scratch_aliased_to_storage`` mutation.
    2. **no two ``__restrict__`` arguments are one allocation.** Two restrict pointers
       to one object is UB whatever the route to it, and NVRTC reorders across it
       without a diagnostic.

    THE SHARED ``kms`` GROUP IS THE ONE DELIBERATE COINCIDENCE and is bound ONCE, so
    it is visited once and is not a collision. A LOSSLESS COMPONENT'S ``cf``/``ci``/
    ``fc`` placeholder is the target volume itself -- the certified launcher's own
    convention, since the preprocessor guarantees the pointer is never dereferenced --
    so those three are visited only where the mask says the branch is live.

    Returns the number of distinct allocations inspected, so a caller can assert that
    something was actually examined.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in SCRATCH_VOLUMES:
        visit(f"{name}_out", state["scratch"][name])
    for name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "fu_Dx", "fu_Dy", "fu_Dz"):
        visit(name, getattr(fields, name))
    if state["variant"] == "conductive":
        for index, (component, live) in enumerate(zip(D_TARGETS, state["cond"])):
            if not live:
                continue
            visit(f"condfac_{component}", fields.condfac_for(component))
            visit(f"condinv_{component}", fields.condinv_for(component))
            visit(f"f_cond_{component}", getattr(fields, f"f_cond_{component}"))
            del index
    else:
        for name in _bfast.BFAST_STATE["step_D"]:
            visit(name, getattr(fields, name))
    for group in ("kms", "sinv", "kps"):
        for axis in "xyz":
            visit(f"{group}_{axis}", state["tables"][group][axis])
    if collisions:
        raise ValueError(
            "this weld binds every field, scratch and table argument __restrict__, "
            "and these arguments alias -- which is undefined behaviour NVRTC "
            "miscompiles silently rather than diagnosing, and which for the scratch "
            "group would make the launch the in-place weld the board refused: "
            + "; ".join(collisions))
    return len(bound)


def _launch_arguments(fields: Any, state: Dict[str, Any]) -> List[Any]:
    """The argument tuple, in the emitted signature's own order.

    WRITTEN OUT RATHER THAN ASSEMBLED FROM A TABLE, and the order is the certified
    launcher's with this weld's three groups around it: a reordering here is a wrong
    launch that no source digest could catch.
    """
    variant = state["variant"]
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    arguments: List[Any] = [state["scratch"][name] for name in SCRATCH_VOLUMES]
    arguments += [getattr(fields, name) for name in
                  ("f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz")]
    # ---- the certified list ----
    arguments += [getattr(fields, name) for name in D_TARGETS]
    arguments += [getattr(fields, f"fu_{name}") for name in D_TARGETS]
    if variant == "bfast":
        arguments += [getattr(fields, name)
                      for name in _bfast.BFAST_STATE["step_D"]]
    arguments += [getattr(fields, name) for name in H_TARGETS]
    if variant == "conductive":
        targets = tuple(getattr(fields, name) for name in D_TARGETS)
        cond = state["cond"]
        arguments += [fields.condfac_for(name) if live else target
                      for name, live, target in zip(D_TARGETS, cond, targets)]
        arguments += [fields.condinv_for(name) if live else target
                      for name, live, target in zip(D_TARGETS, cond, targets)]
        arguments += [getattr(fields, f"f_cond_{name}") if live else target
                      for name, live, target in zip(D_TARGETS, cond, targets)]
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz),
                  np.float32(state["dtdx"])]
    for axis in ("x", "y", "z"):
        arguments += [state["tables"]["kms"][axis], state["tables"]["sinv"][axis]]
    arguments += [np.int32(int(code)) for code in state["boundary_codes"]]
    if variant == "bfast":
        arguments += [np.float32(value) for pair in state["coefficients"]
                      for value in pair]
    # ---- this weld's tail ----
    arguments += [state["tables"]["kps"][axis] for axis in ("x", "y", "z")]
    return arguments


def launch_conductive_bfast_fused_hd_pair(fields: Any, state: Dict[str, Any], *,
                                          kernel: Optional[Any] = None,
                                          threads: int = _FUSED_THREADS
                                          ) -> Dict[str, Any]:
    """Both of :data:`REPLACES` in ONE launch. THE CALLER ROTATES AFTERWARDS.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is: a gate
    compiles a deliberately broken copy of the shipped source and hands it here. A
    launcher that could not be handed its own kernel could not arm a single mutation.

    ``threads`` is the gate's schedule door. The design's whole claim is that the
    answer does NOT depend on which block ran first, and a block size that never moves
    cannot expose the opposite.

    Returns the launch geometry rather than ``None`` so a gate can assert that
    something was actually launched.
    """
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + threads - 1) // threads
    arguments = _launch_arguments(fields, state)
    launched = kernel or _get_kernel(state["variant"], state["cond"])
    launched((blocks,), (threads,), tuple(arguments))
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "variant": state["variant"], "cond": state["cond"],
            "arguments": len(arguments),
            "boundary_codes": tuple(state["boundary_codes"])}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION. The scratch becomes the live H/f_w_H; the retired pair becomes scratch.

    :func:`.fused_hd_pair.rotate_into_fields`'s choreography and its consumer audit,
    reused rather than re-derived: every consumer of the magnetic field in this engine
    resolves it BY NAME at use time, so a rebinding between steps is invisible to all
    of them.

    Returns the mapping the caller should keep as the NEXT launch's scratch.
    """
    return _hd.rotate_into_fields(fields, scratch)


def run_conductive_bfast_fused_hd_pair(fields: Any, grid: Any, pml: Any,
                                       dtdx: Optional[float] = None, *,
                                       sources: Any = None,
                                       state: Optional[Dict[str, Any]] = None,
                                       kernel: Optional[Any] = None,
                                       threads: int = _FUSED_THREADS,
                                       rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the run, launch once, rotate.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``state``, ``kernel``, ``threads`` and ``rotate`` are the gate's doors,
    keyword-only. ``rotate=False`` is how the ``rotation_skipped`` mutation is armed: a
    launcher that always rotated could not measure what the rotation is worth.
    """
    covered, reason = covers_conductive_bfast_fused_hd_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if state is None:
        state = resolve(fields, grid, pml, dtdx=dtdx)
    assert_bindings_are_disjoint(fields, state)
    record = launch_conductive_bfast_fused_hd_pair(fields, state,
                                                   kernel=kernel, threads=threads)
    record["rotated"] = bool(rotate)
    record["scratch"] = (rotate_into_fields(fields, state["scratch"]) if rotate
                         else state["scratch"])
    if rotate:
        state["scratch"] = record["scratch"]
    return record
