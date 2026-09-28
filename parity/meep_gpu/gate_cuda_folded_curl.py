"""Sub-step byte-identity gate for the SHIPPED CUDA real-PML curl pair ON A FOLD.

TWO ROUNDS, AND THE SECOND IS WHAT THIS FILE NOW RUNS.

ROUND 1 (2026-08-19, ``results/cuda_folded_curl_2026-08-19/``) changed no device
byte and asked whether ``covers_real_pml_curl``'s blanket mirror refusal was a
refusal the arithmetic required, and if so WHERE, cell by cell. It answered with a
split: a fold whose every folded axis is METALLIC was ALREADY exact, 48/48 at one
launch and at 60 under both float32 subnormal policies; a fold with a PERIODIC
termination diverged 0/64, with 19,649 differing words of which 19,649 lay on the
mask-delta plane and ZERO anywhere else, always the same entry
``('oracle_only', -1)``.

ROUND 2 -- this run -- measures the branch written against that specification.
``step_curl_kernels.py`` now carries a third boundary code,
``BC_MIRROR_PERIODIC``, whose entire difference from ``BC_METALLIC`` is the
Yee-shift-1 TOP-PLANE mask ``stepping._stored_past_owned`` (stepping.py:1503-1518)
calls for, plus the cell-0 mask a folded axis carries because it is mirrored
rather than because it is metallic. The question is now:

  * under the SHIPPED resolution (``coverage.real_curl_boundary_codes``), is the
    pair byte-identical on a fold at BOTH terminations, per sub-step, at one
    launch and at 60 -- and
  * does the PRE-BRANCH behaviour still diverge, on that one plane and nowhere
    else, when the same folded PERIODIC axis is deliberately handed the metallic
    code? A fix whose control also went identical would be a fix nothing
    distinguishes from a change in the fixture.

The device source IS changed this round, which is why the mutation battery gained
``drop_folded_periodic_top_mask`` and ``drop_folded_periodic_near_mask``: a branch
no mutation can break is a branch this gate cannot claim.

WHY THIS IS NOT THE CONSTITUTIVE QUESTION AGAIN. On 2026-08-19 the same refusal
was retired from ``covers_real_pml_constitutive`` because that sub-step READS NO
NEIGHBOUR: a fold changes the ghost rule and the ownership mask, and an
element-wise product has neither. A CURL HAS BOTH. So the constitutive verdict
transfers nothing, and the answer here is expected to be different in kind: not
"the fold is invisible" but "the fold is invisible EXCEPT ON THESE PLANES".

WHAT THE READING SAYS BEFORE ANY DEVICE RUNS (four facts off ``stepping.py``,
and the sibling track's measured conclusion):

1. THE FOLDED GHOST VALUES ARE DEAD. ``_shift_down``'s MIRROR branch writes
   ``parity * field[2]`` at stored cell 0 (stepping.py:1870-1873,
   ``MIRROR_SOURCE_INDEX = 2``) and ``_shift_up``'s writes
   ``parity * field[reflect_row]`` at the last stored slot of a folded PERIODIC
   axis (:1772-1780) or an exact zero on a folded METALLIC one (:1781-1783).
   Each ghost has exactly ONE consumer plane, and both consumer planes are
   dropped by ``_mask_non_owned_cells`` (:1865). The parity arithmetic therefore
   never reaches a surviving output word.
2. THE MASK IS NOT DEAD. ``_mask_non_owned_cells`` zeroes cell 0 on a mirrored
   axis for every component whose Yee shift there is 0, and -- only on a folded
   PERIODIC axis, where ``_stored_past_owned`` (:1454) is true -- zeroes the LAST
   stored slot for every component whose Yee shift there is 1.
3. THE KERNEL HAD ONE OF THOSE TWO MASKS AND NOT THE OTHER, up to 2026-08-20. It
   masked cell 0 when ``bc == BC_METALLIC`` and never masked a top plane; it had
   no MIRROR code at all and ``real_pml_boundary_codes(kinds)`` RAISED on one.
   Round 1 measured that as a divergence on exactly the missing plane, and the
   pair now carries ``BC_MIRROR_PERIODIC`` with both fold masks. Reading 3 is
   therefore the record of what was fixed, and ``mirror_as_metallic`` is the arm
   that still reproduces it.
4. The sibling Triton track measured the same question first and wired the
   answer: ``triton_kernels/symmetry.py:39-41`` -- "the whole kernel delta is
   (a) admit the fold in the predicate, and (b) add the Yee-shift-1 top-plane
   mask on a folded PERIODIC axis". CUDA predicates have matched Triton within
   0-1 rows on every family ported so far, which is a reason to MEASURE this
   one, not to assume it.

The reading predicted a SPLIT verdict and round 1 measured it:

* a folded METALLIC axis is exactly what ``BC_METALLIC`` already implements --
  zero far ghost, cell-0 mask, no top-plane mask needed. MEASURED 48/48.
* a folded PERIODIC axis DIVERGED, on the last stored plane of every component
  whose Yee shift on that axis is 1, and nowhere else. MEASURED 0/64, 19,649
  differing words, 0 elsewhere.

Round 2 tests the branch built to that specification, and keeps the diverging
substitution as the control that isolates it.

=============================================================================
ONE SHIPPED ARM, TWO MISDECLARATION CONTROLS
=============================================================================

A folded axis resolves to ``mirror`` (``stepping._boundary_kinds``) at BOTH
terminations, so the code triple cannot be a function of the kind strings alone.
``coverage.real_curl_boundary_codes`` splits it off the grid, and three arms are
swept:

* ``mirror_resolved`` -- the SHIPPED split, and the only triple a dispatch can
  produce. This is the arm release is gated on.
* ``mirror_as_metallic`` -- the round-1 substitution, kept as a CONTROL. On a
  folded PERIODIC axis it is exactly the pre-branch kernel and must still diverge.
* ``mirror_as_periodic`` -- the wrap, and no mask at all. Kept for the same reason.

Reporting the licensed arm alone would leave "the branch is what closed the gap"
an inference rather than a measurement.

=============================================================================
WHY A SUB-STEP GATE AND NOT A WHOLE-STEP ONE
=============================================================================

A WHOLE-STEP GATE CANNOT SEE A MASK. The driver's fold repairs
(``fill_symmetry_bc_*``, ``fill_folded_far_ghosts_*``) overwrite exactly the
planes a missing mask would corrupt -- ``symmetry.py:44-48`` records that a
kernel carrying NEITHER mask is bytewise-identical after a complete step, so a
whole-step check certifies a mask-less symmetry kernel as correct. Every
comparison here is ONE SUB-STEP from ONE frozen state.

=============================================================================
LOCALIZATION: WHAT MAKES A DIVERGENCE A FINDING RATHER THAN A FAILURE
=============================================================================

Every diverging case is decomposed by PLANE. For each output array the gate
computes the planes the ORACLE masks and the KERNEL does not (and vice versa),
then counts the differing words that lie on those planes against the differing
words that do not. "Diverged" is a failure; "diverged, and every one of the N
differing words lies on the mask-delta planes, none elsewhere" is a specification
for the kernel change that would fix it.

=============================================================================
WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
=============================================================================

* ``oracle_moved`` -- the fraction of output words the array path changed from
  the frozen input. A case that moved nothing is refused, not passed.
* ``folded_axis_absorbs`` -- ``max|kms-1|`` and ``max|sinv-1|`` ON THE FOLDED
  AXIS. If the folded axis's coefficient vector is the identity everywhere, a
  coefficient-index error on that axis is invisible.
* ``mask_delta_planes_are_live`` -- the planes the mask delta names must carry a
  NONZERO oracle curl. A top-plane mask whose curl is zero anyway is a mask
  nothing can distinguish, and a case at that floor would report IDENTICAL for a
  reason about the fixture rather than about the kernel.
* The mutation battery, including legs that MUST BE UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_folded_curl.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the
same floors and the same HOST mutations against a transcription of the shipped
device tree. IT COMPILES NOTHING AND CERTIFIES NOTHING -- what it settles is
whether the harness can localize a divergence and whether it can fail::

    python -u gate_cuda_folded_curl.py --backend numpy \\
        --out /tmp/folded_curl_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: the NumPy backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields, IYEE_SHIFTS  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260819

#: Consecutive launches in the multi-step leg. 60 is the budget both certified
#: hand-CUDA records are cut at. "Identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence live. Held fixed, the curl operands
#: never move and a 60-launch leg becomes a slow single-launch leg. The same exact
#: float32 scale is applied on both paths, so it cancels out of the comparison.
_ADVANCE = np.float32(0.97)

#: Per sub-step: what it writes, its PML auxiliaries, what it reads, and which
#: PML sub-lattice its coefficient tables come from. ``half_integer`` TRUE for the
#: B curl (``stepping.step_B`` passes ``half_integer=True``, stepping.py:404) and
#: FALSE for D (:486). Backwards is a half-cell error in the absorber profile:
#: converged, smooth and wrong.
SUB_STEP_ARRAYS: Dict[str, Dict[str, Any]] = {
    "step_B": {"targets": ("Bx", "By", "Bz"),
               "aux": ("fu_Bx", "fu_By", "fu_Bz"),
               "sources": ("Ex", "Ey", "Ez"),
               "half_integer": True,
               "backward": False},
    "step_D": {"targets": ("Dx", "Dy", "Dz"),
               "aux": ("fu_Dx", "fu_Dy", "fu_Dz"),
               "sources": ("Hx", "Hy", "Hz"),
               "half_integer": False,
               "backward": True},
}

#: The six curl terms as the SHIPPED KERNEL spells them, transcribed from
#: ``step_curl_kernels.py:1700-1731`` (B) and :1760-1793 (D). Read as
#: ``(target, aux, first source, first source's axis, second source, second
#: source's axis, dsig axis, dsigu axis)``. The two source axes are the axes the
#: stencil shifts along; ``dsig``/``dsigu`` select which coefficient vectors the
#: recurrence pairs -- and the kernel's own note calls confusing them the
#: highest-consequence error in the file.
#:
#: Written out here rather than derived from ``stepping.B_CURL_TERMS`` on purpose:
#: this table is what the NumPy leg transcribes the DEVICE source into, and
#: deriving it from the oracle would make that leg compare the oracle with itself.
#: :func:`check_terms_against_stepping` pins it against ``stepping`` so the
#: transcription cannot drift silently.
CURL_TERMS: Dict[str, Tuple[Tuple[str, str, str, int, str, int, int, int], ...]] = {
    "step_B": (
        ("Bx", "fu_Bx", "Ez", 1, "Ey", 2, 1, 2),
        ("By", "fu_By", "Ex", 2, "Ez", 0, 2, 0),
        ("Bz", "fu_Bz", "Ey", 0, "Ex", 1, 0, 1),
    ),
    "step_D": (
        ("Dx", "fu_Dx", "Hz", 1, "Hy", 2, 1, 2),
        ("Dy", "fu_Dy", "Hx", 2, "Hz", 0, 2, 0),
        ("Dz", "fu_Dz", "Hy", 0, "Hx", 1, 0, 1),
    ),
}

BC_PERIODIC = 0
BC_METALLIC = 1
#: Added to the shipped kernels 2026-08-20, and the whole subject of this re-run.
#: A folded axis whose termination is PERIODIC: same dead ghosts as METALLIC, plus
#: the Yee-shift-1 top-plane mask ``stepping._stored_past_owned`` calls for.
BC_MIRROR_PERIODIC = 2

#: How the folded axis's code is chosen. THREE ARMS, and only one of them is the
#: shipped answer:
#:
#: * ``mirror_resolved`` -- the SHIPPED resolution, ``coverage.real_curl_boundary
#:   _codes``: folded METALLIC -> BC_METALLIC, folded PERIODIC -> BC_MIRROR_PERIODIC.
#:   This is the arm the gate licenses, and the only one a dispatch could ever take.
#: * ``mirror_as_metallic`` -- the 2026-08-19 substitution, KEPT AS A CONTROL. It
#:   hands a folded PERIODIC axis the metallic code, which is exactly the kernel
#:   this pair was before today. Its folded-periodic arm should still diverge, on
#:   the same single plane and nowhere else. That is what shows the identity in the
#:   resolved arm comes from the NEW BRANCH and not from a perturbation elsewhere:
#:   a fix whose control also went identical would be a fix nothing distinguishes.
#: * ``mirror_as_periodic`` -- the other deliberate misdeclaration, kept for the
#:   same reason and because reporting one substitution alone reports a property of
#:   the choice as though it were a property of the kernel.
SUBSTITUTIONS: Tuple[str, ...] = ("mirror_resolved", "mirror_as_metallic",
                                  "mirror_as_periodic")

#: The one arm a dispatch could take, and so the only one release is gated on.
LICENSED_SUBSTITUTION = "mirror_resolved"

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")


def outputs(sub_step: str) -> Tuple[str, ...]:
    """The six arrays this sub-step writes.

    ``fu`` IS STATE. A tree that gets the field right and the auxiliary wrong is
    correct for exactly one launch and wrong forever after, so the auxiliary is
    compared on every case, not only in the multi-step leg. The sibling track
    measured a dropped auxiliary store at 120/120 UNCAUGHT when ``fu`` was not
    compared.
    """
    spec = SUB_STEP_ARRAYS[sub_step]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def state_names(sub_step: str) -> Tuple[str, ...]:
    return outputs(sub_step) + tuple(SUB_STEP_ARRAYS[sub_step]["sources"])


# ---------------------------------------------------------------------------
# The case product
# ---------------------------------------------------------------------------
#
# EVERY AXIS IS FOLDED SOMEWHERE IN THIS LIST, and both terminations appear on
# each. The two terminations are not two spellings of one case here: on a folded
# PERIODIC axis the stored array carries ONE SLOT PAST MEEP's owned window and the
# array path masks it; on a folded METALLIC axis it does not and the top plane is
# genuinely stepped. That difference IS the question, so a sweep carrying one
# termination would answer half of it and report the half as the whole.

FOLD_SPECS: Tuple[Dict[str, Any], ...] = (
    # The unfolded controls. Already certified; carried so the harness is shown to
    # reproduce the standing result under its own fixture rather than only under
    # the fixture that produced the record.
    {"label": "unfolded_periodic", "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "unfolded_metallic", "axes": "", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (8.0, 10.0, 12.0)},

    # Folded Y, both terminations, both parities of the plane, even full count.
    {"label": "fold_Y_periodic", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_metallic", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_periodic_odd_plane", "axes": "Y", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_metallic_odd_plane", "axes": "Y", "phase": -1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 16.0, 12.0)},

    # ODD FULL COUNT on the folded axis, both terminations. MEEP's fold lands on a
    # grid point at both parities but shifts the window, and the reflect row moves
    # with it (``_far_reflect_rows``) -- the sibling gate measured one mutation
    # caught ONLY at an odd count.
    {"label": "fold_Y_periodic_odd_count", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 17.0, 12.0)},
    {"label": "fold_Y_metallic_odd_count", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 17.0, 12.0)},

    # Folded X and folded Z: the axis whose index is the SLOWEST stride and the
    # one whose index is the FASTEST. A decomposition defect that confuses them is
    # only visible if both are folded somewhere in the sweep.
    {"label": "fold_X_periodic", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_metallic", "axes": "X", "phase": 1,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_Z_periodic", "axes": "Z", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 12.0, 16.0)},
    {"label": "fold_Z_metallic", "axes": "Z", "phase": -1,
     "boundaries": ("periodic", "periodic", "metallic"), "cell": (8.0, 12.0, 16.0)},

    # TWO PLANES AT ONCE: the doubly-folded corner, one axis terminated each way.
    {"label": "fold_XY_mixed", "axes": "XY", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (16.0, 16.0, 10.0)},
    {"label": "fold_XZ_mixed", "axes": "XZ", "phase": -1,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 10.0, 16.0)},

    # BOTH FOLDED AXES PERIODIC: two top planes masked at once. The mixed cases
    # above carry one of each, so neither of them can show a defect that only
    # appears when two top-plane masks interact.
    {"label": "fold_XY_periodic", "axes": "XY", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 16.0, 10.0)},
    # BOTH FOLDED AXES METALLIC: the arm the reading predicts is already exact,
    # with no top plane masked on either.
    {"label": "fold_XY_metallic", "axes": "XY", "phase": -1,
     "boundaries": ("metallic", "metallic", "periodic"), "cell": (16.0, 16.0, 10.0)},

    # THREE PLANES AT ONCE, added 2026-08-20. Until this round the predicate
    # refused a three-plane fold BY NAME -- ``CURL_FOLD_ADMISSION`` recorded
    # ``folded_planes_swept = 2`` BY CONSTRUCTION and the clause quoted it back --
    # so the refusal was a statement about this file's case table and not about
    # the kernel. The corpus drives exactly one triply-folded row
    # (``TestLDOS.test_ldos_3D``, mirrored and metallic on all three axes) and it
    # costs four slots at the census.
    #
    # ALL THREE TERMINATION ARMS ARE SWEPT, not only the corpus's. The corpus row
    # is all-metallic, which is the arm the reading says needs no device code at
    # all; the all-PERIODIC one is the arm that fires THREE top-plane masks at
    # once and is the only place a defect in how two of them interact could show a
    # third time; the mixed one carries a different code on each axis, which is the
    # combination a per-axis code triple can get wrong in a way no uniform arm can.
    #
    # THE EXTENTS ARE DELIBERATELY UNEQUAL (9, 10, 11 stored). A cube would let an
    # index decomposition confuse i, j and k and stay bit-identical, which is
    # exactly the defect ``fortran_order_index_decomposition`` plants.
    {"label": "fold_XYZ_metallic", "axes": "XYZ", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"),
     "cell": (16.0, 18.0, 20.0)},
    {"label": "fold_XYZ_periodic", "axes": "XYZ", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (16.0, 18.0, 20.0)},
    {"label": "fold_XYZ_mixed", "axes": "XYZ", "phase": -1,
     "boundaries": ("metallic", "periodic", "metallic"),
     "cell": (16.0, 18.0, 20.0)},
)

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient, so it is a real second draw of the tables.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: ``--fmad=false`` is CORRECTNESS on this sub-step, not tuning: ``(fu*kms) - curl``
#: and ``(f*kms_u) + fu_new`` are both contraction candidates the array path rounds
#: twice. The control is EXPECTED TO DIVERGE at the inexact courant; if it does
#: not, the guard is decorative here and the record must say so rather than
#: implying evidence it does not have.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: Source mutations, borrowed from the shared probe so this gate cannot own a
#: second spelling of a defect the certified record was cut against. Device
#: backend only: they rewrite the module's device strings.
SOURCE_MUTATIONS: Tuple[str, ...] = (
    "regroup_stencil",                    # the parenthesisation the record pins
    "drop_metallic_mask",                 # the wall mask a folded METALLIC axis inherits
    "drop_folded_periodic_top_mask",      # THE BRANCH ADDED 2026-08-20
    "drop_folded_periodic_near_mask",     # its cell-0 twin on the same axis
    "swap_dsig_dsigu",                    # the coefficient pairing
    "drop_fu_store",                      # the auxiliary-only defect
    "read_fprev_after_store",             # the aliasing trap
    "fortran_order_index_decomposition",  # i and k swapped: the curl's index defect
    # THE TWO NULLS. A battery whose every leg must be caught scores identically
    # whether the comparator works or has degenerated into failing everything.
    "commute_dtdx_scale",
    "reload_fu_from_memory",
)

#: A mutation is scored only on legs where its lines can FIRE. Deleting the
#: metallic wall mask on a grid with no metallic-coded axis changes nothing, and
#: deleting the folded-periodic masks on a grid with no folded-periodic axis
#: changes nothing -- an UNCAUGHT that is a fact about the fixture, not about the
#: kernel, and one that would block release for the wrong reason. Each entry names
#: the code a leg's grid must carry for that mutation to be asked there.
#:
#: THIS IS NOT A WAY TO EXCUSE AN UNCAUGHT LEG. Every mutation here still has to
#: be CAUGHT on the legs that do carry its code, and ``_score`` reports NO LEGS --
#: never "inert" -- if the filter leaves none.
SOURCE_MUTATION_REQUIRES_CODE: Dict[str, int] = {
    "drop_metallic_mask": BC_METALLIC,
    "drop_folded_periodic_top_mask": BC_MIRROR_PERIODIC,
    "drop_folded_periodic_near_mask": BC_MIRROR_PERIODIC,
}

#: MEASURED NOT ARMED on 2026-08-19 and dropped from the list above rather than
#: left in it to fail: ``own_axis_to_x_for_all_three`` rewrites the CONSTITUTIVE
#: pair's own-axis coefficient binding (``kps_x[i]`` for all three components) and
#: matched 0 sites in either curl kernel, on both sub-steps. The curl's own
#: indexing defect is ``fortran_order_index_decomposition``, which armed at 1 site
#: and was CAUGHT 5/5. Recorded here because a mutation silently absent from a
#: battery and a mutation that cannot bite look identical in a verdict.
SOURCE_MUTATIONS_WITH_NO_CURL_SITE: Tuple[str, ...] = ("own_axis_to_x_for_all_three",)
NULL_SOURCE_MUTATIONS: Tuple[str, ...] = probe.PML_NULL_MUTATIONS

#: HOST mutations: they corrupt the TABLES rather than the device text, so they
#: run on both backends. ``reverse_folded_axis_coefficients`` is the fold's own --
#: the retired constitutive refusal's reason, armed as a defect on a family that
#: has not retired it.
HOST_MUTATIONS: Tuple[str, ...] = (
    "reverse_folded_axis_coefficients",
    "swap_curl_sublattice",
)
FOLD_ONLY_HOST_MUTATIONS: Tuple[str, ...] = ("reverse_folded_axis_coefficients",)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and that
    is the one thing about the device library a laptop cannot supply. Everything
    else the fixture exercises -- the fold, the stored extent, the coefficient
    vector lengths, the dtype and contiguity -- is a real object either way.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float):
    """A frozen ``(fields, layer, grid)`` triple for one fold spec.

    THE THICKNESS RULE IS THE SLICE'S OWN (``test_step_curl_pml_real.build``):
    skip an axis too thin to hold a layer, and on a MIRRORED axis ask for the HIGH
    face only. ``PML._resolve_mirror_faces`` refuses a named low face on a folded
    axis outright -- cell 0 is the mirror plane, a boundary condition rather than
    a wall -- so this is the shape the engine accepts, and it is also what keeps
    ``folded_axis_absorbs`` above its floor.
    """
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                xp=xp, courant=courant)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def seed_state(fields, grid, sub_step: str, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Physical-band or subnormal-band values in every array the sub-step touches.

    THE AUXILIARIES START NONZERO in both classes. A zero ``fu`` makes ``kms*prev``
    exactly zero on the first launch whatever ``kms`` holds, so a mis-indexed
    coefficient would only show from step two -- precisely what the multi-step leg
    exists to catch, and no reason to hide it from the single launch as well.
    """
    xp = grid.xp
    names = state_names(sub_step)
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
                for name in names}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(names, tuple(grid.shape), rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def snapshot(fields, sub_step: str) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in state_names(sub_step)}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, sub_step: str) -> None:
    """Move the curl operands between launches, identically on both paths.

    A real run's constitutive sub-step rewrites E (or H) before every curl. Held
    fixed, ``fu`` reaches a fixed point and 60 launches measure what one does.
    """
    for name in SUB_STEP_ARRAYS[sub_step]["sources"]:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# The boundary codes, and the substitution that is the experimental variable
# ---------------------------------------------------------------------------

def boundary_codes_for(grid, substitution: str) -> Tuple[Tuple[int, int, int],
                                                         Tuple[str, str, str]]:
    """Map ``stepping._boundary_kinds`` onto the kernels' three codes.

    ``mirror_resolved`` asks the SHIPPED resolver, ``coverage.real_curl_boundary
    _codes`` — the same function ``covers_real_pml_curl`` consults, so this arm
    measures the kernel under exactly the codes a dispatch would hand it, and a
    drift between the predicate's split and the harness's cannot hide here.

    The other two arms are the harness deliberately MISDECLARING the fold, and
    they are controls, not defaults: ``mirror_as_metallic`` reproduces the
    pre-2026-08-20 kernel's behaviour on a folded PERIODIC axis and
    ``mirror_as_periodic`` the wrap. Reporting either as though it were the
    shipped answer would report a property of the choice as a property of the
    kernel.
    """
    kinds = coverage.real_pml_boundary_kinds(grid)
    if substitution == LICENSED_SUBSTITUTION:
        codes, refusal = coverage.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise ValueError(f"the shipped resolver refuses this grid: {refusal}")
        return tuple(int(c) for c in codes), tuple(kinds)
    substitute = BC_METALLIC if substitution == "mirror_as_metallic" else BC_PERIODIC
    codes = []
    for kind in kinds:
        if kind == "mirror":
            codes.append(substitute)
        elif kind == "metallic":
            codes.append(BC_METALLIC)
        elif kind == "periodic":
            codes.append(BC_PERIODIC)
        else:
            raise ValueError(f"boundary {kind!r} has no code and no substitution")
    return tuple(codes), tuple(kinds)


# ---------------------------------------------------------------------------
# The mask delta: what the oracle drops that the kernel keeps, plane by plane
# ---------------------------------------------------------------------------

def oracle_masked_planes(target: str, grid) -> List[Tuple[int, int]]:
    """The planes ``_mask_non_owned_cells`` zeroes for this component.

    Transcribed from stepping.py:1912-1947. ``(axis, index)`` with index 0 for the
    near plane and -1 for the last stored slot.
    """
    iyee = IYEE_SHIFTS[target]
    planes: List[Tuple[int, int]] = []
    for axis in range(3):
        if iyee[axis] != 0:
            if stepping._stored_past_owned(grid, axis):
                planes.append((axis, -1))
            continue
        if grid.is_mirrored(axis) or grid.is_metallic(axis) or grid.is_axis(axis):
            planes.append((axis, 0))
    return planes


def kernel_masked_planes(target: str, codes: Tuple[int, int, int]) -> List[Tuple[int, int]]:
    """The planes the SHIPPED kernel zeroes, transcribed from its own two bodies.

    Cell 0 for every Yee-shift-0 axis coded METALLIC or MIRROR_PERIODIC
    (``if (bc == BC_METALLIC && index == 0)`` beside
    ``if (bc == BC_MIRROR_PERIODIC && index == 0)``), and the LAST stored slot for
    every Yee-shift-1 axis coded MIRROR_PERIODIC
    (``if (bc == BC_MIRROR_PERIODIC && index == n - 1)``). The second family is
    the branch added 2026-08-20; before it the kernel had no top-plane branch
    anywhere in the file, which is the divergence this gate specified.
    """
    iyee = IYEE_SHIFTS[target]
    planes: List[Tuple[int, int]] = []
    for axis in range(3):
        if iyee[axis] == 0:
            if codes[axis] in (BC_METALLIC, BC_MIRROR_PERIODIC):
                planes.append((axis, 0))
        elif codes[axis] == BC_MIRROR_PERIODIC:
            planes.append((axis, -1))
    return planes


def mask_delta(target: str, grid, codes: Tuple[int, int, int]) -> Dict[str, List[List[int]]]:
    oracle = oracle_masked_planes(target, grid)
    kernel = kernel_masked_planes(target, codes)
    return {
        "oracle_only": [list(p) for p in oracle if p not in kernel],
        "kernel_only": [list(p) for p in kernel if p not in oracle],
    }


def _plane_index(axis: int, index: int, shape) -> Tuple[Any, ...]:
    key: List[Any] = [slice(None)] * 3
    key[axis] = index if index >= 0 else shape[axis] - 1
    return tuple(key)


def plane_mask_array(planes: Sequence[Sequence[int]], shape) -> np.ndarray:
    """Boolean volume: True on every named plane."""
    out = np.zeros(shape, dtype=bool)
    for axis, index in planes:
        out[_plane_index(int(axis), int(index), shape)] = True
    return out


def localize(reference: np.ndarray, produced: np.ndarray, target: str, grid,
             codes: Tuple[int, int, int]) -> Dict[str, Any]:
    """WHERE the two answers differ, decomposed against the mask delta.

    This is what turns a divergence into a specification. A kernel that is wrong
    everywhere and a kernel that is wrong on exactly the planes the array path
    masks and it does not are the same verdict at the ``bit_identical`` level and
    completely different findings.
    """
    a = np.ascontiguousarray(reference, dtype=np.float32).view(np.uint32)
    b = np.ascontiguousarray(to_host(produced), dtype=np.float32).view(np.uint32)
    differing = a != b
    delta = mask_delta(target, grid, codes)
    named = plane_mask_array(delta["oracle_only"] + delta["kernel_only"], a.shape)
    return {
        "differing_words": int(np.count_nonzero(differing)),
        "differing_on_mask_delta_planes": int(np.count_nonzero(differing & named)),
        "differing_elsewhere": int(np.count_nonzero(differing & ~named)),
        "mask_delta": delta,
        "words_on_mask_delta_planes": int(np.count_nonzero(named)),
    }


def mask_delta_planes_are_live(frozen: Dict[str, np.ndarray],
                               reference: Dict[str, np.ndarray],
                               sub_step: str, grid,
                               codes: Tuple[int, int, int]) -> Dict[str, Any]:
    """Did the ORACLE move any word on the planes the mask delta names?

    A plane the oracle masks and the kernel does not is only distinguishable if the
    curl there would have been nonzero. Where it would not, the two paths agree for
    a reason about the fixture -- and a case reporting IDENTITY off that agreement
    is measuring nothing. This checks the weaker, sufficient thing: that the oracle
    changed words on those planes at all.
    """
    out: Dict[str, Any] = {"planes": [], "moved_on_delta_planes": 0,
                           "delta_plane_words": 0}
    for target in SUB_STEP_ARRAYS[sub_step]["targets"]:
        delta = mask_delta(target, grid, codes)
        planes = delta["oracle_only"] + delta["kernel_only"]
        if not planes:
            continue
        named = plane_mask_array(planes, tuple(grid.shape))
        before = np.ascontiguousarray(frozen[target], dtype=np.float32).view(np.uint32)
        after = np.ascontiguousarray(reference[target], dtype=np.float32).view(np.uint32)
        moved = int(np.count_nonzero((before != after) & named))
        out["planes"].append({"target": target, "planes": planes, "moved": moved,
                              "words": int(np.count_nonzero(named))})
        out["moved_on_delta_planes"] += moved
        out["delta_plane_words"] += int(np.count_nonzero(named))
    # No delta at all is NOT a floor failure: it is the arm where the substitution
    # reproduces the oracle's masking exactly, and that arm is precisely what this
    # gate is trying to certify.
    out["has_delta"] = out["delta_plane_words"] > 0
    out["meets_floor"] = (not out["has_delta"]) or out["moved_on_delta_planes"] > 0
    return out


# ---------------------------------------------------------------------------
# The coefficient tables, and the host mutations that corrupt them
# ---------------------------------------------------------------------------

def tables_for(sub_step: str, layer) -> Dict[str, Any]:
    """The six flattened kms/sinv views this sub-step reads."""
    suffix = "_h" if SUB_STEP_ARRAYS[sub_step]["half_integer"] else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def wrong_sub_lattice_tables(sub_step: str, layer) -> Dict[str, Any]:
    """The OTHER sub-step's sub-lattice: a half-cell error, not a crash."""
    suffix = "" if SUB_STEP_ARRAYS[sub_step]["half_integer"] else "_h"
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def reversed_folded_axis_tables(sub_step: str, layer, grid) -> Dict[str, Any]:
    """The folded axis's absorber profile read backwards -- IN BOUNDS.

    The constitutive family retired "the fold changes the stored extent" on
    2026-08-19 after this exact mutation was caught 10/10. It is carried here
    because a family that has NOT retired it must still show that its gate could
    see it: an arm certified without this leg would be certified against a defect
    nobody demonstrated was visible.
    """
    tables = dict(tables_for(sub_step, layer))
    xp = grid.xp
    for axis, name in enumerate("xyz"):
        if not grid.is_mirrored(axis):
            continue
        for stem in ("kms", "sinv"):
            key = f"{stem}_{name}"
            tables[key] = xp.ascontiguousarray(tables[key][::-1])
    return tables


def host_mutated_tables(name: str, sub_step: str, layer, grid) -> Dict[str, Any]:
    if name == "reverse_folded_axis_coefficients":
        return reversed_folded_axis_tables(sub_step, layer, grid)
    if name == "swap_curl_sublattice":
        return wrong_sub_lattice_tables(sub_step, layer)
    raise KeyError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

def run_kernel_cuda(sub_step: str, fields, tables: Dict[str, Any],
                    codes: Tuple[int, int, int], dtdx: float) -> None:
    """The SHIPPED kernel, through its own launch wrapper, with codes handed in."""
    from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415

    entry = (step_curl_kernels._step_B_fused_pml_real if sub_step == "step_B"
             else step_curl_kernels._step_D_fused_pml_real)
    entry(fields, tables, tuple(np.int32(c) for c in codes), dtdx)
    cp.cuda.runtime.deviceSynchronize()


def _shift_up_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    """``shift_up`` from ``_REAL_PML_PRELUDE``: neighbour one up, far-face rule.

    ONLY PERIODIC WRAPS. Both the metallic and the folded-periodic code take the
    zero face, which is the shipped ternary
    ``(bc == BC_PERIODIC) ? g[wrap] : 0.0f``.
    """
    shifted = np.roll(field, -1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane_index(axis, -1, field.shape)] = np.float32(0.0)
    return shifted


def _shift_dn_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    """``shift_dn`` from ``_REAL_PML_PRELUDE``: neighbour one down, near-face rule."""
    shifted = np.roll(field, 1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane_index(axis, 0, field.shape)] = np.float32(0.0)
    return shifted


def _broadcast(vector, axis: int) -> np.ndarray:
    shape = [1, 1, 1]
    shape[axis] = int(np.asarray(vector).size)
    return np.asarray(vector, dtype=np.float32).reshape(shape)


def run_kernel_numpy(sub_step: str, fields, tables: Dict[str, Any],
                     codes: Tuple[int, int, int], dtdx: float) -> None:
    """The device tree, transcribed, in float32 -- the laptop backend.

    THE GROUPING IS THE SHIPPED ONE and it is load-bearing::

        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc == BC_METALLIC && index == 0) curl = 0.0f;
        if (bc == BC_MIRROR_PERIODIC && index == 0) curl = 0.0f;        // shift 0
        if (bc == BC_MIRROR_PERIODIC && index == n - 1) curl = 0.0f;    // shift 1
        float fprev = fu[idx];
        float fu_new = ((fprev * kms) - curl) * sinv;
        fu[idx] = fu_new;
        f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;

    NOT ``dtdx * (sf - f1 + f2 - ss)``, which C associates left-to-right; float
    addition is not associative and the two differ in the last bits.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the NVRTC
    contraction guard exists for and it is not the shipped bytes. What it settles
    is whether the harness can localize a divergence and whether it can fail.
    """
    spec = SUB_STEP_ARRAYS[sub_step]
    shift = _shift_dn_numpy if spec["backward"] else _shift_up_numpy
    scale = np.float32(dtdx)
    names = "xyz"
    for (target, aux, first, first_axis, second, second_axis,
         dsig, dsigu) in CURL_TERMS[sub_step]:
        f = getattr(fields, target)
        fu = getattr(fields, aux)
        f1 = getattr(fields, first)
        f2 = getattr(fields, second)
        sf = shift(f1, first_axis, codes[first_axis])
        ss = shift(f2, second_axis, codes[second_axis])
        curl = (scale * ((sf - f1) + (f2 - ss))).astype(np.float32)
        iyee = IYEE_SHIFTS[target]
        for axis in range(3):
            if iyee[axis] == 0:
                if codes[axis] in (BC_METALLIC, BC_MIRROR_PERIODIC):
                    curl[_plane_index(axis, 0, curl.shape)] = np.float32(0.0)
            elif codes[axis] == BC_MIRROR_PERIODIC:
                curl[_plane_index(axis, -1, curl.shape)] = np.float32(0.0)
        kms = _broadcast(tables[f"kms_{names[dsig]}"], dsig)
        sinv = _broadcast(tables[f"sinv_{names[dsig]}"], dsig)
        kms_u = _broadcast(tables[f"kms_{names[dsigu]}"], dsigu)
        sinv_u = _broadcast(tables[f"sinv_{names[dsigu]}"], dsigu)
        fprev = fu.copy()
        fu_new = (((fprev * kms).astype(np.float32) - curl).astype(np.float32)
                  * sinv).astype(np.float32)
        fu[...] = fu_new
        a = ((f * kms_u).astype(np.float32) + fu_new).astype(np.float32)
        f[...] = ((a - fprev).astype(np.float32) * sinv_u).astype(np.float32)


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                 sub_step: str) -> float:
    """Fraction of output WORDS the array path changed from the frozen input.

    Compared as raw uint32, never with ``allclose``: ``-0.0 == 0.0`` and
    ``NaN != NaN`` both lie, and the subnormal-band class puts signed zeros in the
    operands deliberately.
    """
    moved = total = 0
    for name in outputs(sub_step):
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def folded_axis_absorbs(sub_step: str, layer, grid) -> Dict[str, Any]:
    """Does the FOLDED axis's coefficient profile differ from the identity?"""
    tables = tables_for(sub_step, layer)
    out: Dict[str, Any] = {"folded_axes": [], "max_deviation": 0.0}
    for axis, name in enumerate("xyz"):
        if not grid.is_mirrored(axis):
            continue
        deviation = 0.0
        for stem in ("kms", "sinv"):
            values = to_host(tables[f"{stem}_{name}"]).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        out["folded_axes"].append({"axis": axis, "name": name,
                                   "max_deviation_from_identity": deviation})
        out["max_deviation"] = max(out["max_deviation"], deviation)
    out["meets_floor"] = (not out["folded_axes"]) or out["max_deviation"] > 0.0
    return out


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "shape_full": [int(n) for n in grid.shape_full],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "stored_cells": [int(grid.stored_cells(a)) for a in range(3)],
        "owned_cells": [int(grid.owned_cells(a)) for a in range(3)],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
    }


def check_terms_against_stepping() -> Dict[str, Any]:
    """Pin :data:`CURL_TERMS` against ``stepping``'s own term tables.

    The table above is a transcription of the DEVICE source, and the NumPy leg
    consumes it. If it drifted from ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS``
    the laptop leg would compare a wrong stencil against the oracle and report a
    divergence that says nothing about the kernel -- so the correspondence is
    asserted rather than assumed.
    """
    out: Dict[str, Any] = {"agreed": True, "checked": [], "disagreements": []}
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        for mine, theirs in zip(CURL_TERMS[sub_step], source):
            # ``stepping`` spells the two PML cycle directions as letters and the
            # kernel indexes them as axes; the map is the only translation, and it
            # is written here rather than in the table so the table stays a
            # transcription of the DEVICE source.
            axis_of = {"x": 0, "y": 1, "z": 2}
            record = {
                "sub_step": sub_step, "target": mine[0],
                "mine": [mine[0], mine[2], mine[3], mine[4], mine[5], mine[6], mine[7]],
                "stepping": [theirs.target, theirs.first, theirs.first_axis,
                             theirs.second, theirs.second_axis,
                             axis_of[theirs.dsig], axis_of[theirs.dsigu]],
            }
            record["agrees"] = record["mine"] == record["stepping"]
            out["checked"].append(record)
            if not record["agrees"]:
                out["agreed"] = False
                out["disagreements"].append(record)
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], sub_step: str, courant: float,
             value_class: str, guard: str, substitution: str,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML`` objects.
    There is no second transcription on the oracle leg to drift: the kernel is
    compared against the thing it claims to reproduce, byte for byte, from ONE
    frozen state -- oracle runs, state is restored, kernel runs.
    """
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{sub_step}|{courant}|{value_class}".encode()).digest()[:4], "big"))

    fields, layer, grid = build(xp, spec, courant)
    host = seed_state(fields, grid, sub_step, value_class, rng)
    dtdx = float(grid.dt / grid.dx)
    codes, kinds = boundary_codes_for(grid, substitution)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend,
        "substitution": substitution, "host_mutation": host_mutation,
        "fold_axes": spec["axes"], "mirror_phase": spec["phase"],
        "boundaries": list(spec["boundaries"]),
        "boundary_codes": [int(c) for c in codes],
        "resolved_kinds": list(kinds),
        "dtdx": dtdx,
        "structure": structure_facts(grid),
        "operand_census": operand_census(host),
    }

    # THE PREDICATE'S CURRENT ANSWER, recorded rather than acted on. This gate
    # exists to decide whether that answer should change, so it must not be the
    # thing that gates the measurement.
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    absorbs = folded_axis_absorbs(sub_step, layer, grid)
    case["folded_axis_absorbs"] = absorbs
    if not absorbs["meets_floor"]:
        case["skipped"] = ("the folded axis's coefficient profile is the identity "
                           "everywhere; this case cannot distinguish a coefficient "
                           "index error on the axis the fold moved")
        case["seconds"] = time.time() - started
        return case

    frozen = snapshot(fields, sub_step)

    # Leg 1: the oracle.
    if sub_step == "step_B":
        stepping.step_B(fields, layer)
    else:
        stepping.step_D(fields, layer)
    reference = snapshot(fields, sub_step)

    moved = oracle_moved(frozen, reference, sub_step)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    live = mask_delta_planes_are_live(frozen, reference, sub_step, grid, codes)
    case["mask_delta_planes_are_live"] = live
    if not live["meets_floor"]:
        case["skipped"] = ("the planes the mask delta names carry no moved word; "
                           "a mask nothing can distinguish is a mask this case "
                           "cannot score")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    if host_mutation is None:
        tables = tables_for(sub_step, layer)
    else:
        tables = host_mutated_tables(host_mutation, sub_step, layer, grid)
    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(sub_step, fields, tables, codes, dtdx)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(sub_step)}
    case["single_launch"] = combine(parts)

    # WHERE it differs, decomposed against the mask delta. This is the half of the
    # record that turns a refusal into a specification.
    case["localization"] = {
        name: localize(reference[name],
                       getattr(fields, name),
                       name[3:] if name.startswith("fu_") else name,
                       grid, codes)
        for name in outputs(sub_step)
    }
    case["localization_summary"] = {
        "differing_words": sum(v["differing_words"]
                               for v in case["localization"].values()),
        "differing_on_mask_delta_planes": sum(
            v["differing_on_mask_delta_planes"] for v in case["localization"].values()),
        "differing_elsewhere": sum(v["differing_elsewhere"]
                                   for v in case["localization"].values()),
    }

    # Leg 3: the multi-step. ``fu`` IS STATE, and a tree that gets the field right
    # and the auxiliary wrong is correct for exactly one launch and wrong forever
    # after. ONE ``Fields`` object, run twice from the SAME frozen state.
    if host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            if sub_step == "step_B":
                stepping.step_B(fields, layer)
            else:
                stepping.step_D(fields, layer)
            advance_sources(fields, sub_step)
        oracle_multi = snapshot(fields, sub_step)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(sub_step, fields, tables, codes, dtdx)
            advance_sources(fields, sub_step)
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs(sub_step)}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str) -> List[Tuple[Dict[str, Any], str, float, str, str]]:
    specs = FOLD_SPECS if product == "full" else FOLD_SPECS[:6]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    out = []
    for spec in specs:
        for sub_step in SUB_STEPS:
            for courant in courants:
                for value_class in classes:
                    for substitution in SUBSTITUTIONS:
                        out.append((spec, sub_step, courant, value_class, substitution))
    return out


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              product: str, guard: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, (spec, sub_step, courant, value_class, substitution) in enumerate(
            plan, start=1):
        case = one_case(backend, spec, sub_step, courant, value_class, guard,
                        substitution)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"{substitution} SKIPPED: {case['skipped'][:60]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        summary = case["localization_summary"]
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"c={courant} {value_class} {substitution} "
            f"shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"diff={summary['differing_words']} "
            f"on_delta={summary['differing_on_mask_delta_planes']} "
            f"elsewhere={summary['differing_elsewhere']} "
            f"({case['seconds']:.1f} s)")
    return cases


#: The specs a mutation leg is scored on. CHOSEN, not sliced off the front of the
#: case product: a leg taken from the head landed entirely on unfolded controls in
#: the sibling gate, so the fold-only mutation was skipped on every leg and scored
#: 0/0 UNCAUGHT -- a harness defect that reads exactly like a kernel defect.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "unfolded_metallic",     # the control: a table defect must bite here too
    "fold_Y_metallic",       # the halved count, no extra slot
    "fold_Y_periodic",       # a stored slot past the owned window
    "fold_X_metallic",       # the slowest-stride coefficient index
    "fold_Z_metallic",       # the fastest-stride one
    "fold_XY_metallic",      # two planes at once
    # ADDED 2026-08-20 with the top-plane branch. Before it, every folded-PERIODIC
    # leg was excluded by ``scorable_baselines`` -- its unmutated baseline diverged
    # -- so the arm carrying the new masks had ONE spec in the plan and would have
    # scored them on a single stride. These give the two new mutations a slowest-
    # stride axis, a fastest-stride axis and two top planes at once.
    "fold_Z_periodic",       # the fastest-stride axis, folded PERIODIC
    "fold_XY_periodic",      # two top planes masked at once
    # ADDED 2026-08-20 with the third fold plane. A three-plane case that were
    # swept but never MUTATED would raise the cap on the strength of an unarmed
    # arm: every mutation this battery carries must be shown to bite on the shape
    # the cap is being raised to, or "identical at three planes" is consistent
    # with a comparator that stopped looking there.
    "fold_XYZ_metallic",     # three planes, the corpus row's own arm
    "fold_XYZ_periodic",     # three top planes masked at once
)

#: The mutation battery runs under ONE substitution, and since 2026-08-20 it is the
#: SHIPPED resolution rather than a substitution at all. A mutation leg answers
#: "can this gate see this defect", and it can only answer that on an arm where the
#: UNMUTATED case is identical -- on an arm that already diverges, every leg scores
#: CAUGHT for free and the battery measures nothing. Before the top-plane branch
#: existed no such arm covered a folded PERIODIC axis, which is precisely why the
#: two masks it adds could not have been scored under the old choice.
MUTATION_SUBSTITUTION = LICENSED_SUBSTITUTION

#: Memo for :func:`spec_boundary_codes`; a spec's codes are a property of the spec.
_SPEC_CODES: Dict[str, Tuple[int, int, int]] = {}


def spec_boundary_codes(spec: Dict[str, Any]) -> Tuple[int, int, int]:
    """The codes the SHIPPED resolver hands this spec's grid.

    Built on NumPy and ASKED of ``coverage.real_curl_boundary_codes`` rather than
    re-derived from the spec's own strings: a second transcription of the fold
    split is a second thing to drift, and this is what decides which legs a mask
    mutation is scored on. A spec the resolver refuses raises here -- loudly,
    because it would otherwise silently drop every leg on that spec.
    """
    cached = _SPEC_CODES.get(spec["label"])
    if cached is None:
        _, _, grid = build(_NumpyWearingCupysName(), spec, COURANTS[0])
        codes, refusal = coverage.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise ValueError(
                f"the shipped resolver refuses fixture {spec['label']!r}: {refusal}")
        cached = tuple(int(code) for code in codes)
        _SPEC_CODES[spec["label"]] = cached
    return cached


def scorable_baselines(results: Dict[str, Any]) -> Tuple[set, List[Dict[str, Any]]]:
    """The ``(label, sub_step)`` pairs whose UNMUTATED case was bit-identical.

    A MUTATION LEG ONLY MEANS SOMETHING WHERE THE BASELINE IS IDENTICAL. On an arm
    that already diverges, every leg -- including a leg that must be UNCAUGHT --
    scores CAUGHT for free, because the divergence it is credited with was already
    there before the mutation was applied.

    That is not hypothetical. The 2026-08-19 first run scored
    ``commute_dtdx_scale`` and ``reload_fu_from_memory`` -- both declared MUST BE
    UNCAUGHT by ``probe.PML_NULL_MUTATIONS`` -- at "caught 1/5, NULL VIOLATED", and
    the 1 was in every instance the single folded-PERIODIC leg in the plan, whose
    unmutated baseline diverges on the top plane by construction. The kernel had
    not acquired a defect; the plan had asked a question the arm cannot answer.

    Read off the primary sweep rather than re-run: the sweep already contains the
    unmutated case at exactly the mutation legs' courant, value class and
    substitution, so this costs nothing and cannot disagree with itself.
    """
    baseline: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for case in results.get("sweep", {}).get("fmad_false", []):
        if (case.get("courant") != INEXACT_COURANT
                or case.get("value_class") != "uniform"
                or case.get("substitution") != MUTATION_SUBSTITUTION):
            continue
        baseline[(case["label"], case["sub_step"])] = case
    scorable, excluded = set(), []
    for key, case in baseline.items():
        if case.get("skipped"):
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": f"baseline skipped: {case['skipped'][:80]}"})
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical", True)
        if single and multi:
            scorable.add(key)
        else:
            excluded.append({
                "label": key[0], "sub_step": key[1],
                "why": ("the unmutated baseline already diverges here, so every "
                        "leg would score CAUGHT for free"),
                "baseline_differing_words": case["localization_summary"]["differing_words"],
                "baseline_differing_elsewhere":
                    case["localization_summary"]["differing_elsewhere"]})
    return scorable, excluded


def mutation_plan(product: str, scorable: Optional[set] = None
                  ) -> List[Tuple[Dict[str, Any], str, float, str]]:
    by_label = {spec["label"]: spec for spec in FOLD_SPECS}
    labels = MUTATION_SPEC_LABELS if product == "full" else MUTATION_SPEC_LABELS[:3]
    plan = [(by_label[label], sub_step, INEXACT_COURANT, "uniform")
            for label in labels for sub_step in SUB_STEPS]
    if scorable is None:
        return plan
    return [entry for entry in plan if (entry[0]["label"], entry[1]) in scorable]


def leg_caught(case: Dict[str, Any]) -> bool:
    """Did this mutated leg diverge from the array path at EITHER granularity?"""
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def _score(legs: List[Dict[str, Any]], must_be_caught: bool) -> Dict[str, Any]:
    scored = [c for c in legs if not c.get("skipped")]
    caught = sum(1 for c in scored if leg_caught(c))
    # NO LEGS is its own verdict, never "UNCAUGHT". A mutation that was never run
    # has not been shown inert; it has not been asked.
    if not scored:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = ("CAUGHT" if caught == len(scored)
                   else "PARTIAL" if caught else "UNCAUGHT")
    else:
        verdict = "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"
    return {"ran": len(scored), "caught": caught, "uncaught": len(scored) - caught,
            "must_be_caught": must_be_caught, "verdict": verdict}


def run_host_mutations(results: Dict[str, Any], out_path: str, backend: str,
                       product: str, scorable: set) -> Dict[str, Any]:
    """Every table-level defect. A gate that cannot fail certifies nothing."""
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)
    for name in HOST_MUTATIONS:
        legs: List[Dict[str, Any]] = []
        for spec, sub_step, courant, value_class in plan:
            if name in FOLD_ONLY_HOST_MUTATIONS and not spec["axes"]:
                continue
            legs.append(one_case(backend, spec, sub_step, courant, value_class,
                                 "fmad_false", MUTATION_SUBSTITUTION,
                                 host_mutation=name))
        out[name] = dict(_score(legs, True), cases=legs)
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results: Dict[str, Any], out_path: str,
                         product: str, scorable: set) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED strings and recompiled.

    The compile memo is keyed through the SOURCE, so a mutated body is a miss and
    reaches NVRTC. Every leg records how many constructions came from the mutated
    bytes, because a leg reporting a pass for a mutation it never applied is worse
    than no leg -- three measured instances on the sibling track.
    """
    from meep_gpu.cuda_kernels import compile_cache, step_curl_kernels  # noqa: PLC0415

    attribute_for = {"step_B": "_step_B_pml_real_kernel_code",
                     "step_D": "_step_D_pml_real_kernel_code"}
    originals = {name: getattr(step_curl_kernels, attribute)
                 for name, attribute in attribute_for.items()}
    out: Dict[str, Any] = {}
    # FOLDED LEGS ONLY. Every one of these defects is already scored on unfolded
    # grids by the certified record; what this gate adds is whether the same
    # battery still bites once the stored extent is halved.
    plan = [entry for entry in mutation_plan(product, scorable) if entry[0]["axes"]]

    try:
        for sub_step in SUB_STEPS:
            attribute = attribute_for[sub_step]
            for name in SOURCE_MUTATIONS:
                transform = probe.SOURCE_MUTATIONS[name]
                mutated, sites = transform(originals[sub_step])
                key = f"{sub_step}:{name}"
                # Legs whose grid carries the code this mutation's lines test on.
                required = SOURCE_MUTATION_REQUIRES_CODE.get(name)
                leg_plan = [entry for entry in plan
                            if required is None
                            or required in spec_boundary_codes(entry[0])]
                if sites == 0 or mutated == originals[sub_step]:
                    out[key] = {"armed": False,
                                "why": (f"matched {sites} site(s) and changed "
                                        f"nothing; the mutation and the kernel "
                                        f"have drifted apart")}
                    log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                setattr(step_curl_kernels, attribute, mutated)
                step_curl_kernels._clear_kernel_cache()
                compile_cache.clear_compile_log()
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                legs = []
                for spec, leg_sub_step, courant, value_class in leg_plan:
                    if leg_sub_step != sub_step:
                        continue
                    legs.append(one_case("cuda", spec, sub_step, courant,
                                         value_class, "fmad_false",
                                         MUTATION_SUBSTITUTION))
                setattr(step_curl_kernels, attribute, originals[sub_step])
                step_curl_kernels._clear_kernel_cache()
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] == digest)
                scored = _score(legs, name not in NULL_SOURCE_MUTATIONS)
                out[key] = dict(scored, armed=True, sites=sites,
                                requires_code=required,
                                legs_in_plan=len([e for e in plan
                                                  if e[1] == sub_step]),
                                legs_carrying_the_code=len(
                                    [e for e in leg_plan if e[1] == sub_step]),
                                mutated_source_sha256=digest,
                                kernel_constructions_from_mutated_bytes=from_mutated,
                                caught_at_single_launch=sum(
                                    1 for c in legs if not c.get("skipped")
                                    and not c["single_launch"]["bit_identical"]),
                                cases=legs)
                if from_mutated == 0 and scored["ran"]:
                    # A leg reporting a pass for a mutation it never applied is
                    # worse than no leg: it measured the SHIPPED kernel.
                    out[key]["verdict"] = "UNACCOUNTED"
                    out[key]["why"] = ("no kernel construction used the mutated "
                                       "bytes; this leg did not exercise the "
                                       "mutation")
                log(f"[src-mut] {key}: caught {scored['caught']}/{scored['ran']} "
                    f"builds_from_mutated={from_mutated} -> {out[key]['verdict']}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for sub_step, attribute in attribute_for.items():
            setattr(step_curl_kernels, attribute, originals[sub_step])
        step_curl_kernels._clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def _arm_of(case: Dict[str, Any]) -> str:
    """Which arm a folded case belongs to: are ALL its folded axes metallic?

    The reading predicts the split falls here and nowhere else, so the summary is
    partitioned on it rather than on the label -- a label is a name and this is a
    property of the grid the case actually built.
    """
    structure = case["structure"]
    folded = [a for a in range(3) if structure["mirrored"][a]]
    if not folded:
        return "unfolded"
    if all(structure["metallic"][a] for a in folded):
        return "folded_all_metallic"
    if all(not structure["metallic"][a] for a in folded):
        return "folded_all_periodic"
    return "folded_mixed"


def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    """The finding, partitioned by arm, with every clause it rests on named."""
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]

    arms: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        key = f"{_arm_of(case)}::{case['substitution']}"
        entry = arms.setdefault(key, {
            "cases": 0, "single_identical": 0, "multi_cases": 0,
            "multi_identical": 0, "differing_words": 0,
            "differing_on_mask_delta_planes": 0, "differing_elsewhere": 0,
            "sub_steps": set(), "labels": set()})
        entry["cases"] += 1
        entry["sub_steps"].add(case["sub_step"])
        entry["labels"].add(case["label"])
        if case["single_launch"]["bit_identical"]:
            entry["single_identical"] += 1
        if "multi_step" in case:
            entry["multi_cases"] += 1
            if case["multi_step"]["bit_identical"]:
                entry["multi_identical"] += 1
        for field in ("differing_words", "differing_on_mask_delta_planes",
                      "differing_elsewhere"):
            entry[field] += case["localization_summary"][field]
    for entry in arms.values():
        entry["sub_steps"] = sorted(entry["sub_steps"])
        entry["labels"] = sorted(entry["labels"])
        entry["all_identical"] = entry["single_identical"] == entry["cases"] and \
            entry["multi_identical"] == entry["multi_cases"]
        entry["divergence_confined_to_mask_delta"] = (
            entry["differing_elsewhere"] == 0 and entry["differing_words"] > 0)

    # THE ARMS THIS GATE LICENSES: every arm run under the SHIPPED resolution,
    # which is the only code triple a dispatch can produce. Each is released only
    # if it is identical at both granularities AND the battery that would have
    # caught a defect there actually bit. The two misdeclaration substitutions are
    # controls and are reported below, never gated on: their folded arms diverge on
    # purpose.
    #
    # EVERY ARM THE SWEEP CONTAINS IS REQUIRED, not merely allowed, and the two
    # TERMINATIONS are required to be in it at all. A run that swept no folded
    # PERIODIC case would release on the metallic arm alone and say nothing about
    # the branch this round added, which is exactly the shape of a green gate that
    # measured the wrong half. Derived from what ran rather than hard-coded, so the
    # reduced product is judged on the arms it actually carries.
    reasons: List[str] = []
    present = {_arm_of(case) for case in scored}
    for arm in ("folded_all_metallic", "folded_all_periodic"):
        if arm not in present:
            reasons.append(
                f"the sweep contained no {arm} case; the fold's two terminations "
                f"are the split this gate exists to measure and one of them is "
                f"missing")
    required_arms = tuple(arm for arm in ("unfolded", "folded_all_metallic",
                                          "folded_all_periodic", "folded_mixed")
                          if arm in present)
    licensed_keys = [f"{arm}::{LICENSED_SUBSTITUTION}" for arm in required_arms]
    for licensed_key in licensed_keys:
        licensed = arms.get(licensed_key)
        if licensed is None:
            reasons.append(f"no case was scored in the arm {licensed_key}")
            continue
        if licensed["single_identical"] != licensed["cases"]:
            reasons.append(
                f"{licensed_key}: single-launch divergence on "
                f"{licensed['cases'] - licensed['single_identical']} of "
                f"{licensed['cases']} cases")
        if licensed["multi_identical"] != licensed["multi_cases"]:
            reasons.append(
                f"{licensed_key}: multi-step divergence on "
                f"{licensed['multi_cases'] - licensed['multi_identical']} of "
                f"{licensed['multi_cases']} cases")
        if sorted(licensed["sub_steps"]) != sorted(SUB_STEPS):
            reasons.append(f"{licensed_key}: only sub-steps "
                           f"{licensed['sub_steps']} were scored")

    # THE PLANE COUNT IS MEASURED FROM WHAT RAN, never quoted from the case table.
    # ``CURL_FOLD_ADMISSION["folded_planes_swept"]`` may be raised to this number
    # and to no other: a cap taken from the spec list would be a claim about a
    # tuple, and a spec that skipped every one of its cases at a floor contributes
    # nothing to a cap.
    plane_counts = sorted({sum(1 for a in range(3) if c["structure"]["mirrored"][a])
                           for c in scored})
    max_planes = max(plane_counts) if plane_counts else 0
    planes_by_arm = {}
    for case in scored:
        count = sum(1 for a in range(3) if case["structure"]["mirrored"][a])
        planes_by_arm.setdefault(_arm_of(case), set()).add(count)
    planes_by_arm = {arm: sorted(counts) for arm, counts in planes_by_arm.items()}
    # A cap of N rests on N-plane cases in BOTH folded terminations. The
    # all-metallic arm is the one the reading says needs no device code; the
    # all-periodic one is the arm that fires N top-plane masks at once, and it is
    # the only place an interaction between them could appear. Raising the cap on
    # the metallic arm alone would be the same over-claim, one plane further out.
    for arm in ("folded_all_metallic", "folded_all_periodic"):
        if arm in planes_by_arm and max(planes_by_arm[arm]) != max_planes:
            reasons.append(
                f"{arm} was scored at up to {max(planes_by_arm[arm])} simultaneous "
                f"fold planes but the sweep reached {max_planes}; a plane cap must "
                f"rest on both terminations at the count it names")

    terms = results.get("curl_terms_vs_stepping", {})
    if not terms.get("agreed", False):
        reasons.append("the transcribed curl term table disagrees with stepping's")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is not inert")
        elif leg["verdict"] != "CAUGHT":
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']}")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT
               and c["single_launch"]["bit_identical"] is not None]
    # The guard control is only meaningful where the GUARDED leg was identical:
    # on an arm that already diverges, the unguarded leg diverging says nothing.
    guarded_identical = {
        (c["label"], c["sub_step"], c["courant"], c["value_class"], c["substitution"])
        for c in scored if c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["courant"], c["value_class"],
                      c["substitution"]) in guarded_identical]
    control_diverged = [c for c in comparable if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored where "
                    "the guarded leg was identical" if not comparable else
                    "the contraction guard is load-bearing on this sub-step"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here"),
    }

    # THE MISDECLARATION CONTROL. The folded-PERIODIC arm under
    # ``mirror_as_metallic`` IS the pre-2026-08-20 kernel, and it must STILL
    # diverge, on the mask-delta plane and nowhere else. A run where it too went
    # identical would mean the identity in the licensed arm came from something
    # other than the new branch — a quiet fixture change, a mask that never fires —
    # and the gate would be reporting a fix it had not isolated.
    control_key = f"folded_all_periodic::mirror_as_metallic"
    control_arm = arms.get(control_key)
    misdeclaration_control = {
        "arm": control_key,
        "cases": (control_arm or {}).get("cases", 0),
        "single_identical": (control_arm or {}).get("single_identical", 0),
        "differing_words": (control_arm or {}).get("differing_words", 0),
        "differing_elsewhere": (control_arm or {}).get("differing_elsewhere", 0),
        "confined_to_mask_delta":
            (control_arm or {}).get("divergence_confined_to_mask_delta", False),
        "reading": ("the pre-2026-08-20 kernel, reproduced: a folded PERIODIC axis "
                    "handed BC_METALLIC still diverges on the top plane and nowhere "
                    "else, so the licensed arm's identity is the new branch's"),
    }
    if control_arm is None:
        reasons.append(f"the misdeclaration control {control_key} was never scored; "
                       f"without it the fix is not isolated from the fixture")
    elif control_arm["single_identical"] == control_arm["cases"]:
        reasons.append(
            f"{control_key}: the misdeclaration control went IDENTICAL on all "
            f"{control_arm['cases']} cases. A folded PERIODIC axis handed the "
            f"metallic code must still diverge; that it does not means the "
            f"top-plane mask is not what this run measured")
    elif not control_arm["divergence_confined_to_mask_delta"]:
        reasons.append(
            f"{control_key}: {control_arm['differing_elsewhere']} differing words "
            f"lie OFF the mask-delta planes, so the misdeclaration is not the only "
            f"thing separating the two paths on this arm")

    return {
        "released": not reasons,
        "reasons": reasons,
        "licensed_arms": licensed_keys,
        "licensed_substitution": LICENSED_SUBSTITUTION,
        "misdeclaration_control": misdeclaration_control,
        "arms": arms,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        # THE CAP, MEASURED. This is the only number
        # ``CURL_FOLD_ADMISSION["folded_planes_swept"]`` may be set from.
        "folded_planes_scored": plane_counts,
        "max_folded_planes_scored": max_planes,
        "folded_planes_by_arm": planes_by_arm,
        "claim": ("under the SHIPPED resolution (coverage.real_curl_boundary_codes: "
                  "folded METALLIC -> BC_METALLIC, folded PERIODIC -> "
                  "BC_MIRROR_PERIODIC), the CUDA real-PML curl pair is byte-identical "
                  "to stepping.step_B / step_D per sub-step on a mirror fold at both "
                  "terminations, at one launch and at 60, over folded X/Y/Z, both "
                  "plane parities, odd and even full counts, and up to "
                  f"{max_planes} simultaneous fold planes"),
        "does_not_claim": [
            "nothing dispatches these kernels; this gate licenses a predicate "
            "clause, not a wiring",
            f"more than {max_planes} simultaneous mirror planes were not swept; "
            f"the predicate refuses them by name (coverage.CURL_FOLD_ADMISSION). "
            f"Three is every plane a 3-D grid has, so at max_planes = 3 that "
            f"clause is unreachable rather than merely unmet",
            "cylindrical (Dcyl) is untouched by this gate",
            "a fold combined with a conductivity, BFAST, special_kz, a Bloch "
            "phase or complex storage is refused by other clauses and untested "
            "here",
            "no throughput claim: this is a correctness gate and times nothing",
        ],
    }


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_folded_curl",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("with the folded-PERIODIC top-plane mask now in the shipped "
                     "kernels, is the CUDA real-PML curl pair byte-identical to the "
                     "array path on a fold at BOTH terminations under the shipped "
                     "code resolution -- and does the pre-branch behaviour still "
                     "diverge, on that plane and nowhere else, when the same fold is "
                     "deliberately misdeclared as metallic?"),
        "curl_terms_vs_stepping": check_terms_against_stepping(),
        "source_mutations_with_no_curl_site": list(SOURCE_MUTATIONS_WITH_NO_CURL_SITE),
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        # THE OBSERVER GOES IN BEFORE THE POLICY: under 'keep' the policy's strip
        # wraps this, so it records the option tuple NVRTC was really given.
        observer = probe.install_nvrtc_binary_observer()
        results["nvrtc_observer"] = observer
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": ("NumPy backend: compiles nothing, "
                                           "certifies nothing")}
    save(results, args.out)

    for guard, options, _is_primary in GUARD_SETS:
        if args.backend == "numpy" and guard != "fmad_false":
            continue  # there is no compiler on this leg to guard
        if args.backend == "cuda":
            from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
            step_curl_kernels._COMPILE_OPTIONS = tuple(options)
            step_curl_kernels._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        step_curl_kernels._COMPILE_OPTIONS = ("--fmad=false",)
        step_curl_kernels._clear_kernel_cache()

    if not args.skip_mutations:
        # THE BATTERY IS SCOPED TO THE ARM WHOSE BASELINE IS IDENTICAL, off the
        # sweep that just ran. A leg on a diverging arm scores CAUGHT for free.
        scorable, excluded = scorable_baselines(results)
        results["mutation_scope"] = {
            "substitution": MUTATION_SUBSTITUTION,
            "scorable": sorted("::".join(k) for k in scorable),
            "excluded": excluded,
            "why": ("a mutation leg answers 'can this gate see this defect', and "
                    "only an arm whose unmutated baseline is bit-identical can "
                    "answer it"),
        }
        log(f"[mut-scope] {len(scorable)} scorable (label, sub_step) pairs; "
            f"{len(excluded)} excluded because the baseline already diverges")
        save(results, args.out)
        run_host_mutations(results, args.out, args.backend, args.product, scorable)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.product, scorable)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']}")
    for name, arm in sorted(verdict["arms"].items()):
        log(f"[verdict]   {name}: single {arm['single_identical']}/{arm['cases']} "
            f"multi {arm['multi_identical']}/{arm['multi_cases']} "
            f"diff={arm['differing_words']} "
            f"on_delta={arm['differing_on_mask_delta_planes']} "
            f"elsewhere={arm['differing_elsewhere']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
