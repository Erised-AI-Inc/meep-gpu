"""Phase B — the mirror FOLD under complex64 storage and under ``grid.beta``.

DISPATCH IS WIRED AND ON BY DEFAULT for a ``prefer_gpu=True`` driver; the
PLANNER is wired outright. ``fastpath.plan_fast_path`` is consulted by the
driver, and what bounds dispatch is the ENABLE (``fastpath.DISPATCH_BY_DEFAULT``
is True; ``MEEP_GPU_DISPATCH=0`` turns it off for a run) together with each
arm's release entry, not the absence of a caller. Earlier text here said
dispatch was off by default; that described the package before release.
``launch.plan_step`` consults this file through SIX arms — ``folded complex PML``,
``folded real beta PML``, ``folded complex beta PML``, ``folded complex fill``,
``folded complex`` / ``folded beta run`` on the constitutive sides, and the
off-diagonal pair added by the residual-group wiring (``folded complex
off-diagonal PML`` and ``folded complex off-diagonal``) — each behind a gate on
the fold — and the package ``__init__`` exports its entry points.
``fingerprints.json`` still carries no entry for this file: the byte gate binds
its own provenance record inside its results directory instead. Callers are the
gate (``parity/meep_gpu/gate_triton_folded_complex.py``), the composition probe
(``probe_triton_folded_complex_composition.py``), the planner and the laptop
tests.

WHAT THIS IS FOR. Three lifted corpus rows demand it and only three —
``eigsrc_0_complex`` (2-D 14x14 res 30, Mirror(Y) +1, beta = 0.2, complex
storage, a LIVE PML on the folded axis), and the two binary-grating special_kz
rows ``special_kz_0_13_2`` / ``special_kz_1_17_7`` (Mirror(Y) +1 with an
IN-PLANE Bloch kx on the PML'd X axis and beta = -0.685 / -0.912). All three sit
in ONE cell of the case matrix (MIRROR_PERIODIC, EVEN full count, phase +1), so
they are anchors, not coverage: the odd-count arm, the metallic arm and the
phase = -1 arm exist in no corpus row and are synthesised by the gate.

FOUR KERNELS. K1/K3b step complex64 as float32 WORD PAIRS (complex64 is
bit-layout (re, im) interleaved, so a C-contiguous complex volume is a
C-contiguous float32 volume with a doubled last axis and complex cell ``w`` is
words ``2*w`` and ``2*w + 1``); K3a steps real float32.

* :func:`folded_bloch_pml_curl_step` (K1) — ``complex_fields.bloch_pml_curl_step``
  with ``symmetry.py``'s THREE fold deltas and no arithmetic change at all: the
  chunk record's forms per stored part, MEEP's curl stencil with the record's
  presence, and the Bloch connection of an UNFOLDED phased axis on the unshifted
  operand's plane 0;
* :func:`folded_mirror_ghost_fill_complex` (K2) — ``stepping.fill_symmetry_bc_*``
  plus ``stepping.fill_folded_far_ghosts_*`` for ONE family on ONE axis, the parity
  multiply per stored part (see below);
* :func:`folded_beta_pml_curl_step` (K3a) — ``special_kz.beta_pml_curl_step``
  with K1's three deltas, REAL storage;
* :func:`folded_beta_bloch_pml_curl_step` (K3b) — K1 plus ``special_kz``'s beta
  pass in the identical slot, COMPLEX storage.

Both beta bodies run the array path's beta pass: after every component's curl form,
the increment masked by the curl's own ownership clauses (both arms under a fold) and
subtracted through a ladder of its own (``stepping._apply_folded_increment``,
:func:`.forms.fold_ladder`), on the chunk record's forms.

WHY THE FOLDED CURL NEEDS NO PARITY ARITHMETIC UNDER COMPLEX STORAGE.
``symmetry.py``'s argument is an OWNERSHIP argument, not a storage argument, and
it transfers verbatim: ``_shift_down``'s near ghost ``parity * field[2]``
(stepping.py:1871-1872) lands at stored cell 0, whose target has Yee shift 0
there, which the cell-0 arm of ``_mask_non_owned_cells`` (S:1898-1902,
``is_mirrored or is_metallic or is_axis``) zeroes; ``_shift_up``'s far ghost
``parity * field[reflect_row]`` (S:1778-1779) lands at the last stored slot,
whose target has Yee shift 1 there, which the top-plane arm (S:1887-1897, gated
on ``_stored_past_owned``) zeroes ON A FOLDED PERIODIC AXIS ONLY. On a folded
METALLIC axis that top plane IS stepped and the ghost must be exactly zero —
under complex storage that means the ``(+0.0, +0.0)`` word pair which ``other=0.0``
already delivers on BOTH word loads (complex_fields.py:364-365 states the same
for the plain metallic case). Verified per target: on a Y fold, step_D's
y-shifted consumers are Dx and Dz (both Yee shift 0 on y, both cell-0 masked) and
step_B's are Bx and Bz (both shift 1 on y, both top-plane masked on
MIRROR_PERIODIC). That is an ARGUMENT plus a hand check, where ``symmetry.py``
earned the same claim for real storage by MEASUREMENT before it was written —
so the gate's synthetic sub-step sweep is a precondition here, not a formality.

THE GHOST FILL'S PARITY MULTIPLY IS PER PART. ``stepping._write_mirror_ghost``
and ``_fill_folded_far_ghosts`` write ``parity * plane`` through
``host_writes.copy_face``, which forms it with ``part_products.scale``: the Python
int rounded to the part dtype and each stored part multiplied by it, as MEEP's
connection multiplies each ``realnum`` part (a ``-1`` connection is a unary minus
per part, boundaries.cpp:359-366). ``x * (+1.0)`` and ``x * (-1.0)`` are exact on
every word, signed zeros and subnormals included, so K2 is the real fill per part,
with the coefficient a host-rounded float32 word passed per pass
(:func:`mirror_parity_coefficients`). The previous whole-complex product (the
coefficient promoted to ``(+-1, +0)``, with its zero cross terms) diverges from it
in 8/128 engineered words at BOTH parities; that product is the gate's inverted
headline mutation.

THE X, Y, Z FILL ORDER AND THE TWO PASSES COMMUTE BITWISE. Every fill is a
multiply of each part by exactly +-1, so the doubly-unowned corner sees
``c_y * (c_x * z)`` and ``c_x * (c_y * z)`` as the same word, and a fused per-axis
launch stores what the two passes store: ``symmetry.py``'s real-storage commute
null transfers to complex storage part by part. The passes are still launched in
the array path's order (near pass, then the far pass after the driver's wall
clear, X, Y, Z within each), and the gate records the commute as two structural
nulls (``reverse_axis_order``, ``fuse_fill_passes``) rather than assuming it.

WHY THE BETA MASK IS LOAD-BEARING UNDER A FOLD. The beta increment is masked with
the curl's own ownership clauses (``stepping._mask_non_owned_cells``) before the pass's
ladder subtracts it, and the fold WIDENS that mask from METALLIC-only at cell 0 to
non-PERIODIC at cell 0 PLUS MIRROR_PERIODIC at the top plane. Beta touches targets 0 and 1 only.
On a Y fold: step_D masks Dx's beta increment at j = 0 (Dx Yee shift 0 on y) and
Dy's at j = ny-1 (Dy shift 1 on y, MIRROR_PERIODIC); step_B masks By's at j = 0
and Bx's at j = ny-1. One folded axis therefore makes BOTH mask arms fire on BOTH
beta targets on BOTH sub-steps, so a pass that left its increment unmasked would
leave it live on the mirror plane and on the far ghost plane, invisible in the
interior. That is the gate's ``b_increment_unmasked`` and ``b_top_plane_mask_dropped``
needles.

FOLD x BLOCH DO NOT INTERACT AT THE WRAPPED PLANE. ``driver._require_bloch_is_
representable`` (driver.py:1040-1120) refuses a nonzero k on a mirror plane's OWN
axis in the Brillouin zone INTERIOR and refuses the zone EDGE too (MEEP #3155
supports it; this engine has not implemented the full-cell period).
``stepping._shift_up`` (:1768-1780) and ``_shift_down`` (:1819-1826) are mutually
exclusive per axis — PERIODIC applies the Bloch factor, MIRROR applies the parity
and NO wrap factor — and ``_far_reflect_rows`` (:1677-1681) states the same. So
``PH*`` MUST be 0 on every axis whose BC is a mirror code; the kernel does not
check it and the predicate refuses it. The two phases compose only across
DIFFERENT axes, multiplicatively, with no cross term.

THE BETA PARTNER IS UNTOUCHED BY THE FOLD. ``stepping._special_kz_beta_term``
(:727-784) takes a SAME-CELL center snapshot (S:293 electric / S:408 magnetic) —
no shift helper, no ghost rule, no reflect row — so no fold ghost value and no
parity ever multiplies it, and the Bloch connection applies to SHIFTED operands
only. No expression contains both the ±i quarter turn and a mirror parity. They compose one plane-copy downstream, at the FILL: ``fill_
symmetry_bc_D`` overwrites cell 0 with ``parity * cell 2``, and cell 2 already
carries its own beta increment — a composed-layer fact for the gate, not an
in-kernel coupling.

EXTENTS AND REFLECT ROWS, MEASURED ON REAL ``Grid`` OBJECTS (grid.py:1276-1311).
A folded PERIODIC axis stores exactly owned + 1 at BOTH count parities: n_full 20
-> owned 11 / stored 12 / reflect 10 = stored-2; 21 -> 12/13/10 = stored-3; 22 ->
12/13/11 = stored-2; 23 -> 13/14/11 = stored-3. A folded METALLIC axis stores
exactly owned and ``_far_reflect_rows`` returns None (20 -> 11/11; 21 -> 12/12).
The GHOST SLOT never degenerates — it is always the last stored slot and always
shift-1 components only; what moves at an odd count is the IMAGE ROW, one row
further down. A single 3-D grid can carry both arms at once (Mirror(x)+Mirror(y)
at n_full (20, 21) gives stored (12, 13) with reflect (10, 10), stored-2 on x and
stored-3 on y). ``reflect_row`` therefore stays a RUNTIME scalar read from
``stepping._far_reflect_rows``; baking ``n - 2`` is a whole cell wrong at an odd
full count.

GROUPING CHOICES ``stepping.py`` DOES NOT FORCE (the gate holds all of them):

1. ``CODE_PERIODIC``/``CODE_METALLIC``/``CODE_MIRROR_METALLIC``/
   ``CODE_MIRROR_PERIODIC``, ``MIRROR_SOURCE_INDEX`` and ``TARGET_IYEE`` are
   RESTATED here, never imported from ``symmetry.py`` (that file belongs to the
   symmetry tranche and is read-only for this one), and a laptop test pins every
   spelling equal to ``symmetry.py``'s AND to ``kernels.py``'s — the discipline
   ``symmetry.py`` itself used against ``kernels.py``.
2. The near/far parity coefficients are HOST-ROUNDED: the entries carry
   ``(re, im)`` word pairs (:func:`mirror_parity_coefficients`, imaginary word
   bitwise ``+0``), and the fill launches the real word alone, one float32 per
   pass — there is no imaginary coefficient word in the kernel to synthesise.
3. The fill multiplies each stored part by the parity word, one float32 product
   per part, as MEEP's connection forms it.
4. ONE LAUNCH PER FOLDED AXIS, in X, Y, Z order, held as a list the plan walks
   (the array path's order, though the per-part fill commutes, see above).
5. The fill kernel carries NO ``PHASE`` constexpr. The parity enters only as the
   passed coefficient word, so the identity-shortcut mutant (the multiply
   dropped) is caught on the phase = -1 rows and is a structural null on the +1
   rows, where ``x * (+1.0)`` IS ``x``. An unused constexpr in the shipped
   signature would be dead weight the reader has to discount.
7. Two host-rounded scalars per launch for the real beta family, one per sign,
   from ``special_kz.beta_curl_coefficients`` — ``special_kz``'s grouping choice
   1, inherited unchanged.
8. ``HAS_BETA`` is a constexpr; the engine routes always bind 1 (the predicates
   require beta nonzero). The 0 arm exists so the gate can pin the ``HAS_BETA=0``
   build byte-identical to the plain folded kernels.
9. The constitutive sub-steps are NOT new kernels. Complex delegates to
   ``complex_fields.bloch_constitutive_step`` through
   ``complex_fields.ComplexConstitutivePlan``; real + beta delegates to
   ``kernels.constitutive_step`` through ``launch.ConstitutivePlan``. Both are
   bound by NEW predicates that RESTATE the element-wise contract with the fold
   clause replaced by :func:`folded_axis_kinds` — never reached by subtracting
   reasons from another predicate's output.
10. The REAL family's ghost fill needs no new kernel: ``symmetry.mirror_ghost_fill``
    is exact for real storage (``±1 * float32`` is an exact sign flip including
    ``-0.0``) and ``symmetry.mirror_ghost_fill_coverage`` carries no beta clause,
    so it already admits a beta run. Nothing in ``symmetry.py`` is edited, and a
    laptop test PINS that predicate's verdict on a folded real beta grid so the
    coupling is visible if the symmetry tranche's owner adds a beta clause.
11. Fusion configuration: every launch goes out at ``enable_fp_fusion =
    ENABLE_FP_FUSION`` (fusion-OFF), which the artifact states — a
    multiply-subtract tail's bytes change with the setting.
12. The curl predicates check the NAMED SUB-STEP'S nine volumes, not all
    eighteen as a set (``coverage.py:306-315``'s "coverage is a set, not a la
    carte"). That is the per-sub-step scoping the certified complex and beta
    tranches already use; every pointer the launched kernel touches is still
    checked, and the engine route allocates all eighteen together
    (``Fields.enable_pml_storage``) so no engine-built configuration changes
    verdict. A hand-built run missing the OTHER sub-step's volumes is admitted
    here and refused by the shipped clause — recorded rather than left silent.
13. The fill's read/write plane disjointness is CHECKED (0 vs 2 through the
    stored-cell clause, ``last`` vs ``reflect_row <= stored - 2`` through the
    reflect-row bounds clause), never assumed.

DRIFT FOUND WHILE ESTABLISHING THIS (report only; ``symmetry.py`` is read-only
for this tranche). ``symmetry.py:44`` cites the driver's fill passes as
``driver.py:3163-3183``; they are now at ``driver.py:3206-3226``, and the range
names three slots where the driver has FIVE per half — the magnetic half is
step_B(3206) -> inject(3207-3208) -> fill_symmetry_bc_B(3209) -> zero_metal_B(3210)
-> fill_folded_far_ghosts_B(3211) -> update_H(3212), and the electric half is
step_D(3215) -> inject(3216-3221) -> fill_symmetry_bc_D(3222) -> zero_metal_D(3223)
-> fill_folded_far_ghosts_D(3224) -> update_E(3225) -> update_P(3226). The two
metallic passes sit BETWEEN the two fills and the far-ghost fill is the LAST thing
before the constitutive sub-step. A composed plan must not fuse across any of
those seams.

Import contract: this module is importable WITHOUT Triton — the predicates and
plan builders (to ``None``) must answer on the laptop that is the merge bar.
Triton is imported at module scope inside a guard, the kernels degrade to
:class:`_UnavailableKernel`, and ``ENABLE_FP_FUSION`` is imported from
:mod:`kernels` only inside ``run()`` so the guard keeps its single spelling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class _UnavailableKernel:
    """Fail at the launch expression while leaving host predicates importable."""

    __slots__ = ("name", "error")

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self.error = error

    def __getitem__(self, grid):
        raise ImportError(
            f"the optional triton package is required to launch {self.name}; "
            f"host coverage remains available without it: {self.error}"
        ) from self.error


try:  # The host predicate and public refusal path must work without Triton.
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - exercised by the absence test
    _TRITON_IMPORT_ERROR = _exc

    class _MissingTriton:
        @staticmethod
        def jit(function):
            return _UnavailableKernel(function.__name__, _TRITON_IMPORT_ERROR)

    class _MissingLanguage:
        @staticmethod
        def constexpr(value):
            return value

    triton = _MissingTriton()  # type: ignore[assignment]
    tl = _MissingLanguage()  # type: ignore[assignment]

from .complex_fields import (  # noqa: E402 - this package's own module
    ComplexConstitutivePlan,
    _complex_layout_reasons,
    _word_view,
    bloch_phase_table,
    connection_arguments,
)
from .coverage import (  # READ-ONLY imports; nothing here mutates coverage.py
    CONSTITUTIVE_SIDES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
)
from .launch import SUB_STEPS, ConstitutivePlan, CupyPointer, _flat, chunk_bounds_for
from .special_kz import (  # noqa: E402 - this package's own module
    beta_curl_coefficients,
)


# ---------------------------------------------------------------------------
# The constants the kernel and the host agree on — RESTATED, never imported
# ---------------------------------------------------------------------------
#
# ``symmetry.py`` belongs to the symmetry tranche and is read-only for this one,
# so its four codes, its mirror source index and its Yee-shift table are restated
# here and PINNED EQUAL by ``test_triton_folded_complex`` — against BOTH
# ``symmetry.py`` and ``kernels.py``. That is exactly the discipline
# ``symmetry.py`` itself used against ``kernels.py``, and it is what stops a plan
# built here and a plan built there from indexing different tables.

CODE_PERIODIC = 0
CODE_METALLIC = 1
CODE_MIRROR_METALLIC = 2
CODE_MIRROR_PERIODIC = 3

#: ``stepping.MIRROR_SOURCE_INDEX`` (stepping.py:160) — MEEP's ``io = -2`` halved
#: origin, so the near ghost images stored cell 2.
MIRROR_SOURCE_INDEX = 2

#: ``stepping._boundary_kinds``' own strings, in the order this file codes them.
BOUNDARY_KIND_NAMES: Tuple[str, ...] = ("periodic", "metallic", "mirror")

#: Yee shifts of the six curl targets (``fields.IYEE_SHIFTS``, fields.py:214-219).
#: The B family's shifts are 1 on the two axes that are NOT its own; the D
#: family's are 1 on its own axis only — which is why the two sub-steps mask
#: different planes.
TARGET_IYEE: Dict[str, Tuple[int, int, int]] = {
    "Bx": (0, 1, 1), "By": (1, 0, 1), "Bz": (1, 1, 0),
    "Dx": (1, 0, 0), "Dy": (0, 1, 0), "Dz": (0, 0, 1),
}

#: Which family each fill launch covers. ``targets`` are in kernel argument order.
GHOST_FILL_FAMILIES: Dict[str, Dict[str, Any]] = {
    "B": {"targets": ("Bx", "By", "Bz")},
    "D": {"targets": ("Dx", "Dy", "Dz")},
}

PERIODIC = tl.constexpr(CODE_PERIODIC)
METALLIC = tl.constexpr(CODE_METALLIC)
MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

# The forms, the record layout and the connection, read by K1. Imported only when
# Triton is: those modules import it at module scope, and this module's host route
# must answer without it.
if _TRITON_IMPORT_ERROR is None:
    from .complex_fields import _connect  # noqa: E402
    from .forms import (axis_present, curl_form, curl_stencil,  # noqa: E402
                        fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

#: Elements per program — complex CELLS for K1/K2/K3b, real cells for K3a.
#: Restated from ``kernels.DEFAULT_BLOCK`` for the same import reason as the
#: boundary codes; no autotune, because these kernels write their own inputs in
#: place and the tuner would silently apply the update dozens of times
#: (kernels.py:56-59).
DEFAULT_BLOCK = 256


__all__ = [
    "CODE_METALLIC",
    "CODE_MIRROR_METALLIC",
    "CODE_MIRROR_PERIODIC",
    "CODE_PERIODIC",
    "FoldedBetaBlochPmlCurlPlan",
    "FoldedBetaPmlCurlPlan",
    "FoldedComplexPmlCurlPlan",
    "FoldedMirrorGhostFillComplexPlan",
    "GHOST_FILL_FAMILIES",
    "MIRROR_SOURCE_INDEX",
    "TARGET_IYEE",
    "folded_axis_kinds",
    "folded_beta_bloch_pml_curl_coverage",
    "folded_beta_bloch_pml_curl_step",
    "folded_beta_pml_curl_coverage",
    "folded_beta_pml_curl_step",
    "folded_beta_run_constitutive_coverage",
    "folded_bloch_pml_curl_step",
    "folded_complex_constitutive_coverage",
    "folded_complex_pml_curl_coverage",
    "folded_mirror_ghost_fill_complex",
    "folded_mirror_ghost_fill_complex_coverage",
    "ghost_fill_axis_entries",
    "mirror_parity_coefficients",
    "plan_folded_beta_bloch_pml_curl",
    "plan_folded_beta_bloch_pml_curl_from_arrays",
    "plan_folded_beta_pml_curl",
    "plan_folded_beta_pml_curl_from_arrays",
    "plan_folded_beta_run_constitutive",
    "plan_folded_complex_constitutive",
    "plan_folded_complex_pml_curl",
    "plan_folded_complex_pml_curl_from_arrays",
    "plan_folded_mirror_ghost_fill_complex",
    "plan_folded_mirror_ghost_fill_complex_from_arrays",
]


# ---------------------------------------------------------------------------
# K1 — the folded complex curl
# ---------------------------------------------------------------------------

@triton.jit
def folded_bloch_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz or Dx,Dy,Dz (complex64 as words)
    u0, u1, u2,                       # auxiliaries: fu_B* or fu_D* (complex64 as words)
    g0, g1, g2,                       # sources: Ex,Ey,Ez or Hx,Hy,Hz (complex64 as words)
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, float32, one sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kind per axis
    BLOCK: tl.constexpr,
):
    """One COMPLEX curl sub-step on a FOLDED grid, both PML forms included.

    :func:`.complex_fields.bloch_pml_curl_step`'s body with exactly three deltas, all
    of them ``symmetry.pml_curl_step_folded``'s and NONE of them arithmetic:

    1. the boundary constexprs widen from {PERIODIC, METALLIC} to the four codes, and
       BOTH mirror codes take the masked ghost branch (serve ``(+0.0, +0.0)``). That is
       not the array path's ghost VALUE on a fold and it does not have to be: the only
       cell that reads either ghost is a cell one of the two masks below zeroes, and on
       MIRROR_METALLIC the top plane is stepped against exactly that zero pair;
    2. the cell-0 ownership mask widens from ``== METALLIC`` to ``!= PERIODIC``
       (``_mask_non_owned_cells``' ``is_mirrored or is_metallic``);
    3. a folded PERIODIC axis masks its LAST plane for every target whose Yee shift is
       1 there (``_mask_non_owned_cells``' ``_stored_past_owned`` arm).

    Everything else is the unfolded complex body: the parts stepped with the real
    forms (:mod:`.forms`) on the chunk record, MEEP's left-to-right stencil with
    partner presence, ``fu`` read and written on the recurrence's cells only, and the
    Bloch connection on the UNSHIFTED partner's plane 0 of a phased PERIODIC axis.
    A folded axis carries no phase (the driver refuses a nonzero k on a mirror plane's
    axis and the predicate re-checks it), so ``CN*`` is 0 on every folded axis; a
    ghost row of a folded axis reads the membership line its image row reads.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- DELTA 1: the ghost rule, per axis (stepping._shift_up / _shift_down) ---
    # PERIODIC wraps (a plain roll, phased or not); EVERY other rule here serves an
    # exact (+0.0, +0.0) past the face, which `tl.load`'s `other=` delivers on both
    # words. On a folded axis that zero stands in for a value nothing reads (the two
    # masks below zero every cell that reads it).
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vx, vy, vz = live, live, live
    if BCX == PERIODIC:
        si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
    else:
        vx = live & (si >= 0) & (si < nx)
    if BCY == PERIODIC:
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    else:
        vy = live & (sj >= 0) & (sj < ny)
    if BCZ == PERIODIC:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
    else:
        vz = live & (sk >= 0) & (sk < nz)

    ox = si * nyz + j * nz + k
    oy = i * nyz + sj * nz + k
    oz = i * nyz + j * nz + sk

    # --- loads: two words per operand ------------------------------------------
    a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
    a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)
    a_z_re = tl.load(g0 + 2 * oz, mask=vz, other=0.0)
    a_z_im = tl.load(g0 + 2 * oz + 1, mask=vz, other=0.0)
    b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)
    b_x_im = tl.load(g1 + 2 * ox + 1, mask=vx, other=0.0)
    b_z_re = tl.load(g1 + 2 * oz, mask=vz, other=0.0)
    b_z_im = tl.load(g1 + 2 * oz + 1, mask=vz, other=0.0)
    c_x_re = tl.load(g2 + 2 * ox, mask=vx, other=0.0)
    c_x_im = tl.load(g2 + 2 * ox + 1, mask=vx, other=0.0)
    c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)
    c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)

    # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
    # Target 0 reads c along y and b along z, target 1 a along z and c along x,
    # target 2 b along x and a along y (stepping's term table). `tl.where` is a
    # bitwise select, so every cell off plane 0 keeps its loaded words.
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
    b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
    a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
    c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
    b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
    a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
    if CNX:
        r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
        c1_re = tl.where(at_x, r_re, c_re)
        c1_im = tl.where(at_x, r_im, c_im)
        r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
        b2_re = tl.where(at_x, r_re, b_re)
        b2_im = tl.where(at_x, r_im, b_im)
    if CNY:
        r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
        c0_re = tl.where(at_y, r_re, c_re)
        c0_im = tl.where(at_y, r_im, c_im)
        r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
        a2_re = tl.where(at_y, r_re, a_re)
        a2_im = tl.where(at_y, r_im, a_im)
    if CNZ:
        r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
        b0_re = tl.where(at_z, r_re, b_re)
        b0_im = tl.where(at_z, r_im, b_im)
        r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
        a1_re = tl.where(at_z, r_re, a_re)
        a1_im = tl.where(at_z, r_im, a_im)

    # --- the curl (stepping._curl_from_operands), per part ---------------------
    # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
    # association with both partners, the one-partner form where the record says the
    # other partner's derivative axis is invariant, an exact +0 with neither. The
    # dtdx scale is the real scalar times each part (part_products.scale).
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
    curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
    curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
    curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
    curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
    curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

    # --- DELTA 2: ownership mask, cell 0 (stepping._mask_non_owned_cells) -------
    # Every target whose Yee shift is 0 on a non-periodic axis (``is_mirrored or
    # is_metallic``): the unfolded body's ``== METALLIC`` widened to ``!= PERIODIC``.
    # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
    if BACKWARD:
        if BCY != PERIODIC:
            curl0_re = tl.where(at_y, 0.0, curl0_re)
            curl0_im = tl.where(at_y, 0.0, curl0_im)
        if BCZ != PERIODIC:
            curl0_re = tl.where(at_z, 0.0, curl0_re)
            curl0_im = tl.where(at_z, 0.0, curl0_im)
        if BCX != PERIODIC:
            curl1_re = tl.where(at_x, 0.0, curl1_re)
            curl1_im = tl.where(at_x, 0.0, curl1_im)
        if BCZ != PERIODIC:
            curl1_re = tl.where(at_z, 0.0, curl1_re)
            curl1_im = tl.where(at_z, 0.0, curl1_im)
        if BCX != PERIODIC:
            curl2_re = tl.where(at_x, 0.0, curl2_re)
            curl2_im = tl.where(at_x, 0.0, curl2_im)
        if BCY != PERIODIC:
            curl2_re = tl.where(at_y, 0.0, curl2_re)
            curl2_im = tl.where(at_y, 0.0, curl2_im)
    else:
        if BCX != PERIODIC:
            curl0_re = tl.where(at_x, 0.0, curl0_re)
            curl0_im = tl.where(at_x, 0.0, curl0_im)
        if BCY != PERIODIC:
            curl1_re = tl.where(at_y, 0.0, curl1_re)
            curl1_im = tl.where(at_y, 0.0, curl1_im)
        if BCZ != PERIODIC:
            curl2_re = tl.where(at_z, 0.0, curl2_re)
            curl2_im = tl.where(at_z, 0.0, curl2_im)

    # --- DELTA 3: ownership mask, the TOP plane of a folded PERIODIC axis --------
    # Every target whose Yee shift is 1 there: that slot sits past MEEP's owned
    # window and the far fill, not the curl, writes it (symmetry.py's block, both
    # parts).
    last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
    if BACKWARD:
        # Dx:(1,0,0)  Dy:(0,1,0)  Dz:(0,0,1) -- shift 1 on its OWN axis only.
        if BCX == MIRROR_PERIODIC:
            curl0_re = tl.where(last_x, 0.0, curl0_re)
            curl0_im = tl.where(last_x, 0.0, curl0_im)
        if BCY == MIRROR_PERIODIC:
            curl1_re = tl.where(last_y, 0.0, curl1_re)
            curl1_im = tl.where(last_y, 0.0, curl1_im)
        if BCZ == MIRROR_PERIODIC:
            curl2_re = tl.where(last_z, 0.0, curl2_re)
            curl2_im = tl.where(last_z, 0.0, curl2_im)
    else:
        # Bx:(0,1,1)  By:(1,0,1)  Bz:(1,1,0) -- shift 1 on the two OTHER axes.
        if BCY == MIRROR_PERIODIC:
            curl0_re = tl.where(last_y, 0.0, curl0_re)
            curl0_im = tl.where(last_y, 0.0, curl0_im)
        if BCZ == MIRROR_PERIODIC:
            curl0_re = tl.where(last_z, 0.0, curl0_re)
            curl0_im = tl.where(last_z, 0.0, curl0_im)
        if BCX == MIRROR_PERIODIC:
            curl1_re = tl.where(last_x, 0.0, curl1_re)
            curl1_im = tl.where(last_x, 0.0, curl1_im)
        if BCZ == MIRROR_PERIODIC:
            curl1_re = tl.where(last_z, 0.0, curl1_re)
            curl1_im = tl.where(last_z, 0.0, curl1_im)
        if BCX == MIRROR_PERIODIC:
            curl2_re = tl.where(last_x, 0.0, curl2_re)
            curl2_im = tl.where(last_x, 0.0, curl2_im)
        if BCY == MIRROR_PERIODIC:
            curl2_re = tl.where(last_y, 0.0, curl2_re)
            curl2_im = tl.where(last_y, 0.0, curl2_im)

    # --- the PML forms (stepping._apply_curl_regions), per part ----------------
    # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). One record
    # select per target serves both parts; fu is read and written on the
    # recurrence's cells only (the auxiliary contract), both words.
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    if BACKWARD:
        lattice = 0
    else:
        lattice = RECORD_HALF_INTEGER
    du0 = record_in(cb, lattice + 8, k)
    du1 = record_in(cb, lattice + 0, i)
    du2 = record_in(cb, lattice + 4, j)

    p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)
    p_im = tl.load(u0 + 2 * idx + 1, mask=live & du0, other=0.0)
    v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=live, other=0.0), p_re, curl0_re,
                             km_y, si_y, km_z, si_z, du0)
    v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl0_im, km_y, si_y, km_z, si_z, du0)

    p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
    p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
    v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=live, other=0.0), p_re, curl1_re,
                             km_z, si_z, km_x, si_x, du1)
    v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl1_im, km_z, si_z, km_x, si_x, du1)

    p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
    p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
    v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=live, other=0.0), p_re, curl2_re,
                             km_x, si_x, km_y, si_y, du2)
    v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl2_im, km_x, si_x, km_y, si_y, du2)

    # --- stores: fu (recurrence cells only), then the fields, both parts --------
    tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
    tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
    tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
    tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
    tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
    tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


# ---------------------------------------------------------------------------
# K2 — the folded complex ghost fill (THE ONE ARITHMETIC DELTA)
# ---------------------------------------------------------------------------

@triton.jit
def folded_mirror_ghost_fill_complex(
    f0, f1, f2,                       # one family: Bx,By,Bz or Dx,Dy,Dz (complex64 words)
    nx, ny, nz, n_plane,
    reflect_row,                      # RUNTIME: stepping._far_reflect_rows
    near_c,                           # host-rounded float32 word of +phase
    far_c,                            # host-rounded float32 word of -phase
    AXIS: tl.constexpr,               # 0 = x, 1 = y, 2 = z
    DO_NEAR: tl.constexpr,            # 1 on the fill_symmetry_bc_* pass
    DO_FAR: tl.constexpr,             # 1 on the fill_folded_far_ghosts_* pass,
    #                                 # and only on a folded PERIODIC axis
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,   # Yee shift on AXIS
    BLOCK: tl.constexpr,
):
    """The two mirror ghost planes of ONE family on ONE axis, COMPLEX storage.

    Replaces, for that (family, axis) pair, the whole-plane assignments in
    ``stepping._fill_symmetry_ghost_cells`` and ``stepping._fill_folded_far_ghosts``,
    through ``host_writes.copy_face``::

        near:  field[0]  = (+phase) * field[2]              # Yee shift 0 here
        far:   field[-1] = (-phase) * field[reflect_row]    # Yee shift 1 here,
                                                            # folded PERIODIC only

    THE PARITY MULTIPLY IS PER PART: ``part_products.scale`` rounds the Python int to
    the part dtype and multiplies each stored part by it, as MEEP's connection
    multiplies each ``realnum`` part (a ``-1`` connection is a unary minus per part,
    boundaries.cpp:359-366). ``x * (+1.0)`` and ``x * (-1.0)`` are exact on every word,
    signed zeros and subnormals included, so the fill is the real fill per part and
    the two passes and the three axes commute bitwise; they are still launched in the
    array path's order (near pass, then far pass after the driver's wall clear, X, Y,
    Z within each). The coefficient words are host-rounded and passed
    (:func:`mirror_parity_coefficients`).

    ``reflect_row`` IS A RUNTIME SCALAR and must come from
    ``stepping._far_reflect_rows``, ``n_full - stored + 2``: ``stored - 2`` at an even
    full count and ``stored - 3`` at an odd one.

    WORD-PAIR ADDRESSING. The CELL base/stride/last are computed exactly as the real
    kernel computes them, then addressed at ``2*off`` and ``2*off + 1``.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_plane

    # Flatten one plane of the C-contiguous (nx, ny, nz) COMPLEX volume: the CELL
    # offset of index `row` along AXIS, plus the in-plane offset this program owns.
    if AXIS == 0:
        stride = ny * nz
        base = idx
        last = nx - 1
    elif AXIS == 1:
        stride = nz
        base = (idx // nz) * (ny * nz) + (idx % nz)
        last = ny - 1
    else:
        stride = 1
        base = (idx // ny) * (ny * nz) + (idx % ny) * nz
        last = nz - 1

    # --- near face: cell 0 = (+phase) * cell MIRROR_SOURCE_INDEX, per part ------
    # Only the components with Yee shift 0 on this axis have an unowned cell 0.
    if DO_NEAR and S0 == 0:
        src = base + 2 * stride
        tl.store(f0 + 2 * base, near_c * tl.load(f0 + 2 * src, mask=live, other=0.0),
                 mask=live)
        tl.store(f0 + 2 * base + 1,
                 near_c * tl.load(f0 + 2 * src + 1, mask=live, other=0.0), mask=live)
    if DO_NEAR and S1 == 0:
        src = base + 2 * stride
        tl.store(f1 + 2 * base, near_c * tl.load(f1 + 2 * src, mask=live, other=0.0),
                 mask=live)
        tl.store(f1 + 2 * base + 1,
                 near_c * tl.load(f1 + 2 * src + 1, mask=live, other=0.0), mask=live)
    if DO_NEAR and S2 == 0:
        src = base + 2 * stride
        tl.store(f2 + 2 * base, near_c * tl.load(f2 + 2 * src, mask=live, other=0.0),
                 mask=live)
        tl.store(f2 + 2 * base + 1,
                 near_c * tl.load(f2 + 2 * src + 1, mask=live, other=0.0), mask=live)

    # --- far face: last cell = (-phase) * cell reflect_row, per part ------------
    # Only on a folded PERIODIC axis, and only for shift-1 components.
    if DO_FAR:
        if S0 == 1:
            dst = base + last * stride
            src = base + reflect_row * stride
            tl.store(f0 + 2 * dst, far_c * tl.load(f0 + 2 * src, mask=live, other=0.0),
                     mask=live)
            tl.store(f0 + 2 * dst + 1,
                     far_c * tl.load(f0 + 2 * src + 1, mask=live, other=0.0), mask=live)
        if S1 == 1:
            dst = base + last * stride
            src = base + reflect_row * stride
            tl.store(f1 + 2 * dst, far_c * tl.load(f1 + 2 * src, mask=live, other=0.0),
                     mask=live)
            tl.store(f1 + 2 * dst + 1,
                     far_c * tl.load(f1 + 2 * src + 1, mask=live, other=0.0), mask=live)
        if S2 == 1:
            dst = base + last * stride
            src = base + reflect_row * stride
            tl.store(f2 + 2 * dst, far_c * tl.load(f2 + 2 * src, mask=live, other=0.0),
                     mask=live)
            tl.store(f2 + 2 * dst + 1,
                     far_c * tl.load(f2 + 2 * src + 1, mask=live, other=0.0), mask=live)


# ---------------------------------------------------------------------------
# K3a — the folded REAL beta curl
# ---------------------------------------------------------------------------

@triton.jit
def folded_beta_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
    u0, u1, u2,                       # auxiliaries: fu_B*  or  fu_D*
    g0, g1, g2,                       # sources: Ex,Ey,Ez  or  Hx,Hy,Hz
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, one Yee sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,
    beta_plus, beta_minus,            # f32(sign*2*pi*beta*dt), host-rounded once
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    HAS_BETA: tl.constexpr,           # compiled-in only for beta != 0 runs
    BLOCK: tl.constexpr,
):
    """One REAL curl sub-step with MEEP's special-kz beta pass, on a FOLDED grid.

    ``symmetry.pml_curl_step_folded``'s body, character for character, plus
    ``special_kz.beta_pml_curl_step``'s beta pass after the three curl forms and before
    the stores. The pass's increment carries the FOLDED ownership mask, the same two
    blocks the folded curl applies (cell 0 of every non-periodic axis for a target
    whose Yee shift is 0 there; the top plane of a folded PERIODIC axis for a target
    whose shift is 1), so a cell the curl leaves to a fill keeps its word through the
    pass too, as ``stepping._mask_non_owned_cells`` leaves it in the array path.
    ``HAS_BETA = 0`` builds ``symmetry.pml_curl_step_folded`` exactly.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) --------
    # PERIODIC wraps; every other rule here serves an exact 0.0 past the face,
    # which `tl.load`'s `other=` delivers without dereferencing anything. On a
    # folded axis that zero stands in for a value nothing reads (see 1 above).
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vx, vy, vz = live, live, live
    if BCX == PERIODIC:
        si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
    else:
        vx = live & (si >= 0) & (si < nx)
    if BCY == PERIODIC:
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    else:
        vy = live & (sj >= 0) & (sj < ny)
    if BCZ == PERIODIC:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
    else:
        vz = live & (sk >= 0) & (sk < nz)

    ox = si * nyz + j * nz + k
    oy = i * nyz + sj * nz + k
    oz = i * nyz + j * nz + sk

    a = tl.load(g0 + idx, mask=live, other=0.0)
    b = tl.load(g1 + idx, mask=live, other=0.0)
    c = tl.load(g2 + idx, mask=live, other=0.0)
    a_y = tl.load(g0 + oy, mask=vy, other=0.0)
    a_z = tl.load(g0 + oz, mask=vz, other=0.0)
    b_x = tl.load(g1 + ox, mask=vx, other=0.0)
    b_z = tl.load(g1 + oz, mask=vz, other=0.0)
    c_x = tl.load(g2 + ox, mask=vx, other=0.0)
    c_y = tl.load(g2 + oy, mask=vy, other=0.0)

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens
    # kernels.pml_curl_step's three calls: MEEP's left-to-right association with
    # both partners, the one-partner form where the record says the other
    # partner's derivative axis is invariant, an exact +0 with neither.
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
    curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
    curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

    # --- ownership mask, cell 0 (stepping._mask_non_owned_cells) ------------
    # Every target whose Yee shift is 0 on a non-periodic axis. Byte-copied
    # from the shipped kernel with `== METALLIC` widened to `!= PERIODIC`.
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    if BACKWARD:
        if BCY != PERIODIC:
            curl0 = tl.where(at_y, 0.0, curl0)
        if BCZ != PERIODIC:
            curl0 = tl.where(at_z, 0.0, curl0)
        if BCX != PERIODIC:
            curl1 = tl.where(at_x, 0.0, curl1)
        if BCZ != PERIODIC:
            curl1 = tl.where(at_z, 0.0, curl1)
        if BCX != PERIODIC:
            curl2 = tl.where(at_x, 0.0, curl2)
        if BCY != PERIODIC:
            curl2 = tl.where(at_y, 0.0, curl2)
    else:
        if BCX != PERIODIC:
            curl0 = tl.where(at_x, 0.0, curl0)
        if BCY != PERIODIC:
            curl1 = tl.where(at_y, 0.0, curl1)
        if BCZ != PERIODIC:
            curl2 = tl.where(at_z, 0.0, curl2)

    # --- ownership mask, the TOP plane of a folded PERIODIC axis ------------
    # The complement of the block above: every target whose Yee shift is 1
    # there. Written out per (side, target, axis) exactly as that block is —
    # no loop, no derived predicate — so a reader checks it against
    # `_mask_non_owned_cells`'s `if iyee[axis] != 0` arm by eye.
    last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
    if BACKWARD:
        # Dx:(1,0,0)  Dy:(0,1,0)  Dz:(0,0,1) — shift 1 on its OWN axis only.
        if BCX == MIRROR_PERIODIC:
            curl0 = tl.where(last_x, 0.0, curl0)
        if BCY == MIRROR_PERIODIC:
            curl1 = tl.where(last_y, 0.0, curl1)
        if BCZ == MIRROR_PERIODIC:
            curl2 = tl.where(last_z, 0.0, curl2)
    else:
        # Bx:(0,1,1)  By:(1,0,1)  Bz:(1,1,0) — shift 1 on the two OTHER axes.
        if BCY == MIRROR_PERIODIC:
            curl0 = tl.where(last_y, 0.0, curl0)
        if BCZ == MIRROR_PERIODIC:
            curl0 = tl.where(last_z, 0.0, curl0)
        if BCX == MIRROR_PERIODIC:
            curl1 = tl.where(last_x, 0.0, curl1)
        if BCZ == MIRROR_PERIODIC:
            curl1 = tl.where(last_z, 0.0, curl1)
        if BCX == MIRROR_PERIODIC:
            curl2 = tl.where(last_x, 0.0, curl2)
        if BCY == MIRROR_PERIODIC:
            curl2 = tl.where(last_y, 0.0, curl2)

    # --- the PML forms (stepping._apply_curl_regions) ----------------------
    # Byte-copied from kernels.pml_curl_step. The PML coefficient vectors are
    # built at the STORED extent on a folded axis (measured: kms_y.shape ==
    # (1, 22, 1) on a grid storing 22), so the `n_a` indexing needs no
    # fold-aware change.
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    # Which cells take the recurrence: the component's dsigu layer on its own
    # sub-lattice (dsigu axis z for target 0, x for 1, y for 2; half-integer for
    # every B target, integer for every D one), a compile-time record word.
    if BACKWARD:
        lattice = 0
    else:
        lattice = RECORD_HALF_INTEGER
    du0 = record_in(cb, lattice + 8, k)
    du1 = record_in(cb, lattice + 0, i)
    du2 = record_in(cb, lattice + 4, j)

    p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)
    v0, n0 = curl_form(tl.load(f0 + idx, mask=live, other=0.0), p0, curl0,
                       km_y, si_y, km_z, si_z, du0)

    p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)
    v1, n1 = curl_form(tl.load(f1 + idx, mask=live, other=0.0), p1, curl1,
                       km_z, si_z, km_x, si_x, du1)

    p2 = tl.load(u2 + idx, mask=live & du2, other=0.0)
    v2, n2 = curl_form(tl.load(f2 + idx, mask=live, other=0.0), p2, curl2,
                       km_x, si_x, km_y, si_y, du2)

    # --- the special-kz beta pass (stepping._special_kz_beta_term, then
    # stepping._apply_folded_increment) ----------------------------------------
    # MEEP's pass of its own after every component's curl (step_db.cpp:161-175),
    # applied to the registers the forms returned. The partners are the CENTER
    # loads: target 0 takes the second source at sign +1 (Bx <- Ey, Dx <- Hy),
    # target 1 the first at sign -1 (By <- Ex, Dy <- Hx); target 2 gets nothing.
    # The increment is the NEGATED product (no dtdx: an analytic derivative),
    # masked to +0 on the cells the curl's own mask drops, and the beta ladder
    # subtracts it (forms.fold_ladder).
    if HAS_BETA:
        q0 = (beta_plus * b) * -1.0
        q1 = (beta_minus * a) * -1.0
        if BACKWARD:
            if BCY != PERIODIC:
                q0 = tl.where(at_y, 0.0, q0)
            if BCZ != PERIODIC:
                q0 = tl.where(at_z, 0.0, q0)
            if BCX != PERIODIC:
                q1 = tl.where(at_x, 0.0, q1)
            if BCZ != PERIODIC:
                q1 = tl.where(at_z, 0.0, q1)
            if BCX == MIRROR_PERIODIC:
                q0 = tl.where(last_x, 0.0, q0)
            if BCY == MIRROR_PERIODIC:
                q1 = tl.where(last_y, 0.0, q1)
        else:
            if BCX != PERIODIC:
                q0 = tl.where(at_x, 0.0, q0)
            if BCY != PERIODIC:
                q1 = tl.where(at_y, 0.0, q1)
            if BCY == MIRROR_PERIODIC:
                q0 = tl.where(last_y, 0.0, q0)
            if BCZ == MIRROR_PERIODIC:
                q0 = tl.where(last_z, 0.0, q0)
            if BCX == MIRROR_PERIODIC:
                q1 = tl.where(last_x, 0.0, q1)
            if BCZ == MIRROR_PERIODIC:
                q1 = tl.where(last_z, 0.0, q1)
        v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)
        v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)

    # fu is written on the recurrence's cells ONLY (the auxiliary contract).
    tl.store(u0 + idx, n0, mask=live & du0)
    tl.store(u1 + idx, n1, mask=live & du1)
    tl.store(u2 + idx, n2, mask=live & du2)
    tl.store(f0 + idx, v0, mask=live)
    tl.store(f1 + idx, v1, mask=live)
    tl.store(f2 + idx, v2, mask=live)


# ---------------------------------------------------------------------------
# K3b — the folded COMPLEX beta curl
# ---------------------------------------------------------------------------

@triton.jit
def folded_beta_bloch_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz or Dx,Dy,Dz (complex64 as words)
    u0, u1, u2,                       # auxiliaries: fu_B* or fu_D* (complex64 as words)
    g0, g1, g2,                       # sources: Ex,Ey,Ez or Hx,Hy,Hz (complex64 as words)
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, float32, one sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
    beta_plus, beta_minus,            # f32 quarter-turn words per sign (B: +c, D: -c)
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kind per axis
    HAS_BETA: tl.constexpr,           # compiled-in only for beta != 0 runs
    BLOCK: tl.constexpr,
):
    """K1 plus ``special_kz.beta_bloch_pml_curl_step``'s beta pass, per part.

    :func:`folded_bloch_pml_curl_step`'s body, character for character, plus the
    complex beta pass after the six part forms and before the stores: the negated
    quarter turn of the CENTER partner's parts by the side's real word, masked to
    ``(+0, +0)`` by K1's two ownership blocks, through the beta ladder per part. No
    complex product is formed. ``HAS_BETA = 0`` builds K1 exactly.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- DELTA 1: the ghost rule, per axis (stepping._shift_up / _shift_down) ---
    # PERIODIC wraps (a plain roll, phased or not); EVERY other rule here serves an
    # exact (+0.0, +0.0) past the face, which `tl.load`'s `other=` delivers on both
    # words. On a folded axis that zero stands in for a value nothing reads (the two
    # masks below zero every cell that reads it).
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vx, vy, vz = live, live, live
    if BCX == PERIODIC:
        si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
    else:
        vx = live & (si >= 0) & (si < nx)
    if BCY == PERIODIC:
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    else:
        vy = live & (sj >= 0) & (sj < ny)
    if BCZ == PERIODIC:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
    else:
        vz = live & (sk >= 0) & (sk < nz)

    ox = si * nyz + j * nz + k
    oy = i * nyz + sj * nz + k
    oz = i * nyz + j * nz + sk

    # --- loads: two words per operand ------------------------------------------
    a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
    a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)
    a_z_re = tl.load(g0 + 2 * oz, mask=vz, other=0.0)
    a_z_im = tl.load(g0 + 2 * oz + 1, mask=vz, other=0.0)
    b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)
    b_x_im = tl.load(g1 + 2 * ox + 1, mask=vx, other=0.0)
    b_z_re = tl.load(g1 + 2 * oz, mask=vz, other=0.0)
    b_z_im = tl.load(g1 + 2 * oz + 1, mask=vz, other=0.0)
    c_x_re = tl.load(g2 + 2 * ox, mask=vx, other=0.0)
    c_x_im = tl.load(g2 + 2 * ox + 1, mask=vx, other=0.0)
    c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)
    c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)

    # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
    # Target 0 reads c along y and b along z, target 1 a along z and c along x,
    # target 2 b along x and a along y (stepping's term table). `tl.where` is a
    # bitwise select, so every cell off plane 0 keeps its loaded words.
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
    b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
    a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
    c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
    b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
    a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
    if CNX:
        r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
        c1_re = tl.where(at_x, r_re, c_re)
        c1_im = tl.where(at_x, r_im, c_im)
        r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
        b2_re = tl.where(at_x, r_re, b_re)
        b2_im = tl.where(at_x, r_im, b_im)
    if CNY:
        r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
        c0_re = tl.where(at_y, r_re, c_re)
        c0_im = tl.where(at_y, r_im, c_im)
        r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
        a2_re = tl.where(at_y, r_re, a_re)
        a2_im = tl.where(at_y, r_im, a_im)
    if CNZ:
        r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
        b0_re = tl.where(at_z, r_re, b_re)
        b0_im = tl.where(at_z, r_im, b_im)
        r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
        a1_re = tl.where(at_z, r_re, a_re)
        a1_im = tl.where(at_z, r_im, a_im)

    # --- the curl (stepping._curl_from_operands), per part ---------------------
    # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
    # association with both partners, the one-partner form where the record says the
    # other partner's derivative axis is invariant, an exact +0 with neither. The
    # dtdx scale is the real scalar times each part (part_products.scale).
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
    curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
    curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
    curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
    curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
    curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

    # --- DELTA 2: ownership mask, cell 0 (stepping._mask_non_owned_cells) -------
    # Every target whose Yee shift is 0 on a non-periodic axis (``is_mirrored or
    # is_metallic``): the unfolded body's ``== METALLIC`` widened to ``!= PERIODIC``.
    # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
    if BACKWARD:
        if BCY != PERIODIC:
            curl0_re = tl.where(at_y, 0.0, curl0_re)
            curl0_im = tl.where(at_y, 0.0, curl0_im)
        if BCZ != PERIODIC:
            curl0_re = tl.where(at_z, 0.0, curl0_re)
            curl0_im = tl.where(at_z, 0.0, curl0_im)
        if BCX != PERIODIC:
            curl1_re = tl.where(at_x, 0.0, curl1_re)
            curl1_im = tl.where(at_x, 0.0, curl1_im)
        if BCZ != PERIODIC:
            curl1_re = tl.where(at_z, 0.0, curl1_re)
            curl1_im = tl.where(at_z, 0.0, curl1_im)
        if BCX != PERIODIC:
            curl2_re = tl.where(at_x, 0.0, curl2_re)
            curl2_im = tl.where(at_x, 0.0, curl2_im)
        if BCY != PERIODIC:
            curl2_re = tl.where(at_y, 0.0, curl2_re)
            curl2_im = tl.where(at_y, 0.0, curl2_im)
    else:
        if BCX != PERIODIC:
            curl0_re = tl.where(at_x, 0.0, curl0_re)
            curl0_im = tl.where(at_x, 0.0, curl0_im)
        if BCY != PERIODIC:
            curl1_re = tl.where(at_y, 0.0, curl1_re)
            curl1_im = tl.where(at_y, 0.0, curl1_im)
        if BCZ != PERIODIC:
            curl2_re = tl.where(at_z, 0.0, curl2_re)
            curl2_im = tl.where(at_z, 0.0, curl2_im)

    # --- DELTA 3: ownership mask, the TOP plane of a folded PERIODIC axis --------
    # Every target whose Yee shift is 1 there: that slot sits past MEEP's owned
    # window and the far fill, not the curl, writes it (symmetry.py's block, both
    # parts).
    last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
    if BACKWARD:
        # Dx:(1,0,0)  Dy:(0,1,0)  Dz:(0,0,1) -- shift 1 on its OWN axis only.
        if BCX == MIRROR_PERIODIC:
            curl0_re = tl.where(last_x, 0.0, curl0_re)
            curl0_im = tl.where(last_x, 0.0, curl0_im)
        if BCY == MIRROR_PERIODIC:
            curl1_re = tl.where(last_y, 0.0, curl1_re)
            curl1_im = tl.where(last_y, 0.0, curl1_im)
        if BCZ == MIRROR_PERIODIC:
            curl2_re = tl.where(last_z, 0.0, curl2_re)
            curl2_im = tl.where(last_z, 0.0, curl2_im)
    else:
        # Bx:(0,1,1)  By:(1,0,1)  Bz:(1,1,0) -- shift 1 on the two OTHER axes.
        if BCY == MIRROR_PERIODIC:
            curl0_re = tl.where(last_y, 0.0, curl0_re)
            curl0_im = tl.where(last_y, 0.0, curl0_im)
        if BCZ == MIRROR_PERIODIC:
            curl0_re = tl.where(last_z, 0.0, curl0_re)
            curl0_im = tl.where(last_z, 0.0, curl0_im)
        if BCX == MIRROR_PERIODIC:
            curl1_re = tl.where(last_x, 0.0, curl1_re)
            curl1_im = tl.where(last_x, 0.0, curl1_im)
        if BCZ == MIRROR_PERIODIC:
            curl1_re = tl.where(last_z, 0.0, curl1_re)
            curl1_im = tl.where(last_z, 0.0, curl1_im)
        if BCX == MIRROR_PERIODIC:
            curl2_re = tl.where(last_x, 0.0, curl2_re)
            curl2_im = tl.where(last_x, 0.0, curl2_im)
        if BCY == MIRROR_PERIODIC:
            curl2_re = tl.where(last_y, 0.0, curl2_re)
            curl2_im = tl.where(last_y, 0.0, curl2_im)

    # --- the PML forms (stepping._apply_curl_regions), per part ----------------
    # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). One record
    # select per target serves both parts; fu is read and written on the
    # recurrence's cells only (the auxiliary contract), both words.
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    if BACKWARD:
        lattice = 0
    else:
        lattice = RECORD_HALF_INTEGER
    du0 = record_in(cb, lattice + 8, k)
    du1 = record_in(cb, lattice + 0, i)
    du2 = record_in(cb, lattice + 4, j)

    p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)
    p_im = tl.load(u0 + 2 * idx + 1, mask=live & du0, other=0.0)
    v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=live, other=0.0), p_re, curl0_re,
                             km_y, si_y, km_z, si_z, du0)
    v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl0_im, km_y, si_y, km_z, si_z, du0)

    p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
    p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
    v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=live, other=0.0), p_re, curl1_re,
                             km_z, si_z, km_x, si_x, du1)
    v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl1_im, km_z, si_z, km_x, si_x, du1)

    p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
    p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
    v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=live, other=0.0), p_re, curl2_re,
                             km_x, si_x, km_y, si_y, du2)
    v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl2_im, km_x, si_x, km_y, si_y, du2)

    # --- the special-kz beta pass (stepping._special_kz_beta_term, then
    # stepping._apply_folded_increment), per part ------------------------------
    # MEEP's pass of its own after every component's curl (step_db.cpp:161-175),
    # applied to the registers the forms returned. The partners are the CENTER
    # words as loaded, never connected: target 0 takes the second source at sign
    # +1, target 1 the first at sign -1; target 2 gets nothing. The increment is
    # the negated quarter turn (part_products.quarter_turn): i*c*(x + iy) is
    # (-(c*y), c*x), so its negation is (c*y, -(c*x)), the two negations of the
    # real part cancelling exactly. The words beta_plus/beta_minus already carry
    # the side's i or -i (special_kz.beta_curl_coefficients). Masked to (+0, +0)
    # on the cells the curl's own mask drops; each part takes the beta ladder.
    if HAS_BETA:
        q0_re = beta_plus * b_im
        q0_im = (beta_plus * b_re) * -1.0
        q1_re = beta_minus * a_im
        q1_im = (beta_minus * a_re) * -1.0
        if BACKWARD:
            if BCY != PERIODIC:
                q0_re = tl.where(at_y, 0.0, q0_re)
                q0_im = tl.where(at_y, 0.0, q0_im)
            if BCZ != PERIODIC:
                q0_re = tl.where(at_z, 0.0, q0_re)
                q0_im = tl.where(at_z, 0.0, q0_im)
            if BCX != PERIODIC:
                q1_re = tl.where(at_x, 0.0, q1_re)
                q1_im = tl.where(at_x, 0.0, q1_im)
            if BCZ != PERIODIC:
                q1_re = tl.where(at_z, 0.0, q1_re)
                q1_im = tl.where(at_z, 0.0, q1_im)
            if BCX == MIRROR_PERIODIC:
                q0_re = tl.where(last_x, 0.0, q0_re)
                q0_im = tl.where(last_x, 0.0, q0_im)
            if BCY == MIRROR_PERIODIC:
                q1_re = tl.where(last_y, 0.0, q1_re)
                q1_im = tl.where(last_y, 0.0, q1_im)
        else:
            if BCX != PERIODIC:
                q0_re = tl.where(at_x, 0.0, q0_re)
                q0_im = tl.where(at_x, 0.0, q0_im)
            if BCY != PERIODIC:
                q1_re = tl.where(at_y, 0.0, q1_re)
                q1_im = tl.where(at_y, 0.0, q1_im)
            if BCY == MIRROR_PERIODIC:
                q0_re = tl.where(last_y, 0.0, q0_re)
                q0_im = tl.where(last_y, 0.0, q0_im)
            if BCZ == MIRROR_PERIODIC:
                q0_re = tl.where(last_z, 0.0, q0_re)
                q0_im = tl.where(last_z, 0.0, q0_im)
            if BCX == MIRROR_PERIODIC:
                q1_re = tl.where(last_x, 0.0, q1_re)
                q1_im = tl.where(last_x, 0.0, q1_im)
            if BCZ == MIRROR_PERIODIC:
                q1_re = tl.where(last_z, 0.0, q1_re)
                q1_im = tl.where(last_z, 0.0, q1_im)
        v0_re, n0_re = fold_ladder(v0_re, n0_re, q0_re, si_y, si_z, du0)
        v0_im, n0_im = fold_ladder(v0_im, n0_im, q0_im, si_y, si_z, du0)
        v1_re, n1_re = fold_ladder(v1_re, n1_re, q1_re, si_z, si_x, du1)
        v1_im, n1_im = fold_ladder(v1_im, n1_im, q1_im, si_z, si_x, du1)

    # --- stores: fu (recurrence cells only), then the fields, both parts --------
    tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
    tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
    tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
    tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
    tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
    tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


# ---------------------------------------------------------------------------
# Fold classification — RESTATED from symmetry.py's discipline, not imported
# ---------------------------------------------------------------------------

def _stored_past_owned_reader():
    """``stepping._stored_past_owned``, or None when ``stepping`` will not import."""
    try:
        from ..stepping import _stored_past_owned  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no stepping, no coverage
        return None
    return _stored_past_owned


def _far_reflect_rows(grid: Any) -> Optional[Tuple[Optional[int], ...]]:
    """``stepping._far_reflect_rows`` (:1661-1690), or None when unreadable.

    Read from the engine, NEVER computed here: the row is ``n_full - stored + 2``,
    which is ``stored - 2`` at an even full count and ``stored - 3`` at an odd
    one, and a second implementation of that is a second place to get it wrong.
    """
    try:
        from ..stepping import _far_reflect_rows as reader  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    try:
        return tuple(reader(grid))
    except Exception:  # noqa: BLE001 - an unanswerable grid is refused, not crashed on
        return None


def folded_axis_kinds(grid: Any, pml: Any) -> Tuple[Optional[Tuple[int, int, int]],
                                                    Tuple[str, ...]]:
    """The per-axis boundary constexprs, or ``None`` plus the reasons.

    THE CLASSIFICATION IS THE SINGLE POINT OF FAILURE IN THIS FILE, so it is
    derived from the engine's own functions and cross-checked against a second
    one rather than re-derived:

    * the ghost rule comes from ``stepping._boundary_kinds`` — the fold OUTRANKS
      the declaration there, and a configuration it will not resolve (a PML on
      the plane face, a folded axis the layer thinks wraps) raises, which is a
      refusal (``_require_consistent_pml``, S:2250-2288);
    * a folded axis is ``MIRROR_PERIODIC`` iff ``stepping._stored_past_owned``
      says the stored array carries the slot past MEEP's owned window, and
      ``MIRROR_METALLIC`` otherwise;
    * ``grid.is_metallic`` must AGREE with that split. It is a second,
      independent route to the same fact (``Grid.stored_cells`` adds its extra
      slot exactly when the axis is mirrored and not metallic), so a
      disagreement means one of them has drifted and NEITHER may be trusted.

    Returned as ``(codes, reasons)`` rather than raising: a plan builder turns a
    refusal into ``None`` and the array path steps the run.
    """
    reasons: List[str] = []
    kinds = _boundary_kinds(grid, pml)
    if kinds is None:
        return None, ("boundary kinds could not be resolved for this grid",)

    owned = getattr(grid, "owned_cells", None)
    stored_past_owned = _stored_past_owned_reader()
    if stored_past_owned is None:
        return None, ("stepping._stored_past_owned is not importable; the "
                      "MIRROR_METALLIC / MIRROR_PERIODIC split cannot be derived",)

    codes: List[int] = []
    for axis, kind in enumerate(kinds):
        if kind == "periodic":
            codes.append(CODE_PERIODIC)
            continue
        if kind == "metallic":
            codes.append(CODE_METALLIC)
            continue
        if kind != "mirror":
            codes.append(CODE_PERIODIC)
            reasons.append(f"axis {axis} boundary {kind!r} is outside "
                           f"{BOUNDARY_KIND_NAMES}")
            continue

        # A folded axis. Everything below is a requirement, not a description.
        if not callable(owned):
            reasons.append(
                f"axis {axis} is folded but the grid exposes no owned_cells(); "
                f"_stored_past_owned would answer False for a folded PERIODIC "
                f"axis and the kernel would drop its top-plane mask")
            codes.append(CODE_PERIODIC)
            continue
        if _call(grid, "mirror_phase", axis, default=None) not in (1, -1):
            reasons.append(
                f"axis {axis} is folded but its mirror phase is "
                f"{_call(grid, 'mirror_phase', axis, default=None)!r}, "
                f"not +1 or -1")
        stored = _call(grid, "stored_cells", axis, default=None)
        if stored is None or int(stored) <= MIRROR_SOURCE_INDEX:
            reasons.append(
                f"axis {axis} is folded with {stored!r} stored cells; the near "
                f"ghost images stored cell {MIRROR_SOURCE_INDEX}")
        try:
            past_owned = bool(stored_past_owned(grid, axis))
        except Exception as exc:  # noqa: BLE001 - an unanswerable axis is refused
            reasons.append(f"axis {axis}: _stored_past_owned raised {exc!r}")
            codes.append(CODE_PERIODIC)
            continue
        declared_metallic = bool(_call(grid, "is_metallic", axis, default=False))
        if past_owned == declared_metallic:
            reasons.append(
                f"axis {axis}: _stored_past_owned={past_owned} and "
                f"is_metallic={declared_metallic} disagree about the fold's "
                f"termination; the two routes to the same fact have drifted")
        codes.append(CODE_MIRROR_PERIODIC if past_owned else CODE_MIRROR_METALLIC)

    if reasons:
        return None, tuple(reasons)
    return (codes[0], codes[1], codes[2]), ()


def _has_real_fold(grid: Any) -> bool:
    """Whether this grid really owns a mirror fold, not merely fold-capable rules."""
    return bool(_call(grid, "has_symmetry", default=False)) and any(
        bool(_call(grid, "is_mirrored", axis, default=False)) for axis in range(3))


def _fold_clause_reasons(grid: Any, pml: Any,
                         codes: Optional[Sequence[int]]) -> List[str]:
    """The fold-specific refusals every predicate here shares, by name.

    Reflect-row bounds are CHECKED, never assumed: the far fill must not image
    the plane it writes, and must not read outside the allocation. The read/write
    plane disjointness of the near fill (0 vs 2) is the stored-cell clause above.
    """
    reasons: List[str] = []
    if codes is None:
        return reasons
    rows = _far_reflect_rows(grid)
    shape = tuple(getattr(grid, "shape", ()))
    for axis, code in enumerate(codes):
        if code != CODE_MIRROR_PERIODIC:
            continue
        row = rows[axis] if rows is not None else None
        if row is None:
            reasons.append(f"axis {axis} is a folded PERIODIC axis with no "
                           f"reflect row from stepping._far_reflect_rows")
        elif len(shape) == 3 and not (0 <= int(row) < int(shape[axis]) - 1):
            reasons.append(
                f"axis {axis} reflect row {row!r} is outside "
                f"[0, {int(shape[axis]) - 1}); the far ghost would image the "
                f"plane it writes, or read out of the allocation")
    return reasons


def _cylindrical_reasons(grid: Any) -> List[str]:
    """Cylindrical is refused by name: the r = 0 ghost is ``r_to_minus_r`` and the
    axis row belongs to the per-m rules — ``_boundary_kinds`` would report
    ``"axis"``, which this file's classification does not carry."""
    reasons: List[str] = []
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")
    return reasons


def _absorber_reasons(pml: Any) -> List[str]:
    """No active PML means the plain path is the bit-identical one; this whole
    tranche is the split-field product. A PML on the LOW face of a folded axis, or
    one that thinks a folded axis wraps, raises inside
    ``stepping._require_consistent_pml`` (S:2250-2288) — a refusal, not a crash —
    and surfaces here through ``_boundary_kinds`` returning None."""
    if pml is None or not getattr(pml, "is_active", False):
        return ["no active PML layer (this tranche implements the split-field "
                "path only)"]
    return []


def _media_reasons(fields: Any, grid: Any, targets: Sequence[str]) -> List[str]:
    """Conductivity, dispersion, nonlinearity, off-diagonal epsilon, BFAST.

    A missing or non-callable ``condfac_for`` is refused OUTRIGHT — the complex
    tranche's stricter reading (complex_fields.py:186-201): inferring "no
    conductivity" from the ABSENCE of the reader is admission by attribute
    absence, the exact reasoning coverage exists to refuse.
    """
    reasons: List[str] = []
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    else:
        for target in targets:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}: this tranche "
                    f"transcribes the plain split-field recurrence only, and "
                    f"conductivity.py's own predicates refuse beta — no silent "
                    f"overlap in either direction")
    if getattr(fields, "has_magnetic_conductivity", False):
        reasons.append("a magnetic (B) conductivity is installed")
    if getattr(fields, "has_nonlinearity", False):
        reasons.append(
            "chi2/chi3 is installed: the Pade constitutive factor is not "
            "carried, the transverse sums shift component PRODUCTS which carry "
            "no single parity, and driver._require_folded_far_face_is_quiet "
            "refuses a nonlinear fold with a live far face")
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed: the row product reads "
            "neighbours through a transverse Yee average (stepping.py:1219-1254), "
            "and in REAL storage _special_kz_beta_term (S:773-783) raises on the "
            "same pairing, MEEP fields.cpp:548-549's own refusal")
    reasons.extend(_susceptibility_reasons(fields))
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (a second additive curl term, refused in "
                       "both directions)")
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")
    return reasons


def _complex_phase_reasons(grid: Any, kinds: Optional[Sequence[str]],
                           codes: Optional[Sequence[int]]) -> List[str]:
    """Per-axis Bloch legality, INCLUDING the fold clause this tranche adds.

    A phased axis must resolve PERIODIC (stepping._bloch_phases raises there);
    a metallic axis must carry k = 0; an unreadable phase is refused, never
    admitted as unphased (complex_fields.py:144-148's measured lesson); and — the
    fold clause — NO axis carrying a mirror code may carry any nonzero k
    component or any phase, INCLUDING the Brillouin zone edge, because
    ``driver._require_bloch_is_representable`` (driver.py:1040-1120) refuses the
    configuration outright and a kernel cannot lift what the array path will not
    run.
    """
    reasons: List[str] = []
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if kinds is None:
        return reasons
    phase_reader = getattr(grid, "bloch_phase", None)
    if not callable(phase_reader):
        reasons.append("grid.bloch_phase is missing or not callable; an "
                       "unreadable phase table is not an unphased one")
    for axis, kind in enumerate(kinds):
        folded = codes is not None and codes[axis] in (CODE_MIRROR_METALLIC,
                                                       CODE_MIRROR_PERIODIC)
        if callable(phase_reader):
            try:
                phase = phase_reader(axis)
            except Exception as exc:  # noqa: BLE001 - unreadable is refused
                reasons.append(f"grid.bloch_phase({axis}) raised {exc!r}; an "
                               f"unreadable phase is not an unphased one")
            else:
                if phase is not None and kind != "periodic":
                    reasons.append(
                        f"axis {axis} carries Bloch phase {phase!r} but "
                        f"resolved to {kind!r}; only a periodic wrap can carry "
                        f"a phase")
                if phase is not None and folded:
                    reasons.append(
                        f"axis {axis} is folded and carries Bloch phase "
                        f"{phase!r}; driver._require_bloch_is_representable "
                        f"refuses a nonzero k on a mirror plane's own axis, "
                        f"zone EDGE included")
        if kind == "metallic" and float(k_point[axis]) != 0.0:
            reasons.append(
                f"axis {axis} is metallic with k component {k_point[axis]!r}; a "
                f"PEC wall gives the axis no lattice vector for the phase")
        if folded and float(k_point[axis]) != 0.0:
            reasons.append(
                f"axis {axis} is folded with k component {k_point[axis]!r}; "
                f"k must be EXACTLY 0 on every folded axis (zone edge included)")
    return reasons


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# ADMITTED, in one sentence: a Cartesian CuPy run under an ACTIVE split-field
# PML, in complex64 storage (K1/K2/K3b) or real float32 (K3a), with one, two or
# three axes folded by an even or odd mirror plane over a PERIODIC or METALLIC
# outer declaration at either count parity, with k = 0 EXACTLY on every folded
# axis and any Bloch phase confined to unfolded PERIODIC axes, with grid.beta
# zero (K1/K2) or nonzero (K3a/K3b), no susceptibility on the curl beyond
# electric lorentzian/drude poles, stored E, all six PML auxiliaries and both
# source families allocated, C-contiguous grid.shape layout, PML coefficient
# vectors at the STORED extent.


def _complex_fold_grid_reasons(fields: Any, pml: Any, grid: Any,
                               targets: Sequence[str]) -> Tuple[List[str],
                                                                Optional[Tuple[int, int, int]]]:
    """The clauses every COMPLEX-storage predicate here shares, plus the codes."""
    reasons: List[str] = []

    # 1. CuPy backend. The kernels launch against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2 (INVERTED, as the complex tranche inverts it). Complex64 storage REQUIRED.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32 (neither force_complex_fields nor a nonzero "
            "k_point): a real folded run belongs to symmetry.py's kernels or to "
            "this module's real beta kernel")

    # 3. An absorber that actually absorbs.
    reasons.extend(_absorber_reasons(pml))

    # 4/5. Cylindrical, then the fold classification and its reflect rows.
    reasons.extend(_cylindrical_reasons(grid))
    active_pml = pml if (pml is not None and getattr(pml, "is_active", False)) else None
    kinds = _boundary_kinds(grid, active_pml)
    codes, fold_reasons = folded_axis_kinds(grid, active_pml)
    reasons.extend(fold_reasons)
    reasons.extend(_fold_clause_reasons(grid, active_pml, codes))

    # 6. Per-axis Bloch legality, fold clause included.
    reasons.extend(_complex_phase_reasons(grid, kinds, codes))

    # 7-9. Media, dispersion, nonlinearity, BFAST, stored E.
    reasons.extend(_media_reasons(fields, grid, targets))
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: complex-storage ADE is a future "
            "tranche, and no folded complex corpus row carries one")

    return reasons, codes


def folded_complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May :func:`folded_bloch_pml_curl_step` (K1) step this configuration?

    ZERO FOLDED AXES IS ADMITTED here, deliberately and for ``symmetry.py``'s
    reason: with every axis resolving to PERIODIC/METALLIC the constexpr branches
    reduce to ``complex_fields.bloch_pml_curl_step``'s exactly, and the gate
    MEASURES that reduction rather than claiming it. Which of the two a composed
    plan would launch is an integration decision, not a coverage one, and the
    composition verdict below refuses the unfolded case by name.

    ``grid.beta`` must be ZERO: a beta run belongs to K3a/K3b, and selecting
    between two valid products by branch order would make the numerical method
    depend on composer order.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    reasons, _codes = _complex_fold_grid_reasons(fields, pml, grid, CURL_TARGETS)

    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(
            f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero: that "
            f"run belongs to the folded beta kernels, not to this one")

    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))
    return Coverage(not reasons, tuple(reasons))


def folded_complex_composition_curl_coverage(fields: Any, pml: Any,
                                             sub_step: str) -> Coverage:
    """K1's verdict for COMPOSITION, where an actual fold is mandatory.

    :func:`folded_complex_pml_curl_coverage` admits an unfolded grid so its
    standalone gate can prove reduction to the certified unfolded kernel. That is
    an equivalence product, not a routing rule.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    base = folded_complex_pml_curl_coverage(fields, pml, sub_step)
    reasons = list(base.reasons)
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: the folded complex curl is an "
                       "equivalence product here, not a composition candidate")
    return Coverage(not reasons, tuple(reasons))


def folded_mirror_ghost_fill_complex_coverage(fields: Any, family: str) -> Coverage:
    """May :func:`folded_mirror_ghost_fill_complex` (K2) write this family?

    Narrower than the curl's predicate and independent of it: this sub-step reads
    no coefficient, no source and no PML, so it has no Courant number, no
    sub-lattice and no absorber clause. What it needs is complex64 storage, a
    real fold, a readable phase, a stored extent that can hold both planes, and a
    reflect row on every folded PERIODIC axis.

    Refused entirely on a run with no mirror plane: with nothing folded the array
    path's fill passes return immediately and there is nothing to replace.
    """
    if family not in GHOST_FILL_FAMILIES:
        raise ValueError(f"family must be one of {tuple(GHOST_FILL_FAMILIES)}, "
                         f"got {family!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32: the real fold's fill is "
            "symmetry.mirror_ghost_fill, which is EXACT there (+-1 * float32 is "
            "an exact sign flip including -0.0) and is not replaced here")
    reasons.extend(_cylindrical_reasons(grid))
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: the array path's fill passes "
                       "return immediately and there is nothing to replace")

    codes, fold_reasons = folded_axis_kinds(grid, None)
    reasons.extend(fold_reasons)
    reasons.extend(_fold_clause_reasons(grid, None, codes))
    if codes is not None and not any(
            code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC) for code in codes):
        reasons.append("no axis resolves to a mirror boundary")

    # A folded axis may carry no phase and no k at all; the fill carries no wrap
    # factor (stepping._far_reflect_rows:1677-1681 states the same).
    kinds = _boundary_kinds(grid, None)
    reasons.extend(_complex_phase_reasons(grid, kinds, codes))

    names = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    return Coverage(not reasons, tuple(reasons))


def folded_complex_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the CERTIFIED complex constitutive kernel step this folded run?

    The element-wise contract RESTATED with the fold clause replaced by
    :func:`folded_axis_kinds` — never reached by subtracting reasons from another
    predicate's output. ``_apply_constitutive_pml`` (S:2065-2096) reads only
    (field, source, fw) at the same cell with per-axis coefficient vectors, and
    ``update_H``/``update_E`` run over the WHOLE stored array with no ownership
    mask, so the fold contributes only its stored extent — which the PML
    coefficient vectors already carry. The kernel launched is
    ``complex_fields.bloch_constitutive_step``.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons, codes = _complex_fold_grid_reasons(fields, pml, grid, CURL_TARGETS)
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: the folded constitutive "
                       "product has no array-path work to specialize")
    elif codes is not None and not any(
            code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC) for code in codes):
        reasons.append("no axis resolves to a mirror boundary")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        # inv_eps stays float32 under complex storage (stepping.py:37-38,
        # fields.py:1203-1204), so coverage.py's own float32 pin is exactly right.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))
    return Coverage(not reasons, tuple(reasons))


def folded_beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May :func:`folded_beta_bloch_pml_curl_step` (K3b) step this configuration?

    K1's clause set with the beta clause INVERTED (beta must be NONZERO — a
    beta = 0 run belongs to K1). The effective-2-D clause is restated from
    ``special_kz`` rather than inferred from the Grid constructor's guard.

    A FOLD IS MANDATORY, for K3a's reason: without it this predicate and
    ``special_kz.beta_bloch_pml_curl_coverage`` were MEASURED to admit the same
    unfolded complex beta grid, and no equivalence leg needs the unfolded case
    here (``run_identity``'s HAS_BETA=0 reduction runs on a FOLDED grid).
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    reasons, _codes = _complex_fold_grid_reasons(fields, pml, grid, CURL_TARGETS)
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: an UNFOLDED complex beta run "
                       "belongs to special_kz.beta_bloch_pml_curl_step, whose "
                       "predicate admits it — this product may not claim the "
                       "same territory")

    if float(getattr(grid, "beta", 0.0)) == 0.0:
        reasons.append("grid.beta is zero: this product exists only for "
                       "special_kz runs; a beta = 0 folded complex run belongs "
                       "to the plain folded complex kernel")
    if int(getattr(grid, "dimensions", 0)) != 2:
        reasons.append(f"grid dimensions={getattr(grid, 'dimensions', None)!r} "
                       f"is not the effective-2-D grid beta requires "
                       f"(grid.py:668-697; MEEP fields.cpp:546-547)")
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if len(k_point) > 2 and float(k_point[2]) != 0.0:
        reasons.append(
            f"k_point component z = {k_point[2]!r} rides the invariant axis "
            f"whose dependence beta already carries analytically "
            f"(grid.py:668-697 refuses the pairing at construction)")

    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))
    return Coverage(not reasons, tuple(reasons))


def _real_fold_grid_reasons(fields: Any, pml: Any, grid: Any,
                            targets: Sequence[str]) -> Tuple[List[str],
                                                             Optional[Tuple[int, int, int]]]:
    """The clauses the REAL folded beta family shares."""
    reasons: List[str] = []

    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # Real storage REQUIRED: a complex-storage beta fold belongs to K3b.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: a complex-storage folded beta "
                       "run belongs to the folded complex beta kernel")
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}: real "
                       f"storage carries no Bloch phase, and the array path "
                       f"raises on the pairing")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    reasons.extend(_absorber_reasons(pml))
    reasons.extend(_cylindrical_reasons(grid))
    active_pml = pml if (pml is not None and getattr(pml, "is_active", False)) else None
    codes, fold_reasons = folded_axis_kinds(grid, active_pml)
    reasons.extend(fold_reasons)
    reasons.extend(_fold_clause_reasons(grid, active_pml, codes))
    reasons.extend(_media_reasons(fields, grid, targets))

    if int(getattr(grid, "dimensions", 0)) != 2:
        reasons.append(f"grid dimensions={getattr(grid, 'dimensions', None)!r} "
                       f"is not the effective-2-D grid beta requires "
                       f"(grid.py:668-697; MEEP fields.cpp:546-547)")
    if float(getattr(grid, "beta", 0.0)) == 0.0:
        reasons.append("grid.beta is zero: this product exists only for "
                       "special_kz runs; a beta = 0 folded real run belongs to "
                       "symmetry.pml_curl_step_folded")
    return reasons, codes


def folded_beta_pml_curl_coverage(fields: Any, pml: Any,
                                  sub_step: str) -> Coverage:
    """May :func:`folded_beta_pml_curl_step` (K3a) step this configuration?

    The folded REAL clause set with the beta clause INVERTED.

    A FOLD IS MANDATORY, unlike K1's standalone verdict. K1 admits an unfolded
    grid deliberately so its gate can prove reduction to the certified unfolded
    kernel, and pushes the routing question into
    :func:`folded_complex_composition_curl_coverage`. This predicate has no such
    equivalence leg — ``symmetry.pml_curl_step_folded`` is the beta = 0 folded
    real product and ``special_kz.beta_pml_curl_step`` is the UNFOLDED beta one —
    so admitting an unfolded grid claims territory ``special_kz`` already owns.
    MEASURED before the clause was added: on ``Grid(cell_size=(1.6,1.6,0.0),
    dimensions=2, boundaries='periodic', beta=0.2)`` with no symmetry, this
    predicate and ``special_kz.beta_pml_curl_coverage`` BOTH returned an empty
    reason list — the admitted overlap the module's own wiring note argues cannot
    arise.

    The conductivity clause is scoped to ALL SIX curl targets, not the named
    sub-step's three. Every sibling predicate in this file passes
    ``CURL_TARGETS``; scoping it narrower here admitted an ELECTRIC conductivity
    for ``step_B`` that every other predicate refuses for the same run, so a
    composed run could only ever be half-substituted.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    reasons, _codes = _real_fold_grid_reasons(fields, pml, grid, CURL_TARGETS)
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: an UNFOLDED beta run belongs "
                       "to special_kz.beta_pml_curl_step, whose predicate admits "
                       "it — this product may not claim the same territory")
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))
    return Coverage(not reasons, tuple(reasons))


def folded_beta_run_constitutive_coverage(fields: Any, pml: Any,
                                          side: str) -> Coverage:
    """May the CERTIFIED real constitutive kernel step a FOLDED BETA run?

    The folded real element-wise contract with ONLY the beta clause dropped —
    ``special_kz``'s grouping-choice-5 discipline. The constitutive sub-steps read
    nothing beta-dependent and nothing fold-dependent beyond the stored extent, so
    admission delegates the ARITHMETIC to ``kernels.constitutive_step`` unchanged.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons, codes = _real_fold_grid_reasons(fields, pml, grid, CURL_TARGETS)
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: the folded constitutive "
                       "product has no array-path work to specialize")
    elif codes is not None and not any(
            code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC) for code in codes):
        reasons.append("no axis resolves to a mirror boundary")

    if side == "E" and (getattr(fields, "has_polarizations", False)
                        or (getattr(fields, "polarizations", ()) or ())):
        reasons.append(
            "a susceptibility is registered: update_E's source is (D - sum P), "
            "not D — that configuration belongs to the ADE kernel")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))
    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The parity coefficients and the per-axis fill entries
# ---------------------------------------------------------------------------

def mirror_parity_coefficients(phase: int) -> Tuple[Tuple[float, float],
                                                    Tuple[float, float]]:
    """The ((near_re, near_im), (far_re, far_im)) word pairs for one folded axis.

    ``fields.mirror_parity(c, axis, phase) == phase * (1 - 2*iyee[c][axis])``
    (fields.py:117), so the near fill's shift-0 components take ``+phase`` and
    the far fill's shift-1 components take ``-phase``. Each is rounded through
    ``numpy.complex64`` HERE, on the host, exactly once, and passed — never
    synthesised in-kernel (special_kz.py:255-264's rule). ``float()`` of a
    numpy.float32 preserves the zero's sign and Triton types a Python float
    argument as fp32, so the words the kernel receives are the array path's bits.

    ONE PROPERTY OF THIS FUNCTION IS LOAD-BEARING ELSEWHERE. ``complex64(±1)``
    has an imaginary word of BITWISE ``0x00000000`` — not merely equal to zero,
    but the exact pattern a literal ``+0.0`` produces — at both mirror phases and
    for the near and the far coefficient alike. The byte gate's
    ``synthesize_zero_imag`` source mutation, which rewrites the kernel's
    ``near_im`` / ``far_im`` argument to that literal, is therefore VACUOUS BY
    CONSTRUCTION (mutant and shipped kernel receive identical operands) and is
    recorded there as a structural null; the live "the word is READ, never
    synthesised" claim is carried by the host mutation
    ``imaginary_words_are_read_not_synthesised``, which passes a NONZERO word.
    If this ever returns a signed zero or a nonzero imaginary word that
    retirement is wrong, and ``test_triton_folded_complex.py::
    test_the_parity_coefficients_imaginary_word_is_bitwise_zero`` — which asserts
    the property directly — fails on a laptop the moment it stops holding.
    """
    import numpy  # noqa: PLC0415

    if int(phase) not in (1, -1):
        raise ValueError(f"a mirror plane's declared phase is +1 or -1, got {phase!r}")
    out: List[Tuple[float, float]] = []
    for value in (int(phase), -int(phase)):
        rounded = numpy.complex64(value)
        out.append((float(numpy.float32(rounded.real)),
                    float(numpy.float32(rounded.imag))))
    return out[0], out[1]


def ghost_fill_axis_entries(grid: Any, family: str) -> Tuple[Dict[str, Any], ...]:
    """One entry per folded axis, in X, Y, Z order — the kernel's arguments.

    Public and separate from the plan builder so it can be checked on a machine
    with no Triton and no GPU, which is where the three things that silently go
    wrong here live: the X, Y, Z ORDER (load-bearing under complex storage — see
    the module docstring), the REFLECT ROW (``stored - 2`` at an even full count,
    ``stored - 3`` at an odd one), and the PARITY COEFFICIENT WORDS.
    """
    if family not in GHOST_FILL_FAMILIES:
        raise ValueError(f"family must be one of {tuple(GHOST_FILL_FAMILIES)}, "
                         f"got {family!r}")
    codes, reasons = folded_axis_kinds(grid, None)
    if codes is None:
        raise ValueError("this grid's folded axes cannot be classified: "
                         + "; ".join(reasons))
    rows = _far_reflect_rows(grid) or (None, None, None)
    targets = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    entries: List[Dict[str, Any]] = []
    for axis, code in enumerate(codes):  # X, Y, Z — the order the fills apply in.
        if code not in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC):
            continue
        far = code == CODE_MIRROR_PERIODIC
        phase = int(grid.mirror_phase(axis))
        near_words, far_words = mirror_parity_coefficients(phase)
        entries.append({
            "axis": axis,
            "phase": phase,
            "far": far,
            "near_words": near_words,
            "far_words": far_words,
            # -1 is never read when FAR is 0; it is passed rather than left unset
            # so the kernel signature is one shape.
            "reflect_row": int(rows[axis]) if far and rows[axis] is not None else -1,
            "shifts": tuple(TARGET_IYEE[name][axis] for name in targets),
        })
    return tuple(entries)


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class FoldedComplexPmlCurlPlan:
    """A launchable, allocation-free FOLDED COMPLEX split-field PML curl sub-step.

    Deliberately a sibling of ``complex_fields.ComplexPmlCurlPlan`` rather than a
    subclass: the two hold the same bindings but launch different kernels, and a
    subclass inheriting ``run`` would launch the certified unfolded one. Built two
    ways — from the engine's own objects through the predicate, and from bare
    device arrays for the gate — and launched ONE way, so the bytes the gate
    certifies are the bytes the engine would launch.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "connections", "connection_values", "block",
                 "num_warps", "chunk_bounds", "_targets", "_aux", "_sources",
                 "_coefficients", "_cb", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, connections,
                 connection_values, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None, *, chunk_bounds: Any) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                             f"got {sub_step!r}")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        for axis, (code, kind) in enumerate(zip(self.bc, self.connections)):
            if kind and code != CODE_PERIODIC:
                raise ValueError(
                    f"axis {axis} carries a Bloch connection but resolved to code "
                    f"{code}; only a periodic wrap carries a phase")
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "FoldedComplexPmlCurlPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # The override exists for exactly one caller: the gate's mutation legs.
        # Dropping it is not a slowdown, it is a DISARMING.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. ``guard`` is the gate's, not a caller's."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else folded_bloch_pml_curl_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.connection_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            CNX=self.connections[0], CNY=self.connections[1], CNZ=self.connections[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"FoldedComplexPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, connections={self.connections}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_folded_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                                 block: Optional[int] = None,
                                 num_warps: Optional[int] = None,
                                 chunk_layout: Any = None
                                 ) -> Optional[FoldedComplexPmlCurlPlan]:
    """Build a K1 plan from the engine's own objects, or None when out of coverage.

    None is the only refusal (never raise into a caller that would otherwise have
    stepped correctly).
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not folded_complex_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    kinds = resolve(grid, pml)
    phases = bloch_phase_table(grid, kinds)
    connections, values = connection_arguments(phases, backward=bool(spec["backward"]))
    return FoldedComplexPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, connections, values,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_folded_complex_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                             flat: Dict[str, Any], codes,
                                             phases: Sequence[Optional[complex]],
                                             dtdx: float,
                                             block: Optional[int] = None,
                                             kernel: Any = None,
                                             num_warps: Optional[int] = None,
                                             *, chunk_bounds: Any,
                                             ) -> FoldedComplexPmlCurlPlan:
    """Build a K1 plan from bare device arrays — the gate's route.

    ``codes`` is the four-valued per-axis code triple; the direction's connection
    factor is formed HERE, per sub-step, exactly as the engine route forms it.
    ``chunk_bounds`` is the chunk record, with no default (a mutation leg that fell
    back to some default division would stop measuring the forms). No predicate
    runs: the caller is a harness that constructed the configuration deliberately,
    including the deliberately wrong ones, and ``kernel=`` carries the mutation
    override.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(spec["backward"]))
    return FoldedComplexPmlCurlPlan(
        sub_step, shape, dtdx, codes, connections, values,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )


class FoldedMirrorGhostFillComplexPlan:
    """One family's COMPLEX mirror ghost fills, as TWO PASSES over the folded axes.

    :meth:`run_near` is ``stepping.fill_symmetry_bc_*`` and :meth:`run_far` is
    ``stepping.fill_folded_far_ghosts_*``; each walks the folded axes in X, Y, Z
    order, one launch per axis. :meth:`run` is near-then-far, which is the array
    path's order — but the driver puts ``zero_metal_*`` BETWEEN the two passes
    (driver.py:3209-3211 / :3222-3224), so a composed plan must call the two
    halves separately and must not fuse across that seam.

    THE SPLIT FOLLOWS THE DRIVER, not the arithmetic. Per stored part every fill
    is ``+-1.0`` times the word, exact, so near-then-far and far-then-near store the
    same words (the gate's ``fuse_fill_passes`` and ``reverse_axis_order`` are
    structural nulls, as ``symmetry.MirrorGhostFillPlan``'s fused passes are for
    real storage); the passes stay separate because the driver's wall clear sits
    between them.
    """

    __slots__ = ("family", "shape", "block", "axes", "num_warps",
                 "_targets", "_kernel")

    def __init__(self, family: str, shape, block: int, targets,
                 axes: Sequence[Dict[str, Any]], kernel=None,
                 num_warps: Optional[int] = None) -> None:
        if family not in GHOST_FILL_FAMILIES:
            raise ValueError(f"family must be one of {tuple(GHOST_FILL_FAMILIES)}, "
                             f"got {family!r}")
        self.family = family
        self.shape = tuple(int(n) for n in shape)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self.axes = tuple(dict(entry) for entry in axes)
        self._kernel = kernel

    def _launch_pass(self, do_near: int, do_far: int,
                     guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = (self._kernel if self._kernel is not None
                  else folded_mirror_ghost_fill_complex)
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        for entry in self.axes:            # X, Y, Z — the order the array path uses
            far = do_far and bool(entry["far"])
            if not do_near and not far:
                continue
            axis = int(entry["axis"])
            n_plane = (ny * nz, nx * nz, nx * ny)[axis]
            grid = ((n_plane + self.block - 1) // self.block,)
            # The parity is a real scale per part (part_products.scale): the
            # coefficient's real word; its imaginary word is +0 and is not read.
            near_c = float(entry["near_words"][0])
            far_c = float(entry["far_words"][0])
            kernel[grid](
                *self._targets,
                nx, ny, nz, n_plane, int(entry["reflect_row"]),
                near_c, far_c,
                AXIS=axis,
                DO_NEAR=1 if do_near else 0,
                DO_FAR=1 if far else 0,
                S0=int(entry["shifts"][0]), S1=int(entry["shifts"][1]),
                S2=int(entry["shifts"][2]),
                BLOCK=self.block,
                enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
                **extra,
            )

    def run_near(self, guard: Optional[bool] = None) -> None:
        """``stepping.fill_symmetry_bc_*`` — every folded axis's NEAR plane."""
        self._launch_pass(1, 0, guard)

    def run_far(self, guard: Optional[bool] = None) -> None:
        """``stepping.fill_folded_far_ghosts_*`` — every folded PERIODIC axis's
        FAR plane. The driver runs ``zero_metal_*`` between this and
        :meth:`run_near`, so a composed plan calls the two halves separately."""
        self._launch_pass(0, 1, guard)

    def run(self, guard: Optional[bool] = None) -> None:
        """Both passes, near then far — the array path's order.

        Correct in isolation; a COMPOSED plan must call :meth:`run_near` and
        :meth:`run_far` around ``zero_metal_*`` instead, because the driver does.
        """
        self.run_near(guard)
        self.run_far(guard)

    def __repr__(self) -> str:
        return (f"FoldedMirrorGhostFillComplexPlan({self.family}, "
                f"shape={self.shape}, axes={[e['axis'] for e in self.axes]}, "
                f"block={self.block})")


def plan_folded_mirror_ghost_fill_complex(fields: Any, family: str,
                                          block: Optional[int] = None,
                                          num_warps: Optional[int] = None
                                          ) -> Optional[FoldedMirrorGhostFillComplexPlan]:
    """Build one family's K2 plan from the engine's own objects, or None."""
    if not folded_mirror_ghost_fill_complex_coverage(fields, family).covered:
        return None
    grid = fields.grid
    targets = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    axes = ghost_fill_axis_entries(grid, family)
    if not axes:  # pragma: no cover - the predicate already refused
        return None
    return FoldedMirrorGhostFillComplexPlan(
        family, grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets], axes, num_warps=num_warps)


def plan_folded_mirror_ghost_fill_complex_from_arrays(
        family: str, arrays: Dict[str, Any], axes: Sequence[Dict[str, Any]],
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = None) -> FoldedMirrorGhostFillComplexPlan:
    """Build a K2 plan from bare device arrays — the gate's route.

    ``axes`` is the entry list :func:`ghost_fill_axis_entries` builds; the gate
    supplies it directly so it can hand over a deliberately wrong reflect row, a
    reversed axis order or a plane-wise coefficient and watch that be caught.
    """
    targets = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return FoldedMirrorGhostFillComplexPlan(
        family, shape, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets], axes, kernel=kernel,
        num_warps=num_warps)


class FoldedBetaPmlCurlPlan:
    """A launchable, allocation-free FOLDED REAL beta PML curl sub-step (K3a).

    ``symmetry.FoldedPmlCurlPlan``'s bindings plus the two host-rounded beta words and
    ``HAS_BETA``. ``chunk_bounds`` is the chunk record, required
    (``launch.bind_chunk_bounds``).
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "beta_plus",
                 "beta_minus", "has_beta", "backward", "bc", "block",
                 "num_warps", "chunk_bounds", "_targets", "_aux", "_sources",
                 "_coefficients", "_cb", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc,
                 beta_plus: float, beta_minus: float, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None, has_beta: int = 1, *,
                 chunk_bounds: Any) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                             f"got {sub_step!r}")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "FoldedBetaPmlCurlPlan")
        self.dtdx = float(dtdx)
        # Already f32-rounded by the host (special_kz.beta_curl_coefficients);
        # float() keeps the bits and Triton types a Python float argument as fp32.
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.has_beta = int(has_beta)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. ``guard`` is the gate's, not a caller's."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else folded_beta_pml_curl_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.beta_plus, self.beta_minus,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"FoldedBetaPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, beta_plus={self.beta_plus!r}, "
                f"has_beta={self.has_beta}, block={self.block}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_folded_beta_pml_curl(fields: Any, pml: Any, sub_step: str,
                              block: Optional[int] = None,
                              num_warps: Optional[int] = None,
                              chunk_layout: Any = None
                              ) -> Optional[FoldedBetaPmlCurlPlan]:
    """Build a K3a plan from the engine's own objects, or None.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not folded_beta_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    spec = SUB_STEPS[sub_step]
    plus, minus = beta_curl_coefficients(grid.beta, grid.dt,
                                         magnetic=(sub_step == "step_B"),
                                         complex_storage=False)
    return FoldedBetaPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, plus, minus,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_folded_beta_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                          flat: Dict[str, Any], codes,
                                          dtdx: float, beta_plus: float,
                                          beta_minus: float,
                                          block: Optional[int] = None,
                                          kernel: Any = None,
                                          num_warps: Optional[int] = None,
                                          has_beta: int = 1, *,
                                          chunk_bounds: Any
                                          ) -> FoldedBetaPmlCurlPlan:
    """Build a K3a plan from bare device arrays — the gate's route.

    ``chunk_bounds`` has no default: a harness plan built without a record is refused
    by name (``launch.bind_chunk_bounds``).
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return FoldedBetaPmlCurlPlan(
        sub_step, shape, dtdx, codes, beta_plus, beta_minus,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
        chunk_bounds=chunk_bounds,
    )


class FoldedBetaBlochPmlCurlPlan:
    """A launchable, allocation-free FOLDED COMPLEX beta PML curl sub-step (K3b).

    :class:`FoldedComplexPmlCurlPlan`'s bindings (connections, chunk record) plus the
    two quarter-turn words and ``HAS_BETA``.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "connections", "connection_values", "beta_words", "has_beta",
                 "block", "num_warps", "chunk_bounds", "_targets",
                 "_aux", "_sources", "_coefficients", "_cb", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, connections,
                 connection_values, beta_words, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None, has_beta: int = 1, *,
                 chunk_bounds: Any) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                             f"got {sub_step!r}")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        for axis, (code, kind) in enumerate(zip(self.bc, self.connections)):
            if kind and code != CODE_PERIODIC:
                raise ValueError(
                    f"axis {axis} carries a Bloch connection but resolved to code "
                    f"{code}; only a periodic wrap carries a phase")
        words = tuple(beta_words)
        if len(words) != 2 or any(isinstance(word, (tuple, list)) for word in words):
            raise ValueError(f"beta_words must be the two real quarter-turn words of "
                             f"special_kz.beta_curl_coefficients(..., "
                             f"complex_storage=True), got {beta_words!r}")
        self.beta_words = tuple(float(word) for word in words)
        self.has_beta = int(has_beta)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "FoldedBetaBlochPmlCurlPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as the others."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = (self._kernel if self._kernel is not None
                  else folded_beta_bloch_pml_curl_step)
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.connection_values,
            *self.beta_words,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            CNX=self.connections[0], CNY=self.connections[1], CNZ=self.connections[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"FoldedBetaBlochPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, connections={self.connections}, "
                f"beta_words={self.beta_words!r}, has_beta={self.has_beta}, "
                f"block={self.block}, chunk_bounds={self.chunk_bounds.tolist()})")


def plan_folded_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                                    block: Optional[int] = None,
                                    num_warps: Optional[int] = None,
                                    chunk_layout: Any = None
                                    ) -> Optional[FoldedBetaBlochPmlCurlPlan]:
    """Build a K3b plan from the engine's own objects, or None.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not folded_beta_bloch_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    kinds = resolve(grid, pml)
    phases = bloch_phase_table(grid, kinds)
    connections, values = connection_arguments(phases, backward=bool(spec["backward"]))
    beta_words = beta_curl_coefficients(grid.beta, grid.dt,
                                        magnetic=(sub_step == "step_B"),
                                        complex_storage=True)
    return FoldedBetaBlochPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, connections, values,
        beta_words,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_folded_beta_bloch_pml_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], codes,
        phases: Sequence[Optional[complex]], dtdx: float,
        beta_words, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = None,
        has_beta: int = 1, *, chunk_bounds: Any) -> FoldedBetaBlochPmlCurlPlan:
    """Build a K3b plan from bare device arrays — the gate's route.

    The direction's connection factor is formed HERE, per sub-step
    (``complex_fields.connection_arguments``); ``chunk_bounds`` has no default.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(spec["backward"]))
    return FoldedBetaBlochPmlCurlPlan(
        sub_step, shape, dtdx, codes, connections, values, beta_words,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
        chunk_bounds=chunk_bounds,
    )


def plan_folded_complex_constitutive(fields: Any, pml: Any, side: str,
                                     block: Optional[int] = None,
                                     num_warps: Optional[int] = None,
                                     chunk_layout: Any = None
                                     ) -> Optional[ComplexConstitutivePlan]:
    """A CERTIFIED complex constitutive plan for a folded run, or None.

    The kernel and plan class are ``complex_fields``' (certified, then re-cut),
    untouched; only the ADMISSION is this module's.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not folded_complex_constitutive_coverage(fields, pml, side).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ComplexConstitutivePlan(
        side, fields.grid.shape,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
    )


def plan_folded_beta_run_constitutive(fields: Any, pml: Any, side: str,
                                      block: Optional[int] = None,
                                      num_warps: Optional[int] = None,
                                      chunk_layout: Any = None
                                      ) -> Optional[ConstitutivePlan]:
    """A CERTIFIED real constitutive plan for a folded BETA run, or None.

    The arithmetic and the plan class are ``launch.ConstitutivePlan``'s,
    untouched; only the ADMISSION is this module's.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not folded_beta_run_constitutive_coverage(fields, pml, side).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ConstitutivePlan(
        side, fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
    )


# ---------------------------------------------------------------------------
# OFF-DIAGONAL chi1inv on the curls and update_H — predicate scope only
# ---------------------------------------------------------------------------
#
# :func:`_media_reasons`' off-diagonal clause (folded_complex.py:1608-1613) is
# shared by the curl AND the constitutive predicates above, and it is CORRECT —
# but only at ``update_E``. The row product it names lives in
# ``stepping._offdiagonal_terms`` (S:1190-1225), which ``update_E`` alone calls
# (S:972-979). The curls difference the STORED E and H arrays and read no
# ``chi1inv`` at all; ``update_H`` is ``B``/``mu`` with the PML accumulation and
# reads no ``chi1inv`` either.
#
# THE CLAUSE'S REACHABILITY ARGUMENT DOES NOT SURVIVE, and that is worth
# recording because it was the leading hypothesis going in. The clause cites
# ``_special_kz_beta_term`` (S:773-783) and MEEP ``fields.cpp:548-549`` as
# refusing "the same pairing". They do not: both refuse BETA plus an off-diagonal
# epsilon in REAL storage, where the implicit-i trick on the TM half stops
# cancelling. Neither says anything about a FOLD plus an off-diagonal epsilon.
# The corpus census settled it by lifting the objects:
# ``TestArrayMetadata.test_array_metadata``, ``TestHoleyWvgBands.test_fields_at_kx``
# and ``solve-cw.py`` each lift with ``force_complex_fields=True``, a live mirror
# fold and surviving off-diagonal ``chi1inv`` rows AT ONCE — the rows coming from
# subpixel smoothing of the cylinder interfaces all three scripts carry
# (results/residual_triage_2026-08-14/census.jsonl, 3/3 rows).
#
# The device leg then ran the certified bodies against ``stepping.py`` on a
# configuration carrying all three features at once, EIGHT complete cycles, uint32
# over the whole stored inventory. ``moved`` is the MINIMUM over cycles of what the
# sub-step alone wrote, bracketed around that one call on both sides:
#
#   D_complex_fold_offdiag_step_B     IDENTICAL      0 / 110592  (>=26361 moved)
#   D_complex_fold_offdiag_step_D     IDENTICAL      0 / 110592  (>=25601 moved)
#   D_complex_fold_offdiag_update_H   IDENTICAL      0 / 110592  (>=27648 moved)
#   D_complex_fold_offdiag_update_E   DIVERGENT  25041 / 110592   <- and this is
#                                                  the largest divergence of the
#                                                  round
#
# ON THREE SHAPES, and that is not decoration. That fixture is a synthetic 3-D box,
# ``[9,16,16]`` with ONE mirrored axis and no phase — it is NOT the corpus
# configuration, which this comment used to claim it was. The rows are 2-D and
# carry more: ``test_array_metadata`` and ``solve-cw.py`` fold TWO axes at
# ``[201,201,1]``/``[161,161,1]``, and ``test_fields_at_kx`` carries
# ``k=(3.5,0,0)`` on a periodic axis beside its fold. Both were measured:
#
#   D_ROWSHAPE_TWOFOLD_{step_B,step_D,update_H}  IDENTICAL  0 / 8112   (2 folds)
#   D_ROWSHAPE_BLOCH_{step_B,step_D,update_H}    IDENTICAL  0 / 16128  (k=3.5)
#
# (results/residual_closure_2026-08-15/device/bodies/bodies.json; minus0 >= 3456,
# 258 and 504 respectively — the +-0 lattice is held live through every cycle, and
# a constitutive leg whose census reaches zero is reported VACUOUS.)
#
# AND THROUGH THIS MODULE'S OWN BUILDERS. The nine legs above reach the certified
# bodies by patching the INCUMBENT predicate. The closure round's second leg builds
# through ``plan_folded_complex_offdiag_pml_curl`` and
# ``plan_folded_complex_offdiag_constitutive`` with their OWN predicates unpatched
# — a plan of None would be a failed leg — and measures the same nine identities
# (results/residual_closure_2026-08-15/device/newpred/new_predicates.json).
#
# So the refusal at ``update_E`` stands and is now backed by 25041 differing
# words instead of an analogy: a COMPLEX sibling of the certified
# ``folded_offdiag_update_e`` body is the kernel that would be needed, and it is
# NOT built here. The three sub-steps below are admissions of the CERTIFIED
# bodies, unchanged.


def _offdiag_media_reasons(fields: Any, grid: Any,
                           targets: Sequence[str]) -> List[str]:
    """:func:`_media_reasons`, RESTATED with the off-diagonal clause INVERTED.

    Restated rather than parameterised: ``_media_reasons`` is the incumbent
    predicates' contract and narrowing it in place would weaken three shipped
    refusals at once, including the one at ``update_E`` that 25041 differing
    words say is load-bearing. Every other clause — conductivity on either
    family, the magnetic conductivity, chi2/chi3, the susceptibility shape,
    BFAST, stored E — is kept verbatim in force.
    """
    reasons: List[str] = []
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    else:
        for target in targets:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}: this tranche "
                    f"transcribes the plain split-field recurrence only")
    if getattr(fields, "has_magnetic_conductivity", False):
        reasons.append("a magnetic (B) conductivity is installed")
    if getattr(fields, "has_nonlinearity", False):
        reasons.append(
            "chi2/chi3 is installed: the Pade constitutive factor is not "
            "carried, and driver._require_folded_far_face_is_quiet refuses a "
            "nonlinear fold with a live far face")

    # INVERTED. The row product must be INSTALLED: without one the configuration
    # is `folded_complex_pml_curl_coverage`'s and `folded_complex_constitutive_
    # coverage`'s, and two predicates admitting one configuration is how a
    # dispatcher picks a body by ordering. Disjoint by construction; a test pins
    # it on both sub-step families.
    if not getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "no off-diagonal chi1inv row is installed: that configuration is "
            "the incumbent folded-complex curl and constitutive predicates' and "
            "this admission must not overlap them")
    reasons.extend(_susceptibility_reasons(fields))
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (a second additive curl term, refused in "
                       "both directions)")
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")
    return reasons


def _complex_fold_offdiag_grid_reasons(fields: Any, pml: Any, grid: Any,
                                       targets: Sequence[str]
                                       ) -> Tuple[List[str],
                                                  Optional[Tuple[int, int, int]]]:
    """:func:`_complex_fold_grid_reasons`, RESTATED over the inverted media set."""
    reasons: List[str] = []

    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32 (neither force_complex_fields nor a nonzero "
            "k_point): a real folded run with an off-diagonal row belongs to "
            "symmetry.py's kernels and to folded_offdiag_update_e.py")

    reasons.extend(_absorber_reasons(pml))
    reasons.extend(_cylindrical_reasons(grid))
    active_pml = pml if (pml is not None and getattr(pml, "is_active", False)) else None
    kinds = _boundary_kinds(grid, active_pml)
    codes, fold_reasons = folded_axis_kinds(grid, active_pml)
    reasons.extend(fold_reasons)
    reasons.extend(_fold_clause_reasons(grid, active_pml, codes))
    reasons.extend(_complex_phase_reasons(grid, kinds, codes))
    reasons.extend(_offdiag_media_reasons(fields, grid, targets))
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: complex-storage ADE is a future "
            "tranche (the ADE launch raises KeyError: 'complex64' — an unbuilt "
            "kernel, not a rounding gap), and no folded complex off-diagonal "
            "corpus row carries one")
    return reasons, codes


def folded_complex_offdiag_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May K1 step a folded COMPLEX run that also carries an off-diagonal row?

    K1's clause set with the off-diagonal clause INVERTED and nothing else
    changed. A REAL FOLD IS REQUIRED here — unlike
    :func:`folded_complex_pml_curl_coverage`, which admits an unfolded grid so
    its own gate can measure the reduction to the unfolded kernel. This predicate
    has no such equivalence leg to prove, and admitting an unfolded grid would
    put it in competition with ``complex_fields``' own curl predicate.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    reasons, codes = _complex_fold_offdiag_grid_reasons(fields, pml, grid, CURL_TARGETS)

    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: an unfolded complex run with "
                       "an off-diagonal row is complex_fields' curl predicate's")
    elif codes is not None and not any(
            code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC) for code in codes):
        reasons.append("no axis resolves to a mirror boundary")

    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(
            f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero: beta "
            f"WITH an off-diagonal epsilon is the pairing MEEP itself refuses "
            f"(fields.cpp:548-549) and _special_kz_beta_term (S:773-783) raises on")

    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))
    return Coverage(not reasons, tuple(reasons))


def folded_complex_offdiag_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the certified complex constitutive kernel step ``update_H`` here?

    ``side='H'`` ONLY. ``side='E'`` is REFUSED BY NAME rather than raised — it is
    a real side of a real sub-step, it is simply the one the row product enters,
    and the measurement that says so is 25041 of 110592 words differing over eight
    complete cycles. The certified
    ``ComplexConstitutivePlan`` is element-wise; the array path adds
    ``_offdiagonal_terms``' row product, which reads partner volumes through
    ``_shift_up``/``_shift_down`` (S:1206-1224). A complex sibling of the
    certified ``folded_offdiag_update_e`` body is what that would take, and it is
    not built.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons, codes = _complex_fold_offdiag_grid_reasons(fields, pml, grid, CURL_TARGETS)
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: the folded constitutive "
                       "product has no array-path work to specialize")
    elif codes is not None and not any(
            code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC) for code in codes):
        reasons.append("no axis resolves to a mirror boundary")

    if side == "E":
        reasons.append(
            "side='E' is where the off-diagonal row product enters "
            "(stepping.py:1001-1008 via _offdiagonal_terms, S:1190-1225; measured "
            "25041/110592 words differing against the certified element-wise "
            "body): that sub-step needs a complex folded off-diagonal kernel, "
            "which is not built")

    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(
            f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero: beta "
            f"with an off-diagonal epsilon is refused in both directions")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))
    return Coverage(not reasons, tuple(reasons))


def plan_folded_complex_offdiag_pml_curl(fields: Any, pml: Any, sub_step: str,
                                         block: Optional[int] = None,
                                         num_warps: Optional[int] = None,
                                         chunk_layout: Any = None
                                         ) -> Optional[FoldedComplexPmlCurlPlan]:
    """A CERTIFIED K1 plan for a folded complex run with an off-diagonal row, or None.

    The kernel and the plan class are K1's, untouched; only the ADMISSION is new.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not folded_complex_offdiag_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    kinds = resolve(grid, pml)
    phases = bloch_phase_table(grid, kinds)
    connections, values = connection_arguments(phases, backward=bool(spec["backward"]))
    return FoldedComplexPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, connections, values,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_folded_complex_offdiag_constitutive(fields: Any, pml: Any, side: str,
                                             block: Optional[int] = None,
                                             num_warps: Optional[int] = None,
                                             chunk_layout: Any = None
                                             ) -> Optional[ComplexConstitutivePlan]:
    """A CERTIFIED complex ``update_H`` plan for that run, or None.

    ``side='E'`` plans to None here, always: the predicate refuses it by name.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not folded_complex_offdiag_constitutive_coverage(fields, pml, side).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ComplexConstitutivePlan(
        side, fields.grid.shape,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
    )


def explain_folded_complex(fields: Any, pml: Any, sub_step: str = "step_B") -> Coverage:
    """The folded complex curl's coverage verdict with its reasons. Needs no Triton."""
    return folded_complex_pml_curl_coverage(fields, pml, sub_step)


# ---------------------------------------------------------------------------
# WIRING — the planner seam is IN; dispatch is not
# ---------------------------------------------------------------------------
#
# This block described the deferral and was left standing after the deferral
# ended, which made it a false statement about two other files. As it now
# stands: ``launch.plan_step`` consults these predicates through lazy forwarders,
# each arm gated on the fold (and on complex storage or beta where the family
# requires it), and the package ``__init__`` exports the entry points behind the
# same lazy seam. ``fastpath.plan_fast_path`` is untouched BY THIS FILE — but it
# no longer returns None on every branch, so the sentence that used to sit here
# was carrying a retired claim: dispatch is held shut by
# ``fastpath.DISPATCH_BY_DEFAULT``, not by the absence of a caller.
#
# WHAT MADE THE DEFERRAL SAFE IS ALSO WHAT MAKES THE WIRING SAFE, and it is worth
# keeping rather than deleting: ``coverage._grid_reasons``' fold/complex/beta
# clauses and ``complex_fields``' fold refusal refuse every configuration these
# predicates admit, so no shipped predicate admits a configuration this one also
# admits. The composition measures that rather than trusting it — the planner's
# disjointness sweep drives real folded complex and folded beta triples through
# the shipped predicates and asserts the full selected dict — and the coordinated
# change that wired it is the one that owes the byte gate a re-run.
#
# ``special_kz`` is the one that had to be MADE true rather than observed.
# ``special_kz._beta_real_grid_reasons`` clause 5 refuses a fold but does not
# REQUIRE one, so it stays live on an unfolded beta grid — and until 2026-08-12
# this file's two beta predicates did not require one either. Measured on
# ``Grid(cell_size=(1.6,1.6,0.0), dimensions=2, boundaries='periodic',
# beta=0.2)`` with no symmetry, in BOTH storages: this file's predicate and
# ``special_kz``' returned identical (empty) reason lists. Both beta predicates
# here now carry the ``_has_real_fold`` clause the two constitutive predicates
# and K1's composition verdict already carried, and a laptop test pins the
# refusal. Tests and the gate import
# ``meep_gpu.triton_kernels.folded_complex`` directly (the package ``__init__``
# eagerly imports only ``coverage``, so the direct import is safe on a
# Triton-less host).
#
# THE OFF-DIAGONAL SIBLINGS ARE WIRED TOO (added 2026-08-14 by the residual-group
# closure round, which did not own ``launch.py``; made arms in the round after it):
#
#   folded_complex_offdiag_pml_curl_coverage      step_B, step_D
#                                                 -> arm "folded complex off-diagonal PML"
#   folded_complex_offdiag_constitutive_coverage  update_H  (side='E' refuses)
#                                                 -> arm "folded complex off-diagonal"
#
# with ``plan_folded_complex_offdiag_pml_curl`` and
# ``plan_folded_complex_offdiag_constitutive`` as the builders, each gated on
# ``folded_grid_active and complex_storage_active and offdiag_rows_possible``.
# Nine slots across three corpus rows — ``TestArrayMetadata.test_array_metadata``,
# ``TestHoleyWvgBands.test_fields_at_kx`` and ``solve-cw.py``, none of them
# TestLoadDump.
#
# WHAT THAT MAKES OF ``_media_reasons``' OFF-DIAGONAL CLAUSE, which is left EXACTLY
# as it is: the FAMILY now refuses an off-diagonal row at ``update_E`` ALONE, which
# is where the row product lives, instead of across all four sub-steps. The clause
# is not narrowed in place, and narrowing it would be the wrong repair — the
# incumbent curl and constitutive predicates would then admit the same three slots
# these siblings admit, and ``launch._select_slot`` fails CLOSED on a double
# admission: all nine slots would go UNSELECTED and fall to the array path. That is
# a silent coverage LOSS, not an error, which is why the scope change is made by
# ADDING an arm.
#
# ``update_E`` on those three rows STAYS REFUSED and stays the best-motivated
# unbuilt kernel in the queue at 25041/110592 differing words.
