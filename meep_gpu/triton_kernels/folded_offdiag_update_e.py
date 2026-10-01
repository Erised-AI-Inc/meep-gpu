"""The FOLDED off-diagonal (tensor) electric constitutive sub-step — ``update_E``
with off-diagonal chi1inv rows installed on a MIRROR-FOLDED grid.

One kernel, one coverage predicate, one plan, in one file, on the pattern
:mod:`offdiag_update_e` and :mod:`symmetry` set between them. This family is
their intersection, and it exists because the argument that admitted the
DIAGONAL constitutive product onto a folded grid does not transfer — measured,
not argued (see THE ELEMENT-WISE ARGUMENT DOES NOT TRANSFER below).

DISPATCH IS NOT WIRED; the PLANNER is, and this paragraph said otherwise for a
round. The engine's fast-path hook keeps returning ``None`` on every branch and
nothing here changes that, so no production step reaches this family.
``plan_step`` composition is NO LONGER DEFERRED: ``launch.py`` carries the two
lazy forwarders and the ``folded off-diagonal`` ``update_E`` arm, gated on the
fold AND on the off-diagonal disjunction, and ``__init__.py`` exports the entry
points — a later round edited those files, so the sentence that listed them as
"imported, never edited" now describes only this module's own restraint at the
time it was written. ``coverage.py``, ``folded_complex.py``, ``stepping.py``,
``fastpath.py`` and ``backends.py`` remain untouched, and this module still
carries no ``fingerprints.json`` entry — the byte gate binds its own provenance
record inside its results directory. Callers are the gate
(``parity/meep_gpu/gate_triton_folded_offdiag.py``), the laptop reference
validator (``parity/meep_gpu/validate_folded_offdiag_reference_vs_stepping.py``),
the planner and the laptop tests.

ONE THING THE WIRING CHANGED FOR THIS FAMILY IN PARTICULAR. ``launch.py``'s
``_veto_dropped_offdiagonal_coupling`` refuses any ``update_E`` product that does
not implement the row term on a grid where the array path forms it; this family's
arm is one of the two named in ``ROW_PRODUCT_ARMS`` and is therefore exempt.
Wiring the arm WITHOUT that entry would have had the veto empty the one slot this
kernel exists for, reporting "has no row-product term" about a kernel whose whole
body is that term.

WHY THIS FAMILY EXISTS AT ALL — the two refusals it sits between:

* :func:`offdiag_update_e.offdiag_constitutive_coverage` calls
  ``coverage._grid_reasons`` unmodified (offdiag_update_e.py:647), so clause 5
  (coverage.py:171-175) refuses EVERY fold;
* :func:`symmetry.folded_constitutive_coverage` refuses every off-diagonal row
  (symmetry.py:814-817), because that predicate gates the ELEMENT-WISE
  constitutive kernel.

Neither refusal is stale. Each is correct for its own product. The union they
leave uncovered is this file's.

WHAT IT REPLACES. ``stepping.update_E`` (stepping.py:954) with an active PML,
``fields.has_offdiagonal_epsilon`` True, NOT ``fields.has_nonlinearity``, and at
least one axis resolving to ``MIRROR`` through ``stepping._boundary_kinds``
(stepping.py:2192-2197) — the ``elif offdiagonal:`` branch (stepping.py:1001-1008).
With no poles admitted, ``displacement_minus_polarization_volumes`` aliases each
source to its D primary, and all three are alive at once because the coupling
reads the OTHER components' volumes (stepping.py:991-997). Per component ``c``
with own axis ``a``::

    constitutive = D_c * us_c                               # stepping.py:1005
    per surviving partner (offset 1 then 2, cycle X->Y->Z, :1235-1237):
        pair    = g + shift_down(g, partner_axis)           # :1214-1216
        product = pair * coefficient                        # :1217
        term    = 0.25 * (product + shift_up(product, a))   # :1219-1220
        total   accumulates term(offset1) then term(offset2)  # :1221
    _mask_metallic_wall_coupling(total)                     # :1223, :1227-1254
    constitutive = (D_c * us_c) + total                     # :978-979
    prev = f_w_c ; f_w_c = constitutive                     # :2065-2096
    E_c += kps_a_h * f_w_c ;  E_c -= kms_a_h * prev         # half-integer, :986

Every line of that is :mod:`offdiag_update_e`'s, unchanged. The whole delta is
what ``_shift_down`` and ``_shift_up`` do on a MIRROR axis.

================================================================================
THE ELEMENT-WISE ARGUMENT DOES NOT TRANSFER. This is the question the family
turns on, so it is answered with citations and a discriminating measurement
rather than by analogy.
================================================================================

The diagonal constitutive product was admitted onto a folded grid on exactly one
argument (symmetry.py:5-7, :318-320): it READS NO NEIGHBOUR, so the fold's ghost
rules cannot reach it, and the only thing the fold changes — the stored extent —
is already carried by the PML coefficient vectors, which are built at the stored
length. ``coverage._grid_reasons`` clause 5 (coverage.py:166-175) states the same
fact from the other side: the constitutive kernels refuse folds "even though
their sub-step reads no neighbour" purely because a fold changes ``n_a``.

The OFF-DIAGONAL row product reads neighbours (stepping.py:1243-1249), so that
argument is unavailable and the fold's ghost rule reaches it. It reaches it in
one place, and NOT in the other, and both halves matter:

1. **THE PARTNER-AXIS DOWN SHIFT IS LIVE ON A FOLD.** ``_shift_down`` is called
   WITH a component name and the plane's parity (stepping.py:1243-1245), so a
   MIRROR partner axis takes stepping.py:1870-1873::

       shifted[face(axis, 0)] = _symmetry_phase("D"+partner, axis, phase)
                                * field[face(axis, MIRROR_SOURCE_INDEX)]

   — a LIVE parity-weighted interior plane, neither the periodic wrap nor the
   metallic zero. And unlike the metallic case it is NOT masked away afterwards:
   ``_mask_metallic_wall_coupling`` asks ``is_metallic and NOT is_mirrored``
   (stepping.py:1282) and therefore ABSTAINS on the fold plane, deliberately and
   with its own measurement (stepping.py:1266-1277: zeroing the fold plane the
   way the metallic rule does costs 2.0e-02 even / 3.5e-03 odd).

   That is the exact structural inversion of the certified family's recorded
   NULL. There, the metallic near ghost was unobservable BECAUSE the wall mask
   zeroed its entire support before the row sum (offdiag_update_e.py:108-117,
   that family's gate: the wrapped-ghost mutant measured byte-identical on the
   all-metallic sweep with a PTX-verified-different binary). Here the mask
   abstains, so the same plane is live and the ghost VALUE is byte-visible.
   The partner axis is always an axis on which the ROW component's Yee shift is
   0 (``E_c`` has ``iyee[a] = 1`` on its own axis ``a`` and 0 on both partner
   axes, fields.py:214-219) — which is precisely the axis set the mask loops
   over (stepping.py:1279-1281). So "mask fires" and "ghost is live" are the
   same switch, thrown the other way by the fold.

2. **THE OWN-AXIS UP SHIFT IS *NOT* PARITY-WEIGHTED, ON EITHER TERMINATION.**
   ``_offdiagonal_terms`` calls ``_shift_up`` with FOUR arguments (stepping.py:
   1248-1249) — no ``component``, no ``mirror_phase``, no ``reflect_row`` — so
   the folded-PERIODIC reflect branch (stepping.py:1819-1827) cannot fire and
   both mirror terminations fall to stepping.py:1828-1830, an exact ``0.0``.
   That is byte-identical to the METALLIC arm this kernel already carries.

   It is transcription FIDELITY, not a physics claim, and the distinction is
   load-bearing: on a folded PERIODIC axis MEEP's ghost past the stored top is
   the parity-weighted image of ``_far_reflect_rows``' row (stepping.py:1708),
   which is what ``_shift_up`` serves for the CURL. The offdiag coupling does
   not ask for it. See ARRAY-PATH FINDING below — that gap belongs to
   ``stepping.py`` and ``driver.py``, not to this family, and this kernel
   reproduces whatever the array path does, byte for byte, either way.

3. **THE STORED EXTENT** is carried exactly as :mod:`symmetry` carries it: the
   PML ``kps``/``kms`` vectors are built at the folded stored length
   (symmetry.py:318-320), so the half-integer coefficient index needs no
   fold-aware change. The predicate proves it (``_coefficient_reasons``) rather
   than assuming it.

4. **THE WALL MASK NEEDS NO CHANGE.** ``offdiag_update_e.wall_mask_axes``
   (offdiag_update_e.py:521-536) already asks the mask's own question,
   ``is_metallic and not is_mirrored``, so a folded axis already compiles to
   ``WM = 0``. It is imported, not restated.

MEASURED (laptop, NumPy, ``validate_folded_offdiag_reference_vs_stepping.py``;
no device, no byte-identity claim about any kernel yet). Same real
``Grid``/``Fields``/``PML``, same seeds, same installed rows; only the
transcription of the two ghost arms varies, against ``stepping.update_E``:

    down = parity * g[row 2],  up = 0        IDENTICAL   <- the transcription
    down = 0 (the metallic arm)              differs     <- (1), the certified
    down = periodic wrap                     differs         kernel cannot serve
    down = +phase * g[row 2] (sign flipped)  differs         a fold
    up   = parity * g[reflect_row]           differs     <- (2): the array path
    up   = periodic wrap                     differs         really serves 0
    fold plane masked like a metallic wall   differs     <- the mask must abstain

on folded PERIODIC X (even and odd full count, both phases), folded METALLIC Y
(both phases), folded PERIODIC Z, two axes folded at once (three phase
combinations), THREE axes folded at once on both terminations (where every one
of the six row slots takes a ghosted down shift), a reduced 2-D run with a fold,
and Courant 0.5 / 0.35 / 0.3125 — 16 configurations, each byte-identical over
four chained sub-step calls.

REACHABILITY IS SHARP AT SLOT LEVEL — NOT AT ROW LEVEL — AND ONE SUB-CASE NEEDS
NO NEW KERNEL. Because the mirror rule enters only through the PARTNER-axis down
shift (``_shift_down`` is called with ``axis = partner_axis``, stepping.py:
1243-1245), a fold on axis ``a`` changes this sub-step's bytes exactly when some
SURVIVING ROW SLOT TAKES ``E_a`` AS ITS PARTNER. Row ``E_a``'s own slots never
do, which is where the coarser reading ("some live row is not ``E_a``") comes
from — but a row can be live through its OTHER slot and never touch the fold,
and that case is measured, not argued. Same grid, same coefficients, same seeds,
flipping only the folded axis's ghost code, three chained sub-step calls:

    fold X, live slots only in row Ex     MIRROR bytes == METALLIC bytes  True
    fold Y, live slots only in row Ey     MIRROR bytes == METALLIC bytes  True
    fold Z, live slots only in row Ez     MIRROR bytes == METALLIC bytes  True
    fold X, single live slot  Ey <- Ex    MIRROR bytes == METALLIC bytes  False
    fold Y, single live slot  Ez <- Ey    MIRROR bytes == METALLIC bytes  False
    fold Z, single live slot  Ex <- Ez    MIRROR bytes == METALLIC bytes  False
    fold X, single live slot  Ey <- Ez    MIRROR bytes == METALLIC bytes  True
    fold Y, single live slot  Ez <- Ex    MIRROR bytes == METALLIC bytes  True
    fold Z, single live slot  Ex <- Ey    MIRROR bytes == METALLIC bytes  True

The last three rows are the counterexample to the row-level reading: the live
row is not ``E_a``, yet the fold is in NEITHER role for the surviving slot and
the bytes agree. :func:`mirror_arm_is_reachable` tests the slot, the gate's
``identity_host`` leg carries all nine configurations, and a disagreement
between the predicate and the byte measurement fails that leg.

So a folded run in which no surviving slot takes ``E_a`` as its partner, for the
single folded axis ``a``, is the CERTIFIED kernel's with that axis coded
METALLIC. It is NOT routed away here —
routing by predicate order would make the numerical method a function of
composer order (the reason ``symmetry.folded_composition_curl_coverage`` exists,
symmetry.py:713-734) — it is carried as the gate's IDENTITY leg, which is the
measurement that proves the delta is exactly where this docstring says it is.

================================================================================
THE THINGS THAT DECIDE BIT-IDENTITY, each held by the gate rather than assumed
================================================================================

A. **THE MIRROR GHOST WEIGHT IS EXACTLY ``-phase``, ALWAYS.** ``_shift_down``
   passes ``component = "D" + AXIS_NAMES[partner_axis]`` on ``axis =
   partner_axis`` (stepping.py:1243-1245), and every D component has Yee shift 1
   on its OWN axis (fields.py:214-219). ``mirror_parity(c, axis, phase) ==
   phase * (1 - 2*iyee[c][axis])`` (fields.py:117; verified exhaustively over 12
   components x 3 axes x 2 phases) therefore collapses to ``-phase`` for every
   partner and every axis. One signed scalar per folded axis is the entire
   parity input — the same reduction :func:`symmetry.mirror_ghost_fill` makes
   for the far fill (symmetry.py:368-374). A test pins the collapse against
   ``fields.mirror_parity`` so it cannot drift.

B. **THE WEIGHT IS A RUNTIME f32 SCALAR, NEVER A UNARY MINUS.** Platform fact:
   Triton 3.1.0's ``semantic.minus`` lowers ``-x`` as ``0.0 - x``, which
   canonicalizes signed zeros — ``-(+0.0)`` becomes ``+0.0`` where IEEE negation
   gives ``-0.0``, and this was measured REACHABLE at driver level, not
   theoretical. A constant ``x * -1.0`` is within reach of the
   same canonicalization. So the weight arrives as a plain runtime argument
   (``gwx``/``gwy``/``gwz``, exactly ``+1.0`` or ``-1.0``) and is applied as a
   multiply on the ghost lane only. ``1.0 * x`` is exact in f32 for every input
   including subnormals and both zeros, which is what lets the even-phase build
   reduce to the certified body byte for byte — a property the identity leg
   measures rather than a claim. The gate carries a mutation that respells the
   weight as a unary minus and a null control that asks whether the two agree,
   so the platform fact is re-measured on the class where it can bite instead of
   being inherited.

C. **THE GHOST INDEX IS STORED ROW 2**, ``stepping.MIRROR_SOURCE_INDEX``
   (stepping.py:160, ``_mirror_source`` :1582-1588): MEEP's halved origin at
   ``io = -2`` maps the ghost at -1 onto stored cell 2. Restated here rather
   than imported so the module stays engine-import-free at module scope, exactly
   as :mod:`symmetry` restates it (symmetry.py:116-119); a test pins them equal.
   ``_mirror_source`` RAISES below three stored cells, so the predicate refuses
   that axis by name (through :func:`symmetry.folded_axis_kinds`, which already
   carries the clause, symmetry.py:515-519).

D. **BOTH MIRROR CODES BEHAVE IDENTICALLY IN THIS KERNEL — a PREDICTED NULL
   with a derivation, carried as a null control rather than collapsed.**
   :mod:`symmetry` splits ``MIRROR_METALLIC`` (2) from ``MIRROR_PERIODIC`` (3)
   because the CURL masks a folded-periodic axis's top plane
   (``_stored_past_owned``, stepping.py:1503) and reflects past it. This
   sub-step has NO ownership mask at all (``update_E`` never calls
   ``_mask_non_owned_cells``) and, by point 2 above, no reflect row either. So
   the two codes must produce identical bytes here. The codes are still CARRIED
   and still CLASSIFIED — through :func:`symmetry.folded_axis_kinds`, which
   cross-checks ``_stored_past_owned`` against ``grid.is_metallic`` and refuses a
   disagreement (symmetry.py:526-532) — because collapsing them silently would
   discard that cross-check, and because a plan built here and a plan built
   there must index the SAME table (symmetry.py:108-114).

E. **EVERYTHING ELSE IS :mod:`offdiag_update_e`'S, UNCHANGED**, and is
   therefore that family's certified concern rather than re-litigated here: the
   coefficient multiply sitting BETWEEN the two shifts (that family's m1), the two
   shifts going in OPPOSITE directions (m2), the ``0.25`` scaling the sum last
   (a recorded null — an exact power of two commutes with round-to-nearest away
   from underflow), the diagonal-first row sum (a null — f32 addition is
   bitwise commutative), the slot-to-partner binding (m3), ``prev`` read before
   ``f_w`` is written (m9), and the no-surviving-row arm reducing to
   :func:`kernels.constitutive_step`'s body. The gate re-runs the ones the fold
   could plausibly perturb and cites that family's certification for the rest.

F. **``ENABLE_FP_FUSION = False`` IS THE CERTIFIED CONFIGURATION.** Measured
   per tranche: offdiag 0/28 fusion-on rows identical (and nonlinear 0/108,
   bfast 0/52, special_kz 24/96, complex 68/68). This family's tail is the same
   multiply-subtract shape and it gains a ghost-lane multiply, so it certifies
   under fusion-off and the gate stamps that.

G. **NO DIVISION ANYWHERE.** No Pade quotient, no ``div_rn`` concern, no
   host-rounded scalar powers: the coefficients and the inverse epsilon are
   VOLUMES (a scalar inverse epsilon is refused by name). f32 ``/`` lowers to
   ``div.full.f32`` at ~2 ulp and never appears on this path.

H. **NO OVERFLOW NEEDLE PRODUCES A NaN.** A NaN's sign and payload are
   IEEE-unspecified, so extreme-magnitude cases stay finite through the whole
   path and the gate states the bound: with ``|D| <= 1``, ``|u| <= 1`` and
   ``|chi1inv| <= 4``, the row value is bounded by ``|D*us| + 2*0.25*(2|D|)|u|
   <= 4 + 2 <= 6``, and the tail multiplies by ``|kps|, |kms| <= 4`` — three
   orders below the f32 ceiling at every intermediate.

================================================================================
ARRAY-PATH FINDING — reported, not fixed here
================================================================================

On a folded PERIODIC axis, ``_offdiagonal_terms``' own-axis ``_shift_up``
serves an exact ZERO past the stored top (point 2 above), where MEEP's ghost is
the parity-weighted image of ``_far_reflect_rows``' row. This is the same class
``stepping._shift_up``'s own docstring names for the nonlinear transverse sums
(stepping.py:1788-1791) — but the guard that refuses it,
``FdtdDriver._require_folded_far_face_is_quiet`` (driver.py:2979), returns
immediately unless ``fields.has_nonlinearity``, so an off-diagonal folded run
with a live far face is NOT refused. Nor is it pinned:
``test_tensor_epsilon.py::test_tensor_fold_equivalence_is_exact`` trims the far
row from its comparison (``[:-1]``, test_tensor_epsilon.py:554). Measured here
with the face fully live (``max|D|`` on the far plane = 3.997e-01 of the run's
scale): serving the parity-weighted reflect row instead changes the bytes.

Whether the array path or MEEP is right there is not this file's question and
not this file's to change. This kernel reproduces the array path. The finding
belongs to ``stepping.py`` and ``driver.py``, alongside
the STALE DOCSTRING HAZARD :mod:`offdiag_update_e` already records
(stepping.py:1219-1220, fields.py:1237-1241 and driver.py:1202-1204 all claim
the installer refuses folded rows; ``_validated_offdiagonal_rows``,
fields.py:1262-1310, installs them unchanged, and this whole family exists
because it does).

================================================================================
WHAT IS REFUSED, BY NAME
================================================================================

See :func:`folded_offdiag_constitutive_coverage`. In one sentence: everything
:func:`offdiag_update_e.offdiag_constitutive_coverage` refuses EXCEPT the fold,
plus everything :func:`symmetry.folded_pml_curl_coverage` refuses except the
curl-only clauses, with the fold clause replaced by
:func:`symmetry.folded_axis_kinds`. Restated, never subtracted: subtracting a
reason string from another predicate's output would make this file's coverage a
function of that file's phrasing (symmetry.py:548-556).

Import contract: this module is importable WITHOUT Triton — the predicate and
the plan builders (to ``None``) must answer on the laptop that is the merge bar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import offdiag_update_e as _offdiag
from . import symmetry as _symmetry

# ---------------------------------------------------------------------------
# The constants the kernel and the host agree on
# ---------------------------------------------------------------------------
#
# Imported by VALUE from the two parent families wherever they already own the
# fact, so there is one home per fact and a test can assert the identity.

#: The components this sub-step writes, with each component's OWN axis — the
#: axis whose HALF-INTEGER coefficient pair the tail reads (MEEP's ``dsigw``).
#: :mod:`offdiag_update_e`'s table (offdiag_update_e.py:180-181), imported.
E_TERMS: Tuple[Tuple[str, str, int], ...] = _offdiag.E_TERMS

#: The six coefficient slots in plan/kernel argument order, from the same file
#: (offdiag_update_e.py:195-198): for each row component, the offset-1 partner
#: then the offset-2 partner, in MEEP's ``cycle_direction`` order X -> Y -> Z.
ROW_SLOTS: Tuple[Tuple[str, str], ...] = _offdiag.ROW_SLOTS

#: Per component, the axes on which its Yee shift is 0 — the axes whose metallic
#: wall plane the coupling mask zeroes (offdiag_update_e.py:205).
WALL_MASK_AXES: Tuple[Tuple[int, int], ...] = _offdiag.WALL_MASK_AXES

#: The Yee sub-lattice this side reads: half-integer (stepping.py:1015).
HALF_INTEGER = True

#: Elements per program — restated from ``kernels.DEFAULT_BLOCK``; a test pins
#: the two against the source.
DEFAULT_BLOCK = _offdiag.DEFAULT_BLOCK

#: The four ghost-rule codes, :mod:`symmetry`'s own values (symmetry.py:111-114)
#: so a plan built here and a plan built there index the SAME table.
CODE_PERIODIC = _symmetry.CODE_PERIODIC
CODE_METALLIC = _symmetry.CODE_METALLIC
CODE_MIRROR_METALLIC = _symmetry.CODE_MIRROR_METALLIC
CODE_MIRROR_PERIODIC = _symmetry.CODE_MIRROR_PERIODIC
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: ``stepping.MIRROR_SOURCE_INDEX`` (stepping.py:160) — the stored row a mirror
#: ghost reflects from. Restated for the module's engine-import-free contract,
#: exactly as symmetry.py:116-119 restates it; a test pins all three equal.
MIRROR_SOURCE_INDEX = _symmetry.MIRROR_SOURCE_INDEX

#: The shared clause builders this file's predicate composes from, named as data
#: so the laptop test can assert every one still exists in the shared modules
#: this file imports and does not edit.
SHARED_CLAUSES: Tuple[str, ...] = (
    "coverage._susceptibility_reasons", "coverage._layout_reasons",
    "coverage._inverse_epsilon_reasons", "coverage._coefficient_reasons",
    "coverage._volume_reasons", "coverage._call",
    "symmetry.folded_axis_kinds", "symmetry._has_real_fold",
    "offdiag_update_e.wall_mask_axes", "offdiag_update_e.row_volumes_for",
    "offdiag_update_e._row_reasons", "offdiag_update_e._base_address",
    "offdiag_update_e._is_volume",
)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Triton is imported conditionally and the kernel is defined conditionally, for
# the measured reason dispersive_update_e.py records: a kernel defined inside a
# lazy builder resolves its names through the defining module's __globals__ and
# dies at first launch.

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:
    PERIODIC = tl.constexpr(CODE_PERIODIC)
    METALLIC = tl.constexpr(CODE_METALLIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)
    MIRROR_ROW = tl.constexpr(MIRROR_SOURCE_INDEX)
else:  # pragma: no cover - the laptop path
    PERIODIC = CODE_PERIODIC          # type: ignore[assignment]
    METALLIC = CODE_METALLIC          # type: ignore[assignment]
    MIRROR_METALLIC = CODE_MIRROR_METALLIC   # type: ignore[assignment]
    MIRROR_PERIODIC = CODE_MIRROR_PERIODIC   # type: ignore[assignment]
    MIRROR_ROW = MIRROR_SOURCE_INDEX  # type: ignore[assignment]


if triton is not None:

    @triton.jit
    def _folded_offdiag_term(g, u, o_c, o_d, o_u, o_ud,
                             v_c, v_d, v_u, v_ud, w_d,
                             MG: tl.constexpr):
        """One partner's OFFDIAG term, with or without the partner axis's ghost
        weight.

        ``stepping._offdiagonal_terms`` (:1214-1220) associates it as::

            0.25*((g[i] + w*g[i-sx])*u[i] + (g[i+s] + w*g[(i+s)-sx])*u[i+s])

        where ``w`` is the plane's ``-phase`` on the ghost lane of a MIRROR
        partner axis and 1 everywhere else. BOTH ghosted loads take the SAME
        weight and the SAME redirected index: ``o_d`` and ``o_ud`` differ only
        in the OWN-axis coordinate, so the two hit the partner-axis ghost plane
        together.

        ``MG`` IS A CONSTEXPR AND THE ``else`` ARM IS
        :func:`offdiag_update_e._offdiag_term`'S BODY VERBATIM, on purpose:
        an unfolded partner axis then compiles to the certified kernel's
        instruction sequence by CONSTRUCTION rather than by trusting a compiler
        to fold a multiply by 1.0 away. That is what makes the gate's reduction
        row a structural claim the identity leg confirms, instead of a
        compiler-version-dependent one.

        Everything else is that function verbatim either way: the coefficient
        multiply sits BETWEEN the shifts (the near pair takes ``u`` at its own
        node, the far pair at the next node up the component's own axis), and
        0.25 scales the sum, applied last. No unary minus anywhere — ``w_d`` is
        a RUNTIME scalar, never a negation the platform could canonicalize
        (docstring point B).
        """
        if MG:
            near = (tl.load(g + o_c, mask=v_c, other=0.0)
                    + w_d * tl.load(g + o_d, mask=v_d, other=0.0))
            far = (tl.load(g + o_u, mask=v_u, other=0.0)
                   + w_d * tl.load(g + o_ud, mask=v_ud, other=0.0))
        else:
            near = (tl.load(g + o_c, mask=v_c, other=0.0)
                    + tl.load(g + o_d, mask=v_d, other=0.0))
            far = (tl.load(g + o_u, mask=v_u, other=0.0)
                   + tl.load(g + o_ud, mask=v_ud, other=0.0))
        return 0.25 * ((near * tl.load(u + o_c, mask=v_c, other=0.0))
                       + (far * tl.load(u + o_u, mask=v_u, other=0.0)))

    @triton.jit
    def _masked_row_sum(diag, total, at_a, at_b,
                        WMA: tl.constexpr, WMB: tl.constexpr):
        """Wall-mask the coupling, THEN form the row sum — the array path's order
        (``_mask_metallic_wall_coupling`` at stepping.py:1252 runs before the
        ``constitutive + coupling`` add at :1007-1008).

        Byte-copied from :func:`offdiag_update_e._masked_row_sum`. On a folded
        axis ``WM`` is 0 by construction (``wall_mask_axes`` asks the mask's own
        ``is_metallic and not is_mirrored``, stepping.py:1282), which is what
        leaves the fold plane's coupling alive — the thing zeroing it costs
        2.0e-02 (stepping.py:1269-1274)."""
        if WMA:
            total = tl.where(at_a, 0.0, total)
        if WMB:
            total = tl.where(at_b, 0.0, total)
        return diag + total

    @triton.jit
    def folded_offdiag_constitutive_step(
        f0, f1, f2,                     # targets:      Ex, Ey, Ez             (in/out)
        w0, w1, w2,                     # auxiliaries:  f_w_Ex, f_w_Ey, f_w_Ez (in/out)
        g0, g1, g2,                     # sources:      Dx, Dy, Dz             (read-only)
        e0, e1, e2,                     # inverse-epsilon VOLUMES (three distinct or aliased)
        u01, u02,                       # row Ex: partner Ey (down y), partner Ez (down z)
        u11, u12,                       # row Ey: partner Ez (down z), partner Ex (down x)
        u21, u22,                       # row Ez: partner Ex (down x), partner Ey (down y)
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's OWN axis, half-integer
        gwx, gwy, gwz,                  # RUNTIME mirror ghost weights, exactly +1.0 / -1.0
        nx, ny, nz, n_elem,
        R01: tl.constexpr, R02: tl.constexpr,
        R11: tl.constexpr, R12: tl.constexpr,
        R21: tl.constexpr, R22: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        MG_X: tl.constexpr, MG_Y: tl.constexpr, MG_Z: tl.constexpr,
        WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``stepping.update_E`` with off-diagonal rows on a MIRROR-FOLDED grid.

        :func:`offdiag_update_e.offdiag_constitutive_step`'s body with the
        boundary constexpr set widened from two values to four and ONE arm added
        to the DOWN shift. Everything else — the row-mask arms, the wall mask,
        the prev-before-store ordering, the two separate accumulations, the
        own-axis half-integer coefficient index — is that kernel's, unchanged,
        because the array path's is unchanged.

        THE THREE DIFFERENCES FROM THE CERTIFIED KERNEL, all named:

        1. DOWN, on a MIRROR axis (either termination): the ghost lane at face 0
           is redirected to stored row ``MIRROR_ROW`` and weighted by that
           axis's runtime ``gw`` (``-phase``; docstring point A).
           ``stepping._shift_down`` :1823-1826.
        2. UP, on a MIRROR axis: mask and serve ``other=0.0``, byte-identical to
           the inherited METALLIC arm — ``_offdiagonal_terms`` calls
           ``_shift_up`` without ``component``/``reflect_row``, so both mirror
           terminations take stepping.py:1828-1830 (docstring point 2).
        3. The two MIRROR codes therefore behave identically here; both are
           carried anyway, and the gate holds the equality as a null control
           (docstring point D).

        ``MG_*`` say whether each axis carries the mirror ghost weight. They are
        the SAME fact as ``BC* in MIRROR_CODES``, derived once on the host by
        :func:`mirror_ghost_axes` and validated against the codes by the plan,
        and they exist as their own constexprs so the term helper's ``else`` arm
        can be the certified body VERBATIM (see :func:`_folded_offdiag_term`)
        rather than a multiply-by-1.0 a compiler is trusted to remove.

        ``WM_*`` remain the mask's own declaration, carried separately from the
        ghost codes exactly as the fused pairs keep ``ZM_*`` beside ``BC*``; on
        a folded axis they are 0 by construction. Coefficient pointers of dead
        slots are bound by the plan to the component's own D pointer and never
        read.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        at_x, at_y, at_z = i == 0, j == 0, k == 0

        # --- per-axis neighbour indices, both directions, with the ghost rule.
        # stepping._shift_down (:1787) / _shift_up (:1723): PERIODIC wraps;
        # METALLIC masks both faces; MIRROR redirects the DOWN ghost to stored
        # row MIRROR_ROW with the plane's weight and masks the UP ghost.
        di, dj, dk = i - 1, j - 1, k - 1
        ui, uj, uk = i + 1, j + 1, k + 1
        dvx, dvy, dvz = live, live, live
        uvx, uvy, uvz = live, live, live
        wx, wy, wz = 1.0, 1.0, 1.0
        if BCX == PERIODIC:
            di = tl.where(di < 0, nx - 1, di)
            ui = tl.where(ui == nx, 0, ui)
        elif BCX == METALLIC:
            dvx = live & (di >= 0)
            uvx = live & (ui < nx)
        else:  # MIRROR_METALLIC or MIRROR_PERIODIC
            di = tl.where(at_x, MIRROR_ROW, di)
            wx = tl.where(at_x, gwx, 1.0)
            uvx = live & (ui < nx)
        if BCY == PERIODIC:
            dj = tl.where(dj < 0, ny - 1, dj)
            uj = tl.where(uj == ny, 0, uj)
        elif BCY == METALLIC:
            dvy = live & (dj >= 0)
            uvy = live & (uj < ny)
        else:
            dj = tl.where(at_y, MIRROR_ROW, dj)
            wy = tl.where(at_y, gwy, 1.0)
            uvy = live & (uj < ny)
        if BCZ == PERIODIC:
            dk = tl.where(dk < 0, nz - 1, dk)
            uk = tl.where(uk == nz, 0, uk)
        elif BCZ == METALLIC:
            dvz = live & (dk >= 0)
            uvz = live & (uk < nz)
        else:
            dk = tl.where(at_z, MIRROR_ROW, dk)
            wz = tl.where(at_z, gwz, 1.0)
            uvz = live & (uk < nz)

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0: Ex — own axis x; partners Dy (down y) then Dz (down z)
        prev0 = tl.load(w0 + idx, mask=live, other=0.0)   # BEFORE the store.
        gs0 = tl.load(g0 + idx, mask=live, other=0.0)
        us0 = tl.load(e0 + idx, mask=live, other=0.0)
        if R01:
            total0 = _folded_offdiag_term(
                g1, u01, idx,
                i * nyz + dj * nz + k,
                ui * nyz + j * nz + k,
                ui * nyz + dj * nz + k,
                live, dvy, uvx, uvx & dvy, wy, MG_Y)
            if R02:
                total0 = total0 + _folded_offdiag_term(
                    g2, u02, idx,
                    i * nyz + j * nz + dk,
                    ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk,
                    live, dvz, uvx, uvx & dvz, wz, MG_Z)
            src0 = _masked_row_sum(gs0 * us0, total0, at_y, at_z, WM_Y, WM_Z)
        else:
            if R02:
                total0 = _folded_offdiag_term(
                    g2, u02, idx,
                    i * nyz + j * nz + dk,
                    ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk,
                    live, dvz, uvx, uvx & dvz, wz, MG_Z)
                src0 = _masked_row_sum(gs0 * us0, total0, at_y, at_z,
                                       WM_Y, WM_Z)
            else:
                src0 = gs0 * us0
        tl.store(w0 + idx, src0, mask=live)
        a0 = tl.load(f0 + idx, mask=live, other=0.0)
        a0 = a0 + kp_0 * src0
        a0 = a0 - km_0 * prev0
        tl.store(f0 + idx, a0, mask=live)

        # --- component 1: Ey — own axis y; partners Dz (down z) then Dx (down x)
        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        gs1 = tl.load(g1 + idx, mask=live, other=0.0)
        us1 = tl.load(e1 + idx, mask=live, other=0.0)
        if R11:
            total1 = _folded_offdiag_term(
                g2, u11, idx,
                i * nyz + j * nz + dk,
                i * nyz + uj * nz + k,
                i * nyz + uj * nz + dk,
                live, dvz, uvy, uvy & dvz, wz, MG_Z)
            if R12:
                total1 = total1 + _folded_offdiag_term(
                    g0, u12, idx,
                    di * nyz + j * nz + k,
                    i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k,
                    live, dvx, uvy, uvy & dvx, wx, MG_X)
            src1 = _masked_row_sum(gs1 * us1, total1, at_x, at_z, WM_X, WM_Z)
        else:
            if R12:
                total1 = _folded_offdiag_term(
                    g0, u12, idx,
                    di * nyz + j * nz + k,
                    i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k,
                    live, dvx, uvy, uvy & dvx, wx, MG_X)
                src1 = _masked_row_sum(gs1 * us1, total1, at_x, at_z,
                                       WM_X, WM_Z)
            else:
                src1 = gs1 * us1
        tl.store(w1 + idx, src1, mask=live)
        a1 = tl.load(f1 + idx, mask=live, other=0.0)
        a1 = a1 + kp_1 * src1
        a1 = a1 - km_1 * prev1
        tl.store(f1 + idx, a1, mask=live)

        # --- component 2: Ez — own axis z; partners Dx (down x) then Dy (down y)
        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        gs2 = tl.load(g2 + idx, mask=live, other=0.0)
        us2 = tl.load(e2 + idx, mask=live, other=0.0)
        if R21:
            total2 = _folded_offdiag_term(
                g0, u21, idx,
                di * nyz + j * nz + k,
                i * nyz + j * nz + uk,
                di * nyz + j * nz + uk,
                live, dvx, uvz, uvz & dvx, wx, MG_X)
            if R22:
                total2 = total2 + _folded_offdiag_term(
                    g1, u22, idx,
                    i * nyz + dj * nz + k,
                    i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk,
                    live, dvy, uvz, uvz & dvy, wy, MG_Y)
            src2 = _masked_row_sum(gs2 * us2, total2, at_x, at_y, WM_X, WM_Y)
        else:
            if R22:
                total2 = _folded_offdiag_term(
                    g1, u22, idx,
                    i * nyz + dj * nz + k,
                    i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk,
                    live, dvy, uvz, uvz & dvy, wy, MG_Y)
                src2 = _masked_row_sum(gs2 * us2, total2, at_x, at_y,
                                       WM_X, WM_Y)
            else:
                src2 = gs2 * us2
        tl.store(w2 + idx, src2, mask=live)
        a2 = tl.load(f2 + idx, mask=live, other=0.0)
        a2 = a2 + kp_2 * src2
        a2 = a2 - km_2 * prev2
        tl.store(f2 + idx, a2, mask=live)

else:  # pragma: no cover - the laptop path
    folded_offdiag_constitutive_step = None  # type: ignore[assignment]


def folded_offdiag_constitutive_step_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError."""
    if folded_offdiag_constitutive_step is None:
        raise ImportError(
            "the folded off-diagonal update_E kernel needs the optional "
            "`triton` package (pip install triton). The engine runs without "
            "it; only this fast path is unavailable. Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return folded_offdiag_constitutive_step


# ---------------------------------------------------------------------------
# Small host helpers
# ---------------------------------------------------------------------------

def mirror_ghost_weights(grid: Any) -> Tuple[float, float, float]:
    """The ``gwx``/``gwy``/``gwz`` runtime scalars, derived the array path's way.

    ``stepping._offdiagonal_terms`` (:1214-1216) passes ``component =
    "D" + AXIS_NAMES[partner_axis]`` on ``axis = partner_axis`` and the plane's
    declared phase, and ``_shift_down``'s MIRROR branch (:1824-1825) weights the
    ghost by ``_symmetry_phase(...) == fields.mirror_parity(...)``. Every D
    component has Yee shift 1 on its OWN axis, so the weight is ``-phase`` on
    every axis and for every partner (docstring point A).

    Derived through ``fields.mirror_parity`` ITSELF rather than restating the
    collapse, so the two cannot drift; the collapse is asserted by a test.
    ``1.0`` on an unfolded axis, where the ghost lane never fires and the value
    is never read.
    """
    try:  # pragma: no cover - the engine is always importable in practice
        from ..fields import mirror_parity  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - unreadable means the plan refuses
        mirror_parity = None  # type: ignore[assignment]
    out: List[float] = []
    for axis in range(3):
        if not bool(_coverage._call(grid, "is_mirrored", axis, default=False)):
            out.append(1.0)
            continue
        phase = _coverage._call(grid, "mirror_phase", axis, default=None)
        if phase not in (1, -1) or mirror_parity is None:
            # An unreadable plane is a refusal, not a default: the predicate
            # names it (through symmetry.folded_axis_kinds, symmetry.py:511-514)
            # and the plan builder refuses before this value can be launched.
            out.append(float("nan"))
            continue
        out.append(float(mirror_parity("D" + "xyz"[axis], axis, int(phase))))
    return (out[0], out[1], out[2])


def mirror_ghost_axes(boundary_codes: Sequence[int]) -> Tuple[int, int, int]:
    """The ``MG_X``/``MG_Y``/``MG_Z`` constexprs — does this axis carry the
    mirror ghost weight?

    The SAME fact as ``code in MIRROR_CODES``, derived in ONE place so the
    constexpr and the index-redirect branch cannot disagree: the redirect keys
    off ``BC*`` and the weight arm off ``MG_*``, and if the two ever disagreed
    the kernel would read stored row 2 without the parity, or apply the parity
    to an ordinary neighbour — a plane of wrong values, not a crash. The plan
    validates the pair; a test pins the derivation.

    It is a separate constexpr rather than an ``or`` inside the kernel for a
    concrete reason (:func:`_folded_offdiag_term`): it lets the unfolded arm be
    the certified body VERBATIM instead of a multiply by 1.0 the compiler is
    trusted to remove.
    """
    return tuple(  # type: ignore[return-value]
        int(int(code) in MIRROR_CODES) for code in boundary_codes)


def mirror_arm_is_reachable(grid: Any, fields: Any) -> Tuple[bool, Tuple[str, ...]]:
    """Can the mirror ghost arm change a byte for THIS (grid, rows) pair?

    The reachability statement of the module docstring, as a function. THE TEST
    IS AT SLOT LEVEL, NOT ROW LEVEL, and the difference is a measured
    counterexample rather than a nicety: the mirror rule on axis ``a`` enters
    this sub-step ONLY through the down shift of the PARTNER axis
    (``_shift_down`` is called with ``axis = partner_axis``, stepping.py:
    1243-1245), so a fold on ``a`` is byte-visible iff some SURVIVING row slot
    takes ``E_a`` AS ITS PARTNER. A row may be live through its OTHER slot and
    still never touch the fold.

    MEASURED (laptop, NumPy, same grid / coefficients / seeds, flipping only the
    folded axis's ghost code, three chained sub-step calls): fold X with the
    single live slot ``Ey <- Ez`` gives MIRROR bytes == METALLIC bytes, i.e.
    UNREACHABLE — while a row-level test ("some live row is not ``Ex``") answers
    reachable, because the live row IS ``Ey``. That configuration is the gate's
    own ``foldX_single_Ey_Ez_no_x_partner`` reference grid and its
    ``single_Ey_Ez`` sweep case, and it is now carried in the gate's
    ``identity_host`` leg, which fails on any disagreement between this function
    and the byte measurement.

    Returns ``(reachable, notes)``. Not a coverage clause and deliberately not a
    routing rule — routing by predicate order would make the numerical method a
    function of composer order (symmetry.py:713-722). It is what the gate's
    IDENTITY leg asserts against, and what a later ``plan_step`` round may use
    to prefer the certified kernel where the two are measured equal.
    """
    notes: List[str] = []
    rows = _offdiag.row_volumes_for(fields)
    live_slots = tuple((row, partner) for (row, partner), value
                       in zip(ROW_SLOTS, rows) if value is not None)
    reachable = False
    for axis in range(3):
        if not bool(_coverage._call(grid, "is_mirrored", axis, default=False)):
            continue
        own = "E" + "xyz"[axis]
        crossing = sorted(f"{row}<-{partner}" for row, partner in live_slots
                          if partner == own)
        if crossing:
            reachable = True
            notes.append(
                f"axis {axis} is folded and is the PARTNER axis of live slot(s) "
                f"{crossing}: the mirror ghost is byte-visible there")
        else:
            live = sorted(f"{row}<-{partner}" for row, partner in live_slots)
            notes.append(
                f"axis {axis} is folded but no live row slot takes {own} as its "
                f"partner (live slots {live}): the mirror arm is unreachable and "
                f"this axis is byte-identical to the certified kernel's METALLIC "
                f"code")
    return reachable, tuple(notes)


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------

def _folded_offdiag_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """Everything :func:`coverage._grid_reasons` refuses EXCEPT the fold, plus
    the fold's own classification.

    RE-STATED, not imported-and-subtracted, for :mod:`symmetry`'s own reason
    (symmetry.py:548-556): subtracting a reason string from another predicate's
    output would make this file's coverage a function of that file's phrasing,
    and a clause renamed there would silently widen coverage here. Clause
    numbering follows ``coverage._grid_reasons`` so the two read side by side.

    Deliberately NOT :func:`symmetry._shared_grid_reasons` either: that helper
    is the folded CURL's contract and refuses conductivity, which changes the
    curl recurrence and nothing in ``update_E`` (coverage.py:128-130). Inheriting
    a curl-only refusal would silently narrow an independent sub-step.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernel launches against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. Complex fold + offdiag is Phase B's, composed with
    #    folded_complex.py; the recurrence has the same shape and the STORAGE
    #    does not, so a complex run stepped as float32 reads the wrong stride.
    if getattr(fields, "force_complex_fields", False):
        reasons.append(
            "force_complex_fields=True: folded complex64 storage belongs to the "
            "folded_complex family, and folded complex + off-diagonal is a "
            "later composed leg")

    # 3. An absorber that actually absorbs. Without one ``update_E`` takes the
    #    plain ``field[...] = constitutive`` branch (stepping.py:1022), a
    #    different sub-step.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this kernel implements the "
                       "split-field dsigw path only)")

    # 4/5. The ghost rules INCLUDING the two folded ones, classified by
    #      symmetry.folded_axis_kinds — which refuses an unresolvable grid, a
    #      folded axis with no owned_cells(), a mirror phase that is not +/-1, a
    #      folded axis with <= MIRROR_SOURCE_INDEX stored cells (the ghost
    #      images stored row 2; stepping._mirror_source raises below that), and
    #      a folded axis whose two independent routes to "periodic or metallic"
    #      disagree. Nothing is decided here.
    codes, fold_reasons = _symmetry.folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False))
        else None)
    reasons.extend(fold_reasons)

    # 6. Cartesian only. The cylindrical r axis has its own ghost rule (the
    #    r_to_minus_r image, stepping.py:1877-1888) and its own axial extent.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _coverage._call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0 on every axis. A Bloch phase needs complex storage and multiplies
    #    one wrapped plane; a folded axis refuses a nonzero k outright
    #    (driver._require_bloch_is_representable) and a kernel cannot lift what
    #    the array path will not run.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. No instantaneous nonlinearity. The Pade factor REPLACES the
    #     constitutive product (stepping.py:999-1000) and, on a fold, the
    #     transverse sums shift component PRODUCTS carrying no single parity —
    #     which the driver itself refuses once the far face is live
    #     (_require_folded_far_face_is_quiet, driver.py:2979). Refused here
    #     regardless of the far face.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append(
            "chi2/chi3 is installed: the Pade factor replaces the row product "
            "(stepping.py:1095-1116) and a nonlinear fold has no single parity")

    # 11/12. BFAST adds a second additive curl term; beta adds out-of-plane
    #        couplings — and the ENGINE additionally raises for
    #        real-storage + offdiag + beta (stepping.py:800-810, MEEP
    #        fields.cpp:548-549). Both are silent additions, not errors.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    if codes is None and not reasons:  # pragma: no cover - belt and braces
        reasons.append("the per-axis boundary codes could not be resolved")
    return reasons


def folded_offdiag_constitutive_coverage(fields: Any, pml: Any) -> "_coverage.Coverage":
    """May the folded off-diagonal kernel step ``update_E`` for this pair?

    POSITIVE CLAUSES ONLY; a failing clause appends its reason and the scan
    continues, so a refusal reports everything that disqualified the run rather
    than the first thing.

    Grid clauses: :func:`_folded_offdiag_grid_reasons` — ``_grid_reasons``'
    list with clause 5 (the fold) replaced by
    :func:`symmetry.folded_axis_kinds`, and conductivity deliberately absent
    (it changes the CURL sub-steps only, coverage.py:128-130).

    E-side clauses, each a silent wrong answer if missing — the certified
    off-diagonal predicate's own list (offdiag_update_e.py:605-693), restated:

    a. **no registered polarization.** With poles the source is ``D - sum P``
       formed in per-component scratch buffers (fields.py:1107-1138) and the
       coupling would read THOSE, not D. A later fused leg.
    b. **at least one surviving off-diagonal row (INVERTED), counted over the
       six ROW_SLOTS** — deliberately not the bare ``has_offdiagonal_epsilon``
       flag, which a row planted past the installer under a diagonal key sets
       while every slot is dead (the builder would raise where this predicate
       had admitted). A zero-slot folded run is
       :func:`symmetry.folded_constitutive_coverage`'s, disjoint by the
       install-time drop at fields.py:1302-1303.
    c. **stored E** — forced True by any surviving row (fields.py:1254-1255);
       belt and braces with a reason.
    d. every surviving row volume real, f32, C-contiguous, grid-shape, and
       aliasing no E/f_w output — :func:`offdiag_update_e._row_reasons`,
       imported rather than restated because it is the same clause about the
       same installer (fields.py:1296 keeps the caller's array without copying,
       so the alias is reachable without planting).
    e. volume inverse epsilon per component; this kernel carries NO scalar arm.
    f. layout and the half-integer coefficient tables, at the FOLDED STORED
       EXTENT — the clause that stops "the vectors are built at the stored
       length" from being an assumption (symmetry.py:690-694).
    g. **a readable mirror ghost weight on every folded axis.** The weight is a
       RUNTIME argument, so an unreadable plane would launch a NaN rather than
       raise; ``folded_axis_kinds`` already refuses a phase outside +/-1 and
       this clause names the launched value itself.

    ZERO FOLDED AXES IS ADMITTED, and that is not an oversight — it is
    :mod:`symmetry`'s own arrangement (symmetry.py:653-659). With every axis
    resolving to PERIODIC/METALLIC the kernel's constexpr branches reduce to the
    certified off-diagonal kernel's exactly, which the gate MEASURES rather than
    claims. Which of the two a composed plan launches is an integration
    decision, not a coverage one, and
    :func:`folded_offdiag_composition_coverage` is the narrower verdict
    ``plan_step`` will use.

    Wherever this predicate answers covered the plan builder must return a plan,
    never raise: the builder's own raises are backstops for harness callers, and
    each has a predicate clause in front of it on the engine route.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = _folded_offdiag_grid_reasons(fields, pml, grid)
    reasons.extend(_coverage._susceptibility_reasons(fields))

    # (a) No poles: the coupling must read the aliased D primaries.
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if states or getattr(fields, "has_polarizations", False):
        reasons.append(
            "a susceptibility is registered: the offdiag source becomes "
            "D - sum P in per-component scratch buffers (fields.py:1107-1138) "
            "— the fused dispersive+offdiag+fold kernel is a later leg")

    # (b) INVERTED: at least one surviving row SLOT.
    if not any(value is not None for value in _offdiag.row_volumes_for(fields)):
        reasons.append(
            "no off-diagonal chi1inv row survived installation: that "
            "configuration is symmetry.folded_constitutive_coverage(side='E')'s "
            "and this predicate must not overlap it")

    # (c) Stored E.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # (d) The surviving rows, readable and well-formed — the certified clause.
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_offdiag._row_reasons(fields, shape))

    # (e)/(f) The volumes this sub-step reads and writes, and their layout.
    names = tuple(term[0] for term in E_TERMS)
    names += tuple("f_w_" + term[0] for term in E_TERMS)
    names += tuple(term[1] for term in E_TERMS)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_coverage._layout_reasons(fields, shape, names))
    if len(shape) == 3:
        reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))
        if pml is not None and getattr(pml, "is_active", False):
            reasons.extend(_coverage._coefficient_reasons(
                pml, shape, ("kps", "kms"), ("_h",) if HALF_INTEGER else ("",)))

    # (g) The launched ghost weights.
    for axis, weight in enumerate(mirror_ghost_weights(grid)):
        if weight != weight or weight not in (1.0, -1.0):
            reasons.append(
                f"axis {axis} mirror ghost weight is {weight!r}, not +1.0 or "
                f"-1.0; the weight is a RUNTIME argument and an unreadable "
                f"plane would launch rather than raise")

    return _coverage.Coverage(not reasons, tuple(reasons))


def folded_offdiag_composition_coverage(fields: Any, pml: Any) -> "_coverage.Coverage":
    """The narrower verdict for composition, where an actual fold is mandatory.

    :func:`folded_offdiag_constitutive_coverage` deliberately admits an unfolded
    grid so the standalone gate can prove reduction to the certified
    off-diagonal kernel. That is not a routing rule: selecting between two valid
    products by branch order would make the numerical method depend on composer
    order. This is the verdict a later ``plan_step`` round uses — the same split
    :func:`symmetry.folded_composition_curl_coverage` makes, for the same
    reason.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))
    base = folded_offdiag_constitutive_coverage(fields, pml)
    reasons = list(base.reasons)
    if not _symmetry._has_real_fold(grid):
        reasons.append(
            "no mirror plane is active: the folded off-diagonal product is an "
            "equivalence product here, not a composition candidate — the "
            "certified offdiag_update_e family owns the unfolded run")
    return _coverage.Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedOffdiagConstitutivePlan:
    """A launchable, allocation-free folded off-diagonal ``update_E``.

    A SIBLING of :class:`offdiag_update_e.OffdiagConstitutivePlan`, deliberately
    not a subclass: the two hold nearly the same bindings but launch different
    kernels with different signatures, and a subclass that inherited ``run``
    would launch the certified one — the exact trap
    :class:`symmetry.FoldedPmlCurlPlan` documents (symmetry.py:920-923).

    The Yee sub-lattice is chosen at the BUILDERS and nowhere else:
    ``kps_a_h``/``kms_a_h``, the half-integer tables (stepping.py:1015). The six
    ROW SLOTS are bound in :data:`ROW_SLOTS` order; a dead slot's pointer is
    bound to the component's own D pointer (never read — the constexpr arm is
    what stops the read, the pointer still has to type). The WALL AXES triple is
    the mask's own question (:func:`offdiag_update_e.wall_mask_axes`), carried
    separately from the boundary codes exactly as the fused pairs carry ``ZM_*``
    beside ``BC*``. The GHOST WEIGHTS are runtime scalars and are validated to
    be exactly +1.0 or -1.0 here, because a NaN weight would launch silently.

    ``__init__`` REFUSES ALIASING between the outputs (E, f_w) and any input
    (D, inverse epsilon, row volumes, the six coefficient vectors), and among
    the outputs themselves: the coupling re-reads the partner D volumes at
    neighbour offsets — on a folded axis including stored row 2, which is deep
    INSIDE the volume rather than at a face — while E and f_w are being written,
    so an aliased pair would make the result depend on block schedule.
    """

    __slots__ = ("shape", "n_elem", "block", "row_mask", "boundary_codes",
                 "wall_axes", "ghost_weights", "ghost_axes", "_targets", "_aux",
                 "_sources", "_inv_eps", "_rows", "_coefficients", "_kernel")

    def __init__(self, shape, block: int, targets, auxiliaries, sources,
                 inverse_epsilon, rows, coefficients, boundary_codes,
                 wall_axes, ghost_weights, kernel: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)

        self.boundary_codes = tuple(int(code) for code in boundary_codes)
        if len(self.boundary_codes) != 3 or any(
                code not in (CODE_PERIODIC, CODE_METALLIC,
                             CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
                for code in self.boundary_codes):
            raise ValueError(
                f"boundary codes must be three of "
                f"{{{CODE_PERIODIC}, {CODE_METALLIC}, {CODE_MIRROR_METALLIC}, "
                f"{CODE_MIRROR_PERIODIC}}}, got {boundary_codes!r}")
        self.wall_axes = tuple(int(flag) for flag in wall_axes)
        if len(self.wall_axes) != 3 or any(
                flag not in (0, 1) for flag in self.wall_axes):
            raise ValueError(f"wall axes must be three of {{0, 1}}, got "
                             f"{wall_axes!r}")
        for axis, (code, flag) in enumerate(zip(self.boundary_codes,
                                                self.wall_axes)):
            if code in MIRROR_CODES and flag:
                raise ValueError(
                    f"axis {axis} is folded AND wall-masked; "
                    f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                    f"(stepping.py:1282) and zeroing the fold plane costs "
                    f"2.0e-02 (stepping.py:1269-1274)")

        self.ghost_weights = tuple(float(w) for w in ghost_weights)
        if len(self.ghost_weights) != 3:
            raise ValueError(f"ghost weights must be three floats, got "
                             f"{ghost_weights!r}")
        # ONE derivation, checked here: the index redirect keys off BC* and the
        # weight arm off MG_*, and a disagreement reads stored row 2 without the
        # parity (or applies the parity to an ordinary neighbour).
        self.ghost_axes = mirror_ghost_axes(self.boundary_codes)
        for axis, (code, flag) in enumerate(zip(self.boundary_codes,
                                                self.ghost_axes)):
            if bool(flag) != (code in MIRROR_CODES):  # pragma: no cover
                raise ValueError(
                    f"axis {axis}: MG={flag} and boundary code {code} disagree "
                    f"about the mirror ghost")
        for axis, (code, weight) in enumerate(zip(self.boundary_codes,
                                                  self.ghost_weights)):
            if code in MIRROR_CODES and weight not in (1.0, -1.0):
                raise ValueError(
                    f"axis {axis} is folded with ghost weight {weight!r}; the "
                    f"mirror ghost carries mirror_parity('D'+axis, axis, "
                    f"phase) == -phase, exactly +1.0 or -1.0 "
                    f"(fields.py:117, stepping.py:1871-1872)")

        targets = tuple(targets)
        auxiliaries = tuple(auxiliaries)
        sources = tuple(sources)
        inverse_epsilon = tuple(inverse_epsilon)
        rows = tuple(rows)
        if not (len(targets) == len(auxiliaries) == len(sources)
                == len(inverse_epsilon) == 3):
            raise ValueError("a plan carries exactly three components")
        if len(rows) != len(ROW_SLOTS):
            raise ValueError(f"rows must fill the {len(ROW_SLOTS)} slots of "
                             f"ROW_SLOTS (None where dead), got {len(rows)}")
        for index, value in enumerate(inverse_epsilon):
            if not _offdiag._is_volume(value):
                raise ValueError(
                    f"inverse epsilon {index} is not a volume: this family "
                    f"carries no scalar arm")
        # The mirror ghost images stored row MIRROR_SOURCE_INDEX, so a folded
        # axis needs more than that many cells — stepping._mirror_source
        # (:1536-1541) raises, and a kernel would read a neighbour's plane.
        for axis, code in enumerate(self.boundary_codes):
            if code in MIRROR_CODES and self.shape[axis] <= MIRROR_SOURCE_INDEX:
                raise ValueError(
                    f"axis {axis} is folded with {self.shape[axis]} stored "
                    f"cells; the mirror ghost images stored row "
                    f"{MIRROR_SOURCE_INDEX}")

        self.row_mask = tuple(int(value is not None) for value in rows)
        if not any(self.row_mask):
            raise ValueError(
                "no row slot survives: that configuration belongs to "
                "symmetry.plan_folded_constitutive, and building this plan for "
                "it would overlap the two")

        self._require_no_aliasing(targets, auxiliaries, sources,
                                  inverse_epsilon, rows, coefficients)

        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        component_of_slot = tuple(
            next(index for index, term in enumerate(E_TERMS)
                 if term[0] == row)
            for row, _partner in ROW_SLOTS)
        self._rows = tuple(
            CupyPointer(value) if value is not None
            else self._sources[component_of_slot[index]]
            for index, value in enumerate(rows))
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._kernel = kernel

    @property
    def launch_grid(self) -> Tuple[int]:
        return ((self.n_elem + self.block - 1) // self.block,)

    @staticmethod
    def _require_no_aliasing(targets, auxiliaries, sources, inverse_epsilon,
                             rows, coefficients) -> None:
        """Base-address disjointness of outputs vs outputs and inputs vs
        outputs — the six coefficient vectors included in the input inventory.
        Equality of BASE addresses only: two overlapping views with different
        bases pass unseen, the certified plans' shared (accepted) limitation.
        Inputs may alias EACH OTHER freely (three aliased inverse-epsilon
        volumes are the isotropic install)."""
        outputs: Dict[int, str] = {}
        for name_group, group in (("target", targets), ("aux", auxiliaries)):
            for index, array in enumerate(group):
                address = _offdiag._base_address(array)
                if address is None:
                    raise ValueError(
                        f"{name_group} {index} exposes no readable base "
                        f"address; an unverifiable output is not accepted")
                if address in outputs:
                    raise ValueError(
                        f"{name_group} {index} aliases {outputs[address]}; "
                        f"the outputs must be distinct arrays")
                outputs[address] = f"{name_group} {index}"
        inputs = list(sources) + list(inverse_epsilon)
        inputs += [value for value in rows if value is not None]
        inputs += [value for value in coefficients
                   if _offdiag._is_volume(value)]
        for array in inputs:
            address = _offdiag._base_address(array)
            if address is not None and address in outputs:
                raise ValueError(
                    f"an input volume aliases {outputs[address]}: the coupling "
                    f"re-reads the partner D volumes at neighbour offsets — on "
                    f"a folded axis including interior stored row "
                    f"{MIRROR_SOURCE_INDEX} — while the outputs are written, "
                    f"so an alias makes the answer depend on block schedule")

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_offdiag_constitutive_step_kernel())
        nx, ny, nz = self.shape
        kernel[self.launch_grid](
            *self._targets, *self._aux, *self._sources, *self._inv_eps,
            *self._rows, *self._coefficients, *self.ghost_weights,
            nx, ny, nz, self.n_elem,
            R01=self.row_mask[0], R02=self.row_mask[1],
            R11=self.row_mask[2], R12=self.row_mask[3],
            R21=self.row_mask[4], R22=self.row_mask[5],
            BCX=self.boundary_codes[0], BCY=self.boundary_codes[1],
            BCZ=self.boundary_codes[2],
            MG_X=self.ghost_axes[0], MG_Y=self.ghost_axes[1],
            MG_Z=self.ghost_axes[2],
            WM_X=self.wall_axes[0], WM_Y=self.wall_axes[1],
            WM_Z=self.wall_axes[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"FoldedOffdiagConstitutivePlan(shape={self.shape}, "
                f"rows={self.row_mask}, bc={self.boundary_codes}, "
                f"mg={self.ghost_axes}, walls={self.wall_axes}, "
                f"gw={self.ghost_weights}, block={self.block})")


def plan_folded_offdiagonal_constitutive(
        fields: Any, pml: Any, block: Optional[int] = None
        ) -> Optional[FoldedOffdiagConstitutivePlan]:
    """Build the folded off-diagonal ``update_E`` plan from the engine's objects.

    None means REFUSED, and the reasons are available from
    :func:`folded_offdiag_constitutive_coverage`. The sources are taken through
    ``displacement_minus_polarization_volumes`` — the accessor the array path
    itself reads (stepping.py:996, :1004) — which with no poles admitted hands
    back the aliased D primaries. The Triton import stays below the predicate: a
    NumPy host must be able to plan (to ``None``) without the optional
    dependency being importable at all.
    """
    if not folded_offdiag_constitutive_coverage(fields, pml).covered:
        return None
    codes, _reasons = _symmetry.folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    targets = tuple(term[0] for term in E_TERMS)
    volumes = fields.displacement_minus_polarization_volumes()
    return FoldedOffdiagConstitutivePlan(
        fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        [getattr(fields, "f_w_" + name) for name in targets],
        [volumes[name] for name in targets],
        [fields.inverse_epsilon_for(name) for name in targets],
        _offdiag.row_volumes_for(fields),
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        codes,
        _offdiag.wall_mask_axes(fields.grid),
        mirror_ghost_weights(fields.grid),
    )


def plan_folded_offdiagonal_constitutive_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any],
        rows: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[int], wall_axes: Sequence[int],
        ghost_weights: Sequence[float],
        block: Optional[int] = None, kernel: Any = None
        ) -> FoldedOffdiagConstitutivePlan:
    """Build it from bare arrays — the gate's and validator's route.

    ``arrays`` is keyed by component name (``Ex``, ``f_w_Ex``, ``Dx``,
    ``inv_eps_Ex``), ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE
    CALLER already selected, and ``rows`` maps row component -> partner
    component -> coefficient volume, with absent entries marking dead slots (the
    caller performs the install-time zero-row drop itself). No coverage
    predicate runs here: the caller is a harness that constructed the
    configuration on purpose, including the deliberately wrong ones.

    ``kernel=`` IS LOAD-BEARING: the mutation legs route a mutated kernel
    through it, and a builder that drops it launches the shipped kernel and
    reports a pass for a defect it never introduced (the measured harness-disarm
    failure the certified families document).
    """
    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return FoldedOffdiagConstitutivePlan(
        shape, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets],
        [arrays["f_w_" + name] for name in targets],
        [arrays[term[1]] for term in E_TERMS],
        [arrays["inv_eps_" + name] for name in targets],
        [(rows.get(row) or {}).get(partner) for row, partner in ROW_SLOTS],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        boundary_codes,
        wall_axes,
        ghost_weights,
        kernel=kernel,
    )


def explain_folded_offdiag(fields: Any, pml: Any) -> "_coverage.Coverage":
    """The coverage verdict with its reasons. Needs no Triton."""
    return folded_offdiag_constitutive_coverage(fields, pml)


# ---------------------------------------------------------------------------
# INTEGRATION — what the later wiring round has to do, and nothing more
# ---------------------------------------------------------------------------
#
# Additive edits to the shared files, which are NOT touched from here:
#
#   coverage.py  clause 5's fold refusal (coverage.py:171-175) STAYS — it is
#                the disjointness seam for the unfolded families, not a stale
#                refusal. plan_step gains one branch: where the fold clauses are
#                the ONLY E-side refusal and folded_offdiag_composition_coverage
#                admits, install plan_folded_offdiagonal_constitutive in the
#                update_E slot, beside symmetry's folded curls and ghost fills.
#                The composition probe's whole-step byte identity — folded curls
#                and folded ghost fills composing UNCHANGED over the new
#                update_E — is the measurement that licenses the branch. IT NOW
#                EXISTS AND IS GREEN: parity/meep_gpu/probe_triton_folded_
#                offdiag_composition.py, 6 cases, 50 complete driver steps,
#                5,293,056 uint32 words, 0 differing, both armed mutations
#                caught at step 1.
#
#                ONE CONSTRAINT THE WIRING ROUND INHERITS, measured by that
#                probe and NOT this file's to fix: symmetry.MirrorGhostFillPlan
#                combines the near and far ghost passes into one launch, while
#                the driver runs zero_metal_B/D BETWEEN the two fill calls
#                (driver.py:3208-3210, :3222-3224). On a grid carrying a folded
#                PERIODIC EVEN axis AND a metallic axis, installing the combined
#                plan at the near slot reorders the far ghosts across the metal
#                zeroing and is byte-visible (48 words at step 1, starting at
#                Bx). The probe attributes it: a run with NO folded off-diagonal
#                kernel in it diverges identically, and every install carrying
#                this kernel without the combined fill is exact. It belongs to
#                symmetry.py and to the composition install convention.
#   symmetry.py  folded_constitutive_coverage's offdiag clause (symmetry.py:
#                814-817) STAYS: it is the disjointness seam against THIS file.
#   __init__.py  export the kernel accessor, the predicate and the builders.
#   fingerprints.json  add this module and kernel, re-cut against the bytes the
#                gate certified.
#   stepping.py / fields.py / driver.py  the two findings this module records:
#                the stale fold-refusal docstrings (stepping.py:1219-1220,
#                fields.py:1237-1241, driver.py:1202-1204) and the folded
#                PERIODIC far-face zero ghost in _offdiagonal_terms that
#                _require_folded_far_face_is_quiet does not cover.
#
# Dispatch stays disabled throughout: the engine's fast-path hook returns None
# on every branch and nothing in this tranche changes that.
