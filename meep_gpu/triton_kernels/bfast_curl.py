"""BFAST (broadband fixed-angle source technique) PML curl tranche — Phase A:
real-f32 split-field-PML curl only.

NOT WIRED. Production dispatch is untouched: ``fastpath.plan_fast_path`` still
returns None on every branch and this module does not touch it, ``plan_step``
composition is DEFERRED (``coverage._grid_reasons`` clause 11 already refuses
every BFAST run for the shipped kernels, so these predicates cannot create an
admitted-overlap ambiguity until a later coordinated change adds launch-side
forwarders), and ``fingerprints.json`` carries no entry for this file — the
byte gate binds its own provenance record inside its results directory.
Callers are the gate (``parity/meep_gpu/gate_triton_bfast.py``), the
composition probe (``probe_triton_bfast_composition.py``) and the laptop
tests, nothing else.

WHAT THIS IS FOR. One corpus unlock, the last on the round-4 fixed list:
python/tests ``test_refl_angular::test_reflectance_angular_2_35_7`` (array-path
parity 4.415e-06, 1/1 published assertion reproduced). The marquee
configuration (MEEP test_refl_angular.py:15-95): dimensions=3, cell z-only
(9.0 incl. 2 x 1.0 PML), resolution 200, k_point exactly zero => REAL float32
storage, PML both z faces, GaussianSource Ex, two diagonal media n1=1.4 /
n2=3.5, ``bfast_scaled_k = (1.4*sin(35.7 deg), 0, 0)`` ~ (0.816958, 0, 0),
Courant ``(1 - kx)/sqrt(3)`` ~ 0.105679 — natively non-power-of-two. Phase A
(real-f32 split-field-PML curl) covers that demand in full; complex/Bloch +
BFAST is refused by name and queued with the complex family.

WHAT BFAST IS (stepping.py:826-931, grid.py:698-753). The time-sheared
substitution ``t -> t - k.r/c``: dB/dt = -curl E + d/dt(k x E), dD/dt =
+curl H - d/dt(k x H). MEEP runs it as a second additive pass over the SAME
shifted operands the curl gathers, per component right after its curl
(step_db.cpp:124-142), and so does the array path (``stepping._bfast_term``, then
``stepping._apply_folded_increment``'s ladder):

* call sites: step_B and step_D, per component after its curl form. The beta
  and BFAST passes would interleave if both were active, and their ORDER is
  byte-significant, which is why clause 12 (beta refused) is KEPT below;
* arithmetic per component (``stepping._bfast_term``, :func:`.forms.bfast_advance`),
  with first = MEEP g1 = f_p, shifted_first = g1[i+s], second = g2 = f_m::

      k1 = bfast[_bfast_axis(term.second)] if have_m else 0.0
      k2 = bfast[_bfast_axis(term.first)]  if have_p else 0.0
      if D side: k1, k2 = -k1, -k2   in host f64, then rounded once
      F   = (k1*(shifted_first + first) - k2*(shifted_second + second)) - F_prev
      raw = F - F_prev                  (a second rounding)
      state <- F on owned cells         (IN PLACE)
      increment = raw * -1.0, +0 on unowned cells, through the ladder

  With one partner the surviving term alone (MEEP's NULL swap below), and in
  MEEP's plain branch (no dsig or dsigu layer in the chunk) no ``- F_prev``;

  It is the Tustin filter (1 - z^-1)/(1 + z^-1): NO dtdx and NO dt anywhere —
  MEEP passes dtdx to step_bfast and never reads it (:836-837). MARGINALLY
  STABLE on purpose: the homogeneous mode is (-1)^n, undamped forever
  (:841-847) — subnormals and noise in f_bfast never decay;
* K-INDEXING TRAP: ``_bfast_axis`` (:787-796) indexes ``bfast_scaled_k`` by
  the partner component's OWN direction (MEEP component_index, vec.hpp:445),
  CROSS-assigned — k1 is indexed by the SECOND partner but multiplies the
  FIRST's sum. For Bx (first=Ez, second=Ey): k1 = k_y on the Ez sum, k2 = k_z
  on the Ey sum => (k x E)_x = k_y Ez - k_z Ey. Named at :791-794 as "the
  single easiest mistake in the whole pass", and silent;
* INVARIANT-AXIS GUARD: have_p = not is_invariant(term.first_axis), have_m =
  not is_invariant(term.second_axis) (:882-883), cross-gated (have_m gates
  k1). Load-bearing because an invariant axis zeroes the curl's DIFFERENCE
  automatically but the BFAST SUM is 2*g, not zero (:862-871).
  ``Grid.is_invariant`` reads DECLARED dimensionality, never extent == 1
  (grid.py:1165-1183) — the marquee is dimensions=3 with one-cell x/y axes,
  so all flags are TRUE there and the live cross terms survive.

  MEEP DOES TWO THINGS WHEN A FLAG IS FALSE, not one: it zeroes the
  COEFFICIENT (step_db.cpp:130-133) *and* it NULLS THE OPERAND ARRAY
  (step_db.cpp:62-63, ``f_p = have_p ? f[c_p][cmp] : NULL``). ``step_bfast``
  then swaps a null g1 to the g2 slot (step_generic.cpp:342-346, k1 and k2
  swapped with it) and takes its single-operand branch when g2 is null — in
  every split-field-PML branch spelled ``F[i] = k1*(g1[i+s1] + g1[i]) - F[i]``
  (:449, :469, :499 and :525), and in the plain no-PML no-fu no-cnd branch at
  :376 without the ``- F[i]`` (the inconsistency ``Grid._resolve_bfast`` cites
  for cylindrical). The array path takes those one-partner forms
  (``stepping._bfast_term``): the absent partner is not read, the surviving k
  multiplies the surviving partner's sum, and outside a layer chunk (MEEP's
  plain branch) the state drops ``- F``. So does this kernel
  (``forms.bfast_advance``), which is why clause 11a below ADMITS a nonzero k
  on a DECLARED-invariant axis and refuses only an unreadable k assignment: such
  a k multiplies a partner MEEP does not read, and the run stores the k = 0
  run's words (measured in test_triton_bfast.py on a declared-2-D grid, states,
  auxiliaries and fields, both sub-steps);
* D1 (dimensions=1) CARRIES A SECOND MEEP ASYMMETRY, benign and stated: MEEP's
  other reason for a false flag is ``gv.has_field``, which leaves Bx/Bz/Dy/Dz
  unallocated in D1 so ``step_db``'s ``if (f[cc][cmp])`` never steps them,
  where this engine steps all six. For a k along z all six (k1, k2) are zero,
  so from the physical all-zero state every state holds a signed zero (the D
  side's negated k makes a -0 of a positive partner sum) and every field and
  auxiliary word equals the non-BFAST run's or differs from it only in the sign
  of a zero (pinned in the laptop tests). Only an artificially seeded state
  makes them move, which is a harness configuration, not a run;
* NEW I/O: six IIR states ``f_bfast_{Dx..Bz}`` (fields.py:470-483),
  grid.shape, storage dtype, zero-initialized, allocated whenever
  ``grid.bfast_active`` INDEPENDENT of PML/conductivity (fields.py:632-651).
  The state has NO spatial stencil (:873-878) — one extra load + one store
  per target, no ghost read/write on it;
* GHOSTS: none new — the pass reads exactly the operands ``_curl_operands``
  gathered (:1560-1598), same backward-stride selection for D (:1567-1571;
  MEEP negates strides once, before BOTH calls, step_db.cpp:80-83);
* DRIVER INTERACTION: ``synchronize_magnetic_fields`` backs up and RESTORES
  f_bfast_Bx/By/Bz around the magnetic half-step (driver.py:4126-4135; MEEP
  energy_and_flux.cpp:113/:130). Because the IIR is marginally stable, a
  missed restore never decays. Consequence: the kernel MUST mutate
  ``fields.f_bfast_*`` IN PLACE (pointer-identical), or the sync
  backup/restore silently diverges;
* no constitutive involvement: update_H/update_E read nothing
  bfast-dependent (fields.py:227-230) — the pass is on step_db only.

ONE KERNEL: :func:`bfast_pml_curl_step` — ``kernels.pml_curl_step``'s body
(fusion off at the launch guard — the pass is multiply-subtract, so fusion-off
is load-bearing) plus the constexpr-gated BFAST pass after the forms: per target
:func:`.forms.bfast_advance` on the record's presence and layer bits, the
increment masked to ``+0`` and the state kept on the cells the body's owned-cell
predicate drops, the state stored in place, and the increment subtracted through
:func:`.forms.fold_ladder`.

GROUPING CHOICES the gate must hold (stepping.py forces none of these):

1. SIX host-rounded f32 scalars per launch — (k1, k2) per target, the D side
   pre-negated in host f64 then rounded ONCE (:886-887, :896-897
   ``dtype.type(k1)``), via :func:`bfast_curl_coefficients`. f64 negation and
   f32 rounding commute exactly; binding the negated values keeps the
   transcription literal per call site rather than resting on that identity
   (recorded as a predicted null in the gate, not armed).
2. ``total``'s grouping is ``(k1 * (g1s + g1c)) - (k2 * (g2s + g2c))`` with
   the SHIFTED operand first in each sum — the array path's operand order at
   :896-897. The subtraction IS the array path's single binary subtract.
3. The state takes MEEP's two roundings — ``F = S - F_prev`` stored, then
   ``raw = F - F_prev`` — never the one-rounding ``S - 2*F_prev`` (byte-visible;
   armed as ``f_one_rounding_advance``).
4. The increment is ``raw * -1.0`` and the ladder SUBTRACTS it, the array path's
   spelling (``x - (-y)`` is ``x + y`` in IEEE-754, signed zeros included;
   recorded as a predicted null, not armed). It never leans on Triton's
   unary-minus lowering (``0.0 - x`` canonicalizes signed zeros, measured; NO
   unary minus appears anywhere in the pass).
5. The state is stored in place (pointer-identical) and keeps its word on the
   cells the owned-cell predicate drops, as MEEP writes ``F`` inside its owned
   loop only (armed as ``f_state_stored_on_unowned_cells``).
6. The increment mask replicates, per target, exactly the ownership predicate
   the body's curl mask applies — metallic cell 0 on each axis where the
   target's Yee shift is 0 (armed as ``f_increment_unmasked``).
7. ``HAS_BFAST`` is a constexpr: engine-route plans always bind 1 (the
   predicate requires bfast_active; a k = 0 run belongs to the certified
   plain kernels, stepping.py:364/:451 never enter the fold); the 0 arm
   exists so the gate's identity leg can pin the HAS_BFAST=0 build
   byte-identical to ``kernels.pml_curl_step``.
8. NO per-target skip when a target's (k1, k2) are both zero: the array path
   runs the pass for every component with a partner whenever ANY component of k
   is nonzero, and with seeded state a zero-k target's ``F = -F_prev`` is
   byte-visible. No kernel arm specializes it away; a component with neither
   partner keeps its state and takes a ``+0`` increment.
9. Constitutive sub-steps on BFAST runs are NOT a new kernel: the restated
   predicate (:func:`bfast_run_constitutive_coverage`) inverts only the BFAST
   clause and delegates arithmetic to the certified
   ``kernels.constitutive_step`` through ``launch.ConstitutivePlan``.
10. The predicate's allocation/layout clauses are scoped to the NAMED
    sub-step's 12 arrays (3 targets + 3 fu + 3 sources + 3 f_bfast states)
    — the per-sub-step scoping the certified complex tranche uses; the
    engine allocates all of them together, so no engine-built configuration
    changes verdict.

REFUSAL ENUMERATION (Phase A predicate = ``coverage._grid_reasons`` clauses
with clause 11 INVERTED to require bfast_active, plus clause 11a which is this
family's OWN): REQUIRE bfast_active; a READABLE k assignment (11a —
``bfast_scaled_k`` and a callable ``is_invariant`` on every axis; a nonzero k on
a declared-invariant axis is admitted, the one-partner forms above); real
storage only (force_complex_fields AND any nonzero k_point refused — Grid
deliberately allows bfast+Bloch, grid.py:719-724, but the complex composition
is the complex family's queue); active split-field PML (no-PML bfast refused
by name; ``no_pml.py`` keeps its own bfast refusal); covered boundary kinds
only; no mirror plane / folded axis; Cartesian and not is_axis (cylindrical
is UNREACHABLE — ``Grid._resolve_bfast`` raises on MEEP's own
step_generic.cpp:376 missing '- F[i]' inconsistency, grid.py:706-742 — the
clause is kept anyway, per coverage.py's own doctrine of never inferring a
refusal from another module's guard); no chi2/chi3; beta == 0 KEPT (the fold
order beta-then-bfast is byte-significant, so bfast+beta is a coordinated
later change, not an accident — ``special_kz``'s predicates refuse bfast in
return); no conductivity on the curl targets in EITHER direction
(conductivity.py:515-518 already refuses bfast — no silent overlap).
Off-diagonal epsilon is ADMITTED for the curl exactly as the shipped curl
predicate admits it (constitutive-only; the offdiag family's own predicate
refuses bfast through the shared grid clauses). DO NOT invent a Courant
gate: MEEP neither derives nor clamps Courant for BFAST (grid.py:719-724).

Import contract: this module is importable WITHOUT Triton — the predicates
and plan builders (to ``None``) must answer on the laptop that is the merge
bar. Triton is imported at module scope inside a guard, the kernel degrades
to the ``_UnavailableKernel`` pattern, and ``ENABLE_FP_FUSION`` is imported
from :mod:`kernels` only inside ``run()`` so the guard keeps its single
spelling.
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
            f"host coverage remains available without it: {self.error}") from self.error


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

from .coverage import (  # READ-ONLY imports; nothing here mutates coverage.py
    CONSTITUTIVE_SIDES,
    COVERED_BOUNDARIES,
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

# The forms, the BFAST pass and the chunk record's word offsets, read by the kernel
# below. Imported only when Triton is (both modules import it at module scope), so the
# predicates and the plan builders still answer on a host without it.
if _TRITON_IMPORT_ERROR is None:
    from .forms import (axis_present, bfast_advance, curl_form,  # noqa: E402
                        curl_stencil, fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

# Same constexpr codes as ``kernels.PERIODIC``/``kernels.METALLIC``. Restated
# (not imported from kernels.py) because importing kernels.py would import
# Triton unconditionally and defeat this module's host-only coverage route;
# the laptop tests pin the restatements against their originators.
PERIODIC = tl.constexpr(0)
METALLIC = tl.constexpr(1)

#: Elements per program. Restated from ``kernels.DEFAULT_BLOCK`` for the same
#: import reason; no autotune (the kernel writes its own inputs in place,
#: kernels.py:56-59).
DEFAULT_BLOCK = 256

#: Per sub-step, per target index: (first, first_axis, second, second_axis) —
#: restated from ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS``
#: (stepping.py:214-223); a laptop test pins the restatement so it cannot
#: drift. ``first`` is MEEP's g1 = f_p (its sum takes k1), ``second`` is
#: g2 = f_m (its sum takes k2); the axes are the DERIVATIVE axes the guard
#: reads (:882-883), NOT the axes that index k.
BFAST_TERMS: Dict[str, Tuple[Tuple[str, int, str, int], ...]] = {
    "step_B": (("Ez", 1, "Ey", 2), ("Ex", 2, "Ez", 0), ("Ey", 0, "Ex", 1)),
    "step_D": (("Hz", 1, "Hy", 2), ("Hx", 2, "Hz", 0), ("Hy", 0, "Hx", 1)),
}

#: The three per-sub-step IIR state names, keyed like ``SUB_STEPS``.
BFAST_STATE_NAMES: Dict[str, Tuple[str, ...]] = {
    "step_B": ("f_bfast_Bx", "f_bfast_By", "f_bfast_Bz"),
    "step_D": ("f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"),
}

__all__ = [
    "BFAST_STATE_NAMES",
    "BFAST_TERMS",
    "BfastPmlCurlPlan",
    "bfast_curl_coefficients",
    "bfast_pml_curl_coverage",
    "bfast_pml_curl_step",
    "bfast_run_constitutive_coverage",
    "plan_bfast_pml_curl",
    "plan_bfast_pml_curl_from_arrays",
    "plan_bfast_run_constitutive",
]


# ---------------------------------------------------------------------------
# The host coefficients — stepping.py:814-823, :909-914, :923-924, transcribed
# ---------------------------------------------------------------------------

def _own_axis(component: str) -> int:
    """``stepping._bfast_axis`` — MEEP's component_index (vec.hpp:445): the
    component's OWN direction from its name's last letter, NEVER the direction
    its derivative is taken along (stepping.py:814-823)."""
    return "xyz".index(component[-1].lower())


def bfast_curl_coefficients(bfast_scaled_k: Sequence[float],
                            invariant_axes: Sequence[bool],
                            magnetic: bool) -> Tuple[float, ...]:
    """The six (k1, k2) scalars of one sub-step, host-computed and rounded
    EXACTLY as ``stepping._bfast_term`` computes and rounds them.

    Per target (stepping.py:909-914): ``have_p = not invariant[first_axis]``,
    ``have_m = not invariant[second_axis]`` — CROSS-gated, have_m gates k1;
    ``k1 = bfast[own_axis(second)]`` (multiplies the FIRST's sum), ``k2 =
    bfast[own_axis(first)]`` (multiplies the SECOND's sum); the D side negates
    BOTH in host f64 (:886-887); then ONE rounding to f32 (:896-897
    ``dtype.type(k1)`` — rounded at use, once). ``float()`` of a numpy.float32
    preserves the bits, and Triton types a Python float argument as fp32, so
    the words the kernel receives are the array path's.

    Returns ``(k1_0, k2_0, k1_1, k2_1, k1_2, k2_2)`` for the sub-step's three
    targets in term order. NO target is dropped when its pair is zero: the
    array path runs the tail for every component whenever any k component is
    nonzero (grid.py:744-753), and the kernel binds all six uniformly.
    """
    import numpy  # noqa: PLC0415

    bfast = tuple(float(value) for value in bfast_scaled_k)
    invariant = tuple(bool(value) for value in invariant_axes)
    if len(bfast) != 3 or len(invariant) != 3:
        raise ValueError("bfast_scaled_k and invariant_axes must be length-3")
    terms = BFAST_TERMS["step_B" if magnetic else "step_D"]
    out: List[float] = []
    for first, first_axis, second, second_axis in terms:
        have_p = not invariant[first_axis]
        have_m = not invariant[second_axis]
        k1 = bfast[_own_axis(second)] if have_m else 0.0  # :884
        k2 = bfast[_own_axis(first)] if have_p else 0.0   # :885
        if not magnetic:  # MEEP's `if (ft == D_stuff) { k1 = -k1; k2 = -k2; }`
            k1, k2 = -k1, -k2                             # :886-887, host f64
        out.append(float(numpy.float32(k1)))              # :896-897 — once
        out.append(float(numpy.float32(k2)))
    return tuple(out)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

@triton.jit
def bfast_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
    u0, u1, u2,                       # auxiliaries: fu_B*  or  fu_D*
    g0, g1, g2,                       # sources: Ex,Ey,Ez  or  Hx,Hy,Hz
    s0, s1, s2,                       # f_bfast states, one per target
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, one Yee sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,
    k1_0, k2_0, k1_1, k2_1, k1_2, k2_2,   # f32 (k1,k2) per target, host-rounded once
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    HAS_BFAST: tl.constexpr,          # compiled-in only for bfast_active runs
    BLOCK: tl.constexpr,
):
    """One REAL curl sub-step of all three components with MEEP's BFAST pass.

    ``kernels.pml_curl_step``'s body, character for character, plus ONE constexpr-gated
    insert after the three curl forms and before the stores: the BFAST pass
    (``stepping._bfast_term`` then ``stepping._apply_folded_increment``), which MEEP
    runs on each component right after its curl (step_db.cpp:124-142) on the same
    gathered partners. Per target, :func:`.forms.bfast_advance` forms the new state
    (MEEP's two roundings, the one-partner forms by the record's presence bits, the
    plain branch on cells in neither the dsig nor the dsigu chunk) and the increment;
    the increment is masked to ``+0`` and the state kept on the cells the curl's
    ownership mask drops; the state is stored, and the increment goes through the
    BFAST ladder (:func:`.forms.fold_ladder`) into the registers the forms returned.
    ``HAS_BFAST = 0`` builds ``kernels.pml_curl_step`` exactly (the gate's identity
    arm).
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall, which `tl.load`'s
    # `other=` delivers without dereferencing anything.
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vx, vy, vz = live, live, live
    if BCX == METALLIC:
        vx = live & (si >= 0) & (si < nx)
    else:
        si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
    if BCY == METALLIC:
        vy = live & (sj >= 0) & (sj < ny)
    else:
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    if BCZ == METALLIC:
        vz = live & (sk >= 0) & (sk < nz)
    else:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))

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

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
    # One call per target: MEEP's left-to-right association with both partners, the
    # one-partner form where the record says the other partner's derivative axis is
    # invariant, and an exact +0 with neither. The presence word is the record's
    # (stepping.curl_presence, MEEP's have_plus_deriv / have_minus_deriv).
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
    curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
    curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    # Cell 0 of a metallic axis, for every target whose Yee shift there is 0. The
    # B targets have a single zero shift each (Bx:x, By:y, Bz:z); the D targets
    # have two (Dx:y,z  Dy:x,z  Dz:x,y). Masked, then the recurrence runs on zero.
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    if BACKWARD:
        if BCY == METALLIC:
            curl0 = tl.where(at_y, 0.0, curl0)
        if BCZ == METALLIC:
            curl0 = tl.where(at_z, 0.0, curl0)
        if BCX == METALLIC:
            curl1 = tl.where(at_x, 0.0, curl1)
        if BCZ == METALLIC:
            curl1 = tl.where(at_z, 0.0, curl1)
        if BCX == METALLIC:
            curl2 = tl.where(at_x, 0.0, curl2)
        if BCY == METALLIC:
            curl2 = tl.where(at_y, 0.0, curl2)
    else:
        if BCX == METALLIC:
            curl0 = tl.where(at_x, 0.0, curl0)
        if BCY == METALLIC:
            curl1 = tl.where(at_y, 0.0, curl1)
        if BCZ == METALLIC:
            curl2 = tl.where(at_z, 0.0, curl2)

    # --- split-field recurrence (stepping._apply_pml_update) -------------------
    # dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    # sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    # Which cells take the recurrence: the component's dsigu layer, on the component's
    # own sub-lattice. The dsigu axis is z for target 0, x for 1 and y for 2 (vec.hpp
    # cycle_direction); the sub-lattice is half-integer for every B target and integer
    # for every D one (the iyee table), so each target's record line is a compile-time
    # word index, chunk_map.record_index(half_integer, axis, 0).
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

    # --- the BFAST pass (stepping._bfast_term, then stepping._apply_folded_increment)
    # MEEP's STEP_BFAST on each component right after its STEP_CURL
    # (step_db.cpp:124-142), on the SAME gathered partners, applied to the registers
    # the forms returned (forms.bfast_advance: the state's two roundings, the
    # one-partner forms by the record's presence bits, MEEP's plain branch on cells
    # in neither the dsig nor the dsigu chunk). The increment is masked to +0, and
    # the state kept, on the cells the curl's own mask drops (MEEP writes F inside
    # its owned loop only); the BFAST ladder subtracts it (forms.fold_ladder). No
    # dtdx anywhere: MEEP passes it to step_bfast and never reads it.
    if HAS_BFAST:
        ds0 = record_in(cb, lattice + 4, j)
        ds1 = record_in(cb, lattice + 8, k)
        ds2 = record_in(cb, lattice + 0, i)
        st0 = tl.load(s0 + idx, mask=live, other=0.0)
        st1 = tl.load(s1 + idx, mask=live, other=0.0)
        st2 = tl.load(s2 + idx, mask=live, other=0.0)
        bf0, q0 = bfast_advance(st0, c_y, c, b_z, b, k1_0, k2_0, py, pz, ds0 | du0)
        bf1, q1 = bfast_advance(st1, a_z, a, c_x, c, k1_1, k2_1, pz, px, ds1 | du1)
        bf2, q2 = bfast_advance(st2, b_x, b, a_y, a, k1_2, k2_2, px, py, ds2 | du2)
        if BACKWARD:
            if BCY == METALLIC:
                q0 = tl.where(at_y, 0.0, q0)
                bf0 = tl.where(at_y, st0, bf0)
            if BCZ == METALLIC:
                q0 = tl.where(at_z, 0.0, q0)
                bf0 = tl.where(at_z, st0, bf0)
            if BCX == METALLIC:
                q1 = tl.where(at_x, 0.0, q1)
                bf1 = tl.where(at_x, st1, bf1)
            if BCZ == METALLIC:
                q1 = tl.where(at_z, 0.0, q1)
                bf1 = tl.where(at_z, st1, bf1)
            if BCX == METALLIC:
                q2 = tl.where(at_x, 0.0, q2)
                bf2 = tl.where(at_x, st2, bf2)
            if BCY == METALLIC:
                q2 = tl.where(at_y, 0.0, q2)
                bf2 = tl.where(at_y, st2, bf2)
        else:
            if BCX == METALLIC:
                q0 = tl.where(at_x, 0.0, q0)
                bf0 = tl.where(at_x, st0, bf0)
            if BCY == METALLIC:
                q1 = tl.where(at_y, 0.0, q1)
                bf1 = tl.where(at_y, st1, bf1)
            if BCZ == METALLIC:
                q2 = tl.where(at_z, 0.0, q2)
                bf2 = tl.where(at_z, st2, bf2)
        tl.store(s0 + idx, bf0, mask=live)
        tl.store(s1 + idx, bf1, mask=live)
        tl.store(s2 + idx, bf2, mask=live)
        v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)
        v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)
        v2, n2 = fold_ladder(v2, n2, q2, si_x, si_y, du2)

    # fu is written on the recurrence's cells ONLY: outside them the dsig-only form
    # reads and writes the field alone and fu keeps its word (the auxiliary contract).
    tl.store(u0 + idx, n0, mask=live & du0)
    tl.store(u1 + idx, n1, mask=live & du1)
    tl.store(u2 + idx, n2, mask=live & du2)
    tl.store(f0 + idx, v0, mask=live)
    tl.store(f1 + idx, v1, mask=live)
    tl.store(f2 + idx, v2, mask=live)


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors ``coverage._grid_reasons`` so the two can be
# diffed. Clause 11 (BFAST) is INVERTED: this product REQUIRES bfast_active,
# where every shipped kernel requires it off — the same inversion pattern the
# special_kz tranche used for beta, and the reason no admitted-overlap
# ambiguity can arise while dispatch stays disabled. Everything else is KEPT,
# restated rather than imported, because the shipped reason list is built
# inside a function whose BFAST clause cannot be subtracted from outside.


def _bfast_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every BFAST-family predicate shares."""
    reasons: List[str] = []

    # 1. CuPy backend.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage REQUIRED (Phase A). A complex-storage BFAST run belongs
    #    to the complex family's queue, not to this kernel.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: complex-storage BFAST is "
                       "the complex family's queued composition, not Phase A")

    # 3. An absorber that actually absorbs (split-field family only).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "split-field path only; no-PML BFAST is refused by "
                       "name — zero demand, and no_pml.py refuses bfast too)")

    # 4. Only the two ghost rules the curl kernel writes.
    kinds = _boundary_kinds(grid, pml if (pml is not None
                                          and getattr(pml, "is_active", False))
                            else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside "
                               f"{COVERED_BOUNDARIES}")

    # 5. No mirror plane anywhere.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not "
                       "carried by this family)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian only. Cylindrical + BFAST is UNREACHABLE — the Grid itself
    #    raises (Grid._resolve_bfast, grid.py:706-742, on MEEP's own
    #    step_generic.cpp:376 missing '- F[i]') — but the clause is KEPT:
    #    this file's own doctrine forbids inferring a refusal from another
    #    module's guard (coverage.py:214-215).
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried (and "
                       "the grid itself refuses bfast there, grid.py:706-742)")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0. Grid DELIBERATELY allows bfast + Bloch (grid.py:719-724 — the
    #    two are independent MEEP constructor slots); this predicate refuses
    #    the pairing by name because a Bloch phase needs complex storage and
    #    the composition belongs to the complex family's queue.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r} "
                       f"(Grid allows bfast+Bloch; this Phase A kernel "
                       f"refuses it by name — the complex family's queue)")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. No instantaneous nonlinearity.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor "
                       "is not carried; that family is a separate tranche)")

    # 11 (INVERTED). BFAST must be ACTIVE: a k = 0 run belongs to the
    #    certified base kernels, and the array path never enters the fold
    #    (stepping.py:364/:451).
    if not getattr(grid, "bfast_active", False):
        reasons.append("grid.bfast_active is False: this product exists only "
                       "for BFAST runs; a k = 0 run belongs to the certified "
                       "plain kernels")

    # 11a. The k assignment reads ``bfast_scaled_k`` and the declared invariance of
    #     every axis (``bfast_curl_coefficients``, stepping._bfast_term). A missing or
    #     unanswerable reader is refused outright rather than read as "not
    #     invariant": inferring admission from an absent reader is the
    #     attribute-absence trap _curl_conductivity_reasons already names. A nonzero
    #     k on a declared-invariant axis is ADMITTED: the array path now takes MEEP's
    #     one-partner forms there (the absent partner is not read, step_db.cpp:62-63
    #     and step_generic.cpp:342-346), and so does this kernel.
    bfast_k = getattr(grid, "bfast_scaled_k", None)
    reader = getattr(grid, "is_invariant", None)
    if bfast_k is None or not callable(reader):
        reasons.append("grid does not expose both bfast_scaled_k and a "
                       "callable is_invariant; the k assignment cannot be "
                       "formed, and an unanswerable question is not coverage")
    else:
        for axis in range(3):
            try:
                bool(reader(axis))
                float(bfast_k[axis])
            except Exception as exc:  # noqa: BLE001 - unreadable is not covered
                reasons.append(f"the k assignment could not be read on axis "
                               f"{axis}: {exc!r}")

    # 12. Beta KEPT refused: the beta-then-bfast fold ORDER (stepping.py
    #     :384-391 before :392-396) is byte-significant, so the composition
    #     is a coordinated later change, not an accident — and special_kz's
    #     own predicates refuse bfast in return (no silent overlap).
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is "
                       f"nonzero: the beta-then-bfast fold order is "
                       f"byte-significant; bfast+beta is a coordinated "
                       f"follow-up, not this tranche")

    # 9c. Stored E — the invariant behind admitting dispersion for the curl.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return reasons


def _curl_conductivity_reasons(fields: Any) -> List[str]:
    """Conductivity refused on ALL six curl targets, BOTH directions — the
    complex tranche's stricter reading. The conductive product is
    ``conductivity.py``'s, and ITS predicates refuse bfast
    (conductivity.py:515-518) — no silent overlap either way. A missing or
    non-callable reader is refused OUTRIGHT: inferring "no conductivity" from
    the absence of ``condfac_for`` is admission by attribute absence."""
    reasons: List[str] = []
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
        return reasons
    for target in CURL_TARGETS:
        try:
            conductive = reader(target) is not None
        except Exception as exc:  # noqa: BLE001 - unreadable means not covered
            reasons.append(f"condfac_for({target!r}) raised {exc!r}")
            continue
        if conductive:
            reasons.append(
                f"a conductivity is installed on {target}: this kernel "
                f"transcribes the plain split-field recurrence only, and the "
                f"conductive family's own predicate refuses bfast "
                f"(conductivity.py:515-518) — the composition is a named "
                f"follow-up, not a silent overlap")
    return reasons


def bfast_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May the BFAST curl kernel step this (fields, pml, sub_step)?

    ``coverage.pml_curl_coverage``'s clause set with clause 11 INVERTED
    (bfast_active required), conductivity strict on all six targets in both
    directions, and the six-state allocation added. Dispersion and
    off-diagonal epsilon are ADMITTED exactly as the shipped curl predicate
    admits them (constitutive-only features; the curl differences stored
    arrays). Allocation/layout are scoped to the NAMED sub-step's 12 arrays
    (grouping choice 10 in the module docstring).
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _bfast_grid_reasons(fields, pml, grid)
    reasons.extend(_curl_conductivity_reasons(fields))

    # 9a/9b. A registered susceptibility must be one this package understands.
    reasons.extend(_susceptibility_reasons(fields))

    # 13. The named sub-step's targets, auxiliaries, sources AND IIR states.
    #     A missing state raises in the array path (stepping.py:916-921);
    #     here it is a refusal by name, before any launch.
    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"])
             + BFAST_STATE_NAMES[sub_step])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    # 14. Layout: float32, C-contiguous, grid.shape, int32 index bound — the
    #     states included (they share the field storage layout,
    #     fields.py:632-651).
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def bfast_run_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the CERTIFIED real constitutive kernel step ``update_H``/``update_E``
    on a BFAST run?

    ``coverage.constitutive_coverage``'s clause set with the BFAST clause
    INVERTED and nothing else changed: the constitutive sub-steps read nothing
    bfast-dependent (fields.py:227-230 — the pass is on step_db only, E/H
    never carry a state), so admission delegates the ARITHMETIC to the
    certified ``kernels.constitutive_step`` unchanged — no new kernel, no new
    sub-step (grouping choice 9). The E side keeps the shipped refusals
    (polarizations belong to the ADE kernel; off-diagonal rows are
    non-element-wise; stores_E).
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _bfast_grid_reasons(fields, pml, grid)
    reasons.extend(_susceptibility_reasons(fields))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is "
                "(D - sum P), not D — that configuration belongs to the ADE "
                "kernel, exactly as the shipped predicate rules")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product "
                "reads neighbours; this sub-step is element-wise) — the "
                "offdiag family owns it, and its predicate refuses bfast "
                "through the shared grid clauses")

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
# The plans
# ---------------------------------------------------------------------------

class BfastPmlCurlPlan:
    """A launchable, allocation-free REAL BFAST PML curl sub-step.

    ``launch.PmlCurlPlan`` plus the three state pointers, the six host-rounded
    scalars and the ``HAS_BFAST`` constexpr. Same two construction routes and one
    launch route: :func:`plan_bfast_pml_curl` from the engine's objects through the
    predicate, :func:`plan_bfast_pml_curl_from_arrays` from bare device arrays for the
    gate (including the deliberately wrong ones), both launched through :meth:`run`.
    ``chunk_bounds`` is the chunk record, required (``launch.bind_chunk_bounds``).

    The state pointers wrap the engine's OWN ``f_bfast_*`` arrays — the in-place,
    pointer-identical mutation the driver's flux backup/restore depends on; the gate
    holds that identity.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "ks", "has_bfast",
                 "backward", "bc", "block", "num_warps", "chunk_bounds", "_targets",
                 "_aux", "_sources", "_states", "_coefficients", "_cb", "_grid",
                 "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc,
                 ks: Sequence[float], block: int,
                 targets, auxiliaries, sources, states, coefficients,
                 kernel=None, num_warps: Optional[int] = None,
                 has_bfast: int = 1, *, chunk_bounds: Any) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "BfastPmlCurlPlan")
        self.dtdx = float(dtdx)
        # Already f32-rounded by the host (bfast_curl_coefficients); float()
        # keeps the bits, and Triton types a Python float argument as fp32.
        self.ks = tuple(float(value) for value in ks)
        if len(self.ks) != 6:
            raise ValueError(f"ks must be the six (k1,k2) scalars, got {ks!r}")
        self.has_bfast = int(has_bfast)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._states = tuple(CupyPointer(a) for a in states)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # The override exists for exactly one caller: the gate's mutation legs.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. ``guard`` is the gate's, not a caller's."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else bfast_pml_curl_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._states,
            *self._coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.ks,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            HAS_BFAST=self.has_bfast,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"BfastPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, ks={self.ks!r}, has_bfast={self.has_bfast}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_bfast_pml_curl(fields: Any, pml: Any, sub_step: str,
                        block: Optional[int] = None,
                        num_warps: Optional[int] = None,
                        chunk_layout: Any = None
                        ) -> Optional[BfastPmlCurlPlan]:
    """Build a BFAST curl plan from the engine's objects, or None.

    None is the only refusal (Y-style). The six scalars come from
    :func:`bfast_curl_coefficients` with the grid's OWN declared-dimensionality
    invariance flags (never a shape test) and the sub-step's side — identical
    arithmetic to the array path's. The state arrays bound are ``fields.f_bfast_*``
    themselves, pointer-identical (the driver-sync requirement). ``chunk_layout`` is
    the run's chunk division, ``None`` for the one-rank default.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not bfast_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    ks = bfast_curl_coefficients(grid.bfast_scaled_k, invariant,
                                 magnetic=(sub_step == "step_B"))
    return BfastPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        ks,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(fields, name) for name in BFAST_STATE_NAMES[sub_step]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_bfast_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                    flat: Dict[str, Any], codes, dtdx: float,
                                    ks: Sequence[float],
                                    block: Optional[int] = None,
                                    kernel: Any = None,
                                    num_warps: Optional[int] = None,
                                    has_bfast: int = 1, *,
                                    chunk_bounds: Any) -> BfastPmlCurlPlan:
    """Build a BFAST curl plan from bare device arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones; ``kernel=`` carries the
    mutation override, ``ks`` the (possibly deliberately wrong) six scalars, and
    ``has_bfast=0`` the identity leg's root-kernel arm. ``arrays`` must carry the
    sub-step's three ``f_bfast_*`` states beside its targets, auxiliaries and sources.
    ``chunk_bounds`` has NO default: a harness plan built without a record is refused
    by name (``launch.bind_chunk_bounds``).
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return BfastPmlCurlPlan(
        sub_step, shape, dtdx, codes, ks,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [arrays[name] for name in BFAST_STATE_NAMES[sub_step]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, has_bfast=has_bfast,
        chunk_bounds=chunk_bounds,
    )


def plan_bfast_run_constitutive(fields: Any, pml: Any, side: str,
                                block: Optional[int] = None,
                                num_warps: Optional[int] = None,
                                chunk_layout: Any = None
                                ) -> Optional[ConstitutivePlan]:
    """A root real constitutive plan for a BFAST run, or None.

    The arithmetic and the plan class are ``launch.ConstitutivePlan``'s, untouched;
    only the ADMISSION is this module's (:func:`bfast_run_constitutive_coverage`, the
    root predicate with the BFAST clause inverted). The binding restates
    ``launch.plan_constitutive``'s: the run's chunk record (``chunk_layout``, ``None``
    for the one-rank default), and on the H side MEEP's start of H from ``fields``.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not bfast_run_constitutive_coverage(fields, pml, side).covered:
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
# WIRING — none, deliberately
# ---------------------------------------------------------------------------
#
# This module is NOT imported by the package ``__init__``, is not consulted by
# ``launch.plan_step``, and does not touch production dispatch, which keeps
# returning None on every branch. That is SAFE to defer, not merely
# convenient — but the safety argument is NARROWER than "no shipped predicate
# admits a BFAST run", which is FALSE as measured:
#
# * every predicate that reaches ``coverage._grid_reasons`` — the shipped curl,
#   the shipped constitutive sides, dispersive/offdiag/nonlinear/complex/no_pml
#   and the conductive and special_kz families — refuses a BFAST run by name.
#   Those are the only predicates whose sub-steps this module also plans, so no
#   admitted-overlap ambiguity exists on step_B/step_D/update_H/update_E;
# * TWO shipped predicates do NOT consult ``_grid_reasons`` and DO admit a
#   BFAST run: ``coverage.ade_update_p_coverage`` (coverage.py:501 — no grid
#   clause, no bfast clause) and ``fused_ade_state.fused_ade_state_coverage``.
#   Measured on a real-f32, PML-active, fully allocated dispersive grid with
#   bfast_scaled_k = (0.31, 0.17, 0.23): both return no non-backend reason.
#   Harmless in substance — BFAST touches only the curl sub-step
#   (fields.py:227-230: E/H carry no f_bfast state; the pass is on step_db),
#   and this module builds no update_P plan — but the universal does not hold,
#   and this package's doctrine forbids inferring a refusal from another
#   module's guard. A laptop test pins the measurement so the exception list
#   cannot silently grow.
#
# An admitted-overlap ambiguity on the sub-steps this module plans therefore
# cannot arise until a later coordinated change adds launch-side forwarders,
# and THAT change re-runs the byte gate. Tests and the gate import
# ``meep_gpu.triton_kernels.bfast_curl`` directly (the package ``__init__``
# eagerly imports only ``coverage``, so the direct import is safe on a
# Triton-less host).
