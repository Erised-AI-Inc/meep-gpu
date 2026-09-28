"""The DISPERSIVE electric constitutive sub-step — ``update_E`` with poles registered.

One kernel, one coverage predicate, one plan, in one file, because that is the whole
of what this sub-step needs and none of it belongs in the shared modules yet.

WHAT IT REPLACES. ``stepping.update_E`` (stepping.py:926) with an active PML and at
least one registered susceptibility. Per component ``c`` the array path runs::

    source       = fields.displacement_minus_polarization(c)      # :981
    constitutive = source * fields.inverse_epsilon_for(c)         # :982
    kps, kms     = _constitutive_coefficients(pml, axis, half_integer=True)   # :986
    _apply_constitutive_pml(E_c, constitutive, kps, kms, f_w_c)   # :987

and ``displacement_minus_polarization`` (fields.py:1079-1105) is::

    contributors = [s for s in fields.polarizations if s.drives(c)]
    if not contributors: return D_c
    scratch[...] = D_c
    for state in contributors: state.subtract_into(c, scratch)    # dispersion.py:693

so the contract this file transcribes, per component, is exactly::

    s    = ((D_c - P_0[c]) - P_1[c]) - ...        left to right, polarizations order
    src  = s * inv_eps_c                          D-minus-P on the LEFT
    prev = f_w_c ; f_w_c = src
    E_c  = (E_c + kps_h[a]*src) - kms_h[a]*prev   a = the component's OWN axis

THREE THINGS DECIDE BIT-IDENTITY, and each is pinned by a MEASURED control rather
than by argument (recon, 30 real Grid/Fields/PML cases on NumPy; the counts below
are that run's):

1. **THE SUM OF P CANNOT BE PRE-ACCUMULATED.** ``(D - P0) - P1`` and
   ``D - (P0 + P1)`` are different float32 numbers — caught 20/20 on the two-pole
   cases, 0/10 at one pole, which is the right split for a multi-pole-only defect.
   The natural design (accumulate one "sum of P" volume, hand the kernel one extra
   input) is therefore WRONG, and it would also cost the extra full-volume pass this
   kernel exists to delete. Anyone reading only the docstring sentence "E = (D - sum
   P) * inv_eps" will write it; this paragraph is here so they do not.
2. **THE POLE ORDER IS LOAD-BEARING**, and the pole SET IS PER COMPONENT.
   ``((D-P1)-P0)`` was caught 20/20 at two poles. ``PolarizationState._driven``
   (dispersion.py:640-642) drops any component whose sigma is identically zero, so
   Ex, Ey and Ez can each see a different subset in a different order; a plan that
   caches one pole list for all three components is wrong on any anisotropic-sigma
   material, and wrong SILENTLY — the field stays smooth and plausible.
3. **THE TWO ACCUMULATIONS STAY SEPARATE.** ``(E + kps*src) - kms*prev`` regrouped
   to ``E + (kps*src - kms*prev)`` was caught 30/30, at BOTH Courant numbers — the
   non-representable multiplicands here are the PML coefficients themselves, not the
   Courant scaling, so there is no exact-in-binary case to exempt. ``ENABLE_FP_FUSION
   = False`` is what stops the multiplies contracting into an FMA on top of that.

A NULL, measured so it is not mistaken for a fourth: ``inv_eps * s`` versus
``s * inv_eps`` is BITWISE IDENTICAL (caught 0/30). float32 multiplication is
commutative to the bit. The array path's operand order is kept for transcription
discipline, and a gate leg that reports it "caught" is comparing something other
than bytes.

WHAT IT DOES NOT DO. ``update_E`` CONSUMES P and does not advance it — measured, P
untouched on 30/30 recon cases. ``update_P`` runs afterwards and reads ``f_w_E``
through ``Fields.drive_field`` (fields.py:1140-1162), which under PML returns
``f_w_<c>`` and NOT the stored E. So this kernel's ``f_w`` store is precisely the
input the already-gated :func:`kernels.ade_update_p` consumes, the two compose
through global memory, and nothing about ``AdeUpdatePPlan`` changes.

Sigma never enters here. It belongs to the ADE recurrence; do not add a clause for
it to the predicate below.

SUBNORMALS ARE THIS SUB-STEP'S OWN HAZARD, AND THE FLUSHING SIDE IS THE ARRAY PATH.
``D - sum P`` is a CANCELLATION between two nearly equal float32 volumes and is the
only one of its kind in the step; ``(D - sum P) * inv_eps`` then underflows wherever
the field is small. On the benchmark's own configuration (``2d_dispersive``, res 20,
NumPy) the first subnormal in ``f_w_Ez`` — the array this kernel writes — appears at
step 42, against the plan's measured Triton/CuPy divergence at step 40; the plain-PML
control lands at 66 against a measured 67 (plan §16). The onset step IS the
divergence step.

WHICH SIDE LEAVES IEEE WAS AN OPEN QUESTION AND IS NOW MEASURED, per operation, with
NumPy as the neutral reference (``results/.../logs/ftz_side.log``):

    subnormal results out of 256   numpy   cupy   triton
      both operands subnormal        256      0      256   triton == numpy, 0 differing
      product underflows              92      0       92   triton == numpy, 0 differing
      control, all normal               0      0        0   all three agree

**CuPy flushes float32 subnormals to zero; Triton and NumPy keep them.** So on this
sub-step the kernel is the IEEE-correct side and the array path is not, and the byte
gate's cancellation-class failures are the ARRAY PATH's departure, not a kernel
defect. That settles which of plan §16's three ways out is the real one — (a), make
the array path stop flushing — and it does so without a claim about the kernels.

The gate's verdict is therefore reported in TWO parts and never merged: 300/300
bit-identical on the ``uniform`` class (normal numbers, guarded) against 0/300
unguarded, and 32/300 on the deliberately-seeded ``cancellation`` class. The
unqualified words "bit-identical about a run" must not be used here.

NOTHING HERE IS WIRED. ``plan_step`` does not know about this module, ``__init__``
does not export it, and the engine's step path is untouched — integration waits for
the shared files to be free. See the module's ``INTEGRATION`` note at the bottom.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage

#: The most poles one component may carry. Ag — the corpus's only real multi-pole
#: material, used by the three ``stochastic_emitter`` scripts — has 6 (1 Drude +
#: 5 Lorentz); ``fused_quartz`` has 3, ``SiO2`` 1, and the benchmark case 1. The
#: predicate refuses more BY NAME rather than silently truncating, because a
#: dropped pole is a smooth, plausible, wrong field.
MAX_POLES = 8

#: The components this sub-step writes, in the order ``stepping.E_CONSTITUTIVE_TERMS``
#: (stepping.py:227) iterates them, with the axis whose HALF-INTEGER coefficient pair
#: each one reads. The axis is the component's OWN — MEEP's ``dsigw``, the absorption
#: a component accumulates along the direction it points in — and is not the
#: ``dsig``/``dsigu`` cycle the curl recurrence uses.
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The Yee sub-lattice this side reads: half-integer, i.e. ``kps_x_h``/``kms_x_h``
#: (``stepping.py:986`` via ``_constitutive_coefficients(..., half_integer=True)``).
#: Swapped for the integer tables it is a half-cell error in the absorber profile —
#: converged, smooth and wrong — which is why the gate carries a mutation for it.
HALF_INTEGER = True


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# TRITON IS IMPORTED CONDITIONALLY AND THE KERNEL IS DEFINED CONDITIONALLY.
# `kernels.py` can import Triton at module scope because nothing imports it unless
# a plan is being launched; this module is different — the coverage predicate below
# is imported by tests that run on the laptop that is the merge bar, where Triton
# is not installable at all (there is no Metal backend).
#
# The kernel is NOT hidden behind a lazy builder function, and that was measured
# rather than chosen: `@triton.jit` resolves a body's names — including the
# `tl.constexpr` ANNOTATIONS — through the defining module's `__globals__`, never
# through an enclosing function's locals. A kernel defined inside a builder
# therefore compiles to `NameError('tl is not defined')` at first launch, which is
# what the first run of the device gate reported on 320 of 320 cases.

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:

    @triton.jit
    def constitutive_step_dispersive(
        f0, f1, f2,                     # targets:      Ex, Ey, Ez          (in/out)
        w0, w1, w2,                     # auxiliaries:  f_w_Ex, f_w_Ey, f_w_Ez (in/out)
        g0, g1, g2,                     # sources:      Dx, Dy, Dz          (in)
        e0, e1, e2,                     # inverse epsilon, per component    (in)
        a0, a1, a2, a3, a4, a5, a6, a7,   # component 0's poles, IN ORDER
        b0, b1, b2, b3, b4, b5, b6, b7,   # component 1's poles, IN ORDER
        c0, c1, c2, c3, c4, c5, c6, c7,   # component 2's poles, IN ORDER
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's OWN axis
        nx, ny, nz, n_elem,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``stepping.update_E`` under PML with poles registered, all three components.

        The body is :func:`kernels.constitutive_step`'s E arm with ONE substitution:
        the source is no longer ``D`` but ``D`` with each driving pole's ``P``
        subtracted from it, sequentially, left to right, in ``fields.polarizations``
        order. Everything else — the ``prev``-before-store ordering, the two separate
        accumulations, the own-axis coefficient index — is that kernel's, unchanged,
        because the array path's is unchanged.

        ``NP0``/``NP1``/``NP2`` are the LIVE pole count per component and are
        ``tl.constexpr``, so the subtraction chain is unrolled at compile time and a
        component with no pole compiles to exactly the non-dispersive body. Unused
        pole slots are bound to the component's own ``D`` pointer by the plan and are
        never read; a device array of pointers would move the count out of the
        compiled specialization, which is where the bit-exact unrolled order lives.

        THE POLES ARE NOT PRE-SUMMED, and that is the whole design. See the module
        docstring, control 1: ``D - (P0 + P1)`` is a different float32 number and was
        caught 20/20 at two poles.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -------------------------------------------------------
        prev0 = tl.load(w0 + idx, mask=live, other=0.0)   # BEFORE the store.
        s0 = tl.load(g0 + idx, mask=live, other=0.0)
        if NP0 > 0:
            s0 = s0 - tl.load(a0 + idx, mask=live, other=0.0)
        if NP0 > 1:
            s0 = s0 - tl.load(a1 + idx, mask=live, other=0.0)
        if NP0 > 2:
            s0 = s0 - tl.load(a2 + idx, mask=live, other=0.0)
        if NP0 > 3:
            s0 = s0 - tl.load(a3 + idx, mask=live, other=0.0)
        if NP0 > 4:
            s0 = s0 - tl.load(a4 + idx, mask=live, other=0.0)
        if NP0 > 5:
            s0 = s0 - tl.load(a5 + idx, mask=live, other=0.0)
        if NP0 > 6:
            s0 = s0 - tl.load(a6 + idx, mask=live, other=0.0)
        if NP0 > 7:
            s0 = s0 - tl.load(a7 + idx, mask=live, other=0.0)
        src0 = s0 * tl.load(e0 + idx, mask=live, other=0.0)
        tl.store(w0 + idx, src0, mask=live)
        v0 = tl.load(f0 + idx, mask=live, other=0.0)
        v0 = v0 + kp_0 * src0
        v0 = v0 - km_0 * prev0
        tl.store(f0 + idx, v0, mask=live)

        # --- component 1 -------------------------------------------------------
        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        s1 = tl.load(g1 + idx, mask=live, other=0.0)
        if NP1 > 0:
            s1 = s1 - tl.load(b0 + idx, mask=live, other=0.0)
        if NP1 > 1:
            s1 = s1 - tl.load(b1 + idx, mask=live, other=0.0)
        if NP1 > 2:
            s1 = s1 - tl.load(b2 + idx, mask=live, other=0.0)
        if NP1 > 3:
            s1 = s1 - tl.load(b3 + idx, mask=live, other=0.0)
        if NP1 > 4:
            s1 = s1 - tl.load(b4 + idx, mask=live, other=0.0)
        if NP1 > 5:
            s1 = s1 - tl.load(b5 + idx, mask=live, other=0.0)
        if NP1 > 6:
            s1 = s1 - tl.load(b6 + idx, mask=live, other=0.0)
        if NP1 > 7:
            s1 = s1 - tl.load(b7 + idx, mask=live, other=0.0)
        src1 = s1 * tl.load(e1 + idx, mask=live, other=0.0)
        tl.store(w1 + idx, src1, mask=live)
        v1 = tl.load(f1 + idx, mask=live, other=0.0)
        v1 = v1 + kp_1 * src1
        v1 = v1 - km_1 * prev1
        tl.store(f1 + idx, v1, mask=live)

        # --- component 2 -------------------------------------------------------
        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        s2 = tl.load(g2 + idx, mask=live, other=0.0)
        if NP2 > 0:
            s2 = s2 - tl.load(c0 + idx, mask=live, other=0.0)
        if NP2 > 1:
            s2 = s2 - tl.load(c1 + idx, mask=live, other=0.0)
        if NP2 > 2:
            s2 = s2 - tl.load(c2 + idx, mask=live, other=0.0)
        if NP2 > 3:
            s2 = s2 - tl.load(c3 + idx, mask=live, other=0.0)
        if NP2 > 4:
            s2 = s2 - tl.load(c4 + idx, mask=live, other=0.0)
        if NP2 > 5:
            s2 = s2 - tl.load(c5 + idx, mask=live, other=0.0)
        if NP2 > 6:
            s2 = s2 - tl.load(c6 + idx, mask=live, other=0.0)
        if NP2 > 7:
            s2 = s2 - tl.load(c7 + idx, mask=live, other=0.0)
        src2 = s2 * tl.load(e2 + idx, mask=live, other=0.0)
        tl.store(w2 + idx, src2, mask=live)
        v2 = tl.load(f2 + idx, mask=live, other=0.0)
        v2 = v2 + kp_2 * src2
        v2 = v2 - km_2 * prev2
        tl.store(f2 + idx, v2, mask=live)

else:  # pragma: no cover - the laptop path
    constitutive_step_dispersive = None  # type: ignore[assignment]


def constitutive_step_dispersive_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError.

    The accessor exists so a caller that needs the kernel gets the same explanation
    :func:`launch.require_triton` gives, rather than a ``None`` that fails later as a
    ``TypeError`` far from its cause.
    """
    if constitutive_step_dispersive is None:
        raise ImportError(
            "the dispersive update_E kernel needs the optional `triton` package "
            "(pip install triton). The engine runs without it; only this fast path "
            f"is unavailable. Original error: {_TRITON_IMPORT_ERROR}")
    return constitutive_step_dispersive


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------
#
# It ENUMERATES what it covers and refuses everything else. Nothing here infers
# coverage from the absence of a known blocker: the two worst defects in this
# project were silent wrong answers, not crashes.
#
# It reuses `coverage`'s own clause builders rather than restating them, so a rule
# that tightens for the other kernels tightens here in the same commit. Those
# helpers are module-private by name; `test_triton_dispersive_update_e` pins that
# they exist, so a rename in the shared file fails at the merge bar instead of
# silently dropping a clause.

#: The shared clause builders this predicate is composed from. Named as data so the
#: test can assert every one of them is still there.
SHARED_CLAUSES: Tuple[str, ...] = (
    "_grid_reasons", "_susceptibility_reasons", "_layout_reasons",
    "_inverse_epsilon_reasons", "_coefficient_reasons", "_volume_reasons")


def dispersive_constitutive_coverage(fields: Any, pml: Any) -> "_coverage.Coverage":
    """May the Triton dispersive kernel step ``update_E`` for this (fields, pml) pair?

    POSITIVE CLAUSES ONLY. Every requirement is named and checked; a failing clause
    appends its reason and the scan continues, so a refusal reports everything that
    disqualified the run rather than the first thing.

    Inherited from :func:`coverage._grid_reasons` and
    :func:`coverage._susceptibility_reasons` — the CuPy backend, real float32
    storage, an active absorber, only the ``periodic``/``metallic`` ghost rules, no
    mirror fold, Cartesian only, ``k_point`` exactly zero, no conductivity, no
    chi2/chi3, no BFAST, no ``special_kz``, electric-only susceptibilities whose kind
    is in ``{lorentzian, drude}``.

    Added here, and each one is a silent wrong answer if it is missing:

    a. **an ACTIVE PML is required.** Without one ``update_E`` is a different
       sub-step — ``field[...] = constitutive`` (stepping.py:993), no ``f_w``, no
       ``kps``/``kms`` — and running this kernel's PML form with unit coefficients
       would be wrong in the auxiliary even where it looked right in ``E``.
       (``_grid_reasons`` already requires it; the clause is repeated by name because
       the REASON is specific to this sub-step.)
    b. **stored E.** Same invariant §13.1 turns on: ``update_E`` writes an array, it
       does not serve ``D*inv_eps`` on demand.
    c. **no off-diagonal chi1inv.** The row product reads the OTHER components'
       volumes at neighbouring cells and the sub-step stops being element-wise
       (stepping.py:972-979).
    d. every state reports ``driven()``, and every driven component is electric.
    e. **at most ``MAX_POLES`` poles per component**, refused by name rather than
       truncated.
    f. every ``state.P[c]`` and ``state.P_prev[c]`` allocated, float32, C-contiguous
       and exactly ``grid.shape``. ``P_prev`` is not read by THIS kernel, and it is
       checked anyway: ``update_P`` rotates the three buffers as a set
       (dispersion.py:687-691), so a malformed ``P_prev`` becomes this kernel's
       ``P`` one step later.
    g. **at least one pole somewhere.** With none, the configuration is
       :func:`coverage.constitutive_coverage`'s and that kernel covers it; two
       predicates admitting one configuration means ``plan_step`` picks by ordering,
       which is how a wrong answer gets chosen at random. The two are DISJOINT by
       construction and a test pins it.

    SIGMA IS NOT CHECKED HERE and must not be: it enters ``update_P``, never this
    sub-step. :func:`coverage.ade_update_p_coverage` is where it belongs.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = list(_coverage._grid_reasons(fields, pml, grid))
    reasons.extend(_coverage._susceptibility_reasons(fields))

    # (a) The split-field PML form is the only one this kernel writes.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append(
            "no active PML: without one update_E writes field[...] = constitutive "
            "(stepping.py:993) with no f_w and no kps/kms — a different sub-step")

    # (b) Stored E.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # (c) Element-wise only.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed (the row product reads "
            "neighbours; this sub-step is element-wise)")

    states = tuple(getattr(fields, "polarizations", ()) or ())

    # (g) Disjointness from constitutive_coverage(side='E').
    if not states:
        reasons.append(
            "no susceptibility is registered: that configuration is "
            "constitutive_coverage(side='E')'s and this predicate must not "
            "overlap it")

    # (d) Readable, electric-only drive sets, and (e) the per-component pole count.
    per_component: Dict[str, int] = {name: 0 for name in _coverage.ELECTRIC_COMPONENTS}
    for index, state in enumerate(states):
        driven = _coverage._call(state, "driven", default=None)
        if driven is None:
            reasons.append(
                f"polarization {index} ({type(state).__name__}) does not report "
                f"driven(); an unreadable susceptibility is not a covered one")
            continue
        for name in tuple(driven):
            if name in per_component:
                per_component[name] += 1
    for name, count in per_component.items():
        if count > MAX_POLES:
            reasons.append(
                f"{name} is driven by {count} poles, more than the kernel's "
                f"MAX_POLES={MAX_POLES} compiled slots")

    shape = tuple(getattr(grid, "shape", ()))

    # (f) The pole volumes themselves.
    for index, state in enumerate(states):
        driven = _coverage._call(state, "driven", default=None)
        if driven is None:
            continue  # already reported
        for name in tuple(driven):
            for slot in ("P", "P_prev"):
                array = (getattr(state, slot, {}) or {}).get(name)
                if array is None:
                    reasons.append(f"polarization {index} {slot}[{name}] is not allocated")
                    continue
                if len(shape) != 3:
                    continue  # the shape clause below reports it once
                reasons.extend(_coverage._volume_reasons(
                    f"polarization {index} {slot}[{name}]", array, shape))

    # The volumes this sub-step reads and writes, and their layout.
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

    return _coverage.Coverage(not reasons, tuple(reasons))


def poles_per_component(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    """The driving states of each E component, IN ``fields.polarizations`` ORDER.

    The single place the order is derived, so the predicate, the plan and the
    per-launch assertion cannot disagree about it — the same discipline
    :func:`coverage.sigma_is_volume` exists for. Getting it out of step is a
    different float32 number in every cell (recon: reversing it was caught 20/20)
    and no error anywhere.

    ``Fields.displacement_minus_polarization`` (fields.py:1097) builds exactly this
    list: ``[state for state in self.polarizations if state.drives(component)]``.
    """
    states = tuple(getattr(fields, "polarizations", ()) or ())
    return {name: tuple(state for state in states
                        if _coverage._call(state, "drives", name, default=False))
            for name in _coverage.ELECTRIC_COMPONENTS}


# ---------------------------------------------------------------------------
# Pole bindings — where the per-launch resolution lives
# ---------------------------------------------------------------------------

class LivePoleBinding:
    """The engine route's pole source: ``state.P[c]``, RESOLVED PER LAUNCH.

    THIS CLASS EXISTS TO SAY WHY THAT IS NOT A CHOICE.
    ``PolarizationState.update`` rotates ``P`` / ``P_prev`` / ``_scratch`` for every
    component on every step (dispersion.py:687-691): this step's result lands in
    ``_scratch``, ``_scratch`` takes over the array that held ``P_prev``, and
    ``P``/``P_prev`` shift down. A plan that cached ``P`` device views the way
    :class:`launch.PmlCurlPlan` caches field views would be stale after the FIRST
    component of the FIRST step — and stale in a way that still computes, giving a
    smooth wrong field. :class:`launch.AdeUpdatePPlan` exists to document the same
    defect from the other side.

    So the ORDER is cached (it is fixed for the run — the driver refuses a
    susceptibility appended mid-run, driver.py) and the POINTERS are re-derived every
    call, with the order re-checked each time. The check costs three list
    comprehensions against a tuple of at most eight objects and the failure it
    catches is a wrong answer rather than a crash.
    """

    __slots__ = ("fields", "order", "counts")

    def __init__(self, fields: Any, order: Dict[str, Sequence[Any]]) -> None:
        self.fields = fields
        self.order = {name: tuple(states) for name, states in order.items()}
        self.counts = tuple(len(self.order[term[0]]) for term in E_TERMS)

    def arrays(self) -> Tuple[Tuple[Any, ...], ...]:
        """The live ``P`` volume of each driving state, per component, in order."""
        live = poles_per_component(self.fields)
        out: List[Tuple[Any, ...]] = []
        for component, _, _ in E_TERMS:
            expected = self.order[component]
            found = live[component]
            if len(found) != len(expected) or any(
                    a is not b for a, b in zip(expected, found)):
                raise RuntimeError(
                    f"the polarization set driving {component} changed since the plan "
                    f"was built ({len(expected)} states -> {len(found)}); the pole "
                    f"ORDER is bit-load-bearing and this plan is no longer valid")
            out.append(tuple(state.P[component] for state in expected))
        return tuple(out)

    def __repr__(self) -> str:
        return f"LivePoleBinding(counts={self.counts})"


class StaticPoleBinding:
    """The gate's and benchmark's pole source: bare device arrays, fixed for the run.

    No rotation happens in a synthetic leg — there is no ``PolarizationState`` — so
    there is nothing to re-resolve, and pretending otherwise would test the harness
    rather than the kernel.
    """

    __slots__ = ("_arrays", "counts")

    def __init__(self, arrays: Sequence[Sequence[Any]]) -> None:
        self._arrays = tuple(tuple(group) for group in arrays)
        if len(self._arrays) != 3:
            raise ValueError("a pole binding carries one group per E component")
        self.counts = tuple(len(group) for group in self._arrays)

    def arrays(self) -> Tuple[Tuple[Any, ...], ...]:
        return self._arrays

    def __repr__(self) -> str:
        return f"StaticPoleBinding(counts={self.counts})"


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class DispersiveConstitutivePlan:
    """A launchable, allocation-free dispersive ``update_E``.

    The Yee sub-lattice is chosen HERE and nowhere else: ``kps_a_h``/``kms_a_h``,
    the half-integer tables (``stepping.py:986``). The kernel takes six coefficient
    pointers and never asks which lattice they came from, so a swap on this line is a
    silent half-cell error in the absorber profile — converged, smooth and wrong. The
    gate carries a mutation for exactly it.

    Everything resolvable ahead of the loop is resolved in ``__init__``: the flat
    coefficient views, the field pointer adapters, the pole ORDER and the three
    ``NP`` constexprs. What is deliberately NOT resolved ahead of the loop is the
    pole POINTERS — see :class:`LivePoleBinding`.
    """

    __slots__ = ("shape", "n_elem", "block", "counts", "_targets", "_aux",
                 "_sources", "_inv_eps", "_coefficients", "_poles", "_grid",
                 "_kernel", "_pointer")

    def __init__(self, shape, block: int, targets, auxiliaries, sources,
                 inverse_epsilon, coefficients, poles, kernel: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self._pointer = CupyPointer
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._poles = poles
        self.counts = tuple(int(n) for n in poles.counts)
        for component, count in zip(E_TERMS, self.counts):
            if count > MAX_POLES:
                raise ValueError(
                    f"{component[0]} is driven by {count} poles; the kernel compiles "
                    f"{MAX_POLES} slots and the predicate refuses more")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as the other plans.

        ``guard=None`` takes the shipped constant; the gate passes True/False
        explicitly so the contraction guard's effect is MEASURED rather than assumed.
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else constitutive_step_dispersive_kernel())
        nx, ny, nz = self.shape
        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not the "
                    f"{self.counts[index]} the plan compiled for")
            # Unused slots take this component's own D pointer. `tl.static_range`
            # is not what stops them being read — the constexpr `if NP > n` chain is
            # — but a pointer argument still has to type, and a null makes the
            # launcher's failure a TypeError far from its cause.
            slots.extend(self._pointer(array) for array in group)
            slots.extend([self._sources[index]] * (MAX_POLES - len(group)))
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._inv_eps,
            *slots, *self._coefficients,
            nx, ny, nz, self.n_elem,
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"DispersiveConstitutivePlan(shape={self.shape}, "
                f"poles={self.counts}, block={self.block})")


def plan_dispersive_constitutive(fields: Any, pml: Any,
                                 block: Optional[int] = None
                                 ) -> Optional[DispersiveConstitutivePlan]:
    """Build the dispersive ``update_E`` plan from the engine's own objects, or None.

    None means REFUSED, and the reasons are available from
    :func:`dispersive_constitutive_coverage`. The Triton import stays BELOW the
    predicate, as it does in every builder in this package: a NumPy host must be able
    to plan (to ``None``) without the optional dependency being importable at all.
    """
    if not dispersive_constitutive_coverage(fields, pml).covered:
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    order = poles_per_component(fields)
    targets = tuple(term[0] for term in E_TERMS)
    return DispersiveConstitutivePlan(
        fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        [getattr(fields, "f_w_" + name) for name in targets],
        [getattr(fields, term[1]) for term in E_TERMS],
        [fields.inverse_epsilon_for(name) for name in targets],
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        LivePoleBinding(fields, order),
    )


def plan_dispersive_constitutive_from_arrays(arrays: Dict[str, Any],
                                             flat: Dict[str, Any],
                                             poles: Dict[str, Sequence[Any]],
                                             block: Optional[int] = None,
                                             kernel: Any = None
                                             ) -> DispersiveConstitutivePlan:
    """Build it from bare device arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name (``Ex``, ``f_w_Ex``, ``Dx``,
    ``inv_eps_Ex``), ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE
    CALLER already selected — deliberately, so the gate can hand over the integer
    tables and watch the swap be caught — and ``poles`` maps each component to its
    ordered list of P volumes. No coverage predicate runs here: the caller is a
    harness that has constructed the configuration on purpose, and refusing it would
    defeat the point of a mutation leg.

    ``kernel=`` IS LOAD-BEARING and is not decoration: the mutation leg routes a
    mutated kernel through it, and a builder that drops it launches the SHIPPED
    kernel and reports a pass for a defect it never introduced (plan §13.4, measured
    at 4/4 real defects reported as 120/120 identical).
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return DispersiveConstitutivePlan(
        shape, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets],
        [arrays["f_w_" + name] for name in targets],
        [arrays[term[1]] for term in E_TERMS],
        [arrays["inv_eps_" + name] for name in targets],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        StaticPoleBinding([poles[name] for name in targets]),
        kernel=kernel,
    )


# ---------------------------------------------------------------------------
# INTEGRATION — what the later round has to do, and nothing more
# ---------------------------------------------------------------------------
#
# Four additive edits, once the shared files are free (they are owned by the
# concurrent cross-sub-step fusion work and are NOT touched from here):
#
#   coverage.py  export `dispersive_constitutive_coverage` (a re-export of this
#                module's, not a second copy), and NARROW the clause at :339-343 so
#                `constitutive_coverage(side="E")` refuses a polarization "HERE, and
#                names this predicate" rather than refusing it flatly. The two must
#                be disjoint and the file must SAY so, which is the per-sub-step rule
#                §13.1 established.
#   launch.py    `plan_step` gains ONE branch: where `constitutive_coverage(...,"E")`
#                refuses and `dispersive_constitutive_coverage` admits, install
#                `plan_dispersive_constitutive(fields, pml, block)` in the
#                `update_E` slot. `STEP_ORDER` is unchanged, and `replaces` then
#                reports all five names on a dispersive run.
#   __init__.py  export the kernel builder, the predicate and the two plan builders.
#   fingerprints.json  add this module to `host_sha256` and the kernel to `kernels`,
#                re-cut against the bytes the gate certified.
#
# Nothing in `stepping.py` changes and no dispatch is wired: the engine's fast-path
# hook still returns None on every branch. The §14.1 whole-step rows are the
# regression test that the wiring changed nothing on the non-dispersive cases.
