"""ARM S — ``update_E`` with STORED E and no absorber: a pointwise store.

Closes residual group (F) and the ``update_E`` third of residual group (H) — 3 of
the 9 slots the no-absorber trio is about. The closure round measured the
configuration DIVERGENT against every built body at 1536 of 7680 words
(``F_stored_e_arm_update_E``,
``results/residual_closure_2026-08-15/device/bodies/bodies.json``); this is the
body that was missing.

``no_pml_constitutive`` SPECIFIED THIS ARM AND DECLINED TO BUILD IT. That decision
rested on a demand figure that has since gone stale, and the record there is
corrected in the same change that builds this file — see
``no_pml_constitutive.STORED_E_ARM``. The transcription below is that record's
``compose_from`` carried out.

WHAT THE ARRAY PATH DOES
------------------------
``stepping.update_E`` (stepping.py:926-993) with an INERT layer and ``stores_E``::

    pml_active = _pml_is_active(pml)                                   # :953
    if not pml_active and not fields.stores_E:
        return                                                          # :954-955
    _require_field_storage(fields, E_STORAGE_ARRAYS)                     # :958-959
    for component, _, axis_name in E_CONSTITUTIVE_TERMS:                 # :969
        source       = fields.displacement_minus_polarization(component) # :981
        constitutive = source * fields.inverse_epsilon_for(component)    # :982-984
        getattr(fields, component)[...] = constitutive                   # :993

and ``Fields.displacement_minus_polarization`` (fields.py:1079-1105) is::

    contributors = [s for s in self.polarizations if s.drives(component)]
    if not contributors: return D_c            # the D array ITSELF, aliased
    scratch[...] = D_c
    for state in contributors: state.subtract_into(component, scratch)   # dispersion.py:693-695

so the contract this file transcribes, per component, is exactly::

    s   = ((D_c - P_0[c]) - P_1[c]) - ...     LEFT TO RIGHT, polarizations order
    E_c = s * inv_eps_c                        D-minus-P on the LEFT, stored

THREE THINGS DECIDE BIT-IDENTITY AND ALL THREE ARE INHERITED
------------------------------------------------------------
They are ``dispersive_update_e``'s, measured there on 30 real Grid/Fields/PML
cases, and they are inherited because the SOURCE half of the two sub-steps is the
same source half:

1. **THE SUM OF P CANNOT BE PRE-ACCUMULATED.** ``(D - P0) - P1`` and
   ``D - (P0 + P1)`` are different float32 numbers — caught 20/20 at two poles,
   0/10 at one, which is the right split for a multi-pole-only defect. Anyone
   reading only the sentence "E = (D - sum P) * inv_eps" will write the wrong one.
2. **THE POLE ORDER IS LOAD-BEARING AND THE POLE SET IS PER COMPONENT.**
   ``PolarizationState._driven`` (dispersion.py:640-642) drops any component whose
   sigma is identically zero, so Ex, Ey and Ez can each see a different subset in a
   different order. Reversing it was caught 20/20.
3. **THE OPERAND ORDER IS KEPT** — ``s * inv_eps``, D-minus-P on the left, as
   stepping.py:1011-1013 writes it. This one is a NULL and is recorded as such:
   float32 multiplication is bitwise commutative and the swap was caught 0/30. It
   is kept for transcription discipline, and a gate leg reporting it "caught" is
   comparing something other than bytes.

WHAT IS DELETED RELATIVE TO ``dispersive_update_e``, AND WHY EACH DELETION IS
FORCED RATHER THAN CHOSEN
---------------------------------------------------------------------------
* **The ``f_w`` store.** Without an absorber ``Fields.drive_field`` returns the
  STORED E, not ``f_w`` (fields.py:1160-1162), and ``f_w_*`` is not allocated at
  all. Writing one would be a store into ``None``; requiring one would refuse every
  configuration this arm exists for. This is the same fact
  ``no_pml_ade.no_pml_ade_update_p_coverage`` inverts on the ``update_P`` side, and
  the two arms are each other's other half: Arm S writes the array that arm reads.
* **The ``prev`` load.** It exists only to be multiplied by ``kms``.
* **The split-field tail.** ``(E + kps*src) - kms*prev`` becomes ``E = src``. There
  are no ``kps``/``kms`` tables on an inert layer to read.

**THIS IS NOT ``kernels.constitutive_step`` UNDER A RESTATED PREDICATE**, and that
was measured rather than assumed (``no_pml_constitutive.STORED_E_ARM``). The
certified body ACCUMULATES and also writes ``f_w``::

    prev = w[i]; w[i] = src; f[i] = (f[i] + kps*src) - kms*prev

Binding ``kps = 1``, ``kms = 0`` and a zeroed ``f`` does not recover ``f[i] = src``:
``0.0 + (-0.0)`` is ``+0.0``, so every negative zero in the source is canonicalised
away, and ``kms * prev`` is ``0.0 * prev``, which is NaN for a non-finite ``prev``.
``f`` is not zero in a real run either — it holds the previous step's E.

THE DEGENERATE CASE IS ADMITTED ON PURPOSE. With no pole driving a component the
array path returns the D array ITSELF from ``displacement_minus_polarization``
(fields.py:1097-1098) and stores ``D * inv_eps``; the kernel's constexpr chain
compiles to zero subtractions and stores the same product. A run with ``stores_E``
and NO poles anywhere is therefore covered, and nothing else covers it — the null
arm refuses it at ``stores_E`` and every other ``update_E`` arm refuses it at
``pml.is_active``. No corpus row is in that state today; the arm is correct there
regardless, and saying so is cheaper than a clause that would have to be justified.

DEVICE RESULT
-------------
``parity/meep_gpu/gate_triton_no_pml_stored_e.py`` ran on the GPU host's RTX A6000
on 2026-08-17 under the installed ``keep`` policy. Five product rows were
byte-identical over eight launches, including the full ``update_E`` then
``update_P`` pointer-rotation row. Reversing two poles, dropping the last pole,
and freezing the rotating pointers diverged by 228, 1,536 and 7,680 words; the
one-pole reversal control stayed identical. The readable provenance entry is
``triton_no_pml_stored_e_device_gate`` in ``fingerprints.json``. This is a
correctness result, not a throughput claim.

SUBNORMALS ARE THIS SUB-STEP'S OWN HAZARD and the finding transfers whole from
``dispersive_update_e``: ``D - sum P`` is a CANCELLATION between two nearly equal
float32 volumes and ``(D - sum P) * inv_eps`` then underflows wherever the field is
small. Measured there, per operation, with NumPy as the neutral reference: **CuPy
flushes float32 subnormals to zero; Triton and NumPy keep them.** So on this
sub-step the kernel is the IEEE-correct side and the array path is not, and a byte
gate's cancellation-class failures are the ARRAY PATH's departure. The gate reports
the two classes separately and never merges them.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage

__all__ = [
    "E_TERMS",
    "MAX_POLES",
    "SHARED_CLAUSES",
    "StoredEConstitutivePlan",
    "explain_stored_e",
    "plan_stored_e_constitutive",
    "plan_stored_e_constitutive_from_arrays",
    "stored_e_constitutive_coverage",
    "stored_e_constitutive_step",
]

#: The most poles one component may carry, and it is THIS KERNEL'S constant rather
#: than a shared one: it is the length of the unrolled ``if NP > n`` chain in the
#: body below, so importing it from another module would let that module's edit
#: silently truncate this kernel's subtraction. ``test_triton_no_pml_stored_e``
#: pins it equal to the chain length AND equal to
#: ``dispersive_update_e.MAX_POLES``, which is the honest way to keep two
#: independently-written bodies in step.
#:
#: 8 is the corpus maximum with headroom: Ag — the only real multi-pole material,
#: used by the three ``stochastic_emitter`` scripts — has 6 (1 Drude + 5 Lorentz),
#: and this arm's own rows carry 5 (``absorber-1d.py``, ``TestAbsorber.test_absorber``)
#: and 2 (``material-dispersion.py``).
MAX_POLES = 8

#: The components this sub-step writes, in the order ``stepping.E_CONSTITUTIVE_TERMS``
#: (stepping.py:228) iterates them, paired with the D volume each one reads.
#:
#: NO AXIS COLUMN, and its absence is the point. ``dispersive_update_e.E_TERMS``
#: carries one because the PML tail indexes a per-axis coefficient table; there is
#: no coefficient on this path, so a third column here would be a field nothing
#: reads and the first reader to trust it would index the wrong thing.
E_TERMS: Tuple[Tuple[str, str], ...] = (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz"))

#: The shared clause builders this predicate composes from, named as data so the
#: laptop test can assert every one still exists in ``coverage.py``. A rename there
#: then fails at the merge bar instead of silently dropping a clause here.
SHARED_CLAUSES: Tuple[str, ...] = (
    "_call", "_layout_reasons", "_volume_reasons", "_inverse_epsilon_reasons",
    "_susceptibility_reasons", "ELECTRIC_COMPONENTS")


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Defined at MODULE scope behind a conditional import, not inside a builder:
# `@triton.jit` resolves a body's names — including the `tl.constexpr` annotations
# — through the defining module's `__globals__`, so a kernel defined inside a
# function compiles to `NameError('tl is not defined')` at first launch
# (dispersive_update_e.py:138-143 records the run that showed it, 320 of 320).

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the byte gate

    @triton.jit
    def stored_e_constitutive_step(
        f0, f1, f2,                     # targets: Ex, Ey, Ez            (out)
        g0, g1, g2,                     # sources: Dx, Dy, Dz            (in)
        e0, e1, e2,                     # inverse epsilon, per component (in)
        a0, a1, a2, a3, a4, a5, a6, a7,   # component 0's poles, IN ORDER
        b0, b1, b2, b3, b4, b5, b6, b7,   # component 1's poles, IN ORDER
        c0, c1, c2, c3, c4, c5, c6, c7,   # component 2's poles, IN ORDER
        n_elem,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``stepping.update_E`` with an inert layer and stored E, all three components.

        The body is :func:`dispersive_update_e.constitutive_step_dispersive`'s with
        the ``f_w`` store, the ``prev`` load and the split-field tail deleted, and
        ``stepping.py:1022``'s plain store put in their place. Everything that
        remains — the left-to-right subtraction chain, the polarizations order, the
        D-minus-P-on-the-left operand order — is that kernel's, unchanged, because
        the array path's is unchanged (stepping.py:1010-1013 is the same pair of lines
        under either layer).

        ``NP0``/``NP1``/``NP2`` are the LIVE pole count per component and are
        ``tl.constexpr``, so the chain is unrolled at compile time and a component
        with no pole compiles to exactly ``E = D * inv_eps`` — which is what
        ``displacement_minus_polarization`` does when nothing drives the component
        (fields.py:1097-1098, returning the D array itself). Unused pole slots are
        bound to the component's own ``D`` pointer by the plan and are never read; a
        device array of pointers would move the count out of the compiled
        specialization, which is where the bit-exact unrolled order lives.

        THE POLES ARE NOT PRE-SUMMED, and that is the whole design. ``D - (P0 + P1)``
        is a different float32 number and was caught 20/20 at two poles.

        NO ``nx``/``ny``/``nz``. This sub-step is element-wise with no stencil, no
        neighbour read and no per-axis coefficient, so the flat index is the only
        geometry it has. Taking the shape and not using it is how a later reader
        concludes the kernel is position-aware when it is not.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem

        # --- component 0 -------------------------------------------------------
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
        tl.store(f0 + idx, s0 * tl.load(e0 + idx, mask=live, other=0.0), mask=live)

        # --- component 1 -------------------------------------------------------
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
        tl.store(f1 + idx, s1 * tl.load(e1 + idx, mask=live, other=0.0), mask=live)

        # --- component 2 -------------------------------------------------------
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
        tl.store(f2 + idx, s2 * tl.load(e2 + idx, mask=live, other=0.0), mask=live)

else:  # pragma: no cover - the laptop path
    stored_e_constitutive_step = None  # type: ignore[assignment]


def stored_e_constitutive_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError."""
    if stored_e_constitutive_step is None:
        raise ImportError(
            "the no-PML stored-E update_E kernel needs the optional `triton` "
            "package (pip install triton). The engine runs without it; only this "
            f"fast path is unavailable. Original error: {_TRITON_IMPORT_ERROR}")
    return stored_e_constitutive_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def stored_e_constitutive_coverage(fields: Any, pml: Any) -> "_coverage.Coverage":
    """May Arm S step ``update_E`` for this ``(fields, pml)`` pair?

    POSITIVE CLAUSES ONLY. Every requirement is named and checked; a failing clause
    appends its reason and the scan continues, so a refusal reports everything that
    disqualified the run rather than the first thing.

    TWO INVERTED CLAUSES CARRY THE WHOLE DISJOINTNESS, and they invert two
    DIFFERENT incumbents, which is why neither one alone would do:

    a. **the layer must be INERT.** ``coverage._grid_reasons`` clause 3 requires an
       active PML and every other ``update_E`` arm inherits it — ``ordinary``,
       ``dispersive``, ``folded``, ``folded dispersive``, ``nonlinear``,
       ``off-diagonal`` and each complex/cylindrical/beta sibling. That is the same
       ``pml.is_active`` call ``stepping._pml_is_active`` (stepping.py:2498-2506)
       makes at stepping.py:982, so an all-zero-face layer lands HERE, not there;
    b. **E must be STORED.** ``no_pml_constitutive.null_constitutive_coverage``
       refuses ``stores_E`` True by name ("that is the STORE arm, specified in
       STORED_E_ARM and not built") and this arm requires it. Same
       ``fields.stores_E`` attribute, opposite sign, and it is exactly what
       stepping.py:983 branches on.

    Every other clause is a REFUSAL of a product this arm does not write, and each
    one names the product:

    * complex storage — the body is float32 and the volumes would be complex64;
    * an off-diagonal chi1inv row — ``update_E`` forms a row product that reads the
      other components at neighbouring cells (stepping.py:1001-1008) and stops being
      element-wise. That is ``offdiag_update_e``'s. Note that a surviving row also
      forces stored E (fields.py:1253-1255), so this clause is the ONLY thing
      holding this arm off the two ``TestMaterialGrid`` rows;
    * chi2/chi3 — the Pade factor REPLACES the constitutive product
      (stepping.py:999-1000). That is ``nonlinear_update_e``'s;
    * a fold, a cylindrical axis, a Bloch phase, beta, BFAST — CONSERVATIVE, and
      labelled so where each fires. The sub-step is element-wise over the stored
      extent, so none of them can change what it computes; they are refused because
      no gate has run this body on those grids and because the folded no-PML
      dispersive configuration, if it is ever wanted, is a separate arm rather than
      a widening of this one. Measured cost: 0 corpus slots
      (``predicate_coverage_2026-08-16_wired_convention/remaining.txt`` lists no
      fold-plus-dispersion-plus-no-PML row);
    * a susceptibility outside ``{lorentzian, drude}`` or driving a magnetic
      component, through ``coverage._susceptibility_reasons``;
    * more than :data:`MAX_POLES` poles on one component, refused BY NAME rather
      than silently truncated — a dropped pole is a smooth, plausible, wrong field;
    * any of ``P``/``P_prev`` missing or malformed. ``P_prev`` is not read by THIS
      kernel and is checked anyway: ``update_P`` rotates the three buffers as a set
      (dispersion.py:687-691), so a malformed ``P_prev`` becomes this kernel's
      ``P`` one step later.

    SIGMA IS NOT CHECKED HERE and must not be: it enters ``update_P``, never this
    sub-step. ``no_pml_ade.no_pml_ade_update_p_coverage`` is where it belongs.

    NO ``f_w`` CLAUSE, in either direction. The incumbent ``dispersive`` arm's
    predicate requires the coefficient tables that go with ``f_w``; this one must
    not require ``f_w`` (no no-PML run allocates it) and must not refuse it either
    — a ``Fields`` carrying a stale ``f_w`` beside an inert layer is refused by the
    PML-storage-mode clause below, which is the clause that actually makes such an
    object dangerous.
    """
    reasons: List[str] = []
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    # 1. CuPy backend. The kernel launches against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage — and the test is NOT ``force_complex_fields`` alone.
    #
    # A NONZERO k_point ALSO GIVES complex64 storage, with force_complex_fields
    # left False. The canonical test is the disjunction, and it lives in
    # complex_fields._complex_grid_reasons: "Either force_complex_fields or a
    # nonzero k_point puts the run here (the array path itself raises on real
    # storage with a phase, stepping.py:1893-1908)".
    #
    # MEASURED 2026-08-17, before the fix: at force_complex_fields=False with
    # has_bloch=True this clause stayed silent and the family ADMITTED — a
    # float32 body over complex64 volumes, which is a wrong answer rather than a
    # missing one. The complex sibling requires force_complex_fields=True and so
    # REFUSED the same configuration, meaning a Bloch run had the wrong family
    # take the slot while the right one declined. Nothing reported it: the two
    # predicates never double-admit, so no disjointness check fires.
    if (getattr(fields, "force_complex_fields", False)
            or _coverage._call(grid, "has_bloch", default=False)):
        reasons.append(
            "complex64 storage (force_complex_fields=True or a nonzero k_point): "
            "this body is float32 while Ex/Dx/P are complex64 there — an unbuilt "
            "kernel, not a rounding gap")

    # (a) THE FIRST INVERTED CLAUSE: the layer must be inert.
    if pml is not None and getattr(pml, "is_active", False):
        reasons.append(
            "an active PML layer is installed: under one update_E writes "
            "(E + kps*src) - kms*prev and stores f_w (stepping.py:1014-1018), which "
            "is dispersive_update_e's product, not this one's")

    # (a2) ...and ``Fields`` must not be in PML STORAGE MODE either.
    #      ``enable_pml_storage`` is a one-way switch (fields.py:678-721) that sets
    #      ``_pml_active`` — the flag ``drive_field`` branches on. A ``Fields``
    #      switched on beside an inert layer would have update_P read ``f_w``, which
    #      THIS sub-step does not write on this path. ``no_pml_ade`` refuses the
    #      same object from the other side, and ``no_pml``'s clause 3b refuses the
    #      curl's mirror image of it.
    if bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "Fields is in PML storage mode while the layer is inert: drive_field "
            "would hand update_P an f_w this sub-step never writes "
            "(fields.py:1160-1162) — a frozen drive field")

    # (b) THE SECOND INVERTED CLAUSE: E must be stored.
    if not getattr(fields, "stores_E", False):
        reasons.append(
            "E is recomputed from D rather than stored: update_E returns at "
            "stepping.py:983-984 and that configuration is "
            "no_pml_constitutive.null_constitutive_coverage(side='E')'s")

    # The products this arm does not write, each named.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed: update_E forms a row product "
            "that reads the other components (stepping.py:1001-1008) and stops being "
            "element-wise — that is offdiag_update_e's")
    if getattr(fields, "has_nonlinearity", False):
        reasons.append(
            "chi2/chi3 is installed: the Pade factor REPLACES the constitutive "
            "product (stepping.py:999-1000) — that is nonlinear_update_e's")

    # CONSERVATIVE, and each says so where it fires. See the docstring.
    if _coverage._call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (CONSERVATIVE: this sub-step is "
                       "element-wise over the stored extent and a fold cannot "
                       "change it, but no gate has run this body on a folded grid)")
    for axis in range(3):
        if _coverage._call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane (CONSERVATIVE, "
                           f"as above)")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried "
                       "(CONSERVATIVE, as above)")
    for axis in range(3):
        if _coverage._call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis "
                           f"(CONSERVATIVE, as above)")
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r} "
                       f"(CONSERVATIVE: a Bloch phase needs complex storage, which "
                       f"clause 2 has already refused)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (CONSERVATIVE: BFAST adds a curl term, not "
                       "a constitutive one)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
                       f"(CONSERVATIVE, as above)")

    reasons.extend(_coverage._susceptibility_reasons(fields))

    states = tuple(getattr(fields, "polarizations", ()) or ())
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
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return _coverage.Coverage(False, tuple(reasons))
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if total >= 2 ** 31:
        reasons.append(f"{total} cells exceeds the kernel's int32 index range")

    for index, state in enumerate(states):
        driven = _coverage._call(state, "driven", default=None)
        if driven is None:
            continue  # already reported
        for name in tuple(driven):
            for slot in ("P", "P_prev"):
                array = (getattr(state, slot, {}) or {}).get(name)
                if array is None:
                    reasons.append(
                        f"polarization {index} {slot}[{name}] is not allocated")
                    continue
                reasons.extend(_coverage._volume_reasons(
                    f"polarization {index} {slot}[{name}]", array, shape))

    names = tuple(term[0] for term in E_TERMS) + tuple(term[1] for term in E_TERMS)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_coverage._layout_reasons(fields, shape, names))
    reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))

    return _coverage.Coverage(not reasons, tuple(reasons))


def poles_per_component(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    """The driving states of each E component, IN ``fields.polarizations`` ORDER.

    The single place the order is derived here, so the predicate, the plan and the
    per-launch assertion cannot disagree about it. Getting it out of step is a
    different float32 number in every cell and no error anywhere.

    ``Fields.displacement_minus_polarization`` (fields.py:1096) builds exactly this
    list: ``[state for state in self.polarizations if state.drives(component)]``.

    NOT imported from ``dispersive_update_e``, deliberately: that module's copy is
    keyed to ITS ``E_TERMS`` and its own launch assertion, and a shared helper that
    two kernels' bit-exactness depends on is a single point at which a change for
    one silently re-orders the other. The two are pinned equal by test instead.
    """
    states = tuple(getattr(fields, "polarizations", ()) or ())
    return {name: tuple(state for state in states
                        if _coverage._call(state, "drives", name, default=False))
            for name in _coverage.ELECTRIC_COMPONENTS}


class LivePoleBinding:
    """The engine route's pole source: ``state.P[c]``, RESOLVED PER LAUNCH.

    ``PolarizationState.update`` rotates ``P``/``P_prev``/``_scratch`` for every
    component on every step (dispersion.py:687-691): this step's result lands in
    ``_scratch``, ``_scratch`` takes over the array that held ``P_prev``, and
    ``P``/``P_prev`` shift down. A plan that cached ``P`` device views would be
    stale after the FIRST component of the FIRST step — and stale in a way that
    still computes, giving a smooth wrong field.

    So the ORDER is cached (it is fixed for the run — the driver refuses a
    susceptibility appended mid-run) and the POINTERS are re-derived every call,
    with the order re-checked each time.
    """

    __slots__ = ("fields", "order", "counts")

    def __init__(self, fields: Any, order: Dict[str, Sequence[Any]]) -> None:
        self.fields = fields
        self.order = {name: tuple(states) for name, states in order.items()}
        self.counts = tuple(len(self.order[term[0]]) for term in E_TERMS)

    def arrays(self) -> Tuple[Tuple[Any, ...], ...]:
        live = poles_per_component(self.fields)
        out: List[Tuple[Any, ...]] = []
        for component, _ in E_TERMS:
            expected = self.order[component]
            found = live[component]
            if len(found) != len(expected) or any(
                    a is not b for a, b in zip(expected, found)):
                raise RuntimeError(
                    f"the polarization set driving {component} changed since the "
                    f"plan was built ({len(expected)} states -> {len(found)}); the "
                    f"pole ORDER is bit-load-bearing and this plan is no longer valid")
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

class StoredEConstitutivePlan:
    """A launchable, allocation-free no-absorber stored-E ``update_E``.

    Everything resolvable ahead of the loop is resolved in ``__init__``: the field
    pointer adapters, the pole ORDER and the three ``NP`` constexprs. What is
    deliberately NOT resolved ahead of the loop is the pole POINTERS — see
    :class:`LivePoleBinding`.

    NO ``f_w`` SLOTS AND NO COEFFICIENT SLOTS, and the absence is structural rather
    than defaulted: a plan that carried them would let a caller bind an active
    layer's tables to a kernel that ignores them, which is a silent half-step.
    """

    __slots__ = ("shape", "n_elem", "block", "counts", "_targets", "_sources",
                 "_inv_eps", "_poles", "_grid", "_kernel", "_pointer")

    def __init__(self, shape, block: int, targets, sources, inverse_epsilon,
                 poles, kernel: Any = None) -> None:
        from .launch import CupyPointer  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self._pointer = CupyPointer
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._poles = poles
        self.counts = tuple(int(n) for n in poles.counts)
        for term, count in zip(E_TERMS, self.counts):
            if count > MAX_POLES:
                raise ValueError(
                    f"{term[0]} is driven by {count} poles; the kernel compiles "
                    f"{MAX_POLES} slots and the predicate refuses more")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as the other plans.

        ``guard=None`` takes the shipped constant; the gate passes True/False
        explicitly so the contraction guard's effect is MEASURED rather than assumed.
        It matters here even though the tail is a bare store: ``s * inv_eps`` with
        ``s`` a chain of subtractions is a subtract feeding a multiply, which is the
        shape LLVM contracts into ``fma.rn.f32`` when the consumer is one.
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else stored_e_constitutive_kernel())
        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not the "
                    f"{self.counts[index]} the plan compiled for")
            # Unused slots take this component's own D pointer. The constexpr
            # `if NP > n` chain is what stops them being read; a pointer argument
            # still has to type, and a null makes the launcher's failure a
            # TypeError far from its cause.
            slots.extend(self._pointer(array) for array in group)
            slots.extend([self._sources[index]] * (MAX_POLES - len(group)))
        kernel[self._grid](
            *self._targets, *self._sources, *self._inv_eps, *slots,
            self.n_elem,
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"StoredEConstitutivePlan(shape={self.shape}, poles={self.counts}, "
                f"block={self.block})")


def plan_stored_e_constitutive(fields: Any, pml: Any, block: Optional[int] = None
                               ) -> Optional[StoredEConstitutivePlan]:
    """Build Arm S from the engine's own objects, or None when refused.

    None means REFUSED, and the reasons are available from
    :func:`stored_e_constitutive_coverage`. The Triton import stays BELOW the
    predicate, as in every builder in this package: a NumPy host must be able to
    plan (to ``None``) without the optional dependency being importable at all.
    """
    if not stored_e_constitutive_coverage(fields, pml).covered:
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    order = poles_per_component(fields)
    targets = tuple(term[0] for term in E_TERMS)
    return StoredEConstitutivePlan(
        fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        [getattr(fields, term[1]) for term in E_TERMS],
        [fields.inverse_epsilon_for(name) for name in targets],
        LivePoleBinding(fields, order))


def plan_stored_e_constitutive_from_arrays(arrays: Dict[str, Any],
                                           poles: Dict[str, Sequence[Any]],
                                           block: Optional[int] = None,
                                           kernel: Any = None
                                           ) -> StoredEConstitutivePlan:
    """Build it from bare device arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name (``Ex``, ``Dx``, ``inv_eps_Ex``) and
    ``poles`` maps each component to its ordered list of P volumes. No coverage
    predicate runs: the caller is a harness that has constructed the configuration
    on purpose, and refusing it would defeat the point of a mutation leg.

    ``kernel=`` IS LOAD-BEARING: the mutation leg routes a mutated kernel through
    it, and a builder that drops it launches the SHIPPED kernel and reports a pass
    for a defect it never introduced.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return StoredEConstitutivePlan(
        shape, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets],
        [arrays[term[1]] for term in E_TERMS],
        [arrays["inv_eps_" + name] for name in targets],
        StaticPoleBinding([poles[name] for name in targets]),
        kernel=kernel)


def explain_stored_e(fields: Any, pml: Any) -> "_coverage.Coverage":
    """The verdict with its reasons, for reports. Needs no Triton."""
    return stored_e_constitutive_coverage(fields, pml)


# ---------------------------------------------------------------------------
# WIRING AND DEVICE CERTIFICATION — COMPLETE
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` now places this arm immediately before ``no-PML null`` in
# the ``update_E`` table, with the same inert-layer gate its predicate reads.
# ``__init__.py`` exposes the two lazy entry points and ``launch.FAMILY_MODULES``
# names this module.  It remains a NEW family for dispatch certification: it may
# not be credited to ``dispersive_composition``, whose body stores ``f_w`` and
# multiplies by kps/kms while this one does neither.
# Its dedicated 2026-08-17 A6000 byte gate is recorded under
# ``triton_no_pml_stored_e_device_gate`` and licenses opted-in fast-path
# dispatch.
#
# AND ONE DELETION THAT IS NOT OPTIONAL. ``no_pml_constitutive.STORED_E_ARM`` is
# the record of the decision NOT to build this arm. It is corrected in the same
# change that adds this file — ``built`` is True and ``measured_demand`` states
# what the arm now buys — because a record that says "not built" beside a built
# arm is worse than no record.
