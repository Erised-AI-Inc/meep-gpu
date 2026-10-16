"""Conductivity + PML Triton kernel: MEEP's conductive ``step_curl``, per structure chunk.

ONE kernel, one predicate, two plan builders. The kernel is
:func:`conductive_pml_curl_step`; ``launch.plan_step`` asks
:func:`conductive_pml_curl_coverage` for its ``conductive PML`` arm on ``step_B`` and
``step_D``, and :mod:`.conductive_fused_electric_pair` lifts the same body into the
``step_D`` / ``update_E`` weld.

* :func:`conductive_pml_curl_step` — ``stepping.step_B`` / ``stepping.step_D`` on a
  grid that carries BOTH an active split-field PML and a material conductivity on
  one or more of that sub-step's three targets. That is the path
  ``stepping._apply_curl`` serves through
  ``stepping._apply_conductive_pml_update``, the engine's most expensive curl: it
  evaluates every form over the whole volume and selects with ``copyto``, where
  this kernel evaluates them per cell and stores once.

THE FORMS ARE CHOSEN PER STRUCTURE CHUNK, from THREE per-chunk facts, as MEEP chooses
them (``step_generic.cpp::step_curl``, step_db.cpp:56-59, structure.cpp:883-900): does
the cell's chunk carry the dsig layer, does it carry the dsigu layer, and does it
keep a conductivity array for the component. The first two are lines of the run's
chunk record (``chunk_map.chunk_bounds_record``, bound as ``cb``), read on the
component's own sub-lattice -- half-integer for every B target, integer for every
D one -- exactly as :func:`.kernels.pml_curl_step` reads its dsigu line. The third
is a per-chunk fact, not a per-axis one, so it is not in the record: it is the
volume ``chunk_map.ChunkLayout.conductive_mask`` of the component's chunks, bound per
component as ``cm`` (``uint8``) when some chunk keeps no conductivity
(``CMASK == 1``) and not bound at all when every chunk keeps one (``CMASK == 0``),
which is the array path's ``conductive=None``. Notation: ``f`` = target, ``u`` =
``fu_<target>``, ``c`` = ``f_cond_<target>``, ``X`` = the curl (already carrying
dtdx), ``cf``/``ci`` = ``condfac``/``condinv``, ``(km1, si1)`` = ``kms``/``sinv`` on
the dsig axis, ``(km2, si2)`` on the dsigu axis; every operation rounds to float32 in
this order (``stepping._apply_conductive_pml_update``)::

    conductive chunk, dsig and dsigu    c' = ((c*cf) - X) * ci
                                        u' = ((u*km1) + (c' - c)) * si1
                                        f' = (((f*km2) + u') - u) * si2
    conductive chunk, dsigu only        u' = ((u*cf) - X) * ci
                                        f' = (((f*km2) + u') - u) * si2     c kept
    conductive chunk, dsig only         c' = ((c*cf) - X) * ci
                                        f' = ((f*km1) + (c' - c)) * si1     u kept
    conductive chunk, neither           f' = ((f*cf) - X) * ci              u, c kept
    chunk with no conductivity          :func:`.forms.curl_form`: the recurrence on
                                        dsigu cells, ``((f*km1) - X) * si1`` elsewhere

The increment ``(c' - c)`` is formed FIRST and then added: MEEP's grouping
(step_generic.cpp:167, :209). ``((u*km1) + c') - c`` is a different float32 number
and is what this kernel carried before the alignment. "kept" is the auxiliary
contract: ``u`` is read and written on dsigu cells only and ``c`` on the dsig cells
of conductive chunks only; everywhere else neither is loaded nor stored. ``cf`` and
``ci`` are FULL VOLUMES of ``grid.shape``, per component (``condfac = 1 -
sigma*dt/2``, ``condinv = 1/(1 + sigma*dt/2)``, MEEP structure.cpp:693-706).

THE SELECTS ARE OUTPUT SELECTS. Every form is evaluated on whatever the masked loads
delivered and ``tl.where`` chooses a word; nothing a discarded form computes feeds a
retained one, and a discarded form never reaches a store. The form a cell takes is
membership, never coefficient identity: a cell inside a chunk whose own sigma is
exactly zero (coefficients exactly 1) still takes its chunk's form, which is where
the two differ.

DO NOT REGROUP THE PARENTHESES ABOVE, and do not let a multiply contract into an FMA.
``ENABLE_FP_FUSION`` is spelled once for the whole package, in :mod:`kernels`, and
this file's single launch site passes it.

PER-COMPONENT CONDUCTIVITY IS A CONSTEXPR. ``_apply_curl`` reads
``fields.condfac_for(term.target)`` PER COMPONENT, and a ``D_conductivity_diag`` with
a zero entry, or a material sigma on one component only, leaves a component lossless
beside two lossy ones. ``COND0``/``COND1``/``COND2`` carry that per component: where
a component is lossless the conductive branch is compiled out and the component runs
:func:`.forms.curl_form`, the same helper :func:`.kernels.pml_curl_step` calls, so a
partially conductive launch is identical on its lossless components by construction.
Both the predicate and the constexpr read :func:`conductive_targets`, so the check
and the compile-time choice cannot disagree.

WHAT ELSE IS TRANSCRIBED. The curl, the ghost rule and the ownership mask are
:func:`.kernels.pml_curl_step`'s: conductivity enters ``_apply_curl`` only after the
curl is formed, so this kernel's head is that kernel's head line for line, stencil
through :func:`.forms.curl_stencil` and the record's presence word included, and
``test_triton_conductivity`` diffs the two textually.

WHAT THE GATE MEASURES (``parity/meep_gpu/gate_triton_conductivity.py``): its
reference against ``stepping._apply_conductive_pml_update`` itself, bytewise on NumPy
(the ``validate`` leg, at the merge bar through ``test_triton_conductivity``); the
single-launch sweep over shapes x boundary sets x dtdx {0.5, 0.35} x {step_B, step_D}
x conductivity pattern x chunk division x conductive-chunk volume, comparing the
target, its ``fu`` and its ``f_cond`` as bytes; the multi-step, decay and control
legs; and the source mutations, with the inversions of KERNELS.md section 7.1 and the
B battery of section 7.2. The 2026-08-10 / 2026-08-19 device record
(``results/triton_conductivity_2026-08-10/``) measured the forms this kernel carried
before the alignment; this body has not been re-gated on a device beyond lane smoke.

Dispatch is governed by ``fastpath`` and the subnormal policy as for every Triton
kernel. Every one of the conductive forms multiplies by ``condfac*condinv < 1`` every
step on top of the PML decay, so this kernel manufactures subnormal inputs faster than
any other in the set; the decay leg exists for that.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import (
    Coverage,
    COVERED_BOUNDARIES,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _layout_reasons,
    _susceptibility_reasons,
    _volume_reasons,
)

# ---------------------------------------------------------------------------
# The constants the kernel and the host agree on
# ---------------------------------------------------------------------------

#: The two sub-steps this kernel serves, and the targets each one writes. Keyed
#: exactly as ``launch.SUB_STEPS`` is, and pinned equal to it by test: a plan
#: built here and a plan built there must name the same components in the same
#: order or the coefficient pairing means nothing.
CONDUCTIVE_SUB_STEPS: Dict[str, Tuple[str, str, str]] = {
    "step_B": ("Bx", "By", "Bz"),
    "step_D": ("Dx", "Dy", "Dz"),
}

#: Which axis pair the split-field recurrence reads per target — vec.hpp's
#: cycle_direction, the same triple on both sides (``launch.DSIG_AXES``).
DSIG_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))


def conductive_targets(fields: Any, sub_step: str) -> Tuple[bool, bool, bool]:
    """Which of this sub-step's three targets carry a conductivity.

    THE SINGLE PLACE THAT DECIDES IT. Both :func:`conductive_pml_curl_coverage`
    and the ``COND0``/``COND1``/``COND2`` constexprs read this function, so the
    clause and the compile-time choice cannot disagree. Getting them out of step
    is the over-covering failure the module docstring names: a lossless component
    stepped through the conductive branch would dereference a placeholder pointer
    and grow a spurious ``f_cond`` history — smooth, plausible, wrong, and no
    exception.

    ``condfac_for`` is MEEP's own granularity (``s->conductivity[c][d]``, per
    component), so "the run is conductive" is not a question this file asks.
    """
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        return (False, False, False)
    flags: List[bool] = []
    for target in CONDUCTIVE_SUB_STEPS[sub_step]:
        try:
            flags.append(reader(target) is not None)
        except Exception:  # noqa: BLE001 - an unanswerable component is not conductive
            flags.append(False)
    return (flags[0], flags[1], flags[2])


def _kernel_module():
    """Import :mod:`kernels` lazily — it imports Triton, this module must not.

    Same discipline as :mod:`launch` and :mod:`symmetry`: a NumPy host with no
    optional dependency must be able to import this file, read its predicate and
    get a refusal.
    """
    from . import kernels  # noqa: PLC0415

    return kernels


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
# Deferred behind a flag so the module imports with Triton absent. Everything
# below the guard is device code; everything above it is the predicate and the
# plan, which run anywhere.

try:  # pragma: no cover - exercised by the Triton-absent test
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - a missing optional dependency is not an error
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if triton is not None:  # pragma: no cover - device code, gated by the shared gate

    from .forms import axis_present, curl_form, curl_stencil, record_in  # noqa: E402
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

    #: Boundary codes, identical to :mod:`kernels`' own. A plan built here and a
    #: plan built there index the same table; the test pins the two equal.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    @triton.jit
    def _conductive_component(f_ptr, u_ptr, c_ptr, cf_ptr, ci_ptr, m_ptr,
                              idx, live, curl, km1, si1, km2, si2, ds, du,
                              COND: tl.constexpr, CMASK: tl.constexpr):
        """One component's curl forms: the conductive ones, or the lossless pair.

        Written once and called three times rather than triplicated, because the
        thing most likely to go wrong here is a parenthesis and three copies of a
        parenthesis is three chances to lose one.

        ``ds`` / ``du`` are the cell's chunk memberships of the component's dsig and
        dsigu layers (the record's lines, on the component's sub-lattice). ``COND ==
        0`` is a lossless component: :func:`.forms.curl_form` and nothing else
        compiles -- no ``f_cond``, ``condfac`` or ``condinv`` load. ``COND == 1``
        evaluates the four conductive forms and selects with ``tl.where``; with
        ``CMASK == 1`` the cell's chunk may keep no conductivity (``m_ptr``, the
        component's conductive-chunk volume), and there the lossless pair is selected
        instead, as the array path selects it.

        THE STORE MASKS ARE THE AUXILIARY CONTRACT. ``u`` is loaded and stored only
        where ``du``, ``c`` only where ``ds`` in a conductive chunk; ``f`` is stored on
        every live cell. ``live`` is ANDed into every mask: past the end of the array
        a membership test still answers, and a store without ``live`` writes past
        the allocation.
        """
        f = tl.load(f_ptr + idx, mask=live, other=0.0)
        u = tl.load(u_ptr + idx, mask=live & du, other=0.0)
        if COND == 0:
            v, n = curl_form(f, u, curl, km1, si1, km2, si2, du)
            tl.store(u_ptr + idx, n, mask=live & du)
            tl.store(f_ptr + idx, v, mask=live)
        else:
            if CMASK == 0:
                kept = live
            else:
                kept = tl.load(m_ptr + idx, mask=live, other=0) != 0
            c = tl.load(c_ptr + idx, mask=live & ds & kept, other=0.0)
            cf = tl.load(cf_ptr + idx, mask=live, other=0.0)
            ci = tl.load(ci_ptr + idx, mask=live, other=0.0)

            # The conductive forms. DO NOT REGROUP THESE PARENTHESES: the increment
            # (c_new - c) is formed before it is added (MEEP's grouping).
            c_new = ((c * cf) - curl) * ci                     # dsig cells
            increment = c_new - c
            u_split = ((u * km1) + increment) * si1            # dsig and dsigu
            u_cond = ((u * cf) - curl) * ci                    # dsigu only
            u_new = tl.where(ds, u_split, u_cond)
            f_split = (((f * km2) + u_new) - u) * si2          # dsigu cells
            f_first = ((f * km1) + increment) * si1            # dsig only
            f_direct = ((f * cf) - curl) * ci                  # neither
            v = tl.where(du, f_split, tl.where(ds, f_first, f_direct))
            if CMASK != 0:
                # A chunk that keeps no conductivity steps the lossless pair.
                v_plain, n_plain = curl_form(f, u, curl, km1, si1, km2, si2, du)
                v = tl.where(kept, v, v_plain)
                u_new = tl.where(kept, u_new, n_plain)

            tl.store(f_ptr + idx, v, mask=live)
            tl.store(u_ptr + idx, u_new, mask=live & du)
            tl.store(c_ptr + idx, c_new, mask=live & ds & kept)

    @triton.jit
    def conductive_pml_curl_step(
        f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
        u0, u1, u2,                       # auxiliaries: fu_B*  or  fu_D*
        c0, c1, c2,                       # f_cond_*  (a placeholder where COND == 0)
        cf0, cf1, cf2,                    # condfac   (a placeholder where COND == 0)
        ci0, ci1, ci2,                    # condinv   (a placeholder where COND == 0)
        m0, m1, m2,                       # conductive chunks (a placeholder where CMASK == 0)
        g0, g1, g2,                       # sources: Ex,Ey,Ez  or  Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, one sub-lattice
        cb,                               # chunk record: the forms' membership and presence
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
        CMASK0: tl.constexpr, CMASK1: tl.constexpr, CMASK2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One curl sub-step with conductivity, every PML form included.

        The head below is :func:`kernels.pml_curl_step`'s, line for line:
        conductivity enters ``_apply_curl`` only AFTER the curl is formed, so the
        ghost rule, the stencil and its partner presence, and the ownership mask are
        the plain path's, unchanged. The record's two membership lines per target
        then key the forms in :func:`_conductive_component`.
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

        # --- the forms, per component ----------------------------------------------
        # dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on
        # both sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        # The component's dsig and dsigu chunk memberships, on its own sub-lattice
        # (half-integer for every B target, integer for every D one): each record line
        # is a compile-time word index, chunk_map.record_index(half_integer, axis, 0).
        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        ds0 = record_in(cb, lattice + 4, j)
        du0 = record_in(cb, lattice + 8, k)
        ds1 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        ds2 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        _conductive_component(f0, u0, c0, cf0, ci0, m0, idx, live, curl0,
                              km_y, si_y, km_z, si_z, ds0, du0, COND0, CMASK0)
        _conductive_component(f1, u1, c1, cf1, ci1, m1, idx, live, curl1,
                              km_z, si_z, km_x, si_x, ds1, du1, COND1, CMASK1)
        _conductive_component(f2, u2, c2, cf2, ci2, m2, idx, live, curl2,
                              km_x, si_x, km_y, si_y, ds2, du2, COND2, CMASK2)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _shared_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """Everything :func:`coverage._grid_reasons` refuses EXCEPT the conductivity.

    RE-STATED, not imported-and-subtracted, for the reason
    :func:`symmetry._shared_grid_reasons` gives: subtracting a reason string from
    another predicate's output would make this file's coverage a function of that
    file's phrasing, and a clause renamed there would silently widen coverage
    here. Every clause below names its own condition, in the same order and for
    the same reason as the original, with clause 8 (the conductivity) replaced by
    the per-sub-step clauses in the caller.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernel launches against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. The recurrence is the same shape in complex64, the STORAGE
    #    is not, and a complex run stepped as float32 reads the wrong stride.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")

    # 3. An absorber that actually absorbs. Without one the array path takes
    #    `_apply_conductive_update` (stepping.py:1938), a DIFFERENT recurrence —
    #    case D alone, with no f_cond and no fu — and this kernel must not claim it.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this kernel implements the conductive "
                       "split-field path only; without PML the array path takes "
                       "stepping._apply_conductive_update)")

    # 4. Only the two ghost rules the curl half writes.
    kinds = _boundary_kinds(grid, pml if (pml is not None and getattr(pml, "is_active", False))
                            else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside {COVERED_BOUNDARIES}")

    # 5. No mirror plane anywhere: a fold changes the ghost rule, adds a parity
    #    mask at cell 0 and CHANGES THE STORED EXTENT, which moves every cell's
    #    coefficient index. The folded kernel is `symmetry.pml_curl_step_folded`
    #    and it carries no conductivity, so the intersection is nobody's yet.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian only. An axial extent moves the coefficient index, and the
    #    cylindrical r axis has its own ghost and ownership rules.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0. A Bloch phase needs complex storage and multiplies one wrapped plane.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 8. The conductivity clause is the one this kernel LIFTS; it is replaced by
    #    the per-target clauses in conductive_pml_curl_coverage and is deliberately
    #    absent here rather than merely unstated.

    # 10. No instantaneous nonlinearity. The Pade factor REPLACES the constitutive
    #     product (stepping.py:970-971) and composes with dispersion.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not carried)")

    # 11/12. BFAST adds a second additive term to every curl; beta adds
    #        out-of-plane couplings. Both are silent additions, not errors.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    return reasons


def conductive_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May :func:`conductive_pml_curl_step` step this sub-step of this run?

    Written POSITIVELY: what is admitted is enumerated and everything else is
    refused. In one sentence, what is admitted is a real-field Cartesian CuPy run
    under an active split-field PML, at k = 0 on every axis, with periodic or
    metallic ghost rules, no fold, no cylindrical axis, no BFAST, no special_kz,
    no instantaneous nonlinearity, whose susceptibilities (if any) are electric
    Lorentz/Drude poles, and **at least one of whose three targets for THIS
    sub-step carries a material conductivity** whose ``condfac``, ``condinv`` and
    ``f_cond`` volumes are all present, float32, C-contiguous and grid-shaped.

    PER SUB-STEP, and that is the whole point of the file. ``coverage._grid_reasons``
    puts the conductivity clause in the SHARED grid reasons, so one conductive
    component disqualifies every sub-step on the grid — which is why the
    ``2d_cond_pml`` benchmark case is refused wholesale even though its magnetic
    side is an ordinary lossless PML curl the shipped kernel already computes.
    Conductivity on the OTHER side's targets is therefore NOT a reason here.

    Dispersion is ADMITTED, for :func:`coverage.pml_curl_coverage`'s reason: the
    curl differences the STORED E and H arrays and never reads a polarization.

    A run with no conductivity on any of this sub-step's targets is REFUSED, not
    silently served: that configuration belongs to :func:`kernels.pml_curl_step`,
    and two kernels claiming one slot is how a composer ends up building both.
    """
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    targets = CONDUCTIVE_SUB_STEPS[sub_step]
    reasons = _shared_grid_reasons(fields, pml, grid)
    reasons.extend(_susceptibility_reasons(fields))

    # STORED E — the invariant behind admitting dispersion (coverage.py:273-281).
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # The three auxiliaries and the three sources THIS sub-step touches. The other
    # side's are not this sub-step's business, per the per-sub-step rule above.
    sources = ("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Hx", "Hy", "Hz")
    names = tuple(targets) + tuple("fu_" + t for t in targets) + sources
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))

    # The conductivity clauses, per target, and the reason this predicate exists.
    reader = getattr(fields, "condfac_for", None)
    inverse_reader = getattr(fields, "condinv_for", None)
    if not callable(reader) or not callable(inverse_reader):
        reasons.append("fields does not expose condfac_for/condinv_for; a "
                       "conductivity this predicate cannot read is not a covered one")
    else:
        flags = conductive_targets(fields, sub_step)
        if not any(flags):
            reasons.append(
                f"no {sub_step} target carries a conductivity: that configuration is "
                f"kernels.pml_curl_step's, and the two kernels must not both claim "
                f"the slot")
        for index, target in enumerate(targets):
            condfac = reader(target)
            condinv = inverse_reader(target)
            if (condfac is None) != (condinv is None):
                reasons.append(
                    f"{target} has condfac={'a volume' if condfac is not None else None} "
                    f"but condinv={'a volume' if condinv is not None else None}; the "
                    f"recurrence needs both or neither")
                continue
            if not flags[index]:
                continue  # A lossless component compiles to the plain recurrence.
            if len(shape) == 3:
                reasons.extend(_volume_reasons(f"condfac[{target}]", condfac, shape))
                reasons.extend(_volume_reasons(f"condinv[{target}]", condinv, shape))
            history = getattr(fields, "f_cond_" + target, None)
            if history is None:
                # `_apply_curl` RAISES on this (stepping.py:487-491). A predicate
                # must refuse it by name instead, or the refusal arrives as a
                # traceback out of a step the array path would have completed.
                reasons.append(f"f_cond_{target} is not allocated, but {target} is "
                               f"conductive (Fields._ensure_conductive_pml_storage)")
            elif len(shape) == 3:
                reasons.extend(_volume_reasons(f"f_cond_{target}", history, shape))
            # WHICH CHUNKS KEEP IT is the forms' third test, and it is read from the
            # component's own conductivity volume (Fields.conductivity_for, MEEP's
            # s->conductivity[c][d]); a component whose volume cannot be read cannot
            # have that test answered.
            sigma_reader = getattr(fields, "conductivity_for", None)
            if not callable(sigma_reader) or sigma_reader(target) is None:
                reasons.append(
                    f"{target} is conductive but fields does not expose its "
                    f"conductivity volume (conductivity_for), so which chunks keep "
                    f"it cannot be decided")

    # The coefficient tables, on THIS sub-step's Yee sub-lattice only: B reads the
    # half-integer set and D the integer one (stepping._curl_coefficients :2417).
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if sub_step == "step_B" else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), suffix))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class ConductivePmlCurlPlan:
    """A launchable, allocation-free conductive PML curl sub-step.

    Deliberately a sibling of :class:`launch.PmlCurlPlan` rather than a subclass:
    the two hold different bindings (this one carries twelve more pointers) and a
    subclass that inherited ``run`` would launch the plain kernel on a conductive
    grid — a wrong answer that never raises. Built two ways —
    :func:`plan_conductive_pml_curl` from the engine's own objects and
    :func:`plan_conductive_from_arrays` from bare device arrays for the gate — and
    launched ONE way, so the bytes the gate certifies are the bytes the engine
    would launch.

    ``chunk_bounds`` is the chunk record (``launch.bind_chunk_bounds``), required:
    its lines key every form, and the plan keeps the read-only host copy as
    :attr:`chunk_bounds` and launches with the device copy as ``cb``.
    ``conductive_chunks`` is, per target, the volume of cells whose chunk keeps the
    component's conductivity (``chunk_map.ChunkLayout.conductive_mask``), or
    ``None`` when every chunk keeps it; a volume is bound as ``uint8`` and its
    target's ``CMASK`` is 1. It is meaningful only for a conductive target.

    A LOSSLESS COMPONENT BINDS ITS OWN TARGET AS THE PLACEHOLDER for the three
    conductive slots, and every component without a conductive-chunk volume does
    for the fourth, never a null. The loads sit behind a ``tl.constexpr`` branch and
    are compiled away, but a pointer argument still has to type, and passing
    ``None`` makes the launcher's failure a ``TypeError`` far from its cause. The
    placeholder is never read and never written.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc", "cond",
                 "cmask", "block", "chunk_bounds", "_targets", "_aux", "_history",
                 "_condfac", "_condinv", "_chunks", "_sources", "_coefficients", "_cb",
                 "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, cond, block: int,
                 targets, auxiliaries, history, condfac, condinv, sources,
                 coefficients, kernel=None, *, chunk_bounds: Any,
                 conductive_chunks: Any = (None, None, None)) -> None:
        from .launch import SUB_STEPS, CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        if sub_step not in CONDUCTIVE_SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                             f"got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "ConductivePmlCurlPlan")
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy/CuPy cast to float32 before the multiply. Triton types a
        # Python float argument as fp32, so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.cond = tuple(1 if flag else 0 for flag in cond)
        if not any(self.cond):
            raise ValueError(
                "a conductive plan with no conductive component is kernels."
                "pml_curl_step's configuration; build that plan instead")
        chunks = tuple(conductive_chunks)
        if len(chunks) != 3:
            raise ValueError("conductive_chunks names one volume (or None) per target")
        self.cmask = tuple(1 if (flag and volume is not None) else 0
                           for flag, volume in zip(self.cond, chunks))
        self.block = int(block)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        # Placeholders for the lossless components, resolved HERE so the launch
        # path never branches (see the class docstring).
        self._history = tuple(CupyPointer(a if a is not None else b)
                              for a, b in zip(history, targets))
        self._condfac = tuple(CupyPointer(a if a is not None else b)
                              for a, b in zip(condfac, targets))
        self._condinv = tuple(CupyPointer(a if a is not None else b)
                              for a, b in zip(condinv, targets))
        self._chunks = tuple(
            CupyPointer(_chunk_volume(volume, self.shape, target) if mask else target)
            for mask, volume, target in zip(self.cmask, chunks, targets))
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's source-mutation leg,
        # which compiles a deliberately broken copy of this kernel. Dropping it is
        # not a slowdown, it is a DISARMING — every mutation leg would then launch
        # the shipped kernel and report the defect as uncaught (§13.4's family).
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as every plan here.

        ``guard`` is not for callers: it exists so the gate can MEASURE the
        contraction guard's effect (identical with, non-identical without) rather
        than assert it.
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = (self._kernel if self._kernel is not None
                  else conductive_pml_curl_step)
        kernel[self._grid](
            *self._targets, *self._aux, *self._history, *self._condfac,
            *self._condinv, *self._chunks, *self._sources, *self._coefficients,
            self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
            CMASK0=self.cmask[0], CMASK1=self.cmask[1], CMASK2=self.cmask[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"ConductivePmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, cond={self.cond}, cmask={self.cmask}, "
                f"block={self.block}, chunk_bounds={self.chunk_bounds.tolist()})")


def _chunk_volume(volume: Any, shape: Tuple[int, ...], like: Any) -> Any:
    """A conductive-chunk volume as the ``uint8`` the kernel reads, on ``like``'s module.

    Refused by name when it does not lie on the plan's shape: a volume of another
    grid would key the third test on the wrong cells.
    """
    import numpy  # noqa: PLC0415

    if tuple(int(n) for n in getattr(volume, "shape", ())) != tuple(shape):
        raise ValueError(
            f"a conductive-chunk volume of shape {getattr(volume, 'shape', None)} does "
            f"not lie on the plan's shape {tuple(shape)}")
    if type(like).__module__.split(".")[0] == "cupy":  # pragma: no cover - device path
        import cupy  # noqa: PLC0415

        return cupy.ascontiguousarray(cupy.asarray(volume).astype(cupy.uint8))
    from ..backends import to_numpy  # noqa: PLC0415

    return numpy.ascontiguousarray(numpy.asarray(to_numpy(volume)).astype(numpy.uint8))


def conductive_chunk_volumes(fields: Any, sub_step: str, chunk_layout: Any,
                             pml: Any = None) -> Tuple[Any, Any, Any]:
    """Per target of ``sub_step``: the cells whose chunk keeps its conductivity, or None.

    ``chunk_map.ChunkLayout.conductive_chunks`` / ``conductive_mask`` of the run's
    division (``chunk_layout``; ``None`` is the one-rank default of the grid and
    ``pml``, as for the chunk record), read on each component's own Yee lattice from
    its conductivity volume (``Fields.conductivity_for``). ``None`` for a lossless
    target and for one whose every chunk keeps its conductivity -- the array path's
    ``conductive=None``, where the kernel reads no volume.
    """
    from ..chunk_map import ChunkLayout  # noqa: PLC0415
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    layout = (chunk_layout if chunk_layout is not None
              else ChunkLayout.one_rank(fields.grid, pml))
    out: List[Any] = []
    for flag, target in zip(conductive_targets(fields, sub_step),
                            CONDUCTIVE_SUB_STEPS[sub_step]):
        if not flag:
            out.append(None)
            continue
        sigma = fields.conductivity_for(target)
        iyee = IYEE_SHIFTS[target]
        if all(layout.conductive_chunks(sigma, iyee)):
            out.append(None)
        else:
            out.append(layout.conductive_mask(sigma, iyee))
    return (out[0], out[1], out[2])


def plan_conductive_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             chunk_layout: Any = None
                             ) -> Optional[ConductivePmlCurlPlan]:
    """Build a conductive curl plan from the engine's own objects, or None when refused.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default;
    the plan binds its record (``launch.chunk_bounds_for``) and, per conductive
    target, its conductive-chunk volume (:func:`conductive_chunk_volumes`).
    """
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not conductive_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    from .launch import SUB_STEPS, chunk_bounds_for  # noqa: PLC0415

    kernels = _kernel_module()
    spec = SUB_STEPS[sub_step]
    targets = CONDUCTIVE_SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = _boundary_kinds(grid, pml)
    flags = conductive_targets(fields, sub_step)
    return ConductivePmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        flags,
        kernels.DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in targets],
        [getattr(fields, "fu_" + n) for n in targets],
        [getattr(fields, "f_cond_" + n) for n in targets],
        [fields.condfac_for(n) for n in targets],
        [fields.condinv_for(n) for n in targets],
        [getattr(fields, n) for n in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        conductive_chunks=conductive_chunk_volumes(fields, sub_step, chunk_layout, pml),
    )


def plan_conductive_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                flat: Dict[str, Any], codes, cond, dtdx: float,
                                block: Optional[int] = None,
                                kernel: Any = None, *,
                                chunk_bounds: Any) -> ConductivePmlCurlPlan:
    """Build a conductive curl plan from bare device arrays — the gate's route.

    ``arrays`` is keyed by component name (``Bx``/``fu_Bx``/``f_cond_Bx``/
    ``condfac_Bx``/``condinv_Bx``/``Ex``/...), ``flat`` by ``kms_x``/``sinv_x``/...
    on the sub-lattice THE CALLER already selected, ``codes`` is the per-axis 0/1
    periodic/metallic triple and ``cond`` the per-target conductive triple. No
    coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.

    ``chunk_bounds`` is REQUIRED and has no default (``launch.plan_from_arrays``'s
    rule): the caller hands over the record of the division its reference steps
    with. A conductive target whose chunks do not all keep the conductivity carries
    its conductive-chunk volume as ``conductive_chunks_<target>`` (bool or integer,
    the target's shape); without one every chunk keeps it.

    A component the caller marks lossless may omit its three conductive arrays
    entirely; the plan binds the placeholder.
    """
    from .launch import SUB_STEPS  # noqa: PLC0415

    kernels = _kernel_module()
    spec = SUB_STEPS[sub_step]
    targets = CONDUCTIVE_SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return ConductivePmlCurlPlan(
        sub_step, shape, dtdx, codes, cond,
        kernels.DEFAULT_BLOCK if block is None else block,
        [arrays[n] for n in targets],
        [arrays["fu_" + n] for n in targets],
        [arrays.get("f_cond_" + n) for n in targets],
        [arrays.get("condfac_" + n) for n in targets],
        [arrays.get("condinv_" + n) for n in targets],
        [arrays[n] for n in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel,
        chunk_bounds=chunk_bounds,
        conductive_chunks=tuple(arrays.get("conductive_chunks_" + n) for n in targets),
    )


def explain_conductive(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """The conductive coverage verdict with its reasons — for reports and refusals."""
    return conductive_pml_curl_coverage(fields, pml, sub_step)


def conductive_sub_steps(fields: Any, pml: Any) -> Tuple[str, ...]:
    """Which sub-steps of this run this kernel covers, in the driver's order.

    The answer is a SUBSET, because a run may be conductive on D and lossless on B — which is
    exactly the ``2d_cond_pml`` benchmark case (``mp.Medium(epsilon=2.0,
    D_conductivity=0.4)``, no absorber, no ``B_conductivity``). Its ``step_B``
    belongs to :func:`kernels.pml_curl_step` and its ``step_D`` to this one.
    """
    return tuple(name for name in ("step_B", "step_D")
                 if conductive_pml_curl_coverage(fields, pml, name).covered)
