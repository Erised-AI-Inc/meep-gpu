"""Conductivity + PML Triton kernel: MEEP's four-case ``step_curl``.

ONE kernel, one predicate, one plan builder. All three are ADDITIONS — nothing in
:mod:`kernels`, :mod:`coverage`, :mod:`launch` or the package ``__init__`` is
edited, and nothing here is wired into ``plan_step``. The integration this file
waits for is listed at the bottom of this docstring.

* :func:`conductive_pml_curl_step` — ``stepping.step_B`` / ``stepping.step_D`` on
  a grid that carries BOTH an active split-field PML and a material conductivity
  on one or more of that sub-step's three targets. That is the path
  ``stepping._apply_conductive_pml_update`` (stepping.py:1954-2062) serves, and it
  is the engine's most expensive curl: measured on the real ``2d_cond_pml``
  layout, 640x640x1, NumPy, per curl term — plain split-field PML 0.619 ms
  (661.9 Mcell/s) against the four-case conductive form's 3.002 ms
  (136.4 Mcell/s), **4.85x**. The array path makes roughly eighty full-volume
  passes per curl term to get there; this kernel reads five volumes and writes
  three.

WHY A BRANCH-PER-CELL KERNEL IS EVEN POSSIBLE HERE, and it is measured rather
than argued. The array path computes ALL FOUR of MEEP's subchunk cases over the
whole volume and selects with ``xp.copyto(..., where=)``. A Triton program
evaluates ONE branch per cell. Those two are the same bits only because the four
branch expressions are mutually independent: ``first_only`` reads ``f_cond_new``
and ``f_cond_previous`` (both defined before it), ``direct`` reads only
``field_previous``, and the discarded branches are restored to their prior values
(stepping.py:2060-2062). Nothing a discarded branch computes ever feeds a
retained one — verified in source order at stepping.py:2011-2062, where
``first_only`` is copied off ``field_previous`` BEFORE ``direct`` mutates
``field_previous`` in place. The recon then measured it: a scalar float32
per-cell loop evaluating only the selected branch is bytewise identical to the
array path, 18/18 over three shapes x {0.5, 0.35} x all three curl targets, with
a flattened-grouping control at 0/18.

THE FOUR CASES, transcribed from ``stepping._apply_conductive_pml_update``
(stepping.py:1954-2062), reached from ``_apply_curl`` (:479-505). Notation:
``f`` = target, ``u`` = ``fu_<target>``, ``c`` = ``f_cond_<target>``, ``cu`` =
the curl (already carrying dtdx), ``cf``/``ci`` = ``condfac``/``condinv``,
``(km1, si1)`` = ``kms``/``sinv`` on the ``dsig`` axis, ``(km2, si2)`` on the
``dsigu`` axis::

    dsig  = (km1 != 1.0) | (si1 != 1.0)          # exact float comparison
    dsigu = (km2 != 1.0) | (si2 != 1.0)

    A  dsig & dsigu   c <- ((c*cf) - cu) * ci
                      u <- (((u*km1) + c_new) - c) * si1
                      f <- (((f*km2) + u_new) - u) * si2
    B  dsigu only     u <- ((u*cf) - cu) * ci
                      f <- (((f*km2) + u_new) - u) * si2      c UNCHANGED
    C  dsig only      c <- ((c*cf) - cu) * ci
                      f <- (((f*km1) + c_new) - c) * si1      u UNCHANGED
    D  neither        f <- ((f*cf) - cu) * ci                 u AND c UNCHANGED

``condfac = 1 - sigma*dt/2`` and ``condinv = 1/(1 + sigma*dt/2)`` (fields.py:821-822,
MEEP structure.cpp:693-706), both FULL VOLUMES of ``grid.shape``. "UNCHANGED" is a
requirement and not a don't-care — the array path restores ``u`` and ``c`` in
their inactive regions so a diagnostic read cannot mistake unused arithmetic for
state; in the kernel that is simply "do not store", which is stronger and
identical.

DO NOT FLATTEN THE PARENTHESES ABOVE, and do not let a multiply contract into an
FMA. The recon's unguarded control flattens only ``((f*km2) + u_new) - u`` into
``(f*km2) + (u_new - u)`` and goes 0/18 on NumPy alone. ``ENABLE_FP_FUSION`` is
spelled once for the whole package, in :mod:`kernels`, and this file's single
launch site passes it.

THE PARTITION IS EXACT, NOT A TOLERANCE. ``dsig_active`` is a bare ``!= 1.0``
(stepping.py:2008-2011). Measured on the real ``2d_cond_pml`` tables: ``kms_x``
has 561/640 entries EXACTLY 1.0 and 561 entries within 1e-7 of 1.0 — the two
counts agree, so there is no near-one band a kernel could partition differently,
and deriving both predicates in-kernel from the loaded coefficients reproduces
the array path's masks exactly. A kernel using ``>=`` or a tolerance is therefore
a defect the REAL tables cannot expose, which is why the gate drives that
mutation with a synthetic table carrying a near-one entry.

PER-COMPONENT CONDUCTIVITY IS THE DANGEROUS CASE AND IT IS A CONSTEXPR.
``_apply_curl`` reads ``fields.condfac_for(term.target)`` PER COMPONENT
(stepping.py:460, :479-508) and ``driver._compose_conductivity_side``
(driver.py:1906-1928) may leave a component out — a ``D_conductivity_diag`` with
a zero entry, or a material sigma on one component only. A component with no
sigma takes the plain path while its neighbours are lossy. ``COND0``/``COND1``/
``COND2`` carry that per component: where a component is lossless the conductive
branch is compiled out entirely and the kernel emits :func:`kernels.pml_curl_step`'s
own two-line recurrence verbatim, so a partially conductive launch is identical
on its lossless components BY CONSTRUCTION rather than by argument. Both the
predicate and the constexpr read :func:`conductive_targets`, so the check and the
compile-time choice cannot disagree — the discipline :func:`coverage.sigma_is_volume`
exists for.

WHAT ELSE IS TRANSCRIBED, and from where. The curl, the ghost rule and the
ownership mask are UNCHANGED from the plain path: conductivity enters only after
the curl is formed. So this kernel's first sixty lines are a verbatim copy of
:func:`kernels.pml_curl_step`'s body —

* term table          ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` (stepping.py:213-223)
* ghost rule          ``stepping._shift_up`` (:1723) / ``_shift_down`` (:1787)
* curl grouping       ``stepping._curl_from_operands`` (:1601)
* ownership mask      ``stepping._mask_non_owned_cells`` (:1865)
* coefficient pairing ``stepping._curl_coefficients`` (:2417)

— which is a SECOND TRANSCRIPTION of the three things most likely to drift, with
no mechanical link to the first. That duplication is forced (this round may not
edit :mod:`kernels`) and it is the main reason the integration note below
recommends folding this kernel INTO ``pml_curl_step`` as three extra constexpr
flags rather than keeping a parallel body: with all three flags zero Triton
compiles the conductive branch away and the existing path stays byte-for-byte
what it is today. Until then ``test_conductivity`` diffs the two curl bodies
textually.

WHAT INTEGRATION WILL NEED, when the shared files are free:

* ``coverage.py``: clause 8 (coverage.py:188-196) must move OUT of the shared
  ``_grid_reasons`` and become per-sub-step. As written it disqualifies EVERY
  sub-step on any conductive grid, which is why the ``2d_cond_pml`` benchmark
  case is refused wholesale. Splitting it admits ``step_B``, ``update_H``,
  ``update_E`` and ``update_P`` on that case with NO new kernel at all, because
  its builder sets ``mp.Medium(epsilon=2.0, D_conductivity=0.4)`` and nothing
  magnetic: only the D curl is conductive there. That is the single
  highest-yield edit in this tranche and it is in a file another workflow owns.
* ``launch.py``/``plan_step``: ``step_B`` and ``step_D`` have ONE slot each and
  ``plan_pml_curl`` claims it. The composer must ask :func:`conductive_pml_curl_coverage`
  FIRST and fall through to the plain builder, and must never build both.
* ``__init__.py``: export the kernel, the predicate and the two plan builders.
* ``fingerprints.json``: a new ``kernels`` entry, this module's ``host_sha256``
  and this gate's verdict, or the version guard certifies a kernel it never saw.
* Dispatch stays disabled regardless. The subnormal question that blocks the
  shipped kernels binds harder here: every one of the four cases multiplies by
  ``condfac*condinv < 1`` every step ON TOP of the PML decay. Measured on the
  real ``2d_cond_pml`` tables, drive-free from 1.0 — ``condfac`` 0.9975,
  ``condinv`` 0.99750626, product 0.9950125, so conductivity alone reaches
  subnormal in 17468 steps, but the DEEPEST absorber plane's combined per-step
  factor is 0.72131777 (D sub-lattice) / 0.7272973 (B), i.e. subnormal from 1.0
  in **268 / 275 steps**. This kernel manufactures the subnormal inputs faster
  than any other in the set.

WHAT THE GATE MEASURED (``parity/meep_gpu/gate_triton_conductivity.py``, one
clear RTX A6000 on the GPU host, ``CUDA_VISIBLE_DEVICES=6``, Triton 3.1.0 / CuPy
13.5.1 / NumPy 2.2.6, artifact
``parity/meep_gpu/results/triton_conductivity_2026-08-10/``):

* the gate's reference against ``stepping._apply_conductive_pml_update`` itself,
  bytewise on NumPy — **48/48**. That leg runs at the merge bar, so the chain
  reaches the array path rather than stopping at "the kernel reproduces whatever
  was written twice";
* single-launch sweep — **240/240 bit-identical GUARDED, 0/240 UNGUARDED**, 32
  skipped ({synthetic, real} coefficient source x 4 shapes x 4 boundary sets x
  dtdx {0.5, **0.35**} x {step_B, step_D} x conductivity pattern {(1,1,1),
  (1,0,1)}, comparing the target, its ``fu`` AND its ``f_cond`` as bytes). All
  four branches were exercised, and the per-case branch census is in the
  artifact rather than assumed;
* **11 mutations, 11 caught** — 8 injected into the kernel source (flattened
  grouping, swapped condfac/condinv, case B collapsed into case A, each store
  mask dropped, swapped dsig/dsigu, regrouped stencil, a tolerance in place of
  ``!= 1.0``) and 3 host-side (the integer table on the B side, a wall told as a
  wrap, and the plan telling the kernel a lossless component is lossy).

**STEP BUDGET: 256 STEPS** for an all-conductive launch — 8/8 runs identical at
every one of 256 consecutive sub-steps, and 320/320 in the drive-free decay leg
that manufactures subnormals. A MIXED launch is certified for **159** steps and
no further. That shorter budget is NOT this kernel's: every mixed divergence is
1-2 SUBNORMAL floats (9.1e-39 to 1.2e-38) in the LOSSLESS component's ``fu``,
i.e. in the plain recurrence ``COND == 0`` compiles, and the SHIPPED
``kernels.pml_curl_step`` run all-lossless on the same tables and seeds first
differs at the SAME step with the same magnitude (177 vs 177, 170 vs 170) —
plan §16's long-horizon plateau, reproduced. Nothing beyond those numbers is
claimed; §16 bounds any whole-run claim independently.
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

    #: Boundary codes, identical to :mod:`kernels`' own. A plan built here and a
    #: plan built there index the same table; the test pins the two equal.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    @triton.jit
    def _conductive_component(f_ptr, u_ptr, c_ptr, cf_ptr, ci_ptr,
                              idx, live, curl, km1, si1, km2, si2,
                              COND: tl.constexpr):
        """One component's recurrence: the four conductive cases, or the plain one.

        Written once and called three times rather than triplicated, because the
        thing most likely to go wrong here is a parenthesis and three copies of a
        parenthesis is three chances to lose one.

        ``COND == 0`` emits :func:`kernels.pml_curl_step`'s two-line recurrence
        verbatim and nothing else compiles — no ``f_cond`` load, no ``condfac``
        load, no predicate. That is what makes a mixed launch (one lossless
        component beside two lossy ones) identical on the lossless component by
        construction.

        ``COND == 1`` evaluates all five branch expressions and selects with
        ``tl.where``. That is bit-exact because the four cases are mutually
        independent — see the module docstring — so a discarded branch cannot
        contaminate a retained one, and ``tl.where`` is a selection, not an
        arithmetic combination.

        THE STORE MASKS ARE THE 'UNCHANGED' COLUMN OF THE CASE TABLE. ``u`` is
        written only where ``dsigu``, ``c`` only where ``dsig``; ``f`` always.
        ``live`` is ANDed into both, and it is load-bearing rather than tidy: the
        coefficient loads use ``other=0.0``, so past the end of the array both
        predicates evaluate ``0.0 != 1.0`` = True. Drop ``live`` from a store mask
        and the kernel writes past the allocation.
        """
        f = tl.load(f_ptr + idx, mask=live, other=0.0)
        u = tl.load(u_ptr + idx, mask=live, other=0.0)
        if COND == 0:
            # stepping._apply_pml_update (:1905), verbatim from kernels.pml_curl_step.
            n = ((u * km1) - curl) * si1
            v = (((f * km2) + n) - u) * si2
            tl.store(u_ptr + idx, n, mask=live)
            tl.store(f_ptr + idx, v, mask=live)
        else:
            c = tl.load(c_ptr + idx, mask=live, other=0.0)
            cf = tl.load(cf_ptr + idx, mask=live, other=0.0)
            ci = tl.load(ci_ptr + idx, mask=live, other=0.0)

            # stepping.py:2008-2011 — an EXACT float comparison, not a tolerance.
            dsig = (km1 != 1.0) | (si1 != 1.0)
            dsigu = (km2 != 1.0) | (si2 != 1.0)

            # The five branch expressions. DO NOT FLATTEN THESE PARENTHESES.
            c_new = ((c * cf) - curl) * ci                     # cases A and C
            u_cond = ((u * cf) - curl) * ci                    # case B
            u_split = (((u * km1) + c_new) - c) * si1          # case A
            u_new = tl.where(dsig, u_split, u_cond)
            f_split = (((f * km2) + u_new) - u) * si2          # cases A and B
            f_first = (((f * km1) + c_new) - c) * si1          # case C
            f_direct = ((f * cf) - curl) * ci                  # case D
            v = tl.where(dsigu, f_split, tl.where(dsig, f_first, f_direct))

            tl.store(f_ptr + idx, v, mask=live)
            tl.store(u_ptr + idx, u_new, mask=live & dsigu)
            tl.store(c_ptr + idx, c_new, mask=live & dsig)

    @triton.jit
    def conductive_pml_curl_step(
        f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
        u0, u1, u2,                       # auxiliaries: fu_B*  or  fu_D*
        c0, c1, c2,                       # f_cond_*  (a placeholder where COND == 0)
        cf0, cf1, cf2,                    # condfac   (a placeholder where COND == 0)
        ci0, ci1, ci2,                    # condinv   (a placeholder where COND == 0)
        g0, g1, g2,                       # sources: Ex,Ey,Ez  or  Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, one sub-lattice
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One curl sub-step with conductivity, PML recurrence included.

        The curl half below the header is byte-copied from
        :func:`kernels.pml_curl_step`: conductivity enters ``_apply_curl`` only
        AFTER the curl is formed (stepping.py:479-505), so the ghost rule, the
        stencil grouping and the ownership mask are the plain path's, unchanged.
        Only the recurrence differs, and it lives in :func:`_conductive_component`.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) --------
        # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall, which
        # `tl.load`'s `other=` delivers without dereferencing anything.
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

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these ------
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- ownership mask (stepping._mask_non_owned_cells) -------------------
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

        # --- the recurrence, per component -------------------------------------
        # dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on
        # both sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        _conductive_component(f0, u0, c0, cf0, ci0, idx, live, curl0,
                              km_y, si_y, km_z, si_z, COND0)
        _conductive_component(f1, u1, c1, cf1, ci1, idx, live, curl1,
                              km_z, si_z, km_x, si_x, COND1)
        _conductive_component(f2, u2, c2, cf2, ci2, idx, live, curl2,
                              km_x, si_x, km_y, si_y, COND2)


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
    the two hold different bindings (this one carries nine more pointers) and a
    subclass that inherited ``run`` would launch the plain kernel on a conductive
    grid — a wrong answer that never raises. Built two ways —
    :func:`plan_conductive_pml_curl` from the engine's own objects and
    :func:`plan_conductive_from_arrays` from bare device arrays for the gate — and
    launched ONE way, so the bytes the gate certifies are the bytes the engine
    would launch.

    A LOSSLESS COMPONENT BINDS ITS OWN TARGET AS THE PLACEHOLDER for the three
    conductive slots, never a null. The loads sit behind a ``tl.constexpr`` branch
    and are compiled away, but a pointer argument still has to type, and passing
    ``None`` makes the launcher's failure a ``TypeError`` far from its cause. The
    placeholder is never read and never written: ``COND == 0`` emits no conductive
    load and no conductive store.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc", "cond",
                 "block", "_targets", "_aux", "_history", "_condfac", "_condinv",
                 "_sources", "_coefficients", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, cond, block: int,
                 targets, auxiliaries, history, condfac, condinv, sources,
                 coefficients, kernel=None) -> None:
        from .launch import SUB_STEPS, CupyPointer, _flat  # noqa: PLC0415

        if sub_step not in CONDUCTIVE_SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                             f"got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
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
            *self._condinv, *self._sources, *self._coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"ConductivePmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, cond={self.cond}, block={self.block})")


def plan_conductive_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None
                             ) -> Optional[ConductivePmlCurlPlan]:
    """Build a conductive curl plan from the engine's own objects, or None when refused.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not conductive_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    from .launch import SUB_STEPS  # noqa: PLC0415

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
    )


def plan_conductive_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                flat: Dict[str, Any], codes, cond, dtdx: float,
                                block: Optional[int] = None,
                                kernel: Any = None) -> ConductivePmlCurlPlan:
    """Build a conductive curl plan from bare device arrays — the gate's route.

    ``arrays`` is keyed by component name (``Bx``/``fu_Bx``/``f_cond_Bx``/
    ``condfac_Bx``/``condinv_Bx``/``Ex``/...), ``flat`` by ``kms_x``/``sinv_x``/...
    on the sub-lattice THE CALLER already selected, ``codes`` is the per-axis 0/1
    periodic/metallic triple and ``cond`` the per-target conductive triple. No
    coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.

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
    )


def explain_conductive(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """The conductive coverage verdict with its reasons — for reports and refusals."""
    return conductive_pml_curl_coverage(fields, pml, sub_step)


def conductive_sub_steps(fields: Any, pml: Any) -> Tuple[str, ...]:
    """Which sub-steps of this run this kernel covers, in the driver's order.

    The shape ``plan_step`` will read when integration lands: the answer is a
    SUBSET, because a run may be conductive on D and lossless on B — which is
    exactly the ``2d_cond_pml`` benchmark case (``mp.Medium(epsilon=2.0,
    D_conductivity=0.4)``, no absorber, no ``B_conductivity``). Its ``step_B``
    belongs to :func:`kernels.pml_curl_step` and its ``step_D`` to this one.
    """
    return tuple(name for name in ("step_B", "step_D")
                 if conductive_pml_curl_coverage(fields, pml, name).covered)
