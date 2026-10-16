"""The CYLINDRICAL (Dcyl) curl sub-steps at m = 0: a radial scan kernel and a curl kernel.

WHAT IT REPLACES. ``stepping.step_B`` and ``stepping.step_D`` on a Dcyl grid with
``grid.m == 0``, real storage and an active split-field PML -- the whole sub-step,
including the radial sum MEEP calls ``f_rderiv_int``. Two ``@triton.jit`` bodies serve
both sub-steps: :func:`cyl_radial_difference` (the column-serial scan) and
:func:`cyl_pml_curl_step` (the three curls, the PML forms and the m = 0 axis rules).
The complex family (:mod:`.cylindrical_complex`) and the four cylindrical fused pairs
launch the same scan kernel; it has one home, here.

THE FORMS ARE THE ARRAY PATH'S, AND THE ARRAY PATH'S ARE MEEP'S (1.33.0, single
precision, no contraction). Every one is read from :mod:`meep_gpu.stepping` of this
tree and transcribed with its grouping:

* THE RADIAL SUM (``stepping.cylindrical_rderiv_difference``, step_db.cpp:99-119). The
  Z component's update needs ``(1/r)·d(r·Fp)/dr``, and MEEP builds it as a running sum
  in EVERY STRUCTURE CHUNK over that chunk's own rows::

      inc[g] = rinv[g-1] * ((Fp[g]*w[g]) - (Fp[g-1]*w[g-1]))       g >= 1
      F_s[s] = +0,  F_s[g] = F_s[g-1] + inc[g]                       g > s

  with ``w[g] = realnum(g + ir0)`` and ``rinv[g-1] = realnum(1.0 / ((g + ir0) -
  0.5))`` (``stepping._cylindrical_rderiv_rows``, bound as two device rows, so the
  kernel performs no division), every operation rounded once and the sum sequential
  along r. A target cell reads the sum of the chunk that owns it:
  ``F_s[j+1] - F_s[j]`` for Bz (forward, half-integer rows ``j``), ``F_s[g-1] -
  F_s[g]`` for Dz (MEEP's negated stride, integer rows ``g``). The cut row between two
  chunks therefore holds two values, which is why the kernel writes the DIFFERENCE and
  never a prefix array. ``s`` per target row is
  ``stepping._cylindrical_scan_starts`` (the run's division), bound as an int32 line,
  and its distinct values as the segment list. A division whose r cuts differ along z
  has no such line and is refused to the array path by name (DESIGN D8).

  The scan is COLUMN-SERIAL: one lane per (phi, z) column -- per stored part, on
  complex storage, because a complex64 volume viewed as float32 words is one column
  per part -- walks the rows in order. A parallel scan (a tree reduction, or a library
  cumsum) associates the sum differently and is a different float32 word on most
  rows; nothing here may replace the loop with one. The B side's Ep carries MEEP's
  stored WALL ROW (``nr + 1`` rows, the wall's Ep held at ``+0``): the kernel reads row
  ``nr`` as a masked ``+0`` load, so the last row's forward difference is
  ``0 - F[nr-1]`` -- the local value MEEP reads, not minus a sum over a zero ghost.

* THE CURLS (``stepping.step_B`` / ``step_D``). Bz's and Dz's whole curl is ``dtdx``
  times the radial difference: their phi partner is the derivative MEEP drops on a
  Dcyl grid (step_db.cpp:86-92). The other two components take
  ``stepping._curl_from_operands`` through :func:`.forms.curl_stencil` with the
  record's partner-presence word, exactly as :func:`.kernels.pml_curl_step` does:
  phi is invariant, so Bx/Dx take the one-partner form ``(-dtdx) * (g2s - g2)`` and
  By/Dy MEEP's left-to-right ``dtdx * (((g1s - g1) + g2) - g2s)``.

* THE PML FORMS (``stepping._apply_curl_regions``) through :func:`.forms.curl_form`,
  keyed on the chunk record ``cb``: the split-field recurrence on the cells whose
  chunk carries the component's dsigu layer, the dsig-only form elsewhere, and ``fu``
  read and written on the recurrence's cells only (the auxiliary contract). The B
  targets read the half-integer lines, the D targets the integer ones.

* THE OWNERSHIP MASK (``stepping._mask_non_owned_cells``): row 0 of every target whose
  r Yee shift is 0 (MEEP's ``little_owned_corner0`` excludes the axis, which the per-m
  rules own), and cell 0 of the metallic z wall for every target whose z shift is 0.

* THE m = 0 AXIS RULES (``stepping._cylindrical_axis_zero_B`` / ``_D``), applied to
  the updated words: ``Br[0] = 0``; ``Dz[0] += realnum(double(4*Courant) *
  double(Hp[0]))`` -- a DOUBLE product rounded once (step_db.cpp:324), with the
  double Courant read from a bound ``float64`` vector, because a scalar argument rides
  the fp32 kernel ABI and ``4*Courant`` is not a float32 word at most Courant numbers
  -- then ``Dp[0] = 0``. The add stays a post-add on the updated field, never folded
  into the curl, as the array path writes it.

THE r_to_minus_r NEAR GHOST IS NOT IMPLEMENTED, and cannot be observed. The only
targets taking a shift-down along r are Dy (partner Hz) and Dz (whose curl is the
radial difference and reads no ghost); Dy's r Yee shift is 0, so the ownership mask
drops its row-0 curl, the only row the near ghost would reach. The kernel serves an
exact ``+0`` there (a masked load's ``other=``), and no byte gate can certify that
choice: a mutation that puts a faithful ghost back must come out NOT caught, and if it
ever comes out caught the ownership mask has broken, not the ghost.

WHAT IS NOT COVERED: m != 0 (complex storage, the i*m/r pass, the per-|m| axis rules;
:mod:`.cylindrical_complex`), a fold, a conductivity, BFAST, beta, a Bloch phase and a
nonlinearity. :func:`cylindrical_curl_coverage` names each refusal.

The CONSTITUTIVE sub-steps need no cylindrical kernel: ``stepping.update_H`` and
``update_E`` carry no cylindrical branch, so :func:`plan_cylindrical_constitutive`
builds the root :class:`.launch.ConstitutivePlan` (with MEEP's start of H) behind its
own predicate, which keeps the Cartesian predicates' blanket Dcyl refusal intact.

``launch.plan_step`` selects these plans for the strict predicates below; the driver
order is unchanged, and the fast path (``fastpath``) reaches them only through
``plan_step``, never by importing this module. Every launch passes :data:`.kernels.ENABLE_FP_FUSION`.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from .coverage import Coverage

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# The configuration this kernel is compiled for, spelled as data
# ---------------------------------------------------------------------------

#: The ghost rule per axis on a Dcyl grid, as ``stepping._boundary_kinds`` resolves
#: it: r = 0 is the cylindrical axis, phi is the one-cell invariant axis (which
#: resolves to PERIODIC, and whose curl partners the record's presence word marks
#: absent), z is walled.
#: CONSTANT for this kernel, so it is spelled here and compiled in rather than
#: passed — a Dcyl grid that resolves to anything else is refused by the predicate,
#: not stepped with different constexprs.
CYLINDRICAL_BOUNDARY_KINDS: Tuple[str, str, str] = ("axis", "periodic", "metallic")

#: ``grid.is_axis`` on the three axes. Refused unless it is exactly this.
CYLINDRICAL_AXIS_FLAGS: Tuple[bool, bool, bool] = (True, False, False)

#: The one azimuthal order this kernel carries. See the module docstring.
COVERED_M: int = 0

#: Targets, auxiliaries and sources per sub-step, in target order — the same table
#: ``launch.SUB_STEPS`` holds, restated here rather than imported so this module
#: does not depend on a file another track owns. ``suffix`` selects the Yee
#: sub-lattice the curl's kms/sinv come from: HALF-INTEGER for B, integer for D
#: (``stepping._curl_coefficients``); swapped, it is a silent half-cell error in
#: the absorber profile, which is why the gate carries ``coeff_lattice_swap``.
SUB_STEPS: Dict[str, Dict[str, Any]] = {
    "step_B": {
        "targets": ("Bx", "By", "Bz"),
        "sources": ("Ex", "Ey", "Ez"),
        "backward": 0,
        "suffix": "_h",
        # stepping.step_B: the radial sum runs over Ep = Ey and its stored zero
        # wall row (nr + 1 rows), at ir0 = 0.0; Bz reads the forward difference.
        "prefix_component": "Ey",
        "prefix_ir0": 0.0,
        "extend_wall_row": True,
    },
    "step_D": {
        "targets": ("Dx", "Dy", "Dz"),
        "sources": ("Hx", "Hy", "Hz"),
        "backward": 1,
        "suffix": "",
        # stepping.step_D: the radial sum runs over Hp = Hy, no wall row (the
        # backward difference reads rows g - 1 and g), at ir0 = 0.5 = half of Hp's
        # r-shift; Dz reads the backward difference.
        "prefix_component": "Hy",
        "prefix_ir0": 0.5,
        "extend_wall_row": False,
    },
}

#: The six curl targets and their auxiliaries, plus every source volume the two
#: sub-steps read. Enumerated so the predicate can check allocation and layout of
#: exactly what the kernel dereferences.
CURL_TARGETS: Tuple[str, ...] = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
CURL_SOURCES: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: Which axis pair the split-field recurrence reads for target 0, 1 and 2 —
#: vec.hpp's cycle_direction, the same triple on both sides.
DSIG_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: The axis letters the kernel's mask spells, per axis index. phi never appears:
#: it is periodic and masks nothing.
MASK_AXIS_LETTERS: Dict[int, str] = {0: "r", 2: "z"}


def ownership_mask_axes(sub_step: str) -> Tuple[Tuple[int, ...], ...]:
    """Which axes each target's curl is zeroed at cell 0 on — DERIVED, not restated.

    ``stepping._mask_non_owned_cells`` drops cell 0 of every axis whose
    boundary is ``mirror``/``metallic``/``axis`` and on which the target's Yee shift
    is 0. Both inputs are the engine's own — ``fields.IYEE_SHIFTS`` and
    :data:`CYLINDRICAL_BOUNDARY_KINDS` — so this function computes what the kernel
    hard-codes, and ``test_triton_cylindrical`` compares the two. Hard-coding it in
    the kernel is right (it is a compile-time fact for this configuration); leaving
    it uncheckable would not be.

    The far-face clause of that function (``_stored_past_owned``) is not
    reproduced: it fires only on a FOLDED periodic axis, which the predicate refuses.
    """
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    masking = {axis for axis, kind in enumerate(CYLINDRICAL_BOUNDARY_KINDS)
               if kind in ("mirror", "metallic", "axis")}
    return tuple(
        tuple(axis for axis in range(3)
              if IYEE_SHIFTS[target][axis] == 0 and axis in masking)
        for target in SUB_STEPS[sub_step]["targets"])

#: Elements per program. The same value and the same reason as
#: ``kernels.DEFAULT_BLOCK`` — not autotuned, because the kernel writes its own
#: inputs in place and ``triton.autotune`` re-runs a kernel to time it (measured on
#: the Cartesian curl: every float wrong, max_abs 1.4e4, no error raised). The scan
#: kernel's BLOCK counts columns, the curl kernel's cells; both take this value.
#:
#: It is RESTATED rather than imported because ``kernels.py`` imports Triton at
#: module scope and a plan must be buildable on a Triton-less host. The two values
#: are pinned equal by ``test_triton_cylindrical``, which reads the constant off
#: that file's SOURCE.
DEFAULT_BLOCK: int = 256




#: Word index of each double constant in the bound ``float64`` vector the m = 0 D side
#: reads (:func:`courant_constants`): ``4*Courant`` for the on-axis Dz add. The complex
#: family's |m| = 1 axis increments read ``+Courant`` (D) and ``-Courant`` (B) from the
#: same vector.
COURANT_WORDS: Dict[str, int] = {"four": 0, "plus": 1, "minus": 2}


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------

if triton is not None:

    # The forms and the record layout. Imported only when Triton is: both modules
    # import it at module scope, and this module's host route (the predicates, the
    # plan builders to None) must answer without it.
    from .forms import axis_present, curl_form, curl_stencil, record_in
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE

    #: :data:`COURANT_WORDS`' ``four`` word, as a constexpr a kernel body may read.
    COURANT_FOUR = tl.constexpr(0)

    @triton.jit
    def cyl_radial_difference(
        fp,                           # Fp's words: (rows_stored, ny, ncol/ny) (in)
        w, rinv,                      # float32 rows: w[g] (rows_src), rinv[g-1] (rows_src-1)
        starts, segs,                 # int32: each target row's chunk start; the distinct starts
        out,                          # the difference, (n_target, ny, ncol/ny) words (out)
        rows_stored, rows_src, n_target, ncol, n_segs,
        BACKWARD: tl.constexpr,       # 0 = B (forward, Bz), 1 = D (backward, Dz)
        BLOCK: tl.constexpr,
    ):
        """MEEP's per-chunk radial difference of ``f_rderiv_int``, column by column.

        ``stepping.cylindrical_rderiv_difference`` with ``_cylindrical_segment_scan``,
        transcribed structurally. A lane owns one column of words (a (phi, z) column,
        or one stored part of it on complex storage) and walks the rows in order:

        * ``out`` starts at ``+0`` everywhere (the array path's ``out[...] = 0``); a row
          no segment writes keeps it;
        * per segment start ``s``: ``F = +0`` at row ``s``, then
          ``F = F + rinv[g-1] * ((Fp[g]*w[g]) - (Fp[g-1]*w[g-1]))`` for ``g > s``, one
          rounding per operation, in row order;
        * the target row that reads row ``g`` takes the difference where its chunk
          starts at ``s``: row ``g - 1`` takes ``F[g] - F[g-1]`` (forward), row ``g``
          takes ``F[g-1] - F[g]`` (backward).

        ``rows_src`` is ``rows_stored + 1`` on the B side, whose extra row is MEEP's
        stored wall (a masked ``+0`` load), and ``rows_stored`` on the D side; the plan
        pins ``n_target = rows_src - 1`` (forward) or ``rows_src`` (backward), which is
        what keeps every written row inside the target. Rows past a chunk's end are
        summed and never read, as in the array path.
        """
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = col < ncol
        zero = tl.zeros([BLOCK], dtype=tl.float32)
        for g in range(0, n_target):
            tl.store(out + g * ncol + col, zero, mask=live)
        for t in range(0, n_segs):
            s = tl.load(segs + t)
            total = zero
            prev = tl.load(w + s) * tl.load(fp + s * ncol + col,
                                            mask=live & (s < rows_stored), other=0.0)
            for g in range(s + 1, rows_src):
                here = tl.load(w + g) * tl.load(fp + g * ncol + col,
                                                mask=live & (g < rows_stored), other=0.0)
                after = total + tl.load(rinv + (g - 1)) * (here - prev)
                if BACKWARD:
                    row = g
                    diff = total - after
                else:
                    row = g - 1
                    diff = after - total
                owned = tl.load(starts + row) == s
                tl.store(out + row * ncol + col, diff, mask=live & owned)
                total = after
                prev = here

    @triton.jit
    def cyl_pml_curl_step(
        f0, f1, f2,                   # targets:    Bx,By,Bz  or  Dx,Dy,Dz   (in/out)
        u0, u1, u2,                   # auxiliaries: fu_B*    or  fu_D*      (in/out)
        g0, g1, g2,                   # sources:    Ex,Ey,Ez  or  Hx,Hy,Hz   (in)
        rad,                          # the radial difference (cyl_radial_difference)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, ONE sub-lattice
        cb,                           # chunk record: the forms' membership and presence
        courant,                      # float64 [4*Courant, Courant, -Courant]
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,       # 0 = B (forward differences), 1 = D
        BLOCK: tl.constexpr,
    ):
        """One cylindrical curl sub-step, m = 0: three curls, the PML forms, the axis rules.

        The index arithmetic is the general 3-D one (``i*ny*nz + j*nz + k``) although
        ``ny`` is 1 on a Dcyl grid. r (axis 0) is CYL_AXIS: the far face is the
        metallic zero (``stepping._shift_up``) and the near ghost is not implemented
        (unobservable; module docstring). phi (axis 1) is PERIODIC on one cell and its
        curl partners are absent by the record's presence word. z (axis 2) is
        METALLIC.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis ---------------------------------------------
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vr = live & (si >= 0) & (si < nx)
        vz = live & (sk >= 0) & (sk < nz)
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        vp = live

        o_r = si * nyz + j * nz + k
        o_p = i * nyz + sj * nz + k
        o_z = i * nyz + j * nz + sk

        a = tl.load(g0 + idx, mask=live, other=0.0)
        b = tl.load(g1 + idx, mask=live, other=0.0)
        c = tl.load(g2 + idx, mask=live, other=0.0)
        a_z = tl.load(g0 + o_z, mask=vz, other=0.0)
        b_z = tl.load(g1 + o_z, mask=vz, other=0.0)
        c_r = tl.load(g2 + o_r, mask=vr, other=0.0)
        c_p = tl.load(g2 + o_p, mask=vp, other=0.0)

        # --- the curls (stepping._curl_from_operands, by partner presence) ----------
        # Target 0 differences g2 = Ez/Hz along phi and g1 = Ey/Hy along z; target 1
        # g0 = Ex/Hx along z and g2 along r (stepping's term table). phi carries no
        # partner on a Dcyl grid, so the presence word selects the one-partner form for
        # target 0 and the full left-to-right bracket for target 1. Target 2's WHOLE
        # curl is dtdx times the radial difference (step_db.cpp:86-119).
        presence = tl.load(cb + RECORD_PRESENCE)
        pr = axis_present(presence, 0)
        pp = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0 = curl_stencil(c_p, c, b, b_z, dtdx, pp, pz)
        curl1 = curl_stencil(a_z, a, c, c_r, dtdx, pz, pr)
        curl2 = dtdx * tl.load(rad + idx, mask=live, other=0.0)

        # --- ownership mask (stepping._mask_non_owned_cells, is_axis + metallic) ------
        # Per axis, for every target whose Yee shift there is 0 (fields.IYEE_SHIFTS):
        #   B: Bx(0,1,1) -> r ;  By(1,0,1) -> phi only, which is periodic: nothing ;
        #      Bz(1,1,0) -> z
        #   D: Dx(1,0,0) -> phi (nothing) + z ;  Dy(0,1,0) -> r + z ;  Dz(0,0,1) -> r
        at_r, at_z = i == 0, k == 0
        if BACKWARD:
            curl0 = tl.where(at_z, 0.0, curl0)
            curl1 = tl.where(at_r, 0.0, curl1)
            curl1 = tl.where(at_z, 0.0, curl1)
            curl2 = tl.where(at_r, 0.0, curl2)
        else:
            curl0 = tl.where(at_r, 0.0, curl0)
            curl2 = tl.where(at_z, 0.0, curl2)

        # --- the PML forms (stepping._apply_curl_regions) ---------------------------
        # dsig/dsigu follow vec.hpp's cycle_direction: target 0 takes (phi, z),
        # target 1 (z, r), target 2 (r, phi). The recurrence runs where the target's
        # chunk carries its dsigu layer -- the record line of that axis on the
        # target's own sub-lattice (half-integer for B, integer for D) -- and fu is
        # read and written there only.
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

        p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)
        v0, n0 = curl_form(tl.load(f0 + idx, mask=live, other=0.0), p0, curl0,
                           km_y, si_y, km_z, si_z, du0)
        p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)
        v1, n1 = curl_form(tl.load(f1 + idx, mask=live, other=0.0), p1, curl1,
                           km_z, si_z, km_x, si_x, du1)
        p2 = tl.load(u2 + idx, mask=live & du2, other=0.0)
        v2, n2 = curl_form(tl.load(f2 + idx, mask=live, other=0.0), p2, curl2,
                           km_x, si_x, km_y, si_y, du2)

        # --- the m = 0 axis rules, on the UPDATED words ------------------------------
        # B (stepping._cylindrical_axis_zero_B): Br[0] = 0.
        # D (_cylindrical_axis_zero_D): Dz[0] += realnum(double(4*Courant) *
        # double(Hp[0])) -- a double product rounded once, then a float32 add into the
        # updated field; Hp is the g1 centre `b` (the stored Hy under the layer) --
        # then Dp[0] = 0. The two touch different components.
        if BACKWARD:
            v1 = tl.where(at_r, 0.0, v1)
            dfcnd = tl.cast(tl.cast(b, tl.float64) * tl.load(courant + COURANT_FOUR),
                            tl.float32)
            v2 = tl.where(at_r, v2 + dfcnd, v2)
        else:
            v0 = tl.where(at_r, 0.0, v0)

        # fu on the recurrence's cells ONLY (the auxiliary contract); the field everywhere.
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

else:  # pragma: no cover - the laptop path
    cyl_radial_difference = None  # type: ignore[assignment]
    cyl_pml_curl_step = None  # type: ignore[assignment]


def cylindrical_curl_kernel() -> Any:
    """The shipped curl ``@triton.jit`` kernel, or a diagnosable ImportError.

    The accessor exists so a caller that needs the kernel gets an explanation rather
    than a ``None`` that fails later as a ``TypeError`` far from its cause.
    """
    if cyl_pml_curl_step is None:
        raise ImportError(
            "the cylindrical curl kernel needs the optional `triton` package "
            "(pip install triton). The engine runs without it; only this fast path "
            f"is unavailable. Original error: {_TRITON_IMPORT_ERROR}")
    return cyl_pml_curl_step


def radial_scan_kernel() -> Any:
    """The shipped column-serial scan kernel, or a diagnosable ImportError."""
    if cyl_radial_difference is None:
        raise ImportError(
            "the cylindrical radial scan kernel needs the optional `triton` package "
            "(pip install triton). The engine runs without it; only this fast path "
            f"is unavailable. Original error: {_TRITON_IMPORT_ERROR}")
    return cyl_radial_difference


# ---------------------------------------------------------------------------
# Host-side bindings the kernels read: the scan's rows and segments, the Courant words
# ---------------------------------------------------------------------------

def courant_constants(courant: float) -> Any:
    """The double constants of the axis rules, as a host ``float64`` vector.

    ``[4*Courant, Courant, -Courant]`` in double, each exact (a power-of-two multiple
    or a negation of the run's double ``grid.courant``). The axis rules multiply a
    ``realnum`` operand by them in double and round once (step_db.cpp:324, :377;
    ``stepping._realnum_double_product``). A Python float kernel argument is typed
    fp32 and would round the constant first, so the words travel by pointer.
    """
    import numpy  # noqa: PLC0415

    value = float(courant)
    words = numpy.empty(len(COURANT_WORDS), dtype=numpy.float64)
    words[COURANT_WORDS["four"]] = 4.0 * value
    words[COURANT_WORDS["plus"]] = 1.0 * value
    words[COURANT_WORDS["minus"]] = -1.0 * value
    return words


def radial_rows(sub_step: str, rows_stored: int) -> Tuple[Any, Any]:
    """``(w, rinv)``: the scan's two invariant row vectors, as host float32 arrays.

    ``stepping._cylindrical_rderiv_rows`` -- the array path's own builder, called, not
    re-derived -- over the scan's source rows: ``rows_stored + 1`` on the B side (Ep
    and its wall row), ``rows_stored`` on the D side, at the side's ``ir0``.
    """
    import numpy  # noqa: PLC0415

    from ..stepping import _cylindrical_rderiv_rows  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    rows_src = int(rows_stored) + (1 if spec["extend_wall_row"] else 0)
    weights, rinv = _cylindrical_rderiv_rows(numpy, rows_src, spec["prefix_ir0"],
                                             numpy.dtype(numpy.float32))
    return (numpy.ascontiguousarray(weights.reshape(-1)),
            numpy.ascontiguousarray(rinv.reshape(-1)))


def radial_scan_starts(fields: Any, pml: Any, sub_step: str,
                       chunk_layout: Any = None) -> Any:
    """Per target row, the first stored r row of its owning chunk: a host int32 line.

    ``stepping._cylindrical_scan_starts`` of the target (Bz or Dz) -- the array path's
    own function, over the division it steps with. ``None`` (one chunk) becomes the
    all-zero line, which selects the same rows. Refused by name
    (:class:`~meep_gpu.chunk_map.ChunkLayoutRefusal`):

    * a division whose r cuts differ along z: the array path then keys the sum per
      cell, which no per-row line describes (DESIGN D8);
    * a division other than the one the plan's chunk record is built from
      (``chunk_layout``, or the one-rank default when it is None): the kernel's forms
      and its sums would then follow two different divisions.
    """
    import numpy  # noqa: PLC0415

    from ..chunk_map import ChunkLayout, ChunkLayoutRefusal, chunk_bounds_record  # noqa: PLC0415
    from ..stepping import _chunk_layout, _cylindrical_scan_starts, _host_ints  # noqa: PLC0415
    from .launch import chunk_bounds_for  # noqa: PLC0415

    target = "Bz" if sub_step == "step_B" else "Dz"
    grid = fields.grid
    rows = int(grid.shape[0])
    stepped = _chunk_layout(fields, pml)
    stepped_record = chunk_bounds_record(
        stepped if stepped is not None else ChunkLayout.one_rank(grid, pml))
    if not numpy.array_equal(stepped_record, chunk_bounds_for(fields, pml, chunk_layout)):
        raise ChunkLayoutRefusal(
            "the cylindrical scan would sum over the division the array path steps with "
            "while the plan's chunk record describes another; the two must be one "
            "division")
    starts = _cylindrical_scan_starts(fields, pml, target)
    if starts is None:
        return numpy.zeros(rows, dtype=numpy.int32)
    host = numpy.asarray(_host_ints(starts))
    if host.shape != (rows, 1, 1):
        raise ChunkLayoutRefusal(
            f"the chunk division's r cuts differ along z (the array path keys the "
            f"radial sum of {target} per cell, shape {host.shape}); the kernels serve a "
            f"division with global r cuts only")
    return host.reshape(-1).astype(numpy.int32)


class RadialScan:
    """One sub-step's radial difference, bound once and launched every step.

    Holds the source volume (Ep on B, Hp on D), the two row vectors, the starts line
    and segment list, and the output volume the curl reads. Complex64 storage is
    viewed as float32 words, one scan column per stored part.
    """

    __slots__ = ("sub_step", "backward", "block", "rows_stored", "rows_src", "n_target",
                 "ncol", "n_segs", "starts", "segments", "out", "_source", "_w", "_rinv",
                 "_starts", "_segs", "_out", "_grid", "_kernel", "_pointer")

    def __init__(self, sub_step: str, source: Any, starts: Any, xp: Any,
                 block: Optional[int] = None, kernel: Any = None) -> None:
        import numpy  # noqa: PLC0415

        from ..chunk_map import ChunkLayoutRefusal  # noqa: PLC0415
        from .launch import CupyPointer  # noqa: PLC0415

        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        spec = SUB_STEPS[sub_step]
        self.sub_step = sub_step
        self.backward = int(spec["backward"])
        self.block = DEFAULT_BLOCK if block is None else int(block)
        shape = tuple(int(n) for n in source.shape)
        if len(shape) != 3:
            raise ValueError(f"the radial scan reads a 3-D volume, got shape {shape}")
        self.rows_stored = shape[0]
        self.rows_src = self.rows_stored + (1 if spec["extend_wall_row"] else 0)
        self.n_target = self.rows_stored
        line = numpy.asarray(starts)
        if line.shape != (self.n_target,) or not numpy.issubdtype(line.dtype, numpy.integer):
            raise ChunkLayoutRefusal(
                f"the radial scan needs one integer chunk start per target row "
                f"({self.n_target}), got dtype {line.dtype} and shape {line.shape}")
        rows = numpy.arange(self.n_target)
        lowest = rows if not self.backward else rows - 1
        if numpy.any(line < 0) or numpy.any(line > numpy.maximum(lowest, 0)):
            raise ChunkLayoutRefusal(
                "a chunk start lies above the row that reads its sum: the scan's line "
                f"is {line.tolist()}")
        self.starts = line.astype(numpy.int32)
        self.starts.setflags(write=False)
        self.segments = numpy.unique(self.starts).astype(numpy.int32)
        self.segments.setflags(write=False)
        self.n_segs = int(self.segments.size)
        words = source.view(numpy.float32) if source.dtype.kind == "c" else source
        if str(words.dtype) != "float32" or not words.flags.c_contiguous:
            raise ValueError(
                f"the radial scan reads contiguous float32 words, got {words.dtype}")
        self.ncol = int(words.shape[1] * words.shape[2])
        w, rinv = radial_rows(sub_step, self.rows_stored)
        self._source = words
        self._w = xp.asarray(w)
        self._rinv = xp.asarray(rinv)
        self._starts = xp.asarray(self.starts)
        self._segs = xp.asarray(self.segments)
        self.out = xp.zeros((self.n_target,) + shape[1:], dtype=source.dtype)
        self._out = self.out.view(numpy.float32) if source.dtype.kind == "c" else self.out
        self._grid = ((self.ncol + self.block - 1) // self.block,)
        self._kernel = kernel
        self._pointer = CupyPointer

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the scan: ``out`` holds this step's radial difference afterwards."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = self._kernel if self._kernel is not None else radial_scan_kernel()
        pointer = self._pointer
        kernel[self._grid](
            pointer(self._source), pointer(self._w), pointer(self._rinv),
            pointer(self._starts), pointer(self._segs), pointer(self._out),
            self.rows_stored, self.rows_src, self.n_target, self.ncol, self.n_segs,
            BACKWARD=self.backward,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )


# ---------------------------------------------------------------------------
# The coverage predicates
# ---------------------------------------------------------------------------
#
# WRITTEN POSITIVELY AND ENUMERATING. Every requirement is named and checked and
# coverage is never inferred from the absence of a known blocker — the two worst
# defects in this project were silent wrong answers, not crashes.
#
# THEY DO NOT CALL ``coverage._grid_reasons``. That function's clause 6
# (coverage.py:172-178) is a BLANKET cylindrical refusal shared by every Cartesian
# predicate in that file; composing it here would make these predicates vacuously
# False. Its clauses are restated below, individually, with the cylindrical ones
# replaced by the positive Dcyl requirements. That duplication is deliberate and it
# is the reason the shared clause must NOT be widened to admit Dcyl: it is now
# load-bearing in two directions at once, and whoever owns ``coverage.py`` should be
# told so.

#: The shared helpers these predicates DO reuse, named as data so a test can assert
#: every one of them still exists and a rename in the shared file fails at the merge
#: bar instead of silently dropping a clause.
SHARED_CLAUSES: Tuple[str, ...] = (
    "_layout_reasons", "_volume_reasons", "_coefficient_reasons",
    "_susceptibility_reasons", "_call")


def cylindrical_axis_policy(grid: Any) -> Optional[bool]:
    """``grid.accurate_fields_near_cylorigin``, READ rather than assumed.

    It is IRRELEVANT AT m = 0 — it selects how many near-axis rows the |m| >= 2
    stability hack holds at zero (``stepping._cylindrical_axis_rows``, :601-624), and
    at m = 0 there is no such hack. It is read and reported anyway, and a grid that
    cannot answer is REFUSED, so that a later widening to |m| >= 2 cannot inherit
    this module's silence about it.
    """
    value = getattr(grid, "accurate_fields_near_cylorigin", None)
    return None if value is None else bool(value)


def _shared_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The non-cylindrical half of ``coverage._grid_reasons``, restated.

    Clause numbering follows that function so the two can be diffed by eye. Clauses
    5 and 6 (fold / Cartesian-only) are NOT here — they are replaced by the positive
    Dcyl requirements in :func:`cylindrical_curl_coverage`.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernels launch against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. At m = 0 Dcyl storage IS real float32 (measured on a lifted
    #    grid: Ez float32 at m = 0, complex64 at m = 1 and 2), which is the whole
    #    reason this kernel needs no complex arithmetic.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")

    # 3. An active split-field absorber. Without one the plain path is the
    #    bit-identical one and this recurrence is the wrong sub-step entirely.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this kernel implements the split-field "
                       "path only)")

    # 7. k = 0. A Bloch phase needs complex storage and multiplies one wrapped plane.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 8. No conductivity on any curl target: that routes to the three-history
    #    conductive-PML recurrence, not to this one.
    reader = getattr(fields, "condfac_for", None)
    if callable(reader):
        for target in CURL_TARGETS:
            if reader(target) is not None:
                reasons.append(f"a conductivity is installed on {target} (mp.Absorber path)")
    if getattr(fields, "has_magnetic_conductivity", False):
        reasons.append("a magnetic (B) conductivity is installed")

    # 10. No instantaneous nonlinearity: the Pade factor REPLACES the constitutive
    #     product and composes with dispersion.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not carried)")

    # 11/12. BFAST adds a second additive term to every curl; beta adds out-of-plane
    #        couplings. Both are silent additions, not errors.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    # 9a/9b. A registered susceptibility must be one this package understands. Not a
    #        refusal of dispersion: it is constitutive-only and the curl differences
    #        the STORED E and H arrays. The clause is vacuous against the engine as
    #        it stands and is written anyway.
    reasons.extend(_coverage._susceptibility_reasons(fields))

    return reasons


def _cylindrical_geometry_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The POSITIVE Dcyl requirements — what replaces the blanket clauses 5 and 6."""
    reasons: List[str] = []

    if not getattr(grid, "cylindrical", False):
        reasons.append("grid is not cylindrical (this kernel steps Dcyl grids only)")

    # m = 0 EXACTLY, refused by name. Everything else needs complex storage, the
    # i*m/r coupling and per-|m| axis rules — a second kernel, not a flag.
    m = getattr(grid, "m", None)
    if m is None:
        reasons.append("grid does not report m")
    elif int(m) != COVERED_M:
        reasons.append(
            f"grid.m = {m!r}; this kernel carries m = {COVERED_M} ONLY. |m| >= 1 forces "
            "complex64 storage, the i*m/r pass and the per-|m| axis rules "
            "(stepping._cylindrical_imr_increment, _cylindrical_axis_increment_B/D) — "
            "the complex family's kernels")

    flags = tuple(bool(_coverage._call(grid, "is_axis", axis, default=False))
                  for axis in range(3))
    if flags != CYLINDRICAL_AXIS_FLAGS:
        reasons.append(f"grid.is_axis {flags!r} is not {CYLINDRICAL_AXIS_FLAGS!r} "
                       "(r must be the leading axis)")

    kinds = _coverage._boundary_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    elif tuple(kinds) != CYLINDRICAL_BOUNDARY_KINDS:
        reasons.append(f"boundary kinds {tuple(kinds)!r} are not "
                       f"{CYLINDRICAL_BOUNDARY_KINDS!r}")

    # No fold anywhere. Grid already refuses a mirror on a Dcyl cell; the clause is
    # written because "another module already guards it" is exactly the reasoning
    # this file exists to refuse.
    if _coverage._call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _coverage._call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
    elif int(shape[1]) != 1:
        reasons.append(f"phi extent is {shape[1]} cells, not 1 (Dcyl is 2.5-D: the "
                       "exp(i*m*phi) dependence is analytic)")

    if cylindrical_axis_policy(grid) is None:
        reasons.append("grid does not report accurate_fields_near_cylorigin; the flag "
                       "is irrelevant at m = 0 but must be readable before any |m| >= 2 "
                       "admission inherits this predicate's silence about it")

    # A configuration the array path itself refuses is not one a kernel may step.
    try:
        from ..stepping import _require_cylindrical_steppable  # noqa: PLC0415

        _require_cylindrical_steppable(grid)
    except Exception as exc:  # noqa: BLE001 - an unsteppable grid is refused, not crashed on
        reasons.append(f"stepping refuses this cylindrical grid: {exc!r}")

    return reasons


def _installed_layout(fields: Any) -> Any:
    """The division the driver installed in ``fields``, or None (read without building)."""
    try:
        return getattr(fields, "chunk_layout", None)
    except Exception:  # noqa: BLE001 - an unreadable attribute is not an installed layout
        return None


def cylindrical_curl_coverage(fields: Any, pml: Any) -> Coverage:
    """May the Triton cylindrical curl kernel step this (fields, pml) pair?

    Every clause names its own requirement and appends its own reason; the scan
    continues after a failure so a refusal reports everything that disqualified the
    run rather than the first thing.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _shared_reasons(fields, pml, grid)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    # STORED E — the invariant behind differencing E while a pole is live, and the
    # one an edit that ever makes stores_E optional under PML must be caught by.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # THE DIVISION'S r CUTS ARE GLOBAL. The radial sum restarts at each chunk's first
    # row; a lifted division whose r cuts differ along z keys the array path's sum per
    # cell, which the scan's per-row line cannot carry (DESIGN D8). Read from the
    # installed division only (the one-rank default is a product layout).
    layout = _installed_layout(fields)
    if layout is not None and len(getattr(layout, "boxes", ())) > 1:
        try:
            cuts = layout.boundaries(0)
        except Exception as exc:  # noqa: BLE001 - an unreadable division is refused
            reasons.append(f"the chunk division's r cuts could not be read: {exc!r}")
        else:
            if cuts is None:
                reasons.append(
                    "the chunk division's r cuts differ along z (the radial sum is keyed "
                    "per cell there; the scan kernel carries global r cuts only)")

    # Every volume the kernel dereferences: six targets, six auxiliaries, six
    # sources. The D side additionally reads the stored Hp for the axis add, which
    # `Fields.get_H` returns as `fields.Hy` under an active PML -- the same array
    # already checked here.
    names = (CURL_TARGETS + tuple("fu_" + target for target in CURL_TARGETS)
             + CURL_SOURCES)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_coverage._layout_reasons(fields, shape, names))

    # The curl plans EITHER sub-step, so both Yee sub-lattices must be present.
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coverage._coefficient_reasons(pml, shape, ("kms", "sinv"),
                                                      ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def cylindrical_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the EXISTING ``kernels.constitutive_step`` run on a Dcyl grid?

    IT NEEDS NO NEW KERNEL, and that is a finding rather than an assumption:
    ``stepping.update_H`` and ``stepping.update_E`` carry NO cylindrical branch at
    all. They are element-wise with per-axis coefficient tables indexed on the
    component's OWN axis, and cylindrical changes nothing there except that axis 1
    has n = 1. What refuses them today is ``coverage._grid_reasons`` clause 6, the
    blanket cylindrical refusal — which is shared, is right for the Cartesian
    predicates, and must stay.

    The enumeration is :func:`cylindrical_curl_coverage`'s minus the curl-specific
    clauses (no ghost rule, no ownership mask, no prefix), plus the E side's own
    refusals: a registered polarization (the source becomes ``D - sum P`` and that
    configuration belongs to the ADE kernel), an off-diagonal chi1inv row (the row
    product reads neighbours and the sub-step stops being element-wise), and
    ``stores_E`` false.

    The gate's ``constitutive`` leg runs the shipped ``ConstitutivePlan`` against
    ``stepping.update_H``/``update_E`` on a real Dcyl grid, word for word.
    """
    if side not in _coverage.CONSTITUTIVE_SIDES:
        raise ValueError(
            f"side must be one of {tuple(_coverage.CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = _coverage.CONSTITUTIVE_SIDES[side]
    reasons = _shared_reasons(fields, pml, grid)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is (D - sum P), not D "
                "(fields.py:1096-1105) — that configuration belongs to the ADE kernel")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product reads "
                "neighbours; this sub-step is element-wise)")
        if not getattr(fields, "stores_E", False):
            reasons.append("E is recomputed from D rather than stored")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_coverage._layout_reasons(fields, shape, names))

    if side == "E" and len(shape) == 3:
        reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))

    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coverage._coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))




def plan_cylindrical_constitutive(fields: Any, pml: Any, side: str,
                                  block: Optional[int] = None,
                                  chunk_layout: Any = None) -> Any:
    """Build the root elementwise constitutive plan for the admitted Dcyl slice.

    The compiled arithmetic is :func:`.kernels.constitutive_step`; cylindrical m = 0
    changes neither its pointwise forms nor its coefficient layout. This wrapper owns
    the distinct predicate, so the Cartesian builder cannot silently broaden its blanket
    Dcyl refusal. ``chunk_layout`` is the run's division (``None`` for the one-rank
    default), and the H side carries MEEP's start of H (``h_state``), as the Cartesian
    engine route does.
    """
    if side not in _coverage.CONSTITUTIVE_SIDES:
        raise ValueError(
            f"side must be one of {tuple(_coverage.CONSTITUTIVE_SIDES)}, got {side!r}")
    if not cylindrical_constitutive_coverage(fields, pml, side).covered:
        return None

    from .launch import ConstitutivePlan, chunk_bounds_for  # noqa: PLC0415

    spec = _coverage.CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    grid = fields.grid
    return ConstitutivePlan(
        side, grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
    )


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalCurlPlan:
    """A launchable cylindrical curl sub-step: the radial scan, then the curl kernel.

    Built two ways and launched ONE way, exactly as :class:`launch.PmlCurlPlan` is:
    :func:`plan_cylindrical_curl` is the engine route (reads a real ``Fields``/``PML``
    and passes through the predicate), :func:`plan_cylindrical_curl_from_arrays` is the
    harness route (bare device arrays, no predicate, the chunk record and the scan's
    starts line REQUIRED). Both produce this object and both go through :meth:`run`,
    so the bytes a gate certifies are the bytes the engine would launch.

    Allocation-free per step: the scan's output volume and its row vectors, starts
    line and segment list, and the Courant words are bound once, at plan time.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "courant", "backward", "block",
                 "chunk_bounds", "scan", "_targets", "_aux", "_sources", "_coefficients",
                 "_cb", "_courant", "_grid", "_kernel", "_pointer")

    def __init__(self, sub_step: str, shape, dtdx: float, courant: float, block: int,
                 targets, auxiliaries, sources, coefficients, xp: Any, *,
                 chunk_bounds: Any, scan_starts: Any, kernel: Any = None,
                 scan_kernel: Any = None) -> None:
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        spec = SUB_STEPS[sub_step]
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "CylindricalCurlPlan")
        # float(): the array path multiplies a float32 volume by a Python float, which
        # NumPy/CuPy cast to float32 before the multiply, and Triton types a Python
        # float argument as fp32 -- so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        # The double Courant of the m = 0 axis add travels as float64 words.
        self.courant = float(courant)
        self._courant = CupyPointer(xp.asarray(courant_constants(self.courant)))
        self.backward = int(spec["backward"])
        self.block = int(block)
        self._pointer = CupyPointer
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        source_map = dict(zip(spec["sources"], sources))
        self.scan = RadialScan(sub_step, source_map[spec["prefix_component"]],
                               scan_starts, xp, block=self.block, kernel=scan_kernel)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override exists for exactly one caller: the gate's source-mutation leg,
        # which compiles a deliberately broken copy of the shipped kernel.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step: the scan, then the curl. In place; references stay valid.

        ``guard`` is not for callers: it exists so a gate can MEASURE the contraction
        guard's effect rather than assert it. Everything else takes the module constant.
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        self.scan.run(guard)
        kernel = self._kernel if self._kernel is not None else cylindrical_curl_kernel()
        nx, ny, nz = self.shape
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            self._pointer(self.scan.out),
            *self._coefficients,
            self._cb, self._courant,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"CylindricalCurlPlan({self.sub_step}, shape={self.shape}, m=0, "
                f"block={self.block})")


def plan_cylindrical_curl(fields: Any, pml: Any, sub_step: str,
                          block: Optional[int] = None,
                          chunk_layout: Any = None) -> Optional[CylindricalCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    ``chunk_layout`` is the run's division (``FdtdDriver.chunk_layout``), ``None`` for
    the one-rank default; the chunk record and the scan's starts line both follow it
    (:func:`radial_scan_starts` refuses by name when they would not). The Triton import
    stays below the predicate, so a NumPy host can plan (to ``None``) without the
    optional dependency being importable at all.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not cylindrical_curl_coverage(fields, pml).covered:
        return None

    from .launch import chunk_bounds_for  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    return CylindricalCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, float(grid.courant),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, "fu_" + n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        grid.xp,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        scan_starts=radial_scan_starts(fields, pml, sub_step, chunk_layout),
    )


def plan_cylindrical_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                      flat: Dict[str, Any], dtdx: float, courant: float,
                                      xp: Any, *, chunk_bounds: Any, scan_starts: Any,
                                      block: Optional[int] = None, kernel: Any = None,
                                      scan_kernel: Any = None) -> CylindricalCurlPlan:
    """Build a plan from bare device arrays -- the gate's and benchmark's route.

    ``arrays`` is keyed by component name, ``flat`` by ``kms_x``/``sinv_x``/... on the
    sub-lattice the caller ALREADY SELECTED (half-integer for B, integer for D), so a
    gate can hand over a swapped pair and measure that the swap is caught.
    ``chunk_bounds`` (``chunk_map.chunk_bounds_record``) and ``scan_starts`` (one chunk
    start per r row, :func:`radial_scan_starts`) are REQUIRED: a harness plan with a
    defaulted division would stop measuring the predicate it exists to measure. No
    coverage predicate runs here.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return CylindricalCurlPlan(
        sub_step, shape, dtdx, courant, DEFAULT_BLOCK if block is None else block,
        [arrays[n] for n in spec["targets"]],
        [arrays["fu_" + n] for n in spec["targets"]],
        [arrays[n] for n in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        xp, chunk_bounds=chunk_bounds, scan_starts=scan_starts, kernel=kernel,
        scan_kernel=scan_kernel,
    )
