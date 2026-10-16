"""The complex conductive ``step_D`` welded into complex stored-E ``update_E``.

ONE LAUNCH FOR ALL THREE COMPONENTS, and the displacement never leaves a register.
The separate composition for this configuration is TWO launches — one whole-grid
conductive curl plus one pointwise stored-E — and this is one.

It claims the D->E cell the Triton fusion matrix ranks second among the unbuilt
cells after the Dcyl one: ``(complex conductive no-PML curl, complex no-PML stored
E)``, four corpus rows —
``TestLoadDump.test_load_dump_{structure,structure_sharded,chunk_layout_file,
chunk_layout_sim}_3d`` — measured at ``D->E 4, ceiling 4`` in
``parity/meep_gpu/results/fusion_matrix_triton_2026-08-30_electric``.

THE SIBLING IS :mod:`..metal_kernels.complex_conductive_fused_pair`, which serves
the same cell 4/4 on Metal and was released 2026-08-29. This module is that
product's Triton twin: the same two certified halves, the same four refusals in the
seam, the same priced decision not to carry the wall. Where the two backends differ
is only in how the arithmetic is spelled — MSL there, ``triton.jit`` here — and
each half below is transcribed from ITS OWN backend's certified body, never from
the sibling's.

EVERYTHING HERE IS TRANSCRIBED, and every arithmetic line carries the function it
came from:

* the complex conductive D curl —
  :func:`.complex_no_pml_conductive.complex_conductive_no_pml_curl_step`, character
  for character through its ghost rule, its Bloch connection of the unshifted
  partners on plane 0, MEEP's per-part stencil with the chunk record's partner
  presence, its ownership mask and its per-target conductive tail;
* the constitutive half —
  :func:`.complex_no_pml_stored_e.complex_stored_e_step`, whose pole chain and
  ``inv_eps`` multiply transcribe ``stepping.update_E``:981-984 through
  ``Fields.displacement_minus_polarization``.

A reader can diff this body against those two and see that nothing moved. If you
find yourself deriving here, you have taken a wrong turn.

===========================================================================
THE SEAM, AND WHY IT IS EMPTY ON THIS CELL — the whole design question
===========================================================================

The driver runs FOUR passes between ``step_D`` and ``update_E``
(driver.py:3293-3304): the electric injection (:3294-3299, which on a conductive
row takes the ``condinv``-scaled path ``FdtdDriver._inject_electric_through_
conductivity`` at :3296), ``fill_symmetry_bc_D`` (:3300), ``zero_metal_D`` (:3301)
and ``fill_folded_far_ghosts_D`` (:3302). THIS FAMILY CARRIES NONE OF THEM. Each is
instead refused by a named clause that makes the pass INERT, and that is a
different and stronger thing than not carrying it:

* **the electric sources — REFUSED BY NAME, and it costs this cell NOTHING.**
  The injection lands between the two halves, so a fused pair would consume a
  pre-injection D. MEASURED COST ON THE CORPUS: ZERO of the four rows — every one
  of them declares a single MAGNETIC source, which is injected in the B/H seam and
  does not reach this one. That is why :data:`CARRIES_DEPOSIT_REPAIR` is False here
  while the two D->E COMPLEX pairs beside it are True: those cells are all-electric
  and would serve nothing without the repair, and this one is served in full
  without it. Ignorance is still never an empty set: ``Fields`` does not hold the
  source list, so an undeclared ``sources`` is itself a REFUSAL. THE CONDUCTIVE
  INJECTION IS THE SAME CLAUSE, not a second one: ``FdtdDriver.step`` guards it
  with ``if electric and self.fields.has_conductivity`` (driver.py:3295), so with
  no electric source it cannot run whatever the conductivity is;
* **``fill_symmetry_bc_D`` — INERT, because a mirror plane is refused.** Both
  certified halves refuse a live mirror already; the clause is restated here by
  name because a reader should not have to chase a shared grid predicate to learn
  which driver pass that refusal empties. Without a mirror the array path's fill
  returns at its first line (stepping.py:1481-1482);
* **``fill_folded_far_ghosts_D`` — INERT for the same reason**
  (stepping.py:1565-1566);
* **``zero_metal_D`` — REFUSED BY NAME, and the refusal is priced.** It writes zero
  into stored cell 0 of every D component whose Yee shift on a walled axis is 0
  (stepping.py:2206-2247), which ``update_E`` then reads.
  :mod:`.complex_fused_electric_pair` and :mod:`.folded_fused_pair` carry it inline
  and this family could too — but a METALLIC axis buys ZERO seam-instances on the
  measured corpus: all four rows of this cell are
  ``boundary_kinds == ['periodic', 'periodic', 'periodic']``. A widening that buys
  nothing is declined, and the count is recorded so the decision can be revisited
  when the corpus moves rather than re-argued. THE CLAUSE IS NOT A COMMENT: an
  unreadable boundary table is a refusal too, because a family that cannot
  establish the pass is inert has not established it.

So under this family's clauses the D->E seam holds NOTHING, which is what makes a
single launch legitimate rather than merely convenient.

===========================================================================
WHY THERE IS NO STENCIL PROBLEM HERE, and why that is not general
===========================================================================

The fusion matrix refuses EIGHTEEN D->E seam-instances as STRUCTURALLY UNFUSABLE on
any backend: ``stepping._offdiagonal_terms`` (stepping.py:1235-1253) reads each
partner component's D volume at FOUR indices while ``step_D`` writes those volumes
in place in the first half of the same launch, and no grid-wide barrier exists
inside one launch. THIS CONSTITUTIVE IS NOT THAT ONE. The certified complex stored
E reads ``g0 + word`` — its OWN component, at its OWN index, the word the same
program computed in the curl half — so the value stays in a register and no lane
reads a word another lane writes. An off-diagonal ``chi1inv`` row is refused by the
E half already, and restated here by name because THIS is the reason the cell is
buildable at all.

THE POLE ARRAYS ARE READ, NEVER WRITTEN, by either half. ``update_P`` runs after
``update_E`` (driver.py:3306) and is not in this seam; the ``P`` volumes this
kernel loads are the ones the previous timestep left, exactly as the certified
stored-E body loads them.

===========================================================================
WHAT MOVES, AND IT IS AT MOST ONE LINE PER COMPONENT
===========================================================================

The certified conductive tail ALREADY names the stepped displacement in a
register::

    v0_re = tl.load(f0 + 2 * idx, ...)          # D, before the step
    if COND0: v0 = _mul_field_left(v0, condfac)
    v0_re = v0_re - curl0_re
    if COND0: v0 = _mul_field_left(v0, condinv)
    tl.store(f0 + 2 * idx, v0_re, ...)          # D, after the step

so NOTHING MOVES AT ALL: the constitutive half reads ``v0_re``/``v0_im`` where the
certified body's :func:`.complex_no_pml_stored_e._subtract_complex_poles` read
``tl.load(source + word)``. That single substitution is the whole weld, and it is
the same one :func:`.kernels.fused_curl_constitutive_D` makes on the real path.
THE D STORE IS KEPT: the array path leaves the stepped displacement in the D
volume and the next timestep's ``update_P`` reads it, so a weld that dropped the
store would be fusing away a value the run still needs.

:func:`_subtract_complex_poles_in_registers` below is that substitution written
once. It is :func:`.complex_no_pml_stored_e._subtract_complex_poles` with its two
opening loads replaced by the two arguments and NOTHING else changed — same eight
``if NP > k:`` arms, same order, same left-to-right subtraction. The order is
bit-load-bearing (the module docstring of the certified body says so) and the gate
carries a mutation that reverses it.

===========================================================================
THE PRODUCTS
===========================================================================

Every product on this path has a real operand and is formed one float32 product per
stored part, as MEEP's ``DOCMP`` loop forms it: the conductive and inverse-permittivity
scales through :func:`.complex_fields._mul_field_left`, the Bloch connection through
:func:`.complex_fields._connect` (the unfused product, or a negation spelled
``* -1.0``), and the stencil's ``dtdx`` scale inside ``forms.curl_stencil``. No
complex-by-complex product and no multiply-add is emitted. The seam adds no
multiply at all — it removes two loads.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans two of them; there is no slot it can
claim without a composition rule nothing has measured. Nothing in ``launch.py``
names this module, ``fastpath.plan_fast_path`` is unchanged, and dispatch stays
disabled — the same deferral :mod:`.complex_fused_electric_pair` and
:mod:`.cylindrical_fused_electric_pair` ship under.

Import contract: importable WITHOUT Triton — the predicates and the plan builder
(to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    CONNECTION_KINDS,
    DEFAULT_BLOCK,
    METALLIC,
    PERIODIC,
    _TRITON_IMPORT_ERROR,
    _connect,
    _mul_field_left,
    _word_view,
    bloch_phase_table,
    connection_arguments,
    tl,
    triton,
)
from .complex_no_pml_conductive import (
    complex_conductive_no_pml_curl_coverage,
)
from .complex_no_pml_curl import source_arrays
from .complex_no_pml_stored_e import (
    E_TERMS,
    MAX_POLES,
    LiveComplexPoleBinding,
    StaticComplexPoleBinding,
    complex_stored_e_coverage,
    poles_per_component,
)
from .coverage import Coverage, _boundary_kinds, _call
from .launch import SUB_STEPS, CupyPointer, _flat, chunk_bounds_for
from .no_pml_conductive import conductive_no_pml_targets
from .. import deposit_repair as _deposit_repair

# The curl forms and the record's presence word, read by the curl half. Imported only
# when Triton is; the predicate and the builder to None answer without it.
if _TRITON_IMPORT_ERROR is None:
    from .forms import axis_present, curl_stencil
    from .kernels import RECORD_PRESENCE

#: Does this product bracket its fused launch with the deposit repair? FALSE, and
#: unlike the two D->E COMPLEX pairs beside it that is not a limitation: every one
#: of this cell's four corpus rows declares a MAGNETIC source only, so the seam
#: clause below refuses nothing the corpus drives. The flag is a claim about the
#: PLAN this module builds -- that the leading slot saves and the trailing slot
#: restores -- and is only ever changed in the same edit as that wiring. While False
#: the clause refuses every in-seam electric deposit outright.
CARRIES_DEPOSIT_REPAIR = False

#: The curl sub-step this product starts at. There is no ``CONSTITUTIVE_SIDE``
#: constant: the stored-E half is not one of ``coverage.CONSTITUTIVE_SIDES`` — it is
#: the no-PML branch of ``update_E``, which writes E directly from ``(D - sum P) *
#: inv_eps`` and touches no ``f_w`` or PML lattice at all.
CURL_SUB_STEP = "step_D"

#: The three volumes the curl half writes, in kernel argument order.
CURL_TARGETS: Tuple[str, ...] = tuple(SUB_STEPS[CURL_SUB_STEP]["targets"])

#: The driver call sites ONE launch of this plan performs, in driver order. TWO,
#: not five: the three passes between them are REFUSED by the clauses above rather
#: than carried, and a launch may only claim to replace what it actually performs.
#: The sibling Metal product declares the same two for the same reason.
REPLACES: Tuple[str, ...] = ("step_D", "update_E")

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 1

#: The multiply device functions this kernel is allowed to call, and the only ones:
#: the partner connection (:func:`.complex_fields._connect`) and the real-coefficient
#: scale of each stored part (:func:`.complex_fields._mul_field_left`, the conductive
#: tail and ``inv_eps``); the stencil's ``dtdx`` scale is ``forms.curl_stencil``'s.
#: The test scans the shipped body for any other.
MULTIPLY_HELPERS: Tuple[str, ...] = ("_connect", "_mul_field_left")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CURL_SUB_STEP", "CURL_TARGETS",
    "MULTIPLY_HELPERS", "REPLACES",
    "ComplexConductiveFusedPairPlan",
    "complex_conductive_fused_curl_constitutive_D",
    "complex_conductive_fused_curl_constitutive_D_kernel",
    "complex_conductive_fused_pair_coverage",
    "plan_complex_conductive_fused_pair",
    "plan_complex_conductive_fused_pair_from_arrays",
]


@triton.jit
def _subtract_complex_poles_in_registers(
    real, imag, p0, p1, p2, p3, p4, p5, p6, p7,
    word, live,
    NP: tl.constexpr,
):
    """Subtract up to eight complex P arrays from a REGISTER pair, in exact order.

    :func:`.complex_no_pml_stored_e._subtract_complex_poles` with its two opening
    loads replaced by the two arguments and nothing else changed — the one
    substitution this weld makes, written once so both the reader and the gate's
    transcription leg can compare it to the certified helper statement for
    statement.

    THE ORDER IS BIT-LOAD-BEARING. The poles may not be pre-summed and may not be
    reordered: either changes float32 rounding at two or more poles, which is why
    the certified body spells eight separate arms and why the gate arms a mutation
    that reverses them.
    """
    if NP > 0:
        real = real - tl.load(p0 + word, mask=live, other=0.0)
        imag = imag - tl.load(p0 + word + 1, mask=live, other=0.0)
    if NP > 1:
        real = real - tl.load(p1 + word, mask=live, other=0.0)
        imag = imag - tl.load(p1 + word + 1, mask=live, other=0.0)
    if NP > 2:
        real = real - tl.load(p2 + word, mask=live, other=0.0)
        imag = imag - tl.load(p2 + word + 1, mask=live, other=0.0)
    if NP > 3:
        real = real - tl.load(p3 + word, mask=live, other=0.0)
        imag = imag - tl.load(p3 + word + 1, mask=live, other=0.0)
    if NP > 4:
        real = real - tl.load(p4 + word, mask=live, other=0.0)
        imag = imag - tl.load(p4 + word + 1, mask=live, other=0.0)
    if NP > 5:
        real = real - tl.load(p5 + word, mask=live, other=0.0)
        imag = imag - tl.load(p5 + word + 1, mask=live, other=0.0)
    if NP > 6:
        real = real - tl.load(p6 + word, mask=live, other=0.0)
        imag = imag - tl.load(p6 + word + 1, mask=live, other=0.0)
    if NP > 7:
        real = real - tl.load(p7 + word, mask=live, other=0.0)
        imag = imag - tl.load(p7 + word + 1, mask=live, other=0.0)
    return real, imag


@triton.jit
def complex_conductive_fused_curl_constitutive_D(
    f0, f1, f2,                       # curl targets: Dx,Dy,Dz, complex words
    g0, g1, g2,                       # curl sources: stored Hx,Hy,Hz, complex words
    cf0, cf1, cf2,                    # condfac, float32 volumes or dead placeholders
    ci0, ci1, ci2,                    # condinv, float32 volumes or dead placeholders
    h0, h1, h2,                       # constitutive targets: Ex,Ey,Ez, complex words
    iv0, iv1, iv2,                    # inverse epsilon, float32 VOLUMES
    a0, a1, a2, a3, a4, a5, a6, a7,   # Ex poles, registration order
    b0, b1, b2, b3, b4, b5, b6, b7,   # Ey poles, registration order
    q0, q1, q2, q3, q4, q5, q6, q7,   # Ez poles, registration order
    cb,                               # chunk record: the curl's partner presence
    nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
    BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kind per axis
    COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
    NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complex conductive ``step_D`` + complex stored-E ``update_E``, one launch.

    Everything through the conductive tail is line-for-line
    :func:`.complex_no_pml_conductive.complex_conductive_no_pml_curl_step`; the
    constitutive half is
    :func:`.complex_no_pml_stored_e.complex_stored_e_step` with its ``D`` load
    replaced by the register the tail just produced. Do not flatten the conductive
    passes: each in-place complex64 operation rounds before the next begins.

    ``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim copy of
    the certified body rather than a hand-specialised one. It cannot be 0: the B
    half's seam ends at ``update_H``, which is a different constitutive entirely,
    and no product welds this curl to it.

    ``BCX``/``BCY``/``BCZ`` take ``PERIODIC`` or ``METALLIC``, and the predicate
    refuses METALLIC on every axis — ``zero_metal_D`` runs inside this seam and this
    family does not carry it. The arms are transcribed anyway, because they are the
    certified body's and a body that dropped them would stop being a transcription.
    ``CNX``/``CNY``/``CNZ`` are the connection kinds of
    :func:`.complex_fields.connection_arguments` and ``cb`` the chunk record, whose
    presence word the curl half reads, both as the certified curl plan binds them.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # ======================= the curl half ================================
    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    # PERIODIC wraps (a plain roll, phased or not); METALLIC serves an exact 0.0 past
    # the wall -- a (+0.0, +0.0) word pair from `other=` IS the complex metallic ghost.
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

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
    if BACKWARD:
        if BCY == METALLIC:
            curl0_re = tl.where(at_y, 0.0, curl0_re)
            curl0_im = tl.where(at_y, 0.0, curl0_im)
        if BCZ == METALLIC:
            curl0_re = tl.where(at_z, 0.0, curl0_re)
            curl0_im = tl.where(at_z, 0.0, curl0_im)
        if BCX == METALLIC:
            curl1_re = tl.where(at_x, 0.0, curl1_re)
            curl1_im = tl.where(at_x, 0.0, curl1_im)
        if BCZ == METALLIC:
            curl1_re = tl.where(at_z, 0.0, curl1_re)
            curl1_im = tl.where(at_z, 0.0, curl1_im)
        if BCX == METALLIC:
            curl2_re = tl.where(at_x, 0.0, curl2_re)
            curl2_im = tl.where(at_x, 0.0, curl2_im)
        if BCY == METALLIC:
            curl2_re = tl.where(at_y, 0.0, curl2_re)
            curl2_im = tl.where(at_y, 0.0, curl2_im)
    else:
        if BCX == METALLIC:
            curl0_re = tl.where(at_x, 0.0, curl0_re)
            curl0_im = tl.where(at_x, 0.0, curl0_im)
        if BCY == METALLIC:
            curl1_re = tl.where(at_y, 0.0, curl1_re)
            curl1_im = tl.where(at_y, 0.0, curl1_im)
        if BCZ == METALLIC:
            curl2_re = tl.where(at_z, 0.0, curl2_re)
            curl2_im = tl.where(at_z, 0.0, curl2_im)

    # _apply_conductive_update: three separate complex64 in-place passes.
    v0_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
    v0_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
    if COND0:
        factor = tl.load(cf0 + idx, mask=live, other=0.0)
        v0_re, v0_im = _mul_field_left(v0_re, v0_im, factor)
    v0_re = v0_re - curl0_re
    v0_im = v0_im - curl0_im
    if COND0:
        inverse = tl.load(ci0 + idx, mask=live, other=0.0)
        v0_re, v0_im = _mul_field_left(v0_re, v0_im, inverse)

    v1_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
    v1_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
    if COND1:
        factor = tl.load(cf1 + idx, mask=live, other=0.0)
        v1_re, v1_im = _mul_field_left(v1_re, v1_im, factor)
    v1_re = v1_re - curl1_re
    v1_im = v1_im - curl1_im
    if COND1:
        inverse = tl.load(ci1 + idx, mask=live, other=0.0)
        v1_re, v1_im = _mul_field_left(v1_re, v1_im, inverse)

    v2_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
    v2_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
    if COND2:
        factor = tl.load(cf2 + idx, mask=live, other=0.0)
        v2_re, v2_im = _mul_field_left(v2_re, v2_im, factor)
    v2_re = v2_re - curl2_re
    v2_im = v2_im - curl2_im
    if COND2:
        inverse = tl.load(ci2 + idx, mask=live, other=0.0)
        v2_re, v2_im = _mul_field_left(v2_re, v2_im, inverse)

    # THE D STORE IS KEPT. The array path leaves the stepped displacement in the D
    # volume and the NEXT sub-step to read it is update_P (driver.py:3306), which
    # this launch does not span. Dropping the store would fuse away a value the run
    # still needs, which is a different defect from a wrong arithmetic line and one
    # no per-step E comparison would see.
    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)

    # ==================== the constitutive half ===========================
    # complex_stored_e_step, with `_subtract_complex_poles`'s two opening loads
    # replaced by the register pair the conductive tail just produced. Everything
    # else is verbatim: the pole chain in registration order, then ONE
    # `_mul_field_left` by the float32 inverse permittivity at the COMPLEX cell
    # index (`+ idx`, not `+ word`), then the store.
    word = 2 * idx

    s0_re, s0_im = _subtract_complex_poles_in_registers(
        v0_re, v0_im, a0, a1, a2, a3, a4, a5, a6, a7, word, live, NP0)
    o0_re, o0_im = _mul_field_left(
        s0_re, s0_im, tl.load(iv0 + idx, mask=live, other=0.0))
    tl.store(h0 + word, o0_re, mask=live)
    tl.store(h0 + word + 1, o0_im, mask=live)

    s1_re, s1_im = _subtract_complex_poles_in_registers(
        v1_re, v1_im, b0, b1, b2, b3, b4, b5, b6, b7, word, live, NP1)
    o1_re, o1_im = _mul_field_left(
        s1_re, s1_im, tl.load(iv1 + idx, mask=live, other=0.0))
    tl.store(h1 + word, o1_re, mask=live)
    tl.store(h1 + word + 1, o1_im, mask=live)

    s2_re, s2_im = _subtract_complex_poles_in_registers(
        v2_re, v2_im, q0, q1, q2, q3, q4, q5, q6, q7, word, live, NP2)
    o2_re, o2_im = _mul_field_left(
        s2_re, s2_im, tl.load(iv2 + idx, mask=live, other=0.0))
    tl.store(h2 + word, o2_re, mask=live)
    tl.store(h2 + word + 1, o2_im, mask=live)


def complex_conductive_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or the shared named-ImportError object on a laptop.

    Unlike the two COMPLEX pairs beside it, this module's kernel object is always
    present: :mod:`.complex_fields` hands out an ``_UnavailableKernel`` when Triton
    is absent, and launching one raises with the original import error. The
    accessor exists so callers have one spelling across the fused families.
    """
    return complex_conductive_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def complex_conductive_fused_pair_coverage(fields: Any, pml: Any,
                                           sources: Any = None) -> Coverage:
    """May ONE launch span complex conductive ``step_D`` -> complex ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said it
    — the construction
    :func:`.complex_fused_electric_pair.complex_fused_electric_pair_coverage` and
    :func:`.coverage.fused_pair_coverage` use, and the one the Metal sibling uses.
    """
    reasons: List[str] = []

    curl = complex_conductive_no_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = complex_stored_e_coverage(fields, pml)
    if not electric.covered:
        reasons.extend(f"E half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM, and for this family it is THE clause that costs nothing on
    # the measured corpus. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. A MAGNETIC source is
    # injected in the B/H half and does NOT disqualify the pair -- which is exactly
    # the polarity that makes this cell's four corpus rows reachable with
    # CARRIES_DEPOSIT_REPAIR at False.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294-3299), and on a conductive row through "
            f"_inject_electric_through_conductivity (driver.py:3296)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS. Both halves already refuse a mirror plane; this restates by
    # NAME what that refusal buys on THIS seam — a reader should not have to chase a
    # shared grid clause to learn that two driver passes sit between the halves on a
    # folded grid.
    if _call(grid, "has_symmetry", default=False):
        reasons.append(
            "a mirror plane is active: stepping.fill_symmetry_bc_D (driver.py:3300) "
            "and fill_folded_far_ghosts_D (:3302) both run inside this seam and "
            "neither is carried by this family")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    # THE WALL. zero_metal_D writes stored cell 0 of every D component whose Yee
    # shift on a walled axis is 0 (stepping.py:2206-2247) and update_E then reads
    # it. This family REFUSES the wall rather than carrying it, and the module
    # docstring prices that decision: a metallic axis buys zero seam-instances on
    # the measured corpus, where all four rows are periodic on all three axes.
    kinds = _boundary_kinds(grid, None)
    if kinds is None:
        # UNREADABLE IS A REFUSAL, NEVER A CRASH. `_boundary_kinds` returns None for
        # a grid the array path will not resolve, and a clause that iterated it
        # would raise where the contract says refuse.
        reasons.append(
            "the per-axis boundary rule is unreadable, so this family cannot "
            "establish that stepping.zero_metal_D (driver.py:3301) is inert")
        kinds = ()
    for axis, kind in enumerate(kinds):
        if kind == "metallic":
            reasons.append(
                f"axis {axis} is metallic: stepping.zero_metal_D runs inside this "
                f"seam (driver.py:3301) and this family does not carry it")

    # THE STENCIL, restated. The eighteen structurally unfusable D->E seam-instances
    # on this board are the OFF-DIAGONAL constitutive reading partner D volumes at
    # four indices (stepping.py:1235-1253) while step_D writes them in place. The E
    # half refuses that row already; it is named again because THIS is the fact that
    # makes the present cell buildable at all.
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append(
            "an off-diagonal chi1inv row makes update_E a STENCIL over the D "
            "volumes step_D writes in place (stepping.py:1235-1253), and no "
            "grid-wide barrier exists inside one launch")

    # THE POLE COUNTS MUST FIT the kernel's compiled slots. The E half already
    # refuses more than MAX_POLES; evaluated here as well so an over-ceiling
    # configuration is a refusal BY NAME rather than a ValueError at plan time, and
    # so the from-arrays route's ceiling and this one cannot drift apart.
    try:
        order = poles_per_component(fields)
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        return Coverage(False, tuple(dict.fromkeys(
            reasons + [f"the pole partition is unreadable ({exc!r})"])))
    for component, _displacement in E_TERMS:
        count = len(tuple(order.get(component, ())))
        if count > MAX_POLES:
            reasons.append(
                f"{component} is driven by {count} poles, more than the kernel's "
                f"MAX_POLES={MAX_POLES} compiled slots")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL, so `Fields` must be able
    # to hand one over per E component. An absent accessor would be a TypeError
    # inside the builder rather than a refusal, which is the wrong way for an
    # uncovered configuration to fail.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half "
            "cannot bind the inv_eps volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class ComplexConductiveFusedPairPlan:
    """One allocation-free launch for the driver's ``step_D`` and ``update_E``.

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_no_pml_conductive.ComplexConductiveNoPmlCurlPlan` and
    :class:`.complex_no_pml_stored_e.ComplexStoredEPlan` make, for the same reason.
    The ``condfac``/``condinv``/``inv_eps`` volumes are the exception and are bound
    RAW: each is float32 with one real coefficient per complex cell, indexed at
    ``+ idx``.

    THE POLE POINTERS ARE RESOLVED PER LAUNCH, never cached, exactly as
    ``ComplexStoredEPlan`` resolves them: ``PolarizationState.update`` rotates
    ``P``/``P_prev``/scratch, so a pointer captured at plan time would be one
    timestep stale. What IS cached is pole ORDER, which is bit-load-bearing and
    which :class:`.complex_no_pml_stored_e.LiveComplexPoleBinding` refuses to let
    change after planning.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "bc", "cond", "connections",
        "connection_values", "counts", "block", "num_warps",
        "chunk_bounds", "_targets", "_sources", "_condfac", "_condinv",
        "_e_targets", "_inv_eps", "_poles", "_cb", "_grid", "_kernel",
    )

    #: The driver call sites one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape: Sequence[int], dtdx: float,
                 codes: Sequence[int], phases: Sequence[Optional[complex]],
                 cond: Sequence[bool], block: int,
                 targets: Sequence[Any], sources: Sequence[Any],
                 condfac: Sequence[Any], condinv: Sequence[Any],
                 e_targets: Sequence[Any], inverse_epsilon: Sequence[Any],
                 poles: Any, kernel: Any = None,
                 num_warps: Optional[int] = 1, *, chunk_bounds: Any) -> None:
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # and Triton types a Python float argument as fp32, so the two are the same
        # bits (ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        self.cond = tuple(int(bool(flag)) for flag in cond)
        if len(self.cond) != 3:
            raise ValueError("a curl plan needs three per-target conductivity flags")
        # The certified curl plan's connection binding, for D's backward differences:
        # refused on an axis that does not wrap, as ComplexConductiveNoPmlCurlPlan
        # refuses it.
        connections, values = connection_arguments(tuple(phases),
                                                   backward=bool(self.backward))
        self.connections = tuple(int(kind) for kind in connections)
        for axis, (kind, code) in enumerate(zip(self.connections, self.bc)):
            if kind and code != getattr(PERIODIC, "value", PERIODIC):
                raise ValueError(f"axis {axis} carries a connection but its boundary "
                                 f"code {code} does not wrap")
        if any(kind not in CONNECTION_KINDS.values() for kind in self.connections):
            raise ValueError(f"unknown connection kinds {self.connections!r}")
        self.connection_values = tuple(float(value) for value in values)
        self.counts = tuple(int(value) for value in poles.counts)
        if len(self.counts) != len(E_TERMS):
            raise ValueError("a plan requires one pole count per E component")
        for (component, _), count in zip(E_TERMS, self.counts):
            if count > MAX_POLES:
                raise ValueError(
                    f"{component} has {count} poles; MAX_POLES={MAX_POLES}")
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        target_words = tuple(_word_view(array) for array in targets)
        self._targets = tuple(CupyPointer(array) for array in target_words)
        self._sources = tuple(CupyPointer(_word_view(array)) for array in sources)
        # A dead coefficient slot binds its target's float32 word view. The pointer
        # is correctly typed and COND=0 compiles every load away — the certified
        # curl plan's own convention, kept rather than re-invented.
        self._condfac = tuple(
            CupyPointer(_flat(array)) if array is not None
            else CupyPointer(target_words[index])
            for index, array in enumerate(condfac))
        self._condinv = tuple(
            CupyPointer(_flat(array)) if array is not None
            else CupyPointer(target_words[index])
            for index, array in enumerate(condinv))
        self._e_targets = tuple(CupyPointer(_word_view(a)) for a in e_targets)
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder "
                "binding would be read as a coefficient")
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._poles = poles
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "ComplexConductiveFusedPairPlan")
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in int32,
        # so the complex cell count must leave room for the doubling. Refused here
        # as well because the from-arrays route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which
        # compile deliberately broken copies of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING — every mutation leg would then launch the
        # shipped kernel and report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Resolve current P pointers and launch the fused pair, in place."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not the "
                    f"planned {self.counts[index]}")
            slots.extend(CupyPointer(_word_view(array)) for array in group)
            # The dead slots bind this component's D word view: correctly typed,
            # and NP compiles every load away. ComplexStoredEPlan's own convention.
            slots.extend([self._targets[index]] * (MAX_POLES - len(group)))
        kernel = (self._kernel if self._kernel is not None
                  else complex_conductive_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._sources, *self._condfac, *self._condinv,
            *self._e_targets, *self._inv_eps, *slots, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx, *self.connection_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            CNX=self.connections[0], CNY=self.connections[1], CNZ=self.connections[2],
            COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"ComplexConductiveFusedPairPlan(shape={self.shape}, bc={self.bc}, "
                f"cond={self.cond}, connections={self.connections}, "
                f"poles={self.counts}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_complex_conductive_fused_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None,
        chunk_layout: Any = None,
        ) -> Optional[ComplexConductiveFusedPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly. ``chunk_layout`` is the run's chunk division (``None``
    for the one-rank default); with no absorber every division binds the same
    words, as the certified curl plan binds them.
    """
    if not complex_conductive_fused_pair_coverage(fields, pml, sources).covered:
        return None

    grid = fields.grid
    spec = SUB_STEPS[CURL_SUB_STEP]
    targets = tuple(spec["targets"])
    # `None` for the PML argument, not `pml`: this family's own predicate has
    # already refused an active absorber, and `_boundary_kinds` consulted WITH one
    # would report the absorber's softening rather than the grid's declaration.
    kinds = _boundary_kinds(grid, None)
    phases = bloch_phase_table(grid, kinds)
    flags = conductive_no_pml_targets(fields, CURL_SUB_STEP)
    order = poles_per_component(fields)
    return ComplexConductiveFusedPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        phases, flags, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        source_arrays(fields, CURL_SUB_STEP),
        [fields.condfac_for(name) if flags[index] else None
         for index, name in enumerate(targets)],
        [fields.condinv_for(name) if flags[index] else None
         for index, name in enumerate(targets)],
        [getattr(fields, component) for component, _ in E_TERMS],
        [fields.inverse_epsilon_for(component) for component, _ in E_TERMS],
        LiveComplexPoleBinding(fields, order),
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, None, chunk_layout),
    )


def plan_complex_conductive_fused_pair_from_arrays(
        arrays: Dict[str, Any], poles: Dict[str, Sequence[Any]],
        phases: Sequence[Optional[complex]], dtdx: float, codes: Sequence[int],
        cond: Sequence[bool] = (True, True, True),
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, *,
        chunk_bounds: Any) -> ComplexConductiveFusedPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays`` is
    keyed by component name plus ``condfac_D*``/``condinv_D*`` and
    ``inv_eps_Ex``... (float32); ``poles`` is keyed by E component and holds the
    complex P volumes in registration order, and ``kernel=`` carries the mutation
    override — dropping it silently disarms every mutation leg. ``chunk_bounds`` has
    no default: the harness states the record it binds.
    """
    spec = SUB_STEPS[CURL_SUB_STEP]
    targets = tuple(spec["targets"])
    flags = tuple(bool(flag) for flag in cond)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return ComplexConductiveFusedPairPlan(
        shape, dtdx, codes, phases, flags,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets],
        [arrays[name] for name in spec["sources"]],
        [arrays[f"condfac_{name}"] if flags[index] else None
         for index, name in enumerate(targets)],
        [arrays[f"condinv_{name}"] if flags[index] else None
         for index, name in enumerate(targets)],
        [arrays[component] for component, _ in E_TERMS],
        [arrays["inv_eps_" + component] for component, _ in E_TERMS],
        StaticComplexPoleBinding([poles[component] for component, _ in E_TERMS]),
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
