"""Complex-field (force_complex_fields / Bloch k_point) PML curl + constitutive tranche.

WIRED, as of the plan_step integration. ``launch.plan_step`` calls
:func:`complex_pml_curl_coverage` and :func:`complex_constitutive_coverage`
through the launch-side forwarders and ``fastpath._decide`` composes through it,
with ``fastpath.ARM_CERTIFICATION`` carrying the 'complex PML' and 'complex' arms.
The four complex slots are dispatchable by default, and what keeps a configuration
out of a kernel is the refusal enumeration below. Every complex product is formed part by part (``part_products``), as MEEP
forms it and as the array path does, so no platform fact about a complex multiply
binds a kernel.

``fingerprints.json`` still carries no entry for this file — the byte gate binds
its own provenance record inside its results directory instead — so an edit here
invalidates no recorded certification through that mechanism, and the gate is the
only thing that can re-establish one.

WHAT THIS IS FOR. Seven corpus scripts demand it, and only seven: five Bloch-phase
runs (refl-angular.py, antenna_pec_ground_plane_1D.py, binary_grating_oblique.py,
mode_coeff_phase.py, oblique-planewave.py) and two complex-storage-at-k=0 runs
(wvg-src.py, solve-cw.py — only its complex-storage class; the mirror stays with
``symmetry.py``). None of the nine carries a susceptibility, a conductivity, BFAST
or chi2/chi3, which is why every one of those is refused outright rather than
carried. ``special_kz`` scripts (refl-angular-kz2d.py, parallel-wvgs-force.py) are
``grid.beta``, NOT ``bloch_phase``, and are refused by name.

TWO KERNELS, one per sub-step family, complex64 stepped as float32 WORD PAIRS
(complex64 is bit-layout (re, im) interleaved; a C-contiguous complex volume is a
C-contiguous float32 volume of shape ``(nx, ny, nz, 2)``, so complex cell ``w``
is words ``2*w`` and ``2*w + 1``):

* :func:`bloch_pml_curl_step` — complex ``stepping.step_B`` / ``step_D`` under
  split-field PML, with MEEP's Bloch connection applied to plane 0 of each
  UNSHIFTED operand;
* :func:`bloch_constitutive_step` — complex ``update_H`` / ``update_E`` (dsigw).
  The constitutive sub-step never sees the wrap, so it takes NO phase.

WHY THE PLANE TRANSCRIPTION IS LEGAL (from source): every recurrence on the
complex step path has real coefficients — curl (stepping.py:1648-1683),
split-field PML (:1952-1982), constitutive dsigw (:2112-2143) — and the imaginary
plane enters ONLY through the Bloch wrap multiply (:1767-1771 up / :1818-1822
down, both via ``_apply_bloch_phase`` :1846-1862) and through a complex source
amplitude, which is a ``sources``-side fact outside these kernels.
stepping.py:41-50 states this and adds the load-bearing invariant: ``inv_eps``
and every PML coefficient are float32 in BOTH storage modes (fields.py:571-573
against fields.py:1203-1204).

WHY IT IS NOT BYTE-TRIVIAL, and what the kernels step (MEEP's own forms, which
the array path steps too):

1. MEEP stores each complex field as two ``realnum`` arrays and steps them with
   REAL coefficients, part by part. Every real-coefficient product here is
   therefore the real scalar times each stored part (``part_products.scale``):
   ``c * f_re`` and ``c * f_im``, one rounding each, no cross terms and no fused
   multiply-add (``enable_fp_fusion=False``; the gate counts ``fma.f32`` in the PTX
   and requires zero).
2. The Bloch connection is MEEP's (boundaries.cpp:203-226): within
   ``part_products.CONNECT_TOLERANCE`` of +1 a copy, of -1 a negation per part,
   otherwise the unfused product with the phase rounded to float32 per part
   (:func:`connection_arguments`, :func:`_connect`). It applies to the not-owned
   word of the UNSHIFTED operand, ``conj(eikna)`` below for the forward (B)
   differences and ``eikna`` above for the backward (D) ones. The negation is
   spelled ``* -1.0``, never unary ``-``: Triton lowers ``-x`` as ``0.0 - x``, which
   gives ``+0`` for ``x = +0`` where MEEP's negation gives ``-0`` (semantic.py:386-391),
   and the composition probe's zero census measures that at stored bytes.
3. The curl, the split-field recurrence and the constitutive are
   :mod:`.forms`' MEEP forms, selected by the chunk record (``chunk_map``): the
   left-to-right stencil ``((g1s - g1) + g2) - g2s`` with the record's partner
   presence, the recurrence on the cells whose chunk carries the layer (the
   auxiliary read and written there only) and the dsig-only form elsewhere, and
   ``f + ((kps * src) - (kms * prev))`` on the accumulation's cells, ``src``
   elsewhere. ``update_H`` starts from ``B`` on MEEP's first step (``start_h``).

WHAT IS TRANSCRIBED, and from where (stepping.py unless noted):

* term table            ``B_CURL_TERMS`` / ``D_CURL_TERMS`` (:213-223)
* ghost rule            ``_shift_up`` (:1723) / ``_shift_down`` (:1787),
                        PERIODIC and METALLIC branches only
* Bloch connection      ``_bloch_ghost_operand`` / ``part_products.connect_parts``:
                        a copy, a negation or the unfused product on the
                        unshifted operand's plane 0; an unphased axis connects by
                        copy, which keeps k = 0 bit-identical to the plain engine
* curl grouping         ``_curl_from_operands`` / :func:`.forms.curl_stencil`
* ownership mask        ``_mask_non_owned_cells`` (:1865-1902)
* PML recurrence        ``_apply_pml_update`` (:1905-1935)
* constitutive dsigw    ``_apply_constitutive_pml`` (:2065-2096)
* coefficient pairing   ``_curl_coefficients`` (:2418) / ``_constitutive_coefficients``
                        (:2428); host-bound, never chosen in-kernel
* phase legality        ``_bloch_phases`` (:2313-2367): phased axis must resolve
                        PERIODIC; PML on a phased axis is ADMITTED (measured
                        2.82e-07); the module docstring at :119-125 contradicts
                        this and is stale — nothing here transcribes from it
* phase value           ``grid.bloch_phase`` (grid.py:1011-1017): exp(2*pi*i*k*L),
                        Brillouin edge EXACTLY -1+0j

GROUPING CHOICES THE GATE MUST HOLD (Triton gives no way to force them):

* No multiply-add contraction: ``enable_fp_fusion=False`` on every launch, and the
  gate reads ``fma.f32`` out of the PTX and requires none.
* The connection's negation is spelled ``* -1.0``; the composition probe's armed
  negation-mutation leg compiles the unary spelling back in and must see it diverge
  (on a state whose stencils are all zero, where the sign of a negated zero is the
  only thing that survives).
* The ``@triton.jit`` helpers (:func:`_connect` and :mod:`.forms`) are inlined by
  Triton; the transcription assumes inlining does not reassociate their bodies.
* The connection is computed on every lane and SELECTED onto plane 0 with
  ``tl.where`` (Triton has no per-lane scalar branch); ``where`` is a bitwise
  select, so every other lane keeps its loaded bits by construction, but any
  compiler transform merging the two paths is the gate's to catch.
* The curl's left-to-right grouping ``((g1s - g1) + g2) - g2s`` is held by the
  gate empirically, exactly as kernels.py declares for the real-field kernel; a
  Triton version bump is a correctness event for this file too.
* Word addressing computes ``2*idx`` in int32; the coverage predicates halve the
  int32 element bound to ``2*ncells < 2**31`` to compensate.
* Within one complex cell the two word stores (re then im) replace the array
  path's single complex store — non-semantic (no aliasing between planes).

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
            f"host coverage remains available without it: {self.error}") from self.error


try:  # The host predicate and public refusal path must work without Triton.
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - exercised by the absence test
    _TRITON_IMPORT_ERROR = _exc

    class _MissingTriton:
        @staticmethod
        def jit(function=None, **_options):
            if function is None:  # the @triton.jit(do_not_specialize=...) spelling
                return lambda inner: _UnavailableKernel(inner.__name__,
                                                        _TRITON_IMPORT_ERROR)
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
    _susceptibility_reasons,
)
from .launch import SUB_STEPS, CupyPointer, _flat

# Same constexpr codes as ``kernels.PERIODIC``/``kernels.METALLIC``. Importing
# kernels.py here would import Triton unconditionally and defeat this module's
# host-only coverage route (the same reason no_pml.py restates them).
PERIODIC = tl.constexpr(0)
METALLIC = tl.constexpr(1)

#: The connection kind of one axis (``part_products.connect_parts``), classified on
#: the host by :func:`connection_arguments`: none (an unphased axis, or a phase
#: within ``part_products.CONNECT_TOLERANCE`` of +1, which MEEP connects as a copy),
#: the unfused product with the phase's float32 parts, or a negation (a phase within
#: the tolerance of -1). Each is a compile-time branch of :func:`_connect`.
CONNECTION_KINDS: Dict[str, int] = {"none": 0, "phase": 1, "negate": 2}
CONNECT_NONE = tl.constexpr(CONNECTION_KINDS["none"])
CONNECT_PHASE = tl.constexpr(CONNECTION_KINDS["phase"])
CONNECT_NEGATE = tl.constexpr(CONNECTION_KINDS["negate"])

# The forms and the record layout, read by the kernels below. Imported only when
# Triton is: both modules import it at module scope, and this module's host route
# (the predicates, the plan builders to None) must answer without it.
if _TRITON_IMPORT_ERROR is None:
    from .forms import (axis_present, constitutive_form, curl_form, curl_stencil,
                        record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE

#: Elements per program — complex CELLS, not words. Restated from
#: ``kernels.DEFAULT_BLOCK`` for the same import reason as the boundary codes;
#: no autotune, because these kernels write their own inputs in place and the
#: tuner would silently apply the update dozens of times (kernels.py:56-59).
DEFAULT_BLOCK = 256

__all__ = [
    "bloch_pml_curl_step",
    "bloch_constitutive_step",
    "bloch_phase_table",
    "connection_arguments",
    "CONNECTION_KINDS",
    "complex_pml_curl_coverage",
    "complex_constitutive_coverage",
    "plan_complex_pml_curl",
    "plan_complex_pml_curl_from_arrays",
    "plan_complex_constitutive",
    "plan_complex_constitutive_from_arrays",
    "ComplexPmlCurlPlan",
    "ComplexConstitutivePlan",
]


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------

@triton.jit
def _rotate_field_left(g_re, g_im, p_re, p_im):
    """``complex<realnum>(phase) * word`` as MEEP's contraction-off build forms it.

    libc++'s ``operator*`` with every product and every sum rounded on its own
    (step.cpp:183-192; ``part_products.connect_parts``)::

        re = (p_re * g_re) - (p_im * g_im)
        im = (p_re * g_im) + (p_im * g_re)

    written field-left, which is the same word: IEEE multiplication and addition are
    each commutative on every non-NaN word.
    """
    out_re = (g_re * p_re) - (g_im * p_im)
    out_im = (g_re * p_im) + (g_im * p_re)
    return out_re, out_im


@triton.jit
def _mul_field_left(z_re, z_im, c):
    """A real coefficient times a complex word, one float32 product per part.

    ``part_products.scale``: MEEP's ``DOCMP`` loops multiply each stored part by the
    ``realnum`` coefficient once (step_db.cpp:47, update_eh.cpp:149), and the array
    path does the same. There are no zero cross terms to carry: MEEP never forms
    ``(c + 0i) * z``.
    """
    return z_re * c, z_im * c


@triton.jit
def _mul_coefficient_left(c, z_re, z_im):
    """:func:`_mul_field_left` with the coefficient spelled first: the same two words."""
    return c * z_re, c * z_im


@triton.jit
def _connect(v_re, v_im, c_re, c_im, KIND: tl.constexpr):
    """MEEP's not-owned word from its owned source, by the connection the host classified.

    ``part_products.connect`` (boundaries.cpp:359-366, step.cpp:183-192): within
    ``CONNECT_TOLERANCE`` of -1 a unary minus per part, spelled ``* -1.0`` because
    Triton lowers ``-x`` as ``0.0 - x``, which gives ``+0`` for ``x = +0`` where the
    negation gives ``-0``; otherwise the unfused product with the phase's float32 parts
    ``(c_re, c_im)`` (:func:`_rotate_field_left`). A connection within the tolerance of
    +1 is a copy, which the kernel never calls this for (:data:`CONNECT_NONE`).
    """
    if KIND == CONNECT_NEGATE:
        out_re = v_re * -1.0
        out_im = v_im * -1.0
    else:
        out_re = (c_re * v_re) - (c_im * v_im)
        out_im = (c_re * v_im) + (c_im * v_re)
    return out_re, out_im


@triton.jit
def bloch_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz or Dx,Dy,Dz (complex64 as words)
    u0, u1, u2,                       # auxiliaries: fu_B* or fu_D* (complex64 as words)
    g0, g1, g2,                       # sources: Ex,Ey,Ez or Hx,Hy,Hz (complex64 as words)
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, float32, one Yee sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kind per axis
    BLOCK: tl.constexpr,
):
    """One complex curl sub-step of all three components, both PML forms included.

    :func:`.kernels.pml_curl_step` run once per stored part, with ONE addition: the
    Bloch connection of the stored window. Every field access is a float32 word pair
    at ``2*idx`` / ``2*idx + 1``, and every product has a real operand, so each part
    takes the real kernel's forms (:mod:`.forms`) with the same coefficient words and
    the same record selects -- MEEP's ``DOCMP`` loop, which steps ``f[c][0]`` and
    ``f[c][1]`` with real operands alone (update_eh.cpp:149, step_db.cpp:47).

    THE CONNECTION (``stepping._curl_operands`` / ``_bloch_ghost_operand``). On a
    phased axis the shifted operand is a plain roll of the stored array -- the word one
    stored cell over is always an owned word of the stored window -- and the UNSHIFTED
    operand's plane 0 along the partner's derivative axis is replaced by MEEP's
    not-owned word: ``conj(eikna)`` times stored cell 0 under the B side's forward
    differences, ``eikna`` times it under the D side's backward ones. The host
    classifies each axis once (:func:`connection_arguments`): no connection
    (``CN* = 0``, unphased or a copy), a product (1) or a negation (2). The two
    unshifted partners of one target connect on different axes, so each target forms
    its own connected copy of its two partners.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

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


@triton.jit(do_not_specialize=["start_h"])
def bloch_constitutive_step(
    f0, f1, f2,                       # targets: Hx,Hy,Hz or Ex,Ey,Ez (complex64 as words)
    w0, w1, w2,                       # auxiliaries: f_w_H* or f_w_E* (complex64 as words)
    g0, g1, g2,                       # sources: Bx,By,Bz or Dx,Dy,Dz (complex64 as words)
    e0, e1, e2,                       # inverse epsilon, float32 VOLUMES, E side only
    kp0, km0, kp1, km1, kp2, km2,     # kps/kms on each component's OWN axis, float32
    cb,                               # chunk record: the forms' dsigw membership
    nx, ny, nz, n_elem, start_h,      # n_elem = COMPLEX cells; start_h: MEEP's start of H
    SCALE: tl.constexpr,              # 0 = H (source is B), 1 = E (source is D*inv_eps)
    BLOCK: tl.constexpr,
):
    """One complex constitutive sub-step, both forms, all three components.

    :func:`.kernels.constitutive_step` run once per stored part. The record's
    membership line for the component's own (dsigw) axis, on its own sub-lattice,
    picks the form (:func:`.forms.constitutive_form`): MEEP's accumulation inside,
    ``f = src`` outside, with ``f_w`` read and written inside only, ``prev`` read
    before ``f_w`` is written, and MEEP's start of H on the run's first H launch
    (``start_h``, as the real kernel carries it). ``src`` is B on the H side and ``D *
    inv_eps`` per part on the E side (``part_products.scale``): ``inv_eps`` is one
    float32 word per COMPLEX cell, indexed by the complex cell, never word-doubled.
    No neighbour is read, so there is no ghost rule, no ownership mask and no
    connection: the constitutive sub-step never sees the wrap.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    if SCALE:
        lattice = RECORD_HALF_INTEGER
    else:
        lattice = 0
    dw0 = record_in(cb, lattice + 0, i)
    dw1 = record_in(cb, lattice + 4, j)
    dw2 = record_in(cb, lattice + 8, k)
    # MEEP's start of H; always 0 on the E side.
    start = start_h != 0

    # Component 0 takes its coefficient from axis x, 1 from y, 2 from z (dsigw).
    kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
    km_0 = tl.load(km0 + i, mask=live, other=0.0)
    kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
    km_1 = tl.load(km1 + j, mask=live, other=0.0)
    kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
    km_2 = tl.load(km2 + k, mask=live, other=0.0)

    # --- component 0 -----------------------------------------------------------
    prev_re = tl.load(w0 + 2 * idx, mask=live & dw0, other=0.0)   # BEFORE the store.
    prev_im = tl.load(w0 + 2 * idx + 1, mask=live & dw0, other=0.0)
    src_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    src_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    if SCALE:
        ie = tl.load(e0 + idx, mask=live, other=0.0)
        src_re = src_re * ie
        src_im = src_im * ie
    tl.store(w0 + 2 * idx, src_re, mask=live & dw0)
    tl.store(w0 + 2 * idx + 1, src_im, mask=live & dw0)
    a_re = tl.load(f0 + 2 * idx, mask=live & dw0, other=0.0)
    a_im = tl.load(f0 + 2 * idx + 1, mask=live & dw0, other=0.0)
    if SCALE == 0:                    # H starts as a copy of B on its own cells
        prev_re = tl.where(start, src_re, prev_re)
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
    tl.store(f0 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_0, km_0, dw0),
             mask=live)
    tl.store(f0 + 2 * idx + 1, constitutive_form(a_im, prev_im, src_im, kp_0, km_0, dw0),
             mask=live)

    # --- component 1 -----------------------------------------------------------
    prev_re = tl.load(w1 + 2 * idx, mask=live & dw1, other=0.0)
    prev_im = tl.load(w1 + 2 * idx + 1, mask=live & dw1, other=0.0)
    src_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    src_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    if SCALE:
        ie = tl.load(e1 + idx, mask=live, other=0.0)
        src_re = src_re * ie
        src_im = src_im * ie
    tl.store(w1 + 2 * idx, src_re, mask=live & dw1)
    tl.store(w1 + 2 * idx + 1, src_im, mask=live & dw1)
    a_re = tl.load(f1 + 2 * idx, mask=live & dw1, other=0.0)
    a_im = tl.load(f1 + 2 * idx + 1, mask=live & dw1, other=0.0)
    if SCALE == 0:
        prev_re = tl.where(start, src_re, prev_re)
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
    tl.store(f1 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_1, km_1, dw1),
             mask=live)
    tl.store(f1 + 2 * idx + 1, constitutive_form(a_im, prev_im, src_im, kp_1, km_1, dw1),
             mask=live)

    # --- component 2 -----------------------------------------------------------
    prev_re = tl.load(w2 + 2 * idx, mask=live & dw2, other=0.0)
    prev_im = tl.load(w2 + 2 * idx + 1, mask=live & dw2, other=0.0)
    src_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    src_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    if SCALE:
        ie = tl.load(e2 + idx, mask=live, other=0.0)
        src_re = src_re * ie
        src_im = src_im * ie
    tl.store(w2 + 2 * idx, src_re, mask=live & dw2)
    tl.store(w2 + 2 * idx + 1, src_im, mask=live & dw2)
    a_re = tl.load(f2 + 2 * idx, mask=live & dw2, other=0.0)
    a_im = tl.load(f2 + 2 * idx + 1, mask=live & dw2, other=0.0)
    if SCALE == 0:
        prev_re = tl.where(start, src_re, prev_re)
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
    tl.store(f2 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_2, km_2, dw2),
             mask=live)
    tl.store(f2 + 2 * idx + 1, constitutive_form(a_im, prev_im, src_im, kp_2, km_2, dw2),
             mask=live)


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# The clause numbering deliberately mirrors ``coverage._grid_reasons`` so the two
# can be diffed. Clause 2 is INVERTED (this product REQUIRES complex64 storage,
# where every shipped kernel requires real) and clause 7 (k = 0) is REPLACED by
# the per-axis phase-consistency clause: has_bloch is ADMITTED, and with no phase
# anywhere the kernel must reduce to the plain complex path (PH* = 0), which the
# gate's unphased rows measure. ``coverage._grid_reasons``/``pml_curl_coverage``/
# ``constitutive_coverage`` are NOT reusable here — their clauses 2 and 7 refuse
# this module's whole domain — and ``_layout_reasons``/``_volume_reasons`` pin
# float32, so the complex twins below exist rather than a widened shared check.


def _complex_volume_reasons(label: str, array: Any, shape: Sequence[int]) -> List[str]:
    """Shape, dtype and contiguity of one COMPLEX volume, reported by name."""
    out: List[str] = []
    if tuple(getattr(array, "shape", ())) != tuple(shape):
        out.append(f"{label} shape {tuple(getattr(array, 'shape', ()))!r} != grid shape "
                   f"{tuple(shape)!r}")
    if str(getattr(array, "dtype", None)) != "complex64":
        out.append(f"{label} dtype {getattr(array, 'dtype', None)} is not complex64 "
                   f"(this product steps complex storage as float32 word pairs)")
    flags = getattr(array, "flags", None)
    if not bool(getattr(flags, "c_contiguous", False)):
        out.append(f"{label} is not C-contiguous")
    return out


def _complex_layout_reasons(fields: Any, shape: Sequence[int],
                            names: Sequence[str]) -> List[str]:
    """Every named field volume must be complex64, C-contiguous, ``grid.shape``.

    The int32 bound is HALVED against the real path's (coverage.py:622-626):
    word offsets are ``2*idx`` in int32, so ``2*ncells`` is what must stay below
    ``2**31``, not ``ncells``.
    """
    out: List[str] = []
    if len(shape) != 3:
        return [f"grid shape {tuple(shape)!r} is not three-dimensional"]
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if 2 * total >= 2 ** 31:
        out.append(f"{total} cells is {2 * total} float32 words, which exceeds the "
                   f"kernel's int32 word-offset range (2*ncells must stay below 2**31)")
    for name in names:
        array = getattr(fields, name, None)
        if array is None:
            continue  # Already reported by the allocation clause.
        out.extend(_complex_volume_reasons(name, array, shape))
    return out


def _complex_grid_reasons(fields: Any, pml: Any, grid: Any, *,
                          require_active_pml: bool = True) -> List[str]:
    """The clauses the complex predicates share.

    ``require_active_pml`` selects which SIDE of clause 3 this caller wants, and
    that clause alone: True demands an absorber that absorbs (the split-field
    product in this module), False demands its absence (the degenerate
    no-absorber product in :mod:`complex_no_pml_curl`). The two are a PARTITION
    of ``stepping._pml_is_active``, which is the same test the array path
    branches on at stepping.py:508 — so no configuration can be admitted by both
    and none can fall between them, and the loss mode a second admitter would
    cause (an UNSELECTED slot falling silently back to the array path) cannot
    arise. Every other clause is shared verbatim, deliberately: a duplicated
    copy of clause 5's phase logic is two things that must be kept in step
    forever, and the 2026-08-11 coverage audit is what that costs.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernels launch against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2 (INVERTED). Complex64 storage REQUIRED. A real run belongs to the shipped
    #    real-field kernels; real storage read as float32 word pairs of a complex
    #    array is the wrong-stride wrong answer in the other direction. Either
    #    force_complex_fields or a nonzero k_point puts the run here (the array
    #    path itself raises on real storage with a phase, stepping.py:1893-1908);
    #    the storage dtype itself is verified by the layout clause.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32 (neither force_complex_fields nor a nonzero "
            "k_point): a real run belongs to the shipped real-field kernels")

    # 3. An absorber that actually absorbs: this is the split-field product only.
    #    A plain complex curl (pml=None) is a separate product and is refused.
    #    UNDER ``require_active_pml=False`` this clause is INVERTED and nothing
    #    else changes — see the docstring for why that is a partition.
    active = pml is not None and bool(getattr(pml, "is_active", False))
    if require_active_pml:
        if not active:
            reasons.append("no active PML layer (this product implements the complex "
                           "split-field path only)")
    elif active:
        reasons.append("an active PML layer is installed (that is the complex "
                       "split-field product's path, not this one)")

    # 4. Only the two ghost rules the curl kernel writes; a fold or the
    #    cylindrical axis changes the stored extent for the constitutive side too.
    kinds = _boundary_kinds(grid, pml if (pml is not None
                                          and getattr(pml, "is_active", False))
                            else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside {COVERED_BOUNDARIES}")
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 5. Per-axis phase consistency. A phased axis must resolve PERIODIC
    #    (stepping._bloch_phases raises there, stepping.py:2401-2415; the grid
    #    guarantees it, grid.py:973-1001 — re-checked because a mis-bound PH
    #    constexpr is a plane of wrong values, not a crash), and a metallic axis
    #    must carry a k component of exactly 0. has_bloch FALSE IS ADMITTED: with
    #    PH*=0 the kernel must reduce to the plain complex path, and the gate's
    #    unphased rows are that reduction measured. The phase is read
    #    DEFENSIVELY, not through ``_call``: an unreadable phase is not an
    #    unphased one (the rule _susceptibility_reasons applies to driven(),
    #    coverage.py:224-227), and admitting one would let ``bloch_phase_table``
    #    raise into the plan builder, violating its None-only refusal contract
    #    (demonstrated by the 2026-08-11 coverage audit).
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if kinds is not None:
        phase_reader = getattr(grid, "bloch_phase", None)
        if not callable(phase_reader):
            reasons.append(
                "grid.bloch_phase is missing or not callable; an unreadable "
                "phase table is not an unphased one")
        for axis, kind in enumerate(kinds):
            if callable(phase_reader):
                try:
                    phase = phase_reader(axis)
                except Exception as exc:  # noqa: BLE001 - unreadable is refused
                    reasons.append(
                        f"grid.bloch_phase({axis}) raised {exc!r}; an unreadable "
                        f"phase is not an unphased one")
                else:
                    if phase is not None and kind != "periodic":
                        reasons.append(
                            f"axis {axis} carries Bloch phase {phase!r} but "
                            f"resolved to {kind!r}; only a periodic wrap can "
                            f"carry a phase")
            if kind == "metallic" and float(k_point[axis]) != 0.0:
                reasons.append(
                    f"axis {axis} is metallic with k component {k_point[axis]!r}; "
                    f"a PEC wall gives the axis no lattice vector for the phase "
                    f"(stepping._bloch_phases raises on the same pairing)")

    # 6. No conductivity anywhere on the curl targets: conductive complex is a
    #    separate product (stepping.py:2001-2109), with no corpus demand. A
    #    missing or non-callable reader is refused OUTRIGHT — deliberately
    #    stricter than the shipped predicate's magnetic-flag fallback
    #    (coverage.py:275-291): inferring "no conductivity" from the ABSENCE of
    #    condfac_for is admission by attribute absence, the exact reasoning
    #    coverage exists to refuse, and an electric conductivity has no fallback
    #    flag at all (demonstrated by the 2026-08-11 coverage audit; the real
    #    Fields always defines the reader, fields.py:832, so no engine-built
    #    configuration changes verdict).
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append(
            "fields does not expose condfac_for; an unreadable conductivity "
            "table is not an absent one")
    else:
        for target in CURL_TARGETS:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}; conductive complex "
                    f"stepping is a separate, unbuilt product")

    # 7. No dispersion, refused OUTRIGHT: none of the nine demand scripts carries
    #    a susceptibility, and complex ADE belongs to a future tranche. The shape
    #    clauses still run so an unreadable susceptibility is named, not shrugged at.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: complex-storage ADE is a future "
            "tranche, and no corpus script combines dispersion with complex fields")
    reasons.extend(_susceptibility_reasons(fields))

    # 8. No instantaneous nonlinearity (Pade factor + the DOCMP split,
    #    stepping.py:999-1000, :1080-1116), no BFAST (:392-396, complex cast
    #    :923-924), no special_kz beta (:754-811 — refl-angular-kz2d and
    #    parallel-wvgs-force are grid.beta, NOT bloch_phase, grid.py:378-383).
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not carried)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
                       f"(that is grid.beta, not a Bloch phase)")

    # 9. Stored E. Under an active PML always true (fields.py:678-700), but the
    #    clause is the REASON: an edit that makes stores_E optional under PML
    #    must be caught here, exactly as coverage.py:296-304 argues. Under
    #    ``require_active_pml=False`` it is NOT automatic and does real work: with
    #    E unstored, stepping._read_component derives D * inv_eps into a scratch
    #    buffer (stepping.py:2440-2449) instead of differencing storage, which is
    #    no_pml.py's DERIVE product and is not transcribed here.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return reasons


def complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May the complex split-field PML curl kernel step this (fields, pml, sub_step)?

    Positive clauses only; a failing clause appends its reason and the scan
    continues. Off-diagonal epsilon is ADMITTED here (constitutive-only, its
    whole effect is inside ``update_E``, stepping.py:1001-1008) and refused by the
    E-side constitutive predicate — the same per-sub-step split coverage.py makes.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _complex_grid_reasons(fields, pml, grid)

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        # Both sub-lattices, as pml_curl_coverage checks (coverage.py:316-317):
        # the suffix the plan binds is the sub-step's own, and a swap is the
        # half-cell mutation the gate carries.
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def complex_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the complex constitutive kernel step ``update_H`` ('H') / ``update_E`` ('E')?

    The E side additionally refuses everything that changes what ``source`` is:
    a registered polarization is already refused for the whole module (clause 7),
    and an off-diagonal chi1inv row makes the sub-step non-element-wise
    (stepping.py:1001-1008), so it is refused HERE and admitted by the curl.
    With no polarization, ``displacement_minus_polarization`` returns the D
    array itself (fields.py:1096-1098), so binding D directly IS the array
    path's source.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _complex_grid_reasons(fields, pml, grid)

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed (the row product reads "
            "neighbours; this sub-step is element-wise)")

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


# ---------------------------------------------------------------------------
# The phase table and its argument encoding
# ---------------------------------------------------------------------------

def bloch_phase_table(grid: Any, kinds: Sequence[str]) -> Tuple[Optional[complex],
                                                                Optional[complex],
                                                                Optional[complex]]:
    """Per-axis Bloch wrap factor, None on an unphased axis — never ``1+0j``.

    Resolved through ``grid.bloch_phase`` exactly as ``stepping._bloch_phases``
    resolves it (stepping.py:2398-2416): the grid value is exp(2*pi*i*k*L) with
    the Brillouin edge EXACTLY -1+0j (grid.py:1011-1017), and None means the
    multiply is SKIPPED — the constexpr skip is the bit-identity of k = 0.
    The non-periodic check mirrors the array path's raise and is unreachable
    behind an admitting predicate; it is kept so a caller who skips the
    predicate is refused loudly rather than handed a mis-bound PH constexpr.
    """
    if not getattr(grid, "has_bloch", False):
        return (None, None, None)
    phases: List[Optional[complex]] = []
    for axis in range(3):
        phase = grid.bloch_phase(axis)
        if phase is not None and kinds[axis] != "periodic":
            raise ValueError(
                f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                f"{kinds[axis]!r}; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same configuration)")
        phases.append(phase)
    return (phases[0], phases[1], phases[2])


def _phase_arguments(phases: Sequence[Optional[complex]],
                     backward: bool) -> Tuple[Tuple[int, int, int],
                                              Tuple[float, ...]]:
    """Encode the phase table as (PH flags, six float32 components).

    The phase is rounded to complex64 BEFORE splitting — the array path's
    ``shifted.dtype.type(phase)`` (stepping.py:1909) — and for the backward
    sub-step the imaginary part is NEGATED, which is the conjugate of
    ``_shift_down`` (stepping.py:1865-1869); negation is exact, so the order of
    conjugation and rounding is immaterial. An unphased axis passes (1.0, 0.0)
    under PH = 0, and the kernel emits no multiply for it at all.
    """
    import numpy  # noqa: PLC0415
    flags: List[int] = []
    values: List[float] = []
    for phase in phases:
        if phase is None:
            flags.append(0)
            values.extend((1.0, 0.0))
            continue
        rounded = numpy.complex64(phase)
        real = float(numpy.float32(rounded.real))
        imag = float(numpy.float32(rounded.imag))
        if backward:
            imag = -imag
        flags.append(1)
        values.extend((real, imag))
    return (flags[0], flags[1], flags[2]), tuple(values)


def connection_arguments(phases: Sequence[Optional[complex]],
                         backward: bool) -> Tuple[Tuple[int, int, int],
                                                  Tuple[float, ...]]:
    """MEEP's connection of each axis's not-owned word: ``(kinds, six float32 parts)``.

    ``stepping._bloch_ghost_operand``, per axis, for one difference direction. The
    forward (B) differences read the word half a cell BELOW the stored window, so the
    factor is ``conj(eikna)``; the backward (D) ones read the word ABOVE it, ``eikna``
    (``below = not backward``). MEEP forms ``thephase = 1.0 * factor`` in
    ``complex<double>`` (boundaries.cpp:203-226; ``readback.cmul``) and connects by
    ``part_products.connect_parts``: within ``CONNECT_TOLERANCE`` of +1 a copy
    (:data:`CONNECT_NONE`: the stored word itself), of -1 a negation
    (:data:`CONNECT_NEGATE`), otherwise the unfused product with the phase rounded to
    the part dtype (:data:`CONNECT_PHASE`, the parts rounded to float32 here so the
    launch passes exactly representable words). An unphased axis is
    :data:`CONNECT_NONE` with ``(1.0, 0.0)``.
    """
    import numpy  # noqa: PLC0415

    from ..part_products import CONNECT_TOLERANCE  # noqa: PLC0415
    from ..readback import cmul  # noqa: PLC0415

    kinds: List[int] = []
    values: List[float] = []
    for phase in phases:
        if phase is None:
            kinds.append(CONNECTION_KINDS["none"])
            values.extend((1.0, 0.0))
            continue
        eikna = complex(phase)
        below = not backward
        factor = (eikna.real, -eikna.imag) if below else (eikna.real, eikna.imag)
        pr, pi = cmul((1.0, 0.0), factor)
        if abs(pi) < CONNECT_TOLERANCE and abs(pr - 1.0) < CONNECT_TOLERANCE:
            kinds.append(CONNECTION_KINDS["none"])
            values.extend((1.0, 0.0))
        elif abs(pi) < CONNECT_TOLERANCE and abs(pr + 1.0) < CONNECT_TOLERANCE:
            kinds.append(CONNECTION_KINDS["negate"])
            values.extend((-1.0, 0.0))
        else:
            kinds.append(CONNECTION_KINDS["phase"])
            values.extend((float(numpy.float32(pr)), float(numpy.float32(pi))))
    return (kinds[0], kinds[1], kinds[2]), tuple(values)


def _word_view(array: Any) -> Any:
    """The float32 word view of one complex64 volume — the pointer the kernel gets.

    ``complex64`` is bit-layout (re, im) interleaved, so a C-contiguous complex
    volume viewed as float32 is the SAME allocation with a doubled last axis;
    no copy, no move, and the engine's own references stay live. Anything that
    is not a C-contiguous complex64 volume is refused at plan time — a strided
    view would be read in the wrong order by the flat word index.
    """
    import numpy  # noqa: PLC0415
    if str(getattr(array, "dtype", None)) != "complex64":
        raise ValueError(f"expected a complex64 volume, got dtype "
                         f"{getattr(array, 'dtype', None)!r}")
    if not bool(getattr(getattr(array, "flags", None), "c_contiguous", False)):
        raise ValueError("complex volume is not C-contiguous; its float32 word "
                         "view would not be either")
    return array.view(numpy.float32)


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class ComplexPmlCurlPlan:
    """A launchable, allocation-free COMPLEX split-field PML curl sub-step.

    Same two construction routes and one launch route as ``launch.PmlCurlPlan``:
    :func:`plan_complex_pml_curl` from the engine's objects through the
    predicate, :func:`plan_complex_pml_curl_from_arrays` from bare device arrays
    for the gate (including the deliberately wrong ones), both launched through
    :meth:`run` so the bytes the gate certifies are the bytes the engine would
    launch. The complex volumes are bound as float32 WORD VIEWS here, once, at
    plan time; ``n_elem`` stays the COMPLEX cell count.

    ``connections`` / ``connection_values`` are :func:`connection_arguments`' kinds
    and float32 parts for this sub-step's direction. ``chunk_bounds`` is the chunk
    record, required and bound as ``launch.PmlCurlPlan`` binds it
    (``launch.bind_chunk_bounds``).
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
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path scales each part by the float32-rounded scalar
        # (part_products.scale); Triton types a Python float argument as fp32, so the
        # two are the same bits.
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.connections = tuple(int(kind) for kind in connections)
        if len(self.connections) != 3 or any(
                kind not in CONNECTION_KINDS.values() for kind in self.connections):
            raise ValueError(f"connections must be three kinds in "
                             f"{sorted(CONNECTION_KINDS.values())}, got {connections!r}")
        for axis, (kind, code) in enumerate(zip(self.connections, self.bc)):
            # A connection exists only on a wrapping axis (stepping._bloch_ghost_operand
            # returns the field itself otherwise): on a wall the kernel would connect a
            # stored plane the array path reads unconnected.
            if kind and code != getattr(PERIODIC, "value", PERIODIC):
                raise ValueError(f"axis {axis} carries a connection but its boundary "
                                 f"code {code} does not wrap")
        self.connection_values = tuple(float(value) for value in connection_values)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "ComplexPmlCurlPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # The override exists for exactly one caller: the gate's mutation legs,
        # which compile deliberately broken copies of the shipped kernel.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. ``guard`` is the gate's, not a caller's."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else bloch_pml_curl_step
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
        return (f"ComplexPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, connections={self.connections}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                          block: Optional[int] = None,
                          num_warps: Optional[int] = None,
                          chunk_layout: Any = None) -> Optional[ComplexPmlCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal (Y-style: never raise into a caller that would
    otherwise have stepped correctly). ``chunk_layout`` is the run's chunk division,
    ``None`` for the one-rank default (``launch.chunk_bounds_for``).
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not complex_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .launch import chunk_bounds_for  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    phases = bloch_phase_table(grid, kinds)
    connections, values = connection_arguments(phases, backward=bool(spec["backward"]))
    return ComplexPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        connections, values,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_complex_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                      flat: Dict[str, Any], codes,
                                      phases: Sequence[Optional[complex]],
                                      dtdx: float,
                                      block: Optional[int] = None,
                                      kernel: Any = None,
                                      num_warps: Optional[int] = None,
                                      *, chunk_bounds: Any,
                                      ) -> ComplexPmlCurlPlan:
    """Build a plan from bare device arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name (complex64 volumes), ``flat`` by
    ``kms_x``/``sinv_x``/... on the sub-lattice the caller already selected,
    ``codes`` is the 0/1 periodic/metallic triple, ``phases`` the per-axis
    Optional[complex] table (None = unphased; the direction's connection factor is
    formed HERE, per sub-step, exactly as the engine route forms it) and
    ``chunk_bounds`` the chunk record, with NO default: a harness plan built without
    one is refused by name (``launch.bind_chunk_bounds``), because a mutation leg
    that fell back to some default division would stop measuring the forms it
    exists to measure. No
    predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones, and ``kernel=`` carries the
    mutation override.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(spec["backward"]))
    return ComplexPmlCurlPlan(
        sub_step, shape, dtdx, codes, connections, values,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )


class ComplexConstitutivePlan:
    """A launchable, allocation-free COMPLEX ``dsigw`` constitutive sub-step.

    The Yee sub-lattice is chosen at build time and nowhere else — integer
    (``kps_a``/``kms_a``) for H, half-integer (``_h``) for E — and the kernel never
    asks which lattice its six pointers came from, so the swap stays the gate's
    half-cell mutation. The H side binds the source word views as inverse-epsilon
    placeholders, the same device as ``launch.ConstitutivePlan``: the loads sit
    behind the ``SCALE`` constexpr and are compiled away, but a pointer argument still
    has to type.

    ``chunk_bounds`` is the chunk record, required (``launch.bind_chunk_bounds``).
    ``h_state`` is the ``Fields`` the H side starts from, or ``None``, with
    ``launch.ConstitutivePlan``'s contract: :meth:`run` passes ``start_h = 1`` while
    ``h_state.h_separated`` is false and sets the flag after the first launch that has
    programs (a warm pass over an empty grid starts nothing); ``None`` -- the
    from-arrays route, the gates -- is a state that has already started.
    """

    __slots__ = ("side", "shape", "n_elem", "scale", "block",
                 "num_warps", "chunk_bounds", "_targets", "_aux", "_sources",
                 "_inv_eps", "_coefficients", "_cb", "_grid", "_kernel", "_h_state")

    def __init__(self, side: str, shape, block: int,
                 targets, auxiliaries, sources, inverse_epsilon, coefficients,
                 kernel=None, num_warps: Optional[int] = None, *,
                 chunk_bounds: Any, h_state: Any = None) -> None:
        if side not in CONSTITUTIVE_SIDES:
            raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.side = side
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.scale = 1 if side == "E" else 0
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        # inv_eps volumes stay float32 and are indexed by the COMPLEX cell index
        # in-kernel — bound directly, never word-viewed.
        self._inv_eps = (tuple(CupyPointer(a) for a in inverse_epsilon)
                         if inverse_epsilon is not None else self._sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "ComplexConstitutivePlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel
        self._h_state = h_state if side == "H" else None

    def start_h(self) -> int:
        """1 on the one H launch that starts H from B, 0 otherwise (and always on E)."""
        state = self._h_state
        return 0 if state is None or state.h_separated else 1

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as the curl plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else bloch_constitutive_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        start = self.start_h()
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._inv_eps,
            *self._coefficients, self._cb,
            nx, ny, nz, self.n_elem, start,
            SCALE=self.scale,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        if start and self._grid[0]:  # an empty grid (the warm pass) starts nothing
            self._h_state.h_separated = True

    def __repr__(self) -> str:
        return (f"ComplexConstitutivePlan({self.side}, shape={self.shape}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_complex_constitutive(fields: Any, pml: Any, side: str,
                              block: Optional[int] = None,
                              num_warps: Optional[int] = None,
                              chunk_layout: Any = None
                              ) -> Optional[ComplexConstitutivePlan]:
    """Build a complex constitutive plan from the engine's objects, or None.

    With no polarization registered (clause 7 refused any), the E source IS the
    D array — ``displacement_minus_polarization`` returns it unchanged — so the
    binding below is the array path's source, not an optimisation of it.
    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default;
    the H side starts from ``fields`` (MEEP's start of H).
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not complex_constitutive_coverage(fields, pml, side).covered:
        return None
    from .launch import chunk_bounds_for  # noqa: PLC0415

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


def plan_complex_constitutive_from_arrays(side: str, arrays: Dict[str, Any],
                                          flat: Dict[str, Any],
                                          block: Optional[int] = None,
                                          kernel: Any = None,
                                          num_warps: Optional[int] = None,
                                          *, chunk_bounds: Any,
                                          ) -> ComplexConstitutivePlan:
    """Build a complex constitutive plan from bare device arrays — the gate's route.

    ``arrays`` is keyed by component name (plus ``inv_eps_Ex``... on the E side,
    float32), ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE CALLER
    selected and ``chunk_bounds`` the chunk record, with no default (see
    :func:`plan_complex_pml_curl_from_arrays`). No start of H: a harness state has
    already started. ``kernel=`` carries the mutation override; dropping it silently
    disarms every mutation leg.
    """
    spec = CONSTITUTIVE_SIDES[side]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return ComplexConstitutivePlan(
        side, shape,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays[name] for name in spec["aux"]],
        [arrays[name] for name in spec["sources"]],
        ([arrays["inv_eps_" + name] for name in spec["targets"]]
         if side == "E" else None),
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )


# ---------------------------------------------------------------------------
# WIRING — live, and what still holds the line
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` consults these predicates through its own forwarders
# (launch.py:232-257 import them lazily; the four complex arms are selected at
# :2045 step_B/step_D, :2147 update_H, :2223 update_E).
# ``fastpath.ARM_CERTIFICATION`` carries 'complex PML' and 'complex', so these are
# dispatchable slots, and what keeps a configuration out of a kernel is the refusal
# enumeration above.
#
# Still true, and load-bearing for a Triton-less host: this module is NOT
# imported by the package ``__init__`` (which eagerly imports only ``coverage``),
# and launch's forwarders import it lazily inside the function body, so the
# import cost and the Triton dependency are paid only by a run that actually has
# complex storage. Tests and the gate import it directly.
