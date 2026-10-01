"""The TENSOR (off-diagonal epsilon) electric constitutive sub-step, on Metal.

One shader, one coverage predicate, one plan, in one file, on the pattern the
Triton package's ``offdiag_update_e`` set — because this family's whole difference
from the certified constitutive kernel is the OFFDIAG row coupling (a coefficient
multiply sitting BETWEEN two shifts) and the metallic wall-coupling mask, and none
of it belongs in the shared modules yet.

WHAT IT REPLACES. ``stepping.update_E`` (stepping.py:954) with an active PML,
``fields.has_offdiagonal_epsilon`` True (fields.py:1313-1315) and NOT
``fields.has_nonlinearity`` — the ``elif offdiagonal:`` branch (stepping.py:
1001-1008). With no poles admitted, ``displacement_minus_polarization_volumes``
aliases each source to its D primary (fields.py:1107-1138), and all three are
alive at once because the coupling reads the OTHER components' volumes
(stepping.py:991-997). Per component ``c`` with own axis ``a``::

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

THIS IS THE FIRST METAL FAMILY WHOSE STENCIL READS NEIGHBOURS THROUGH A
TRANSVERSE YEE AVERAGE. The certified constitutive sub-step is element-wise; this
one is not, which is exactly why ``coverage.constitutive_coverage(side="E")``
refuses every off-diagonal run at that sub-step (coverage.py's own clause) and why
that refusal STAYS — it is the DISJOINTNESS SEAM between the two families, not a
stale refusal.

THE THINGS THAT DECIDE BIT-IDENTITY, each held by the gate rather than assumed:

1. **THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS** (stepping.py:
   1243-1249): ``u[i]`` multiplies the two-point partner average AT ITS OWN NODE
   and ``u[i+s]`` the average at the next node up — MEEP registers the off-diagonal
   entry at the component's Yee site minus half a cell along its own axis, the
   integer node (anisotropic_averaging.cpp:248-257, ``here - shift1``). A plain
   four-point-average-times-``u[i]`` hoist is the same ALGEBRA only for a uniform
   coefficient, and is bitwise DIFFERENT even there, because distributivity
   (``a*u + b*u`` vs ``(a+b)*u``) is not a bitwise identity in f32. The gate's
   spatially varying coefficient cases are what make the REGISTRATION — not merely
   the rounding — byte-visible; its m1 mutation is the hoist, and the UNIFORM-row
   arm of that mutation at 2**-16 amplitude is THE ALLCLOSE-BLINDNESS CONTROL:
   bytes fail, ``allclose`` passes. That case is the sharpest justification of byte
   gating anywhere in this project and it is carried here deliberately.
2. **THE TWO SHIFTS GO IN OPPOSITE DIRECTIONS** — half a cell DOWN the partner's
   axis on the raw partner volume (``g[i-sx]``), then the already-multiplied
   product half a cell UP the component's own axis (``(pair*u)[i+s]``, which is
   where the corner ``g[(i+s)-sx]`` arises). Per-axis ghost rules compose
   independently on the corner, which is why :func:`_two_way_ghost` keeps the up
   and down validity flags separate and the corner ANDs them at the load site.
   Taking both shifts the same way is the half-cell registration error; the gate's
   m2 mutation is exactly it.
3. **0.25 SCALES THE SUM, APPLIED LAST** (stepping.py:1248-1249). 0.25 is exact in
   f32 and the association is the transcribed one — but it is NOT a byte-observable
   choice: an exact power-of-two factor commutes with round-to-nearest away from
   underflow, so the distributed form is bitwise identical wherever nothing is
   subnormal. The gate carries the distribution as a recorded NULL control; the
   transcribed association is fidelity, not a pinned grouping. Contrast the
   coefficient hoist (m1): ARBITRARY-coefficient distributivity rounds differently
   and IS caught.
4. **THE METALLIC WALL-COUPLING MASK** (``_mask_metallic_wall_coupling``,
   stepping.py:1256-1283): the coupling total is zeroed at FACE 0 of every metallic
   non-mirrored axis on which the component's Yee shift is 0, BEFORE the row sum.
   The wall's tangential E is never stepped and the diagonal product is already
   zero there via ``zero_metal_D``, but the coupling reads live PARTNER volumes
   beside the wall (measured 2.6e-02 unmasked against a 2.0e-07 floor masked). Only
   the low wall is stored; the high wall is the shift-up zero ghost. The mask asks
   the grid's own DECLARATION (``is_metallic and not is_mirrored``), deliberately
   NOT the resolved ghost codes — which is why :data:`MetalOffdiagPlan.wall_axes`
   is carried separately from ``boundary_codes``, exactly as the fused pairs keep
   ``ZM_*`` separate from ``BC*``. m6 (dropped) and m7 (over-applied) hold both
   directions. **WITHIN THIS FAMILY'S ADMITTED SPACE THE TWO ARE LOCKED, AND THAT
   IS MEASURED RATHER THAN LEFT IMPLIED** (``test_metal_offdiag``'s lock test,
   all eight boundary triples): mirrors are refused by the predicate, so
   ``walls[a] == (codes[a] == METALLIC)`` on every grid ``plan_offdiag_constitutive``
   will build for. The separation is therefore FUTURE-PROOFING, not a live degree
   of freedom — it becomes one the moment a fold is admitted, which is exactly
   when conflating the two would zero a plane MEEP steps. The independent pairs
   are reachable only through ``plan_offdiag_constitutive_from_arrays``, which is
   how m6/m7 exercise them.
5. **THE ROW SUM IS DIAGONAL-FIRST**: ``(D_c * us_c) + total`` (stepping.py:
   1007-1008), and the accumulation order inside ``total`` is offset 1 then offset 2
   (:1221). Both additions are bitwise commutative in f32, which is why the gate
   carries the commuted row sum as a NULL control rather than a mutation.
6. **A COMPONENT WITH NO SURVIVING ROW KEEPS THE PURE DIAGONAL ARITHMETIC** —
   ``_offdiagonal_terms`` returns ``None`` and the caller never forms a ``+ 0``
   copy (stepping.py:1222-1227). The row-mask specialisation compiles that arm to
   the certified constitutive kernel's component body verbatim, and the gate's
   identity leg pins the rows-all-dropped build byte-identical to the certified
   kernel on the same seeds.
7. **``prev`` IS READ BEFORE ``f_w`` IS WRITTEN**, and the tail keeps the two
   separate accumulations of ``_apply_constitutive_pml`` (stepping.py:2112-2143) —
   byte-for-byte the certified family's recurrence, under the same file-scope
   contraction directive.

NO DIVISION ANYWHERE, so the ``fast::divide`` hazard (2471 mismatches, measured on
this host) cannot arise in this family. No ``min``/``max`` builtin either. The only
negation is the one inside the shifted-index arithmetic, which is integer.

GHOSTS. Exactly the two plain rules of ``_shift_up`` (stepping.py:1770) /
``_shift_down`` (:1834): the periodic wrap and the metallic zero ghost, per axis,
composed independently for the corner. One of them is transcription fidelity rather
than a byte-observable choice, and it is recorded as a predicted null rather than
claimed as a catch: the partner-axis metallic NEAR (down) zero ghost's entire
support is the partner-axis face-0 plane, which the wall-coupling mask zeroes
BEFORE the row sum — a metallic partner axis is always a Yee-shift-0 transverse
axis of the component, so the mask covers exactly that plane. Mirrors, where the
mask abstains, are refused by this family's predicate.

ON A REDUCED (n = 1) AXIS the wrap returns the same plane and the partner pair is
``2*g`` — NOT the curl's exact zero — matching MEEP's stride(d)=0 double-read. A
sweep case pins it.

STALE DOCSTRING HAZARD (code wins, and the refusals leg measures it):
stepping.py:1219-1220 claims "a folded axis never reaches this function:
``Fields.set_epsilon_volumes`` refuses the combination at install", and
fields.py:1237-1241 repeats it — but ``_validated_offdiagonal_rows``
(fields.py:1262-1310) installs rows UNCHANGED on folded grids. This family refuses
folds anyway (the shared extent/coefficient-index clause), so the refusal is THIS
KERNEL FAMILY'S, not the engine's, and the gate's refusals leg shows the predicate
refusing a folded run that ``stepping`` demonstrably steps. The doc fix belongs in
those files.

DISPATCH IS STILL NOT WIRED; THE ARM NOW IS. ``plan_step`` composes this family as
of tranche 2, after the composition sweep measured the co-admission question over a
constructed configuration matrix; ``fastpath.plan_fast_path`` still returns ``None``
on every branch, which is the separate decision a benchmark informs. The sweep
found this arm's boundary with ``coverage.constitutive_coverage(side="E")`` to be
the one that rests on an ENGINE coupling rather than an inverted clause — the flag
and the live row slots move together only because the installer drops a zero row —
so the planted flag-false/slot-live configuration is carried permanently as the
proof that the pair fails closed rather than picking a winner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import arms, coverage as _coverage, templates
from .device import compile_source
from .plans import KernelPlan

#: The registry family name. One spelling.
FAMILY = "offdiag_constitutive"

#: The slot this family claims.
SLOT = "update_E"

#: The components this sub-step writes, in ``stepping.E_CONSTITUTIVE_TERMS`` order
#: (stepping.py:228), with each component's source volume and OWN axis — the axis
#: whose HALF-INTEGER coefficient pair the tail reads (MEEP's ``dsigw``).
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The coupling partners of each component, in MEEP's ``cycle_direction`` order
#: X -> Y -> Z (vec.hpp:586; stepping.py:1235-1237): own axis + 1 first, own axis
#: + 2 second. Ez therefore takes the Ex-partner volume then the Ey-partner volume.
#: The order BINDS COEFFICIENTS TO PARTNERS — the plan fills the six row slots by
#: this table, and the gate's m3 mutation mispairs them.
TRANSVERSE_PARTNERS: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: The six coefficient slots, in plan/kernel argument order: for each row
#: component (E_TERMS order), the offset-1 partner then the offset-2 partner.
#: Keys are ``fields._chi1inv_offdiagonal``'s own (row, partner) pairs.
ROW_SLOTS: Tuple[Tuple[str, str], ...] = (
    ("Ex", "Ey"), ("Ex", "Ez"),
    ("Ey", "Ez"), ("Ey", "Ex"),
    ("Ez", "Ex"), ("Ez", "Ey"))

#: Per component, the axes on which its Yee shift is 0 — the axes whose metallic
#: wall plane the coupling mask zeroes (``_mask_metallic_wall_coupling`` loops axes
#: ascending, stepping.py:1279-1283; ``fields.IYEE_SHIFTS``: Ex (1,0,0),
#: Ey (0,1,0), Ez (0,0,1)). A test pins this table against ``fields.IYEE_SHIFTS``
#: so the two cannot drift.
WALL_MASK_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))

#: The Yee sub-lattice this side reads: half-integer, ``kps_a_h``/``kms_a_h``
#: (stepping.py:1015 via ``_constitutive_coefficients(..., half_integer=True)``).
HALF_INTEGER = True

#: Buffer bindings the shader declares. Metal's ceiling is 31 (indices 0..30) and a
#: 32nd is a COMPILE ERROR, so the number is asserted by a test rather than left to
#: be discovered by a future edit: 24 buffers + 4 scalars = 28, three of headroom.
BINDING_COUNT = 28

#: Boundary kind string -> the kernel's specialisation code. The strings are
#: ``stepping``'s own; the codes are the package's (shaders.py:71-72).
BOUNDARY_CODES: Dict[str, int] = {"periodic": templates.PERIODIC,
                                  "metallic": templates.METALLIC}


# ---------------------------------------------------------------------------
# The shader
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void offdiag_constitutive_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       w0      [[buffer(3)]],
    device float*       w1      [[buffer(4)]],
    device float*       w2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device const float* e0      [[buffer(9)]],
    device const float* e1      [[buffer(10)]],
    device const float* e2      [[buffer(11)]],
    device const float* u01     [[buffer(12)]],
    device const float* u02     [[buffer(13)]],
    device const float* u11     [[buffer(14)]],
    device const float* u12     [[buffer(15)]],
    device const float* u21     [[buffer(16)]],
    device const float* u22     [[buffer(17)]],
    device const float* kp0     [[buffer(18)]],
    device const float* km0     [[buffer(19)]],
    device const float* kp1     [[buffer(20)]],
    device const float* km1     [[buffer(21)]],
    device const float* kp2     [[buffer(22)]],
    device const float* km2     [[buffer(23)]],
    constant uint&      nx      [[buffer(24)]],
    constant uint&      ny      [[buffer(25)]],
    constant uint&      nz      [[buffer(26)]],
    constant uint&      n_elem  [[buffer(27)]],
    uint idx [[thread_position_in_grid]])
{
__GUARD__

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;

    // --- BOTH neighbours per axis (stepping._shift_down:1787 / _shift_up:1723) --
    // The row product's two shifts go in OPPOSITE directions, so one axis can be
    // read both ways inside one component's term and the corner sample needs the
    // up flag of one axis with the down flag of another. The flags stay separate
    // and are ANDed at the load site.
    int di = i - 1, dj = j - 1, dk = k - 1;
    int ui = i + 1, uj = j + 1, uk = k + 1;
    bool dvx = true, dvy = true, dvz = true;
    bool uvx = true, uvy = true, uvz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    // Wall-plane predicates for the coupling mask (face 0 only; the high wall is
    // already the shift-up zero ghost).
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);

    // The coefficient index is on the component's OWN axis: kps_x/kms_x for
    // target 0, _y for 1, _z for 2 (E_CONSTITUTIVE_TERMS, stepping.py:227). That
    // is MEEP's dsigw, NOT the dsig/dsigu cycle the curl recurrence uses.
    float kp_0 = kp0[i], km_0 = km0[i];
    float kp_1 = kp1[j], km_1 = km1[j];
    float kp_2 = kp2[k], km_2 = km2[k];

    // --- component 0: Ex — own axis x; partners Dy (down y) then Dz (down z) ---
    // `prev` is read BEFORE `fw` is written: the array path copies fw first
    // (stepping.py:2090). Reversed, this is wrong only where kms != 0 -- INSIDE
    // THE PML ONLY -- and looks like a slightly worse absorber, not like a bug.
    float prev0 = w0[ii];
__SRC0__
    w0[ii] = src0;
    float a0 = f0[ii];
    // Two SEPARATE accumulations, left to right: ((f + kps*src) - kms*prev).
    // Flattening to f + (kps*src - kms*prev) is a different float32 number, and
    // neither multiply may contract into an fma -- the file-scope contraction
    // directive is what stops the second.
    a0 = a0 + kp_0 * src0;
    a0 = a0 - km_0 * prev0;
    f0[ii] = a0;

    // --- component 1: Ey — own axis y; partners Dz (down z) then Dx (down x) ---
    float prev1 = w1[ii];
__SRC1__
    w1[ii] = src1;
    float a1 = f1[ii];
    a1 = a1 + kp_1 * src1;
    a1 = a1 - km_1 * prev1;
    f1[ii] = a1;

    // --- component 2: Ez — own axis z; partners Dx (down x) then Dy (down y) ---
    float prev2 = w2[ii];
__SRC2__
    w2[ii] = src2;
    float a2 = f2[ii];
    a2 = a2 + kp_2 * src2;
    a2 = a2 - km_2 * prev2;
    f2[ii] = a2;
}
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 227->228, 2090->2137

#: Per axis, the names the two-way neighbour block declares.
_TWO_WAY_NAMES: Dict[str, Tuple[str, str, str, str, str]] = {
    "x": ("di", "ui", "dvx", "uvx", "nxi"),
    "y": ("dj", "uj", "dvy", "uvy", "nyi"),
    "z": ("dk", "uk", "dvz", "uvz", "nzi"),
}

#: Per axis, the flat-index term contributed when the index is shifted on it, and
#: the term when it is not. Written as data so the six (component, partner) index
#: triples below are ASSEMBLED rather than transcribed six times — six hand-written
#: index expressions is six chances to write ``dj`` where ``dk`` belongs, and that
#: defect is a half-cell registration error that stays smooth and plausible.
_STRIDE: Dict[int, str] = {0: "{} * nyz", 1: "{} * nzi", 2: "{}"}
_HOME: Tuple[str, str, str] = ("i", "j", "k")


def _index(shifts: Dict[int, str]) -> str:
    """A flat index with the named per-axis substitutions applied."""
    parts = [_STRIDE[axis].format(shifts.get(axis, _HOME[axis]))
             for axis in range(3)]
    return " + ".join(parts)


def _two_way_ghost(axis: str, code: int) -> str:
    """One axis's ghost rule for a stencil that reads BOTH neighbours.

    METALLIC serves an exact 0.0 in both directions — past a perfect electric
    conductor there is no field (``stepping._shift_down``:1826-1828,
    ``_shift_up``:1782-1784) — and the flag is what the load's ternary reads, never
    a clamped load. PERIODIC wraps both ways; at k = 0 no Bloch factor multiplies
    the wrapped plane (``_shift_up``:1769-1771 skips the multiply entirely), which
    is what keeps the wrap bit-identical to the plain periodic engine. The wrap is
    integer and exact, so no float rounds on this path.
    """
    down, up, dvalid, uvalid, extent = _TWO_WAY_NAMES[axis]
    if code == templates.METALLIC:
        return (f"    {dvalid} = ({down} >= 0);\n"
                f"    {uvalid} = ({up} < {extent});")
    if code != templates.PERIODIC:
        raise ValueError(f"boundary code {code!r} is neither PERIODIC nor METALLIC")
    return (f"    {down} = ({down} < 0) ? ({extent} - 1) : {down};\n"
            f"    {up} = ({up} == {extent}) ? 0 : {up};")


def _term(component: int, partner_axis: int, coefficient: str,
          tag: str) -> str:
    """One partner's OFFDIAG term — MEEP step_generic.cpp:582-583 as
    ``stepping._offdiagonal_terms`` (:1214-1220) associates it::

        0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])

    THE COEFFICIENT MULTIPLY SITS BETWEEN THE SHIFTS. The near pair takes ``u`` at
    its own node; the far pair — the shifted PRODUCT's value — takes ``u`` at the
    next node up the component's OWN axis. Loading ``u`` at the composed up index
    is the same bits as shifting the formed product: a shift is data movement, the
    coefficient is periodic and phase-free on a wrapped axis, and a metallic far
    face zeroes the whole product through the flags (``far`` is 0 + 0 and ``u_far``
    is 0, so the product is an exact +0.0 — which is precisely what the array
    path's ``_shift_up`` writes into that plane).

    0.25 scales the sum and is applied LAST.
    """
    own_axis = E_TERMS[component][2]
    down, up, dvalid, uvalid, _extent = _TWO_WAY_NAMES["xyz"[partner_axis]]
    _odown, oup, _odvalid, ouvalid, _oextent = _TWO_WAY_NAMES["xyz"[own_axis]]
    partner_volume = f"g{partner_axis}"

    home = "ii"
    down_index = _index({partner_axis: down})
    up_index = _index({own_axis: oup})
    corner_index = _index({own_axis: oup, partner_axis: down})
    return "\n".join([
        f"    float near_{tag} = {partner_volume}[{home}]",
        f"        + ({dvalid} ? {partner_volume}[{down_index}] : 0.0f);",
        f"    float far_{tag} = ({ouvalid} ? {partner_volume}[{up_index}] : 0.0f)",
        f"        + (({ouvalid} && {dvalid}) ? {partner_volume}[{corner_index}]"
        f" : 0.0f);",
        f"    float unear_{tag} = {coefficient}[{home}];",
        f"    float ufar_{tag} = {ouvalid} ? {coefficient}[{up_index}] : 0.0f;",
        f"    float term_{tag} = 0.25f * ((near_{tag} * unear_{tag})"
        f" + (far_{tag} * ufar_{tag}));",
    ])


def _wall_mask(component: int, walls: Sequence[int]) -> str:
    """``stepping._mask_metallic_wall_coupling`` for one component's total.

    Face 0 of every axis on which this component's Yee shift is 0 AND the grid
    DECLARES metallic-not-mirrored. Ascending axis order, the mask's own loop
    order. Emitted only for the axes whose flag is live, so the 'no wall' build
    carries no branch at all.
    """
    flags = ("at_x", "at_y", "at_z")
    lines = [f"    total{component} = {flags[axis]} ? 0.0f : total{component};"
             for axis in WALL_MASK_AXES[component] if walls[axis]]
    return "\n".join(lines) or (
        f"    // no declared metallic wall axis touches component {component}")


#: The coefficient buffer name per ROW_SLOTS entry, in kernel argument order.
_ROW_BUFFERS: Tuple[str, ...] = ("u01", "u02", "u11", "u12", "u21", "u22")


def _component_source(component: int, row_mask: Sequence[int],
                      walls: Sequence[int]) -> str:
    """The whole ``src{c}`` block for one component, on the arm its rows select.

    Four arms per component — none / first / second / both — and the NONE arm is
    the certified constitutive kernel's component body verbatim, which is what
    makes the rows-all-dropped build byte-identical to it (docstring point 6).
    """
    gs = f"    float gs{component} = g{component}[ii];"
    us = f"    float us{component} = e{component}[ii];"
    first, second = row_mask[2 * component], row_mask[2 * component + 1]
    if not (first or second):
        return "\n".join([gs, us,
                          f"    float src{component} = gs{component}"
                          f" * us{component};"])

    lines = [gs, us]
    accumulated: List[str] = []
    for offset in (0, 1):
        if not row_mask[2 * component + offset]:
            continue
        partner_axis = TRANSVERSE_PARTNERS[component][offset]
        tag = f"{component}{offset}"
        lines.append(_term(component, partner_axis,
                           _ROW_BUFFERS[2 * component + offset], tag))
        accumulated.append(tag)

    # `total` accumulates offset 1 then offset 2 (stepping.py:1250), and the mask
    # runs BEFORE the row sum (:1252 precedes the add at :1007-1008).
    lines.append(f"    float total{component} = term_{accumulated[0]};")
    for tag in accumulated[1:]:
        lines.append(f"    total{component} = total{component} + term_{tag};")
    lines.append(_wall_mask(component, walls))
    lines.append(f"    float src{component} = (gs{component} * us{component})"
                 f" + total{component};")
    return "\n".join(lines)


def offdiag_source(row_mask: Sequence[int], codes: Sequence[int],
                   walls: Sequence[int],
                   contract: str = templates.CONTRACT_OFF) -> str:
    """The specialised ``offdiag_constitutive_step`` source.

    ``row_mask`` is the six-flag :data:`ROW_SLOTS` liveness triple-pair, ``codes``
    the per-axis PERIODIC/METALLIC ghost triple ``_boundary_kinds`` resolves, and
    ``walls`` the per-axis metallic-wall DECLARATION triple — a different question
    from ``codes``, carried separately for the reason the module docstring gives.
    All three are baked into the string, as Triton bakes its constexprs.
    """
    row_mask = tuple(int(flag) for flag in row_mask)
    codes = tuple(int(code) for code in codes)
    walls = tuple(int(flag) for flag in walls)
    if len(row_mask) != len(ROW_SLOTS):
        raise ValueError(f"row_mask must fill the {len(ROW_SLOTS)} slots of "
                         f"ROW_SLOTS, got {row_mask!r}")
    if not any(row_mask):
        raise ValueError(
            "no row slot survives: that configuration is the certified plain "
            "constitutive kernel's, and emitting this source for it would overlap "
            "the two families")
    if len(codes) != 3 or len(walls) != 3:
        raise ValueError(f"codes and walls are per-axis triples, got {codes!r} "
                         f"and {walls!r}")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__GUARD__": templates.GUARD,
        "__GHOST_X__": _two_way_ghost("x", codes[0]),
        "__GHOST_Y__": _two_way_ghost("y", codes[1]),
        "__GHOST_Z__": _two_way_ghost("z", codes[2]),
        "__SRC0__": _component_source(0, row_mask, walls),
        "__SRC1__": _component_source(1, row_mask, walls),
        "__SRC2__": _component_source(2, row_mask, walls),
    })


def compile_offdiag(row_mask: Sequence[int], codes: Sequence[int],
                    walls: Sequence[int],
                    contract: str = templates.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (rows, boundaries, walls, mode)."""
    return compile_source(
        offdiag_source(row_mask, codes, walls, contract)
    ).offdiag_constitutive_step


# ---------------------------------------------------------------------------
# The specialisation corpus and its fingerprint
# ---------------------------------------------------------------------------

def specialisations() -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...],
                                     Tuple[int, ...]], ...]:
    """Every (row_mask, codes, walls) triple this family can emit, canonically ordered.

    63 live row masks x 8 boundary triples x 8 wall triples = 4,032 sources. Too
    many to list one sha256 each without burying the fingerprint file, and NOT too
    many to hash — so :func:`corpus_digest` hashes the whole enumeration into one
    value. A single changed character anywhere in the family moves that digest,
    which is the property a per-source list was providing; the per-source hashes
    the gate actually launches are recorded beside it in the gate's own artifact.

    THE ENUMERATION IS DELIBERATELY WIDER THAN THE REACHABLE SET, and the artifact
    should not be read as claiming otherwise: within this family's coverage
    ``walls`` is LOCKED to ``codes`` (module docstring point 4, measured), so
    ``plan_offdiag_constitutive`` can only ever emit the 63 x 8 diagonal subset —
    504 of these 4,032. The other 3,528 are reachable through
    ``plan_offdiag_constitutive_from_arrays`` alone, which is the harness route the
    wall-mask mutations take, and they are hashed so that a change to the emitter
    is a fingerprint event on the whole surface rather than only on the part a
    predicate currently admits.
    """
    out = []
    for mask in range(1, 1 << len(ROW_SLOTS)):
        row_mask = tuple((mask >> bit) & 1 for bit in range(len(ROW_SLOTS)))
        for codes in _TRIPLES:
            for walls in _TRIPLES:
                out.append((row_mask, codes, walls))
    return tuple(out)


_TRIPLES: Tuple[Tuple[int, int, int], ...] = tuple(
    (x, y, z) for x in (templates.PERIODIC, templates.METALLIC)
    for y in (templates.PERIODIC, templates.METALLIC)
    for z in (templates.PERIODIC, templates.METALLIC))


def corpus_digest(contract: str = templates.CONTRACT_OFF) -> Dict[str, Any]:
    """One sha256 over every specialisation, in canonical order, plus the count.

    A digest rather than a list, for the reason :func:`specialisations` gives. It
    is a FINGERPRINT, not a certification: what the byte gate certified is the
    subset it launched, and the gate records those individually.
    """
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    count = 0
    for row_mask, codes, walls in specialisations():
        digest.update(offdiag_source(row_mask, codes, walls, contract).encode())
        count += 1
    return {"count": count, "sha256": digest.hexdigest(), "contract": contract}


# ---------------------------------------------------------------------------
# Small host helpers
# ---------------------------------------------------------------------------

def _is_volume(value: Any) -> bool:
    return bool(getattr(value, "shape", None))


def _base_address(array: Any) -> Optional[int]:
    """The host base address of a volume, or None when unreadable."""
    interface = getattr(array, "__array_interface__", None)
    if isinstance(interface, dict):
        return int(interface["data"][0])
    return None


def wall_mask_axes(grid: Any) -> Tuple[int, int, int]:
    """The per-axis wall declaration, decided the way the mask decides.

    ``_mask_metallic_wall_coupling`` (stepping.py:1279-1283) asks, per axis,
    ``grid.is_metallic(axis) and not grid.is_mirrored(axis)`` — the grid's own
    DECLARATION, deliberately not ``_boundary_kinds``' resolved ghost rule. One
    function so the predicate-side reasoning and the compile-time choice cannot
    disagree; folds are refused by the predicate so the mirrored arm is belt and
    braces here.
    """
    return tuple(  # type: ignore[return-value]
        int(bool(_coverage._call(grid, "is_metallic", axis, default=False))
            and not bool(_coverage._call(grid, "is_mirrored", axis, default=False)))
        for axis in range(3))


def _source_addresses(fields: Any) -> Dict[int, str]:
    """The host base addresses of the three volumes this sub-step mirrors as SOURCES.

    The D primaries, by name — which, with poles refused (clause (a)), is exactly
    what ``displacement_minus_polarization_volumes`` hands the plan builder
    (fields.py:1107-1138). They are mirrored NON-CONSTANT
    (``residency.mirror(term[1], a)``), and that is the whole point of this table:
    see :func:`_alias_reasons`.
    """
    out: Dict[int, str] = {}
    for _target, source, _axis in E_TERMS:
        address = _base_address(getattr(fields, source, None))
        if address is not None:
            out.setdefault(address, source)
    return out


def _alias_reasons(label: str, value: Any, outputs: Dict[int, str],
                   sources: Dict[int, str]) -> List[str]:
    """Why one READ-ONLY input volume may not be the array it aliases.

    TWO DISTINCT HAZARDS, and only the first was refused before 2026-08-15:

    1. **it aliases an OUTPUT** (E or f_w). The coupling re-reads the partner D
       volumes at neighbour offsets while the outputs are written, so the answer
       would depend on thread schedule.
    2. **it aliases a SOURCE** (Dx/Dy/Dz) — MEASURED, and fatal in composition
       rather than in this sub-step. The residency registry keys mirrors by NAME,
       so one host array registered as ``Dx`` (non-constant, because ``step_D``
       writes it) and again as ``chi1inv_offdiag:Ex:Ey`` (constant, because nothing
       writes a coefficient) becomes TWO device tensors. ``step_D``'s device write
       lands in the first; the second keeps the bytes it was born with, and the
       coupling reads a coefficient frozen at step 0. Measured on this host
       (2026-08-15) with ``chi1inv_offdiag[Ex][Ey] = fields.Dy``: the single
       sub-step is byte-identical — both mirrors are fresh at plan time, which is
       exactly why the byte gate cannot see it — while TWO complete driver steps
       diverge by 19,620 words and ``Residency.verify`` reports
       ``{'chi1inv_offdiag:Ex:Ey': 1080}``.

    Hazard 2 is refused HERE, in the family, rather than in
    :class:`~.device.Residency`. The general rule — one host array may not carry
    both a constant and a non-constant mirror — belongs in the spine and would bind
    six other families whose gates this round does not own; it is recorded as a
    measured recommendation instead of taken unilaterally.

    Inputs may still alias EACH OTHER freely: three aliased inverse-epsilon volumes
    are the isotropic install and one coefficient array on two row slots is a legal
    tensor. Both are CONSTANT mirrors of one host array, which nothing writes, so
    neither can go stale.
    """
    address = _base_address(value)
    if address is None:
        return []
    if address in outputs:
        return [f"{label} aliases output {outputs[address]}: the coupling would "
                f"read the volume while the sub-step writes it, and the plan "
                f"builder refuses the alias — the predicate must refuse it first"]
    if address in sources:
        return [f"{label} aliases source {sources[address]}: the residency registry "
                f"keys mirrors by NAME, so one host array bound as a written source "
                f"and again as a read-only coefficient becomes two device tensors "
                f"and the coefficient freezes at plan time. Measured: identical for "
                f"one sub-step, 19,620 differing words over two complete steps"]
    return []


def row_volumes_for(fields: Any) -> Tuple[Optional[Any], ...]:
    """The six coefficient slots in :data:`ROW_SLOTS` order, ``None`` where dead.

    The single place the slot binding is derived from the engine's own rows
    (``Fields.chi1inv_offdiagonal_for``, fields.py:1317-1319), so the predicate,
    the plan and the gate cannot disagree about which coefficient volume pairs with
    which partner term — the mispairing is the gate's m3 mutation.
    """
    return tuple(
        _coverage._call(fields, "chi1inv_offdiagonal_for", row,
                        default={}).get(partner)
        for row, partner in ROW_SLOTS)


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------

def _row_reasons(fields: Any, shape: Sequence[int]) -> List[str]:
    """Every surviving row volume must be a real f32 C-contiguous grid volume that
    aliases neither this sub-step's outputs nor its sources.

    THE INSTALLER DOES NOT NORMALIZE ALL OF THIS AWAY, and this docstring said it
    did. ``_validated_offdiagonal_rows`` (fields.py:1262-1310) casts to float32,
    refuses complex and a shape mismatch, refuses non-finite values and drops an
    identically zero row — but it casts with ``astype(..., copy=False)``
    (fields.py:1296), whose default ``order="K"`` PRESERVES Fortran order. Measured
    on this host (2026-08-15): a row built with ``numpy.asfortranarray`` installs
    through the PUBLIC installer and arrives here ``F_CONTIGUOUS`` with
    ``C_CONTIGUOUS`` False, and this clause list is what refuses it. The same is
    true with less qualification of the inverse-epsilon maps, which
    ``set_epsilon_volumes`` stores with no normalization at all
    (fields.py:1251-1252). So some of these clauses are reachable through the front
    door and the rest are for a row planted past it; either way the package's rule
    is that such a row is refused by name rather than stepped.

    THE ALIAS CLAUSES ARE BEYOND THE INSTALLER'S REACH ENTIRELY — it keeps the
    caller's array without copying — and the plan builder refuses both with a
    raise, so the predicate must refuse them FIRST to keep the builder's
    None-means-refused contract: a covered verdict must never meet a ValueError.
    See :func:`_alias_reasons` for the two hazards and the measurement behind the
    second.
    """
    out: List[str] = []
    outputs: Dict[int, str] = {}
    for term in E_TERMS:
        for name in (term[0], "f_w_" + term[0]):
            address = _base_address(getattr(fields, name, None))
            if address is not None:
                outputs.setdefault(address, name)
    sources = _source_addresses(fields)
    for name in ("Ex", "Ey", "Ez"):
        volume = _coverage._call(fields, "inverse_epsilon_for", name, default=None)
        out.extend(_alias_reasons(f"inv_eps[{name}]", volume, outputs, sources))
    for row, partner in ROW_SLOTS:
        entries = _coverage._call(fields, "chi1inv_offdiagonal_for", row,
                                  default={}) or {}
        value = entries.get(partner)
        if value is None:
            continue
        label = f"chi1inv_offdiag[{row}][{partner}]"
        if not _is_volume(value):
            out.append(f"{label} is not a volume")
            continue
        # FAIL-CLOSED, and this line USED TO RAISE. It read
        # ``str(getattr(value, "dtype", ""))[0]``, which is an IndexError on any
        # volume-shaped object that answers no dtype — measured, 2026-08-15:
        # ``IndexError: string index out of range`` out of the PREDICATE. Every
        # other clause in this module is duck-typed with a default precisely
        # because a row planted past the installer is the case these clauses
        # exist for, and a predicate that raises is neither a refusal nor a
        # verdict: ``plan_offdiag_constitutive``'s None-means-refused contract
        # and ``arms``' raising-predicate-is-a-refusal contract disagree about
        # what happened. An unreadable width is refused BY NAME instead.
        dtype = str(getattr(value, "dtype", ""))
        if not dtype:
            out.append(f"{label} exposes no dtype; a buffer width that cannot be "
                       f"read cannot be bound, and an unreadable volume is "
                       f"refused rather than inspected further")
            continue
        if dtype.startswith("c"):
            out.append(f"{label} is complex; the inverse-permittivity tensor of a "
                       f"lossless medium is real")
            continue
        out.extend(_alias_reasons(label, value, outputs, sources))
        if len(shape) == 3:
            out.extend(_coverage._volume_reasons(label, value, shape))
    return out


def offdiag_constitutive_coverage(fields: Any, pml: Any,
                                  residency: Any = None) -> "_coverage.Coverage":
    """May the Metal off-diagonal kernel step ``update_E`` for this pair?

    POSITIVE CLAUSES ONLY; a failing clause appends its reason and the scan
    continues. The grid clauses are ``coverage._grid_reasons``' — this module's
    Metal ones, called DIRECTLY and not restated — because this predicate inverts
    NO shared clause: chi2/chi3 stays refused (the row-product-times-Pade
    most-general case is a later fused leg), beta stays refused (the engine itself
    additionally raises for real+offdiag+beta, stepping.py:800-810, mirroring MEEP
    fields.cpp:548-549), the fold clause stays, and everything else is wanted
    verbatim. The offdiag INVERSION lives in the E-side clauses below, exactly
    where the certified predicate's refusal lives:

    a. **no registered polarization.** With poles the source is ``D - sum P``
       formed in per-component scratch buffers (fields.py:1107-1138) and the
       coupling would read THOSE, not D — a later fused leg's configuration.
    b. **at least one surviving off-diagonal row (INVERTED), counted over the six
       ROW_SLOTS** — deliberately not the bare ``has_offdiagonal_epsilon`` flag,
       which a row planted past the installer under a diagonal key sets while every
       slot is dead (the builder would raise where this predicate had admitted). A
       zero-slot run is bit-identically the diagonal engine's (the install-time
       drop, fields.py:1302-1303) and belongs to
       ``coverage.constitutive_coverage(side="E")`` — the two predicates are
       disjoint by construction and a test pins it.
    c. **stored E** — forced True by any surviving row (fields.py:1254-1255); belt
       and braces with a reason.
    d. every surviving row volume real, f32, C-contiguous, grid-shape, and aliasing
       neither an E/f_w OUTPUT nor a D SOURCE — the second measured, and fatal only
       in composition (:func:`_alias_reasons`). The inverse-epsilon volumes take
       the same two alias clauses, for the same reason and by the same route.
    e. volume inverse epsilon per component — this kernel carries NO scalar arm.
    f. layout, the half-integer coefficient tables, and the RESIDENCY declaration
       (the Metal-only clause: two sub-steps that mirror one volume separately each
       hold a private copy and the second launch reads the first one's stale bytes).

    Conductivity is NOT a clause, deliberately: it changes the CURL sub-steps only,
    never the constitutive one.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = _coverage._grid_reasons(fields, pml, grid)
    reasons.extend(_coverage._susceptibility_reasons(fields))
    reasons.extend(_coverage._residency_declaration_reasons(residency))

    # (a) No poles: the coupling must read the aliased D primaries.
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if states or getattr(fields, "has_polarizations", False):
        reasons.append(
            "a susceptibility is registered: the offdiag source becomes "
            "D - sum P in per-component scratch buffers (fields.py:1107-1138) — "
            "the fused dispersive+offdiag kernel is a later leg")

    # (b) INVERTED: at least one surviving row SLOT.
    if not any(value is not None for value in row_volumes_for(fields)):
        reasons.append(
            "no off-diagonal chi1inv row survived installation: that "
            "configuration is constitutive_coverage(side='E')'s and this "
            "predicate must not overlap it")

    # (c) Stored E.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # (d) The surviving rows, readable and well-formed.
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_row_reasons(fields, shape))

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

    return _coverage.Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalOffdiagConstitutivePlan(KernelPlan):
    """A launchable, allocation-free off-diagonal ``update_E``.

    The Yee sub-lattice is chosen at the BUILDERS and nowhere else:
    ``kps_a_h``/``kms_a_h``, the half-integer tables (stepping.py:1015). The kernel
    takes six coefficient pointers and never asks which lattice they came from — a
    swap is the silent half-cell absorber error the certified family's gate carries
    a mutation for, and this one carries it too.

    The six ROW SLOTS are bound in :data:`ROW_SLOTS` order; a dead slot's pointer
    is bound to the component's own D mirror (never read — the specialisation is
    what stops the read, the buffer still has to bind). The WALL AXES triple is the
    mask's own question, carried separately from the boundary codes.

    ``__init__`` REFUSES ALIASING between the outputs (E, f_w) and any input (D,
    inverse epsilon, row volumes, the six coefficient vectors), and among the
    outputs themselves: the coupling re-reads the partner D volumes at neighbour
    offsets while E and f_w are being written, so an aliased pair would make the
    result depend on thread schedule — a wrong answer that varies run to run. An
    all-dead row mask is refused toward the certified plain kernel.
    """

    __slots__ = ("shape", "n_elem", "row_mask", "boundary_codes", "wall_axes",
                 "residency", "volumes")

    REPR_FIELDS = ("shape", "row_mask", "boundary_codes", "wall_axes")

    def __init__(self, shape, residency, targets, auxiliaries, sources,
                 inverse_epsilon, rows, coefficients, boundary_codes, wall_axes,
                 functions: Dict[str, Any], volumes: Sequence[str],
                 host_rows: Sequence[Any] = (),
                 host_arrays: Sequence[Any] = ()) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.boundary_codes = tuple(int(code) for code in boundary_codes)
        if len(self.boundary_codes) != 3 or any(
                code not in (templates.PERIODIC, templates.METALLIC)
                for code in self.boundary_codes):
            raise ValueError(f"boundary codes must be three of "
                             f"{{{templates.PERIODIC}, {templates.METALLIC}}}, "
                             f"got {boundary_codes!r}")
        self.wall_axes = tuple(int(flag) for flag in wall_axes)
        if len(self.wall_axes) != 3 or any(flag not in (0, 1)
                                           for flag in self.wall_axes):
            raise ValueError(f"wall axes must be three of {{0, 1}}, got "
                             f"{wall_axes!r}")
        rows = tuple(rows)
        if len(rows) != len(ROW_SLOTS):
            raise ValueError(f"rows must fill the {len(ROW_SLOTS)} slots of "
                             f"ROW_SLOTS (None where dead), got {len(rows)}")
        self.row_mask = tuple(int(value is not None) for value in rows)
        if not any(self.row_mask):
            raise ValueError(
                "no row slot survives: that configuration belongs to the certified "
                "plain constitutive kernel, and building this plan for it would "
                "overlap the two")
        self.residency = residency
        self.volumes = tuple(volumes)

        self._require_no_aliasing(host_arrays, host_rows)

        # Dead slots bind the row component's own D mirror; the specialisation
        # stops the read but the buffer argument still has to bind.
        component_of_slot = tuple(
            next(index for index, term in enumerate(E_TERMS) if term[0] == row)
            for row, _partner in ROW_SLOTS)
        bound_rows = tuple(
            value if value is not None else tuple(sources)[component_of_slot[index]]
            for index, value in enumerate(rows))

        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(inverse_epsilon) + bound_rows + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem))

    @staticmethod
    def _require_no_aliasing(host_arrays: Sequence[Any],
                             host_rows: Sequence[Any]) -> None:
        """Base-address disjointness of outputs vs outputs and inputs vs outputs.

        ``host_arrays`` is ``(targets, auxiliaries, sources, inverse_epsilon,
        coefficients)`` as HOST arrays — the device tensors cannot answer the
        question, because two mirrors of the same host array are the SAME tensor
        by the residency registry's design, and that is legal for constants and
        fatal for outputs. Equality of BASE addresses only: two overlapping views
        with different bases pass unseen, which is the certified plans' shared and
        accepted limitation.

        READ-ONLY INPUTS MAY ALIAS EACH OTHER FREELY — three aliased
        inverse-epsilon volumes are the isotropic install, and one coefficient
        array installed on two row slots is a legal tensor — because those are
        CONSTANT mirrors of one host array and nothing writes them. What they may
        NOT alias is a volume this plan mirrors as WRITTEN: an OUTPUT (E, f_w) or
        a SOURCE (D). The source case is MEASURED and is invisible to a single
        sub-step, which is why it went unrefused until 2026-08-15; see
        :func:`_alias_reasons`. ``sources`` counts as a write-side name here even
        though this kernel only reads it, because ``step_D`` writes it in the
        composition the residency layer exists to serve.
        """
        if not host_arrays:
            return
        targets, auxiliaries, sources, inverse_epsilon, coefficients = host_arrays
        outputs: Dict[int, str] = {}
        for group_name, group in (("target", targets), ("aux", auxiliaries)):
            for index, array in enumerate(group):
                address = _base_address(array)
                if address is None:
                    raise ValueError(
                        f"{group_name} {index} exposes no readable base address; "
                        f"an unverifiable output is not accepted")
                if address in outputs:
                    raise ValueError(
                        f"{group_name} {index} aliases {outputs[address]}; the "
                        f"outputs must be distinct arrays")
                outputs[address] = f"{group_name} {index}"
        written = dict(outputs)
        for index, array in enumerate(sources):
            address = _base_address(array)
            if address is not None and address in outputs:
                raise ValueError(
                    f"an input volume aliases {outputs[address]}: the coupling "
                    f"re-reads the partner D volumes at neighbour offsets while "
                    f"the outputs are written, so an alias makes the answer depend "
                    f"on thread schedule")
            if address is not None:
                written.setdefault(address, f"source {index}")
        read_only = list(inverse_epsilon)
        read_only += [value for value in host_rows if value is not None]
        read_only += [value for value in coefficients if _is_volume(value)]
        for array in read_only:
            address = _base_address(array)
            if address is not None and address in written:
                raise ValueError(
                    f"a read-only input volume aliases {written[address]}: an "
                    f"OUTPUT alias makes the answer depend on thread schedule, and "
                    f"a SOURCE alias gives one host array both a written and a "
                    f"constant device mirror — so the coefficient freezes at plan "
                    f"time and the field goes smoothly, plausibly wrong one "
                    f"complete step later (measured: 19,620 words over two steps)")


def _functions(row_mask, codes, walls,
               contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_offdiag(row_mask, codes, walls, mode)
            for mode in contract_variants}


def plan_offdiag_constitutive(fields: Any, pml: Any, residency: Any = None,
                              contract_variants: Sequence[str] = (
                                  templates.CONTRACT_OFF,),
                              ) -> Optional[MetalOffdiagConstitutivePlan]:
    """Build the off-diagonal ``update_E`` plan from the engine's own objects.

    None means REFUSED, and the reasons come from
    :func:`offdiag_constitutive_coverage`. The sources are taken through
    ``displacement_minus_polarization_volumes`` — the accessor the array path
    itself reads (stepping.py:996, :1004) — which with no poles admitted hands back
    the aliased D primaries (fields.py:1107-1138).
    """
    if not offdiag_constitutive_coverage(fields, pml, residency).covered:
        return None
    kinds = _coverage._boundary_kinds(fields.grid, pml)
    codes = tuple(BOUNDARY_CODES[kind] for kind in kinds)
    walls = wall_mask_axes(fields.grid)
    targets = tuple(term[0] for term in E_TERMS)
    volumes = fields.displacement_minus_polarization_volumes()
    rows = row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)

    host_targets = [getattr(fields, n) for n in targets]
    host_aux = [getattr(fields, "f_w_" + n) for n in targets]
    host_sources = [volumes[n] for n in targets]
    host_inv = [fields.inverse_epsilon_for(n) for n in targets]
    host_coefficients = [getattr(pml, f"{stem}_{axis}_h")
                         for axis in "xyz" for stem in ("kps", "kms")]

    mirrored_targets = [residency.mirror(n, a) for n, a in zip(targets, host_targets)]
    mirrored_aux = [residency.mirror("f_w_" + n, a)
                    for n, a in zip(targets, host_aux)]
    mirrored_sources = [residency.mirror(term[1], a)
                        for term, a in zip(E_TERMS, host_sources)]
    mirrored_inv = [residency.mirror("inv_eps_" + n, a, constant=True)
                    for n, a in zip(targets, host_inv)]
    mirrored_rows = [
        None if value is None
        else residency.mirror(f"chi1inv_offdiag:{row}:{partner}", value,
                              constant=True)
        for (row, partner), value in zip(ROW_SLOTS, rows)]
    mirrored_coefficients = [
        residency.mirror(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"),
                         constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    volume_names = (tuple(targets) + tuple("f_w_" + n for n in targets)
                    + tuple(term[1] for term in E_TERMS))
    return MetalOffdiagConstitutivePlan(
        fields.grid.shape, residency, mirrored_targets, mirrored_aux,
        mirrored_sources, mirrored_inv, mirrored_rows, mirrored_coefficients,
        codes, walls, _functions(row_mask, codes, walls, contract_variants),
        volume_names,
        host_rows=rows,
        host_arrays=(host_targets, host_aux, host_sources, host_inv,
                     host_coefficients))


def plan_offdiag_constitutive_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any],
        rows: Dict[str, Dict[str, Any]], boundary_codes: Sequence[int],
        wall_axes: Sequence[int], residency: Any,
        functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (templates.CONTRACT_OFF,),
        ) -> MetalOffdiagConstitutivePlan:
    """Build it from bare host arrays — the gate's and probe's route.

    ``arrays`` is keyed by component name (``Ex``, ``f_w_Ex``, ``Dx``,
    ``inv_eps_Ex``), ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE
    CALLER already selected, and ``rows`` maps row component -> partner component ->
    coefficient volume, with absent entries marking dead slots (the caller performs
    the install-time zero-row drop itself; an explicitly present all-zero volume is
    treated as LIVE, which is deliberate — the array path never sees one, so a
    harness wanting the degenerate byte-equality case must drop it the way the
    installer does). No coverage predicate runs here: the caller is a harness that
    constructed the configuration on purpose, including the deliberately wrong ones.

    ``functions=`` IS LOAD-BEARING: the mutation legs route a mutated kernel
    through it, and a builder that drops it launches the SHIPPED kernel and reports
    a pass for a defect it never introduced.
    """
    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    row_values = tuple((rows.get(row) or {}).get(partner)
                       for row, partner in ROW_SLOTS)
    row_mask = tuple(int(value is not None) for value in row_values)

    host_targets = [arrays[n] for n in targets]
    host_aux = [arrays["f_w_" + n] for n in targets]
    host_sources = [arrays[term[1]] for term in E_TERMS]
    host_inv = [arrays["inv_eps_" + n] for n in targets]
    host_coefficients = [flat[f"{stem}_{axis}"]
                         for axis in "xyz" for stem in ("kps", "kms")]

    mirrored_targets = [residency.mirror(n, a) for n, a in zip(targets, host_targets)]
    mirrored_aux = [residency.mirror("f_w_" + n, a)
                    for n, a in zip(targets, host_aux)]
    mirrored_sources = [residency.mirror(term[1], a)
                        for term, a in zip(E_TERMS, host_sources)]
    mirrored_inv = [residency.mirror("inv_eps_" + n, a, constant=True)
                    for n, a in zip(targets, host_inv)]
    mirrored_rows = [
        None if value is None
        else residency.mirror(f"chi1inv_offdiag:{row}:{partner}", value,
                              constant=True)
        for (row, partner), value in zip(ROW_SLOTS, row_values)]
    mirrored_coefficients = [
        residency.mirror(f"pml:{stem}_{axis}:offdiag", flat[f"{stem}_{axis}"],
                         constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    volume_names = (tuple(targets) + tuple("f_w_" + n for n in targets)
                    + tuple(term[1] for term in E_TERMS))
    return MetalOffdiagConstitutivePlan(
        shape, residency, mirrored_targets, mirrored_aux, mirrored_sources,
        mirrored_inv, mirrored_rows, mirrored_coefficients, boundary_codes,
        wall_axes,
        functions if functions is not None
        else _functions(row_mask, boundary_codes, wall_axes, contract_variants),
        volume_names,
        host_rows=row_values,
        host_arrays=(host_targets, host_aux, host_sources, host_inv,
                     host_coefficients))


# ---------------------------------------------------------------------------
# Registration — WIRED, as of tranche 2
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> "_coverage.Coverage":
    return offdiag_constitutive_coverage(context.fields, context.pml,
                                         context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalOffdiagConstitutivePlan]:
    return plan_offdiag_constitutive(context.fields, context.pml,
                                     context.residency,
                                     context.contract_variants)


ARM = arms.register(
    family=FAMILY, slot=SLOT, label="offdiag",
    coverage=_arm_coverage, plan=_arm_plan,
    prefix="offdiag: ", noun="off-diagonal constitutive",
    # WIRED, as of tranche 2, and this is the one arm whose disjointness was NOT
    # settled by an inverted clause. `coverage.constitutive_coverage(side="E")`
    # refuses on the FLAG (`has_offdiagonal_epsilon`) while this predicate requires
    # a LIVE ROW SLOT, and the two agree only because the installer drops a
    # zero row and the flag is a read-only property over the very dict the row
    # accessor reads. That coupling is an engine fact, not a composition fact, so
    # the composition sweep does two things rather than assume it: it runs the
    # reachable direction (flag True, every slot dead — both arms refuse, no
    # ambiguity) and it PLANTS the unreachable one (flag False, a slot live) to
    # prove the pair FAILS CLOSED, leaving `update_E` on the array path and naming
    # both claimants, instead of one arm silently winning.
    wired=True)
