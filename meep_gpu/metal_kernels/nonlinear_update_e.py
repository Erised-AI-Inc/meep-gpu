"""The INSTANTANEOUS chi2/chi3 (Pade) electric constitutive sub-step, on Metal.

One shader, one coverage predicate, one plan, in one file, on the pattern
:mod:`.offdiag_update_e` set — that module is this one's TEMPLATE and the shape is
deliberately copied rather than reinvented: same slot, same half-integer tail, same
two-way ghost block, same alias refusals, same plan/from_arrays pair. What differs
is the SOURCE term, and only that.

WHAT IT REPLACES. ``stepping.update_E`` (stepping.py:954) with an active PML and
``fields.has_nonlinearity`` True — the ``if nonlinear:`` branch (stepping.py:
999-1000) reaching ``_nonlinear_constitutive`` (:1080). With no poles admitted,
``displacement_minus_polarization_volumes`` aliases each source to its D primary
(fields.py:1107-1138), and all three are alive at once because ``Dsqr`` reads the
OTHER components' volumes (stepping.py:991-997). Per component ``c`` with own axis
``a``, when ``fields.is_nonlinear(c)``::

    per partner (offset 1 then 2, cycle X->Y->Z, stepping.py:1183-1185):
        pair  = g + shift_down(g, partner_axis)          # :1187-1189
        gNs   = pair + shift_up(pair, a)                 # :1162
    Dsqr  = gs*gs + 0.0625*(g1s*g1s + g2s*g2s)           # :1118
    c2    = (gs   * chi2) * (us*us)                      # :1025
    c3    = (Dsqr * chi3) * ((us*us)*us)                 # :1026
    u     = ((1 + c2) + 2*c3) / ((1 + 2*c2) + 3*c3)      # :1027
    src   = (gs * us) * u                                # :1078-1080
    prev  = f_w_c ; f_w_c = src                          # :2065-2096
    E_c  += kps_a_h * f_w_c ;  E_c -= kms_a_h * prev     # half-integer, :986

and, when the component is LINEAR in a partly nonlinear run, ``src = gs * us`` —
MEEP's ``else if (u)`` branch (stepping.py:1100-1102), which is the certified plain
constitutive kernel's component body verbatim.

THIS IS THE FIRST METAL FAMILY IN THIS PACKAGE THAT DIVIDES, and that is not a
remark. ``shaders.py``'s transcription rule 4 states the certified pair's position
plainly — "Neither body divides, calls ``sqrt``, or takes a min or a max, which is
why this pair is the lowest-risk one to certify first" — so the Pade quotient walks
off the end of every measurement this package had. It was therefore MEASURED before
this file was written, not after: see :func:`~parity.meep_gpu` probe
``probe_metal_divide_and_negation.py`` and the numbers in point 1 below.

THE THINGS THAT DECIDE BIT-IDENTITY, each held by measurement rather than assumed:

1. **THE DIVISION IS THE PLAIN ``/`` — MEASURED ON THIS HOST, NOT INHERITED.** The
   sibling Triton track carries ``tl.math.div_rn`` because Triton's plain ``/``
   lowers to ``div.full.f32`` (~2 ulp) and every nonlinear sweep case diverged
   there. **That reason does not reproduce here.** Measured 2026-08-16 (arm64,
   macOS 26.2, torch 2.10.0, metalfe-32023.850.10): Metal's ``/`` agrees with
   ``numpy.float32`` division on **0 differing normal-result words** across five
   operand classes — ``pade_band`` (both operands anchored at 1.0, 65,536 lanes),
   ``random_wide``, ``near_pole``, ``exact_powers`` and the EXHAUSTIVE 256-pair
   special-value table — and the whole Pade expression end to end is 0 differing on
   all four of its operand classes including ``large_u`` (u down to 0.666) and
   ``wall_zero``. ``fast::divide`` DIVERGES by up to 2 ulp on 102,578 words and is
   this family's must-catch source mutation; ``precise::divide`` agrees with the
   plain operator and is a recorded null. Porting Triton's workaround on faith
   would have been the mistake here, exactly as porting its negation idiom would.

2. **THE ONLY DIVERGENCE THE DIVIDE SHOWS IS THE SUBNORMAL FLUSH, and it is the
   PLATFORM'S, not the operator's.** On the special-value table 16 lanes differ;
   all 16 have a SUBNORMAL IEEE result and the device produced the correctly-signed
   zero. ``multiply`` — an operation four certified families already ship — shows
   the identical signature on 8 lanes, which is what proves the effect belongs to
   the number system rather than to the quotient. Metal flushes and cannot be
   unflushed (no pragma, no env var, no torch API), so ``keep`` is NOT offerable
   and every claim here rides on the CHECKED subnormal-free precondition in
   :mod:`.subnormal`, never on a tolerance.

3. **THE PADE GROUPINGS ARE THE TRANSCRIBED ONES**, and each paren is load-bearing
   in the way ``shaders.py`` rule 2 describes: ``(gs*chi2)*(us*us)``,
   ``(Dsqr*chi3)*((us*us)*us)``, ``((1 + c2) + 2*c3) / ((1 + 2*c2) + 3*c3)``, then
   ``(gs*us)*u`` — Python's left-to-right association in ``calc_nonlinear_u``
   (stepping.py:1054-1056) and the row-then-scale order of ``_nonlinear_constitutive``
   (:1107-1109). Flattening any of them is a different float32 number.

4. **THE FOUR-CORNER ASSOCIATION IS THE SHIFTED-PAIR ONE, NOT MEEP C'S.**
   ``_nonlinear_transverse_sums`` (stepping.py:1187-1191) forms
   ``(g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1])`` — pair first, then the pair shifted
   — while MEEP C sums left to right (step_generic.cpp:646-648). **The byte arbiter
   is stepping.py**, so the pair association is what this kernel carries and the
   left-to-right form is a gate mutation, not a comment.

5. **THE TWO SHIFTS GO IN OPPOSITE DIRECTIONS** — half a cell DOWN the partner's
   axis on the raw partner volume, then the formed pair half a cell UP the
   component's OWN axis (stepping.py:1160-1171). Taking both the same way is the
   half-cell registration error that survives every scalar test. Per-axis ghost
   rules compose independently on the corner, which is why :func:`_two_way_ghost`
   keeps the up and down validity flags separate and the corner ANDs them at the
   load site — the partner-axis flag is index-invariant along the own axis, so
   evaluating it at ``i`` and at ``i+s`` is the same flag.

6. **0.0625 SCALES THE TRANSVERSE SUM OF SQUARES, APPLIED LAST** (stepping.py:1147).
   ``0.0625 = (1/4)^2`` turns each UNNORMALIZED four-point sum back into a mean
   before it is squared. It is an exact power of two, so — like the off-diagonal
   family's 0.25 — distributing it is bitwise identical away from underflow and is
   carried as a recorded NULL control rather than claimed as a catch.

7. **A VOLUME INVERSE EPSILON IS REQUIRED, AND THAT IS A NUMERICAL CLAUSE RATHER
   THAN A CONVENIENCE.** With a PYTHON-FLOAT ``us`` NumPy evaluates
   ``(chi1inv * chi1inv)`` in DOUBLE and rounds once when the product meets the
   float32 array, which is a different word from the float32 power this kernel
   forms: measured on this host, **51,218 differing words of 65,536**. The Triton
   family answers that by shipping three host-rounded scalars; this one answers it
   by refusing a scalar epsilon BY NAME (the shared
   ``coverage._inverse_epsilon_reasons`` clause already does exactly that), so the
   powers are float32 on both sides by construction. The chi2/chi3 SCALARS are a
   different case and are admitted: a Python float meeting a float32 array is a
   weak scalar rounded ONCE to float32 under NEP 50, measured 0 differing words
   over seven scalars spanning 1e-30 to 3.4e38.

8. **``prev`` IS READ BEFORE ``f_w`` IS WRITTEN**, and the tail keeps the two
   separate accumulations of ``_apply_constitutive_pml`` (stepping.py:2112-2143) —
   byte-for-byte the certified constitutive family's recurrence, under the same
   file-scope contraction directive. **The directive is load-bearing on THIS
   expression and that was measured too**: with ``contract(fast)`` the Pade body
   diverges by 10,770 words (up to 2 ulp), because the numerator and denominator
   sums contract into fmas before the divide ever runs.

NO ``min``/``max`` BUILTIN and NO NEGATION anywhere in this body — the Pade factor
is built from multiplies, adds and one divide, and the tail subtracts rather than
negates. The package's ``-x`` rule was re-measured this round anyway (0 differing
on random, exhaustive-special and subnormal-band classes; ``0.0f - x`` diverges,
``x * -1.0f`` also exact) and is recorded in the probe artifact; this family simply
has no site that depends on it.

GHOSTS. Exactly the two plain rules of ``_shift_up`` (stepping.py:1770) /
``_shift_down`` (:1834): the periodic wrap and the metallic zero ghost, per axis,
composed independently for the corner. No Bloch phase (k = 0 only, so
``_shift_up``:1769-1771 skips the multiply entirely), no mirror parity, no far
reflect-row — ``_shift_up``'s docstring names the nonlinear sums as exactly the
caller that leaves ``reflect_row`` unset and keeps the zero face (:1741-1744), and
the driver refuses a nonlinearity on a live-faced periodic fold rather than serve
that zero as an answer. This family refuses every fold anyway.

THE POLE GUARD IS NOT A PREDICATE CLAUSE. ``nonlinear_margin`` (stepping.py:1344)
is sampled OUTSIDE the step loop by the driver's ``_NonlinearityGuard`` and is
unchanged by this kernel. What IS a predicate clause is the MAGNITUDE refusal —
see clause (g) in :func:`nonlinear_constitutive_coverage`, which is a coverage
refusal and not a tolerance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import arms, coverage as _coverage, templates
from .coverage import _boundary_kinds, _call, _metal_backend_reasons
from .device import compile_source
from .plans import KernelPlan

from ..triton_kernels.coverage import COVERED_BOUNDARIES

#: The registry family name. One spelling.
FAMILY = "nonlinear_constitutive"

#: The slot the dedicated Pade shader claims.  The same family also owns three
#: *admissions* below: ``step_B``, ``step_D``, and ``update_H`` reuse their
#: already-certified ordinary Metal shaders because the engine does not read χ²/χ³
#: there.  They are separate arms rather than a widening of the ordinary predicates,
#: which preserves fail-closed disjointness on nonlinear configurations.
SLOT = "update_E"

#: The components this sub-step writes, in ``stepping.E_CONSTITUTIVE_TERMS`` order
#: (stepping.py:228), with each component's source volume and OWN axis — the axis
#: whose HALF-INTEGER coefficient pair the tail reads (MEEP's ``dsigw``).
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The transverse partners of each component, in MEEP's ``cycle_direction`` order
#: X -> Y -> Z (vec.hpp:586; stepping.py:1183-1185): own axis + 1 first, own axis
#: + 2 second. Ez therefore takes the Dx partner volume then the Dy one. The order
#: BINDS PARTNERS TO SUM SLOTS and a test pins it against ``stepping``'s own
#: ``(own_axis + offset) % 3``, because mispairing them is a smooth wrong answer.
TRANSVERSE_PARTNERS: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: The Yee sub-lattice this side reads: half-integer, ``kps_a_h``/``kms_a_h``
#: (stepping.py:1015 via ``_constitutive_coefficients(..., half_integer=True)``).
HALF_INTEGER = True

#: Buffer bindings the shader declares. Metal's ceiling is 31 (indices 0..30) and a
#: 32nd is a COMPILE ERROR, so the number is asserted by a test rather than left to
#: be discovered by a future edit: 25 buffers + 4 scalars = 29, two of headroom.
#:
#: THE PACKED SCALAR BUFFER EXISTS BECAUSE OF THIS CEILING, and it is the design
#: decision the limit FORCED rather than informed. Six separate ``constant float&``
#: chi scalars — the shape the Triton kernel uses, where the ABI has no such limit
#: — would have been 24 + 6 + 4 = 34 bindings and would not have compiled at all.
#: One buffer holding six floats costs one binding. The same reasoning is what the
#: ADE ``update_P`` port will need for its per-pole coefficients.
BINDING_COUNT = 29

#: Boundary kind string -> the kernel's specialisation code. The strings are
#: ``stepping``'s own; the codes are the package's (shaders.py:71-72).
BOUNDARY_CODES: Dict[str, int] = {"periodic": templates.PERIODIC,
                                  "metallic": templates.METALLIC}

#: THE MAGNITUDE CEILING, and it is a COVERAGE REFUSAL rather than a tolerance.
#:
#: THE SCIENCE IS SETTLED AND IS NOT RE-DERIVED HERE. Measured over five real
#: lifted rows at 20,000 steps each, 218,841,786 state words: the Pade tree DOES
#: reach the subnormal band hard (1,031,503 subnormal words on one row, 792,830 of
#: them named intermediates) and it is PROVABLY INVISIBLE — flushing what the tree
#: itself produces never diverged, because the band cannot propagate past the
#: ``1 + ...`` additions. The numerator, the denominator, ``one_plus_c2``,
#: ``one_plus_two_c2`` and ``u`` all censused ZERO subnormal words and ``u`` stayed
#: in [0.998, 1.000]. A flushed subnormal perturbs ``c3`` by at most
#: ``1.18e-38 * chi3 * |chi1inv|^3`` while half an ulp at 1.0 is 5.96e-8, so the
#: crossing sits at ``chi3 * |chi1inv|^3 = 1e31`` — which a needle leg located.
#:
#: This constant is TWO DECADES UNDER that crossing. Above it the plan REFUSES BY
#: NAME rather than stepping a configuration whose flush could become visible.
CHI_MAGNITUDE_CEILING = 1e29


# ---------------------------------------------------------------------------
# The shader
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void nonlinear_constitutive_step(
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
    device const float* q20     [[buffer(12)]],
    device const float* q21     [[buffer(13)]],
    device const float* q22     [[buffer(14)]],
    device const float* q30     [[buffer(15)]],
    device const float* q31     [[buffer(16)]],
    device const float* q32     [[buffer(17)]],
    device const float* chis    [[buffer(18)]],
    device const float* kp0     [[buffer(19)]],
    device const float* km0     [[buffer(20)]],
    device const float* kp1     [[buffer(21)]],
    device const float* km1     [[buffer(22)]],
    device const float* kp2     [[buffer(23)]],
    device const float* km2     [[buffer(24)]],
    constant uint&      nx      [[buffer(25)]],
    constant uint&      ny      [[buffer(26)]],
    constant uint&      nz      [[buffer(27)]],
    constant uint&      n_elem  [[buffer(28)]],
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
    // The four-corner sum's two shifts go in OPPOSITE directions, so one axis can
    // be read both ways inside one component's term and the corner sample needs
    // the up flag of one axis with the down flag of another. The flags stay
    // separate and are ANDed at the load site.
    int di = i - 1, dj = j - 1, dk = k - 1;
    int ui = i + 1, uj = j + 1, uk = k + 1;
    bool dvx = true, dvy = true, dvz = true;
    bool uvx = true, uvy = true, uvz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    // The coefficient index is on the component's OWN axis: kps_x/kms_x for
    // target 0, _y for 1, _z for 2 (E_CONSTITUTIVE_TERMS, stepping.py:227). That
    // is MEEP's dsigw, NOT the dsig/dsigu cycle the curl recurrence uses.
    float kp_0 = kp0[i], km_0 = km0[i];
    float kp_1 = kp1[j], km_1 = km1[j];
    float kp_2 = kp2[k], km_2 = km2[k];

    // --- component 0: Ex — own axis x; partners Dy then Dz ---------------------
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

    // --- component 1: Ey — own axis y; partners Dz then Dx ---------------------
    float prev1 = w1[ii];
__SRC1__
    w1[ii] = src1;
    float a1 = f1[ii];
    a1 = a1 + kp_1 * src1;
    a1 = a1 - km_1 * prev1;
    f1[ii] = a1;

    // --- component 2: Ez — own axis z; partners Dx then Dy ---------------------
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

#: The chi volume buffer names, per component, in kernel argument order.
_CHI2_BUFFERS: Tuple[str, ...] = ("q20", "q21", "q22")
_CHI3_BUFFERS: Tuple[str, ...] = ("q30", "q31", "q32")

#: Where each component's SCALAR chi lives in the packed six-float buffer.
_CHI2_SLOTS: Tuple[int, int, int] = (0, 1, 2)
_CHI3_SLOTS: Tuple[int, int, int] = (3, 4, 5)

#: The packed scalar buffer's length. Bound even when every arm is a volume, so the
#: signature never changes shape — a kernel whose ARGUMENT COUNT depended on the
#: specialisation would make the plan's argument tuple specialisation-dependent
#: too, and that is one more place a mispairing can hide.
CHI_SCALAR_SLOTS = 6


def _index(shifts: Dict[int, str]) -> str:
    """A flat index with the named per-axis substitutions applied."""
    parts = [_STRIDE[axis].format(shifts.get(axis, _HOME[axis]))
             for axis in range(3)]
    return " + ".join(parts)


def _two_way_ghost(axis: str, code: int) -> str:
    """One axis's ghost rule for a stencil that reads BOTH neighbours.

    METALLIC serves an exact 0.0 in both directions — past a perfect electric
    conductor there is no field (``stepping._shift_down``:1827-1829,
    ``_shift_up``:1781-1783) — and the flag is what the load's ternary reads, never
    a clamped load. PERIODIC wraps both ways; at k = 0 no Bloch factor multiplies
    the wrapped plane (``_shift_up``:1768-1771 skips the multiply entirely), which
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


def _four_point_sum(component: int, partner_axis: int, tag: str) -> str:
    """MEEP's ``g1s``/``g2s`` for one partner — ``stepping._nonlinear_transverse_sums``
    (:1158-1162) as it ASSOCIATES them::

        pair = g + shift_down(g, partner_axis)      -> g[i] + g[i-s1]
        gNs  = pair + shift_up(pair, own_axis)      -> ... + (g[i+s] + g[i+s-s1])

    so the bytes are ``(g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1])`` — THE PAIR
    ASSOCIATION, not MEEP C's left-to-right sum (step_generic.cpp:646-648). The far
    pair is the near pair's elementwise sum moved up one cell, so loading the two
    far corners and adding them is the same bits as shifting the formed pair: a
    shift is data movement, and no float rounds between the two spellings.

    THE CORNER TAKES BOTH AXES' RULES. ``shift_up`` is applied to the ALREADY
    SHIFTED pair, so the partner-axis down-flag is evaluated at ``i+s`` rather than
    at ``i`` — and those are the same flag, because a shift along the OWN axis does
    not move the partner-axis coordinate. That is why ANDing the two flags at the
    load site reproduces applying them in sequence.

    A metallic own axis at the top row makes ``far`` an exact ``0.0f + 0.0f``,
    which is precisely the ``+0.0`` plane ``_shift_up`` writes there (:1782).
    """
    own_axis = E_TERMS[component][2]
    down, _up, dvalid, _uvalid, _extent = _TWO_WAY_NAMES["xyz"[partner_axis]]
    _odown, oup, _odvalid, ouvalid, _oextent = _TWO_WAY_NAMES["xyz"[own_axis]]
    partner_volume = f"g{partner_axis}"

    down_index = _index({partner_axis: down})
    up_index = _index({own_axis: oup})
    corner_index = _index({own_axis: oup, partner_axis: down})
    return "\n".join([
        f"    float near_{tag} = {partner_volume}[ii]",
        f"        + ({dvalid} ? {partner_volume}[{down_index}] : 0.0f);",
        f"    float far_{tag} = ({ouvalid} ? {partner_volume}[{up_index}] : 0.0f)",
        f"        + (({ouvalid} && {dvalid}) ? {partner_volume}[{corner_index}]"
        f" : 0.0f);",
        f"    float sum_{tag} = near_{tag} + far_{tag};",
    ])


def _component_source(component: int, nonlinear: Sequence[int],
                      chi2_volume: Sequence[int],
                      chi3_volume: Sequence[int]) -> str:
    """The whole ``src{c}`` block for one component, on the arm its chi selects.

    THE LINEAR ARM IS THE CERTIFIED PLAIN BODY VERBATIM — ``gs * us``, MEEP's
    ``else if (u)`` branch (stepping.py:1100-1102). That is what makes a partly
    nonlinear run's linear components byte-identical to the ordinary constitutive
    kernel rather than merely close to it, and it is why the per-component split
    the array path performs at runtime is a compile-time specialisation here.
    """
    gs = f"    float gs{component} = g{component}[ii];"
    us = f"    float us{component} = e{component}[ii];"
    if not nonlinear[component]:
        return "\n".join([gs, us,
                          f"    float src{component} = gs{component}"
                          f" * us{component};"])

    lines = [gs, us]
    tags = []
    for offset in (0, 1):
        partner_axis = TRANSVERSE_PARTNERS[component][offset]
        tag = f"{component}{offset}"
        lines.append(_four_point_sum(component, partner_axis, tag))
        tags.append(tag)

    # Dsqr = gs*gs + 0.0625*(g1s*g1s + g2s*g2s) — stepping.py:1147, with 0.0625
    # scaling the SUM and applied last.
    lines.append(
        f"    float dsqr{component} = gs{component} * gs{component}"
        f" + 0.0625f * ((sum_{tags[0]} * sum_{tags[0]})"
        f" + (sum_{tags[1]} * sum_{tags[1]}));")

    chi2 = (f"{_CHI2_BUFFERS[component]}[ii]" if chi2_volume[component]
            else f"chis[{_CHI2_SLOTS[component]}]")
    chi3 = (f"{_CHI3_BUFFERS[component]}[ii]" if chi3_volume[component]
            else f"chis[{_CHI3_SLOTS[component]}]")
    lines.append(f"    float chi2_{component} = {chi2};")
    lines.append(f"    float chi3_{component} = {chi3};")

    # calc_nonlinear_u (stepping.py:1054-1056), term for term. The powers are
    # formed in float32 because `us` is a VOLUME — a scalar epsilon would make
    # NumPy round the double power once and is refused by name (docstring point 7).
    lines.append(f"    float us_sq{component} = us{component} * us{component};")
    lines.append(f"    float us_cu{component} = (us{component} * us{component})"
                 f" * us{component};")
    lines.append(f"    float c2_{component} = (gs{component} * chi2_{component})"
                 f" * us_sq{component};")
    lines.append(f"    float c3_{component} = (dsqr{component} * chi3_{component})"
                 f" * us_cu{component};")
    lines.append(f"    float num{component} = (1.0f + c2_{component})"
                 f" + 2.0f * c3_{component};")
    lines.append(f"    float den{component} = (1.0f + 2.0f * c2_{component})"
                 f" + 3.0f * c3_{component};")
    # THE PLAIN OPERATOR, measured IEEE round-to-nearest on this host (docstring
    # point 1). `fast::divide` diverges by up to 2 ulp and is the must-catch
    # mutation; Triton's div_rn workaround exists for a reason that does not
    # reproduce here and is deliberately NOT ported.
    lines.append(f"    float u{component} = num{component} / den{component};")
    lines.append(f"    float src{component} = (gs{component} * us{component})"
                 f" * u{component};")
    return "\n".join(lines)


def _normalize(nonlinear: Sequence[int], chi2_volume: Sequence[int],
               chi3_volume: Sequence[int]
               ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Canonicalise a specialisation triple, forcing dead flags to 0.

    A chi arm flag on a LINEAR component names an operand the emitted source never
    reads, so leaving it free would make two different keys compile the same string
    — and a memo keyed by the source would then hide the disagreement while a
    fingerprint enumeration counted it twice. One canonical spelling.
    """
    live = tuple(int(bool(flag)) for flag in nonlinear)
    if len(live) != 3:
        raise ValueError(f"nonlinear must be a per-component triple, got "
                         f"{nonlinear!r}")
    second = tuple(int(bool(flag)) if live[c] else 0
                   for c, flag in enumerate(chi2_volume))
    third = tuple(int(bool(flag)) if live[c] else 0
                  for c, flag in enumerate(chi3_volume))
    if len(second) != 3 or len(third) != 3:
        raise ValueError(f"chi arms must be per-component triples, got "
                         f"{chi2_volume!r} and {chi3_volume!r}")
    return live, second, third


def nonlinear_source(nonlinear: Sequence[int], chi2_volume: Sequence[int],
                     chi3_volume: Sequence[int], codes: Sequence[int],
                     contract: str = templates.CONTRACT_OFF) -> str:
    """The specialised ``nonlinear_constitutive_step`` source.

    ``nonlinear`` is the per-component liveness triple, ``chi2_volume`` /
    ``chi3_volume`` say whether that component's coefficient is a VOLUME (else the
    packed scalar slot), and ``codes`` is the per-axis PERIODIC/METALLIC ghost
    triple ``_boundary_kinds`` resolves. All four are baked into the string, as
    Triton bakes its constexprs.
    """
    live, second, third = _normalize(nonlinear, chi2_volume, chi3_volume)
    codes = tuple(int(code) for code in codes)
    if not any(live):
        raise ValueError(
            "no component is nonlinear: that configuration is the certified plain "
            "constitutive kernel's, and emitting this source for it would overlap "
            "the two families")
    if len(codes) != 3:
        raise ValueError(f"codes is a per-axis triple, got {codes!r}")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__GUARD__": templates.GUARD,
        "__GHOST_X__": _two_way_ghost("x", codes[0]),
        "__GHOST_Y__": _two_way_ghost("y", codes[1]),
        "__GHOST_Z__": _two_way_ghost("z", codes[2]),
        "__SRC0__": _component_source(0, live, second, third),
        "__SRC1__": _component_source(1, live, second, third),
        "__SRC2__": _component_source(2, live, second, third),
    })


def compile_nonlinear(nonlinear: Sequence[int], chi2_volume: Sequence[int],
                      chi3_volume: Sequence[int], codes: Sequence[int],
                      contract: str = templates.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (arms, boundaries, mode)."""
    return compile_source(
        nonlinear_source(nonlinear, chi2_volume, chi3_volume, codes, contract)
    ).nonlinear_constitutive_step


# ---------------------------------------------------------------------------
# The specialisation corpus and its fingerprint
# ---------------------------------------------------------------------------

_TRIPLES: Tuple[Tuple[int, int, int], ...] = tuple(
    (x, y, z) for x in (templates.PERIODIC, templates.METALLIC)
    for y in (templates.PERIODIC, templates.METALLIC)
    for z in (templates.PERIODIC, templates.METALLIC))


def specialisations() -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...],
                                     Tuple[int, ...], Tuple[int, ...]], ...]:
    """Every (nonlinear, chi2_volume, chi3_volume, codes) tuple this family emits.

    Per component there are FIVE canonical arms — linear, or nonlinear with each of
    the four (chi2 volume?, chi3 volume?) combinations — so 5^3 = 125 component
    triples, minus the all-linear one this family refuses toward the certified plain
    kernel, times 8 boundary triples: **992 sources**. Small enough to enumerate and
    too many to list one sha256 each, so :func:`corpus_digest` hashes the whole
    enumeration into one value, exactly as the off-diagonal family does.
    """
    out = []
    per_component = []
    for live in (0, 1):
        if not live:
            per_component.append((0, 0, 0))
            continue
        for q2 in (0, 1):
            for q3 in (0, 1):
                per_component.append((1, q2, q3))
    for arm0 in per_component:
        for arm1 in per_component:
            for arm2 in per_component:
                live = (arm0[0], arm1[0], arm2[0])
                if not any(live):
                    continue
                second = (arm0[1], arm1[1], arm2[1])
                third = (arm0[2], arm1[2], arm2[2])
                for codes in _TRIPLES:
                    out.append((live, second, third, codes))
    return tuple(out)


def corpus_digest(contract: str = templates.CONTRACT_OFF) -> Dict[str, Any]:
    """One sha256 over every specialisation, in canonical order, plus the count.

    A digest rather than a list, for the reason :func:`specialisations` gives. It
    is a FINGERPRINT, not a certification: what a byte gate certifies is the subset
    it launched, and a gate records those individually.
    """
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    count = 0
    for live, second, third, codes in specialisations():
        digest.update(nonlinear_source(live, second, third, codes,
                                       contract).encode())
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


def nonlinear_components(fields: Any) -> Tuple[int, int, int]:
    """Which of the three components take the Pade correction, as a 0/1 triple.

    Asked of ``Fields.is_nonlinear`` (fields.py:973) and of nothing else, so the
    predicate, the plan and the emitted source cannot disagree about which
    components branch. ``is_nonlinear`` is membership in ``_chi2_components``, and
    the installer stores BOTH or NEITHER per component (fields.py:838-885, MEEP's
    structure.cpp:815-816 rule), which is what makes one flag enough for two
    coefficients.
    """
    return tuple(  # type: ignore[return-value]
        int(bool(_call(fields, "is_nonlinear", term[0], default=False)))
        for term in E_TERMS)


def chi_pair_for(fields: Any, component: str) -> Tuple[Any, Any]:
    """``(chi2, chi3)`` for one component — each a float or a float32 volume.

    ``Fields.chi2_for``/``chi3_for`` (fields.py:976-980) answer 0.0 for a linear
    component, which is MEEP's ``s->chi2[ec]`` semantics; this family never reads
    those, because the linear arm compiles the plain body.
    """
    return (_call(fields, "chi2_for", component, default=0.0),
            _call(fields, "chi3_for", component, default=0.0))


def chi_volume_arms(fields: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Per component, whether chi2 / chi3 arrived as a VOLUME rather than a scalar."""
    live = nonlinear_components(fields)
    second: List[int] = []
    third: List[int] = []
    for index, term in enumerate(E_TERMS):
        chi2, chi3 = chi_pair_for(fields, term[0])
        second.append(int(bool(live[index]) and _is_volume(chi2)))
        third.append(int(bool(live[index]) and _is_volume(chi3)))
    return tuple(second), tuple(third)


def _magnitude(chi: Any, inverse_epsilon: Any, power: int) -> Optional[float]:
    """``max |chi| * |chi1inv|^power`` over the grid, in float64, or None.

    IN FLOAT64 DELIBERATELY. The quantity being bounded is allowed to be up to
    1e29, and ``chi3 * |chi1inv|^3`` at the ceiling would OVERFLOW float32 while
    being computed — a check that overflows to ``inf`` and then refuses everything
    is not the clause, it is a different clause with the same name.

    Returns ``None`` when either operand cannot be read as a number or an array, so
    the caller refuses BY NAME rather than this function guessing a magnitude.
    """
    import numpy as np  # noqa: PLC0415

    try:
        chi_values = np.abs(np.asarray(chi, dtype=np.float64))
        epsilon_values = np.abs(np.asarray(inverse_epsilon, dtype=np.float64))
    except Exception:  # noqa: BLE001 - an unreadable operand is not a magnitude
        return None
    if not np.all(np.isfinite(chi_values)) or not np.all(
            np.isfinite(epsilon_values)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        product = chi_values * (epsilon_values ** power)
    if not product.size:
        return None
    return float(np.max(product))


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------

def _nonlinear_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The shared clause list with CLAUSE 10 INVERTED, and nothing else changed.

    ``registry.py``'s contract is that every family's grid-reason list carries the
    same numbered questions and answers exactly ONE of them the other way, and that
    a new family STATES its inversion rather than inheriting disjointness. This is
    the cleanest single inversion in the package:

    * ``coverage._grid_reasons`` clause 10 (coverage.py:191-195) refuses chi2/chi3
      on EVERY sub-step — "the Pade factor REPLACES the constitutive product
      (stepping.py:999-1000) and composes with dispersion, so it is refused on every
      sub-step rather than only on update_E". Nine carried families take that
      clause or restate it;
    * **HERE a nonlinearity is REQUIRED**, refused by name when absent.

    That one inversion separates this predicate from every other family in the
    table, including the off-diagonal ``update_E`` arm it shares a slot with — that
    family restates the shared clause 10 verbatim in
    ``offdiag_constitutive_coverage``'s ``_grid_reasons`` call, so the two are
    disjoint by an inverted clause rather than by an engine coupling. There is no
    planted-configuration caveat here of the kind the constitutive/offdiag pair
    needs: ``has_nonlinearity`` is the same property ``update_E`` itself branches
    on (stepping.py:989, :999), so a run that reaches the Pade body reaches this
    predicate.

    EVERY OTHER CLAUSE IS UN-INVERTED AND IS WHAT KEEPS THE UNCARRIED ROWS
    UNCARRIED, deliberately and by name:

    * clause 2 (real storage) KEPT — the DOCMP split (stepping.py:1084-1091) runs
      the whole nonlinear update once per Cartesian part with ``u`` built from real
      parts only, which is a second body and not a dtype change. ``nonlinear_complex``
      stays uncarried WITH A REASON on this side;
    * clause 5 (no fold) KEPT — ``fold_real_2d_nonlinear`` stays uncarried, and the
      fold interacts with this family specifically rather than generically:
      ``Fields._require_chi2_respects_the_mirror_planes`` (fields.py:887) refuses a
      chi2 on a component a plane makes ODD, because ``c2`` is linear in D and
      therefore carries D's parity while ``c3`` is even. A folded nonlinear kernel
      has to carry that asymmetry;
    * clauses 3, 4, 6, 7, 11, 12 KEPT verbatim.
    """
    reasons: List[str] = []

    # 1. The Metal backend, and the host array module the mirrors copy from.
    #    IMPORTED from this package's own coverage module: one backend clause for
    #    the whole package, so a family cannot quietly admit a run on a host whose
    #    subnormal policy the MPS executor refuses. THAT CLAUSE IS LOAD-BEARING
    #    HERE IN A WAY IT IS NOT ELSEWHERE — see clause (g) below for why this
    #    family's flush exposure is bounded by a magnitude ceiling rather than by
    #    the policy alone.
    reasons.extend(_metal_backend_reasons(grid))

    # 2. Real storage. KEPT UN-INVERTED. MEEP nonlinearizes the two parts of a
    #    complex field INDEPENDENTLY (update_eh.cpp:149's DOCMP loop; transcribed
    #    at stepping.py:1110-1116), so a complex arm is a second kernel body, not
    #    a wider dtype. `nonlinear_complex` is uncarried and this is its refusal.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: MEEP nonlinearizes the two "
                       "parts of a complex field independently (the DOCMP split, "
                       "stepping.py:1084-1091), which is a second kernel body "
                       "rather than a wider dtype — no product carries it")

    # 3. An absorber that actually absorbs. WITHOUT ONE update_E IS A DIFFERENT
    #    SUB-STEP — `field[...] = constitutive` (stepping.py:1019-1022) with no
    #    dsigw accumulation and no f_w at all — so this is not merely an
    #    optimisation boundary.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (these kernels implement the "
                       "split-field path only)")

    # 4. Only the two ghost rules the four-corner sum reads.
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

    # 5. No mirror plane anywhere. KEPT UN-INVERTED — `fold_real_2d_nonlinear`
    #    stays uncarried. See the docstring for why the fold is a real interaction
    #    with THIS family rather than a generic one.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian only.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0. A Bloch phase needs complex storage and multiplies one wrapped
    #    plane; the four-corner sum would carry it on two axes at once.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. INVERTED. THE ONE CLAUSE THIS FAMILY ANSWERS THE OTHER WAY.
    if not getattr(fields, "has_nonlinearity", False):
        reasons.append(
            "no chi2/chi3 is installed: a linear run is bit-identically the "
            "certified plain constitutive kernel's (MEEP deletes the trivial "
            "pair, structure.cpp:822-826, transcribed at fields.py:879-880) and "
            "belongs to coverage.constitutive_coverage(side='E') — this "
            "predicate must not overlap it")

    # 11/12. BFAST adds a second additive term to every curl; beta adds
    #        out-of-plane couplings. Neither reaches update_E, and both are KEPT
    #        because coverage is a set per sub-step and not a la carte.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not "
                       "carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    return reasons


def _chi_reasons(fields: Any, shape: Sequence[int]) -> List[str]:
    """Every live chi2/chi3 must be a finite scalar or a real f32 grid volume.

    AND MUST NOT ALIAS A WRITTEN VOLUME. This is the off-diagonal family's measured
    hazard 2, and it applies here for exactly the same mechanism: the residency
    registry keys mirrors by NAME, so one host array registered as ``Dz``
    (non-constant, because ``step_D`` writes it) and again as ``chi3:Ez``
    (constant, because nothing writes a coefficient) becomes TWO device tensors.
    ``step_D``'s device write lands in the first; the second keeps the bytes it was
    born with, and the kernel reads a coefficient frozen at step 0. The off-diagonal
    family measured that as byte-identical for ONE sub-step and 19,620 differing
    words over two complete steps — invisible to a single-launch gate, which is why
    it is refused in the predicate rather than left to a gate to notice.
    """
    out: List[str] = []
    outputs: Dict[int, str] = {}
    for term in E_TERMS:
        for name in (term[0], "f_w_" + term[0]):
            address = _base_address(getattr(fields, name, None))
            if address is not None:
                outputs.setdefault(address, name)
    sources: Dict[int, str] = {}
    for _target, source, _axis in E_TERMS:
        address = _base_address(getattr(fields, source, None))
        if address is not None:
            sources.setdefault(address, source)

    live = nonlinear_components(fields)
    for index, term in enumerate(E_TERMS):
        if not live[index]:
            continue
        component = term[0]
        for label, value in zip(("chi2", "chi3"),
                                chi_pair_for(fields, component)):
            name = f"{label}[{component}]"
            if not _is_volume(value):
                # THE SCALAR ARM, admitted because a Python float meeting a
                # float32 array is a weak scalar rounded ONCE to float32 under
                # NEP 50 — measured 0 differing words over seven scalars spanning
                # 1e-30 to 3.4e38 (probe_metal_divide_and_negation, leg
                # host_scalar_rounding). Contrast the inverse-epsilon powers,
                # which are a DOUBLE expression and are refused (docstring point 7).
                try:
                    scalar = float(value)
                except Exception:  # noqa: BLE001 - unreadable is refused by name
                    out.append(f"{name} is neither a volume nor a readable float")
                    continue
                if scalar != scalar or scalar in (float("inf"), float("-inf")):
                    out.append(f"{name} is not finite")
                continue
            dtype = str(getattr(value, "dtype", ""))
            if not dtype:
                out.append(f"{name} exposes no dtype; a buffer width that cannot "
                           f"be read cannot be bound")
                continue
            if dtype.startswith("c"):
                out.append(f"{name} is complex; an instantaneous chi is real")
                continue
            address = _base_address(value)
            if address is not None and address in outputs:
                out.append(
                    f"{name} aliases output {outputs[address]}: the four-corner "
                    f"sum re-reads the partner D volumes at neighbour offsets "
                    f"while the outputs are written, so an alias makes the answer "
                    f"depend on thread schedule")
            elif address is not None and address in sources:
                out.append(
                    f"{name} aliases source {sources[address]}: the residency "
                    f"registry keys mirrors by NAME, so one host array bound as a "
                    f"written source and again as a read-only coefficient becomes "
                    f"two device tensors and the coefficient freezes at plan time")
            if len(shape) == 3:
                out.extend(_coverage._volume_reasons(name, value, shape))
    return out


def _magnitude_reasons(fields: Any) -> List[str]:
    """Clause (g): the plan-time magnitude ceiling on the INSTALLED volumes.

    A COVERAGE REFUSAL, NOT A TOLERANCE, and the distinction is the whole point: a
    tolerance would let a run proceed and be judged afterwards, while this refuses
    the configuration by name and leaves the slot on the array path, where the
    engine's own float64-free arithmetic is the arbiter.

    See :data:`CHI_MAGNITUDE_CEILING` for the settled science. It is NOT re-derived
    here and the 20,000-step census is NOT re-run: the crossing was located at
    ``chi3 * |chi1inv|^3 = 1e31`` and this clause sits two decades under it.

    THE QUANTITY IS PER-CELL, not a product of two independent maxima. ``chi`` and
    ``chi1inv`` live on the same grid, so ``max_i(|chi_i| * |chi1inv_i|^p)`` is both
    the right question and strictly tighter than ``max|chi| * max|chi1inv|^p``; the
    looser form would refuse runs whose large chi and large epsilon are nowhere
    near each other.
    """
    out: List[str] = []
    live = nonlinear_components(fields)
    for index, term in enumerate(E_TERMS):
        if not live[index]:
            continue
        component = term[0]
        inverse_epsilon = _call(fields, "inverse_epsilon_for", component,
                                default=None)
        if inverse_epsilon is None:
            continue  # the layout clauses refuse this by name already
        chi2, chi3 = chi_pair_for(fields, component)
        for label, chi, power in (("chi2", chi2, 2), ("chi3", chi3, 3)):
            magnitude = _magnitude(chi, inverse_epsilon, power)
            if magnitude is None:
                out.append(
                    f"{label}[{component}] * |chi1inv|^{power} could not be "
                    f"evaluated; the subnormal-visibility ceiling is a plan-time "
                    f"check on the installed volumes and an unevaluable one is "
                    f"refused rather than assumed to pass")
                continue
            if magnitude > CHI_MAGNITUDE_CEILING:
                out.append(
                    f"{label}[{component}] * |chi1inv|^{power} reaches "
                    f"{magnitude:.3e}, above this family's ceiling of "
                    f"{CHI_MAGNITUDE_CEILING:.0e}. Metal FLUSHES subnormals and "
                    f"cannot be made not to, and the Pade tree's flush is "
                    f"invisible only while the band cannot propagate past the "
                    f"1 + ... additions; the measured crossing is 1e31 and this "
                    f"ceiling is two decades under it. Refused by name rather "
                    f"than stepped to a tolerance")
    return out


def nonlinear_constitutive_coverage(fields: Any, pml: Any,
                                    residency: Any = None
                                    ) -> "_coverage.Coverage":
    """May the Metal Pade kernel step ``update_E`` for this pair?

    POSITIVE CLAUSES ONLY; a failing clause appends its reason and the scan
    continues. The grid clauses are :func:`_nonlinear_grid_reasons`' — the shared
    list with clause 10 INVERTED and every other clause kept, spelled out there
    with the reason each one stays.

    THE E-SIDE CLAUSES:

    a. **no registered polarization.** With poles the source is ``D - sum P``
       formed in per-component scratch buffers (fields.py:1107-1138) and both the
       Pade factor and its transverse sums would read THOSE, not D. That is the
       fused dispersive+nonlinear leg, and nothing on this backend carries it.
    b. **no off-diagonal chi1inv row.** MEEP's MOST GENERAL CASE scales the WHOLE
       row product by the Pade factor — ``fw = (gs*us + OFFDIAG + OFFDIAG) * u``
       (stepping.py:1095-1099) — so the two features do not compose by being
       admitted together; they need a third kernel body. ``offdiag_update_e``
       refuses a nonlinearity through the shared clause 10, so a run carrying both
       is uncovered WITH A REASON FROM BOTH SIDES rather than silently.
    c. **at least one component actually nonlinear**, counted over
       :func:`nonlinear_components` — deliberately not the bare
       ``has_nonlinearity`` flag alone, for the same reason the off-diagonal family
       counts live row slots: the flag and the per-component answer are two reads
       of one dict, and the emitted source branches on the per-component answer.
    d. **stored E** — belt and braces with a reason.
    e. every live chi2/chi3 a finite scalar or a real f32 grid-shaped volume that
       aliases neither an E/f_w OUTPUT nor a D SOURCE (:func:`_chi_reasons`).
    f. VOLUME inverse epsilon per component — this kernel carries NO scalar
       epsilon arm, and that is a NUMERICAL clause: see module docstring point 7
       and the 51,218-word measurement behind it.
    g. **the magnitude ceiling** (:func:`_magnitude_reasons`) — a plan-time check
       on the installed volumes, refused by name.
    h. layout, the half-integer coefficient tables, and the RESIDENCY declaration.

    Conductivity is NOT a clause, deliberately: it changes the CURL sub-steps only,
    never the constitutive one.

    WHAT THIS FAMILY DELIBERATELY DOES NOT CLAIM, stated because a silent omission
    and a decision look the same from outside. A nonlinear run's ``step_B``,
    ``step_D`` and ``update_H`` slots are refused by the SHARED clause 10, which
    calls chi2/chi3 uncarried on every sub-step. Those three are arguably
    over-broad — the curl differences the stored E and H arrays and never reads
    epsilon, and this engine refuses a magnetic chi outright (fields.py:869-873) —
    but clause 10 lives in ``coverage._grid_reasons``, which is pinned to the
    Triton clause list by ``test_grid_reasons_match_the_triton_clause_list``.
    Widening it is a change to nine families' predicates and to a cross-port pin,
    which is a separate measured change and not a side effect of this one. The
    slots stay refused, with a reason, and the over-breadth is FILED rather than
    fixed here — the same disposition the round took when Metal's admitted set was
    found to exceed Triton's by nine slots.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = _nonlinear_grid_reasons(fields, pml, grid)
    reasons.extend(_coverage._susceptibility_reasons(fields))
    reasons.extend(_coverage._residency_declaration_reasons(residency))

    # (a) No poles: the Pade factor must read the aliased D primaries.
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if states or getattr(fields, "has_polarizations", False):
        reasons.append(
            "a susceptibility is registered: the nonlinear source becomes "
            "D - sum P in per-component scratch buffers (fields.py:1107-1138) — "
            "the fused dispersive+nonlinear kernel is a later leg")

    # (b) No off-diagonal row: MEEP's most-general case is a third body.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed: MEEP's most-general case "
            "scales the WHOLE row product by the Pade factor "
            "(stepping.py:1095-1099), which is a third kernel body — no product "
            "on this backend carries the intersection")

    # (c) INVERTED clause 10's per-component half.
    if not any(nonlinear_components(fields)):
        reasons.append(
            "no component answers is_nonlinear(): the emitted source branches "
            "per component, so a run whose flag is set while every component is "
            "linear has no body to compile and belongs to "
            "constitutive_coverage(side='E')")

    # (d) Stored E.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    shape = tuple(getattr(grid, "shape", ()))

    # (e) The live coefficients, readable and well-formed.
    reasons.extend(_chi_reasons(fields, shape))

    # (g) The magnitude ceiling.
    reasons.extend(_magnitude_reasons(fields))

    # (f)/(h) The volumes this sub-step reads and writes, and their layout.
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

class MetalNonlinearConstitutivePlan(KernelPlan):
    """A launchable, allocation-free chi2/chi3 ``update_E``.

    The Yee sub-lattice is chosen at the BUILDERS and nowhere else:
    ``kps_a_h``/``kms_a_h``, the half-integer tables (stepping.py:1015). The kernel
    takes the coefficient pointers and never asks which lattice they came from — a
    swap is the silent half-cell absorber error every constitutive family's gate
    carries a mutation for, and this one carries it too.

    THE SIX CHI VOLUME SLOTS ARE BOUND IN COMPONENT ORDER, chi2 then chi3; a slot
    on a linear component, or on one whose coefficient is a scalar, is bound to
    that component's own D mirror and never read (the specialisation is what stops
    the read, the buffer still has to bind). THE PACKED SCALAR BUFFER IS ALWAYS
    BOUND, at full length, whether or not any arm reads it — a signature whose
    ARGUMENT COUNT depended on the specialisation would make the argument tuple
    specialisation-dependent too, which is one more place a mispairing can hide.

    ``__init__`` REFUSES ALIASING between the outputs (E, f_w) and any input (D,
    inverse epsilon, chi volumes, the six coefficient vectors), and among the
    outputs themselves: the four-corner sum re-reads the partner D volumes at
    neighbour offsets while E and f_w are being written, so an aliased pair would
    make the result depend on thread schedule — a wrong answer that varies run to
    run. An all-linear specialisation is refused toward the certified plain kernel.
    """

    __slots__ = ("shape", "n_elem", "nonlinear", "chi2_volume", "chi3_volume",
                 "boundary_codes", "residency", "volumes", "chi_scalars")

    REPR_FIELDS = ("shape", "nonlinear", "chi2_volume", "chi3_volume",
                   "boundary_codes")

    def __init__(self, shape, residency, targets, auxiliaries, sources,
                 inverse_epsilon, chi2_slots, chi3_slots, chi_scalars,
                 coefficients, nonlinear, chi2_volume, chi3_volume,
                 boundary_codes, functions: Dict[str, Any],
                 volumes: Sequence[str], host_chi: Sequence[Any] = (),
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
        live, second, third = _normalize(nonlinear, chi2_volume, chi3_volume)
        self.nonlinear = live
        self.chi2_volume = second
        self.chi3_volume = third
        if not any(live):
            raise ValueError(
                "no component is nonlinear: that configuration belongs to the "
                "certified plain constitutive kernel, and building this plan for "
                "it would overlap the two")
        self.residency = residency
        self.volumes = tuple(volumes)
        self.chi_scalars = tuple(float(value) for value in chi_scalars)
        if len(self.chi_scalars) != CHI_SCALAR_SLOTS:
            raise ValueError(f"the packed scalar buffer holds "
                             f"{CHI_SCALAR_SLOTS} floats, got "
                             f"{len(self.chi_scalars)}")

        self._require_no_aliasing(host_arrays, host_chi)

        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(inverse_epsilon) + tuple(chi2_slots) + tuple(chi3_slots)
            + (chi_scalars_tensor(chi_scalars, residency),) + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem))

    @staticmethod
    def _require_no_aliasing(host_arrays: Sequence[Any],
                             host_chi: Sequence[Any]) -> None:
        """Base-address disjointness of outputs vs outputs and inputs vs outputs.

        ``host_arrays`` is ``(targets, auxiliaries, sources, inverse_epsilon,
        coefficients)`` as HOST arrays — the device tensors cannot answer the
        question, because two mirrors of the same host array are the SAME tensor by
        the residency registry's design, and that is legal for constants and fatal
        for outputs. Equality of BASE addresses only: two overlapping views with
        different bases pass unseen, which is the certified plans' shared and
        accepted limitation.

        READ-ONLY INPUTS MAY ALIAS EACH OTHER FREELY — three aliased
        inverse-epsilon volumes are the isotropic install, and one chi array
        installed on two components is a legal medium — because those are CONSTANT
        mirrors of one host array and nothing writes them. What they may NOT alias
        is a volume this plan mirrors as WRITTEN: an OUTPUT (E, f_w) or a SOURCE
        (D). ``sources`` counts as a write-side name here even though this kernel
        only reads it, because ``step_D`` writes it in the composition the
        residency layer exists to serve.
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
                    f"an input volume aliases {outputs[address]}: the four-corner "
                    f"sum re-reads the partner D volumes at neighbour offsets "
                    f"while the outputs are written, so an alias makes the answer "
                    f"depend on thread schedule")
            if address is not None:
                written.setdefault(address, f"source {index}")
        read_only = list(inverse_epsilon)
        read_only += [value for value in host_chi if _is_volume(value)]
        read_only += [value for value in coefficients if _is_volume(value)]
        for array in read_only:
            address = _base_address(array)
            if address is not None and address in written:
                raise ValueError(
                    f"a read-only input volume aliases {written[address]}: an "
                    f"OUTPUT alias makes the answer depend on thread schedule, "
                    f"and a SOURCE alias gives one host array both a written and "
                    f"a constant device mirror — so the coefficient freezes at "
                    f"plan time and the field goes smoothly, plausibly wrong one "
                    f"complete step later")


def chi_scalars_tensor(values: Sequence[float], residency: Any) -> Any:
    """The packed six-float scalar buffer as a device mirror.

    MIRRORED BY VALUE-DERIVED NAME, not by a fixed one. Two plans in one
    composition may carry different scalar chi sets — a two-medium run is ordinary
    — and the residency registry refuses one name bound to two host arrays, so a
    fixed name would turn a legal second plan into a raise. The name embeds the
    float32 words, so identical scalar sets share one mirror and different ones do
    not collide.
    """
    import numpy as np  # noqa: PLC0415

    packed = np.ascontiguousarray(
        np.asarray(values, dtype=np.float32).reshape(-1))
    if packed.size != CHI_SCALAR_SLOTS:
        raise ValueError(f"the packed scalar buffer holds {CHI_SCALAR_SLOTS} "
                         f"floats, got {packed.size}")
    key = packed.view(np.uint32).tobytes().hex()
    return residency.mirror(f"nonlinear:chi_scalars:{key}", packed, constant=True)


def _functions(nonlinear, chi2_volume, chi3_volume, codes,
               contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_nonlinear(nonlinear, chi2_volume, chi3_volume, codes,
                                    mode)
            for mode in contract_variants}


def _packed_scalars(fields: Any) -> Tuple[float, ...]:
    """The six chi values in packed-buffer order: chi2 x/y/z then chi3 x/y/z.

    A VOLUME arm contributes 0.0 in its slot — the emitted source reads the volume
    buffer there and never the packed one, so the value is unread; it is written as
    a definite 0.0 rather than left uninitialised so the mirror name (which hashes
    the words) is a function of the configuration and not of whatever was in memory.
    """
    second: List[float] = []
    third: List[float] = []
    live = nonlinear_components(fields)
    for index, term in enumerate(E_TERMS):
        chi2, chi3 = chi_pair_for(fields, term[0])
        second.append(0.0 if (not live[index] or _is_volume(chi2))
                      else float(chi2))
        third.append(0.0 if (not live[index] or _is_volume(chi3))
                     else float(chi3))
    return tuple(second) + tuple(third)


def plan_nonlinear_constitutive(fields: Any, pml: Any, residency: Any = None,
                                contract_variants: Sequence[str] = (
                                    templates.CONTRACT_OFF,),
                                ) -> Optional[MetalNonlinearConstitutivePlan]:
    """Build the chi2/chi3 ``update_E`` plan from the engine's own objects.

    None means REFUSED, and the reasons come from
    :func:`nonlinear_constitutive_coverage`. The sources are taken through
    ``displacement_minus_polarization_volumes`` — the accessor the array path
    itself reads (stepping.py:1069) — which with no poles admitted hands back the
    aliased D primaries (fields.py:1107-1138).
    """
    if not nonlinear_constitutive_coverage(fields, pml, residency).covered:
        return None
    kinds = _boundary_kinds(fields.grid, pml)
    codes = tuple(BOUNDARY_CODES[kind] for kind in kinds)
    targets = tuple(term[0] for term in E_TERMS)
    volumes = fields.displacement_minus_polarization_volumes()
    live = nonlinear_components(fields)
    chi2_volume, chi3_volume = chi_volume_arms(fields)
    chi_values = [chi_pair_for(fields, name) for name in targets]

    host_targets = [getattr(fields, n) for n in targets]
    host_aux = [getattr(fields, "f_w_" + n) for n in targets]
    host_sources = [volumes[n] for n in targets]
    host_inv = [fields.inverse_epsilon_for(n) for n in targets]
    host_coefficients = [getattr(pml, f"{stem}_{axis}_h")
                         for axis in "xyz" for stem in ("kps", "kms")]
    host_chi = [pair[half] for pair in chi_values for half in (0, 1)]

    mirrored_targets = [residency.mirror(n, a) for n, a in zip(targets, host_targets)]
    mirrored_aux = [residency.mirror("f_w_" + n, a)
                    for n, a in zip(targets, host_aux)]
    mirrored_sources = [residency.mirror(term[1], a)
                        for term, a in zip(E_TERMS, host_sources)]
    mirrored_inv = [residency.mirror("inv_eps_" + n, a, constant=True)
                    for n, a in zip(targets, host_inv)]

    # A DEAD CHI SLOT BINDS THE COMPONENT'S OWN D MIRROR. The specialisation is
    # what stops the read; the buffer argument still has to bind, and binding the
    # D mirror rather than a fresh allocation keeps the plan allocation-free.
    def _chi_slot(index: int, half: int, flag: int) -> Any:
        if not flag:
            return mirrored_sources[index]
        label = ("chi2", "chi3")[half]
        return residency.mirror(f"{label}:{targets[index]}",
                                chi_values[index][half], constant=True)

    chi2_slots = [_chi_slot(index, 0, chi2_volume[index]) for index in range(3)]
    chi3_slots = [_chi_slot(index, 1, chi3_volume[index]) for index in range(3)]
    mirrored_coefficients = [
        residency.mirror(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"),
                         constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    volume_names = (tuple(targets) + tuple("f_w_" + n for n in targets)
                    + tuple(term[1] for term in E_TERMS))
    return MetalNonlinearConstitutivePlan(
        fields.grid.shape, residency, mirrored_targets, mirrored_aux,
        mirrored_sources, mirrored_inv, chi2_slots, chi3_slots,
        _packed_scalars(fields), mirrored_coefficients,
        live, chi2_volume, chi3_volume, codes,
        _functions(live, chi2_volume, chi3_volume, codes, contract_variants),
        volume_names,
        host_chi=host_chi,
        host_arrays=(host_targets, host_aux, host_sources, host_inv,
                     host_coefficients))


def plan_nonlinear_constitutive_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any],
        chi: Dict[str, Dict[str, Any]], boundary_codes: Sequence[int],
        residency: Any, functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (templates.CONTRACT_OFF,),
        ) -> MetalNonlinearConstitutivePlan:
    """Build it from bare host arrays — the gate's and probe's route.

    ``arrays`` is keyed by component name (``Ex``, ``f_w_Ex``, ``Dx``,
    ``inv_eps_Ex``), ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE
    CALLER already selected, and ``chi`` maps component -> ``{"chi2": ..., "chi3":
    ...}`` with an absent component marking a LINEAR one. No coverage predicate
    runs here: the caller is a harness that constructed the configuration on
    purpose, including the deliberately wrong ones.

    ``functions=`` IS LOAD-BEARING: the mutation legs route a mutated kernel
    through it, and a builder that drops it launches the SHIPPED kernel and reports
    a pass for a defect it never introduced.
    """
    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)

    live = tuple(int(name in chi) for name in targets)
    pairs = [((chi.get(name) or {}).get("chi2", 0.0),
              (chi.get(name) or {}).get("chi3", 0.0)) for name in targets]
    chi2_volume = tuple(int(bool(live[i]) and _is_volume(pairs[i][0]))
                        for i in range(3))
    chi3_volume = tuple(int(bool(live[i]) and _is_volume(pairs[i][1]))
                        for i in range(3))
    packed = tuple(
        0.0 if (not live[i] or _is_volume(pairs[i][0])) else float(pairs[i][0])
        for i in range(3)) + tuple(
        0.0 if (not live[i] or _is_volume(pairs[i][1])) else float(pairs[i][1])
        for i in range(3))

    host_targets = [arrays[n] for n in targets]
    host_aux = [arrays["f_w_" + n] for n in targets]
    host_sources = [arrays[term[1]] for term in E_TERMS]
    host_inv = [arrays["inv_eps_" + n] for n in targets]
    host_coefficients = [flat[f"{stem}_{axis}"]
                         for axis in "xyz" for stem in ("kps", "kms")]
    host_chi = [pair[half] for pair in pairs for half in (0, 1)]

    mirrored_targets = [residency.mirror(n, a) for n, a in zip(targets, host_targets)]
    mirrored_aux = [residency.mirror("f_w_" + n, a)
                    for n, a in zip(targets, host_aux)]
    mirrored_sources = [residency.mirror(term[1], a)
                        for term, a in zip(E_TERMS, host_sources)]
    mirrored_inv = [residency.mirror("inv_eps_" + n, a, constant=True)
                    for n, a in zip(targets, host_inv)]

    def _chi_slot(index: int, half: int, flag: int) -> Any:
        if not flag:
            return mirrored_sources[index]
        label = ("chi2", "chi3")[half]
        return residency.mirror(f"{label}:{targets[index]}", pairs[index][half],
                                constant=True)

    chi2_slots = [_chi_slot(index, 0, chi2_volume[index]) for index in range(3)]
    chi3_slots = [_chi_slot(index, 1, chi3_volume[index]) for index in range(3)]
    mirrored_coefficients = [
        residency.mirror(f"pml:{stem}_{axis}:nonlinear", flat[f"{stem}_{axis}"],
                         constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    volume_names = (tuple(targets) + tuple("f_w_" + n for n in targets)
                    + tuple(term[1] for term in E_TERMS))
    return MetalNonlinearConstitutivePlan(
        shape, residency, mirrored_targets, mirrored_aux, mirrored_sources,
        mirrored_inv, chi2_slots, chi3_slots, packed, mirrored_coefficients,
        live, chi2_volume, chi3_volume, boundary_codes,
        functions if functions is not None
        else _functions(live, chi2_volume, chi3_volume, boundary_codes,
                        contract_variants),
        volume_names,
        host_chi=host_chi,
        host_arrays=(host_targets, host_aux, host_sources, host_inv,
                     host_coefficients))


# ---------------------------------------------------------------------------
# The nonlinear PML spine — existing arithmetic, inverted admission only
# ---------------------------------------------------------------------------

class _LinearScopeView:
    """Expose a nonlinear run to an unchanged ordinary-kernel predicate.

    The two ordinary PML kernels do not read ``has_nonlinearity``.  They are still
    deliberately refused by the shared predicate because its clause applies to the
    full step.  This view inverts only that one admission clause while forwarding
    every array, PML coefficient, boundary, conductivity, dispersion, and layout
    fact unchanged.  It is used for both the predicate and the reused builder so a
    covered verdict can never reach a builder that re-applies the broader clause.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        if name == "has_nonlinearity":
            return False
        return getattr(object.__getattribute__(self, "_fields"), name)


def _nonlinear_spine_reasons(fields: Any) -> List[str]:
    """The inverted χ²/χ³ clause shared by all three unaffected sub-steps."""
    try:
        active = bool(getattr(fields, "has_nonlinearity", False))
    except Exception as exc:  # noqa: BLE001 - unreadable state is a refusal
        return [f"fields.has_nonlinearity raised {exc!r}"]
    if not active:
        return ["no chi2/chi3 is installed: the ordinary PML arm owns this "
                "linear configuration"]
    return []


def nonlinear_run_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                    residency: Any = None) -> "_coverage.Coverage":
    """Whether an ordinary Metal curl may step a nonlinear run's B or D side.

    ``stepping.step_B`` and ``stepping.step_D`` only read their curl inputs and
    split-field coefficients.  The Pade term enters later in ``update_E``.  This
    is consequently an admission-only port: it requires a nonlinear material, then
    delegates every other check to the existing ordinary PML curl predicate through
    :class:`_LinearScopeView`.
    """
    if sub_step not in _coverage.CURL_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(_coverage.CURL_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))
    reasons = _nonlinear_spine_reasons(fields)
    if reasons:
        return _coverage.Coverage(False, tuple(reasons))
    return _coverage.pml_curl_coverage(_LinearScopeView(fields), pml, sub_step,
                                        residency)


def nonlinear_run_constitutive_coverage(fields: Any, pml: Any, side: str,
                                        residency: Any = None) -> "_coverage.Coverage":
    """Whether the ordinary magnetic constitutive shader may step a nonlinear run.

    The spine owns ``update_H`` only.  ``update_E`` is rejected by name because it
    evaluates the χ²/χ³ Pade factor and belongs to
    :func:`nonlinear_constitutive_coverage`.
    """
    if side not in _coverage.CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(_coverage.CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))
    reasons = _nonlinear_spine_reasons(fields)
    if side == "E":
        reasons.append("side='E' evaluates the χ²/χ³ Pade factor and belongs to "
                       "the nonlinear update_E arm")
    if reasons:
        return _coverage.Coverage(False, tuple(reasons))
    return _coverage.constitutive_coverage(_LinearScopeView(fields), pml, side,
                                            residency)


def plan_nonlinear_run_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Any = None,
        contract_variants: Sequence[str] = (templates.CONTRACT_OFF,),
        ) -> Any:
    """Build the certified ordinary PML curl for one nonlinear B/D sub-step."""
    if not nonlinear_run_pml_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    from . import launch  # noqa: PLC0415 - avoids the registration import cycle

    return launch.plan_pml_curl(_LinearScopeView(fields), pml, sub_step, residency,
                                contract_variants)


def plan_nonlinear_run_constitutive(
        fields: Any, pml: Any, side: str, residency: Any = None,
        contract_variants: Sequence[str] = (templates.CONTRACT_OFF,),
        ) -> Any:
    """Build the certified ordinary magnetic constitutive plan for a nonlinear run."""
    if not nonlinear_run_constitutive_coverage(fields, pml, side, residency).covered:
        return None
    from . import launch  # noqa: PLC0415 - avoids the registration import cycle

    return launch.plan_constitutive(_LinearScopeView(fields), pml, side, residency,
                                    contract_variants)


# ---------------------------------------------------------------------------
# Registration — WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> "_coverage.Coverage":
    return nonlinear_constitutive_coverage(context.fields, context.pml,
                                           context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalNonlinearConstitutivePlan]:
    return plan_nonlinear_constitutive(context.fields, context.pml,
                                       context.residency,
                                       context.contract_variants)


def _spine_curl_coverage(context: Any, slot: str) -> "_coverage.Coverage":
    return nonlinear_run_pml_curl_coverage(context.fields, context.pml, slot,
                                           context.residency)


def _spine_curl_plan(context: Any, slot: str) -> Any:
    return plan_nonlinear_run_pml_curl(context.fields, context.pml, slot,
                                       context.residency, context.contract_variants)


def _spine_h_coverage(context: Any, slot: str) -> "_coverage.Coverage":
    return nonlinear_run_constitutive_coverage(context.fields, context.pml, "H",
                                               context.residency)


def _spine_h_plan(context: Any, slot: str) -> Any:
    return plan_nonlinear_run_constitutive(context.fields, context.pml, "H",
                                           context.residency,
                                           context.contract_variants)


SPINE_ARMS = tuple(
    arms.register(
        family=FAMILY, slot=slot, label="nonlinear PML curl",
        coverage=_spine_curl_coverage, plan=_spine_curl_plan,
        prefix="nonlinear PML curl: ", noun="nonlinear PML curl",
        wired=True)
    for slot in ("step_B", "step_D")) + (
    arms.register(
        family=FAMILY, slot="update_H", label="nonlinear PML magnetic",
        coverage=_spine_h_coverage, plan=_spine_h_plan,
        prefix="nonlinear PML magnetic: ", noun="nonlinear PML magnetic constitutive",
        wired=True),
)


ARM = arms.register(
    family=FAMILY, slot=SLOT, label="nonlinear",
    coverage=_arm_coverage, plan=_arm_plan,
    prefix="nonlinear: ", noun="chi2/chi3 Pade constitutive",
    # WIRED. Its disjointness from the other two ``update_E`` arms rests on an
    # INVERTED CLAUSE in both directions, which is the cleanest case in the table:
    # ``constitutive`` and ``offdiag_constitutive`` both take the shared clause 10
    # (chi2/chi3 refused on every sub-step), and this predicate requires it. There
    # is no engine-coupling caveat of the kind the constitutive/offdiag pair needs,
    # because ``has_nonlinearity`` is the same property ``stepping.update_E``
    # itself branches on — the run that reaches the Pade body is the run that
    # reaches this predicate.
    wired=True)
