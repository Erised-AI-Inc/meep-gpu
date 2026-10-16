"""CYLINDRICAL (Dcyl) curl sub-steps at m = 0 on Metal — real float32 storage.

WHAT IT REPLACES. ``stepping.step_B`` and ``stepping.step_D`` on a Dcyl grid with
``grid.m == 0``, real storage and an active split-field PML, including MEEP's radial
sum ``f_rderiv_int``, which runs on the device here.

THE CURL IS THE CERTIFIED ONE. :func:`cylindrical_curl_source` derives the kernel from
:func:`.shaders.curl_source` by four checked edits, so the bounds block, the ghost
rule, the stencil of targets 0 and 1 (MEEP's left-to-right bracket, with the phi
partner absent: the record's presence bit for the one-cell phi axis is clear), the
ownership mask and the two per-chunk PML forms are the certified text. The r axis
compiles METALLIC: ``_shift_up``'s CYL_AXIS arm is the METALLIC arm, and the
CYL_AXIS near ghost (``_shift_down``) feeds only Dy and Dz, whose r-Yee shift is 0, so
the ownership mask zeroes the one row that reads it. The cylindrical additions are
three, in the array path's order:

1. **Bz and Dz differentiate the per-chunk radial sum** (step_db.cpp:92-119,
   ``stepping.cylindrical_rderiv_difference``). MEEP sums ``f_rderiv_int`` in every
   structure chunk from that chunk's first row, ``F[s] = +0``, ``F[g] = F[g-1] +
   rinv[g] * ((f_p[g]*w[g]) - (f_p[g-1]*w[g-1]))`` in float32, left to right, with
   ``rinv = realnum(1.0 / ((g + ir0) - 0.5))`` MULTIPLIED. The scan kernel walks each
   (phi, z) column serially, restarts the sum where the bound chunk starts name a new
   chunk, and stores the stencil difference a target row reads from its own chunk's
   sum: ``F_s[j+1] - F_s[j]`` for Bz (forward, over Ep extended by its zero wall row)
   and ``F_s[g-1] - F_s[g]`` for Dz (backward). The cut row between two chunks holds
   two sums, so the scan stores differences, not a prefix. The curl's target 2 is
   then ``dtdx * rad[ii]`` on both sides.
2. **The chunk starts are the array path's own** (``stepping._cylindrical_scan_starts``
   for Bz or Dz, read on the target's Yee lattice), bound as an ``int32`` vector, one
   start per r row. A division whose r cuts differ along z has no per-row start and is
   refused by name (:func:`scan_starts`).
3. **The m = 0 axis rules**, after the forms: ``Bx[r=0] = 0``; ``Dz[r=0] +=
   (4*Courant)*Hp[r=0]`` then ``Dy[r=0] = 0`` (``stepping._cylindrical_axis_zero_B`` /
   ``_D``). MEEP forms the on-axis product in double and rounds it once
   (step_db.cpp:324); the kernel's float32 product with the float32 word of
   ``4*Courant`` is that word exactly when ``Courant`` is a float32 word, so the D-side
   predicate refuses any other Courant by name (:func:`courant_reasons`) until a
   software double product exists on this backend.

WHAT IS METAL-SPECIFIC: source specialisation by substitution (direction, boundary
triple); the contraction pragma, reached through :mod:`.templates`; the chunk record
bound as ``constant ChunkBounds&`` after the two cylindrical bindings (23 in all,
under the 31 ceiling of :data:`.device.MAX_BUFFER_BINDINGS`). Subnormals flush on this
backend, so every claim rides on the checked subnormal-free precondition
(:mod:`.subnormal`, through ``coverage._metal_backend_reasons``). There is no
generated-code audit on this backend (``compile_shader`` exposes no AIR), so every
claim here is behavioural.

WIRED. This module registers two curl arms and the certified constitutive pair in
:mod:`.arms`. What separates these arms from every other family is clause 6 INVERTED
(``coverage._grid_reasons`` refuses ``cylindrical`` and ``is_axis`` by name, and this
family requires both); clause 2 (real storage) separates it from the cylindrical
complex family, and clause 14 (``m == 0``) from the engine's own refusal of real
storage at ``|m| >= 1``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
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
# The SAME table `metal_kernels.launch` binds, imported from the Triton package for
# the reason `bfast_curl` imports it: the two ports cannot then disagree about which
# targets, sources, direction or Yee suffix a sub-step has, and this module acquires
# no import edge into the Metal composer.
from ..triton_kernels.launch import SUB_STEPS
from . import shaders, templates
# Bound as `_chunk_bounds`: the plan builders take the RECORD as a keyword named
# `chunk_bounds`, the name the composer's StepContext carries (launch.py does the same).
from . import chunk_bounds as _chunk_bounds
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source

# `.plans.KernelPlan` is DELIBERATELY NOT IMPORTED. This family's curl plan is a
# SIBLING of that base rather than a subclass — see `CylindricalRealCurlPlan` — so an
# import here would be dead weight that reads as an inheritance the class does not
# have. The reason it cannot subclass is stated at the class, not hidden here.

#: The family name the arm table carries. One spelling, so a refusal message, a
#: registry row and an artifact column cannot drift apart.
FAMILY = "cylindrical_real"

__all__ = [
    "CURL_RECORD_BUFFER",
    "CYLINDRICAL_CURL2",
    "FAMILY",
    "SCAN_TARGETS",
    "CylindricalRealCurlPlan",
    "axis_coefficient",
    "compile_cylindrical_curl",
    "compile_scan",
    "courant_reasons",
    "cylindrical_real_constitutive_coverage",
    "cylindrical_real_curl_coverage",
    "cylindrical_curl_source",
    "enumerate_cylindrical_real_sources",
    "int32_tensor",
    "plan_cylindrical_real_constitutive",
    "plan_cylindrical_real_curl",
    "plan_cylindrical_real_curl_from_arrays",
    "prefix_row_vectors",
    "register_arms",
    "scan_rows",
    "scan_source",
    "scan_start_reasons",
    "scan_starts",
]

PERIODIC = shaders.PERIODIC
METALLIC = shaders.METALLIC

#: ``ir0 = origin_r*a + 0.5*iyee_shift(Fp).in_direction(R)`` per sub-step
#: (step_db.cpp:99-103, ``stepping.step_B`` / ``step_D``). The r origin IS the axis
#: here, so the whole value is the half-shift: Ep sits at the node (r-shift 0) and Hp
#: half a cell out (r-shift 1). Getting these two backwards is a HALF-CELL error in
#: the radial weights and is silent, which is why they are a table keyed on the
#: sub-step rather than a literal at each call site.
PREFIX_IR0: Dict[str, float] = {"step_B": 0.0, "step_D": 0.5}

#: Which sub-step's scan carries the ZERO WALL ROW. Only the B side: its Bz curl is a
#: forward difference of the radial sum, and MEEP stores the wall row with Ep held at
#: zero (``stepping.step_B``), so the last row reads ``F[wall] - F[last]`` rather than
#: minus the whole sum. The D side's backward difference never looks past the top.
PREFIX_WALL_ROW: Dict[str, bool] = {"step_B": True, "step_D": False}


def scan_rows(sub_step: str, radial_rows: int) -> int:
    """How many rows this sub-step's radial scan covers — ``nr`` or ``nr + 1``."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    return int(radial_rows) + (1 if PREFIX_WALL_ROW[sub_step] else 0)


# ---------------------------------------------------------------------------
# The radial scan: MEEP's f_rderiv_int per chunk, emitted as its difference
# ---------------------------------------------------------------------------

#: ``stepping.cylindrical_rderiv_difference`` as one thread per (phi, z) column.
#: The thread walks the column's rows strictly serially (float32 addition is not
#: associative, so a blocked or parallel scan is another number), forming each
#: row's increment ``rinv[g-1] * (w[g] - w[g-1])`` with ``w = f_p * weights`` and
#: adding it to the running sum of the chunk that owns the target row. The sum
#: restarts at ``+0`` where ``starts`` names a new chunk (step_db.cpp:104-118), and
#: the kernel stores the stencil difference the curl reads, not the sum: the cut
#: row between two chunks holds two sums (the lower chunk's, carried one row
#: past its last owned row, and the upper chunk's ``+0``), so no single prefix
#: volume represents the division.
#:
#: ``rinv`` is MULTIPLIED, as MEEP stores and multiplies it; dividing by
#: ``(g + ir0) - 0.5`` is another float32 word on part of the ladder.
_SCAN_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void cyl_rderiv_difference(
    device float*       out     [[buffer(0)]],
    device const float* src     [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* rinv    [[buffer(3)]],
    device const int*   starts  [[buffer(4)]],
    constant uint&      nx      [[buffer(5)]],
    constant uint&      ny      [[buffer(6)]],
    constant uint&      nz      [[buffer(7)]],
    constant uint&      n_cols  [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    // One thread per (phi, z) column. The dispatch is sized from `out`, which is
    // nx times wider, so most threads return here.
    if (idx >= n_cols) { return; }
    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int k = int(idx) % nzi;
    int j = int(idx) / nzi;
    if (j >= nyi) { return; }
    int nyz = nyi * nzi;
    int base = j * nzi + k;

__LOOP__
}
"""

#: The B side (Bz, forward): target rows ``j`` read ``F_s[j+1] - F_s[j]`` of the
#: chunk that owns them, over Ep extended by its zero wall row (row ``nx`` reads
#: the literal ``0.0f`` the array path assigns). A new chunk's first target row is
#: its first row ``s`` (checked on the host, :func:`scan_starts`).
_FORWARD_LOOP = """    float w_prev = src[base] * weights[0];
    float acc = 0.0f;              // F_s[j]: +0 on the chunk's first row
    int owner = 0;
    for (int g = 0; g < nxi; ++g) {
        int s = starts[g];
        if (s != owner) { acc = 0.0f; owner = s; }
        float w = (g + 1 < nxi ? src[(g + 1) * nyz + base] : 0.0f) * weights[g + 1];
        float inc = rinv[g] * (w - w_prev);
        float nxt = acc + inc;     // F_s[j+1], the owning chunk's sum
        out[g * nyz + base] = nxt - acc;
        acc = nxt;
        w_prev = w;
    }"""

#: The D side (Dz, backward, D's negated stride): target rows ``g`` read
#: ``F_s[g-1] - F_s[g]``; row 0, the axis, reads ``+0`` (the per-m axis rules own
#: it). A new chunk's first target row is one past its first row ``s``, so its
#: ``F_s[g-1]`` is the restarted ``+0`` (checked on the host, :func:`scan_starts`).
_BACKWARD_LOOP = """    out[base] = 0.0f;
    float w_prev = src[base] * weights[0];
    float acc = 0.0f;              // F_s[g-1]: +0 on the chunk's first row
    int owner = 0;
    for (int g = 1; g < nxi; ++g) {
        int s = starts[g];
        if (s != owner) { acc = 0.0f; owner = s; }
        float w = src[g * nyz + base] * weights[g];
        float inc = rinv[g - 1] * (w - w_prev);
        float nxt = acc + inc;     // F_s[g]
        out[g * nyz + base] = acc - nxt;
        acc = nxt;
        w_prev = w;
    }"""

_SCAN_LOOPS: Dict[str, str] = {"step_B": _FORWARD_LOOP, "step_D": _BACKWARD_LOOP}


def scan_source(sub_step: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """The radial-scan source for one sub-step (its direction and wall row are baked in)."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    return templates.substitute(_SCAN_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__LOOP__": _SCAN_LOOPS[sub_step],
    })


def compile_scan(sub_step: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised radial-scan entry point for one sub-step."""
    return compile_source(scan_source(sub_step, contract)).cyl_rderiv_difference


def prefix_row_vectors(sub_step: str, radial_rows: int, dtype: Any = None
                       ) -> Tuple[Any, Any]:
    """The two row vectors the scan binds — ``(weights, rinv)``, FLAT.

    ``stepping._cylindrical_rderiv_rows`` is CALLED, not restated: ``weights[g] =
    realnum(g + ir0)`` and ``rinv[g-1] = realnum(1.0 / ((g + ir0) - 0.5))``, the
    quotient in double and rounded once, over the sub-step's scan rows (``nr + 1``
    on the B side, whose source carries the wall row).
    """
    import numpy as np  # noqa: PLC0415
    from ..stepping import _cylindrical_rderiv_rows  # noqa: PLC0415

    resolved = np.float32 if dtype is None else np.dtype(dtype).type
    weights, rinv = _cylindrical_rderiv_rows(
        np, scan_rows(sub_step, radial_rows), PREFIX_IR0[sub_step], resolved)
    return (np.ascontiguousarray(weights).reshape(-1),
            np.ascontiguousarray(rinv).reshape(-1))


#: The target each sub-step's scan serves: its chunk starts are read on the target's
#: own Yee lattice (a Dz cut row belongs to the lower chunk, a Bz one to the upper).
SCAN_TARGETS: Dict[str, str] = {"step_B": "Bz", "step_D": "Dz"}


def scan_starts(sub_step: str, starts: Any, radial_rows: int) -> Any:
    """The scan's ``int32`` start vector from ``stepping._cylindrical_scan_starts``'s answer.

    ``starts`` is that function's value for the sub-step's target: ``None`` (one chunk
    from the axis), an ``(nr, 1, 1)`` line, or a full volume. Refused by name
    (``ValueError``): a full volume (r cuts that differ along z, which a per-row start
    cannot describe), a line of the wrong length, a first chunk that does not start at
    the axis, and a chunk change the column walk cannot take — on the B side the new
    chunk's first target row must be its first row ``s``, on the D side ``s + 1``,
    which is what MEEP's ownership gives on a division cut at whole cells.
    """
    import numpy as np  # noqa: PLC0415

    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    rows = int(radial_rows)
    if starts is None:
        return np.zeros(rows, dtype=np.int32)
    get = getattr(starts, "get", None)
    host = np.asarray(get() if callable(get) else starts)
    if host.ndim != 3 or host.shape[1:] != (1, 1):
        raise ValueError(
            f"the chunk starts have shape {host.shape}: the division's r cuts differ "
            f"along z, and the column walk reads one start per row")
    line = host.reshape(-1).astype(np.int64)
    if line.shape[0] != rows:
        raise ValueError(f"the chunk starts hold {line.shape[0]} rows, not {rows}")
    backward = bool(SUB_STEPS[sub_step]["backward"])
    first = 1 if backward and rows > 1 else 0
    if line[first] != 0:
        raise ValueError(f"the first chunk starts at row {int(line[first])}, not at the axis")
    for row in range(first + 1, rows):
        if line[row] == line[row - 1]:
            continue
        expected = row - 1 if backward else row
        if line[row] != expected:
            raise ValueError(
                f"row {row} changes chunk to one starting at row {int(line[row])}; the "
                f"column walk restarts the sum there only for a chunk starting at row "
                f"{expected}")
    return np.ascontiguousarray(line.astype(np.int32))


# ---------------------------------------------------------------------------
# The curl: the certified body with the cylindrical substitution
# ---------------------------------------------------------------------------

#: The chunk record's binding on this family's curl: two past the certified curl's
#: (``shaders.CURL_RECORD_BUFFER``), after the radial difference and the axis scalar.
CURL_RECORD_BUFFER = shaders.CURL_RECORD_BUFFER + 2

#: The bindings the curl takes in front of the record, in order.
_CURL_EXTRA_BINDINGS = (
    f"    device const float* rad       [[buffer({shaders.CURL_RECORD_BUFFER})]],\n"
    f"    constant float&     axis_coef [[buffer({shaders.CURL_RECORD_BUFFER + 1})]],\n")

#: The certified curl's target-2 stencil, which the cylindrical curl replaces: Bz and
#: Dz differentiate the per-chunk radial sum (stepping.step_B / step_D, the Z case),
#: and their phi partner is the derivative MEEP drops.
_CERTIFIED_CURL2 = (
    "    float curl2 = has_x ? (has_y ? (dtdx * (((b_x - b) + a) - a_y)) : (dtdx * (b_x - b)))\n"
    "                        : (has_y ? ((-dtdx) * (a_y - a)) : 0.0f);\n")

#: ``part_products.scale(dtdx, radial)``: one product of the radial difference.
CYLINDRICAL_CURL2 = "    float curl2 = dtdx * rad[ii];\n"

#: The certified body's first guarded auxiliary store. The per-chunk forms end above
#: it, so the m = 0 axis rules go in front of it, on the registers.
_FORMS_END = "    if (du0) { u0[ii] = n0; }\n"

#: The m = 0 axis rules after the forms (stepping._cylindrical_axis_zero_B / _D):
#: Bx[r=0] = 0 on the B side; on the D side the on-axis Dz increment, then Dp = 0.
#: ``axis_coef`` is the float32 word of ``4 * Courant`` and ``g1`` the stored Hp:
#: MEEP's ``dfcnd = g[i] * (Courant * 4)`` is a double product rounded once
#: (step_db.cpp:324), the same word as this float32 product exactly when ``Courant``
#: is a float32 word, which the predicate requires (:func:`courant_reasons`).
_AXIS_TAILS: Dict[str, str] = {
    "step_B": ("    // stepping._cylindrical_axis_zero_B: Bx[r = 0] = 0, the field only.\n"
               "    v0 = at_x ? 0.0f : v0;\n"),
    "step_D": ("    // stepping._cylindrical_axis_zero_D: Dz[r = 0] += 4*Courant*Hp[r = 0],\n"
               "    // then Dy[r = 0] = 0, the fields only.\n"
               "    v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;\n"
               "    v1 = at_x ? 0.0f : v1;\n"),
}


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(
            f"the certified curl source carries {what} ({old.strip()!r}) {count} times, "
            f"not once; the cylindrical curl cannot be derived from it")
    return text.replace(old, new, 1)


def cylindrical_curl_source(sub_step: str, codes: Sequence[int],
                            contract: str = shaders.CONTRACT_OFF) -> str:
    """The cylindrical curl source: :func:`.shaders.curl_source` with four checked edits.

    The certified curl for the sub-step's direction and the ``codes`` triple, with:
    the entry point renamed ``cyl_pml_curl_step``; the radial difference ``rad`` and
    the axis scalar bound in front of the chunk record; target 2's stencil replaced by
    ``dtdx * rad[ii]``; the m = 0 axis rules inserted after the forms. Everything else
    — the bounds block, the ghost rule, targets 0 and 1 with their absent phi partner,
    the ownership mask and the per-chunk forms — is the certified text. Each edit must
    land exactly once.

    ``codes`` is the per-axis PERIODIC/METALLIC triple with the r axis METALLIC: a
    PERIODIC r axis is refused by name rather than emitted with its far ghost wrapping
    onto the axis row.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if codes[0] != METALLIC:
        raise ValueError(
            f"the r axis must compile as METALLIC (code {METALLIC}), got {codes[0]!r}: "
            f"CYL_AXIS shares _shift_up's metallic zero ghost, and a PERIODIC r axis "
            f"would wrap the far face onto the axis row")
    backward = bool(SUB_STEPS[sub_step]["backward"])
    source = shaders.curl_source(codes, backward, contract)
    source = _replace_once(source, "kernel void pml_curl_step(",
                           "kernel void cyl_pml_curl_step(", "its entry point")
    record = (f"    constant ChunkBounds& cb [[buffer({shaders.CURL_RECORD_BUFFER})]],\n")
    source = _replace_once(
        source, record,
        _CURL_EXTRA_BINDINGS
        + f"    constant ChunkBounds& cb [[buffer({CURL_RECORD_BUFFER})]],\n",
        "its chunk-record binding")
    source = _replace_once(source, _CERTIFIED_CURL2, CYLINDRICAL_CURL2,
                           "its target-2 stencil")
    return _replace_once(source, _FORMS_END, _AXIS_TAILS[sub_step] + "\n" + _FORMS_END,
                         "its first guarded auxiliary store")


def compile_cylindrical_curl(sub_step: str, codes: Sequence[int],
                             contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised cylindrical curl entry point for one configuration."""
    return compile_source(
        cylindrical_curl_source(sub_step, codes, contract)).cyl_pml_curl_step


def enumerate_cylindrical_real_sources(contract: str = shaders.CONTRACT_OFF
                                       ) -> Dict[str, str]:
    """Every shipped specialisation, keyed by a stable label.

    TEN sources: two scans (one per sub-step) and eight curls. The curl enumeration
    pins ``codes[0] = METALLIC`` because the predicate admits no other r axis. The
    chunk division and its starts are bound data, so the set does not grow with the
    grid.
    """
    out: Dict[str, str] = {}
    for sub_step in ("step_B", "step_D"):
        out[f"cyl_rderiv_difference/{sub_step}"] = scan_source(sub_step, contract)
        for cy in (PERIODIC, METALLIC):
            for cz in (PERIODIC, METALLIC):
                label = f"cyl_pml_curl_step/{sub_step}/{METALLIC}{cy}{cz}"
                out[label] = cylindrical_curl_source(
                    sub_step, (METALLIC, cy, cz), contract)
    return out


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors ``coverage._grid_reasons`` so the two can be diffed.
# Clause 6 (CARTESIAN ONLY) is INVERTED: this product REQUIRES a Dcyl grid with the
# r axis declared, where every shipped kernel refuses both BY NAME. Clause 14 is
# NEW and is this family's internal split against a future |m| >= 1 product. Clause 4
# is RESTATED to admit 'axis' on the r axis and nowhere else. Everything else is
# KEPT.

#: Which axis of the engine's (r, phi, z) storage is the cylindrical axis. Named
#: rather than spelled ``0`` at five call sites, because "the radial axis is the
#: leading one" is a storage convention (stepping.py:1300-1301) and a reader should
#: be able to find every place this family depends on it.
RADIAL_AXIS = 0


def _cylindrical_real_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every Metal cylindrical m = 0 predicate shares."""
    reasons: List[str] = []

    # 1. The Metal backend, and the host array module the mirrors copy from.
    #    IMPORTED from this package's own coverage module: one backend clause for the
    #    whole package, so a family cannot quietly admit a run on a host whose
    #    subnormal policy the MPS executor refuses.
    reasons.extend(_metal_backend_reasons(grid))

    # 2. Real storage REQUIRED, and KEPT UN-INVERTED. This is THE clause that
    #    separates this family from the cylindrical COMPLEX one, in the SAME
    #    direction the shipped real families are separated from `complex_fields`.
    #    It is not implied by clause 14: a complex-storage m = 0 run is perfectly
    #    constructible (`force_complex_fields` is an independent switch), this kernel
    #    would read the wrong stride on it, and since 2026-09-04 the complex family
    #    carries it (its `M_ZERO` arm) — so the two families partition Dcyl by
    #    storage alone.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: complex-storage cylindrical is "
                       "the cylindrical COMPLEX family's kernel at every m (its "
                       "M_ZERO arm carries m = 0), not this one")

    # 3. An absorber that actually absorbs (split-field family only).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "split-field path only)")

    # 4. RESTATED, and this is a WIDENING that must be spelled PER AXIS. On a Dcyl
    #    grid `_boundary_kinds` returns 'axis' for r (stepping.py:2192), which is
    #    outside COVERED_BOUNDARIES, so the shared clause would refuse every
    #    configuration this family exists for. 'axis' is admitted on the RADIAL axis
    #    only: an axis-kind phi or z would mean the grid disagrees with this family
    #    about which direction is radial, and admitting it globally would compile the
    #    wrong ghost onto two axes silently.
    kinds = _boundary_kinds(grid, pml if (pml is not None
                                          and getattr(pml, "is_active", False))
                            else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            allowed = (("periodic", "metallic", "axis") if axis == RADIAL_AXIS
                       else ("periodic", "metallic"))
            if kind not in allowed:
                reasons.append(f"axis {axis} boundary {kind!r} is outside {allowed}")
        if kinds[RADIAL_AXIS] != "axis":
            reasons.append(
                f"axis {RADIAL_AXIS} resolved to {kinds[RADIAL_AXIS]!r}, not 'axis': "
                f"this product's Bz/Dz substitution IS the cylindrical radial "
                f"derivative and is a wrong answer on a Cartesian leading axis")

    # 5. No mirror plane anywhere, KEPT — and it must be asked TWICE, which is not
    #    tidiness. `_boundary_kinds` puts is_axis AHEAD of is_mirrored
    #    (stepping.py:2191-2195), so a folded r axis reports CYL_AXIS and THE FOLD
    #    VANISHES SILENTLY from the boundary triple; and `_mirror_phases`
    #    (:2291-2308) puts (-1)**grid.m into the SAME SLOT the mirror phase occupies.
    #    The boundary kinds alone therefore cannot answer this question.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried "
                       "by this family, and a folded r axis is INVISIBLE in the "
                       "boundary kinds — is_axis outranks is_mirrored)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6 (INVERTED). Cylindrical REQUIRED, where `coverage._grid_reasons` :176-182
    #    refuses both `cylindrical` and `is_axis` by name. This single inversion is
    #    what separates this family from ALL NINE carried ones, because every one of
    #    them either uses that clause list or restates it.
    if not getattr(grid, "cylindrical", False):
        reasons.append("grid.cylindrical is False: this product exists only for "
                       "Dcyl runs; a Cartesian run belongs to the certified plain "
                       "kernels")
    for axis in range(3):
        declared = bool(_call(grid, "is_axis", axis, default=False))
        if axis == RADIAL_AXIS and not declared:
            reasons.append(f"axis {RADIAL_AXIS} is not the cylindrical r = 0 axis")
        if axis != RADIAL_AXIS and declared:
            reasons.append(f"axis {axis} is declared the cylindrical r = 0 axis; "
                           f"this product's radial substitution is on axis "
                           f"{RADIAL_AXIS}")

    # 6a (THIS FAMILY'S OWN). The phi axis carries ONE cell on a Dcyl grid, which is
    #    what makes its finite difference an exact zero rather than a boundary. The
    #    clause is spelled rather than inferred from `Grid`'s own guard: this
    #    package's doctrine forbids reading a refusal off another module.
    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) == 3 and int(shape[1]) != 1:
        reasons.append(f"the phi axis carries {shape[1]} cells; a Dcyl grid has one, "
                       f"and this kernel's transverse operands assume the invariant "
                       f"wrap that gives")

    # 7. k = 0, KEPT. `Grid` refuses a k on r or phi, and a z Bloch phase would need
    #    complex storage, which clause 2 already refuses — but the pairing is named
    #    rather than left to be implied by another clause.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. No instantaneous nonlinearity.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried; that family is a separate tranche)")

    # 11/12. BFAST and beta, KEPT. Cylindrical + BFAST is refused by `Grid` itself
    #        (grid.py:706-742) and the clause is kept anyway, for the reason
    #        `bfast_curl` keeps its own cylindrical clause: a refusal inferred from
    #        another module's guard is not this predicate's answer.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not "
                       "carried; the grid itself refuses bfast on Dcyl, "
                       "grid.py:706-742)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    # 14 (NEW — this family's internal split). m == 0 REQUIRED, refused by name above
    #    it. Clause 2 alone is NOT sufficient in both directions: `stepping`
    #    :704-709 makes complex storage a hard REQUIREMENT of the i*m/r term, so an
    #    |m| >= 1 real run cannot exist on the engine — but "cannot exist" is an
    #    ENGINE COUPLING, not an inverted clause, and this project has already
    #    recorded that residuals separated only by a coupling are PLANTED and
    #    measured, never assumed. So the m clause is spelled, and the |m| >= 1 family
    #    will spell its inversion.
    m = getattr(grid, "m", 0)
    try:
        nonzero = int(m) != 0
    except Exception as exc:  # noqa: BLE001 - an unreadable m is not a zero one
        reasons.append(f"grid.m could not be read as an integer: {exc!r}")
    else:
        if nonzero:
            reasons.append(
                f"grid.m = {m!r}: |m| >= 1 adds the i*m/r coupling on two targets, "
                f"the |m| = 1 axis-row curl REPLACEMENT on both sides and the "
                f"|m| >= 2 six-component near-axis zeroing, and it forces complex64 "
                f"storage (stepping.py:731-736) — that is the |m| >= 1 family's "
                f"kernel and this one refuses it by name rather than by the absence "
                f"of an i*m/r term")

    # 9c. Stored E — the invariant behind admitting dispersion for the curl.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return reasons


def _curl_conductivity_reasons(fields: Any) -> List[str]:
    """Conductivity refused on ALL six curl targets, BOTH directions.

    This kernel transcribes ``_apply_pml_update`` (stepping.py:1952), the PLAIN
    split-field recurrence; a conductivity routes ``_apply_curl`` to the three-stage
    ``f_cond`` recurrence instead (:487-505). THIS PACKAGE SHIPS NO CONDUCTIVE
    PRODUCT AT ALL, so the refusal is unconditional rather than a hand-off.

    A missing or non-callable reader is refused OUTRIGHT: inferring "no conductivity"
    from the absence of ``condfac_for`` is admission by attribute absence.
    """
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
                f"a conductivity is installed on {target}: this kernel transcribes "
                f"the plain split-field recurrence only, and this package ships no "
                f"conductive product")
    return reasons


def cylindrical_real_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                   residency: Any = None) -> Coverage:
    """May the Metal cylindrical m = 0 curl kernel step this (fields, pml, sub_step)?"""
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _cylindrical_real_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_curl_conductivity_reasons(fields))

    # 9a/9b. A registered susceptibility must be one this package understands.
    reasons.extend(_susceptibility_reasons(fields))

    # 13. The named sub-step's targets, auxiliaries and sources.
    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    # 14b. The D side's axis increment re-reads Hp from storage exactly as
    #      `_cylindrical_axis_zero_D` does (the stored H under an active PML, which
    #      clause 3 requires); named so a future no-PML arm cannot inherit it.
    if sub_step == "step_D" and getattr(fields, "Hy", None) is None:
        reasons.append("Hy is not stored: the m = 0 axis rule adds "
                       "(4*Courant)*Hp[r=0] to Dz and reads the STORED H "
                       "(stepping._cylindrical_axis_zero_D)")

    # 14c. The D side's on-axis product is MEEP's double product with Courant.
    if sub_step == "step_D":
        reasons.extend(courant_reasons(grid, "the m = 0 on-axis Dz increment"))

    # 15. Layout: float32, C-contiguous, grid.shape.
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    # 16. The radial sum's chunk starts: one per r row, on a division the column
    #     walk can restart on.
    if not reasons:
        reasons.extend(scan_start_reasons(fields, pml, sub_step))

    return Coverage(not reasons, tuple(reasons))


def courant_reasons(grid: Any, use: str) -> List[str]:
    """Why a double product with ``Courant`` is not this backend's float32 product.

    MEEP's cylindrical axis rules multiply a ``realnum`` by the double ``Courant`` in
    double and round once (step_db.cpp:324, :377; ``stepping._realnum_double_product``).
    This backend has no float64, and its float32 product with the float32 word of the
    factor is the same word exactly when the factor is a float32 word: two float32
    operands multiply exactly in double. Any other ``Courant`` is refused by name
    rather than stepped one rounding off; ``use`` names the product.
    """
    import numpy  # noqa: PLC0415

    try:
        courant = float(getattr(grid, "courant"))
    except Exception as exc:  # noqa: BLE001 - an unreadable Courant is not a float32 one
        return [f"grid.courant could not be read ({exc!r}); {use} multiplies by it"]
    if float(numpy.float32(courant)) != courant:
        return [f"Courant {courant!r} is not a float32 word: {use} is MEEP's double "
                f"product with Courant rounded once, which this backend forms as a "
                f"float32 product only when Courant is a float32 word (no float64 on "
                f"this device; a software double product is not built)"]
    return []


def scan_start_reasons(fields: Any, pml: Any, sub_step: str) -> List[str]:
    """Why the radial sum's chunk starts cannot be bound; empty when they can.

    The starts are the array path's own (``stepping._cylindrical_scan_starts`` for
    the sub-step's target, on the division the run steps with), and
    :func:`scan_starts` refuses what the column walk cannot take.
    """
    from ..stepping import _cylindrical_scan_starts  # noqa: PLC0415

    try:
        starts = _cylindrical_scan_starts(fields, pml, SCAN_TARGETS[sub_step])
        scan_starts(sub_step, starts, int(fields.grid.shape[0]))
    except ValueError as exc:
        return [f"the radial sum's chunk starts are refused: {exc}"]
    except Exception as exc:  # noqa: BLE001 - an unreadable division is not covered
        return [f"the radial sum's chunk starts could not be read ({exc!r})"]
    return []


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalRealCurlPlan:
    """One cylindrical m = 0 curl sub-step, as TWO launches: the scan, then the curl.

    A sibling of :class:`.plans.KernelPlan` rather than a subclass: that base's
    contract is that ``run`` unpacks ONE argument tuple and calls ONE function, and
    this plan cannot honour it — the curl reads the scan's output, so the two are
    ordered and cannot be one dispatch.

    THE RADIAL DIFFERENCE IS A PLAN-OWNED SCRATCH VOLUME, registered non-constant
    (the device writes it) through :func:`scratch_mirror`, which reuses the array
    already bound under the name, so two plans for one sub-step on one residency share
    it; the scan overwrites every word before the curl reads it. The two row vectors
    are grid invariants, mirrored constant. The chunk starts and the chunk record are
    ``int32`` plan-owned tensors outside the residency registry, as every ``Params``
    record is: nothing on the device writes them and nothing on the host reads them
    back.

    ``launches`` COUNTS KERNEL LAUNCHES, NOT ``run`` CALLS, and it is two per cycle;
    :attr:`scan_launches` and :attr:`curl_launches` are carried separately so "the
    scan ran and the curl did not" is distinguishable from "the plan ran once".
    """

    __slots__ = ("sub_step", "shape", "n_elem", "n_cols", "dtdx",
                 "axis_coef", "backward", "bc", "residency", "volumes",
                 "radial_host", "starts", "chunk_bounds", "launches", "scan_launches",
                 "curl_launches", "_scan_functions", "_scan_args",
                 "_curl_functions", "_curl_args")

    family = "cylindrical m = 0 PML curl"

    #: This plan launches kernels; the composer's planned/null split reads it.
    performs_device_work = True

    #: TWO: ``run`` dispatches the radial scan and then the curl that reads it.
    launches_per_run = 2

    #: The run's first ``update_H`` is not this plan's slot; declared for symmetry.
    REPR_FIELDS = ("sub_step", "shape", "bc")

    def __init__(self, sub_step: str, shape, dtdx: float, axis_coef: float, bc,
                 residency: Residency, radial_host: Any, starts: Any,
                 chunk_bounds: Any, scan_args: Sequence[Any],
                 curl_args: Sequence[Any], scan_functions: Dict[str, Any],
                 curl_functions: Dict[str, Any], volumes: Sequence[str]) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(
                f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.n_cols = self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy casts to float32 before the multiply; Metal binds a Python
        # float into `constant float&` the same way, so the two scalars carry the
        # same bits.
        self.dtdx = float(dtdx)
        self.axis_coef = float(axis_coef)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.residency = residency
        self.volumes = tuple(volumes)
        self.radial_host = radial_host
        self.starts = starts
        self.chunk_bounds = chunk_bounds
        self._scan_functions = dict(scan_functions)
        self._curl_functions = dict(curl_functions)
        self._scan_args = tuple(scan_args)
        self._curl_args = tuple(curl_args)
        self.launches = 0
        self.scan_launches = 0
        self.curl_launches = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        """The contraction modes BOTH kernels were built with (their intersection)."""
        return tuple(sorted(set(self._scan_functions) & set(self._curl_functions)))

    def _function(self, table: Dict[str, Any], mode: str, which: str) -> Any:
        function = table.get(mode)
        if function is None:
            raise KeyError(
                f"this plan holds no {mode!r} {which} variant (it was built with "
                f"{self.variants}); build it with contract_variants={(mode,)} "
                f"rather than launching the pinned one")
        return function

    def run_scan(self, contract: Optional[str] = None) -> None:
        """The radial scan alone — ``cylindrical_rderiv_difference`` on the device."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._scan_functions, mode, "scan")
        self.launches += 1
        self.scan_launches += 1
        function(*self._scan_args)

    def run_curl(self, contract: Optional[str] = None) -> None:
        """The curl alone. Reads the scan's output, so it is never run first."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._curl_functions, mode, "curl")
        self.launches += 1
        self.curl_launches += 1
        function(*self._curl_args)

    def run(self, contract: Optional[str] = None) -> None:
        """The sub-step: scan, then curl. In place.

        THE ORDER IS THE CONTRACT. The curl reads the radial difference the scan
        writes, a function of THIS sub-step's source volume, so a scan run after the
        curl feeds it the previous timestep's derivative: a plausible wrong field
        rather than an error.
        """
        self.run_scan(contract)
        self.run_curl(contract)

    def describe(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}"
                          for name in self.REPR_FIELDS)
        return f"CylindricalRealCurlPlan({parts}, variants={self.variants})"

    def __repr__(self) -> str:
        return self.describe()


def _scan_functions(sub_step: str, contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_scan(sub_step, mode) for mode in contract_variants}


def _curl_functions(sub_step: str, codes, contract_variants: Sequence[str]
                    ) -> Dict[str, Any]:
    return {mode: compile_cylindrical_curl(sub_step, codes, mode)
            for mode in contract_variants}


def scratch_mirror(residency: Residency, name: str, build: Any,
                   constant: bool = False) -> Tuple[Any, Any]:
    """Register a PLAN-OWNED scratch volume, reusing the array already bound.

    ``build`` is called ONLY on the first registration, so a second plan for the same
    sub-step on one residency shares the volume rather than being refused for binding
    a different host array under the same name.
    """
    existing = residency.host(name)
    host = build() if existing is None else existing
    return host, residency.mirror(name, host, constant=constant)


def int32_tensor(values: Any, device: str) -> Any:
    """A plan-owned ``torch.int32`` copy of a small host vector on ``device``.

    Outside the residency registry, as every ``Params`` tensor is: ``Residency.mirror``
    binds float32 and complex64 volumes only, and nothing on the device writes this.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    host = np.ascontiguousarray(values, dtype=np.int32).reshape(-1)
    return torch.from_numpy(host.copy()).to(torch.device(device))


def boundary_codes(kinds: Sequence[str]) -> Tuple[int, int, int]:
    """``_boundary_kinds``' strings as this family's specialisation triple ('axis' -> METALLIC)."""
    return tuple(METALLIC if kind in ("metallic", "axis") else PERIODIC
                 for kind in kinds)  # type: ignore[return-value]


def axis_coefficient(courant: float) -> float:
    """The m = 0 on-axis Dz factor ``4 * Courant``, bound into ``constant float&``.

    MEEP forms ``g[i] * (Courant * 4)`` in double and rounds the product once
    (step_db.cpp:324, ``stepping._realnum_double_product``). Bound as the float32
    word of ``4 * Courant``, the kernel's float32 product is that word exactly when
    ``Courant`` is a float32 word (two float32 operands multiply exactly in double),
    which :func:`courant_reasons` requires.
    """
    return 4.0 * float(courant)


def _build_curl_plan(sub_step: str, shape, dtdx: float, courant: float, codes,
                     residency: Residency, arrays: Dict[str, Any],
                     coefficients: Sequence[Any], starts: Any, chunk_bounds: Any,
                     owner: str, scan_function: Optional[Dict[str, Any]],
                     curl_function: Optional[Dict[str, Any]],
                     contract_variants: Sequence[str]) -> CylindricalRealCurlPlan:
    """The ONE plan assembly both routes take (engine and bare arrays)."""
    import numpy as np  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in shape)
    record = _chunk_bounds.plan_words(chunk_bounds, shape, owner)
    start_line = scan_starts(sub_step, starts, shape[0])

    targets = [residency.mirror(n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n]) for n in spec["sources"]]

    # The scan reads the sub-step's PHI source: Ep on the B side and Hp on the D side,
    # both `sources[1]`, named through the term table rather than by index.
    scan_source_name = spec["sources"][1]
    radial_host, radial = scratch_mirror(
        residency, f"cyl_radial:{sub_step}",
        lambda: np.zeros(shape, dtype=np.float32))
    _, weight_vector = scratch_mirror(
        residency, f"cyl_weights:{sub_step}",
        lambda: prefix_row_vectors(sub_step, shape[0])[0], constant=True)
    _, rinv_vector = scratch_mirror(
        residency, f"cyl_rinv:{sub_step}",
        lambda: prefix_row_vectors(sub_step, shape[0])[1], constant=True)
    starts_tensor = int32_tensor(start_line, residency.device)

    # Zero on the B side, which binds the slot without reading it: one binding layout
    # for both sub-steps.
    axis_coef = axis_coefficient(courant) if sub_step == "step_D" else 0.0
    n_elem = shape[0] * shape[1] * shape[2]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    scan_args = (radial, residency.mirror(scan_source_name, arrays[scan_source_name]),
                 weight_vector, rinv_vector, starts_tensor,
                 shape[0], shape[1], shape[2], shape[1] * shape[2])
    curl_args = (tuple(targets) + tuple(auxiliaries) + tuple(sources)
                 + tuple(coefficients)
                 + (shape[0], shape[1], shape[2], n_elem, float(dtdx))
                 + (radial, float(axis_coef), _chunk_bounds.tensor(record, residency.device)))
    return CylindricalRealCurlPlan(
        sub_step, shape, dtdx, axis_coef, codes, residency, radial_host, start_line,
        record, scan_args, curl_args,
        scan_function if scan_function is not None
        else _scan_functions(sub_step, contract_variants),
        curl_function if curl_function is not None
        else _curl_functions(sub_step, codes, contract_variants),
        volumes)


def plan_cylindrical_real_curl(fields: Any, pml: Any, sub_step: str,
                               residency: Optional[Residency] = None,
                               contract_variants: Sequence[str] = (
                                   shaders.CONTRACT_OFF,),
                               *, chunk_bounds: Any,
                               ) -> Optional[CylindricalRealCurlPlan]:
    """Build a cylindrical m = 0 curl plan from the engine's own objects, or None.

    ``chunk_bounds`` is the record of the division the run steps with (the composer's
    ``StepContext.chunk_bounds``) and has no default; the chunk starts are the array
    path's own (``stepping._cylindrical_scan_starts``). None is the only refusal: a
    configuration this kernel does not carry falls back to the array path.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not cylindrical_real_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from ..stepping import _cylindrical_scan_starts  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    codes = boundary_codes(resolve(grid, pml))
    names = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
             + tuple(spec["sources"]))
    arrays = {name: getattr(fields, name) for name in names}
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    starts = _cylindrical_scan_starts(fields, pml, SCAN_TARGETS[sub_step])
    return _build_curl_plan(
        sub_step, grid.shape, grid.dt / grid.dx, grid.courant, codes, residency,
        arrays, coefficients, starts, chunk_bounds, "plan_cylindrical_real_curl",
        None, None, contract_variants)


def plan_cylindrical_real_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], codes,
        dtdx: float, residency: Residency, *, chunk_bounds: Any,
        courant: Optional[float] = None, starts: Any = None,
        scan_function: Optional[Dict[str, Any]] = None,
        curl_function: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        ) -> CylindricalRealCurlPlan:
    """Build a plan from bare host arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately. ``chunk_bounds`` is required; ``starts`` is
    ``stepping._cylindrical_scan_starts``'s answer for the sub-step's target (``None``:
    one chunk from the axis); ``courant`` defaults to ``dtdx``. ``scan_function`` and
    ``curl_function`` are the mutation seams, separate so a leg can plant a defect in
    one kernel while the other stays shipped.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    return _build_curl_plan(
        sub_step, shape, dtdx, float(dtdx) if courant is None else float(courant),
        tuple(int(code) for code in codes), residency, arrays, coefficients, starts,
        chunk_bounds, "plan_cylindrical_real_curl_from_arrays", scan_function,
        curl_function, contract_variants)


# ---------------------------------------------------------------------------
# The constitutive companion — no new kernel, a restated predicate
# ---------------------------------------------------------------------------
#
# WHY THIS IS SOUND, and it is a READ of stepping.py rather than an analogy.
# `update_H` (S:907) and `update_E` (S:926) contain NO CYLINDRICAL BRANCH AT ALL:
# measured on this tree by reading their own source, the words "cylindrical",
# "is_axis" and "grid.m" do not appear in either. They are element-wise with per-axis
# coefficient tables indexed on the component's OWN axis, and Dcyl changes nothing
# about that except that axis 1 has n = 1. So the ARITHMETIC of the constitutive pair
# on a Dcyl run is byte-for-byte the arithmetic the certified constitutive kernel
# already reproduces, and only the ADMISSION has to change.
#
# THE TRITON TRACK MEASURED THE SAME CLAIM ON ITS OWN KERNEL — the existing
# `kernels.constitutive_step` bytewise identical to `stepping.update_H`/`update_E` on
# a Dcyl grid, 4/4 (`cylindrical_triton.py:84-86`) — which is corroboration on
# another backend and NOT this port's evidence. This one's evidence is its own gate.
#
# THIS IS THE SAME MOVE `bfast_curl.bfast_run_constitutive_coverage` MAKES, for the
# same reason, and it is the cheapest coverage on either track: no kernel, no new
# specialisation. What it buys is WHOLE-STEP composition — without it a Dcyl run
# takes its curls on the device and its constitutive pair on the array path, which
# forces a sync per sub-step and makes the residency verdict refuse. Six of this
# family's twelve slots are this predicate.

def cylindrical_real_constitutive_coverage(fields: Any, pml: Any, side: str,
                                           residency: Any = None) -> Coverage:
    """May the CERTIFIED Metal constitutive kernel step a Dcyl m = 0 run?

    :func:`_cylindrical_real_grid_reasons`' clause set with nothing added and nothing
    removed — clause 6 is already inverted there — plus the constitutive side's own
    clauses, exactly as the certified predicate spells them. The conductivity clause
    is deliberately NOT applied: a conductivity changes the CURL recurrence
    (``_apply_curl`` reads ``condfac_for`` and nothing else does), so refusing it here
    would refuse a configuration this sub-step steps correctly.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _cylindrical_real_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_susceptibility_reasons(fields))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is (D - sum P), "
                "not D (fields.py:1096-1105) — that configuration belongs to the "
                "ADE kernel")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product reads "
                "neighbours; this sub-step is element-wise)")

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


def plan_cylindrical_real_constitutive(fields: Any, pml: Any, side: str,
                                       residency: Any = None,
                                       contract_variants: Sequence[str] = (
                                           templates.CONTRACT_OFF,),
                                       *, chunk_bounds: Any) -> Any:
    """A CERTIFIED constitutive plan for a Dcyl m = 0 run, or None.

    The arithmetic, the source and the plan class are the certified ones; only the
    ADMISSION is this module's. Building the plan through the certified builder —
    rather than re-deriving one here — is what makes "same kernel" a fact instead of
    a claim: a divergence would have to come from the predicate, which is the only
    thing this family contributes.
    """
    from .launch import (  # noqa: PLC0415 - avoids a circular import at module load
        ConstitutivePlan, _constitutive_functions,
    )

    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not cylindrical_real_constitutive_coverage(fields, pml, side,
                                                  residency).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror(n, getattr(fields, n)) for n in spec["aux"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True) for n in spec["targets"]]
        if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ConstitutivePlan(
        side, fields.grid.shape, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        _constitutive_functions(side, contract_variants), volumes,
        chunk_bounds=chunk_bounds)


# ---------------------------------------------------------------------------
# WIRING — two curl arms and the certified constitutive pair
# ---------------------------------------------------------------------------
#
# WHAT SEPARATES THEM, stated because a new family must state its inversion rather
# than inherit disjointness. `coverage._grid_reasons` clause 6 refuses `cylindrical`
# AND `is_axis` BY NAME (:176-182), and every carried family either uses that list
# (`pml_curl`, `constitutive`, `no_pml_constitutive`, `offdiag_update_e`) or restates
# it verbatim (`complex_fields`, `special_kz`, `bfast_curl`, `symmetry`,
# `folded_complex`, `folded_offdiag_update_e`). Clause 6 here is INVERTED — a Dcyl
# grid with the r axis declared is REQUIRED — so the separation from all nine is
# total, and it is one clause rather than a coincidence of several.
#
# THE ONE FAMILY THIS DOES NOT SEPARATE FROM BY CLAUSE 6 IS THE CYLINDRICAL COMPLEX
# ONE, and clause 2 (real storage) is what does: this family REQUIRES float32 and
# that one REQUIRES complex64, inverted in both directions. Clause 14 (m == 0 here)
# was a SECOND separation while the complex family carried |m| >= 1 only; since
# 2026-09-04 that family's M_ZERO arm carries complex64 at m = 0 too, so clause 14
# now separates this family from the ENGINE's own refusal (a float32 run at |m| >= 1
# cannot exist, stepping.py:731-736) rather than from a sibling product, and the two
# families partition Dcyl by storage alone.
#
# ON `update_H`/`update_E` THE ADMISSION IS NOT AN AMBIGUITY EITHER, and the
# reasoning is the same clause: `constitutive` and `no_pml_constitutive` are the only
# other arms there, the first refuses a Dcyl run through the shared clause 6, and the
# second requires an INACTIVE absorber where clause 3 here requires an active one —
# two independent separations, not one. The composition sweep MEASURES that per
# registered arm rather than assuming it.
#
# `update_P` still carries no Metal arm at all and this family does not change that:
# a Dcyl run with a registered susceptibility is refused on the E side by name, so
# nothing here reaches the pole recurrence.


def _curl_arm_coverage(context: Any, slot: str) -> Coverage:
    return cylindrical_real_curl_coverage(context.fields, context.pml, slot,
                                          context.residency)


def _curl_arm_plan(context: Any, slot: str) -> Optional[CylindricalRealCurlPlan]:
    return plan_cylindrical_real_curl(context.fields, context.pml, slot,
                                      context.residency, context.contract_variants,
                                      chunk_bounds=context.chunk_bounds)


#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return cylindrical_real_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency)


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_cylindrical_real_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants,
        chunk_bounds=context.chunk_bounds)


def register_arms() -> Tuple[Any, ...]:
    """This family's four arms: two curls and the certified constitutive pair."""
    from . import arms  # noqa: PLC0415 - deferred: `arms` imports nothing of ours

    registered = [
        arms.register(family=FAMILY, slot=slot, label="cylindrical m=0",
                      coverage=_curl_arm_coverage, plan=_curl_arm_plan,
                      prefix="cylindrical m=0: ",
                      noun="cylindrical m = 0 PML curl", wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="cylindrical m=0",
                      coverage=_constitutive_arm_coverage,
                      plan=_constitutive_arm_plan,
                      prefix="cylindrical m=0: ",
                      noun="cylindrical m = 0 constitutive", wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    return tuple(registered)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[Any, ...] = register_arms()
