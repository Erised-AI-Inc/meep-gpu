"""The off-diagonal ``update_E`` device source, emitted — with no CuPy in sight.

WHY THE EMITTER IS A MODULE OF ITS OWN, and not a few functions inside
``offdiag_constitutive_kernels``. That module holds ``cp.RawKernel`` objects and
therefore imports ``cupy`` at scope, so nothing in it can be imported — let alone
exercised — on a machine without a GPU. The certified siblings live with that
because their device code is a pair of module-level STRING CONSTANTS, which a test
can evaluate off the syntax tree without importing anything. This family's device
code is not a constant: it is specialized per row mask, so the thing a test has to
check is a FUNCTION, and a function has to be imported to be called.

So the emitter goes where ``coverage.py`` and ``compile_cache.py`` already are:
in a stdlib-only sibling, on the package's own rule that the parts of this track
whose failure mode is a SILENT WRONG ANSWER rather than a crash are the parts that
must be exercisable at the merge bar. Emitting ``dj`` where ``dk`` belongs is a
half-cell registration error — smooth, converged, plausible and wrong — and it is
caught here on a laptop in a tenth of a second rather than in a device slot.

NOTHING HERE IMPORTS CUPY OR NUMPY. The three transcribed tables come from
``coverage``, which imports nothing either.

WHAT IS SPECIALIZED AND WHAT IS NOT — the whole design decision of this family, in
one paragraph. The sibling Triton kernel carries TWELVE compile-time axes (six row
flags, three boundary codes, three wall flags), a static product of 4,096 kernels
of which the 186-row corpus drives seven. This emitter splits them by what they do
to the ARITHMETIC:

* the boundary and wall axes are BRANCH axes — they select an index or a
  predicated zero and change no float operation and no association — so they are
  ordinary runtime ``int`` arguments, exactly as the certified
  ``step_B_pml_real`` already takes ``bc_x``/``bc_y``/``bc_z``
  (step_curl_kernels.py:1687). Sixty-four variants collapse to one source;
* the six ROW-LIVENESS flags are an ARITY axis — they change how many terms the
  inner sum has, and therefore its association — so they stay compile-time and are
  this emitter's only parameter.

Of the 63 live row masks the corpus asks for TWO.

WHY THE ARITY AXIS STAYED COMPILE-TIME, with both facts stated. The ZERO-PADDING
fold — bind a dead slot to an all-zero coefficient so a dropped term contributes
``+0.0`` — is measurably NOT bit-identical: ``x + 0.0f`` turns ``-0.0f`` into
``+0.0f``, and a ``-0.0f`` coupling total is reachable.
``test_offdiag_constitutive_pml_real.py`` carries that witness. A DYNAMIC LOOP is a
different fold and IS licensed: ``the design notes (cuda-kernel-triton-transfer-assessment)``
§3.2 measured a runtime-bounded loop bit-identical to the unrolled form at every
arity 0-8 (156/156 cases, zero differing words, max ULP 0), because NVCC leaves the
arithmetic a strictly serial dependence chain on one accumulator. This family's row
accumulation is such a chain, so the axis COULD go runtime; this round did not take
it. Doing so needs all six pointers bound and needs the accumulation kept in its own
accumulator with the final ``+ total`` itself conditional on a non-zero trip count —
because the array path sums the coupling separately and adds it once
(stepping.py:1250 then :1007-1008), which is a shape §3.2 did not measure. It would
buy one compiled source on this corpus instead of two.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .coverage import (OFFDIAG_ROW_SLOTS, OFFDIAG_TRANSVERSE_PARTNERS,
                       OFFDIAG_WALL_MASK_AXES)

#: The one kernel this family emits. Its SOURCE varies with the row mask; its NAME
#: does not, because it is one family and NVRTC compiles one module per source.
KERNEL_NAME = "update_E_pml_real_offdiag"

#: The components this sub-step writes, in ``stepping.E_CONSTITUTIVE_TERMS`` order
#: (stepping.py:228), with each component's source volume and OWN axis — the axis
#: whose HALF-INTEGER coefficient pair the tail reads (MEEP's ``dsigw``).
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The D volume of each axis. The coupling reads the PARTNER axis's volume, which
#: with no poles admitted is that axis's D primary (fields.py:1107-1138).
PARTNER_VOLUMES: Tuple[str, str, str] = ("Dx", "Dy", "Dz")

#: The kernel parameter name of each row coefficient, in
#: :data:`coverage.OFFDIAG_ROW_SLOTS` order — named for the pair it carries, so a
#: mispairing is visible in the emitted source rather than only in the physics.
ROW_PARAMETERS: Tuple[str, ...] = tuple(
    f"chi1inv_{row}_{partner}" for row, partner in OFFDIAG_ROW_SLOTS)

#: Every row mask this family can emit: the 63 non-empty subsets of the six slots.
#: An all-dead mask is NOT here — that configuration is the certified plain
#: constitutive kernel's, and the two families are disjoint by the install-time
#: zero-row drop (fields.py:1302-1303).
LIVE_ROW_MASKS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple((value >> slot) & 1 for slot in range(len(OFFDIAG_ROW_SLOTS)))
    for value in range(1, 1 << len(OFFDIAG_ROW_SLOTS))
)


# =============================================================================
# THE DEVICE CODE
# =============================================================================
#
# SEVEN THINGS DECIDE BIT-IDENTITY HERE. The first three are the certified
# constitutive pair's, unchanged, because the tail is unchanged. The last four are
# this family's own, and every one of them is a SILENT wrong answer: a smooth,
# converged, plausible field with the wrong tensor in it.
#
# 1. THE TAIL IS TWO SEPARATE ACCUMULATIONS, LEFT TO RIGHT
#    (``stepping._apply_constitutive_pml``, stepping.py:2133-2134, and the scratch
#    branch :2140-2143, which is the same tree by a different route).
#    ``constitutive_apply`` below is character-for-character the certified
#    sibling's, and a test welds the two texts together.
# 2. THE COEFFICIENT INDEX IS THE COMPONENT'S OWN AXIS -- MEEP's ``dsigw``, not
#    the dsig/dsigu cycle the curl recurrence uses.
# 3. ``prev`` IS READ BEFORE ``fw`` IS WRITTEN.
# 4. THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS -- see ``offdiag_term``.
# 5. THE TWO SHIFTS GO IN OPPOSITE DIRECTIONS: half a cell DOWN the partner's axis
#    on the raw partner volume, then the already-multiplied product half a cell UP
#    this component's own axis.
# 6. THE WALL MASK RUNS BEFORE THE ROW SUM (stepping.py:1252 precedes the add at
#    :1007-1008) and asks the grid's DECLARATION (``is_metallic and not
#    is_mirrored``, :1253), which is a different question from the resolved ghost
#    rule -- hence ``wm_*`` beside ``bc_*`` rather than derived from it.
# 7. A COMPONENT WITH NO SURVIVING ROW KEEPS THE PURE DIAGONAL ARITHMETIC:
#    ``_offdiagonal_terms`` returns None and ``update_E`` then never forms a
#    ``+ 0`` copy (stepping.py:1222-1227, :1007).

PRELUDE = r'''
#define BC_PERIODIC 0
#define BC_METALLIC 1

// stepping._shift_up (stepping.py:1723), PERIODIC and METALLIC branches only:
// the neighbour COORDINATE one cell up an axis, or -1 where the metallic far-face
// ghost makes the sample an exact zero. MIRROR and CYL_AXIS are refused by
// covers_real_pml_offdiag_constitutive, and real storage cannot carry a Bloch
// phase, so those branches have no code here.
//
// WHY A COORDINATE AND NOT A VALUE, where the certified curl's shift_up returns a
// float (step_curl_kernels.py:1646-1651): this stencil needs the SAME shifted
// position three times over -- the partner volume at the shifted node, the
// partner volume at the CORNER (up this component's axis and down the partner's),
// and the COEFFICIENT at the shifted node, because the multiply sits between the
// two shifts and u is read at the up index rather than shifted with the product.
// Resolving once to a coordinate and loading three times is the same arithmetic
// and one rule instead of three.
__device__ __forceinline__ int coord_up(int a, int n, int bc) {
    if (a + 1 < n) return a + 1;
    return (bc == BC_METALLIC) ? -1 : 0;
}

// stepping._shift_down (stepping.py:1787), same two branches, same convention.
// An invariant axis (n == 1, always PERIODIC) needs no case of its own: both
// wraps return the same cell, so the partner pair is 2*g rather than the curl's
// exact zero -- which is what MEEP's stride(d) = 0 double-read computes.
__device__ __forceinline__ int coord_dn(int a, int n, int bc) {
    if (a > 0) return a - 1;
    return (bc == BC_METALLIC) ? -1 : n - 1;
}

// The flat index of a cell, with a -1 on ANY axis propagating. The two shifts of
// one term go in opposite directions on DIFFERENT axes, so the corner sample is a
// ghost when either leg is one; the per-axis rules compose independently, exactly
// as they do in the array path where the two shift helpers are applied in turn.
__device__ __forceinline__ int flat(int i, int j, int k, int nyz, int nz) {
    return (i < 0 || j < 0 || k < 0) ? -1 : (i * nyz + j * nz + k);
}

// A ghosted load: index -1 is the metallic zero ghost, which the array path
// writes into that plane as an exact +0.0 (_shift_up:1782-1784,
// _shift_down:1826-1828).
__device__ __forceinline__ float ghosted(const float* g, int index) {
    return (index < 0) ? 0.0f : g[index];
}

// One partner's OFFDIAG term -- MEEP step_generic.cpp:582-583 as
// stepping._offdiagonal_terms (stepping.py:1214-1220) associates it:
//
//     0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])
//
// THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS. u[i] multiplies the
// two-point partner average AT ITS OWN NODE and u[i+s] the average at the next
// node up this component's axis, because MEEP registers the off-diagonal entry at
// the component's Yee site minus half a cell along its own axis -- the integer
// node (anisotropic_averaging.cpp:248-257, `here - shift1`). Hoisting to a plain
// four-point average times u[i] is the same ALGEBRA only for a uniform
// coefficient, and a different float32 number even then: a*u + b*u and (a+b)*u do
// not round alike.
//
// LOADING u AT THE COMPOSED UP INDEX IS THE SAME BITS AS SHIFTING THE FORMED
// PRODUCT. A shift is data movement; the coefficient wraps plainly on a periodic
// axis (it is periodic and phase-free), and on a metallic far face both the pair
// and the coefficient read the -1 ghost, so the product is an exact +0.0 -- which
// is precisely what the array path's _shift_up writes into that plane.
//
// 0.25 SCALES THE SUM AND IS APPLIED LAST. It is exact in float32, so the
// distributed form is bitwise identical away from underflow; the association here
// is transcription fidelity rather than a pinned grouping.
//
// NO UNARY MINUS ANYWHERE ON THIS PATH. CUDA lowers -x to neg.f32 rather than to
// 0.0f - x, so it does NOT canonicalize signed zeros the way the sibling track's
// platform does. That track's negation idiom must not be ported here, and this
// body is written so the answer never matters.
__device__ __forceinline__ float offdiag_term(
    const float* g, const float* u, int home, int down, int up, int corner
) {
    float near_pair = g[home] + ghosted(g, down);
    float far_pair = ghosted(g, up) + ghosted(g, corner);
    return 0.25f * ((near_pair * u[home]) + (far_pair * ghosted(u, up)));
}

// stepping._apply_constitutive_pml (stepping.py:2065), MEEP's step_update_EDHB
// with dsigw active. CHARACTER FOR CHARACTER the certified sibling's
// constitutive_apply (constitutive_kernels.py); a test welds the two texts so the
// tail cannot drift between the two families that share it.
__device__ __forceinline__ void constitutive_apply(
    float* __restrict__ f, float* __restrict__ fw, int idx, float src,
    float kps, float kms
) {
    float prev = fw[idx];
    fw[idx] = src;
    float a = f[idx] + kps * src;
    f[idx] = a - kms * prev;
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1723->1770, 1787->1834, 1214-1220->1243-1249, 2065->2112

# THE SIGNATURE CARRIES ONLY THE LIVE ROW COEFFICIENTS. The sibling track binds all
# six pointers and points the dead ones at the component's own D volume, on the
# argument that a specialized branch stops the read. That works, and it leaves a
# pointer bound to an array it must never touch -- a shape a later edit can read by
# accident, and one that makes __restrict__ on the D arguments a promise the caller
# cannot keep. Emitting only the live parameters removes both.
#
# WHICH POINTERS CARRY __restrict__, AND WHY THE READ-ONLY ONES DO NOT. The six
# outputs are pairwise distinct and disjoint from every input -- the predicate's
# alias clause and the launcher's check are what make that true -- so they carry
# the promise, as do the six PML coefficient vectors, which no field can alias.
# The read-only VOLUMES do not: an isotropic install hands the same inverse
# permittivity three times (fields.py:1321-1326), and a row coefficient may legally
# alias another row's, an epsilon volume or a D volume. Restrict on mutually
# aliasing arguments is a promise the caller cannot keep, and nothing is lost by
# not making it but a load hint.
TEMPLATE = r'''
extern "C" __global__ void update_E_pml_real_offdiag(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    const float* Dx, const float* Dy, const float* Dz,
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
__ROW_PARAMETERS__
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z,
    int bc_x, int bc_y, int bc_z,
    int wm_x, int wm_y, int wm_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int nyz = ny * nz;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Both neighbour coordinates on every axis: one term reads an axis DOWN (the
    // partner's) and another reads an axis UP (its own), and a component's two
    // terms take different partners, so every axis can be needed either way.
    int di = coord_dn(i, nx, bc_x), ui = coord_up(i, nx, bc_x);
    int dj = coord_dn(j, ny, bc_y), uj = coord_up(j, ny, bc_y);
    int dk = coord_dn(k, nz, bc_z), uk = coord_up(k, nz, bc_z);

    // Wall-plane predicates for the coupling mask. FACE 0 ONLY: the high wall is
    // already the shift-up zero ghost, and stepping._mask_metallic_wall_coupling
    // writes only _face(axis, 0) (stepping.py:1254).
    int at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);

    // The three components are independent, and that is read off the loop rather
    // than assumed: the coupling reads D and the inverse permittivity, the tail
    // writes E and f_w_E. Four disjoint sets, so the order the array path runs
    // them in (stepping.py:969-989) is not a data dependence -- which is what
    // makes a NEIGHBOUR read across components safe here.
__SRC_Ex__

__SRC_Ey__

__SRC_Ez__
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1254->1283, 969-989->998-1018


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------
#
# The three index expressions per term are ASSEMBLED from a per-axis table rather
# than written out eighteen times. Eighteen hand-written flat indices is eighteen
# chances to write ``dj`` where ``dk`` belongs, and that defect is a half-cell
# registration error: smooth, converged and wrong.

_HOME_COORDS = ("i", "j", "k")
_UP_COORDS = ("ui", "uj", "uk")
_DOWN_COORDS = ("di", "dj", "dk")
_WALL_FLAGS = ("wm_x", "wm_y", "wm_z")
_WALL_PREDICATES = ("at_x", "at_y", "at_z")
_OWN_AXIS_INDEX = ("i", "j", "k")


def _flat_index(shifted: Dict[int, str]) -> str:
    """``flat(...)`` with the named per-axis substitutions applied."""
    coords = [shifted.get(axis, _HOME_COORDS[axis]) for axis in range(3)]
    return f"flat({coords[0]}, {coords[1]}, {coords[2]}, nyz, nz)"


def _term_lines(component: int, offset: int) -> List[str]:
    """One partner's term for one component, as source lines.

    ``offset`` is 0 for MEEP's ``cycle_direction(dim, d_ec, 1)`` and 1 for
    ``(..., 2)`` (stepping.py:1235-1237). The partner axis, the coefficient slot,
    the partner volume and all three shifted indices come from that one number, so
    a mispairing would have to be introduced deliberately rather than by a typo.
    """
    own_axis = E_TERMS[component][2]
    partner_axis = OFFDIAG_TRANSVERSE_PARTNERS[component][offset]
    coefficient = ROW_PARAMETERS[2 * component + offset]
    volume = PARTNER_VOLUMES[partner_axis]
    tag = f"{E_TERMS[component][0]}_{offset}"
    down = _flat_index({partner_axis: _DOWN_COORDS[partner_axis]})
    up = _flat_index({own_axis: _UP_COORDS[own_axis]})
    corner = _flat_index({own_axis: _UP_COORDS[own_axis],
                          partner_axis: _DOWN_COORDS[partner_axis]})
    return [
        f"    // partner {volume}: the pair half a cell DOWN axis {partner_axis},",
        f"    // then the product half a cell UP axis {own_axis} (its own).",
        f"    float term_{tag} = offdiag_term(",
        f"        {volume}, {coefficient}, idx,",
        f"        {down},",
        f"        {up},",
        f"        {corner});",
    ]


def _wall_mask_lines(component: int) -> List[str]:
    """``stepping._mask_metallic_wall_coupling`` for one component's total.

    Face 0 of every axis whose Yee shift is 0
    (:data:`coverage.OFFDIAG_WALL_MASK_AXES`), ascending — the mask's own loop
    order (stepping.py:1279-1283). WHICH axes can be masked is fixed by the
    component and compiled in; WHETHER each one is masked is the runtime ``wm_*``
    flag, because that is a property of the grid rather than of the arithmetic. A
    select, never an arithmetic operation, so it cannot round.
    """
    total = f"total_{E_TERMS[component][0]}"
    return [f"    {total} = ({_WALL_FLAGS[axis]} && {_WALL_PREDICATES[axis]})"
            f" ? 0.0f : {total};"
            for axis in OFFDIAG_WALL_MASK_AXES[component]]


def _component_source(component: int, row_mask: Sequence[int]) -> str:
    """One component's whole block, on the arm its two row slots select.

    Four arms: none, offset-1 only, offset-2 only, both. THE NONE ARM IS THE
    CERTIFIED PLAIN CONSTITUTIVE KERNEL'S COMPONENT BODY — ``src = D * inv_eps``
    and the same tail — because ``_offdiagonal_terms`` returns None for a component
    with no surviving row and ``update_E`` then never forms a ``+ 0`` copy
    (stepping.py:1222-1227, :1007).
    """
    name, source, own_axis = E_TERMS[component]
    axis_letter = "xyz"[own_axis]
    index_letter = _OWN_AXIS_INDEX[own_axis]
    live = [offset for offset in (0, 1) if row_mask[2 * component + offset]]

    lines = [f"    // --- {name}: own axis {axis_letter}; source {source}; "
             f"dsigw = {axis_letter}.",
             f"    float gs_{name} = {source}[idx];",
             f"    float us_{name} = inv_eps_{name}[idx];"]
    if not live:
        lines.append(f"    float src_{name} = gs_{name} * us_{name};")
    else:
        for offset in live:
            lines.extend(_term_lines(component, offset))
        # ``total`` accumulates offset 1 then offset 2 (stepping.py:1250). Both
        # additions are bitwise commutative in float32, so this is transcription
        # fidelity rather than a pinned grouping.
        lines.append(f"    float total_{name} = term_{name}_{live[0]};")
        for offset in live[1:]:
            lines.append(f"    total_{name} = total_{name} + term_{name}_{offset};")
        # The mask runs BEFORE the row sum: stepping.py:1252 precedes :1007-1008.
        lines.extend(_wall_mask_lines(component))
        lines.append(f"    float src_{name} = (gs_{name} * us_{name})"
                     f" + total_{name};")
    lines.append(f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},"
                 f" kps_{axis_letter}[{index_letter}],"
                 f" kms_{axis_letter}[{index_letter}]);")
    return "\n".join(lines)


def _row_parameter_lines(row_mask: Sequence[int]) -> str:
    """The live coefficient parameters, in :data:`coverage.OFFDIAG_ROW_SLOTS` order."""
    return "\n".join(f"    const float* {ROW_PARAMETERS[slot]},"
                     for slot, flag in enumerate(row_mask) if flag)


def normalized_row_mask(row_mask: Sequence[int]) -> Tuple[int, ...]:
    """Validate and canonicalize a row mask; raises on the two ways it can be wrong."""
    mask = tuple(int(flag) for flag in row_mask)
    if len(mask) != len(OFFDIAG_ROW_SLOTS) or any(
            flag not in (0, 1) for flag in mask):
        raise ValueError(
            f"a row mask is {len(OFFDIAG_ROW_SLOTS)} flags of {{0, 1}} in "
            f"OFFDIAG_ROW_SLOTS order, got {row_mask!r}")
    if not any(mask):
        raise ValueError(
            "no row slot survives: that configuration belongs to the certified "
            "plain constitutive kernel (update_E_pml_real), and emitting "
            "this family's source for it would overlap the two")
    return mask


def offdiag_source(row_mask: Sequence[int]) -> str:
    """The device source specialized on one row mask.

    The BOUNDARY and WALL axes are NOT baked in — they are runtime ``int``
    arguments, for the reason the module docstring gives — so the row mask is this
    family's only source axis, and the corpus drives two of its 63 values.
    """
    mask = normalized_row_mask(row_mask)
    body = TEMPLATE.replace("__ROW_PARAMETERS__", _row_parameter_lines(mask))
    for component, term in enumerate(E_TERMS):
        body = body.replace(f"__SRC_{term[0]}__",
                            _component_source(component, mask))
    return PRELUDE + body


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Sixty-three sources is too many to pin one digest each without burying the
    record and not too many to hash, so the whole enumeration hashes to one value:
    a single changed character anywhere in the emitter moves it. The per-source
    digests of the masks a gate actually launches belong in that gate's artifact.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for mask in LIVE_ROW_MASKS:
        digest.update(repr(mask).encode("ascii"))
        digest.update(offdiag_source(mask).encode("utf-8"))
    return digest.hexdigest()


def shipped_kernel_names(source: str) -> Any:
    """Every kernel NVRTC could be asked to compile in ``source``, from the text."""
    import re  # noqa: PLC0415 - stdlib, imported at the one call site

    return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))


def offdiag_launch_source(row_mask: Sequence[int]) -> str:
    """The text NVRTC compiles for the single: :func:`offdiag_source` in the register view.

    TWO TEXTS PER ROW MASK, ONE CERTIFIED. :func:`offdiag_source` is the certified
    statement form: the text the fused products lift, the NumPy evaluator executes and
    :func:`corpus_digest` hashes. This is the same text through the port rule,
    ``own_cell_hoist.hoisted_kernel_code``: every own-cell word loaded into a register
    before the first store, the certified ``constitutive_apply`` stepping the registers,
    the write-back in the certified store order. The rule issues it only when its inverse
    returns :func:`offdiag_source` byte for byte. ``offdiag_source`` is looked up when
    this is called, so a harness that rewrites the certified text reaches this one too.

    Measured on one RTX A6000 before it landed (a kernel screen, 2026-09-28), all six
    rows live on ``pml_3d``: 0 differing words at 512,000, 2,097,152 and 7,077,888
    cells, 1.91x-2.06x per launch, 40 registers before and after.
    """
    from . import own_cell_hoist  # noqa: PLC0415 - stdlib-only sibling, one call site

    mask = normalized_row_mask(row_mask)
    certified = offdiag_source(mask)
    # ONE DERIVATION PER CERTIFIED TEXT, NOT PER LAUNCH. The forward, its inverse and the
    # hazard check cost about 0.9 ms of host time a call, and ``_get_kernel`` asks for this
    # text on every launch (it rebuilds its key per call). Keyed on the certified text
    # itself, so a harness that rewrites that text is never served another text's hoist;
    # bounded by the distinct certified texts one process emits.
    issued = _LAUNCH_TEXTS.get(certified)
    if issued is None:
        issued = own_cell_hoist.hoisted_kernel_code(
            certified, KERNEL_NAME, f"offdiag_source({mask!r})")
        _LAUNCH_TEXTS[certified] = issued
    return issued


#: The launch texts already issued by :func:`offdiag_launch_source`, keyed on the certified
#: text each was derived from.
_LAUNCH_TEXTS: Dict[str, str] = {}


def launch_corpus_digest() -> str:
    """One sha256 over every text NVRTC can be asked to compile for the single.

    :func:`corpus_digest` hashes the CERTIFIED texts, which the hoist leaves unchanged,
    so it cannot see an edit to the port rule that moves the compiled bytes. The gate
    compiles four of the 63 launch texts; this digest, recorded beside
    ``emitter_corpus_digest`` when the gate runs, is what the other 59 rest on.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for mask in LIVE_ROW_MASKS:
        digest.update(repr(mask).encode("ascii"))
        digest.update(offdiag_launch_source(mask).encode("utf-8"))
    return digest.hexdigest()
