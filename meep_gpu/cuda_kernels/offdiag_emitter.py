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

NOTHING HERE IMPORTS CUPY OR NUMPY. The transcribed tables come from ``coverage``,
which imports nothing either. The presence words the kernel reads are BUILT on the
host by ``offdiag_presence`` (NumPy); this module only spells how the device reads
them.

WHAT IS SPECIALIZED AND WHAT IS NOT — the whole design decision of this family, in
one paragraph. The sibling Triton kernel carries TWELVE compile-time axes (six row
flags, three boundary codes, three wall flags), a static product of 4,096 kernels
of which the 186-row corpus drives seven. This emitter splits them by what they do
to the ARITHMETIC:

* the boundary and wall axes are BRANCH axes — they select an index or a
  predicated zero and change no float operation and no association — so they are
  ordinary runtime ``int`` arguments, exactly as the certified
  ``step_B_pml_real`` already takes ``bc_x``/``bc_y``/``bc_z``. Sixty-four
  variants collapse to one source. So are the run's chunk record ``cb`` and the
  per-row presence words ``kept_*``: they choose a form or whether a term is
  added, never how one is rounded;
* the six ROW-LIVENESS flags are an ARITY axis — they change how many terms the
  inner sum has, and therefore its association — so they stay compile-time and are
  this emitter's only parameter.

Of the 63 live row masks the corpus asks for TWO.

WHY THE ARITY AXIS STAYED COMPILE-TIME, with both facts stated. The ZERO-PADDING
fold — bind a dead slot to an all-zero coefficient so a dropped term contributes
``+0.0`` — is measurably NOT bit-identical: ``x + 0.0`` turns ``-0.0`` into
``+0.0``, and a ``-0.0`` row is reachable. ``test_offdiag_constitutive_pml_real.py``
carries that witness. A DYNAMIC LOOP is a different fold and IS licensed: the
design notes (cuda-kernel-triton-transfer-assessment) §3.2 measured a
runtime-bounded loop bit-identical to the unrolled form at every arity 0-8
(156/156 cases, zero differing words, max ULP 0), because NVCC leaves the
arithmetic a strictly serial dependence chain on one accumulator. This family's
row accumulation is such a chain, so the axis COULD go runtime; it has not been
taken. It would buy one compiled source on this corpus instead of two.

THE ROW IS MEEP'S, IN DOUBLE (0.10). MEEP's ``step_update_EDHB`` forms
``fw[i] = gs*us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)`` where each OFFDIAG
is ``0.25*(...)`` with ``0.25`` a DOUBLE literal (step_generic.cpp:580-583, :611):
the pair sums, the two products and their sum round to float32, the scale and the
row sum are double, and the store rounds once. ``stepping._offdiagonal_terms``
returns that double row and ``stepping._rounded_row`` rounds it; this emitter
writes the same tree. A term is added only in a chunk that keeps the partner's
``chi1inv`` array (MEEP passes NULL for a deleted one, update_eh.cpp:191-196), and
the constitutive tail takes MEEP's per-chunk form (``constitutive_apply``'s
membership flag, the certified sibling's helper).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .coverage import (OFFDIAG_ROW_SLOTS, OFFDIAG_TRANSVERSE_PARTNERS,
                       OFFDIAG_WALL_MASK_AXES)

#: The one kernel this family emits. Its SOURCE varies with the row mask; its NAME
#: does not, because it is one family and NVRTC compiles one module per source.
KERNEL_NAME = "update_E_pml_real_offdiag"

#: The components this sub-step writes, in ``stepping.E_CONSTITUTIVE_TERMS`` order,
#: with each component's source volume and OWN axis — the axis whose HALF-INTEGER
#: coefficient pair the tail reads (MEEP's ``dsigw``).
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The D volume of each axis. The coupling reads the PARTNER axis's volume, which
#: with no poles admitted is that axis's D primary (``Fields.
#: displacement_minus_polarization_volumes``).
PARTNER_VOLUMES: Tuple[str, str, str] = ("Dx", "Dy", "Dz")

#: The kernel parameter name of each row coefficient, in
#: :data:`coverage.OFFDIAG_ROW_SLOTS` order — named for the pair it carries, so a
#: mispairing is visible in the emitted source rather than only in the physics.
ROW_PARAMETERS: Tuple[str, ...] = tuple(
    f"chi1inv_{row}_{partner}" for row, partner in OFFDIAG_ROW_SLOTS)

#: The kernel parameter name of each row's PRESENCE WORD, in the same slot order: bit
#: ``lx + 3*ly + 9*lz`` is set when the chunk with those three labels keeps that row's
#: ``chi1inv`` array (``offdiag_presence.presence_words`` builds them on the host).
KEPT_PARAMETERS: Tuple[str, ...] = tuple(
    f"kept_{row}_{partner}" for row, partner in OFFDIAG_ROW_SLOTS)

#: Every chunk kept: the word of a row whose array no chunk deletes.
ALL_CHUNKS_KEPT = (1 << 27) - 1

#: Every row mask this family can emit: the 63 non-empty subsets of the six slots.
#: An all-dead mask is NOT here — that configuration is the certified plain
#: constitutive kernel's, and the two families are disjoint by the install-time
#: zero-row drop.
LIVE_ROW_MASKS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple((value >> slot) & 1 for slot in range(len(OFFDIAG_ROW_SLOTS)))
    for value in range(1, 1 << len(OFFDIAG_ROW_SLOTS))
)

#: The chunk-record entries this kernel decodes: both Yee sub-lattices on every axis.
#: The tail reads each component's own axis on the half-integer lattice (``in_h_*``,
#: ``bounds_block.constitutive_entries``); the presence labels read each component's
#: other two axes on the integer lattice (an E component is integer there).
RECORD_ENTRIES: Tuple[Tuple[bool, int], ...] = tuple(
    (half, axis) for half in (False, True) for axis in range(3))

#: The chunk-record block, spelt as ``bounds_block.bounds_block(RECORD_ENTRIES)``
#: emits it (``test_offdiag_constitutive_pml_real.py`` pins the two equal); written
#: out here because this module imports nothing that pulls in NumPy.
RECORD_BLOCK = (
    "    // THE CHUNK RECORD (chunk_map.chunk_bounds_record): row r of this lattice and axis\n"
    "    // lies in a chunk that carries the axis's layer iff (a0 <= r < a1) || (b0 <= r < b1).\n"
    "    const int cbi_x_a0 = cb[0], cbi_x_a1 = cb[1], cbi_x_b0 = cb[2], cbi_x_b1 = cb[3];\n"
    "    const int in_i_x = (i >= cbi_x_a0 && i < cbi_x_a1) || (i >= cbi_x_b0 && i < cbi_x_b1);\n"
    "    const int cbi_y_a0 = cb[4], cbi_y_a1 = cb[5], cbi_y_b0 = cb[6], cbi_y_b1 = cb[7];\n"
    "    const int in_i_y = (j >= cbi_y_a0 && j < cbi_y_a1) || (j >= cbi_y_b0 && j < cbi_y_b1);\n"
    "    const int cbi_z_a0 = cb[8], cbi_z_a1 = cb[9], cbi_z_b0 = cb[10], cbi_z_b1 = cb[11];\n"
    "    const int in_i_z = (k >= cbi_z_a0 && k < cbi_z_a1) || (k >= cbi_z_b0 && k < cbi_z_b1);\n"
    "    const int cbh_x_a0 = cb[12], cbh_x_a1 = cb[13], cbh_x_b0 = cb[14], cbh_x_b1 = cb[15];\n"
    "    const int in_h_x = (i >= cbh_x_a0 && i < cbh_x_a1) || (i >= cbh_x_b0 && i < cbh_x_b1);\n"
    "    const int cbh_y_a0 = cb[16], cbh_y_a1 = cb[17], cbh_y_b0 = cb[18], cbh_y_b1 = cb[19];\n"
    "    const int in_h_y = (j >= cbh_y_a0 && j < cbh_y_a1) || (j >= cbh_y_b0 && j < cbh_y_b1);\n"
    "    const int cbh_z_a0 = cb[20], cbh_z_a1 = cb[21], cbh_z_b0 = cb[22], cbh_z_b1 = cb[23];\n"
    "    const int in_h_z = (k >= cbh_z_a0 && k < cbh_z_a1) || (k >= cbh_z_b0 && k < cbh_z_b1);\n"
    "\n")

#: The chunk label lines: per component, the code ``lx + 3*ly + 9*lz`` of the chunk its
#: cell belongs to, each axis read on the component's own sub-lattice there (half-integer
#: on its own axis, integer on the other two) at the row ``chunk_row`` resolves.
CHUNK_LINES = (
    "    // THE CHUNK EACH COMPONENT'S CELL BELONGS TO, as the code lx + 3*ly + 9*lz of its\n"
    "    // three record labels (chunk_coverage.chunk_owner_label), each axis on the\n"
    "    // component's own Yee sub-lattice there. A presence word is indexed by it.\n"
    "    const int wi = chunk_row(i, bc_x == BC_METALLIC);\n"
    "    const int wj = chunk_row(j, bc_y == BC_METALLIC);\n"
    "    const int wk = chunk_row(k, bc_z == BC_METALLIC);\n"
    "    const int chunk_Ex = chunk_label(i, cbh_x_a0, cbh_x_a1, cbh_x_b0, cbh_x_b1)\n"
    "        + 3 * chunk_label(wj, cbi_y_a0, cbi_y_a1, cbi_y_b0, cbi_y_b1)\n"
    "        + 9 * chunk_label(wk, cbi_z_a0, cbi_z_a1, cbi_z_b0, cbi_z_b1);\n"
    "    const int chunk_Ey = chunk_label(wi, cbi_x_a0, cbi_x_a1, cbi_x_b0, cbi_x_b1)\n"
    "        + 3 * chunk_label(j, cbh_y_a0, cbh_y_a1, cbh_y_b0, cbh_y_b1)\n"
    "        + 9 * chunk_label(wk, cbi_z_a0, cbi_z_a1, cbi_z_b0, cbi_z_b1);\n"
    "    const int chunk_Ez = chunk_label(wi, cbi_x_a0, cbi_x_a1, cbi_x_b0, cbi_x_b1)\n"
    "        + 3 * chunk_label(wj, cbi_y_a0, cbi_y_a1, cbi_y_b0, cbi_y_b1)\n"
    "        + 9 * chunk_label(k, cbh_z_a0, cbh_z_a1, cbh_z_b0, cbh_z_b1);\n")


# =============================================================================
# THE DEVICE CODE
# =============================================================================
#
# NINE THINGS DECIDE BIT-IDENTITY HERE, and every one of them is a SILENT wrong
# answer: a smooth, converged, plausible field with the wrong tensor in it.
#
# 1. THE TAIL IS MEEP'S PER-CHUNK CONSTITUTIVE FORM (``stepping.
#    _apply_constitutive_regions``): on a cell whose chunk carries the component's
#    own layer the accumulation ``f + ((kps*src) - (kms*prev))`` with ``fw = src``;
#    elsewhere ``f = src`` and ``fw`` untouched. ``constitutive_apply`` below is
#    character-for-character the certified sibling's, and a test welds the two.
# 2. THE COEFFICIENT INDEX IS THE COMPONENT'S OWN AXIS -- MEEP's ``dsigw``, not
#    the dsig/dsigu cycle the curl recurrence uses.
# 3. ``prev`` IS READ BEFORE ``fw`` IS WRITTEN.
# 4. THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS -- see ``offdiag_term``.
# 5. THE TWO SHIFTS GO IN OPPOSITE DIRECTIONS: half a cell DOWN the partner's axis
#    on the raw partner volume, then the already-multiplied product half a cell UP
#    this component's own axis.
# 6. THE WALL MASK ZEROES EACH TERM'S FLOAT32 SUM, before its scale
#    (``stepping._mask_metallic_wall_coupling`` inside the partner loop), and asks
#    the grid's DECLARATION (``is_metallic and not is_mirrored``), which is a
#    different question from the resolved ghost rule -- hence ``wm_*`` beside
#    ``bc_*`` rather than derived from it.
# 7. A COMPONENT WITH NO SURVIVING ROW KEEPS THE PURE DIAGONAL ARITHMETIC:
#    ``_offdiagonal_terms`` returns None and ``update_E`` then forms ``gs*us`` in
#    float32 and never a double row.
# 8. THE SCALE AND THE ROW SUM ARE DOUBLE: ``0.25 * (double)sum`` is exact, the row
#    starts at ``(double)(gs * us)`` (one float32 product, widened), the terms are
#    added left to right in partner order, and ``(float)row`` rounds once.
# 9. A TERM IS ADDED ONLY WHERE THE CELL'S CHUNK KEEPS THE ROW'S ARRAY: MEEP drops
#    the term in a chunk that deleted it, which differs from adding a zero exactly
#    when the row so far is -0.

PRELUDE = r'''
#define BC_PERIODIC 0
#define BC_METALLIC 1

// stepping._shift_up, PERIODIC and METALLIC branches only: the neighbour COORDINATE
// one cell up an axis, or -1 where the metallic far-face ghost makes the sample an
// exact zero. MIRROR and CYL_AXIS are refused by covers_real_pml_offdiag_constitutive,
// and real storage cannot carry a Bloch phase, so those branches have no code here.
//
// WHY A COORDINATE AND NOT A VALUE, where the certified curl's shift_up returns a
// float: this stencil needs the SAME shifted position three times over -- the partner
// volume at the shifted node, the partner volume at the CORNER (up this component's
// axis and down the partner's), and the COEFFICIENT at the shifted node, because the
// multiply sits between the two shifts and u is read at the up index rather than
// shifted with the product. Resolving once to a coordinate and loading three times is
// the same arithmetic and one rule instead of three.
__device__ __forceinline__ int coord_up(int a, int n, int bc) {
    if (a + 1 < n) return a + 1;
    return (bc == BC_METALLIC) ? -1 : 0;
}

// stepping._shift_down, same two branches, same convention. An invariant axis
// (n == 1, always PERIODIC) needs no case of its own: both wraps return the same
// cell, so the partner pair is 2*g rather than the curl's exact zero -- which is what
// MEEP's stride(d) = 0 double-read computes.
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

// A ghosted load: index -1 is the metallic zero ghost, which the array path writes
// into that plane as an exact +0.0 (_shift_up, _shift_down).
__device__ __forceinline__ float ghosted(const float* g, int index) {
    return (index < 0) ? 0.0f : g[index];
}

// One partner's OFFDIAG sum -- MEEP step_generic.cpp:580-583 as
// stepping._offdiagonal_terms associates it:
//
//     OFFDIAG(u, g, sx) = 0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])
//
// THIS RETURNS THE FLOAT32 SUM INSIDE THE SCALE. The pair sums, the two products and
// their sum are realnum (float32) operations; 0.25 is a double literal, so the scale
// and everything after it is double, and the caller applies the wall mask to this sum
// and then scales it in double.
//
// THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS. u[i] multiplies the two-point
// partner average AT ITS OWN NODE and u[i+s] the average at the next node up this
// component's axis, because MEEP registers the off-diagonal entry at the component's
// Yee site minus half a cell along its own axis -- the integer node
// (anisotropic_averaging.cpp:248-257, `here - shift1`). Hoisting to a plain four-point
// average times u[i] is the same ALGEBRA only for a uniform coefficient, and a
// different float32 number even then: a*u + b*u and (a+b)*u do not round alike.
//
// LOADING u AT THE COMPOSED UP INDEX IS THE SAME BITS AS SHIFTING THE FORMED PRODUCT.
// A shift is data movement; the coefficient wraps plainly on a periodic axis (it is
// periodic and phase-free), and on a metallic far face both the pair and the
// coefficient read the -1 ghost, so the product is an exact +0.0 -- which is
// precisely what the array path's _shift_up writes into that plane.
//
// NO UNARY MINUS ANYWHERE ON THIS PATH. CUDA lowers -x to neg.f32 rather than to
// 0.0f - x, so it does NOT canonicalize signed zeros the way the sibling track's
// platform does. That track's negation idiom must not be ported here, and this body
// is written so the answer never matters.
__device__ __forceinline__ float offdiag_term(
    const float* g, const float* u, int home, int down, int up, int corner
) {
    float near_pair = g[home] + ghosted(g, down);
    float far_pair = ghosted(g, up) + ghosted(g, corner);
    return (near_pair * u[home]) + (far_pair * ghosted(u, up));
}

// The record label of the chunk a stored row belongs to along one axis --
// chunk_coverage.chunk_owner_label: 0 inside the low run [a0, a1), 2 inside the high
// run [b0, b1), 1 otherwise. The composer admits a division only where these labels
// name its chunk intervals one to one along every axis
// (chunk_coverage.chunk_division_refusal with CHUNK_AXES_OFFDIAG).
__device__ __forceinline__ int chunk_label(int r, int a0, int a1, int b0, int b1) {
    if (r >= a0 && r < a1) return 0;
    if (r >= b0 && r < b1) return 2;
    return 1;
}

// The row a cell's chunk is read at along one axis on the INTEGER lattice: the
// cell's own row, except the metallic wall row 0, which no chunk owns and which
// ChunkLayout.owner_volume reads at the nearest owned row, row 1. The half-integer
// lattice has no unowned row on a metallic axis, and a wrapping axis none at all.
__device__ __forceinline__ int chunk_row(int a, int wall) {
    return (wall && a == 0) ? 1 : a;
}

// Whether the chunk with code c keeps a row's chi1inv array: bit c of that row's
// presence word (ChunkLayout.offdiag_kept; MEEP passes NULL for an array the chunk
// deleted, update_eh.cpp:191-196, and step_update_EDHB then drops the term there).
__device__ __forceinline__ int chunk_keeps(int kept, int c) {
    return (kept >> c) & 1;
}

// stepping._apply_constitutive_regions, MEEP's step_update_EDHB per structure chunk.
// CHARACTER FOR CHARACTER the certified sibling's constitutive_apply
// (constitutive_kernels.py); a test welds the two texts so the tail cannot drift
// between the two families that share it.
__device__ __forceinline__ void constitutive_apply(
    float* __restrict__ f, float* __restrict__ fw, int idx, float src,
    float kps, float kms, int dsigw
) {
    float prev = dsigw ? fw[idx] : 0.0f;
    if (dsigw) fw[idx] = src;
    float value = dsigw ? (f[idx] + ((kps * src) - (kms * prev))) : src;
    f[idx] = value;
}
'''

# THE SIGNATURE CARRIES ONLY THE LIVE ROW COEFFICIENTS AND THEIR PRESENCE WORDS. The
# sibling track binds all six pointers and points the dead ones at the component's
# own D volume, on the argument that a specialized branch stops the read. That works,
# and it leaves a pointer bound to an array it must never touch -- a shape a later
# edit can read by accident, and one that makes __restrict__ on the D arguments a
# promise the caller cannot keep. Emitting only the live parameters removes both.
#
# WHICH POINTERS CARRY __restrict__, AND WHY THE READ-ONLY ONES DO NOT. The six
# outputs are pairwise distinct and disjoint from every input -- the predicate's
# alias clause and the launcher's check are what make that true -- so they carry
# the promise, as do the six PML coefficient vectors, which no field can alias, and
# the chunk record. The read-only VOLUMES do not: an isotropic install hands the same
# inverse permittivity three times, and a row coefficient may legally alias another
# row's, an epsilon volume or a D volume. Restrict on mutually aliasing arguments is
# a promise the caller cannot keep, and nothing is lost by not making it but a load
# hint.
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
    int wm_x, int wm_y, int wm_z,
    const int* __restrict__ cb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int nyz = ny * nz;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

__RECORD_BLOCK__    // Both neighbour coordinates on every axis: one term reads an axis DOWN (the
    // partner's) and another reads an axis UP (its own), and a component's two
    // terms take different partners, so every axis can be needed either way.
    int di = coord_dn(i, nx, bc_x), ui = coord_up(i, nx, bc_x);
    int dj = coord_dn(j, ny, bc_y), uj = coord_up(j, ny, bc_y);
    int dk = coord_dn(k, nz, bc_z), uk = coord_up(k, nz, bc_z);

    // Wall-plane predicates for the coupling mask. FACE 0 ONLY: the high wall is
    // already the shift-up zero ghost, and stepping._mask_metallic_wall_coupling
    // writes only _face(axis, 0).
    int at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);

__CHUNK_LINES__
    // The three components are independent, and that is read off the loop rather
    // than assumed: the coupling reads D and the inverse permittivity, the tail
    // writes E and f_w_E. Four disjoint sets, so the order the array path runs
    // them in is not a data dependence -- which is what makes a NEIGHBOUR read
    // across components safe here.
__SRC_Ex__

__SRC_Ey__

__SRC_Ez__
}
'''


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
    """One partner's term sum for one component, as source lines.

    ``offset`` is 0 for MEEP's ``cycle_direction(dim, d_ec, 1)`` and 1 for
    ``(..., 2)``. The partner axis, the coefficient slot, the partner volume and all
    three shifted indices come from that one number, so a mispairing would have to be
    introduced deliberately rather than by a typo.
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


def _wall_mask_lines(component: int, offset: int) -> List[str]:
    """``stepping._mask_metallic_wall_coupling`` for one term's float32 sum.

    Face 0 of every axis whose Yee shift is 0
    (:data:`coverage.OFFDIAG_WALL_MASK_AXES`), ascending — the mask's own loop
    order. WHICH axes can be masked is fixed by the component and compiled in;
    WHETHER each one is masked is the runtime ``wm_*`` flag, because that is a
    property of the grid rather than of the arithmetic. A select, never an
    arithmetic operation, so it cannot round; it writes +0.0, as the array path does.
    """
    term = f"term_{E_TERMS[component][0]}_{offset}"
    return [f"    {term} = ({_WALL_FLAGS[axis]} && {_WALL_PREDICATES[axis]})"
            f" ? 0.0f : {term};"
            for axis in OFFDIAG_WALL_MASK_AXES[component]]


def _component_source(component: int, row_mask: Sequence[int]) -> str:
    """One component's whole block, on the arm its two row slots select.

    Four arms: none, offset-1 only, offset-2 only, both. THE NONE ARM IS THE
    CERTIFIED PLAIN CONSTITUTIVE KERNEL'S COMPONENT BODY — ``src = D * inv_eps`` in
    float32 and the same tail — because ``_offdiagonal_terms`` returns None for a
    component with no surviving row and ``update_E`` then never forms a double row.
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
            # The mask zeroes the term's float32 sum before its scale.
            lines.extend(_wall_mask_lines(component, offset))
        # The row: gs*us in float32, widened; each kept term's sum scaled by the double
        # 0.25 and added in partner order (cycle 1, then 2); rounded once by the store.
        lines.append(f"    double row_{name} = (double)(gs_{name} * us_{name});")
        for offset in live:
            kept = KEPT_PARAMETERS[2 * component + offset]
            lines.append(f"    if (chunk_keeps({kept}, chunk_{name}))"
                         f" row_{name} = row_{name} + (0.25 * (double)term_{name}_{offset});")
        lines.append(f"    float src_{name} = (float)row_{name};")
    lines.append(f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},"
                 f" kps_{axis_letter}[{index_letter}],"
                 f" kms_{axis_letter}[{index_letter}], in_h_{axis_letter});")
    return "\n".join(lines)


def _row_parameter_lines(row_mask: Sequence[int]) -> str:
    """The live coefficient parameters, then their presence words, in slot order."""
    live = [slot for slot, flag in enumerate(row_mask) if flag]
    return "\n".join([f"    const float* {ROW_PARAMETERS[slot]}," for slot in live]
                     + [f"    int {KEPT_PARAMETERS[slot]}," for slot in live])


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

    The BOUNDARY and WALL axes, the chunk record and the presence words are NOT baked
    in — they are runtime arguments, for the reason the module docstring gives — so the
    row mask is this family's only source axis, and the corpus drives two of its 63
    values.
    """
    mask = normalized_row_mask(row_mask)
    body = (TEMPLATE.replace("__ROW_PARAMETERS__", _row_parameter_lines(mask))
            .replace("__RECORD_BLOCK__", RECORD_BLOCK)
            .replace("__CHUNK_LINES__", CHUNK_LINES))
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
    before the first store (the auxiliary and the old field only under the cell's
    membership flag), the certified ``constitutive_apply`` stepping the registers, the
    write-back in the certified store order. The rule issues it only when its inverse
    returns :func:`offdiag_source` byte for byte. ``offdiag_source`` is looked up when
    this is called, so a harness that rewrites the certified text reaches this one too.
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
