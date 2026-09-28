"""Raw-CUDA off-diagonal (tensor epsilon) ``update_E`` ON A MIRROR-FOLDED GRID.

THE FAMILY THE PREVIOUS ROUND SPECIFIED AND DID NOT BUILD.
``coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION`` records ``installed: False`` and
carries the measurement that says what is missing: on the GPU host's RTX A6000, under
both float32 subnormal policies, the SHIPPED ``update_E_pml_real_offdiag``
run on folded grids scored 96/96 identical on the configurations where the fold
cannot reach the arithmetic and 352/352 DIVERGENT on the complementary ones -- with
every one of the ~132,514 differing words lying on the ghost-delta planes and ZERO
elsewhere. A divergence that is exactly on the planes whose ghost rule a kernel
does not implement is not a failure; it is a specification. That record also names
the work in as many words (``what_would_move_the_twenty``):

    "a MIRROR branch in the emitted coord_dn returning
     stepping.MIRROR_SOURCE_INDEX, plus a per-term parity argument the kernel has
     no concept of today (a signature change, not a branch), plus a third boundary
     code and the launcher mapping for it -- a new kernel variant with its own
     byte gate, not a line in this module"

This module is that variant. It changes NOT ONE BYTE of ``offdiag_emitter.py`` or
``offdiag_constitutive_kernels.py``, whose ``offdiag_2026-08-16`` certification
block is pinned by digest, and it does not touch
``covers_real_pml_offdiag_constitutive``: that predicate's two fold refusals STAY,
and they are the disjointness seam against this file, exactly as
``symmetry.folded_constitutive_coverage``'s off-diagonal clause is the seam on the
sibling track.

WHAT IT TRANSCRIBES. ``stepping.update_E`` (stepping.py:954) with an active PML,
at least one surviving off-diagonal ``chi1inv`` row, no nonlinearity, no
polarization, and at least one axis resolving to ``MIRROR`` through
``stepping._boundary_kinds`` -- the ``elif offdiagonal:`` branch (stepping.py:
1001-1008). With no poles admitted ``displacement_minus_polarization_volumes``
aliases each source to its D primary (fields.py:1107-1138). Per component ``c``
with own axis ``a``::

    constitutive = D_c * us_c                                # stepping.py:1005
    per surviving partner (offset 1 then 2, cycle X->Y->Z, :1235-1237):
        pair    = g + shift_down(g, partner_axis)            # :1214-1216
        product = pair * coefficient                         # :1217
        term    = 0.25 * (product + shift_up(product, a))    # :1219-1220
        total   accumulates term(offset 1) then term(offset 2)  # :1221
    _mask_metallic_wall_coupling(total)                      # :1223, :1250-1254
    constitutive = (D_c * us_c) + total                      # :978-979
    prev = f_w_c ; f_w_c = constitutive                      # :2065-2096
    E_c += kps_a_h * f_w_c ;  E_c -= kms_a_h * prev          # half-integer, :986

Every line of that is ``offdiag_emitter``'s, unchanged. THE WHOLE DELTA IS WHAT
THE TWO SHIFT HELPERS DO ON A MIRROR AXIS, and it is three facts, each transcribed
with its line and each measured rather than argued.

=============================================================================
FACT 1. THE FOLD ENTERS THROUGH EXACTLY ONE LEG: THE PARTNER-AXIS DOWN SHIFT
=============================================================================

``stepping._offdiagonal_terms`` calls ``_shift_down`` WITH a component name and
the plane's parity, on ``axis = partner_axis`` (stepping.py:1243-1245). On a
MIRROR boundary that helper takes stepping.py:1870-1872::

    shifted[_face(axis, 0)] = (_symmetry_phase(component, axis, mirror_phase)
                               * _mirror_source(field, axis))

``_mirror_source`` (stepping.py:1582-1588) is ``field[_face(axis,
MIRROR_SOURCE_INDEX)]`` with ``MIRROR_SOURCE_INDEX = 2`` (stepping.py:160) -- a
LIVE, parity-weighted INTERIOR plane, which is neither ``coord_dn``'s periodic
wrap nor its metallic ``-1`` zero. So the certified kernel moves bytes on that
plane whichever of its two codes the launcher hands the folded axis, which is what
the device sweep measured.

AND THE WALL MASK DOES NOT TAKE IT BACK. ``_mask_metallic_wall_coupling`` asks
``is_metallic(axis) and not is_mirrored(axis)`` (stepping.py:1282) and therefore
ABSTAINS on a fold plane -- deliberately, with the array path's own measurement
beside it (stepping.py:1266-1277: zeroing the fold plane the way the metallic rule
does costs 2.0e-02 even / 3.5e-03 odd). The fold plane's coupling is alive, and
its value is the mirror ghost's.

=============================================================================
FACT 2. THE OWN-AXIS UP SHIFT IS AN EXACT ZERO ON A FOLD, WHICH ``coord_up``
        ALREADY SPELLS AS ITS METALLIC ARM
=============================================================================

``_offdiagonal_terms`` calls ``_shift_up`` with FOUR arguments (stepping.py:
1248-1249) -- no ``component``, no ``mirror_phase``, no ``reflect_row`` -- so the
folded-PERIODIC reflect branch (stepping.py:1819-1827) cannot fire and BOTH mirror
terminations fall through to ``shifted[_face(axis, -1)] = 0`` (stepping.py:
1828-1830). That is byte-identical to what ``coord_up`` writes for METALLIC and is
NOT what it writes for PERIODIC. It is transcription fidelity, not a physics
claim: the sibling track records the same reading as an ARRAY-PATH FINDING
(``triton_kernels/folded_offdiag_update_e.py``, "ARRAY-PATH FINDING"), because on
a folded PERIODIC axis MEEP's ghost past the stored top is the parity-weighted
image of ``_far_reflect_rows``' row and the coupling does not ask for it. That
finding belongs to the sessions that own ``stepping.py`` and ``driver.py``. THIS
KERNEL REPRODUCES THE ARRAY PATH, byte for byte, either way.

=============================================================================
FACT 3. THE TWO FOLD TERMINATIONS ARE ONE CODE HERE, AND THAT IS A PREDICTION
=============================================================================

``coverage.REAL_CURL_BC_CODES`` splits a folded PERIODIC axis from a folded
METALLIC one because the CURL masks the folded-periodic top plane
(``stepping._stored_past_owned``, stepping.py:1503-1518) and reflects past it.
``update_E`` has NO ownership mask at all -- ``_mask_non_owned_cells``
(stepping.py:1912) has exactly three call sites, ``step_B``, ``step_D`` and
``_bfast_term``, and no constitutive function is among them -- and by FACT 2 it
has no reflect row either. So ONE ``BC_MIRROR`` code serves both terminations.

That is a prediction about the arithmetic, so it is carried as a NULL CONTROL in
the gate rather than assumed: the sweep folds each axis under BOTH declared
terminations and the two must be bit-identical. The cross-check that the
termination is READABLE at all is kept anyway
(``coverage._fold_termination_problem``): a grid whose ``stored_cells >
owned_cells`` and whose ``is_metallic`` disagree has one of the two drifted, and
neither can then be trusted about anything.

=============================================================================
WHAT THE MIRROR WEIGHT IS, EXACTLY, AND WHY IT IS A RUNTIME FLOAT
=============================================================================

``_shift_down`` is called with ``component = "D" + AXIS_NAMES[partner_axis]`` on
``axis = partner_axis`` (stepping.py:1243-1245), and every D component has Yee
shift 1 on its OWN axis (fields.py:214-219). ``mirror_parity`` (fields.py:117)
returns ``phase * (-1 if flipped else 1)`` with ``flipped = (own_axis == axis) !=
is_pseudovector``, so for every partner and every axis it collapses to exactly
``-phase``. MEASURED here rather than transcribed: :func:`mirror_ghost_weights`
derives the value through ``fields.mirror_parity`` ITSELF, and
``test_folded_offdiag.py`` pins the collapse over 3 axes x 2 phases.

IT ARRIVES AS A RUNTIME ``float`` ARGUMENT, exactly ``+1.0f`` or ``-1.0f``, and is
applied as a multiply on the ghost lane alone. THE SIBLING TRACK'S REASON FOR THIS
DOES NOT TRANSFER AND A DIFFERENT ONE DOES. There, a unary minus is avoided
because Triton 3.1.0 lowers ``-x`` as ``0.0 - x`` and canonicalizes signed zeros;
CUDA lowers ``-x`` to ``neg.f32`` and does not, which
``offdiag_emitter.PRELUDE`` already records. The reason here is the PREDICATE'S:
the weight is data the launcher reads off the grid, so a plane whose phase is
unreadable must be a REFUSAL rather than a NaN that launches, and a value the
predicate can inspect is one it can refuse.

THE WEIGHT IS APPLIED ONLY ON THE GHOST LANE, never as a blanket ``1.0f *``. The
array path multiplies exactly one plane (stepping.py:1871-1872) and leaves every
other sample untouched, so a kernel that multiplied every lane by a runtime 1.0f
would be introducing an operation the oracle does not perform. ``1.0f * x`` is the
identity on the bits for every float32 x under ``-ftz=false`` -- which is what
NVRTC compiles by default and what this family's options leave alone -- but "is
the identity today under this flag" is a weaker claim than "is not there", and the
subnormal-band value class is precisely where the difference could bite. So
:data:`PRELUDE`'s ``ghosted_mirror`` selects between the weighted and the plain
load, and the ``mg = 0`` arm of it is ``ghosted``'s body verbatim.

THAT IS ALSO WHAT MAKES THE UNFOLDED REDUCTION STRUCTURAL. With no axis folded,
every ``mg`` is 0 on every lane, ``ghosted_mirror`` returns ``ghosted``'s value
by BRANCH rather than by a multiply a compiler is trusted to fold away, and the
emitted body is the certified family's arithmetic exactly. The gate measures the
reduction against the certified kernel rather than claiming it, and
``test_folded_offdiag.py`` measures it at the merge bar against the certified
EMITTER through the same NumPy evaluator.

=============================================================================
WHAT IS SPECIALIZED AND WHAT IS NOT
=============================================================================

``offdiag_emitter``'s split, unchanged, and for its reasons: the boundary and wall
axes are BRANCH axes (they select an index or a predicated zero and change no
float operation and no association) so they are runtime ``int`` arguments; the six
ROW-LIVENESS flags are an ARITY axis (they change how many terms the inner sum has
and therefore its association) so they stay the emitter's only source parameter.

THE GHOST WEIGHTS FOLLOW THE BOUNDARY AXES AND ARE RUNTIME TOO, and the SIBLING
TRACK'S EXTRA CONSTEXPR IS DELIBERATELY NOT PORTED. ``folded_offdiag_update_e.py``
carries ``MG_X``/``MG_Y``/``MG_Z`` beside ``BCX``/``BCY``/``BCZ`` -- the same fact
twice -- so that its unfolded arm can be the certified body VERBATIM instead of a
multiply by 1.0. Here ``bc_*`` is already a runtime value, that benefit does not
exist, and a second spelling of one fact is a place for the index redirect and the
parity weight to disagree about which lane is the ghost: a plane of wrong values,
not a crash. So the kernel derives ``mg_* = (bc_* == BC_MIRROR) && at_*`` from
``bc_*`` in ONE line and nothing else knows the answer.

=============================================================================
PLATFORM FACTS THIS FILE DEPENDS ON
=============================================================================

* ``--fmad=false`` is CORRECTNESS, not tuning, and the option tuple is SPELLED
  HERE rather than imported so that loading this file by path cannot pick up a
  different one than a gate compiled. The certified sibling measured its unguarded
  control diverging on 384/384, and this family's tail and row product are that
  one plus a select, so the guard is a precondition here too -- carried as the
  gate's own control rather than inherited.
* SIGNED ZERO: CUDA lowers ``-x`` to ``neg.f32``, not to ``0.0f - x``. No unary
  minus appears on any float path here regardless; the weight is a bound argument.
* DIVISION: none anywhere on this path.
* THE DEVICE SOURCE IS PURE ASCII, a compile requirement rather than a style rule:
  ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a bare
  ``open(..., 'w')``, so the bytes go through the interpreter's LOCALE encoding --
  ASCII under C/POSIX, which is what a non-interactive shell on the validation
  host gets. The tests scan AND encode every emitted source, per row mask.

=============================================================================
IMPORTABLE WITHOUT CUPY, AND NOT CERTIFIED UNTIL THE RECORD SAYS SO
=============================================================================

``cupy`` is imported INSIDE the launcher, never at module scope, on
``no_pml_curl.py``'s precedent and for its reason: the predicate is consumed by
the coverage census, which runs on a laptop. Everything above the launcher --
emitter, predicate, tables -- is stdlib plus one lazy ``fields.mirror_parity``.

NOTHING DISPATCHES THIS. No module in ``meep_gpu/`` imports ``cuda_kernels`` at
all (``test_package_boundary.py`` pins the absence in both directions), so a
predicate returning True licenses a MEASUREMENT and not a production step. What
has and has not been measured on a device is :data:`FOLDED_OFFDIAG_ADMISSION`, and
nothing here may be read as a device verdict while that record's ``host`` field is
None.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import offdiag_emitter as _flat
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)

# ---------------------------------------------------------------------------
# The constants the kernel and the host agree on
# ---------------------------------------------------------------------------
#
# Imported BY VALUE from the certified family wherever it already owns the fact,
# so there is one home per fact and a test can assert the identity rather than
# diff two spellings.

#: The one kernel this family emits. Its SOURCE varies with the row mask; its NAME
#: does not, because it is one family and NVRTC compiles one module per source.
#: Deliberately NOT ``offdiag_emitter.KERNEL_NAME``: a second kernel wearing the
#: certified one's name would make ``certification.json``'s partition test and any
#: NVRTC binary observation ambiguous about which body it saw.
KERNEL_NAME = "update_E_pml_real_folded_offdiag"

#: Byte-identical to the array path with a device verdict behind it. The evidence is
#: :data:`FOLDED_OFFDIAG_ADMISSION` and ``certification.json``'s
#: ``cuda_folded_offdiag_2026-08-21`` block, which names the same two artifacts; a
#: name here without a record block is the failure ``test_kernel_partition.py``
#: exists to catch. Spelled as a LITERAL rather than as ``(KERNEL_NAME,)`` because
#: the partition readers evaluate this with ``ast.literal_eval`` on a host with no
#: CuPy, where a name reference is not evaluable; the walk pins it against the
#: kernel declaration in the emitted device text, so the two cannot drift.
CERTIFIED_KERNELS = ("update_E_pml_real_folded_offdiag",)

#: Shipped but not gated. EMPTY, and the partition requires every shipped kernel to
#: be in exactly one of the two sets, so a kernel added to this file without a gate
#: verdict fails there rather than shipping unmeasured. Spelled as a dict WITHOUT a
#: type annotation for the reason ``constitutive_kernels.py`` records: the partition
#: readers walk the syntax tree so they run where there is no CuPy, and an annotated
#: assignment is an ``ast.AnnAssign`` the plain-assignment readers do not match --
#: annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}

#: ``stepping.MIRROR_SOURCE_INDEX`` (stepping.py:160) -- the stored row a mirror
#: ghost reflects from. Restated here for this package's engine-import-free
#: contract, exactly as ``coverage.py`` restates ``stepping._boundary_kinds``; a
#: test pins the two equal.
MIRROR_SOURCE_INDEX = 2

#: The three boundary codes this kernel implements. :data:`coverage.BC_CODES`
#: plus one: ``stepping._boundary_kinds`` answers ``"mirror"`` at BOTH
#: terminations of a fold (``coverage._boundary_kinds_from``), and FACT 3 of the
#: module docstring is why one code serves both here where the curl pair needs
#: two.
BC_MIRROR_CODE = 2
FOLDED_BC_CODES: Dict[str, int] = dict(_coverage.BC_CODES, mirror=BC_MIRROR_CODE)

#: The components this sub-step writes, with each component's source volume and
#: OWN axis. ``offdiag_emitter.E_TERMS``, imported.
E_TERMS: Tuple[Tuple[str, str, int], ...] = _flat.E_TERMS

#: The D volume of each axis, and the kernel parameter name of each row
#: coefficient in :data:`coverage.OFFDIAG_ROW_SLOTS` order. Both the certified
#: emitter's.
PARTNER_VOLUMES: Tuple[str, str, str] = _flat.PARTNER_VOLUMES
ROW_PARAMETERS: Tuple[str, ...] = _flat.ROW_PARAMETERS

#: Every row mask this family can emit -- the 63 non-empty subsets. An all-dead
#: mask belongs to the plain constitutive kernel, which since 2026-08-19 admits a
#: fold on its own evidence (:data:`coverage.CONSTITUTIVE_FOLD_ADMISSION`), so the
#: two are disjoint here by exactly the seam they are disjoint by unfolded.
LIVE_ROW_MASKS: Tuple[Tuple[int, ...], ...] = _flat.LIVE_ROW_MASKS

#: Lanes per block. The geometry every certified hand-CUDA family uses.
_FOLDED_OFFDIAG_THREADS = 256

#: NVRTC compile options -- CORRECTNESS, not performance. Spelled here rather than
#: imported so that loading this file by path cannot pick up a different tuple
#: than the one a gate compiled; a test pins it equal to both siblings'.
_COMPILE_OPTIONS = ('--fmad=false',)


# =============================================================================
# THE DEVICE CODE
# =============================================================================
#
# THE FOUR HELPERS THE CERTIFIED FAMILY ALREADY OWNS ARE READ OUT OF ITS SOURCE,
# NOT COPIED. ``offdiag_emitter.PRELUDE`` is a module-level string constant in a
# stdlib-only sibling, so it can be sliced rather than re-typed -- the route
# ``no_pml_curl._sibling_prelude`` takes, for its reason: two copies of a helper
# are equal only until someone edits one, and a silent divergence in ``flat`` or
# in the welded ``constitutive_apply`` tail would be a wrong answer no diff of
# THIS file would show. A test asserts each extracted block still appears verbatim
# in the sibling and in every emitted source.


def _sibling_block(name: str) -> str:
    """One ``__device__`` helper, with its comment, out of the certified prelude.

    Anchored on the DECLARATION line and terminated by the first line that is
    exactly ``}`` -- the shape every helper in that prelude has. The preceding
    comment block travels with it, because the comment is where the transcription
    citation lives and a helper carried without its citation is a helper whose
    provenance has to be remembered.

    RAISES if the block cannot be found. A silent fallback to a local copy is the
    one failure this indirection exists to prevent.
    """
    lines = _flat.PRELUDE.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("__device__ __forceinline__") and f" {name}(" in line:
            start = index
            break
    if start is None:
        raise RuntimeError(
            f"offdiag_emitter.PRELUDE no longer declares {name!r}; this family "
            f"reads that helper out of the certified sibling rather than copying "
            f"it, so a rename there must fail here instead of forking the "
            f"arithmetic")
    head = start
    while head > 0 and lines[head - 1].startswith("//"):
        head -= 1
    end = start
    while end < len(lines) and lines[end] != "}":
        end += 1
    if end >= len(lines):
        raise RuntimeError(f"{name!r} in offdiag_emitter.PRELUDE has no closing "
                           f"brace on its own line")
    return "\n".join(lines[head:end + 1])


#: The helpers taken verbatim from the certified family, named as data so a test
#: can assert every one is still found and still lands in the emitted source.
SIBLING_HELPERS: Tuple[str, ...] = ("flat", "ghosted", "constitutive_apply")


def _shared_prelude() -> str:
    return "\n\n".join(_sibling_block(name) for name in SIBLING_HELPERS)


#: This family's own device code. Everything here is either NEW (the mirror arms)
#: or DIFFERENT from the certified sibling's by exactly the mirror arm, and each
#: piece carries the ``stepping`` line it transcribes.
_OWN_PRELUDE = r'''
#define BC_PERIODIC 0
#define BC_METALLIC 1
#define BC_MIRROR 2

// stepping.MIRROR_SOURCE_INDEX (stepping.py:159). MEEP symmetry.cpp runs the
// transform on doubled coordinates, so with the halved grid's origin at io = -2
// the ghost at index -1 maps onto stored cell 2 for the iyee = 1 components the
// backward differences read. stepping._mirror_source (stepping.py:1535-1541)
// RAISES below that many stored cells, which is why the predicate refuses a
// folded axis with <= MIRROR_ROW cells by name.
#define MIRROR_ROW 2

// stepping._shift_up (stepping.py:1723), the PERIODIC branch and the joint
// METALLIC / MIRROR one: the neighbour COORDINATE one cell up an axis, or -1
// where the ghost makes the sample an exact zero.
//
// MIRROR SHARES METALLIC'S ARM HERE AND THAT IS THE TRANSCRIPTION, NOT A
// SHORTCUT. stepping._offdiagonal_terms calls _shift_up with four arguments
// (stepping.py:1219-1220) -- no component, no mirror_phase, no reflect_row -- so
// the folded-PERIODIC reflect branch (stepping.py:1772-1780) cannot fire and both
// mirror terminations fall through to shifted[_face(axis, -1)] = 0
// (stepping.py:1781-1783). A folded axis that wrapped here instead would move
// bytes on the last stored plane, which is exactly what the 2026-08-20 device
// sweep measured for the mirror_as_periodic substitution.
__device__ __forceinline__ int coord_up(int a, int n, int bc) {
    if (a + 1 < n) return a + 1;
    return (bc == BC_PERIODIC) ? 0 : -1;
}

// stepping._shift_down (stepping.py:1787), all three branches.
//
// THE MIRROR ARM IS THIS FAMILY'S REASON FOR EXISTING. On a folded axis the array
// path writes _symmetry_phase(component, axis, mirror_phase) *
// _mirror_source(field, axis) into face 0 (stepping.py:1823-1825) -- the stored
// row MIRROR_ROW, weighted by the plane's parity. It is neither the periodic wrap
// nor the metallic zero, which is why the certified kernel's two branches cannot
// serve a fold and why this one returns a real interior row here. The WEIGHT is
// applied by the caller, on the ghost lane alone; see ghosted_mirror.
//
// An invariant axis (n == 1, always PERIODIC) needs no case of its own: both
// wraps return the same cell, so the partner pair is 2*g rather than an exact
// zero, which is what MEEP's stride(d) = 0 double-read computes.
__device__ __forceinline__ int coord_dn(int a, int n, int bc) {
    if (a > 0) return a - 1;
    if (bc == BC_METALLIC) return -1;
    if (bc == BC_MIRROR) return MIRROR_ROW;
    return n - 1;
}

// A ghosted load that carries the mirror parity ON THE GHOST LANE AND NOWHERE
// ELSE. mg is 1 only at face 0 of an axis whose code is BC_MIRROR, and it is
// derived from bc_* in one place so the index redirect above and this weight
// cannot disagree about which lane is the ghost.
//
// THE mg == 0 ARM IS ``ghosted``'S BODY VERBATIM, on purpose: an unfolded axis
// then computes the certified family's arithmetic by CONSTRUCTION rather than by
// trusting 1.0f * x to be the identity under whatever flush policy the run
// carries. That is what makes the unfolded reduction a structural claim the gate
// confirms instead of a compiler-version-dependent one.
//
// The array path performs exactly one multiply, on exactly this plane
// (stepping.py:1824-1825), so a blanket weight would be an operation the oracle
// does not have.
__device__ __forceinline__ float ghosted_mirror(const float* g, int index,
                                                int mg, float w) {
    if (index < 0) return 0.0f;
    float value = g[index];
    return mg ? (w * value) : value;
}

// One partner's OFFDIAG term -- offdiag_emitter's offdiag_term with the two
// PARTNER-AXIS loads routed through ghosted_mirror and nothing else touched.
// MEEP step_generic.cpp:582-583 as stepping._offdiagonal_terms (stepping.py:
// 1214-1220) associates it:
//
//     0.25*((g[i] + w*g[i-sx])*u[i] + (g[i+s] + w*g[(i+s)-sx])*u[i+s])
//
// BOTH GHOSTED DOWN LOADS TAKE THE SAME WEIGHT AND THE SAME REDIRECT: ``down``
// and ``corner`` differ only in the OWN-axis coordinate, so the two hit the
// partner-axis ghost plane together and mg is one predicate for both.
//
// THE UP LEG IS UNWEIGHTED AND UNREDIRECTED. It is the component's OWN axis,
// where the array path serves an exact zero on a fold (see coord_up), so it stays
// the certified ``ghosted`` call character for character.
//
// Everything else is the certified body: the coefficient multiply sits BETWEEN
// the two shifts (u[i] multiplies the pair at its own node and u[i+s] the pair at
// the next node up this component's axis, because MEEP registers the off-diagonal
// entry at the component's Yee site minus half a cell along its own axis --
// anisotropic_averaging.cpp:248-257, `here - shift1`), and 0.25f scales the sum
// last.
__device__ __forceinline__ float folded_offdiag_term(
    const float* g, const float* u, int home, int down, int up, int corner,
    int mg, float w
) {
    float near_pair = g[home] + ghosted_mirror(g, down, mg, w);
    float far_pair = ghosted(g, up) + ghosted_mirror(g, corner, mg, w);
    return 0.25f * ((near_pair * u[home]) + (far_pair * ghosted(u, up)));
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 159->160, 1535-1541->1582-1588, 1723->1770, 1219-1220->1248-1249, 1772-1780->1819-1827, 1781-1783->1828-1830, 1787->1834, 1823-1825->1870-1872, 1824-1825->1871-1872, 1214-1220->1243-1249

# THE SIGNATURE CARRIES ONLY THE LIVE ROW COEFFICIENTS -- the certified emitter's
# choice, for its reason: binding a dead slot to some other volume leaves a
# pointer aimed at an array the kernel must never touch, which a later edit can
# read by accident.
#
# THE THREE GHOST WEIGHTS ARE ALWAYS BOUND, even with no axis folded. They are
# scalars, not pointers, so there is no array to aim wrongly; and the launcher
# validates each to be exactly +1.0f or -1.0f on a folded axis, where a NaN would
# otherwise launch rather than raise.
TEMPLATE = r'''
extern "C" __global__ void update_E_pml_real_folded_offdiag(
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
    float gw_x, float gw_y, float gw_z
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

    // The mirror ghost lane: face 0 of a folded axis, which is the one plane
    // _shift_down weights (stepping.py:1823-1825). DERIVED FROM bc_* AND NOWHERE
    // ELSE, so the index redirect in coord_dn and the weight in ghosted_mirror
    // cannot disagree about which lane is the ghost -- a disagreement would read
    // stored row MIRROR_ROW without the parity, or apply the parity to an
    // ordinary neighbour, and both are a plane of wrong values rather than a
    // crash.
    //
    // A FOLDED AXIS IS NEVER ALSO WALL-MASKED: stepping._mask_metallic_wall_
    // coupling asks is_metallic AND NOT is_mirrored (stepping.py:1253), and
    // coverage.offdiag_wall_mask_flags asks the same question of the same grid,
    // so wm_* is 0 on a folded axis by construction. The launcher refuses the
    // combination rather than relying on that.
    int mg_x = (bc_x == BC_MIRROR) && at_x;
    int mg_y = (bc_y == BC_MIRROR) && at_y;
    int mg_z = (bc_z == BC_MIRROR) && at_z;

    // The three components are independent, and that is read off the loop rather
    // than assumed: the coupling reads D and the inverse permittivity, the tail
    // writes E and f_w_E. Four disjoint sets, so the order the array path runs
    // them in (stepping.py:969-989) is not a data dependence -- which is what
    // makes a NEIGHBOUR read across components safe here.
__SRC_Ex__

__SRC_Ey__

__SRC_Ez__
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1254->1283, 1823-1825->1870-1872, 1253->1282, 969-989->998-1018


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------
#
# The three index expressions per term are ASSEMBLED from a per-axis table rather
# than written out eighteen times, and the ghost lane and weight come from the
# SAME ``partner_axis`` number that picks the index -- so a term cannot weight one
# axis's ghost while shifting another's.

_HOME_COORDS = ("i", "j", "k")
_UP_COORDS = ("ui", "uj", "uk")
_DOWN_COORDS = ("di", "dj", "dk")
_WALL_FLAGS = ("wm_x", "wm_y", "wm_z")
_WALL_PREDICATES = ("at_x", "at_y", "at_z")
_GHOST_LANES = ("mg_x", "mg_y", "mg_z")
_GHOST_WEIGHTS = ("gw_x", "gw_y", "gw_z")
_OWN_AXIS_INDEX = ("i", "j", "k")


def _flat_index(shifted: Dict[int, str]) -> str:
    """``flat(...)`` with the named per-axis substitutions applied."""
    coords = [shifted.get(axis, _HOME_COORDS[axis]) for axis in range(3)]
    return f"flat({coords[0]}, {coords[1]}, {coords[2]}, nyz, nz)"


def _term_lines(component: int, offset: int) -> List[str]:
    """One partner's term for one component, as source lines.

    ``offset`` is 0 for MEEP's ``cycle_direction(dim, d_ec, 1)`` and 1 for
    ``(..., 2)`` (stepping.py:1235-1237). The partner axis, the coefficient slot,
    the partner volume, all three shifted indices AND the ghost lane/weight pair
    come from that one number, so a mispairing would have to be introduced
    deliberately rather than by a typo.
    """
    own_axis = E_TERMS[component][2]
    partner_axis = _coverage.OFFDIAG_TRANSVERSE_PARTNERS[component][offset]
    coefficient = ROW_PARAMETERS[2 * component + offset]
    volume = PARTNER_VOLUMES[partner_axis]
    tag = f"{E_TERMS[component][0]}_{offset}"
    down = _flat_index({partner_axis: _DOWN_COORDS[partner_axis]})
    up = _flat_index({own_axis: _UP_COORDS[own_axis]})
    corner = _flat_index({own_axis: _UP_COORDS[own_axis],
                          partner_axis: _DOWN_COORDS[partner_axis]})
    return [
        f"    // partner {volume}: the pair half a cell DOWN axis {partner_axis},",
        f"    // then the product half a cell UP axis {own_axis} (its own). The",
        f"    // mirror ghost, if axis {partner_axis} carries one, is on the DOWN leg.",
        f"    float term_{tag} = folded_offdiag_term(",
        f"        {volume}, {coefficient}, idx,",
        f"        {down},",
        f"        {up},",
        f"        {corner},",
        f"        {_GHOST_LANES[partner_axis]}, {_GHOST_WEIGHTS[partner_axis]});",
    ]


def _wall_mask_lines(component: int) -> List[str]:
    """``stepping._mask_metallic_wall_coupling`` for one component's total.

    Face 0 of every axis whose Yee shift is 0
    (:data:`coverage.OFFDIAG_WALL_MASK_AXES`), ascending -- the mask's own loop
    order (stepping.py:1279-1283). Unchanged from the certified emitter, because
    the mask itself is unchanged: it already abstains on a fold, by asking the
    grid's DECLARATION rather than the resolved ghost rule.
    """
    total = f"total_{E_TERMS[component][0]}"
    return [f"    {total} = ({_WALL_FLAGS[axis]} && {_WALL_PREDICATES[axis]})"
            f" ? 0.0f : {total};"
            for axis in _coverage.OFFDIAG_WALL_MASK_AXES[component]]


def _component_source(component: int, row_mask: Sequence[int]) -> str:
    """One component's whole block, on the arm its two row slots select.

    Four arms: none, offset-1 only, offset-2 only, both. THE NONE ARM IS THE
    CERTIFIED PLAIN CONSTITUTIVE KERNEL'S COMPONENT BODY -- ``src = D * inv_eps``
    and the same tail -- because ``_offdiagonal_terms`` returns None for a
    component with no surviving row and ``update_E`` then never forms a ``+ 0``
    copy (stepping.py:1222-1227, :1007).
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
    """Validate and canonicalize a row mask -- the certified emitter's, imported.

    Its refusal message names the PLAIN constitutive kernel, which is the family a
    zero-slot run belongs to folded or not, so it is the right message here too.
    """
    return _flat.normalized_row_mask(row_mask)


def folded_offdiag_source(row_mask: Sequence[int]) -> str:
    """The device source specialized on one row mask.

    The BOUNDARY, WALL and GHOST-WEIGHT axes are NOT baked in -- they are runtime
    arguments -- so the row mask is this family's only source axis, exactly as in
    the certified sibling, and the corpus drives two of its 63 values.
    """
    mask = normalized_row_mask(row_mask)
    body = TEMPLATE.replace("__ROW_PARAMETERS__", _row_parameter_lines(mask))
    for component, term in enumerate(E_TERMS):
        body = body.replace(f"__SRC_{term[0]}__",
                            _component_source(component, mask))
    return _shared_prelude() + "\n" + _OWN_PRELUDE + body


def prelude() -> str:
    """The whole device prelude -- shared helpers first, then this family's."""
    return _shared_prelude() + "\n" + _OWN_PRELUDE


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    The certified sibling's arrangement and its reason: 63 sources is too many to
    pin one digest each without burying the record and not too many to hash, so a
    single changed character anywhere in the emitter moves one value. The
    per-source digests of the masks a gate launches belong in that gate's artifact.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for mask in LIVE_ROW_MASKS:
        digest.update(repr(mask).encode("ascii"))
        digest.update(folded_offdiag_source(mask).encode("utf-8"))
    return digest.hexdigest()


def shipped_kernel_names(source: str) -> Any:
    """Every kernel NVRTC could be asked to compile in ``source``, from the text."""
    import re  # noqa: PLC0415 - stdlib, imported at the one call site

    return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))


# ---------------------------------------------------------------------------
# The host-side facts a launch needs
# ---------------------------------------------------------------------------

def folded_offdiag_boundary_codes(grid: Any, pml: Any = None) -> Tuple[int, ...]:
    """The three ``bc_*`` arguments: the resolved ghost rule per axis, as codes.

    ``coverage.real_pml_boundary_kinds`` reproduces ``stepping._boundary_kinds``,
    and this family's table is :data:`coverage.BC_CODES` plus ``mirror`` -- which
    is the ONE difference between this launcher and
    ``offdiag_constitutive_kernels.offdiag_boundary_codes``, whose KeyError on a
    folded axis is the reason the certified predicate could not be widened.

    Every kind the resolution can return that is NOT in :data:`FOLDED_BC_CODES`
    (today: the cylindrical ``axis``) is refused by the predicate, so a KeyError
    here would mean the launcher ran on a configuration the predicate declined. It
    is left to raise rather than defaulted, because a default would turn that into
    a silent wrong ghost rule.

    ``pml`` is accepted and unused, so that every launcher on this track takes the
    same shape.
    """
    return tuple(FOLDED_BC_CODES[kind]
                 for kind in _coverage.real_pml_boundary_kinds(grid))


def mirror_ghost_weights(grid: Any) -> Tuple[float, float, float]:
    """``gw_x``/``gw_y``/``gw_z`` -- the parity the mirror ghost carries per axis.

    ``stepping._offdiagonal_terms`` passes ``component = "D" +
    AXIS_NAMES[partner_axis]`` on ``axis = partner_axis`` (stepping.py:1243-1245),
    and ``_shift_down``'s MIRROR branch weights the ghost by
    ``_symmetry_phase(...)``, which is ``fields.mirror_parity``
    (stepping.py:2457-2469). Every D component has Yee shift 1 on its OWN axis
    (fields.py:214-219), so the weight collapses to ``-phase`` for every partner
    and every axis.

    DERIVED THROUGH ``fields.mirror_parity`` ITSELF rather than by restating the
    collapse, so the two cannot drift; the collapse is asserted by a test over all
    3 axes x 2 phases.

    ``1.0`` on an unfolded axis, where ``mg`` is never 1 and the value is never
    read. ``nan`` where the plane cannot be read: the weight is a RUNTIME argument,
    so an unreadable plane would otherwise LAUNCH rather than raise, and the
    predicate names the NaN by clause.
    """
    try:  # pragma: no cover - the engine is importable wherever this is called
        from ..fields import mirror_parity  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - unreadable means the plan refuses
        mirror_parity = None  # type: ignore[assignment]
    out: List[float] = []
    for axis in range(3):
        try:
            mirrored = bool(grid.is_mirrored(axis))
        except Exception:  # noqa: BLE001
            out.append(float("nan"))
            continue
        if not mirrored:
            out.append(1.0)
            continue
        try:
            phase = grid.mirror_phase(axis)
        except Exception:  # noqa: BLE001
            phase = None
        if phase not in (1, -1) or mirror_parity is None:
            out.append(float("nan"))
            continue
        out.append(float(mirror_parity("D" + "xyz"[axis], axis, int(phase))))
    return (out[0], out[1], out[2])


def folded_offdiag_constitutive_tables(pml: Any) -> dict:
    """The six HALF-INTEGER kps/kms views this sub-step reads.

    The sub-lattice is asked of :func:`coverage.constitutive_sub_lattice`, the same
    function the predicate asks and both certified launchers ask, so no call site
    can pair the E side with the INTEGER tables -- a half-cell error in the
    absorber profile that is converged, smooth and wrong.
    """
    from .constitutive_kernels import real_constitutive_tables  # noqa: PLC0415

    return real_constitutive_tables(pml, _coverage.constitutive_sub_lattice("E"))


def fold_axes(grid: Any) -> Tuple[int, ...]:
    """The folded axes, as a tuple. One reader, so a caller cannot ask differently."""
    return tuple(axis for axis in range(3) if bool(grid.is_mirrored(axis)))


# ---------------------------------------------------------------------------
# THE COVERAGE PREDICATE
# ---------------------------------------------------------------------------

#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, and nothing more.
#:
#: ``host`` is the only thing that makes any number here a device verdict; a
#: ``host: None`` record means the predicate returning True licenses a MEASUREMENT
#: and nothing else. Filled 2026-08-20 by
#: ``parity/meep_gpu/gate_cuda_folded_offdiag_kernel.py`` on the GPU host's RTX A6000,
#: GPU index 7 (verified physically empty by UUID before launch), under BOTH
#: float32 subnormal policies, each in its own process with its own CuPy cache
#: directory because CuPy's disk-cache key is computed above the strip seam.
#:
#: STILL NOT DISPATCHED. No module in ``meep_gpu/`` imports ``cuda_kernels``, so
#: nothing reaches this kernel from a production step; what the record licenses is
#: a coverage claim and a future planner branch, not a run.
FOLDED_OFFDIAG_ADMISSION: dict = {
    "kernel": KERNEL_NAME,
    "gate": "parity/meep_gpu/gate_cuda_folded_offdiag_kernel.py",
    "oracle": ("stepping.update_E on real Grid/Fields/PML with rows installed "
               "through Fields.set_epsilon_volumes, compared as uint32 words"),
    "specified_by": "coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION",
    "recorded_utc": "2026-08-21T23:04:40Z",
    "host": "the GPU host, NVIDIA RTX A6000, GPU index 7",
    "artifacts": ("parity/meep_gpu/results/cuda_folded_offdiag_kernel_2026-08-21_rename/"
                  "keep/gate.json",
                  "parity/meep_gpu/results/cuda_folded_offdiag_kernel_2026-08-21_rename/"
                  "flush/gate.json"),
    #: HOW THIS RECORD IS BOUND TO THOSE RUNS, and why it is NOT an artifact hash.
    #: A digest of ``gate.json`` inside this module is circular: the artifact's own
    #: ``source_sha256.txt`` records THIS FILE's bytes as of the run, so writing
    #: the artifact's digest here changes this file and invalidates the manifest
    #: that was supposed to bind it. The non-circular binding is the EMITTER
    #: CORPUS DIGEST below -- one sha256 over all 63 sources this family can emit,
    #: carried in both artifacts and asserted equal to the shipped emitter by
    #: ``test_folded_offdiag.py``. That ties the verdict to the exact DEVICE CODE,
    #: which is what the verdict is about; a docstring edit moves the file hash
    #: and not the arithmetic, and this record is a docstring edit.
    "bound_by": "emitter_corpus_digest, carried in both artifacts",
    "gate_sha256_at_run":
        "b9a8b539c38fe2d79e4a4a95f56edea2f83a41240a08039d29a9504615e62e16",
    "post_gate_record_edits": (
        "this FOLDED_OFFDIAG_ADMISSION block was filled in after the run it "
        "describes; it touches no device string and no predicate clause, and the "
        "emitter corpus digest is unchanged across the edit",),
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    #: PER POLICY. The two produced identical case counts and identical verdicts,
    #: which is why one copy carries both; each ran in its own process.
    "cases_scored": 544,
    "cases_refused_as_vacuous": 0,
    "folded_single_launch_identical": 240,
    "folded_cases": 240,
    "folded_at_sixty_launches_identical": 40,
    "folded_at_sixty_launches": 40,
    "unfolded_reduction_identical": 32,
    "unfolded_reduction_cases": 32,
    #: The two arms the mirror rule can reach, counted separately, because a total
    #: alone cannot show that the arm this family exists for was exercised.
    "cases_where_a_fold_is_a_live_partner_axis": 168,
    "cases_where_a_fold_is_a_live_own_axis": 168,
    #: THE PREDICTION THE CURL PAIR'S SPLIT DOES NOT TRANSFER (module docstring
    #: FACT 3), measured: ``update_E`` has no ownership mask and no reflect row,
    #: so ONE ``BC_MIRROR`` code serves both declared terminations.
    "folded_periodic_identical": "144/144",
    "folded_metallic_identical": "96/96",
    "simultaneous_planes_identical": {"1": "176/176", "2": "32/32", "3": "32/32"},
    #: ``--fmad=false`` is CORRECTNESS here, measured rather than inherited: the
    #: unguarded control is 0/136 identical at the inexact courant.
    "guard_control_identical_at_inexact_courant": "0/136",
    "mutation_legs_as_required": "23/23",
    #: The two that make the battery two-sided rather than a list of catches.
    "null_controls_confirmed": ("distribute_the_quarter",
                                "commute_the_near_product"),
    #: The shipped CERTIFIED kernel's only available answer for a folded axis,
    #: armed as a host defect against THIS kernel: caught wherever the mirror ghost
    #: is reachable, a NULL where it is not. That is
    #: ``coverage.offdiag_fold_roles``' rule re-measured on a device.
    "shipped_kernels_answer_armed_as_a_defect":
        "hand_the_fold_the_metallic_code: 42/68 caught, as_required",
    "release_verdict_shown_to_flip_against_a_planted_defect": True,
    "subnormal_band_is_non_vacuous": True,
    "certified": True,
    #: WHAT THE RECORD DOES NOT SAY. It is a verdict about the EMITTER at the
    #: corpus digest below, exercised on four of its 63 row masks. A mask outside
    #: those is covered by the emitter's own tests and by that digest, not by a
    #: byte gate. No throughput claim is made or possible: the box was shared.
    "row_masks_exercised": ((1, 1, 1, 1, 1, 1), (1, 0, 0, 1, 0, 0),
                            (1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)),
    "emitter_corpus_digest":
        "9cb8ccc8f8bceb008446a811731bf547ef32bcb4c43544c9ada59fa26a92e2bd",
    #: WHAT IT IS WORTH, recomputed 2026-08-20 on ONE tree with the union analyzer
    #: (positional path) -- the family present and absent on the SAME census, so
    #: the delta is attributable to this leg rather than to a tree that moved
    #: between two rounds.
    "corpus": {
        "census": ("parity/meep_gpu/results/"
                   "cuda_predicate_coverage_2026-08-20_folded_offdiag"),
        "rows": 186, "slots": 759,
        "union_without_this_family": 664,
        "union_with_this_family": 683,
        "slots_gained": 19,
        "rows_covered_at_every_sub_step_gained": 19,
        "overlaps_introduced": 0,
        #: Every one of the 19 was refused FIRST by
        #: ``covers_real_pml_offdiag_constitutive``'s mirror clause, drives row
        #: mask (1, 0, 0, 1, 0, 0), and is folded on X and/or Y -- which is
        #: structural rather than a sampling accident, as
        #: ``FOLDED_OFFDIAG_ROW_MASK_ADMISSION["corpus"]`` already recorded.
        "predicted_by_the_specification": 19,
        #: The twentieth live-partner row, ``examples/absorbed_power_density.py``,
        #: is refused by the dispersion clause as well and is NOT gained.
        "slots_still_needing_a_fused_dispersive_arm": 1,
    },
}


def _fold_reasons(facts: dict, grid: Any) -> List[str]:
    """Everything this family needs of a fold, as accumulated reasons.

    THE ONE PLACE THE FOLD IS CLASSIFIED. The certified predicate refuses it twice
    (``has_symmetry`` and per axis); this one has to admit it, so the two
    accessors are still both read -- an object that answers them inconsistently is
    refused rather than dispatched -- and then the axis is checked for the three
    things the kernel and its launcher actually need:

    a. MORE THAN ``MIRROR_ROW`` STORED CELLS. ``coord_dn``'s mirror arm returns
       stored row 2 unconditionally at face 0, and ``stepping._mirror_source``
       (:1536-1541) RAISES below that many cells. A thinner axis would have the
       kernel read a neighbouring plane where the array path refuses to run.
    b. A READABLE PLANE PHASE, exactly +1 or -1. The ghost weight is a runtime
       float argument, so an unreadable plane launches a NaN rather than raising.
       Checked here by NAME and again on the launched value itself.
    c. THE TWO ROUTES TO THE TERMINATION AGREEING
       (``coverage._fold_termination_problem``). This kernel serves both
       terminations with one code -- FACT 3 of the module docstring -- so the
       cross-check is not load-bearing for the arithmetic; it is load-bearing for
       TRUST: a grid whose ``stored_cells > owned_cells`` and whose
       ``is_metallic`` disagree has one of the two drifted, and neither can then
       be believed about the boundary kinds this kernel indexes on.
    """
    reasons: List[str] = []
    mirrored = facts["mirrored"]
    if facts["has_symmetry"] and not any(mirrored):
        reasons.append(
            "grid.has_symmetry() is True but no axis reports is_mirrored: the "
            "two accessors disagree and neither can be trusted")
    if any(mirrored) and not facts["has_symmetry"]:
        reasons.append(
            "an axis reports is_mirrored but grid.has_symmetry() is False: the "
            "two accessors disagree and neither can be trusted")
    weights = mirror_ghost_weights(grid)
    for axis in range(3):
        if not mirrored[axis]:
            continue
        if int(facts["shape"][axis]) <= MIRROR_SOURCE_INDEX:
            reasons.append(
                f"axis {axis} is folded with {facts['shape'][axis]} stored "
                f"cells; the mirror ghost images stored row "
                f"{MIRROR_SOURCE_INDEX} and stepping._mirror_source raises "
                f"below that")
        problem = _coverage._fold_termination_problem(facts, axis)
        if problem is not None:
            reasons.append(problem)
        weight = weights[axis]
        if weight != weight or weight not in (1.0, -1.0):
            reasons.append(
                f"axis {axis} mirror ghost weight is {weight!r}, not +1.0 or "
                f"-1.0; the weight is a RUNTIME argument and an unreadable "
                f"plane would launch rather than raise")
        if facts["metallic"][axis] and _coverage._offdiag_wall_flags_from(
                facts)[axis]:  # pragma: no cover - unreachable by construction
            reasons.append(
                f"axis {axis} is folded AND wall-masked; "
                f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                f"(stepping.py:1253)")  # stepping.py live lines for the frozen device-text citation(s) in this string: 1253->1282
    return reasons


def covers_folded_offdiag_constitutive(fields: Any, pml: Any, grid: Any) -> tuple:
    """Whether ``update_E_pml_real_folded_offdiag`` may serve this run.

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal -- the
    convention every predicate in ``coverage.py`` uses, and the reason this chain
    is WRITTEN OUT rather than delegated to
    ``covers_real_pml_offdiag_constitutive`` with the fold clauses stripped: that
    function SHORT-CIRCUITS, so "refused for the fold and nothing else" is not a
    question its return value can answer, and delegating would admit a
    configuration whose ``stores_E``, array-layout or coefficient-table clause was
    never reached. ``test_folded_offdiag.py`` pins the two chains equal on every
    axis except the fold one, so the duplication cannot drift silently.

    ZERO FOLDED AXES IS ADMITTED HERE, and that is deliberate rather than an
    oversight: with every axis PERIODIC or METALLIC the emitted body computes the
    certified family's arithmetic exactly, and the gate MEASURES that reduction
    instead of claiming it. Which of the two a planner would launch is an
    integration decision, not a coverage one, and routing by predicate order would
    make the numerical method a function of composer order. THE CENSUS ASKS
    :func:`covers_folded_offdiag_composition`, which requires a real fold, so the
    two families partition the corpus rather than overlapping on it.

    THE DISJOINTNESS SEAM, both halves:

    * against ``covers_real_pml_offdiag_constitutive``: that predicate's two fold
      refusals STAY, and :func:`covers_folded_offdiag_composition` requires a fold;
    * against ``covers_real_pml_constitutive(side='E')``: that predicate refuses
      every off-diagonal run, and clause (a) below REQUIRES a surviving row slot.
    """
    xp, backend = _coverage._backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if getattr(fields, "force_complex_fields", False):
        return False, ("complex64 storage: the recurrence is the same but the "
                       "storage is not")
    if not (pml is not None and getattr(pml, "is_active", False)):
        # Without an absorber ``update_E`` takes the plain assignment
        # (stepping.py:1022) -- no ``f_w``, no recurrence. A different sub-step.
        return False, "no active PML layer"
    facts, unreadable = _coverage._grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    if facts["cylindrical"]:
        return False, ("cylindrical (Dcyl): the r axis has its own ghost rule "
                       "(the r_to_minus_r image, stepping.py:1877-1888) and its "
                       "own axial extent")
    for axis in range(3):
        if facts["axis"][axis]:
            return False, f"axis {axis} is the cylindrical r = 0 axis"
    fold_problems = _fold_reasons(facts, grid)
    if fold_problems:
        return False, fold_problems[0]
    for axis, kind in enumerate(_coverage._boundary_kinds_from(facts)):
        # Stated positively: the three ghost rules the emitter writes, named.
        if kind not in FOLDED_BC_CODES:
            return False, (f"axis {axis} resolves to boundary {kind!r}, which "
                           f"has no kernel")
    if facts["has_bloch"]:
        return False, ("nonzero Bloch k: the wrapped plane carries a phase real "
                       "storage cannot hold")
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, ("special_kz (grid.beta != 0): extra out-of-plane coupling "
                       "terms")
    # Both maps by name, never ``has_nonlinearity`` -- that property reads
    # ``_chi2_components`` alone (fields.py:966-967), so a chi3-only run would
    # pass a predicate that asked it. MEEP's most general case scales the WHOLE
    # row product (coupling included) by the Pade factor, step_generic.cpp:590-601
    # as ``stepping._nonlinear_constitutive`` (:1066-1073) transcribes it.
    if (getattr(fields, "_chi2_components", None)
            or getattr(fields, "_chi3_components", None)):
        return False, ("instantaneous chi2/chi3: the Pade factor scales the whole "
                       "row product, coupling included")
    if getattr(fields, "polarizations", None):
        # With poles the sources are per-component ``D - sum P`` scratch buffers
        # (fields.py:1107-1138) and the coupling reads THOSE, not the D primaries
        # the launcher binds.
        return False, ("dispersion: update_E's source is (D - sum P), not D, "
                       "and update_P closes the step")
    # (a) INVERTED. Counted over the six slots, deliberately not the bare
    #     ``has_offdiagonal_epsilon`` flag: a row planted past the installer under
    #     a diagonal key sets the flag with every slot dead, and the emitter
    #     refuses an all-dead mask by raising.
    rows = _coverage.offdiag_row_volumes(fields)
    if not any(volume is not None for volume in rows):
        return False, ("no off-diagonal chi1inv row survived installation: that "
                       "configuration is covers_real_pml_constitutive(side='E')'s "
                       "and this predicate must not overlap it")
    if not getattr(fields, "stores_E", False):
        return False, "E is recomputed from D rather than stored"

    spec = _coverage.CONSTITUTIVE_SIDES["E"]
    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        return False, f"{cells} cells exceeds the kernel's int32 index range"
    outputs = tuple(spec["targets"]) + tuple(spec["aux"])
    for name in outputs + tuple(spec["sources"]):
        problem = _coverage._array_problem(name, getattr(fields, name, None),
                                           xp, shape)
        if problem is not None:
            return False, problem
    reader = getattr(fields, "inverse_epsilon_for", None)
    if not callable(reader):
        return False, "fields does not expose inverse_epsilon_for"
    for component in ("Ex", "Ey", "Ez"):
        try:
            volume = reader(component)
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal
            return False, f"inverse_epsilon_for({component!r}) raised {exc!r}"
        if volume is None:
            return False, f"inverse_epsilon_for({component!r}) is None"
        if not getattr(volume, "shape", ()):
            return False, (f"inverse_epsilon_for({component!r}) is a scalar, not "
                           f"a volume")
        problem = _coverage._array_problem(
            f"inverse_epsilon_for({component!r})", volume, xp, shape)
        if problem is not None:
            return False, problem
    # (b) and (c): the surviving rows, readable, well-formed, and not an output.
    #     The alias clause is stronger here than for the plain constitutive pair
    #     for the certified sibling's reason, and the fold sharpens it: the
    #     coupling re-reads the partner volumes at neighbour offsets, on a folded
    #     axis including INTERIOR stored row MIRROR_ROW rather than a face.
    output_addresses: Dict[int, str] = {}
    for name in outputs:
        address = _coverage._base_address(getattr(fields, name, None))
        if address is not None:
            output_addresses.setdefault(address, name)
    for index, volume in enumerate(rows):
        if volume is None:
            continue
        row, partner = _coverage.OFFDIAG_ROW_SLOTS[index]
        label = f"chi1inv_offdiagonal[{row!r}][{partner!r}]"
        if not getattr(volume, "shape", ()):
            return False, f"{label} is a scalar, not a volume"
        problem = _coverage._array_problem(label, volume, xp, shape)
        if problem is not None:
            return False, problem
        address = _coverage._base_address(volume)
        if address is not None and address in output_addresses:
            return False, (f"{label} aliases output {output_addresses[address]}: "
                           f"the coupling re-reads the partner volumes at "
                           f"neighbour offsets -- on a folded axis including "
                           f"interior stored row {MIRROR_SOURCE_INDEX} -- while "
                           f"the outputs are written, so the answer would depend "
                           f"on block schedule")
    # HALF-INTEGER ONLY, the E side's own sub-lattice (stepping.py:1015). ON A
    # FOLDED AXIS THE VECTOR IS BUILT AT THE FOLDED STORED EXTENT, which is what
    # ``_coefficient_vector_problem`` checks by SHAPE rather than by size -- the
    # clause that stops "the vectors are the right length" from being an
    # assumption on exactly the axis the fold moved.
    suffix = "_h" if spec["half_integer"] else ""
    for axis, name in enumerate(("x", "y", "z")):
        for label in ("kps", "kms"):
            attribute = f"{label}_{name}{suffix}"
            problem = _coverage._coefficient_vector_problem(
                attribute, getattr(pml, attribute, None), xp, axis, shape)
            if problem is not None:
                return False, problem
    return True, "covered"


def covers_folded_offdiag_composition(fields: Any, pml: Any, grid: Any) -> tuple:
    """The narrower verdict, where an ACTUAL FOLD IS MANDATORY.

    :func:`covers_folded_offdiag_constitutive` admits an unfolded grid so that the
    gate can prove reduction to the certified family. That is not a routing rule:
    selecting between two valid products by branch order would make the numerical
    method depend on composer order. THIS is the verdict a planner -- and the
    coverage census -- asks, and it is what keeps this family disjoint from
    ``cuda_offdiag`` on every corpus row.
    """
    covered, reason = covers_folded_offdiag_constitutive(fields, pml, grid)
    if not covered:
        return False, reason
    try:
        folded = fold_axes(grid)
    except Exception as exc:  # noqa: BLE001
        return False, f"the grid could not say whether it is folded: {exc!r}"
    if not folded:
        return False, ("no mirror plane is active: an unfolded off-diagonal "
                       "update_E is covers_real_pml_offdiag_constitutive's, and "
                       "admitting it here would overlap the two families")
    return True, "covered"


# ---------------------------------------------------------------------------
# THE LAUNCHER
# ---------------------------------------------------------------------------

#: The emitted sources a gate has replaced, keyed by row mask -- the mutation
#: seam. Paired with :func:`clear_kernel_cache`: the compile memo keys on the
#: SOURCE, so a mutated body is a miss and reaches NVRTC without the clear, but
#: the clear is what makes "how many compiles came from the mutated bytes" exact.
_SOURCE_OVERRIDES: Dict[Tuple[int, ...], str] = {}


def kernel_source(row_mask: Sequence[int]) -> str:
    """The device text a launch would compile for this mask -- override or emit."""
    mask = normalized_row_mask(row_mask)
    return _SOURCE_OVERRIDES.get(mask) or folded_offdiag_source(mask)


def set_kernel_source(row_mask: Sequence[int], source: Optional[str]) -> None:
    """Replace (or, with ``None``, restore) one mask's device text.

    THE GATE'S DOOR, and it is load-bearing: a mutation harness that could not
    route its own bytes through the launcher would launch the shipped kernel and
    report a pass for a defect it never introduced.
    """
    mask = normalized_row_mask(row_mask)
    if source is None:
        _SOURCE_OVERRIDES.pop(mask, None)
    else:
        _SOURCE_OVERRIDES[mask] = source


def clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return _clear_cache()


def _get_kernel(row_mask: Sequence[int]):
    """Compile on first use, memoized on (name, options, policy, source).

    THE ROW MASK REACHES THE KEY THROUGH THE SOURCE, which is already in it, so
    two specializations cannot be served each other's binary -- and neither can a
    mutation harness that rewrote the emitter's output be served the unmutated
    one. ``cupy`` is imported HERE, never at scope.
    """
    import cupy as cp  # noqa: PLC0415 - device-only, in a laptop-importable module

    code = kernel_source(row_mask)
    key = _kernel_cache_key(KERNEL_NAME, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _require_no_aliasing(fields: Any, rows: Sequence[Any]) -> None:
    """Outputs pairwise distinct and disjoint from every input volume.

    The certified sibling's check, and the fold makes it stronger rather than
    weaker: the coupling re-reads the partner volumes at neighbour offsets, and on
    a folded axis one of those offsets is stored row 2 -- deep INSIDE the volume
    rather than at a face -- while the outputs are being written. An alias makes
    the answer depend on block schedule: wrong differently on each run, which no
    single comparison catches reliably.

    Equality of BASE addresses only: two overlapping views with different bases
    pass unseen, the accepted limitation every plan on this track shares.
    """
    outputs: Dict[int, str] = {}
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        address = _coverage._base_address(getattr(fields, name, None))
        if address is None:
            raise ValueError(
                f"{name} exposes no readable base address; an unverifiable "
                f"output is not accepted")
        if address in outputs:
            raise ValueError(f"{name} aliases {outputs[address]}; the outputs "
                             f"must be distinct arrays")
        outputs[address] = name
    inputs = [fields.Dx, fields.Dy, fields.Dz,
              fields.inverse_epsilon_for("Ex"),
              fields.inverse_epsilon_for("Ey"),
              fields.inverse_epsilon_for("Ez")]
    inputs.extend(rows)
    for volume in inputs:
        address = _coverage._base_address(volume)
        if address is not None and address in outputs:
            raise ValueError(
                f"an input volume aliases {outputs[address]}: the coupling "
                f"re-reads the partner volumes at neighbour offsets -- on a "
                f"folded axis including interior stored row "
                f"{MIRROR_SOURCE_INDEX} -- while the outputs are written, so "
                f"the answer would depend on block schedule")


def _validate_launch_arguments(shape: Sequence[int], codes: Sequence[int],
                               walls: Sequence[int],
                               weights: Sequence[float]) -> None:
    """Everything a launch would otherwise get silently wrong.

    Each of these is a WRONG ANSWER rather than a crash if it is not checked here:
    a NaN weight multiplies one plane into NaN, a folded axis that is also
    wall-masked zeroes a plane MEEP steps, and a folded axis with too few cells
    reads a neighbouring plane. The predicate names all three, and a gate can
    reach the launcher without the predicate, which is why they are checked twice.
    """
    if len(codes) != 3 or any(int(code) not in FOLDED_BC_CODES.values()
                              for code in codes):
        raise ValueError(
            f"boundary codes must be three of "
            f"{sorted(set(FOLDED_BC_CODES.values()))}, got {tuple(codes)!r}")
    if len(walls) != 3 or any(int(flag) not in (0, 1) for flag in walls):
        raise ValueError(f"wall flags must be three of {{0, 1}}, got "
                         f"{tuple(walls)!r}")
    if len(weights) != 3:
        raise ValueError(f"ghost weights must be three floats, got "
                         f"{tuple(weights)!r}")
    for axis in range(3):
        if int(codes[axis]) != BC_MIRROR_CODE:
            continue
        if int(walls[axis]):
            raise ValueError(
                f"axis {axis} is folded AND wall-masked; "
                f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                f"(stepping.py:1282) and zeroing the fold plane the way the "
                f"metallic rule does costs 2.0e-02 (stepping.py:1269-1274)")
        if float(weights[axis]) not in (1.0, -1.0):
            raise ValueError(
                f"axis {axis} is folded with ghost weight {weights[axis]!r}; the "
                f"mirror ghost carries mirror_parity('D'+axis, axis, phase) == "
                f"-phase, exactly +1.0 or -1.0 (fields.py:117, "
                f"stepping.py:1824-1825)")  # stepping.py live lines for the frozen device-text citation(s) in this string: 1824-1825->1871-1872
        if int(shape[axis]) <= MIRROR_SOURCE_INDEX:
            raise ValueError(
                f"axis {axis} is folded with {shape[axis]} stored cells; the "
                f"mirror ghost images stored row {MIRROR_SOURCE_INDEX}")


def _launch(kernel, fields: Any, rows: Sequence[Any], tables: dict,
            codes: Sequence[int], walls: Sequence[int],
            weights: Sequence[float]) -> None:
    """One launch. ``rows`` are the LIVE coefficient volumes, in slot order."""
    import numpy as np  # noqa: PLC0415 - only the scalar types are needed

    nx, ny, nz = fields.Ex.shape
    blocks = ((nx * ny * nz + _FOLDED_OFFDIAG_THREADS - 1)
              // _FOLDED_OFFDIAG_THREADS)
    arguments = [
        fields.Ex, fields.Ey, fields.Ez,
        fields.f_w_Ex, fields.f_w_Ey, fields.f_w_Ez,
        fields.Dx, fields.Dy, fields.Dz,
        fields.inverse_epsilon_for("Ex"),
        fields.inverse_epsilon_for("Ey"),
        fields.inverse_epsilon_for("Ez"),
    ]
    arguments.extend(rows)
    arguments.extend([
        np.int32(nx), np.int32(ny), np.int32(nz),
        tables["kps_x"], tables["kms_x"],
        tables["kps_y"], tables["kms_y"],
        tables["kps_z"], tables["kms_z"],
        np.int32(codes[0]), np.int32(codes[1]), np.int32(codes[2]),
        np.int32(walls[0]), np.int32(walls[1]), np.int32(walls[2]),
        np.float32(weights[0]), np.float32(weights[1]), np.float32(weights[2]),
    ])
    kernel((blocks,), (_FOLDED_OFFDIAG_THREADS,), tuple(arguments))


def update_E_folded_offdiag_fused_pml_real(fields: Any, pml: Any = None, *,
                                           tables: dict = None, codes=None,
                                           walls=None, weights=None):
    """``stepping.update_E`` with off-diagonal rows on a folded grid, in one launch.

    SUPPLY ``pml`` AND EVERY DERIVED ARGUMENT IS DECIDED HERE -- the half-integer
    tables through :func:`coverage.constitutive_sub_lattice`, the boundary codes
    through ``real_pml_boundary_kinds`` and :data:`FOLDED_BC_CODES`, the wall flags
    through the grid's own declaration, the ghost weights through
    ``fields.mirror_parity``. That is what makes "the launcher and the predicate
    cannot disagree" an enforced property of this function rather than a
    convention a caller may keep.

    ``tables``/``codes``/``walls``/``weights`` ARE THE GATE'S DOOR and stay,
    keyword-only: a byte gate feeds synthetic tables no ``PML`` produces, and its
    mutations feed DELIBERATELY WRONG ones -- a mis-paired sub-lattice, a dropped
    wall flag, a fold handed the metallic code, a flipped parity. A launcher that
    could not take its own could not arm any of them. Passing ``pml`` together
    with an override, or neither, is refused: a caller with two answers to a
    one-answer question has a bug either way.
    """
    overrides = (tables, codes, walls, weights)
    supplied = [value is not None for value in overrides]
    if (pml is not None) and any(supplied):
        raise ValueError(
            "pml and an override were both supplied; the derived answer and the "
            "supplied one cannot both be the one this launch used")
    if pml is None and not all(supplied):
        raise ValueError(
            "pass exactly one of pml (tables, codes, walls and weights are all "
            "derived from it and the grid) or ALL FOUR of tables, codes, walls "
            "and weights (the gate supplies its own, including deliberately "
            "wrong ones); got "
            f"tables={'set' if tables is not None else 'None'}, "
            f"codes={'set' if codes is not None else 'None'}, "
            f"walls={'set' if walls is not None else 'None'}, "
            f"weights={'set' if weights is not None else 'None'}")
    if pml is not None:
        tables = folded_offdiag_constitutive_tables(pml)
        codes = folded_offdiag_boundary_codes(fields.grid, pml)
        walls = _coverage.offdiag_wall_mask_flags(fields.grid)
        weights = mirror_ghost_weights(fields.grid)

    mask = normalized_row_mask(_coverage.offdiag_row_mask(fields))
    rows = [volume for volume in _coverage.offdiag_row_volumes(fields)
            if volume is not None]
    _validate_launch_arguments(tuple(fields.Ex.shape), codes, walls, weights)
    _require_no_aliasing(fields, rows)
    return _launch(_get_kernel(mask), fields, rows, tables, codes, walls, weights)
