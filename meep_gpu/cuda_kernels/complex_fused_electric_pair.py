"""The COMPLEX-storage hand-CUDA fused pair: ``step_D`` -> ``zero_metal_D`` -> ``update_E``.

THE FIRST COMPLEX PRODUCT ON THE ELECTRIC SEAM, and it takes the board's largest
unserved Cartesian D cell: ``D_to_E (cuda_complex/complex, cuda_complex/complex)``,
16 seam-instances of demand of which exactly ONE clears the source seam on the driver
fact alone (``parity/meep_gpu/results/fusion_matrix_cuda_2026-08-30_electric/board.log``,
"THE BOARD"). The other 15 carry an ELECTRIC deposit inside the seam, so what this
module is worth is the BRACKET rather than the launch count: without a product on the
cell, the board prices every one of those 15 on the driver fact and they can never be
reached, whatever the fitness verdict says.

=============================================================================
WHY IT IS A SEPARATE MODULE AND NOT AN ARM OF THE COMPLEX MAGNETIC PAIR
=============================================================================

:mod:`.complex_fused_magnetic_pair` welds ``step_B`` to ``update_H``; this welds
``step_D`` to ``update_E``. They share an emitter and not a byte of spliced text: the
D curl takes ``cshift_dn`` and the CONJUGATED Bloch factor (``bloch_phase_arguments(grid,
True)``, ``stepping._shift_down``), the E constitutive multiplies by three
inverse-permittivity volumes the H side does not have, and the two sub-lattice pairings
are MIRROR IMAGES of each other. One module answering for both would need a predicate
that is the INTERSECTION of two clause sets, which is the reason
``covers_real_pml_complex_curl`` takes a sub-step argument in the first place.

The two occupy DIFFERENT SEAMS, so they can never make one seam ambiguous:
``fused_pairs.install_fused_pairs`` decides one seam at a time and skips every product
whose ``curl_slot`` is not the one it is deciding.

WHAT IT IS DISJOINT FROM, ON ITS OWN SEAM. ``fused_electric_pair``'s predicate opens
with ``covers_real_pml_curl``, which refuses complex64 storage BY NAME; this one opens
with :func:`covers_real_pml_complex_curl`, which refuses a run that is neither
``force_complex_fields`` nor Bloch-phased and refuses a Dcyl grid. A pair of predicates
that overlapped would leave the seam UNFUSED naming both, so disjointness is what buys
the slots rather than a tidiness.

=============================================================================
WHAT ONE LAUNCH PERFORMS, AND WHAT IT DOES NOT
=============================================================================

:data:`REPLACES` is ``step_D`` (driver.py:3302) -> ``zero_metal_D`` (:3310) ->
``update_E`` (:3313), declared rather than inferred from the two slots.

THE TWO MIRROR FILLS ARE ABSENT BY REFUSAL, not by omission: ``fill_symmetry_bc_D``
(driver.py:3309) and ``fill_folded_far_ghosts_D`` (:3311) do nothing at all on an
unfolded grid, and :func:`covers_complex_fused_electric_pair` refuses every folded one
-- twice over, once through the complex curl half's own clause ("mirror symmetry:
different ghost rule, a parity mask and two fill passes",
``coverage._complex_grid_refusal``) and once in this file, so the guarantee that makes
:data:`REPLACES` honest does not rest on a clause another module could widen. The
ownership inversion ``fused_magnetic_pair`` uses to carry its two fills is not ported
here: there is no in-seam fill to carry on an admitted row, and porting the machinery
anyway would be untested device text on every one of them.

``zero_metal_D`` IS carried, and IT IS THE OFF-DIAGONAL TABLE -- two components per
walled axis, the complement of the B side's diagonal. That is the single most likely
slip in porting from the magnetic sibling: ``fields.IYEE_SHIFTS`` gives Dx (1,0,0),
Dy (0,1,0), Dz (0,0,1), so a D component has Yee shift 0 on the OTHER TWO axes and sits
on both of their walls. A pair that reused the B family's diagonal would clear one
wrong component and leave two right ones standing, on every walled run, and it is armed
as a mutation rather than left to a reading.

=============================================================================
THE ALIASING HAZARD, AND THE ONE ANSWER TO IT
=============================================================================

The curl half writes D in place and the constitutive half reads it. Binding that one
allocation twice -- once writable for the curl, once ``const __restrict__`` for the
constitutive source -- is undefined behaviour that NVRTC miscompiles without a
diagnostic. So D appears EXACTLY ONCE in :data:`_SIGNATURE` and the constitutive half
has no flux-density source pointers at all: its three ``cf_load(g*, idx)`` loads are
replaced by the three registers the curl already holds.

That substitution is also the whole performance claim, and it is exact rather than
approximate: the certified ``update_E_pml_complex_bloch`` opens each component with a
reload of the word pair ``step_D_pml_complex_bloch`` just stored, and a float32 stored
to global and reloaded is the identity on the bits.

THE THREE ``inv_eps`` POINTERS ARE DELIBERATELY NOT ``__restrict__``, and they are
lifted from the certified ``update_E`` rather than decided here: an isotropic run hands
one allocation three times (``fields.py:1323-1325``), and ``restrict`` on mutually
aliasing arguments is a promise the caller cannot keep. They are read-only, so nothing
is lost but the promise.

=============================================================================
THE TWO NAME COLLISIONS THE SPLICE MUST RESOLVE, AND WHY THEY MATTER
=============================================================================

The two certified bodies were emitted from ONE template family, so they use the same
local names for different things. Spliced verbatim into one scope:

* ``f0/f1/f2`` is D in the curl body and E in the constitutive body, and ``g0/g1/g2``
  is H in the curl body and D in the constitutive one. A collision here is a kernel
  that does not compile, which is the harmless half.
* ``kms_x/kms_y/kms_z`` is the INTEGER sub-lattice for the D curl
  (``stepping._curl_coefficients``, :2418 with ``complex_emitter.HALF_INTEGER['step_D']
  = False``) and the HALF-INTEGER one for ``update_E`` (``HALF_INTEGER['E'] = True``,
  stepping.py:1015). THE PAIRING IS THE MIRROR IMAGE OF THE B/H SEAM'S. Those two
  compile perfectly and differ by half a cell in the absorber profile -- converged,
  smooth and entirely wrong. This is the collision that matters, and it is why the
  constitutive group is renamed ``kms_half_*`` in the signature and in the spliced
  body both.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch. This
module ships a predicate, an emitter and a launcher; ``fused_pairs`` can plan it
opt-in (``arms.plan_step(..., fuse=True)``), and no shipped code path launches it.
"""

from __future__ import annotations

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

# THE CERTIFIED TEXT. ``complex_emitter`` imports nothing and runs at the merge bar,
# so this import is unconditional and the emitter below works on a laptop.
try:
    from . import complex_emitter
    from .coverage import (covers_real_pml_complex_constitutive,
                           covers_real_pml_complex_curl)
    from .complex_pml_kernels import (bloch_phase_arguments,
                                      complex_boundary_codes,
                                      complex_constitutive_tables,
                                      complex_curl_tables, word_view)
    from .in_seam_coverage import zero_metal_axes
    from . import compile_cache
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    def _load(stem):
        here = _os.path.dirname(_os.path.abspath(__file__))
        spec = _importlib_util.spec_from_file_location(
            f"cuda_kernels_{stem}", _os.path.join(here, f"{stem}.py"))
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    complex_emitter = _load("complex_emitter")
    _coverage = _load("coverage")
    covers_real_pml_complex_constitutive = _coverage.covers_real_pml_complex_constitutive
    covers_real_pml_complex_curl = _coverage.covers_real_pml_complex_curl
    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")
    bloch_phase_arguments = complex_boundary_codes = None
    complex_constitutive_tables = complex_curl_tables = word_view = None

# The source-seam clause, shared with every other fused-pair predicate on all three
# tracks. Imported rather than re-spelled: writing it out per product produced a real
# bug once (a Metal pair asked about the wrong seam and would have refused a magnetic
# source while ADMITTING an electric one).
try:
    from .. import deposit_repair as _deposit_repair
except ImportError:  # loaded by path, outside the package
    import importlib.util as _importlib_util
    import os as _os

    _spec = _importlib_util.spec_from_file_location(
        "meep_gpu_deposit_repair",
        _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "deposit_repair.py"))
    _deposit_repair = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_deposit_repair)


# =============================================================================
# THE PARTITION — EMPTY ON THE CERTIFIED SIDE UNTIL A GATE MOVES IT
# =============================================================================
#
# Both are PLAIN assignments with NO type annotation, for the reason every sibling in
# this directory spells them that way: the partition readers walk the syntax tree
# WITHOUT importing the module (they have to — these modules import cupy) and an
# annotated assignment is an ``ast.AnnAssign`` those readers do not match.

#: Byte-identical to the array path with a gate verdict AND a ``certification.json``
#: record behind it. A fused product's bit-identity is a claim about the WELD, which no
#: verdict on either half establishes, so this stays empty until the fused gate has run
#: on a device.
#: RELEASED ON DEVICE 2026-09-11 by the campaign ``certification.json:cuda_complex_fused_electric_pair_2026-09-11_regate``
#: (parity/meep_gpu/gate_cuda_fused_complex_pairs.py --family complex_electric, keep and flush legs, 18/18 cases identical, 27/27 mutations caught).
#: Moved here from ``UNCERTIFIED_KERNELS`` by that verdict and by nothing
#: else: the gate ran BEFORE this edit, on the bytes that ship, and the
#: weld in ``cuda_kernels/fingerprints.json`` binds them.
CERTIFIED_KERNELS = (
    "fused_electric_pair_pml_complex",
)
#: Shipped without a gate verdict. The kernel is here to BE gated.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its fused launch with the deposit repair? TRUE, AND THE
#: WIRING IS ``fused_pairs._install_fused_pair``'s -- the same block that brackets the
#: real electric pair, reached because this family has a row in
#: ``fused_pairs.FUSED_PRODUCTS``. The flag and the wiring change together or not at all
#: (``deposit_repair.py:221-226``); a product that declared it without the two slots
#: would compute ``update_E`` against a pre-injection D and report success.
#:
#: WHAT THE REPAIR HAS TO INVERT HERE IS THE SAME ARITHMETIC IT ALREADY INVERTS.
#: ``deposit_repair.apply`` recomputes ``(f + kps*fw_fresh) - kms*fw_prev`` with ``xp``
#: array ops, with ``fw_fresh`` being ``D * inv_eps`` on this side rather than ``B``
#: (``deposit_repair.SEAMS["D"]``); under complex64 storage those are complex adds and
#: float32 x complex64 products, which is word for word what ``constitutive_apply``
#: computes. The two D-only clauses in ``deposit_repair.repairable`` (an off-diagonal
#: chi1inv row, an instantaneous chi2/chi3) are refused by the constitutive half's own
#: predicate before this predicate reaches the seam clause, and the fold clauses in
#: ``_folded_seam_reasons`` -- INCLUDING the one that names the cylindrical r = 0 axis
#: -- are vacuous on every row this admits, because it refuses a folded grid outright.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_complex_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_electric_pair_pml_complex"

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3302, :3310, :3313). Declared rather than inferred from the two slots.
#:
#: THE TWO FILLS ARE ABSENT BY REFUSAL, not by omission — see the module docstring.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ cf pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the helper names "
            "and returns the word pair it already computed. The expression, its "
            "parenthesisation and the store to f are unchanged, so the global state it "
            "leaves is the certified one."},
    {"line": "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), "
             "sinv_u));",
     "became": "    cf value = mul_field_left(cf_sub(cf_add(a, fu_new), fprev), "
               "sinv_u);\n    cf_store(f, idx, value);\n    return value;",
     "why": "same right-hand side, same tree, same store; the result is additionally "
            "named so it can be returned. A float32 word pair stored to global and "
            "reloaded is the identity on the bits."},
    {"line": "        pml_apply(f0, u0, idx, curl, ...);",
     "became": "        d0 = pml_apply_reg(f0, u0, idx, curl, ...);",
     "why": "capture the register. Three lines, one per component; the argument lists "
            "are untouched. Three 'cf d* = cf_zero();' declarations are hoisted above "
            "the certified braced blocks because a value declared inside one does not "
            "outlive it."},
    {"line": "    cf s0 = cf_load(g0, idx);",
     "became": "    cf s0 = d0;",
     "why": "THE SEAM. The certified constitutive body opens each component with a "
            "reload of the flux density the curl just stored; this reads the register "
            "instead. It also removes the constitutive half's only use of g0/g1/g2, "
            "which is what lets D be bound exactly once. The inverse-permittivity "
            "multiply on the NEXT line, its operand order (D on the left, "
            "stepping.py:1011) and the fact that all three products are formed before "
            "any store are untouched."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i]);",
     "became": "    constitutive_apply(h0, w0, idx, s0, kps_x[i], kms_half_x[i]);",
     "why": "TWO RENAMES, NO ARITHMETIC. f0 is D in the curl body and E here; kms_x is "
            "the INTEGER sub-lattice for the D curl and the HALF-INTEGER one for "
            "update_E -- the mirror image of the B/H seam's pairing -- and letting one "
            "shadow the other is a half-cell error in the absorber profile rather than "
            "a compile failure."},
    {"line": "        Dy[base] = 0.0f; Dz[base] = 0.0f;   "
             "(in_seam_passes._zero_metal_D_kernel_code)",
     "became": "    if (wall_x && i == 0) { d1 = cf_zero(); cf_store(f1, idx, d1); "
               "d2 = cf_zero(); cf_store(f2, idx, d2); }",
     "why": "THE ONE RE-SPELLING, and it is launch geometry rather than arithmetic. "
            "The certified pass walks a FACE with face_geometry(); this launch walks "
            "the VOLUME and already holds i/j/k, and stored cell 0 of an axis is "
            "exactly that face. Same OFF-DIAGONAL table (two components per wall), "
            "same (+0.0f, +0.0f) word pair. The register is cleared beside the store so "
            "update_E reads the wiped value."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body", "certified_curl_body",
    "complex_fused_electric_pair_prelude", "complex_fused_electric_pair_source",
    "complex_fused_electric_pair_tables", "covers_complex_fused_electric_pair",
    "device_sources", "inverse_epsilon_bindings",
    "launch_complex_fused_electric_pair", "run_complex_fused_electric_pair",
    "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================
#
# Every anchor below is an exact line of certified device text. If one stops matching,
# the certified string changed, and the splice raises rather than emitting a kernel
# that is quietly missing a mask, a store or a seam.

#: Separates a certified kernel's signature from its body. Both emitted strings close
#: their parameter list on its own line, so ONE anchor lifts either body.
_BODY_ANCHOR = "\n) {\n"

#: The last line of the index decomposition, emitted by BOTH bodies. The curl half
#: keeps it; the constitutive half's copy (and the bounds guard above it) is dropped,
#: because splicing both would redeclare i/j/k.
_DECODE_END = "    int i = idx / (ny * nz);\n"

#: The certified split-field helper, and the one edit that turns it into a value.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ cf pml_apply_reg(\n"
_PML_APPLY_STORE = (
    "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u));\n")
_PML_APPLY_STORE_REG = (
    "    // THE ONE EDIT TO THIS HELPER: the right-hand side, its\n"
    "    // parenthesisation and the store are the certified ones; the value is\n"
    "    // additionally NAMED so the constitutive half can read it from a\n"
    "    // register instead of reloading the word pair from global memory.\n"
    "    cf value = mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u);\n"
    "    cf_store(f, idx, value);\n"
    "    return value;\n")

#: The three component registers the seam carries, in target order.
_CARRIED: Tuple[str, ...] = ("0", "1", "2")

#: ``stepping._zero_metal``'s OFF-DIAGONAL table for the D side, as (flag, coordinate,
#: the two component indices wiped there). SPELLED HERE rather than imported from
#: anything B-shaped: the B family's table is the DIAGONAL (each component on its own
#: axis), and a pair that reused it would clear one wrong component and leave two right
#: ones standing on every walled run. ``fields.IYEE_SHIFTS`` gives Dx (1,0,0), Dy
#: (0,1,0), Dz (0,0,1), so a D component has Yee shift 0 on the OTHER TWO axes -- x
#: wall: Dy and Dz; y wall: Dx and Dz; z wall: Dx and Dy, which is
#: ``in_seam_passes._zero_metal_D_kernel_code``'s own table (:352-363).
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("wall_x", "i", ("1", "2")),
    ("wall_y", "j", ("0", "2")),
    ("wall_z", "k", ("0", "1")),
)

#: The fused entry point. The ONLY hand-written device text in this module, and it is a
#: signature: no arithmetic lives here. D (f0/f1/f2) appears EXACTLY ONCE -- see the
#: module docstring's aliasing section, which is the reason the constitutive half has no
#: flux-density source pointers at all.
_SIGNATURE = r'''
extern "C" __global__ void fused_electric_pair_pml_complex(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 D.
    // The curl half writes it and the constitutive half reads it from a register;
    // binding it a second time as a const __restrict__ source would be two restrict
    // pointers to one allocation, which is UB and which NVRTC miscompiles without a
    // diagnostic.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ u0, float* __restrict__ u1, float* __restrict__ u2,
    // H, the curl's operands.
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    // E and its constitutive history.
    float* __restrict__ h0, float* __restrict__ h1, float* __restrict__ h2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    // NOT __restrict__, and lifted from the certified update_E rather than decided
    // here: an isotropic run hands the same device pointer three times
    // (fields.py:1323-1325), and restrict on mutually aliasing arguments is a promise
    // the caller cannot keep. They stay FLOAT32 under complex storage and are indexed
    // by the COMPLEX CELL index, never word-doubled (stepping.py:41-50).
    const float* inv_eps_0, const float* inv_eps_1, const float* inv_eps_2,
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx,
    // The D curl's INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The E constitutive's HALF-INTEGER coefficients. Its kms is renamed kms_half_*:
    // the two sub-lattices collide on the bare name in one scope, and a shadow there
    // is a half-cell error in the absorber profile, not a compile failure. THE PAIRING
    // IS THE MIRROR IMAGE OF THE B/H SEAM'S.
    const float* __restrict__ kps_x, const float* __restrict__ kms_half_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_half_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_half_z,
    int bc_x, int bc_y, int bc_z,
    // The Bloch table: a flag per axis and the rounded complex64 factor split into
    // words. THE IMAGINARY PART IS THE CONJUGATE ONE -- this is the BACKWARD sub-step
    // and bloch_phase_arguments(grid, True) negates it (stepping._shift_down,
    // :1818-1822). ph = 0 SKIPS the multiply entirely rather than doing it against
    // 1+0j, which is what keeps k = 0 bit-identical to the plain complex engine.
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,
    // zero_metal_D's three walled-axis flags, from in_seam_coverage.zero_metal_axes.
    // A folded metallic axis is EXCLUDED there (stepping.py:2237-2239); this predicate
    // refuses a folded grid outright, so the two agree by construction.
    int wall_x, int wall_y, int wall_z
) {
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 2237-2239->2284-2286


def _split_body(source: str, prelude: str, name: str) -> str:
    """One emitted kernel string, reduced to its body.

    Strips the prelude it was built from and the signature that follows it, and the
    closing brace. Raises rather than returning a truncated body: a splice that
    silently dropped a component would produce a kernel that steps two fields and
    reports success.
    """
    if not source.startswith(prelude):
        raise AssertionError(
            f"{name} no longer begins with the prelude this module lifts separately; "
            f"the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, not 1; "
            f"the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    A missing anchor is certified text that has changed under this family, and splicing
    around it would produce a kernel that compiles and is quietly not the certified
    arithmetic; TWO matches would mean the anchor no longer identifies a single
    statement.
    """
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line starting "
            f"{prefix!r}); this family LIFTS that line rather than retyping it and "
            f"cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the lift "
            f"of {what} would take an arbitrary one")
    return matches[0]


def _prelude_for(arm) -> str:
    """The emitted prelude for one arm, exactly as ``complex_source`` builds it."""
    code = complex_emitter.normalized_expansion(arm)
    return (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[code]
            + complex_emitter._TAIL)


def complex_fused_electric_pair_prelude(arm) -> str:
    """The certified complex prelude with ``pml_apply`` turned into a value.

    ``constitutive_apply`` is emitted UNTOUCHED: the seam changes where its ``src``
    argument comes from and not what it does with it.
    """
    prelude = _prelude_for(arm)
    if prelude.count(_PML_APPLY_SIGNATURE) != 1:
        raise AssertionError(
            f"the certified complex prelude declares pml_apply "
            f"{prelude.count(_PML_APPLY_SIGNATURE)} times, not once; the value rewrite "
            f"has no anchor")
    if prelude.count(_PML_APPLY_STORE) != 1:
        raise AssertionError(
            "the certified pml_apply no longer closes with the store this module turns "
            "into a named value; the arithmetic may have moved")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_PML_APPLY_STORE, _PML_APPLY_STORE_REG, 1)
    return prelude


def certified_curl_body(arm) -> str:
    """``step_D_pml_complex_bloch``'s body, lifted, with the three registers captured.

    Not a transcription: this is what ``complex_emitter.complex_source("step_D", arm)``
    emits, minus its prelude and signature. ``step_B``'s FORWARD shifts are a different
    product and never appear here -- the backward ``cshift_dn`` and the conjugated
    phase are what make this the D side.
    """
    body = _split_body(complex_emitter.complex_source("step_D", arm),
                       _prelude_for(arm), "complex_source('step_D')")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the carried "
            "registers have nowhere to be declared")
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END,
        "\n    // The three carried registers. Declared HERE because each\n"
        "    // certified component below is a braced scope, and a value declared\n"
        "    // inside one does not outlive it.\n"
        "    cf d0 = cf_zero();\n"
        "    cf d1 = cf_zero();\n"
        "    cf d2 = cf_zero();\n"
        "\n",
        tail,
    ))
    for target in _CARRIED:
        old = f"pml_apply(f{target}, u{target}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} {body.count(old)} "
                f"times, not once; the capture has no anchor")
        body = body.replace(old, f"d{target} = pml_apply_reg(f{target}, u{target}, ", 1)
        # Matched as a WHOLE LINE rather than by a `);` tail: a tail anchor would attach
        # to whichever statement happened to end first if the certified body ever moved
        # this call onto two lines.
        prefix = f"        d{target} = pml_apply_reg(f{target}, u{target}, "
        call = _line_starting(body, prefix, f"target {target}'s captured pml_apply")
        if not call.endswith(");"):
            raise AssertionError(
                f"the certified curl body no longer closes target {target}'s pml_apply "
                f"on one line ({call!r})")
    if "pml_apply(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be written "
            "to global memory and never read into the seam")
    return body


def zero_metal_carry() -> str:
    """``zero_metal_D`` (driver.py:3310), carried between the two halves.

    ``stepping._zero_metal`` writes the integer 0 into stored cell 0 of every component
    whose Yee shift on a walled axis is 0 -- for D that is the OFF-DIAGONAL, TWO
    components per wall (:data:`_ZERO_METAL_ROWS`). The register is cleared BESIDE the
    store, because the constitutive half below reads the register and not the volume.

    A FOLDED METALLIC AXIS CARRIES NO WALL (``stepping._zero_metal``'s ``is_metallic
    and not is_mirrored``, :2237-2239). This kernel never has to model that: the
    predicate refuses every folded grid, so ``zero_metal_axes`` and the boundary codes
    cannot disagree on any admitted row.
    """
    lines = [
        "\n    // stepping._zero_metal (:2206-2245) / in_seam_passes.zero_metal_D,\n"
        "    // CARRIED. The OFF-DIAGONAL: a D component sits ON the wall of each\n"
        "    // axis whose Yee shift is 0, which for D is the other two -- so two\n"
        "    // components are wiped per walled axis. Stored cell 0 is the LOW wall;\n"
        "    // the high wall is the zero ghost cshift_dn already supplies.\n"
    ]
    for flag, coordinate, targets in _ZERO_METAL_ROWS:
        stores = " ".join(
            f"d{target} = cf_zero(); cf_store(f{target}, idx, d{target});"
            for target in targets)
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {stores} }}\n")
    return "".join(lines)


def certified_constitutive_body(arm) -> str:
    """``update_E_pml_complex_bloch``'s body, lifted, reading the registers.

    Drops the second index decomposition (the curl body above already declares ``idx``,
    ``i``, ``j`` and ``k`` and its bounds guard has already returned), turns the three
    source reloads into the seam, and resolves the two name collisions the module
    docstring names.
    """
    body = _split_body(complex_emitter.complex_source("update_E", arm),
                       _prelude_for(arm), "complex_source('update_E')")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own line; the "
            "duplicate decomposition cannot be identified")
    tail = body.split(_DECODE_END, 1)[1]

    # THE SEAM. Each certified reload becomes the register the curl already holds, which
    # is also what removes this half's only use of g0/g1/g2. The inverse-permittivity
    # multiply on the following line is untouched: it reads `s0`, which is now the
    # register rather than the reload, and its operand order is the certified one.
    for target in _CARRIED:
        old = f"    cf s{target} = cf_load(g{target}, idx);\n"
        if tail.count(old) != 1:
            raise AssertionError(
                f"the certified constitutive body loads s{target} {tail.count(old)} "
                f"times, not once; the seam has no anchor")
        tail = tail.replace(
            old,
            f"    cf s{target} = d{target};   // THE SEAM: the register the curl half "
            f"just stored\n", 1)
    if "cf_load(g" in tail:
        raise AssertionError(
            "a source reload survived the seam rewrite; the constitutive half would "
            "need D bound a second time, which is the aliasing hazard this signature "
            "exists to avoid")

    # THE INVERSE-PERMITTIVITY PRODUCTS ARE ASSERTED, NOT EDITED. They are the one part
    # of this body that has no counterpart on the H seam, so a change to them would be
    # exactly the kind of drift that compiles: the line is READ and required to be the
    # certified one, and nothing is substituted into it.
    for target in _CARRIED:
        expected = (f"    s{target} = mul_field_left(s{target}, "
                    f"inv_eps_{target}[idx]);")
        line = _line_starting(tail, f"    s{target} = mul_field_left(",
                              f"target {target}'s inverse-permittivity product")
        if line != expected:
            raise AssertionError(
                f"target {target}'s inverse-permittivity product is {line!r}, not the "
                f"certified {expected!r}; this family binds inv_eps_* on the strength "
                f"of that line and does not edit it")

    # THE TWO RENAMES. Matched as whole lines so a moved argument list is a named
    # failure rather than a partial substitution.
    for target, axis in zip(_CARRIED, ("x", "y", "z")):
        prefix = f"    constitutive_apply(f{target}, w{target}, idx, s{target}, "
        call = _line_starting(tail, prefix, f"target {target}'s constitutive_apply")
        coordinate = "ijk"[int(target)]
        expected = (f"{prefix}kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);")
        if call != expected:
            raise AssertionError(
                f"target {target}'s constitutive_apply is {call!r}, not the certified "
                f"{expected!r}; the sub-lattice rename would be applied to an argument "
                f"list this module has not read")
        tail = tail.replace(
            call,
            call.replace(f"(f{target},", f"(h{target},", 1)
                .replace(f"kms_{axis}[", f"kms_half_{axis}[", 1), 1)
    for target in _CARRIED:
        if f"(f{target}," in tail:
            raise AssertionError(
                f"f{target} survived the target rename; it is D in the curl body and E "
                f"here, and the two would collide")
    for target, axis in zip(_CARRIED, ("x", "y", "z")):
        coordinate = "ijk"[int(target)]
        if f"kms_{axis}[{coordinate}])" in tail:
            raise AssertionError(
                f"kms_{axis} survived the sub-lattice rename; it is the INTEGER vector "
                f"in the curl body and the HALF-INTEGER one here, and the two would "
                f"collide on the bare name")
    return tail


def complex_fused_electric_pair_source(arm) -> str:
    """The whole fused kernel for one expansion arm: prelude, signature, body."""
    source = "".join((
        complex_fused_electric_pair_prelude(arm),
        _SIGNATURE,
        certified_curl_body(arm),
        zero_metal_carry(),
        certified_constitutive_body(arm),
        "}\n",
    ))
    # PURE ASCII IS A COMPILE REQUIREMENT, not a style rule; checked here so a mutation
    # leg that inserts a non-ASCII character is refused at emission with the reason
    # instead of at NVRTC three frames away.
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by arm name -- for a digest."""
    return {name: complex_fused_electric_pair_source(name)
            for name in sorted(complex_emitter.EXPANSIONS)}


#: NVRTC compile options -- CORRECTNESS, not performance, and identical to both
#: certified halves'. Spelled here rather than imported so that loading this file by
#: path (which the bit-identity probe does) cannot pick up a different tuple than the
#: one the gate compiled.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane -- what every complex sibling landed on.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm, name: str = KERNEL_NAME):
    """Compile the fused kernel under one arm, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING, for the reason every
    sibling rebuilds its code map per call: the source is part of the memo key, so it
    has to be built before there is a key to miss on, and a gate mutates this family by
    monkeypatching the emitter. A source memoized at first call would hand back the
    pre-mutation string forever -- a leg reporting a pass for a mutation it never
    applied.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    code = complex_fused_electric_pair_source(arm)
    key = compile_cache.kernel_cache_key(
        f"{name}_arm{complex_emitter.normalized_expansion(arm)}", True,
        _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_complex_fused_electric_pair(fields: Any, pml: Any, grid: Any,
                                       sources: Any = None, license: Any = None,
                                       subnormal_policy: Any = None
                                       ) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``zero_metal_D`` -> ``update_E``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal, which is
    this directory's convention.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own certified
    predicate refuses is refused here with that half's reason, prefixed so a reader can
    tell which side said it. What this predicate ADDS is the seam clauses: the source
    slot, the injection ROUTE, the two UNCARRIED fills, and the grid's ability to say
    which axes are walled.

    ``license`` and ``subnormal_policy`` are the complex family's arm arbitration and
    are passed straight through to both halves. They are REQUIRED in practice: the arm
    is compiled into the binary at this seam and there is no later rung to check it at,
    and a wrong arm is a wrong answer rather than a crash.
    """
    covered, reason = covers_real_pml_complex_curl(
        fields, pml, grid, "step_D", license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "E", license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE TWO FILLS ARE REFUSED, NOT CARRIED, and the refusal is stated HERE rather than
    # inherited. The curl half above already refuses every folded grid, so this clause
    # is unreachable today -- and that is precisely why it is written: the guarantee
    # that `fill_symmetry_bc_D` (driver.py:3309) and `fill_folded_far_ghosts_D` (:3311)
    # do nothing inside this seam is what makes REPLACES honest, and it may not depend
    # on a clause in another module that a future device verdict could licence away.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_D and "
                       "fill_folded_far_ghosts_D run inside it")
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry or any(folded):
        return False, ("a mirror plane is active: fill_symmetry_bc_D "
                       "(driver.py:3309) and fill_folded_far_ghosts_D (:3311) run "
                       "inside this seam and this pair carries neither -- the "
                       "ownership inversion that carries the B family's two fills is "
                       "fused_magnetic_pair's and is not ported here")

    # THE INJECTION ROUTE, REFUSED BY NAME, and it is a D-seam clause with no B-seam
    # counterpart. The driver has TWO ways to deposit an electric source in this seam:
    # the plain `source.inject` loop (driver.py:3308) and
    # `_inject_electric_through_conductivity` (:3305), which snapshots the D components,
    # injects, and then rescales only the increment by `condinv` (MEEP step.cpp:294-317).
    # THE COMPLEX HALVES BOTH ADMIT A CONDUCTIVITY BY NAME -- `fields.condfac_for` is
    # read in `stepping._apply_curl` and nowhere else, so neither constitutive sub-step
    # sees it (coverage.covers_real_pml_complex_constitutive's own paragraph) -- so
    # unlike the real electric pair this refusal is NOT inherited from a half and has to
    # be made here. The deposit repair has never been measured against the scaled route,
    # and a repair that reconstructed the unscaled increment would be a silently wrong
    # field at every deposit point.
    if getattr(fields, "has_conductivity", False):
        return False, ("the engine carries a conductivity, so the driver deposits "
                       "this seam's electric sources through "
                       "_inject_electric_through_conductivity (driver.py:3305), which "
                       "rescales the increment by condinv; the deposit repair has no "
                       "verdict on that route")

    # THE SOURCE SEAM, CARRIED RATHER THAN REFUSED. An ELECTRIC source is injected
    # BETWEEN the two halves (driver.py:3305/:3308), so a fused pair computes update_E
    # against a pre-injection D; `fused_pairs._install_fused_pair` brackets the launch
    # with LeadingRepairPlan/TrailingRepairPlan, CARRIES_DEPOSIT_REPAIR says so, and the
    # shared clause then asks what the repair can actually reconstruct. IGNORANCE IS
    # NEVER AN EMPTY SET: `Fields` does not hold the source list, so a predicate that
    # inferred "no sources" from not being told would be exactly the over-covering this
    # clause exists to prevent. A MAGNETIC source is injected in the B/H half and does
    # not disqualify this pair.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3308)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

    # zero_metal_D IS CARRIED, so the grid must be able to say which axes are walled. A
    # grid that cannot answer would silently be treated as unwalled, which is a plane of
    # wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_D cannot be "
                           f"carried in registers")
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE STORED SHAPE THE KERNEL INDEXES MUST BE THE ONE THE WALL PLANE IS DERIVED
    # FROM. `nx, ny, nz` come from `fields.Dx.shape`; the wall is stored cell 0 of a
    # `grid.stored_cells` axis. If the two ever disagree the wipe is right for an array
    # this launch is not walking.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Dx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks Dx.shape "
                       f"{extents}; the wall plane is derived from the first and "
                       f"indexed into the second")

    # THE THREE INVERSE-PERMITTIVITY VOLUMES MUST BE ASKABLE AND MUST MATCH THE SHAPE
    # THIS LAUNCH WALKS. The constitutive half's own predicate already checks them
    # against `_grid_facts`' shape; this asks the same question against the extents THIS
    # LAUNCH walks (`fields.Dx.shape`), which is the array the flat idx indexes.
    accessor = getattr(fields, "inverse_epsilon_for", None)
    if not callable(accessor):
        return False, ("fields does not expose inverse_epsilon_for; update_E's three "
                       "inverse-permittivity volumes cannot be bound")
    for component in _INVERSE_EPSILON_COMPONENTS:
        try:
            volume = accessor(component)
        except Exception as exc:  # noqa: BLE001
            return False, (f"fields.inverse_epsilon_for({component!r}) raised: "
                           f"{type(exc).__name__}: {exc}")
        if volume is None:
            return False, f"fields.inverse_epsilon_for({component!r}) is None"
        shape = tuple(int(n) for n in getattr(volume, "shape", ()))
        if shape != extents:
            return False, (f"inverse_epsilon_for({component!r}) has shape {shape} but "
                           f"the launch walks {extents}; the kernel indexes it with "
                           f"this launch's flat index")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The fifteen complex volumes, in SIGNATURE ORDER. Nothing else keeps the launch and
#: the kernel in step, so the order is spelled once and both the binding check and the
#: launch read it. The three inv_eps pointers are NOT here: they are bound separately
#: because they may legitimately alias each other.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz",
    "fu_Dx", "fu_Dy", "fu_Dz",
    "Hx", "Hy", "Hz",
    "Ex", "Ey", "Ez",
    "f_w_Ex", "f_w_Ey", "f_w_Ez",
)

#: The three E components whose inverse permittivity ``update_E`` multiplies by, in
#: signature order.
_INVERSE_EPSILON_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: The curl's INTEGER coefficient keys and the constitutive's HALF-INTEGER ones, in
#: signature order. THE PAIRING IS THE OPPOSITE OF THE B/H SEAM'S and the kernel cannot
#: tell -- a swap on either group is a half-cell error in the absorber profile,
#: converged, smooth and wrong.
_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def complex_fused_electric_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads.

    THE SUB-LATTICE IS DECIDED HERE, not by the caller: while it was an argument, every
    call site was one more place the pairing could be got backwards, and backwards is a
    half-cell error rather than a crash. The D curl reads the INTEGER positions
    (``complex_emitter.HALF_INTEGER['step_D']`` is False) and ``update_E`` the
    HALF-INTEGER ones (``HALF_INTEGER['E']`` is True, stepping.py:1015) -- the opposite
    pairing to :mod:`.complex_fused_magnetic_pair`'s.
    """
    return {"curl": complex_curl_tables(pml, False),
            "constitutive": complex_constitutive_tables(pml, True)}


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three per-component inverse-permittivity volumes, in signature order.

    ``Fields.inverse_epsilon_for`` and never ``fields.inv_eps``, which is the Ez view
    (fields.py:1259-1260) -- binding it for all three is the defect the twelve
    uncertified complex kernels in ``step_curl_kernels`` carry, named in
    ``complex_pml_kernels.update_fused_pml_complex``'s own docstring.
    """
    return tuple(  # type: ignore[return-value]
        fields.inverse_epsilon_for(component)
        for component in _INVERSE_EPSILON_COMPONENTS)


def assert_disjoint_bindings(fields: "Fields",
                             tables: Dict[str, Dict[str, Any]]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes.

    ONCE PER FROZEN CONFIGURATION, OFF THE LAUNCH PATH. The shared flux density is
    bound once by construction -- the signature has no second D group to hand it to --
    so what is left to check is that no OTHER two restrict arguments are the same
    allocation. Two ``__restrict__`` pointers to one object is UB whatever the route to
    it, and NVRTC reorders across it without a diagnostic.

    THE THREE inv_eps VOLUMES ARE CHECKED DIFFERENTLY AND DELIBERATELY. They are not
    ``__restrict__`` in the signature, so they may alias EACH OTHER -- an isotropic run
    hands one pointer three times and the certified kernel is built for it. What they
    may NOT do is alias a restrict-qualified argument, which would break that argument's
    promise from the other side, so that is the check they get.

    Returns the number of distinct allocations checked, so a caller can assert that
    something was actually inspected; a check that examined nothing and reported success
    is the vacuity this whole track guards against.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in _FIELD_BINDINGS:
        visit(name, getattr(fields, name))
    for key in _CURL_TABLE_KEYS:
        visit(f"curl:{key}", tables["curl"][key])
    for key in _CONSTITUTIVE_TABLE_KEYS:
        visit(f"constitutive:{key}", tables["constitutive"][key])
    for component, volume in zip(_INVERSE_EPSILON_COMPONENTS,
                                 inverse_epsilon_bindings(fields)):
        pointer = int(volume.data.ptr)
        if pointer in bound:
            collisions.append(
                f"inv_eps_{component} and {bound[pointer]} are the same allocation, "
                f"and the second is bound __restrict__")
    if collisions:
        raise ValueError(
            "the complex fused electric pair binds every field and table argument "
            "__restrict__, and these arguments alias, which is undefined behaviour "
            "NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_complex_fused_electric_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], phase_flags: Sequence[Any],
        phase_values: Sequence[Any], walls: Sequence[int], dtdx: float,
        arm: Any, kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch.

    ``boundary_codes`` are ``complex_pml_kernels.complex_boundary_codes(grid)``;
    ``phase_flags``/``phase_values`` are ``bloch_phase_arguments(grid, True)`` -- THE
    BACKWARD TABLE, whose imaginary parts are the conjugates of the magnetic sibling's,
    and handing this launch the forward one is a wrong answer on every phased row rather
    than a crash. ``walls`` are ``in_seam_coverage.zero_metal_axes(grid)``. THE CODES
    AND THE WALLS ANSWER DIFFERENT QUESTIONS and confusing them is a plane of wrong
    values: the codes are ``_boundary_kinds``' ghost-rule resolution and the walls are
    ``_zero_metal``'s own declaration.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is: a gate
    compiles a deliberately broken copy of the shipped source and hands it here. A
    launcher that could not be handed its own kernel could not arm a single mutation,
    and every mutation leg would silently launch the shipped one and report the defect
    as uncaught.

    Returns the launch geometry rather than ``None`` so a gate can assert that something
    was actually launched.
    """
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(word_view(getattr(fields, name)) for name in _FIELD_BINDINGS) + (
        tuple(inverse_epsilon_bindings(fields))
    ) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + tuple(np.int32(code) for code in boundary_codes) + tuple(
        np.int32(flag) for flag in phase_flags
    ) + tuple(np.float32(value) for value in phase_values) + tuple(
        np.int32(int(bool(walls[axis]))) for axis in range(3))
    (kernel or _get_kernel(arm))((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "arm": complex_emitter.normalized_expansion(arm)}


def run_complex_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, arm: Any, *,
        sources: Any = None, license: Any = None, subnormal_policy: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``tables`` and ``kernel`` are the gate's doors, keyword-only. Passing ``tables`` is
    how a swapped-sub-lattice mutation is armed: a launcher that always derived its own
    could not be handed mis-paired ones.
    """
    covered, reason = covers_complex_fused_electric_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = complex_fused_electric_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    # THE BACKWARD PHASE TABLE. `True` is `complex_emitter.KERNELS['step_D'][1]`, read
    # off the same table `complex_pml_kernels.step_fused_pml_complex` reads it from, and
    # it is the one argument of this call a copy of the magnetic sibling would get
    # silently wrong.
    flags, values = bloch_phase_arguments(grid, True)
    return launch_complex_fused_electric_pair(
        fields, tables, complex_boundary_codes(grid), flags, values,
        zero_metal_axes(grid), dtdx, arm, kernel)
