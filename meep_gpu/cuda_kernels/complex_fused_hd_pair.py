"""The H->D weld under COMPLEX64 storage: ``update_H`` welded into ``step_D``.

The complex sibling of :mod:`.fused_hd_pair`, on the same seam and with the same
shape -- ONE launch, scratch output, foreign-cell recompute, a rotation after --
and the same two flags (:data:`INSTALLABLE` False, :data:`HOISTS_THE_WITHDRAW`
False). What differs is the arithmetic, the storage and one structural fact this
backend's real family does not have: on the complex side a MIRROR FOLD is a
SEPARATE DEVICE TEXT rather than a runtime boundary code, so this product emits
two variants.

=============================================================================
THE CELL, AND WHY THE FOLDED NEIGHBOUR COMES WITH IT
=============================================================================

``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-07_cyl/fusion_matrix_cuda.json``,
``h_to_d_seam.instances`` filtered to ``buildable_not_built``:

* ``(update_H cuda_complex/complex -> step_D cuda_complex/complex)`` -- **17
  instances**, the largest unbuilt H->D cell on this backend;
* ``(update_H cuda_complex/complex -> step_D cuda_complex_folded/folded complex)``
  -- **5 instances**, the neighbouring cell.

THE TWO CELLS SHARE THEIR H HALF EXACTLY. ``complex_folded_kernels`` ships NO
constitutive kernel: its own docstring says the constitutive sides "are served by
the certified complex pair unchanged", and the census records
``cuda_complex/complex`` at ``update_H`` on all five folded rows. So the folded
cell differs from the plain one in the CURL half alone -- and that half is not a
rewrite either: ``complex_folded_kernels.folded_source`` is
``complex_emitter.complex_source`` put through three anchored text deltas plus a
rename (a third boundary code, the ghost widened to the metallic zero, and the
mask split by termination).

This product therefore takes the SAME weld transform over BOTH curl texts and
emits two kernels, :data:`KERNEL_NAMES`. **22 seam-instances, one transform, two
variants** -- which is stated here because the alternative (a second module for
the folded five) would duplicate the recompute machinery and give the two cells
two different chances of being wrong.

THE VARIANT IS RESOLVED FROM THE GRID, not chosen: :func:`variant_for` reads
``grid.is_mirrored`` per axis, and the predicate refuses a configuration whose
variant its caller contradicted. A fold handed to the plain text compiles and runs
and is wrong on one plane of one component -- the exact failure
``complex_folded_kernels``' site counts exist to prevent -- so nothing here
defaults.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE
=============================================================================

Identical in structure to :mod:`.fused_hd_pair`, and identical in JUSTIFICATION,
so the argument is not restated at length here. In brief:

* ``step_D``'s curl reads ``H`` at the thread's own cell and at three BACKWARD
  neighbours (``cshift_dn``); ``update_H`` writes ``H`` and ``f_w_H``. Welded in
  place that is a race on both volumes -- and on ``f_w_H`` especially, since the
  newly stored ``f_w_H`` IS ``B`` exactly (``constitutive_apply``'s
  ``cf_store(fw, idx, src)`` with ``src`` the loaded ``B``), so an in-place write
  hands a racing neighbour ``B`` where its recurrence needs ``B_prev``.
* **The constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go
  to LAUNCH-LOCAL SCRATCH; ``H``, ``f_w_H`` and ``B`` are ``const`` for the whole
  launch.
* **The foreign read is a RECOMPUTE**, through :func:`raw_update_H_cell_source` --
  the certified ``update_H_pml_complex_bloch`` body, whole, evaluated at an
  arbitrary cell. ``stepping.update_H`` is POINTWISE under complex storage exactly
  as it is under real (``_apply_constitutive_pml``, stepping.py:2112-2143; the
  emitted ``_CONSTITUTIVE_TEMPLATE`` reads no neighbour, carries no ghost rule, no
  mask and no phase and says so in its own comment), so evaluating it twice gives
  the same bits by construction.
* The launcher then ROTATES the ``H``/``f_w_H`` bindings
  (:func:`.fused_hd_pair.rotate_into_fields`, reused rather than re-spelled).

WHERE THE BLOCH PHASE SITS, AND WHY THE RECOMPUTE DOES NOT CARRY ONE. The phase is
applied by ``cshift_dn`` to the WRAPPED LANE ONLY and AFTER the load
(``_TAIL``: ``cf w = cf_load(g, idx + (na - 1) * stride); if (ph) w =
rotate_field_left(w, phase);``). :func:`cshift_dn_recompute_source` replaces the
LOAD and leaves the branch, the wrap arithmetic, the ``ph`` guard and the
``rotate_field_left`` call exactly where the certified helper puts them, so the
recomputed neighbour is phased under precisely the conditions the stored one would
have been. A recompute that carried a phase of its own, or applied one to the
near-neighbour branch, would be a band structure that converges to the wrong
dispersion.

THE METALLIC AND MIRROR GHOSTS ARE NOT RECOMPUTED. ``cshift_dn`` returns
``cf_zero()`` on those branches without touching ``g``, and the resolved helper
keeps that return verbatim: the ghost is a constant, not a magnetic field.

=============================================================================
THE ARITHMETIC IS THE CERTIFIED FAMILY'S, INCLUDING ITS ARM
=============================================================================

Every complex operation in this kernel comes from ``complex_emitter``'s prelude:
``cf_load``/``cf_store``'s word-pair addressing, ``cf_add``/``cf_sub``,
``cf_zero``, ``rotate_field_left``, ``mul_field_left``, ``mul_coefficient_left``,
``pml_apply`` and ``constitutive_apply``. NOTHING is retyped. The arm
(``NAIVE``/``FMA_V1``) is an ARGUMENT with no default, for the reason
``complex_emitter.normalized_expansion`` refuses rather than defaulting: a wrong
arm is a wrong ANSWER, not a crash, and the two differ in the last bits of about a
quarter of the words.

TWO CuPy SPELLINGS MEASURED ON THIS BACKEND ARE RESPECTED RATHER THAN
REDISCOVERED (``lanes/cyl_round/cupy_probe``, 2026-09-06/07, one RTX A6000, CuPy
13.5.1, both float32 subnormal policies):

* ``complex64 * float32`` is the four-product form CONTRACTED by NVRTC. The
  UNCONTRACTED transcription is 0 on 11 of 12 spelling cases and 6 of 4,005,000
  words on (flush, edge): the sign of a FLUSHED zero. The spelling that is 0 on all
  twelve under ``--fmad=false`` is ``_ARM_SOURCE[FMA_V1].mul_field_left``, which is
  what this kernel calls -- through the emitter's own text, under EITHER arm's name
  binding, because the arm block is lifted whole.
* ``complex64 / float32`` is CuPy's SCALED complex/complex algorithm with the
  zero-valued terms kept, NOT numpy's reciprocal multiply. **This family performs
  NO DIVISION** -- the reciprocal its recurrence needs is ``sinv``, computed
  host-side by ``PML`` -- so the divide spelling has no site here. That is recorded
  rather than omitted, because a reader coming from the cylindrical H->D products
  (where the increment DOES divide) will look for it.

Both facts are armed as gate mutations anyway, on a PLANTED row-0 tiny-normal
class: a random battery does not exercise the class where the wrong multiply shows.

=============================================================================
WHAT SITS IN THE SEAM, AND WHY THIS PRODUCT REFUSES THE WITHDRAW ROWS
=============================================================================

The same one statement its real sibling documents: ``for source in electric:
getattr(source, "withdraw", _no_withdraw)(self.fields)`` (driver.py:3313-3314).
Nothing is INJECTED here, so :data:`CARRIES_DEPOSIT_REPAIR` is False as a FACT and
:mod:`..withdraw_hoist` -- not :mod:`..deposit_repair` -- is this seam's contract.

:data:`HOISTS_THE_WITHDRAW` is False, and the predicate therefore refuses BY NAME
every row whose electric withdraw does work. On the 22 instances of the two cells
that costs NOTHING measurable: all 22 carry ``integrated_electric_sources = 0`` on
the standing board, and none of them is in the ``withdraw_seam`` bucket. The
clause is still asked, because a predicate that admitted a configuration it cannot
serve would be wrong on the first row that acquired an integrated source.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS THE SIBLING'S MEASURED VERDICT
=============================================================================

:data:`INSTALLABLE` is False. Over the driver's ``step_B - update_H - step_D -
update_E`` slot path launches are ``4 - (installed pairs)``, and a product spanning
``update_H``/``step_D`` takes one slot from EACH neighbour -- and on THIS cell both
neighbours are released products (``cuda_complex_fused_magnetic_pair`` at
``step_B``/``update_H``, ``cuda_complex_fused_electric_pair`` at
``step_D``/``update_E``), so the span can only TIE or LOSE. That is the Cartesian
algebra ``fused_hd_pair.INSTALLABLE_REASON`` prices, and unlike the cylindrical
cell there is nothing here to re-price: this cell's seam is 2 launches today and
this product takes it to 1, saving 1, exactly what each neighbour saves.

WHAT KEEPS THE RELEASED B->H PAIR IN ITS SLOT IS NOT THIS FLAG. It is
``fused_pairs._neighbouring_seam_claimant``'s end-edge guard together with
``_spans_may_absorb`` reading the live ``selected``. The flag is belt and braces in
front of them, and the gate's arbitration leg measures both arrangements.

=============================================================================
NOT DISPATCHED, AND NOT YET IN THE COMPOSER'S TABLES
=============================================================================

Nothing under ``meep_gpu/`` imports ``cuda_kernels`` outside tests;
``fastpath.plan_fast_path`` returns ``None`` on every branch. Beyond that, this
family is not in ``fused_pairs.FUSED_PRODUCTS``, ``registry`` or ``arms`` at all:
the wiring is a separate change (``lanes/unbuilt_round/cuda_complex/wiring.patch``)
that a later round applies. The gate PLANS EVERY ARRANGEMENT FROM ARRAYS and drives
the array path itself; its ``composition_today`` reference is the two released
complex pairs launched through their own entry points, and the composer building a
composition that includes THIS product is measured in-process with the wiring rows
patched in (the arbitration leg) and awaits the wiring round.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Every import here is CuPy-FREE at module scope, so the predicate, the emitter and
# the whole transform stay callable on a laptop with no device. CuPy is imported
# inside the launchers and the compile memo, at their call sites.
from . import complex_emitter
from . import complex_folded_kernels
from . import fused_hd_pair as _hd
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import covers_real_pml_complex_constitutive, covers_real_pml_complex_curl
from .. import withdraw_hoist as _withdraw_hoist

# The complex TABLE builders and the word view import CuPy at scope; taken
# defensively so this module still imports where there is none.
try:
    from . import complex_pml_kernels
except Exception:  # noqa: BLE001 - no CuPy on this host
    complex_pml_kernels = None  # type: ignore[assignment]

FAMILY = "cuda_complex_fused_hd_pair"
SLOT = "update_H"
REPLACES: Tuple[str, ...] = ("update_H", "step_D")
SEAM: str = _withdraw_hoist.SEAM

#: The two variants and the kernel each emits. ``plain`` welds
#: ``complex_emitter.complex_source("step_D", arm)``; ``folded`` welds
#: ``complex_folded_kernels.folded_source("step_D", arm)``, which is the same text
#: through that module's three deltas. The names are spelled as LITERALS in
#: :func:`signature` as well, because ``test_kernel_partition.py`` reads shipped
#: kernel names off this file's own text with a regex that requires a C identifier.
KERNEL_NAMES: Dict[str, str] = {
    "plain": "fused_hd_pair_pml_complex_bloch",
    "folded": "fused_hd_pair_pml_complex_folded",
}

VARIANTS: Tuple[str, ...] = ("plain", "folded")

#: The default the tooling that expects a single ``KERNEL_NAME`` reads. It is the
#: PLAIN variant's, which is the 17-instance cell; every launcher and every source
#: function takes the variant explicitly and none of them consults this.
KERNEL_NAME: str = KERNEL_NAMES["plain"]

#: The six volumes the rotation swaps -- ``fused_hd_pair``'s own tuple, reused so
#: the two products cannot disagree about what a rotation covers.
SCRATCH_VOLUMES: Tuple[str, ...] = _hd.SCRATCH_VOLUMES
H_TARGETS: Tuple[str, str, str] = _hd.H_TARGETS

#: The curl's own magnetic source parameters, in the emitted template's spelling.
CURL_SOURCE_NAMES: Tuple[str, str, str] = ("g0", "g1", "g2")

#: How many shifted magnetic taps and own-cell magnetic loads the certified
#: ``step_D`` curl makes. Both are ASSERTED against the lifted text on every emit: a
#: tap left standing would read a word another block is writing, and an own load left
#: standing would read the PRE-launch H.
HALO_TAPS = 6
OWN_LOAD_EDITS = 6

#: BYTE-IDENTICAL TO FOUR INDEPENDENT ARRANGEMENTS OF THE SEAM, WITH A DEVICE VERDICT
#: BEHIND IT AND A RECORD BEHIND THE VERDICT. ``certification.json``'s
#: ``cuda_complex_fused_hd_pair_2026-09-07`` block names the two artifacts these lines
#: rest on (``parity/meep_gpu/results/cuda_complex_fused_hd_pair_2026-09-07/
#: {keep,flush}/gate.json``), and ``test_kernel_partition.py`` refuses a name here
#: that no record block claims.
#:
#: THE DECLARATION IS THE FINAL BYTES: the campaign was cut against THESE bytes with
#: the names still in ``UNCERTIFIED_KERNELS``, and the record binds the digests the
#: campaign observed at NVRTC, which this move does not change (the partition sets are
#: read off the syntax tree and are not part of any emitted device string).
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "fused_hd_pair_pml_complex_bloch",
    "fused_hd_pair_pml_complex_folded",
)

#: EMPTY, and what emptied it was the run above rather than an argument. Spelled as a
#: dict WITHOUT a type annotation, for the reason every sibling records: the partition
#: readers walk the syntax tree so they run where there is no CuPy, and an annotated
#: assignment is an ``ast.AnnAssign`` the plain-assignment readers do not match --
#: annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}

#: Nothing is injected between the two consults, so there is no deposit to bracket.
#: A fact about the driver, not a choice -- see the module docstring.
CARRIES_DEPOSIT_REPAIR = False

#: Not in this round: the only wiring that performs the hoist is
#: ``fused_pairs._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, which
#: ``_declared_uninstallable`` makes unreachable for a product declaring INSTALLABLE
#: False. True here would be a claim about wiring that cannot fire.
HOISTS_THE_WITHDRAW = False

INSTALLABLE = False
INSTALLABLE_REASON = (
    "slot arbitration, and on THIS cell the Cartesian algebra applies unchanged. Over "
    "the driver's step_B - update_H - step_D - update_E slot path launches are "
    "4 - (installed pairs); both of this cell's neighbours are RELEASED products "
    "(cuda_complex_fused_magnetic_pair holds step_B/update_H, "
    "cuda_complex_fused_electric_pair holds step_D/update_E), and a span taking one "
    "slot from each can only TIE (one neighbour installs) or LOSE (both do). The "
    "cylindrical H->D products re-priced this because their seam is 7 launches and "
    "the span takes it to 3; here the seam is 2 launches and the span takes it to 1, "
    "saving exactly what each neighbour saves. So INSTALLABLE stays False -- credited "
    "on the board as served by predicate admission, as fused_hd_pair is -- and the "
    "only shape that could pay is a four-slot step_B -> update_H -> step_D -> update_E "
    "weld, which is not built here")
WHAT_A_RELEASE_DOES_NOT_LICENSE = (
    "SERVED on the fusion board is PREDICATE ADMISSION by the board's own definition, "
    "so a released product credits its admitted seam-instances while executing "
    "NOWHERE: nothing under meep_gpu/ imports cuda_kernels outside tests, this family "
    "declares INSTALLABLE = False so the composer refuses to install it on every "
    "configuration once wired, it is not in the composer's tables until the wiring "
    "change lands, and NO TIMING of any kind has been taken of this shape. The fused "
    "route does MORE memory traffic than the two singles for the same step-level "
    "launch count -- up to six extra pointwise complex constitutive evaluations per "
    "thread -- and whether the saved launch and the saved H write-then-read round trip "
    "pay for them is a HYPOTHESIS. Every launch figure in this module is a COUNT.")

_FUSED_THREADS = 256

#: ``--fmad=false`` is CORRECTNESS and is the certified complex family's own tuple
#: (``complex_pml_kernels._COMPILE_OPTIONS``), restated here rather than imported so
#: that loading this file by path cannot pick up a different one than the gate
#: compiled. The constitutive half's two separate accumulations and the curl's
#: ``((sf - f_1) + (f_2 - ss))`` grouping are contraction candidates; the arm's own
#: fusions are explicit ``__fmaf_rn`` and survive the flag.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CURL_SOURCE_NAMES", "FAMILY", "HALO_TAPS",
    "HOISTS_THE_WITHDRAW", "H_TARGETS", "INSTALLABLE", "INSTALLABLE_REASON",
    "KERNEL_NAME", "KERNEL_NAMES", "OWN_LOAD_EDITS", "REPLACES", "SCRATCH_VOLUMES",
    "SEAM", "SLOT", "UNCERTIFIED_KERNELS", "VARIANTS",
    "WHAT_A_RELEASE_DOES_NOT_LICENSE", "assert_bindings_are_disjoint",
    "constitutive_prelude", "covers_complex_fused_hd_pair",
    "cshift_dn_recompute_source", "curl_pieces", "device_sources", "kernel_name",
    "kernel_source", "launch_complex_fused_hd_pair", "raw_update_H_cell_source",
    "resolution_source", "resolve", "rotate_into_fields",
    "run_complex_fused_hd_pair", "signature", "source_digest", "variant_for",
    "weld_args_construction", "weld_args_struct",
]


# ---------------------------------------------------------------------------
# The lift anchors
# ---------------------------------------------------------------------------

#: The signature terminator every emitted kernel closes on.
_BODY_ANCHOR = "\n) {\n"

#: Where a sub-step source stops being the prelude and starts being the kernel.
_KERNEL_MARKER = '\nextern "C" __global__ void '

_THREAD_PREAMBLE = ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                    "    if (idx >= nx * ny * nz) return;\n")
_INDEX_DECODE = ("    int k = idx % nz;\n"
                 "    int j = (idx / nz) % ny;\n"
                 "    int i = idx / (ny * nz);\n")

#: The curl body splits here -- the first target block. Everything above is the
#: preamble, the strides, the decode and the three phase constants; everything below
#: is the three target blocks whose magnetic reads this weld redirects.
_FIRST_TARGET = "    // Target 0:"

_APPLY_SIGNATURE = "__device__ __forceinline__ void constitutive_apply(\n"
_APPLY_SIGNATURE_PURE = "__device__ __forceinline__ cf constitutive_apply_pure(\n"
_APPLY_PARAMETERS = "    float* f, float* fw, int idx, cf src, float kps, float kms\n)"
_APPLY_PARAMETERS_PURE = ("    const float* f, const float* fw, int idx, cf src, "
                          "float kps, float kms,\n    cf* fw_out\n)")
_APPLY_FW_STORE = "    cf_store(fw, idx, src);\n"
_APPLY_FW_STORE_PURE = (
    "    // THE FIRST STORE, HANDED BACK INSTEAD OF PERFORMED: the caller writes it to\n"
    "    // the SCRATCH volume for its own cell and DISCARDS it for a foreign\n"
    "    // recompute, so nothing this launch reads is a word this launch wrote.\n"
    "    *fw_out = src;\n")
_APPLY_F_STORE = "    cf_store(f, idx, a);\n"
_APPLY_F_STORE_PURE = (
    "    // THE SECOND STORE, LIKEWISE: the same two separate accumulations in the same\n"
    "    // order, the value RETURNED so the caller stores it to scratch for its own\n"
    "    // cell and consumes it from a register for a foreign one.\n"
    "    return a;\n")

_CSHIFT_DN_SIGNATURE = ("__device__ __forceinline__ cf cshift_dn(\n"
                        "    const float* g, int idx, int ia, int na, int stride, "
                        "int bc, int ph, cf phase\n)")
_CSHIFT_DN_SIGNATURE_RECOMPUTE = (
    "__device__ __forceinline__ cf cshift_dn_recompute(\n"
    "    int comp, int idx, int ia, int na, int stride, int bc, int ph, cf phase,\n"
    "    const WeldArgs& weld\n)")

#: The argument pack, as ``(C type, member name)``. Every member is READ-ONLY for the
#: whole launch: that is the property the recompute rests on, and binding H, f_w_H and
#: B ``const`` here is where it is stated in the type system rather than in a comment.
WELD_ARGS_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("int", "nx"), ("int", "ny"), ("int", "nz"),
    ("const float* __restrict__", "Hx"),
    ("const float* __restrict__", "Hy"),
    ("const float* __restrict__", "Hz"),
    ("const float* __restrict__", "f_w_Hx"),
    ("const float* __restrict__", "f_w_Hy"),
    ("const float* __restrict__", "f_w_Hz"),
    ("const float* __restrict__", "Bx"),
    ("const float* __restrict__", "By"),
    ("const float* __restrict__", "Bz"),
    ("const float* __restrict__", "kps_x"),
    ("const float* __restrict__", "kps_y"),
    ("const float* __restrict__", "kps_z"),
    ("const float* __restrict__", "kms_x"),
    ("const float* __restrict__", "kms_y"),
    ("const float* __restrict__", "kms_z"),
)

#: Which KERNEL PARAMETER each pack member is built from. Three of them are not
#: spelled alike and that is deliberate rather than sloppy: the pre-launch magnetic
#: field arrives under the certified curl template's OWN names ``g0``/``g1``/``g2``
#: (which is what lets the lifted curl body be carried across character for
#: character), while the pack calls it ``Hx``/``Hy``/``Hz`` because that is what the
#: lifted CONSTITUTIVE body calls it. Assigning ``weld.Hx = Hx`` would name a
#: parameter this kernel does not have -- there is only ``Hx_out``, the SCRATCH -- and
#: the mapping is written out so the two spellings meet in exactly one place.
WELD_ARGS_SOURCE: Dict[str, str] = {
    "Hx": "g0", "Hy": "g1", "Hz": "g2",
}

#: The four anchored edits that make ``constitutive_apply`` pure, for the record and
#: for the host test that inverts them.
CONSTITUTIVE_LIFT_EDITS: Tuple[Tuple[str, str], ...] = (
    (_APPLY_SIGNATURE, _APPLY_SIGNATURE_PURE),
    (_APPLY_PARAMETERS, _APPLY_PARAMETERS_PURE),
    (_APPLY_FW_STORE, _APPLY_FW_STORE_PURE),
    (_APPLY_F_STORE, _APPLY_F_STORE_PURE),
)


def _needle(source: str, old: str, new: str, what: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(
            f"the certified text carries {count} copies of {what}, not one; this "
            f"product LIFTS that text rather than retyping it and cannot splice "
            f"around its absence")
    return source.replace(old, new, 1)


def _check_variant(variant: str) -> str:
    if variant not in KERNEL_NAMES:
        raise ValueError(
            f"variant must be one of {sorted(KERNEL_NAMES)}, got {variant!r}; a fold "
            f"handed to the plain text compiles, runs, and is wrong on one plane of "
            f"one component, so there is no default here")
    return variant


def _arm(expansion) -> int:
    return complex_emitter.normalized_expansion(expansion)


def kernel_name(variant: str) -> str:
    """The kernel this variant emits."""
    return KERNEL_NAMES[_check_variant(variant)]


def _curl_source(variant: str, arm: int) -> str:
    """The certified ``step_D`` source this variant welds -- plain or folded."""
    if _check_variant(variant) == "plain":
        return complex_emitter.complex_source("step_D", arm)
    return complex_folded_kernels.folded_source("step_D", arm)


def _split_at_kernel(source: str, what: str) -> Tuple[str, str]:
    """``(prelude, kernel)`` for one emitted sub-step source."""
    if source.count(_KERNEL_MARKER) != 1:
        raise AssertionError(
            f"{what} declares {source.count(_KERNEL_MARKER)} kernels, not one; this "
            f"weld splits its prelude from its body on that declaration")
    prelude, _, rest = source.partition(_KERNEL_MARKER)
    return prelude, _KERNEL_MARKER + rest


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

def constitutive_prelude(variant: str, expansion) -> str:
    """This variant's certified prelude with ``constitutive_apply`` made PURE.

    THE PRELUDE IS THE CURL SOURCE'S OWN, which is what carries the folded variant's
    two prelude deltas -- ``#define BC_MIRROR_PERIODIC 2`` and the widened ghost --
    into the weld without this module restating either. Four anchored edits on top,
    all of them in :data:`CONSTITUTIVE_LIFT_EDITS`; the arithmetic (``prev``, the two
    separate accumulations and their order) is untouched.

    ``cshift_up`` rides along unused: it is the certified TAIL's, lifting the tail
    whole is what keeps the shared mutation needles resolving, and NVRTC drops an
    unused ``__forceinline__`` helper.
    """
    prelude, _ = _split_at_kernel(_curl_source(variant, _arm(expansion)),
                                 f"the certified step_D source for variant {variant!r}")
    for old, new in CONSTITUTIVE_LIFT_EDITS:
        prelude = _needle(prelude, old, new, "constitutive_apply's " + (
            "declaration" if old is _APPLY_SIGNATURE else
            "parameter list" if old is _APPLY_PARAMETERS else
            "split-field store" if old is _APPLY_FW_STORE else "field store"))
    return prelude


def weld_args_struct() -> str:
    """The argument pack, declared."""
    lines = [
        "",
        "// THE PRE-LAUNCH STATE, packed once. Every member is read-only for the whole",
        "// launch: that is the property the recompute rests on, and binding H, f_w_H",
        "// and B const here is where it is stated in the type system rather than in a",
        "// comment. The pointers are float32 WORD VIEWS of complex64 volumes; cf_load",
        "// does the doubling and nothing else in this file does.",
        "struct WeldArgs {",
    ]
    for kind, name in WELD_ARGS_FIELDS:
        lines.append(f"    {kind} {name};")
    lines.append("};")
    return "\n".join(lines) + "\n"


def weld_args_construction(indent: str = "    ") -> str:
    """The struct, built from the kernel's own parameters. One line per member."""
    lines = [f"{indent}WeldArgs weld;"]
    for _kind, name in WELD_ARGS_FIELDS:
        lines.append(f"{indent}weld.{name} = {WELD_ARGS_SOURCE.get(name, name)};")
    return "\n".join(lines) + "\n"


def raw_update_H_cell_source(expansion) -> str:
    """The certified ``update_H_pml_complex_bloch`` body as a function of a CELL.

    THE BODY IS ARM-DEPENDENT AND VARIANT-INDEPENDENT, and both halves of that are
    load-bearing. Arm-dependent because ``constitutive_apply`` calls
    ``mul_coefficient_left``, whose text the arm decides. Variant-independent because
    ``complex_folded_kernels`` ships NO constitutive kernel: the folded rows are
    served at ``update_H`` by this very body, unchanged, which is the fact that lets
    one product cover both cells.

    Only the thread preamble and the three ``constitutive_apply`` call lines are
    edited, each through an exact anchor. Both results come back through
    out-parameters: ``h_out[c]`` is the stepped magnetic field and ``w_out[c]`` the
    split-field history, which the caller stores to scratch for its own cell and
    discards for a foreign recompute.
    """
    arm = _arm(expansion)
    prelude, kernel = _split_at_kernel(
        complex_emitter.complex_source("update_H", arm),
        "the certified update_H source")
    del prelude
    if kernel.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            "the certified update_H signature terminator is not unique; this weld "
            "cuts its body on that anchor")
    body = kernel.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified update_H body does not end with a brace")
    body = body[: -len("}\n")]
    body = _needle(
        body, _THREAD_PREAMBLE,
        "    // THE CELL ARRIVES AS A FLAT INDEX. The bounds guard left with the thread\n"
        "    // index: every caller is either the kernel's own guarded thread or an\n"
        "    // index the certified cshift_dn composed, which that helper's own branch\n"
        "    // keeps inside the volume. THE DECODE BELOW STAYS -- it is what indexes\n"
        "    // the absorber profile at the recomputed cell rather than at the\n"
        "    // thread's own.\n",
        "the constitutive body's thread preamble")
    if body.count(_INDEX_DECODE) != 1:
        raise AssertionError(
            f"the certified update_H body carries {body.count(_INDEX_DECODE)} copies "
            f"of the index decode, not one; this weld keeps that block verbatim and "
            f"cannot vouch for a changed one")
    for component, axis in enumerate("xyz"):
        coordinate = "ijk"[component]
        old = (f"    constitutive_apply(f{component}, w{component}, idx, "
               f"s{component}, kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);\n")
        new = (f"    h_out[{component}] = constitutive_apply_pure(f{component}, "
               f"w{component}, idx, s{component}, kps_{axis}[{coordinate}], "
               f"kms_{axis}[{coordinate}], &w_out[{component}]);\n")
        body = _needle(body, old, new,
                       f"component {component}'s certified constitutive_apply call")
    if "constitutive_apply(" in body:
        raise AssertionError(
            "a storing constitutive_apply call survived the capture; this weld writes "
            "nothing in place on the magnetic side")
    unpack = [
        "    // The certified body's own names, bound to the pack.",
        "    const int nx = weld.nx; const int ny = weld.ny;",
        "    const int nz = weld.nz;",
        "    (void) nx;   // the removed bounds guard's operand",
        "    const float* __restrict__ f0 = weld.Hx;",
        "    const float* __restrict__ f1 = weld.Hy;",
        "    const float* __restrict__ f2 = weld.Hz;",
        "    const float* __restrict__ w0 = weld.f_w_Hx;",
        "    const float* __restrict__ w1 = weld.f_w_Hy;",
        "    const float* __restrict__ w2 = weld.f_w_Hz;",
        "    // The H source is Bx/By/Bz DIRECTLY (stepping.py:921): mu = 1 is baked",  # stepping.py live lines for the frozen device-text citation(s) in this string: 921->949
        "    // into H_CONSTITUTIVE_TERMS and there is no permeability volume here.",
        "    const float* __restrict__ g0 = weld.Bx;",
        "    const float* __restrict__ g1 = weld.By;",
        "    const float* __restrict__ g2 = weld.Bz;",
    ]
    for stem in ("kps", "kms"):
        for axis in "xyz":
            unpack.append(
                f"    const float* __restrict__ {stem}_{axis} = weld.{stem}_{axis};")
    return ("\n// The certified update_H_pml_complex_bloch, evaluated at ONE ARBITRARY\n"
            "// CELL and storing nothing. This is the function the curl half calls for\n"
            "// every foreign tap it needs: a pure function of H, f_w_H and B, none of\n"
            "// which this launch writes, so a foreign evaluation cannot depend on\n"
            "// which block ran first.\n"
            "__device__ __forceinline__ void raw_update_H_cell(\n"
            "    int idx, const WeldArgs& weld, cf* h_out, cf* w_out\n"
            ") {\n" + "\n".join(unpack) + "\n" + body + "}\n")


def resolution_source() -> str:
    """``resolve_H`` -- ``update_H``'s result for ONE component at any cell.

    A thin selector over :func:`raw_update_H_cell_source`, so the thread's own cell
    and every foreign recompute go through ONE code path and cannot disagree.
    ``comp`` is a compile-time constant at every call site, so the selection folds and
    the other two components' arithmetic is dead-code eliminated.
    """
    return "\n".join([
        "",
        "// update_H's stepped magnetic field for ONE component at ONE cell. The",
        "// component tag is a compile-time constant at every call site.",
        "__device__ __forceinline__ cf resolve_H(int comp, int idx,",
        "                                       const WeldArgs& weld) {",
        "    cf h[3];",
        "    cf w[3];",
        "    raw_update_H_cell(idx, weld, h, w);",
        "    return h[comp];",
        "}",
        "",
    ])


def _balanced_call(text: str, opener: str) -> Tuple[str, str, str]:
    """``(head, argument text, tail)`` for the first ``opener(`` call in ``text``.

    The argument text is taken with BALANCED PARENTHESES rather than to the first
    ``)``: the certified wrap load spells its index ``idx + (na - 1) * stride``, and
    a scan that stopped at the first closing bracket would cut it in half and emit a
    kernel that reads a different cell.
    """
    start = text.index(opener)
    cursor = start + len(opener)
    depth = 1
    while cursor < len(text) and depth:
        if text[cursor] == "(":
            depth += 1
        elif text[cursor] == ")":
            depth -= 1
            if depth == 0:
                break
        cursor += 1
    if depth:
        raise AssertionError(
            f"the certified text opens {opener!r} and never closes it; the index "
            f"expression this weld carries across cannot be read off it")
    return text[:start], text[start + len(opener):cursor], text[cursor + 1:]


def cshift_dn_recompute_source(variant: str, expansion) -> str:
    """The certified ``cshift_dn`` with its two leaf LOADS RESOLVED.

    THE BRANCHES AND THE INDEX EXPRESSIONS ARE THE CERTIFIED HELPER'S OWN, parsed out
    of the prelude this variant emits rather than retyped: the near-neighbour branch,
    the ghost return, the wrap arithmetic, the ``ph`` guard and the
    ``rotate_field_left`` call are carried across character for character, and only
    ``cf_load(g, EXPR)`` becomes ``resolve_H(comp, EXPR, weld)``.

    That the FOLDED variant's widened ghost (``bc == BC_METALLIC || bc ==
    BC_MIRROR_PERIODIC``) arrives here automatically is the whole reason this function
    reads the variant's own prelude instead of ``complex_emitter._TAIL``: a folded
    ghost that recomputed a magnetic field instead of returning ``cf_zero()`` would be
    wrong on one plane of one component and would converge.
    """
    prelude, _ = _split_at_kernel(_curl_source(variant, _arm(expansion)),
                                 f"the certified step_D source for variant {variant!r}")
    if prelude.count(_CSHIFT_DN_SIGNATURE) != 1:
        raise AssertionError(
            f"the certified prelude carries {prelude.count(_CSHIFT_DN_SIGNATURE)} "
            f"copies of cshift_dn's declaration, not one; the recompute has no anchor")
    body = prelude.split(_CSHIFT_DN_SIGNATURE, 1)[1]
    if not body.startswith(" {\n"):
        raise AssertionError(
            "the certified cshift_dn no longer opens its body on the declaration's "
            "own line; this weld lifts that body and cannot find it")
    body = body[len(" {\n"):]
    end = body.find("}\n")
    if end < 0:
        raise AssertionError("the certified cshift_dn does not close")
    # EVERYTHING PAST THE CLOSING BRACE IS DROPPED, and dropping it is load-bearing
    # rather than tidy: the rest of the prelude is ``pml_apply`` and
    # ``constitutive_apply``, which :func:`constitutive_prelude` already emits.
    # Carrying them a second time is a duplicate definition NVRTC refuses outright.
    body = body[:end]
    if body.count("cf_load(g,") != 2:
        raise AssertionError(
            f"the certified cshift_dn makes {body.count('cf_load(g,')} loads of g, "
            f"not 2; this weld resolves exactly the near-neighbour load and the "
            f"periodic wrap")
    for _ in range(2):
        head, expression, tail = _balanced_call(body, "cf_load(g,")
        body = f"{head}resolve_H(comp,{expression}, weld){tail}"
    if "g[" in body or "(g," in body or " g," in body:
        raise AssertionError(
            "the resolved cshift_dn still reads the pointer g; in this helper there "
            "is no such parameter and every magnetic read must be a recompute")
    return ("\n// stepping._shift_down's certified device helper, with the leaf loads\n"
            "// RESOLVED: the value the curl needs at a backward neighbour is not a\n"
            "// stored word but update_H's result there, recomputed from pre-launch\n"
            "// state. The branch, the ghost return, the wrap arithmetic and the Bloch\n"
            "// rotation ON THE WRAPPED LANE ONLY are the certified helper's own -- the\n"
            "// recompute carries no phase of its own and none is applied to the\n"
            "// near-neighbour branch.\n"
            + _CSHIFT_DN_SIGNATURE_RECOMPUTE + " {\n" + body + "}\n")


def curl_pieces(variant: str, expansion) -> Tuple[str, str]:
    """``(head, tail)`` of the certified ``step_D`` body, every H read redirected.

    ``head`` is the certified preamble, the stride constants, the index decode and the
    three phase constants, lifted VERBATIM. ``tail`` is the three target blocks with
    the six own-cell magnetic loads taken from registers and the six shifted ones
    recomputed. The ownership mask -- ``complex_emitter``'s on the plain variant and
    ``complex_folded_kernels.fold_mask_lines``' three-arm block on the folded one --
    rides in ``tail`` untouched: it zeroes ``curl``, never a magnetic read.
    """
    _check_variant(variant)
    source = _curl_source(variant, _arm(expansion))
    _, kernel = _split_at_kernel(
        source, f"the certified step_D source for variant {variant!r}")
    if kernel.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            "the certified step_D signature terminator is not unique")
    body = kernel.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified step_D body does not end with a brace")
    body = body[: -len("}\n")]
    if body.count(_THREAD_PREAMBLE) != 1 or body.count(_INDEX_DECODE) != 1:
        raise AssertionError(
            "the certified step_D body no longer carries exactly one thread preamble "
            "and one index decode; this weld keeps both verbatim")
    if body.count(_FIRST_TARGET) != 1:
        raise AssertionError(
            f"the certified step_D body carries {body.count(_FIRST_TARGET)} target-0 "
            f"markers, not one; this weld splices its constitutive half directly "
            f"above the first target block")
    cut = body.index(_FIRST_TARGET)
    head, tail = body[:cut], body[cut:]

    own = 0
    for component, pointer in enumerate(CURL_SOURCE_NAMES):
        for variable in ("f_1", "f_2"):
            old = f"        cf {variable} = cf_load({pointer}, idx);\n"
            if old not in tail:
                continue
            tail = _needle(tail, old,
                           f"        cf {variable} = own_h[{component}];\n",
                           f"{pointer}'s own-cell load into {variable}")
            own += 1
    if own != OWN_LOAD_EDITS:
        raise AssertionError(
            f"the certified step_D body makes {own} own-cell magnetic loads of the "
            f"shape this weld redirects, not {OWN_LOAD_EDITS}; a load left standing "
            f"would read the PRE-launch H")

    taps = 0
    for line in [line + "\n" for line in tail.splitlines() if "cshift_dn(" in line]:
        line_head, _, rest = line.partition("cshift_dn(")
        pointer, _, arguments = rest.partition(", ")
        if pointer not in CURL_SOURCE_NAMES:
            raise AssertionError(
                f"the certified step_D body reads {pointer!r} through cshift_dn; this "
                f"weld resolves exactly the three magnetic sources "
                f"{CURL_SOURCE_NAMES}")
        if not arguments.rstrip("\n").endswith(");"):
            raise AssertionError(
                f"the certified cshift_dn call {line!r} does not close on its own "
                f"line; the weld appends its argument pack to that closing line")
        component = CURL_SOURCE_NAMES.index(pointer)
        carried = arguments.rstrip("\n")[: -len(");")]
        tail = _needle(
            tail, line,
            f"{line_head}cshift_dn_recompute({component}, {carried}, weld);\n",
            f"the shifted magnetic load {line.strip()!r}")
        taps += 1
    if taps != HALO_TAPS:
        raise AssertionError(
            f"the certified step_D body makes {taps} shifted magnetic loads, not "
            f"{HALO_TAPS}; this weld redirects every one of them and a missed tap "
            f"would read a word another block is writing")
    for pointer in CURL_SOURCE_NAMES:
        if f"cf_load({pointer}," in tail or f"({pointer}," in tail:
            raise AssertionError(
                f"the welded curl half still reads {pointer}; in this signature that "
                f"pointer is the PRE-LAUNCH magnetic field and every read must be the "
                f"register or a recompute")
    if "cshift_dn(" in tail:
        raise AssertionError(
            "the welded curl half still calls the certified cshift_dn, which loads a "
            "stored H; every shifted tap must go through cshift_dn_recompute")
    return head, tail


#: The signature, per variant. Spelled OUT rather than formatted from
#: :data:`KERNEL_NAMES` so that ``test_kernel_partition.py``'s strict declaration
#: regex -- which requires a C identifier after ``void`` -- can read both shipped
#: kernel names off this file without importing it.
_SIGNATURE_HEAD: Dict[str, str] = {
    "plain": '\nextern "C" __global__ void fused_hd_pair_pml_complex_bloch(\n',
    "folded": '\nextern "C" __global__ void fused_hd_pair_pml_complex_folded(\n',
}

_SIGNATURE_BODY = '''    // THE SCRATCH OUTPUTS (float32 word views of complex64 volumes). A DIFFERENT
    // ALLOCATION from the pre-launch group below -- assert_bindings_are_disjoint
    // checks it by base address before every launch -- and that separation is the
    // whole design: nothing this launch reads is ever a word this launch wrote, so
    // the curl half's foreign taps are a pure function of unwritten memory rather
    // than a race with another block.
    float* __restrict__ Hx_out, float* __restrict__ Hy_out,
    float* __restrict__ Hz_out,
    float* __restrict__ f_w_Hx_out, float* __restrict__ f_w_Hy_out,
    float* __restrict__ f_w_Hz_out,
    // D AND ITS SPLIT-FIELD AUXILIARY, written IN PLACE -- safe by construction,
    // since pml_apply reads and writes f[idx] and fu[idx] at the THREAD'S OWN CELL
    // only and no thread reads a displacement another thread wrote. The names are
    // the certified curl template's own (f0/f1/f2, u0/u1/u2), so the lifted body
    // below arrives character for character.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ u0, float* __restrict__ u1, float* __restrict__ u2,
    // THE PRE-LAUNCH MAGNETIC FIELD, read at ANY cell by ANY thread and written by
    // none. g0/g1/g2 are the curl template's names for Hx/Hy/Hz; after the weld the
    // curl half reads NEITHER -- its own cell comes from a register and every
    // shifted tap from a recompute -- and curl_pieces asserts exactly that.
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    // The rest of the pre-launch constitutive state: the split-field history and
    // the source B.
    const float* __restrict__ f_w_Hx, const float* __restrict__ f_w_Hy,
    const float* __restrict__ f_w_Hz,
    const float* __restrict__ Bx, const float* __restrict__ By,
    const float* __restrict__ Bz,
    // Extents in COMPLEX CELLS. cf_load does the word doubling.
    int nx, int ny, int nz, float dtdx,
    // The D curl's INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // update_H's kps. Its kms IS the curl's kms above and is bound ONCE: step_D
    // reads the INTEGER sub-lattice (complex_emitter.HALF_INTEGER["step_D"] is
    // False) and update_H reads the INTEGER one too (HALF_INTEGER["H"] is False),
    // so the two groups are the same three volumes. There is no kms_*_h here and
    // there must not be -- :func:`resolve` ASSERTS the identity by base address
    // rather than assuming it, because binding the half-integer set instead
    // compiles and is a half-cell error in the absorber profile.
    const float* __restrict__ kps_x, const float* __restrict__ kps_y,
    const float* __restrict__ kps_z,
    int bc_x, int bc_y, int bc_z,
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi
) {
'''


def signature(variant: str) -> str:
    """The kernel's parameter list, for one variant."""
    return _SIGNATURE_HEAD[_check_variant(variant)] + _SIGNATURE_BODY


def kernel_source(variant: str, expansion) -> str:
    """The whole device source for one variant under one arm."""
    head, tail = curl_pieces(variant, expansion)
    weld = "\n".join([
        "",
        weld_args_construction().rstrip("\n"),
        "",
        "    // --- THE WELD: update_H, computed into registers and stored to SCRATCH.",
        "    // Nothing written here is read by this launch. The curl half below takes",
        "    // its OWN cell's magnetic field from these registers and recomputes",
        "    // every foreign tap from PRE-LAUNCH state through the same",
        "    // raw_update_H_cell, so no thread observes another thread's store; the",
        "    // launcher rotates H/f_w_H against the scratch afterwards.",
        "    cf own_h[3];",
        "    cf own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    cf_store(Hx_out, idx, own_h[0]);",
        "    cf_store(Hy_out, idx, own_h[1]);",
        "    cf_store(Hz_out, idx, own_h[2]);",
        "    cf_store(f_w_Hx_out, idx, own_w[0]);",
        "    cf_store(f_w_Hy_out, idx, own_w[1]);",
        "    cf_store(f_w_Hz_out, idx, own_w[2]);",
        "",
        "    // --- step_D (stepping.step_D / _apply_pml_update), the certified body",
        "    // from its first target block down, with the twelve magnetic reads",
        "    // redirected and NOTHING else touched.",
        "",
    ])
    return (constitutive_prelude(variant, expansion) + weld_args_struct()
            + raw_update_H_cell_source(expansion) + resolution_source()
            + cshift_dn_recompute_source(variant, expansion)
            + signature(variant) + head + weld + tail + "}\n")


def device_sources() -> Dict[str, str]:
    """``{kernel name: source}`` -- the shape every family on this track publishes.

    The FMA_V1 arm, which is the one every corpus row of both cells licenses on this
    hardware; the NAIVE arm's texts are reachable through :func:`kernel_source` and
    the gate compiles both.
    """
    return {KERNEL_NAMES[variant]: kernel_source(variant, "FMA_V1")
            for variant in VARIANTS}


def source_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Two variants times two arms is four sources; a single changed character here, in
    ``complex_emitter`` or in ``complex_folded_kernels`` moves this value.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for variant in sorted(KERNEL_NAMES):
        for name in sorted(complex_emitter.EXPANSIONS):
            digest.update(f"{variant}|{name}".encode("ascii"))
            digest.update(kernel_source(variant, name).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The compile memo
# ---------------------------------------------------------------------------

#: The mutable copy the launcher compiles, keyed by ``(variant, arm)``. The gate
#: mutates through :func:`set_kernel_source`; ONE seam, so a mutation cannot miss a
#: second copy.
_SOURCES: Dict[Tuple[str, int], str] = {}


def current_source(variant: str, expansion) -> str:
    """The device text this family would compile now -- the mutated one if any."""
    key = (_check_variant(variant), _arm(expansion))
    if key not in _SOURCES:
        _SOURCES[key] = kernel_source(*key)
    return _SOURCES[key]


def set_kernel_source(variant: str, expansion, source: str) -> None:
    """Replace one variant's device text -- the gate's mutation seam, and only that."""
    _SOURCES[(_check_variant(variant), _arm(expansion))] = source


def reset_kernel_sources() -> int:
    """Drop every mutated body; returns how many entries went. The gate's undo."""
    count = len(_SOURCES)
    _SOURCES.clear()
    return count


def _get_kernel(variant: str, expansion, source: Optional[str] = None,
                options: Optional[Tuple[str, ...]] = None):
    """Compile on first use, memoized in the PACKAGE'S SHARED MEMO.

    THE SHARED MEMO AND NOT A PRIVATE DICT, which is a measurement property rather
    than tidiness: every gate on this track counts launches by wrapping
    ``compile_cache``'s memo, and a family that kept its own dict would be INVISIBLE
    to that counter -- its launches would read as zero and a launch-structure leg
    would pass a weld that never executed. Measured on this product's first smoke
    (2026-09-07): ``memo_launches_per_step`` came back 0.0 against an arrangement
    tally of 1.0.

    THE SOURCE AND THE OPTIONS ARE IN THE KEY, so a mutation harness that handed in a
    rewritten string cannot be served the shipped binary, the contraction-guard
    control cannot be served the guarded build, and a late policy install cannot be
    served an earlier one. ``cupy`` is imported HERE, never at scope.
    """
    import cupy as cp  # noqa: PLC0415 - device-only, in a laptop-importable module

    variant = _check_variant(variant)
    arm = _arm(expansion)
    code = current_source(variant, arm) if source is None else source
    build = _COMPILE_OPTIONS if options is None else tuple(options)
    name = KERNEL_NAMES[variant]
    key = _kernel_cache_key(name, True, build, code)
    return _get_or_compile(key, lambda: cp.RawKernel(code, name, options=build))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# ---------------------------------------------------------------------------
# The variant, resolved from the grid
# ---------------------------------------------------------------------------

def variant_for(grid: Any) -> str:
    """``"folded"`` if any axis carries a mirror fold, else ``"plain"``.

    READ FROM THE GRID, never chosen by a caller. ``Grid.is_mirrored`` is the same
    reader ``complex_folded_kernels._fold_reasons`` and
    ``coverage._complex_grid_refusal`` use, so the three cannot disagree about what a
    fold is.
    """
    reader = getattr(grid, "is_mirrored", None)
    if not callable(reader):
        raise ValueError(
            "grid does not expose is_mirrored; this product cannot infer from a grid "
            "without it that no axis is folded, and a fold handed to the plain text "
            "is wrong on one plane of one component")
    return "folded" if any(bool(reader(axis)) for axis in range(3)) else "plain"


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def covers_complex_fused_hd_pair(fields: Any, pml: Any, grid: Any,
                                 sources: Any = None, license: Any = None,
                                 subnormal_policy: Any = None) -> Tuple[bool, str]:
    """May this product span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A CONJUNCTION OF THE TWO CERTIFIED HALVES' OWN PREDICATES plus this seam's
    withdraw clause and the rotation's six volumes -- never a re-derivation of either
    half's clause set, which is what keeps this product from admitting a
    configuration a half refuses:

    * the H side: ``coverage.covers_real_pml_complex_constitutive(..., "H")``, asked
      FIRST because it asks the expansion licence first of all;
    * the D side: ``coverage.covers_real_pml_complex_curl(..., "step_D")`` on the
      plain variant, ``complex_folded_kernels.covers_complex_folded_curl(...,
      "step_D")`` on the folded one -- SELECTED BY THE GRID through
      :func:`variant_for`, so the text that would be compiled and the predicate that
      admits it are decided by the same reading;
    * the seam: ``withdraw_hoist.seam_withdraw_reasons``, which refuses BY NAME every
      row carrying a standing integrated electric withdraw;
    * the rotation: the six volumes :data:`SCRATCH_VOLUMES` names.

    THE H SIDE'S PREDICATE ADMITS MORE THAN THIS PRODUCT SERVES -- it admits a Dcyl
    grid, which the curl half refuses -- and that is fine and deliberate: the
    conjunction takes the INTERSECTION, and the cylindrical H->D products are where a
    Dcyl row is served.
    """
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "H", license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"
    try:
        variant = variant_for(grid)
    except ValueError as error:
        return False, f"variant: {error}"
    if variant == "plain":
        covered, reason = covers_real_pml_complex_curl(
            fields, pml, grid, "step_D", license, subnormal_policy)
    else:
        covered, reason = complex_folded_kernels.covers_complex_folded_curl(
            fields, pml, grid, "step_D", license, subnormal_policy)
    if not covered:
        return False, f"curl half ({variant}): {reason}"
    seam_reasons = _withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False, so nothing would perform the withdraw "
            f"before its launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES)
    if seam_reasons:
        return False, seam_reasons[0]
    # THE BOUNDARY RESOLUTION THE LAUNCH BINDS, asked here rather than left to raise
    # at launch time: on the folded variant it is the fold-termination cross-check
    # that decides BC_METALLIC against BC_MIRROR_PERIODIC, and a grid it cannot
    # resolve is a refusal with a line behind it.
    try:
        if variant == "folded":
            _codes, refusal = complex_folded_kernels.folded_complex_boundary_codes(grid)
            if refusal is not None:
                return False, f"boundary resolution ({variant}): {refusal}"
        elif complex_pml_kernels is not None:
            complex_pml_kernels.complex_boundary_codes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise here is a refusal
        return False, (f"the curl's boundary resolution refuses this grid: "
                       f"{type(exc).__name__}: {exc}")
    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this product rotates it against "
                           f"a launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def _word_view(array: Any) -> Any:
    if complex_pml_kernels is None:  # pragma: no cover - laptop guard
        raise RuntimeError("complex_pml_kernels did not import; there is no device here")
    return complex_pml_kernels.word_view(array)


def resolve(fields: Any, grid: Any, pml: Any, expansion, *,
            variant: Optional[str] = None,
            tables: Optional[Dict[str, Any]] = None,
            constitutive: Optional[Dict[str, Any]] = None,
            boundary_codes: Optional[Any] = None,
            phase_flags: Optional[Any] = None,
            phase_values: Optional[Any] = None,
            scratch: Optional[Dict[str, Any]] = None,
            dtdx: Optional[float] = None) -> Dict[str, Any]:
    """Everything the launch needs, derived once per frozen configuration.

    DERIVED HERE BY THE SAME FUNCTIONS THE CERTIFIED LAUNCHERS ASK, which is what
    makes "the weld and the singles cannot disagree about which sub-lattice, which
    boundary code or which phase this seam reads" an enforced property rather than a
    convention. THE OVERRIDES ARE THE GATE'S DOOR and stay, keyword-only and named for
    what they are: a gate feeds deliberately wrong tables (the swapped sub-lattice),
    wrong boundary codes (the dropped metallic wall) and wrong phases (the
    unconjugated backward factor).

    THE kms IDENTITY IS ASSERTED, NOT ASSUMED. ``step_D`` and ``update_H`` both read
    the INTEGER sub-lattice, so the curl's ``kms_*`` and the constitutive's are the
    same three device allocations and the signature binds them once. Binding the
    half-integer set instead COMPILES and is a half-cell error in the absorber
    profile, so the two are compared by base address here and a mismatch RAISES --
    except where the caller supplied one of the two deliberately, which is the
    mutation the gate arms.
    """
    import cupy as cp  # noqa: PLC0415 - device-only

    if complex_pml_kernels is None:  # pragma: no cover - laptop guard
        raise RuntimeError("complex_pml_kernels did not import; there is no device here")
    arm = _arm(expansion)
    variant = _check_variant(variant if variant is not None else variant_for(grid))
    supplied = tables is not None or constitutive is not None
    if tables is None:
        tables = complex_pml_kernels.complex_curl_tables(
            pml, complex_emitter.HALF_INTEGER["step_D"])
    if constitutive is None:
        constitutive = complex_pml_kernels.complex_constitutive_tables(
            pml, complex_emitter.HALF_INTEGER["H"])
    if not supplied:
        for axis in "xyz":
            if int(tables[f"kms_{axis}"].data.ptr) != int(
                    constitutive[f"kms_{axis}"].data.ptr):
                raise ValueError(
                    f"the curl's kms_{axis} and the constitutive's are different "
                    f"allocations; step_D and update_H both read the INTEGER Yee "
                    f"sub-lattice and this signature binds that group ONCE, so a "
                    f"disagreement here is a half-cell error in the absorber profile")
    if boundary_codes is None:
        if variant == "folded":
            # ``folded_complex_boundary_codes`` returns ``(codes, refusal)`` with
            # exactly one of them None -- it carries the fold-termination cross-check
            # (``Grid.stored_cells > Grid.owned_cells`` against ``Grid.is_metallic``)
            # that decides BC_METALLIC against BC_MIRROR_PERIODIC on a folded axis.
            # Getting that wrong is a top-plane mask applied where MEEP steps the
            # plane; a refusal here is RAISED rather than dropped on the floor.
            boundary_codes, refusal = (
                complex_folded_kernels.folded_complex_boundary_codes(grid))
            if refusal is not None:
                raise ValueError(
                    f"the folded curl's boundary resolution refuses this grid: "
                    f"{refusal}")
        else:
            boundary_codes = complex_pml_kernels.complex_boundary_codes(grid)
    derived_flags, derived_values = complex_pml_kernels.bloch_phase_arguments(
        grid, complex_emitter.KERNELS["step_D"][1])
    if phase_flags is None:
        phase_flags = derived_flags
    if phase_values is None:
        phase_values = derived_values
    if dtdx is None:
        dtdx = float(grid.dt / grid.dx)  # stepping.py:314, :431.
    if scratch is None:
        scratch = {}
        for name in SCRATCH_VOLUMES:
            source = getattr(fields, name, None)
            if source is None:
                raise ValueError(
                    f"fields.{name} is not allocated; this weld rotates a scratch "
                    f"shaped like it and has nothing to shape one from")
            # UNINITIALIZED IS CORRECT: every cell of all six is written by every
            # launch, so there is no cell whose prior contents a reader could observe.
            scratch[name] = cp.empty_like(source)
    return {"variant": variant, "arm": arm, "tables": tables,
            "constitutive": constitutive, "boundary_codes": tuple(boundary_codes),
            "phase_flags": tuple(phase_flags), "phase_values": tuple(phase_values),
            "dtdx": float(dtdx), "scratch": scratch, "launches": 0}


def assert_bindings_are_disjoint(fields: Any, state: Dict[str, Any]) -> int:
    """Every pointer this launch binds ``__restrict__``, checked by base address."""
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in SCRATCH_VOLUMES:
        visit(f"{name}_out", state["scratch"][name])
    for name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz",
                 "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"):
        visit(name, getattr(fields, name))
    for axis in "xyz":
        visit(f"kms_{axis}", state["tables"][f"kms_{axis}"])
        visit(f"sinv_{axis}", state["tables"][f"sinv_{axis}"])
        visit(f"kps_{axis}", state["constitutive"][f"kps_{axis}"])
    if collisions:
        raise ValueError(
            "this product binds every field, scratch and table argument "
            "__restrict__, and these arguments alias: " + "; ".join(collisions))
    return len(bound)


def launch_complex_fused_hd_pair(fields: Any, state: Dict[str, Any],
                                 kernel: Optional[Any] = None,
                                 threads: int = _FUSED_THREADS) -> Dict[str, Any]:
    """THE ONE LAUNCH. ``kernel`` and ``threads`` are the gate's doors."""
    import numpy as np  # noqa: PLC0415 - device-only path

    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + threads - 1) // threads
    tables, constitutive = state["tables"], state["constitutive"]
    arguments: List[Any] = [_word_view(state["scratch"][name])
                            for name in SCRATCH_VOLUMES]
    arguments += [_word_view(getattr(fields, name))
                  for name in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
                               "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                               "Bx", "By", "Bz")]
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz),
                  np.float32(state["dtdx"])]
    for axis in "xyz":
        arguments += [tables[f"kms_{axis}"], tables[f"sinv_{axis}"]]
    arguments += [constitutive[f"kps_{axis}"] for axis in "xyz"]
    arguments += [np.int32(code) for code in state["boundary_codes"]]
    arguments += [np.int32(flag) for flag in state["phase_flags"]]
    arguments += [np.float32(value) for value in state["phase_values"]]
    launcher = kernel or _get_kernel(state["variant"], state["arm"])
    launcher((blocks,), (threads,), tuple(arguments))
    state["launches"] += 1
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "variant": state["variant"],
            "kernel": KERNEL_NAMES[state["variant"]],
            "arm": complex_emitter.EXPANSION_NAMES[state["arm"]]}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION -- ``fused_hd_pair.rotate_into_fields``, reused verbatim.

    The consumer audit it rests on is that module's, and it is a statement about the
    ENGINE rather than about a storage class: every consumer of the magnetic field
    resolves it BY NAME at use time, so a rebinding between steps is invisible to all
    of them. Calling it here rather than re-spelling it is what keeps the two products
    from ever disagreeing about what a rotation covers.
    """
    return _hd.rotate_into_fields(fields, scratch)


def run_complex_fused_hd_pair(fields: Any, grid: Any, pml: Any, expansion, *,
                              sources: Any = None, license: Any = None,
                              subnormal_policy: Any = None,
                              state: Optional[Dict[str, Any]] = None,
                              check: bool = True,
                              rotate: bool = True) -> Dict[str, Any]:
    """The whole product: the predicate, the launch, the rotation.

    Returns the launch record with ``rotated`` and the state the caller should keep.
    """
    if check:
        covered, reason = covers_complex_fused_hd_pair(
            fields, pml, grid, sources, license, subnormal_policy)
        if not covered:
            raise ValueError(f"this product does not cover this configuration: {reason}")
    if state is None:
        state = resolve(fields, grid, pml, expansion)
    assert_bindings_are_disjoint(fields, state)
    record = launch_complex_fused_hd_pair(fields, state)
    if rotate:
        state["scratch"] = rotate_into_fields(fields, state["scratch"])
    record["rotated"] = bool(rotate)
    record["replaces"] = REPLACES
    record["state"] = state
    return record
