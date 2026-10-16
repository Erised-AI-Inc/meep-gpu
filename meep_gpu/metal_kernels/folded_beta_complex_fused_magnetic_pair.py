"""The FOLDED BETA COMPLEX fused Metal pair: a folded ``special_kz`` Bloch ``step_B``
welded into ``update_H``.

:mod:`.folded_complex_fused_magnetic_pair` WITH MEEP'S BETA PASS CARRIED IN, and the
pass is the whole module. The source is the certified weld's own output
(:func:`.folded_complex_fused_magnetic_pair.folded_complex_fused_magnetic_pair_source`)
with :func:`.special_kz.beta_pass` inserted in register by
:func:`.folded_pass.insert_split_pass`. MEEP runs the beta pass after every curl
(step_db.cpp:161-175) through the additive ladder (``stepping._apply_folded_increment``),
and the pass reads the curl's partner fields only, so the weld's lines are untouched and
only the pass is added: the near mirror carry, the far ghost carry, the parity chain,
the wall clear, the ghost coefficient reload and the constitutive blocks are the weld's
own. :mod:`.beta_fused_magnetic_pair` carries the same pass on the REAL unfolded seam.

This is the cell ``(folded beta complex, folded beta complex)`` at ``B_to_H``, and it
is the one B/H cell neither shipped folded weld can stand in for.
:mod:`.folded_complex_fused_magnetic_pair` refuses ``grid.beta`` through both halves'
own clause (``folded_complex._folded_complex_grid_reasons`` clause 8, which names "the
folded beta product" as the owner); :mod:`.beta_fused_magnetic_pair` refuses a fold by
name ("axis {a} is folded: stepping.fill_symmetry_bc_B and
stepping.fill_folded_far_ghosts_B both run inside this seam and neither is carried by
this family") and is real-storage besides. This module is the product they name.

===========================================================================
WHAT IT REACHES, AND WHAT IT CANNOT
===========================================================================

The cell is driven by THREE corpus rows and only TWO of them are reachable:

* ``tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2``
* ``tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7``
* ``tests:TestSpecialKz.test_eigsrc_kz_0_complex`` — **UNREACHABLE ON ANY BACKEND**.
  It declares a MAGNETIC source, which ``driver.step`` deposits BETWEEN ``step_B`` and
  ``update_H`` (driver.py:3292-3293). No launch can straddle a deposit, so the clause
  is a driver fact rather than this kernel's scope, and it is refused by name below.

**THE ROW COUNT IS NOT RE-DERIVED HERE AND THIS MODULE CLAIMS NO CELL MOVE.** The Metal
census that placed those rows finished 2026-08-19 02:15 and ``folded_beta.py`` was
edited at 05:16; the re-close round ``results/metal_coverage_special_kz_reclose_2026-08-19``
re-ran ``TestSpecialKz.test_special_kz`` only, not these two rows. The standing board
(``results/fusion_matrix_metal_2026-08-26_canonical``) could bound that staleness
because NO fused product covered any folded-beta cell; this product is the first that
would, so the +2 is a MATRIX RUN this module does not perform and does not assert. What
is asserted here is the predicate and the binding arithmetic, both of which are
measured on every build.

The two in-seam fills are LIVE on both reachable rows and are therefore CARRIED, not
refused: ``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B``. ``zero_metal_B`` is
NOT live on either, and is carried anyway — the emitter compiles it out on an unwalled
grid, and leaving it uncarried would make the walled specialisation a silent gap rather
than a compiled-out block.

===========================================================================
THE FILL CARRY IS THE RISK, AND IT IS TAKEN DELIBERATELY
===========================================================================

:mod:`.folded_beta` DECLINES to claim ``fill_B`` and ``fill_folded_far_ghosts_B``
(folded_beta.py:46-53) — "registering a third fill arm here would make both fill slots
AMBIGUOUS and drop them to the array path, which is a coverage LOSS dressed as
completeness". That decision is about an ARM TABLE ROW, not about the arithmetic: the
fills copy ``parity * field[plane]`` and touch no beta term, so
:mod:`.folded_complex`'s fill predicate already admits a folded beta run and its
composition matrix has pinned exactly that for ``fold_complex_2d_beta`` since before
that family existed.

So the arithmetic this weld carries inline for those two passes is
:func:`.folded_complex.folded_mirror_fill_complex_source`'s own — reached through
:func:`.folded_complex_fused_magnetic_pair._carry_blocks`, which is the shipped weld's
emitter and is CALLED rather than copied. This product takes over two driver passes its
own curl family declines to own an ARM for, and it does so through the emitter of the
family that does own them. Registering unwired (below) is what keeps that from
disturbing either arm table.

===========================================================================
THE SIGNATURE — 27 pointers plus one packed struct, unmoved
===========================================================================

    3 B  +  3 fu_B  +  3 E   (float2, the curl half)
  + 3 H  +  3 f_w_H          (float2, the constitutive half)
  + 6 curl coefficients  +  6 constitutive coefficients   (float32)
  = 27 pointers

:data:`PACKED_BINDINGS` is IMPORTED rather than re-spelled, because the beta pass adds
NO POINTER: its two words are real SCALARS and they ride inside the same
``constant Params&`` the phases and parities already ride in, in front of the chunk
record. There is no inverse-epsilon volume, for
:mod:`.folded_complex_fused_magnetic_pair`'s reason: ``update_H`` reads ``B`` and
``mu`` and never an inverse epsilon, so the count is 27 rather than 30. 28 of the 31
bindings the platform allows (device.py:72). The record grows from 184 to
:data:`PARAMS_ITEMSIZE` bytes and the signature does not move.

:data:`SEPARATE_SCALAR_BINDINGS` is spelled HERE: 27 + 4 uint + 1 ``dtdx`` + 6 phase
+ 4 beta = 42, twelve over the ceiling, the separate-scalar style of the retired beta
curl template. :func:`refuted_separate_scalar_source` builds that signature so a gate
can require the COMPILE FAILURE; the beta words this family binds are two real floats
inside the struct, and any separate-scalar spelling of the 27 pointers is over the
ceiling either way.

===========================================================================
WHAT IS LIFTED, AND WHERE THE PASS LANDS
===========================================================================

* **the weld** -- :func:`.folded_complex_fused_magnetic_pair.folded_complex_fused_magnetic_pair_source`'s
  own output, whole: the folded head, the per-chunk forms, the carries, the wall clear
  and the constitutive blocks;
* **the beta pass** -- :func:`.special_kz.beta_pass` on the mask rows that head carries
  (:func:`.folded_pass.ownership_rows`): the quarter turn of the partner by the real
  word, negated, masked where the curl is masked, then
  ``stepping._apply_folded_increment``'s ladder. :func:`.folded_pass.insert_split_pass`
  puts the increment and the ``fu`` update before the ``fu`` store and each target's
  displacement update right after that displacement is formed, inside the ownership
  guard and before the wall clear and the store -- the order MEEP's curl, pass, fill,
  wall clear take;
* **the constitutive half** -- nothing beta-dependent reaches it:
  ``_apply_constitutive_pml`` never sees ``grid.beta``, which enters the curl passes
  only -- the measurement :func:`.folded_beta.folded_beta_complex_constitutive_coverage`
  already ships on.

===========================================================================
THE COMPLEX MULTIPLIES
===========================================================================

One ``templates.complex_helpers()`` emission serves the whole kernel. The mirror
fill's ``c_mul(parity, v)``, the Bloch rotation and the field-left and
coefficient-left multiplies are the parent's; the beta pass forms no complex product
(a quarter turn by a real word).

===========================================================================
WHAT THIS FAMILY REFUSES, EACH BY NAME
===========================================================================

* ``beta == 0`` — :mod:`.folded_complex_fused_magnetic_pair` owns that seam; the
  refusal arrives through ``folded_beta``'s clause 12 on both halves, and
  :func:`folded_beta_complex_fused_magnetic_pair_source` additionally refuses to emit a
  body whose beta insert is absent;
* an UNFOLDED complex beta run — :mod:`.special_kz`'s certified Bloch beta curl and
  ``beta_run_complex_constitutive_coverage`` own those slots separately, and no weld is
  built over them here. The refusal arrives through ``folded_beta``'s clause 5;
* a REAL folded beta run — :mod:`.beta_fused_magnetic_pair` is the real seam's weld and
  refuses a fold; the real FOLDED B/H weld is not built on this backend. Clause 2
  (storage) is the inversion;
* a MAGNETIC source anywhere in the seam — refused by name, and on this cell that
  clause is exactly one row of three (driver.py:3292-3293). Ignorance is never an empty
  set: ``Fields`` does not hold the source list, so an undeclared ``sources`` is a
  REFUSAL and not an assumed ``()``. An ELECTRIC source is injected in the D/E half and
  does NOT disqualify this pair;
* the D/E seam — ``step_D``'s negated strides, conjugated phase, ELECTRIC beta sign and
  off-diagonal constitutive are a different product entirely; :data:`BACKWARD` is False
  and never a parameter, and :data:`SLOT` is ``step_B``;
* ``has_beta=False`` — not a parameter. The identity arm belongs to
  :func:`.folded_beta.folded_beta_bloch_curl_source`'s own gate, and a ``has_beta=0``
  weld would be the shipped folded complex weld under a second name;
* a folded axis carrying a Bloch phase — refused by the certified emitter itself
  (``folded_beta.folded_beta_bloch_curl_source``), and not re-checked here: one raise,
  in the emitter that owns the rule;
* a susceptibility, a conductivity, a nonlinearity, BFAST, cylindrical coordinates, and
  an off-diagonal ``chi1inv`` row on the E side — all through the two halves' own
  predicates, which are conjoined here unweakened;
* a stored extent too shallow for the near fill's source plane, a reflect row outside
  the stored extent, and an axis reported both folded and walled — restated on this
  family because its carry writes destination cells from a DIFFERENT thread, where a
  violated row is an out-of-range write rather than a soft error.

===========================================================================
NOT WIRED
===========================================================================

``register_arms`` registers this product with ``wired=False``, as every other Metal
fused pair is. ``arms.arms_for`` skips an unwired arm, so ``plan_step`` cannot select it
and no existing arm's selection changes — in particular the wired ``folded beta
complex`` curl and constitutive arms, which admit exactly the configurations this weld
admits and would leave both slots UNSELECTED if this contended for them. Registering
unwired keeps it ENUMERABLE for the disjointness sweep (``arms.registered``). The seam
instances this family reaches are a PREDICATE VERDICT; the number actually stepped by a
fused kernel in production remains zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from ..triton_kernels.launch import SUB_STEPS
from ..triton_kernels.symmetry import (
    CODE_MIRROR_PERIODIC, CODE_PERIODIC, folded_axis_kinds,
)
from . import folded_beta, folded_pass, shaders, special_kz
from . import chunk_bounds as _chunk_bounds
from . import folded_complex_fused_magnetic_pair as _folded_complex_fused_magnetic_pair
from .complex_fused_magnetic_pair import PACKED_BINDINGS
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .folded_complex_fused_magnetic_pair import (
    NEAR_SOURCE_INDEX,
    _far_carry_reasons,
    _mirror_phase_reasons,
    far_fill_axes,
    far_parity_words,
    near_fill_axes,
    parity_words,
)
from .plans import KernelPlan
from .symmetry import MIRROR_CODES
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it does on the
#: two welds this one is made of. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
#:
#: TRUE SINCE 2026-08-30, in the same edit as the ``launch.FUSED_PAIR_ARMS`` row that
#: lets the seam loop bracket this family (the ``('folded beta complex', 'folded beta
#: complex')`` entry). Until that row existed this family had NO route into
#: ``_install_fused_pair`` at all -- the seam loop refused it by name on every
#: configuration, "has no absorb declaration" -- so the flag could not truthfully be
#: anything but False, and the fusion board recorded the cell as
#: ``product_exists_but_refuses`` on its ``in_seam_magnetic_deposit_clears`` clause.
#: Both halves of the claim now hold and neither is new here:
#:
#: * THE WIRING. ``launch._install_fused_pair`` puts a ``LeadingRepairPlan`` in
#:   ``step_B`` and a ``TrailingRepairPlan`` in ``update_H`` whenever the seam carries
#:   a deposit, and it reaches this family through the absorb row above.
#: * THE FOLDED SEAM. What held every folded family at ``False`` was that a folded
#:   seam runs ``fill_symmetry_bc_B``/``fill_folded_far_ghosts_B`` AFTER the injection
#:   and a POINT repair never visits the MIRROR IMAGE of a deposit index. That is
#:   retired by ``deposit_repair.repair_cells`` (deposit_repair.py:217-258), which
#:   extends the saved and restored set to the CLOSURE of the cells those two fills
#:   image each deposit point into -- the same machinery that flipped
#:   ``folded_fused_magnetic_pair`` and ``folded_fused_pair`` on 2026-08-28. It is
#:   storage-agnostic: it indexes and accumulates element-wise over one array per
#:   component, which under complex storage is one ``complex64`` array (fields.py:573).
#:   ``repairable`` still refuses BY NAME the two folds whose fill map it cannot read
#:   -- the cylindrical r = 0 axis and a folded axis storing no more than
#:   ``MIRROR_SOURCE_INDEX`` cells -- and this family inherits both refusals rather
#:   than restating them.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_beta_complex_fused_magnetic_pair"

#: The sub-step slot this arm holds a row on. It spans five; a refusal is NAMED on the
#: slot the fusion starts at rather than being invisible to the table.
SLOT = "step_B"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
#: ``BACKWARD`` is the direction ``SUB_STEPS[CURL_SUB_STEP]`` declares, restated as a
#: bool so the one legal binding is visible without importing :mod:`launch`.
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"
BACKWARD = False

#: Never a parameter. A ``has_beta=False`` weld would be
#: :mod:`.folded_complex_fused_magnetic_pair` under a second name, and every device leg
#: would still pass because beta = 0 IS that family's arithmetic.
HAS_BETA = True

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3288-3296) and in :data:`.coverage.RESIDENCY_ORDER`'s spelling. Read by
#: the whole-step gate's ``covered_passes``; declared, never inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_B", "fill_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: What the same 27 pointers need with the non-pointer arguments bound separately in
#: the retired beta curl template's style: 27 + 4 uint + 1 ``dtdx`` + 6 phase floats
#: + 4 beta floats. Over the ceiling by twelve; :func:`refuted_separate_scalar_source`
#: builds it and a gate requires the compile failure. Any separate-scalar spelling of
#: these 27 pointers is over the ceiling, so the refutation stands for the packed
#: record this family ships.
SEPARATE_SCALAR_BINDINGS = 42

#: The packed struct's size in bytes: the parent's record
#: (:func:`.folded_complex_fused_magnetic_pair.params_record_dtype`) with the two float
#: beta words in front of the chunk record -- 80 + 8 + 100 = 188, padded to the
#: ``float2`` members' 8-byte alignment. :func:`params_record_dtype` checks it.
PARAMS_ITEMSIZE = 192

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the folded beta complex fused magnetic pair binds {PACKED_BINDINGS} buffers "
        f"and Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}; the signature "
        f"cannot be built at all")

if SEPARATE_SCALAR_BINDINGS <= MAX_BUFFER_BINDINGS:  # pragma: no cover - an invariant
    raise RuntimeError(
        f"the refuted separate-scalar signature binds {SEPARATE_SCALAR_BINDINGS} "
        f"buffers, which is NOT over Metal's {MAX_BUFFER_BINDINGS}; the packing this "
        f"family ships would then be a preference rather than a forced choice, and the "
        f"gate leg that requires the compile failure would be measuring nothing")

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY", "HAS_BETA",
    "NEAR_SOURCE_INDEX", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalFoldedBetaComplexFusedMagneticPairPlan",
    "beta_words", "compile_folded_beta_complex_fused_magnetic_pair",
    "folded_beta_complex_fused_magnetic_pair_source",
    "metal_folded_beta_complex_fused_magnetic_pair_coverage", "params_record_dtype",
    "plan_metal_folded_beta_complex_fused_magnetic_pair",
    "refuted_separate_scalar_source", "signature_binding_census",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: The two beta words, as the ``Params`` member line and the unpack line this family
#: adds to the parent's own (:func:`.folded_pass.with_params_members`), in front of
#: the embedded chunk record and its alias.
_BETA_MEMBERS = "    float beta_plus; float beta_minus;"
_BETA_UNPACK = "    float beta_plus = prm.beta_plus, beta_minus = prm.beta_minus;"


def folded_beta_complex_fused_magnetic_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded quadruple, Bloch flags, walls, arm, mode).

    THE CERTIFIED :func:`.folded_complex_fused_magnetic_pair.folded_complex_fused_magnetic_pair_source`'S OWN OUTPUT, with
    MEEP's beta pass carried in (:func:`.folded_pass.insert_split_pass`) and the two
    beta words added to its ``Params`` (:func:`.folded_pass.with_params_members`).
    The pass's statements are :func:`.special_kz.beta_pass`'s complex ones on the mask
    rows the folded head carries -- the cell-0 rows and, on a folded PERIODIC axis, the
    top-plane rows -- so the increment is masked exactly where the curl is. Its
    increment and ``fu`` update land before the ``fu`` store; each target's
    displacement update lands right after the displacement is formed, inside the
    ownership guard and before the wall clear and the store. Every other line is the
    parent's: the folded head and the per-chunk forms, the near and far carries, the
    parity chain, the wall clear and the constitutive blocks.
    """
    owner = "folded_beta_complex_fused_magnetic_pair.folded_beta_complex_fused_magnetic_pair_source"
    parent = _folded_complex_fused_magnetic_pair.folded_complex_fused_magnetic_pair_source(
        codes, phased, zero_metal, contract)
    source = folded_pass.with_params_members(
        parent, ("folded_complex_fused_magnetic_pair_step", "folded_beta_complex_fused_magnetic_pair_step"), _BETA_MEMBERS,
        _BETA_UNPACK, owner)
    block = special_kz.beta_pass(folded_pass.ownership_rows(source), True)
    return folded_pass.insert_split_pass(source, block, owner)


def compile_folded_beta_complex_fused_magnetic_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, Bloch flags, walls, arm, mode)."""
    return compile_source(folded_beta_complex_fused_magnetic_pair_source(
        codes, phased, zero_metal,
        contract)).folded_beta_complex_fused_magnetic_pair_step


# ---------------------------------------------------------------------------
# The binding arithmetic, measured off the EMITTED TEXT
# ---------------------------------------------------------------------------

def signature_binding_census(source: Optional[str] = None) -> Dict[str, int]:
    """Count the ``[[buffer(n)]]`` attributes in an emitted source and check them.

    THE POINT IS THAT THE NUMBER IS READ BACK OUT OF THE STRING. :data:`PACKED_BINDINGS`
    is imported from a sibling and :data:`PARAMS_ITEMSIZE` is spelled by hand; a
    signature that drifted from either would otherwise be discovered on a device, or
    not at all. This counts what the emitter actually wrote, requires the attribute
    indices to be the DENSE range ``0..n-1`` (a gap would mean a buffer nobody binds and
    a plan whose argument tuple is silently off by one), and requires exactly one of
    them to be the packed ``constant Params&``.

    ``source`` defaults to a representative specialisation -- a Bloch phase on the
    unfolded x axis with the fold on y, which is the shape both reachable corpus rows
    carry -- so the census can be taken without a caller choosing a configuration.
    """
    import re  # noqa: PLC0415

    if source is None:
        source = folded_beta_complex_fused_magnetic_pair_source(
            (CODE_PERIODIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC), (1, 0, 0),
            (False, False, False))
    signature = source.split("uint idx [[thread_position_in_grid]])", 1)[0]
    indices = sorted(int(value) for value in
                     re.findall(r"\[\[buffer\((\d+)\)\]\]", signature))
    if indices != list(range(len(indices))):
        raise AssertionError(
            f"the emitted signature's buffer attributes are not the dense range "
            f"0..{len(indices) - 1}: {indices!r}. A gap is a buffer nobody binds and a "
            f"plan whose argument tuple is off by one")
    pointers = signature.count("device float2*") + signature.count("device const float2*") \
        + signature.count("device const float*")
    packed = signature.count("constant Params&")
    if packed != 1:
        raise AssertionError(
            f"the emitted signature carries {packed} packed Params& bindings, not one; "
            f"the whole 42 -> 28 argument is that ALL the scalars ride in exactly one")
    if len(indices) != PACKED_BINDINGS:
        raise AssertionError(
            f"the emitted signature binds {len(indices)} buffers and PACKED_BINDINGS "
            f"says {PACKED_BINDINGS}")
    if pointers + packed != PACKED_BINDINGS:
        raise AssertionError(
            f"{pointers} pointers + {packed} packed struct != {PACKED_BINDINGS}")
    if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - checked at import
        raise AssertionError((PACKED_BINDINGS, MAX_BUFFER_BINDINGS))
    if SEPARATE_SCALAR_BINDINGS <= MAX_BUFFER_BINDINGS:  # pragma: no cover - at import
        raise AssertionError((SEPARATE_SCALAR_BINDINGS, MAX_BUFFER_BINDINGS))
    return {"bindings": len(indices), "pointers": pointers, "packed": packed,
            "ceiling": MAX_BUFFER_BINDINGS,
            "refuted_separate_scalar": SEPARATE_SCALAR_BINDINGS}


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 42-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so a gate can compile it and require the
    failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from an argument into
    a measurement. This is the separate-scalar signature the retired beta curl
    template used, scaled to this pair's fifteen volumes -- so what it measures is "the
    packing is FORCED by the six extra volumes the fusion brings", not that separate
    scalars are bad style. Seven bindings wider than
    :func:`.complex_fused_magnetic_pair.refuted_separate_scalar_source`'s 35: the three
    float2 phases as six floats plus four beta words.

    The body touches every buffer -- what is being measured is the SIGNATURE, and a body
    the compiler could drop would let dead-code elimination decide the answer.
    """
    written = ["f0", "f1", "f2", "u0", "u1", "u2", "h0", "h1", "h2", "w0", "w1", "w2"]
    read_complex = ["g0", "g1", "g2"]
    read_real = ["kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                 "kp0", "km0", "kp1", "km1", "kp2", "km2"]
    scalars = ("dtdx", "pxr", "pxi", "pyr", "pyi", "pzr", "pzi",
               "bpr", "bpi", "bmr", "bmi")
    lines: List[str] = []
    slot = 0
    for name in written:
        lines.append(f"    device float2*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in read_complex:
        lines.append(f"    device const float2* {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in read_real:
        lines.append(f"    device const float*  {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("nx", "ny", "nz", "n_elem"):
        lines.append(f"    constant uint&       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in scalars:
        lines.append(f"    constant float&      {name:<8}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - a construction invariant
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    body = "\n".join(f"    {name}[idx] = {name}[idx] * dtdx;" for name in written)
    touch_real = " + ".join(f"{name}[0]" for name in read_real)
    touch_complex = " + ".join(f"{name}[0].x" for name in read_complex)
    touch_scalar = " + ".join(scalars)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalar(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = ({touch_real}) + ({touch_complex}) + ({touch_scalar});",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz);",
        body,
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_beta_complex_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None) -> Coverage:
    """May ONE dispatch span folded beta complex ``step_B`` -> fills -> wall -> ``update_H``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. That is the
    construction every shipped fused pair uses; the halves are
    :mod:`.folded_beta`'s ROUTING verdicts (a fold MANDATORY), which is what keeps this
    product disjoint from :mod:`.beta_fused_magnetic_pair` and from
    :mod:`.folded_complex_fused_magnetic_pair` in both directions.
    """
    reasons: List[str] = []

    curl = folded_beta.folded_beta_composition_bloch_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency)
    if not curl.covered:
        reasons.extend(f"folded beta complex curl half: {reason}"
                       for reason in curl.reasons)
    magnetic = folded_beta.folded_beta_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency)
    if not magnetic.covered:
        reasons.extend(f"folded beta complex constitutive half: {reason}"
                       for reason in magnetic.reasons)

    # THE PARITY ARM IS THIS PRODUCT'S OWN QUESTION and neither half asks it: the two
    # halves are folded_beta's, which never emits a mirror-parity product because that
    # family registers NO fill arm. The fill arithmetic this weld carries inline is
    # folded_complex's, so folded_complex's artifact is required too.

    # THE SOURCE SEAM. A MAGNETIC source is injected BETWEEN the two halves
    # (driver.py:3292-3293), so a fused pair would consume a pre-injection B.
    # IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. An ELECTRIC source is
    # injected in the D/E half and does NOT disqualify this pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the driver injects "
            f"it BETWEEN step_B and update_H (driver.py:3292-3293), which is work "
            f"inside the seam this kernel closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # ASKED AFTER THE SOURCE CLAUSE, deliberately: that clause reads no grid, and a
    # degenerate object should still get its polarity named rather than only "no grid".
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    reasons.extend(_mirror_phase_reasons(grid, codes))
    reasons.extend(_far_carry_reasons(grid, codes))

    # THE WALL CLEAR IS CARRIED INLINE, so the grid must be able to answer which axes
    # are walled. A grid that cannot would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        # THE NEAR FILL IMAGES STORED CELL 2, and the source thread must exist. The
        # curl half's predicate already refuses a folded axis storing too few cells;
        # restated because THIS family writes the destination from a DIFFERENT thread
        # and a missing source plane is an out-of-range write, not a soft error.
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; the "
                    f"near fill images stored cell {NEAR_SOURCE_INDEX} and this kernel "
                    f"images it from that cell's own thread")

        # THE WALL CLEAR AND THE FILL MUST NOT MEET. On this family both act on
        # component m at stored cell 0 of axis m, and `_zero_metal` skips a folded axis
        # so they are disjoint by construction. CHECKED rather than inferred: reading
        # coverage off another module's guard is what this package's rule forbids, and
        # an overlap would be a plane of wrong values.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall there; "
                    f"stepping._zero_metal skips a folded axis, so the two passes have "
                    f"drifted and the carry's disjointness no longer holds")

    # THE RUN'S FIRST update_H, the parent's clause: the array path starts H and f_w
    # from B on the dsigw cells at that update (stepping.separate_h_from_b) and the
    # lifted H body does not, so this pair takes that update only where the start
    # cannot change a word (launch.h_start_reasons), asked with no source list because
    # a magnetic deposit is this family's bracket's to start.
    from .launch import h_start_reasons  # noqa: PLC0415 - launch imports this family's peers
    reasons.extend(f"folded beta complex constitutive half: {reason}"
                   for reason in h_start_reasons(fields, ()))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedBetaComplexFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs five driver passes on a folded complex beta run.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no special
    case for it.

    ``runs`` is an alias of ``launches`` rather than a second counter, because with one
    dispatch per run the two ARE the same number.
    """

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values", "phases",
                 "parity_values", "far_parity_values", "beta_values", "reflect",
                 "zero_metal", "shape", "dtdx", "params", "carried_axes",
                 "far_carried_axes", "chunk_bounds")

    family = "folded beta complex fused B-curl/mirror-fill/stored-H pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phased", "phases", "zero_metal", "beta_values",
                   "carried_axes", "far_carried_axes", "reflect")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 phases: Sequence[int],
                 parity_values: Sequence[float],
                 far_parity_values: Sequence[float],
                 beta_values: Sequence[float],
                 reflect: Sequence[Optional[int]],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float, params: Any,
                 pointers: Sequence[Any],
                 functions: Mapping[str, Any], *, chunk_bounds: Any) -> None:
        self.residency = residency
        # The host words of the record ``params`` carries, kept for inspection.
        self.chunk_bounds = chunk_bounds
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.phases = tuple(int(value) for value in phases)
        self.parity_values = tuple(float(word) for word in parity_values)
        self.far_parity_values = tuple(float(word) for word in far_parity_values)
        # float(): `beta_curl_coefficients` already rounded ONCE to float32 and
        # float() of a numpy.float32 preserves the word.
        self.beta_values = tuple(float(word) for word in beta_values)
        self.reflect = tuple(None if value is None else int(value)
                             for value in reflect)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        self.params = params
        self.carried_axes = tuple(
            near_fill_axes(self.codes, target) for target in range(3))
        self.far_carried_axes = tuple(
            far_fill_axes(self.codes, target) for target in range(3))
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


#: The two beta words this family's ``Params`` adds to the parent's, in front of the
#: embedded chunk record.
_BETA_FIELDS = (("beta_plus", "<f4"), ("beta_minus", "<f4"))


def params_record_dtype() -> Any:
    """The parent's ``Params`` record with the two beta words in front of ``cb``.

    :func:`.folded_pass.extend_params_dtype` of :func:`.folded_complex_fused_magnetic_pair.params_record_dtype`,
    so every parent member keeps its offset (the three ``float2`` phases first, for
    the alignment reason the parent records) and the chunk record keeps its place
    last, as the source's struct does: itemsize :data:`PARAMS_ITEMSIZE`.
    """
    record = folded_pass.extend_params_dtype(_folded_complex_fused_magnetic_pair.params_record_dtype(),
                                             _BETA_FIELDS)
    if record.itemsize != PARAMS_ITEMSIZE:  # pragma: no cover - an invariant
        raise AssertionError((record.itemsize, PARAMS_ITEMSIZE))
    return record


def beta_words(grid: Any) -> Tuple[float, float]:
    """``(beta_plus, beta_minus)`` -- the CURL ARM'S OWN words, called.

    :func:`.special_kz.beta_curl_coefficients` is the array path's coefficient
    evaluated on the objects the array path evaluates it on, so the words this kernel
    receives are its bits. ``magnetic`` is ``CURL_SUB_STEP == "step_B"`` -- NOT a
    default, because the D side's word carries the negation the B side's does not;
    ``complex_storage=True`` because this is the complex arm. Neither is a parameter.
    """
    return special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(CURL_SUB_STEP == "step_B"),
        complex_storage=True)


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]],
                   near_parities: Sequence[float], far_parities: Sequence[float],
                   beta_values: Sequence[float], reflect: Sequence[Optional[int]],
                   device: str, chunk_words: Any) -> Any:
    """The parent's record, built by the parent's own ``_params_tensor`` on the host,
    re-laid with the two beta words (:func:`.folded_pass.extend_params_words`).

    ``-1`` stands for an axis with no reflect row, the parent's sentinel and for its
    reason. ``chunk_words`` is the chunk record's checked words.
    """
    import torch  # noqa: PLC0415

    beta = tuple(beta_values)
    if len(beta) != 2:
        raise ValueError(
            f"beta_values must be the (plus, minus) pair "
            f"special_kz.beta_curl_coefficients returns, got {beta_values!r}")
    parent = _folded_complex_fused_magnetic_pair._params_tensor(
        shape, dtdx, phase_values, near_parities, far_parities, reflect, "cpu",
        chunk_words)
    words = folded_pass.extend_params_words(
        _folded_complex_fused_magnetic_pair.params_record_dtype(), parent.numpy(), _BETA_FIELDS, beta)
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_folded_beta_complex_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None, *, chunk_bounds: Any,
        ) -> Optional[MetalFoldedBetaComplexFusedMagneticPairPlan]:
    """Build the fused folded beta complex B/H plan, or ``None`` when refused.

    ``None`` is the only refusal, for the reason
    :func:`.folded_beta.plan_folded_beta_bloch_pml_curl` gives: a configuration this
    kernel does not carry must fall back to the array path, never raise into a caller
    that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. A gate compiles a deliberately broken copy of
    the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING -- every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """

    if not metal_folded_beta_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    chunk_words = _chunk_bounds.plan_words(
        chunk_bounds, fields.grid.shape, "MetalFoldedBetaComplexFusedMagneticPairPlan")
    from ..stepping import _boundary_kinds, _far_reflect_rows  # noqa: PLC0415
    from . import complex_fields  # noqa: PLC0415

    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    kinds = _boundary_kinds(grid, pml)
    walls = zero_metal_axes(grid)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    # backward=BACKWARD: this is the B seam. The phase table is resolved through the
    # certified helper the parent uses, so the conjugation rule lives in one place.
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds), backward=BACKWARD)
    parities = parity_words(codes, phases)
    far_parities = far_parity_words(codes, phases)
    beta = beta_words(grid)
    # `stepping._far_reflect_rows` is the ONE derivation of the row a folded PERIODIC
    # axis's far face images; asked here rather than re-derived, and passed as a runtime
    # int because nothing about an integer row can move a bit.
    reflect = _far_reflect_rows(grid)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step, so
    # the groups are built in the order the kernel declares them. There is NO
    # inverse-mu volume: the H product is `source = B` with mu = 1 baked into the array
    # path too, which is what makes this signature 27 pointers rather than 30.
    flux = [bind_complex(name, getattr(fields, name))
            for name in curl_spec["targets"]]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in curl_spec["targets"]]
    electric = [bind_complex(name, getattr(fields, name))
                for name in curl_spec["sources"]]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in side_spec["targets"]]
    workspace = [bind_complex(name, getattr(fields, name))
                 for name in side_spec["aux"]]
    # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER one
    # (SUB_STEPS['step_B']['suffix'] == '_h'; CONSTITUTIVE_SIDES['H']['half_integer'] is
    # False). That is the OPPOSITE pairing to the folded D/E pair, and the kernel cannot
    # tell -- a gate must carry a host mutation for each group.
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}{curl_spec['suffix']}",
                  getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}"))
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + electric + magnetic + workspace
                + curl_coefficients + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_beta_complex_fused_magnetic_pair(
                codes, flags, walls, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor would read
    # "mps:0" where the mirrors were built with "mps" and put the struct on a nominally
    # different device from the volumes it describes.
    dtdx = grid.dt / grid.dx
    return MetalFoldedBetaComplexFusedMagneticPairPlan(
        residency, volumes, codes, flags, values, phases, parities, far_parities,
        beta, reflect, walls, grid.shape, dtdx,
        _params_tensor(grid.shape, dtdx, values, parities, far_parities, beta,
                       reflect, residency.device, chunk_words),
        pointers, selected, chunk_bounds=chunk_words)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False,
            (f"folded beta complex fused magnetic pair cannot fill {slot}",))
    return metal_folded_beta_complex_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalFoldedBetaComplexFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_beta_complex_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, chunk_bounds=context.chunk_bounds)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY, and it is every other fused pair's.
    ``plan_step`` assigns at most one arm per slot and this product spans five, so there
    is no slot it could claim without a composition rule nothing has measured. It would
    additionally contend on ``step_B`` with :mod:`.folded_beta`'s wired ``folded beta
    complex`` curl arm, which admits exactly the same configurations, and
    ``_select_slot`` would then leave the slot UNSELECTED -- the certified curl would
    come off the device as well.

    Registering unwired keeps it ENUMERABLE for the disjointness sweep
    (``arms.registered``) while ``arms.arms_for`` skips it, so ``plan_step`` cannot
    select it and no existing arm's selection changes.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "folded beta complex fused B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded beta complex fused B/H pair: ",
                          noun=("folded beta complex PML B-curl/mirror-fill/"
                                "stored-H pair"),
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
