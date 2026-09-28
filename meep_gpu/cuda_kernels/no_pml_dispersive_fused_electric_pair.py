"""The NO-ABSORBER DISPERSIVE-STORE hand-CUDA fused pair: ``step_D`` -> ``zero_metal_D``
-> plain stored-E ``update_E``.

THE LAST TWO POINTWISE-BUILDABLE ``D_to_E`` CELLS ON THIS BOARD that are not a stencil
and not an arm that launches nothing. Both are occupied by this ONE product, because
their only difference is a compile-time flag:

    (cuda_conductive/conductive,  cuda_no_pml_dispersive/no-PML dispersive store)  2
    (cuda_no_pml_curl/no-PML curl, cuda_no_pml_dispersive/no-PML dispersive store) 1

read from ``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-02_tierfix``'s
``taxonomy.instances.buildable_not_built``. The three corpus rows are
``examples:absorber-1d.py`` and ``tests:TestAbsorber.test_absorber`` (conductive on
every D component, five poles, METALLIC on z) and ``examples:material-dispersion.py``
(lossless, two poles, periodic on all three).

SIBLINGS. ``metal_kernels/no_pml_conductive_fused_electric_pair.py`` and
``triton_kernels/no_pml_fused_electric_pair.py``. This is a PORT of that shape, not an
invention -- CUDA was the only backend without it.

=============================================================================
BUILT AND RELEASED, AND NOT INSTALLED ON THIS CORPUS -- READ THIS FIRST
=============================================================================

The arithmetic is gated and RELEASED under both float32 subnormal policies
(``certification.json:cuda_no_pml_dispersive_fused_electric_pair_2026-09-02b``: 14 of
14 cases bit-identical over 60 complete driver steps, 67 of 83 mutation legs caught,
none unarmed). The predicate admits all three corpus rows and nothing else.

WHAT THE COMPOSER DOES WITH IT IS A DIFFERENT QUESTION, AND THE ANSWER TODAY IS
"NOTHING". This product owns ``step_D`` AND ``update_E``;
``cuda_no_pml_fused_polarization_pair`` owns ``update_E`` AND ``update_P``. One slot,
two claimants. And all THREE rows of this family's two cells are EXACTLY the three
rows that pair serves at the E->P seam -- measured on the stamped census (each row's
``cuda_no_pml_ade`` covering ``update_P``) and on the board's own served ledger, which
records that pair at 3. So installing here is +3 at D_to_E and -3 at E_to_P: NET ZERO,
paid for by displacing a released, certified product.

``fused_pairs._later_seam_claimant`` therefore leaves the slot where it was. THAT IS
NOT A VERDICT ABOUT THIS WELD -- the predicate still admits, the gate still certifies,
and the module is ready. What is missing is a reason to prefer this arrangement over
the one already there, and there are exactly two ways to get one, both a release decision:

* a THREE-SLOT weld, ``step_D`` -> ``update_E`` -> ``update_P``, which would serve
  BOTH seams on these rows (+3 and +3) instead of trading one for the other;
* a ruling on the seam ORDER in ``install_fused_pairs``, which before 2026-09-02
  decided the trade by walking D_to_E first rather than by comparing what each
  arrangement serves.

THE SAME ARITHMETIC HOLDS FOR TWO PRODUCTS THAT SHIPPED BEFORE THIS ONE --
``cuda_dispersive_fused_electric_pair`` (7 rows against ``cuda_fused_polarization_
pair``'s same 7) and ``cuda_no_pml_complex_fused_electric_pair`` (4 against
``cuda_complex_no_pml_fused_polarization_pair``'s same 4) -- and until 2026-09-02 the
board counted BOTH sides of every one of those trades. ``_loses_the_shared_slot`` in
``build_cuda_fusion_matrix.py`` is where that over-count was closed.

=============================================================================
WHY ONE PRODUCT AND NOT TWO, AND WHY THAT IS A MEASUREMENT
=============================================================================

``conductive_kernels`` bakes the per-component conductivity into three ``COND``
defines and emits the certified no-PML curl's OWN plain tail behind ``#else``
(``_no_pml_template``: it builds ``step_D_no_pml_conductive`` by transforming
``no_pml_curl.kernel_source("step_D_no_pml_real")``'s shipped bytes). So the arity
``(False, False, False)`` build IS the lossless curl, character for character, and the
two board cells differ by three preprocessor lines rather than by arithmetic.

The two PREDICATES stay disjoint and are not merged: ``covers_conductive_curl``
refuses a run where no target of ``step_D`` carries a sigma BY NAME, and
``covers_no_pml_curl`` refuses one where any target does. This product asks BOTH and
takes whichever admits -- a DISJUNCTION over two predicates that cannot both hold,
which is a lookup rather than a widening. ``fused_pairs.FUSED_PAIR_EXTRA_ARMS`` is
where the second cell is declared, the mechanism that table exists for.

=============================================================================
THE SEAM, PASS BY PASS -- what is carried, what is refused, what is inert
=============================================================================

The driver runs three passes between ``step_D`` and ``update_E``
(``driver.py:3306-3313``). This family's disposition of each is a clause, not a
comment:

* **the electric injection -- CARRIED, through the PLAIN deposit repair.** See the
  section below; it is the whole reason this cell was unbuildable until
  :data:`..deposit_repair.PLAIN_PATH` existed.
* **``zero_metal_D`` -- CARRIED INLINE, and it had to be.** It writes zero into stored
  cell 0 of the two TANGENTIAL D components of every metallic axis
  (``stepping.py:2206-2247``) and ``update_E`` then reads them. TWO of this product's
  three corpus rows are METALLIC on z, so declining the wall would cost two thirds of
  the coverage. :func:`zero_metal_carry` clears the REGISTER beside the store, because
  the constitutive half reads the register and not the volume.
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` -- INERT, because a mirror
  plane is REFUSED.** Both siblings refuse it the same way and none of the three
  corpus rows carries one (``has_symmetry`` is False on all three in the stamped
  census). The clause is restated by name in :func:`covers_no_pml_dispersive_fused_
  electric_pair` rather than inherited silently, because a reader should not have to
  chase a shared grid predicate to learn which driver pass that refusal empties.
  Without a mirror both passes return at their first line (``stepping.py:1481-1482``,
  ``:1516-1517``).

=============================================================================
THE PLAIN REPAIR, AND WHY THE SPLIT-FIELD ONE WOULD BE WRONG HERE
=============================================================================

All three corpus rows declare an in-seam electric source (``source_field_types ==
["D"]`` on every one), so without a deposit carry this product would serve ZERO. That
is what ``deposit_repair`` is for -- and until :data:`..deposit_repair.PLAIN_PATH` was
measured it could not do it here.

The layer is INACTIVE on this cell, so ``update_E`` takes the PURE OVERWRITE at
``stepping.py:1019-1022`` -- ``E[...] = (D - sum_n P_n) * inv_eps``, with no ``f_w``, no
coefficient vector and no previous value. The split-field repair inverts
``field + kps*fresh - kms*fw_prev`` and NOTHING ELSE, so applying it here would write
an answer the step never computed. :data:`REPAIR_PATHS` therefore names the plain
repair, and ``deposit_repair.repairable`` refuses by name any configuration whose
recurrence is not among the paths its caller declares -- so a wrong declaration is a
REFUSAL rather than a silent mis-repair.

THAT REFUSAL IS MEASURED, NOT ASSERTED.
``parity/meep_gpu/probe_cuda_no_pml_dispersive_electric_pair.py``'s
``split_field_declared`` leg runs this exact product declaring the WRONG path on all
seven of its configurations and records the named refusal each time, beside a
``bracketed`` leg that is byte-identical to the driver's own order and an
``unbracketed`` null control that diverges on every one.

=============================================================================
THE ARITY IS A COMPILE-TIME AXIS ON BOTH SIDES
=============================================================================

This family emits one device string per ``((COND0, COND1, COND2), (n_Ex, n_Ey, n_Ez))``
pair. The conductive triple is ``conductive_kernels.conductive_targets``' own answer --
the same function the predicate reads, which is the discipline that keeps a clause and
a compile-time choice from drifting apart. The pole triple is
``dispersive_kernels``': the chain length is baked, a component with no contributor
forms NO subtraction at all (fields.py:1096-1098 returns the D array itself), and at
arity ``(0, 0, 0)`` the emitted constitutive lines are character-identical to the
non-dispersive family's. The ENTRY POINT NAME is fixed and only the parameter list and
the chain vary, which is ``ARM_KERNEL_NAMES``' own convention;
``compile_cache.kernel_cache_key`` keys the memo on the source.

=============================================================================
THE ALIASING HAZARD
=============================================================================

D appears EXACTLY ONCE in the signature: the curl writes it and the constitutive half
reads it from a register, so binding it a second time as ``const __restrict__`` would
be two restrict pointers to one allocation, which NVRTC miscompiles without a
diagnostic.

THE POLE POINTERS ARE ``__restrict__`` and the three inverse-epsilon ones are not, both
lifted from ``dispersive_kernels.dispersive_source`` rather than decided here: two
susceptibilities never share a ``P`` buffer (``PolarizationState.__init__`` allocates
its own, dispersion.py:645-647) while an isotropic run hands ONE inverse-epsilon
pointer three times (fields.py:1321-1326).

THE CONDUCTIVITY POINTERS ARE NOT ``__restrict__`` EITHER, and that is lifted from
``conductive_kernels`` rather than decided here: a run whose three targets share one
profile hands the same device pointer three times. A LOSSLESS component binds its own
TARGET array in those two slots, which is that launcher's own convention
(``step_conductive_curl``) and is safe because ``#if CONDn`` is 0 and the pointer is
never dereferenced.

THE POLE POINTERS ROTATE. ``PolarizationState.update`` moves ``P``/``P_prev``/scratch
every step (dispersion.py:689-691), so the launcher resolves ``state.P[c]`` immediately
before each launch through ``dispersive_kernels.resolve_pole_plan`` and never caches a
view.

=============================================================================
DISJOINT FROM EVERY OTHER ``step_D`` PRODUCT
=============================================================================

``install_fused_pairs`` leaves a seam UNFUSED when two products admit it, so an overlap
would COST coverage rather than add it. This family requires an INACTIVE layer and a
registered susceptibility and stored E; every other ``step_D`` row on
``fused_pairs.FUSED_PRODUCTS`` requires an ACTIVE layer (the two real ones, the two
complex ones, the dispersive one, BFAST, beta, the folded and off-diagonal ones) or
complex storage or Dcyl, each refused here by name through the two conjoined
predicates. The one no-PML neighbour, ``cuda_no_pml_complex_fused_electric_pair``,
requires COMPLEX storage and this requires real.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import deposit_repair as _deposit_repair
from . import compile_cache

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

try:  # pragma: no cover - the device modules import CuPy at scope
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore

try:  # pragma: no cover
    from . import conductive_kernels
except Exception:  # pragma: no cover
    conductive_kernels = None  # type: ignore

try:  # pragma: no cover
    from . import dispersive_kernels
except Exception:  # pragma: no cover
    dispersive_kernels = None  # type: ignore

try:  # pragma: no cover
    from . import no_pml_curl
except Exception:  # pragma: no cover
    no_pml_curl = None  # type: ignore

from .in_seam_coverage import zero_metal_axes

#: RELEASED 2026-09-02 under BOTH float32 subnormal policies. What moved this kernel
#: across the partition is the RUN, not an argument:
#: ``gate_cuda_no_pml_dispersive_fused_electric_pair.py`` on one RTX A6000, 14 of 14
#: cases bit-identical over 60 COMPLETE driver steps across both curl arms, four
#: conductive masks and six pole arities including both corpus ones, every stored
#: field volume AND every pole buffer compared as uint32 words, 67 of 83 mutation legs
#: caught with three confirmed nulls and NONE unarmed, the deposit legs byte-identical
#: with the unbracketed control diverging on all three fixtures and the WRONG repair
#: declaration refused by name on all three, and the block written into
#: ``certification.json`` as
#: ``cuda_no_pml_dispersive_fused_electric_pair_2026-09-02``.
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "no_pml_dispersive_fused_electric_pair_real",
)

#: EMPTY since 2026-09-02; what emptied it was the run above.
UNCERTIFIED_KERNELS: Dict[str, str] = {}

#: Does this product bracket its fused launch with the deposit repair? TRUE, and the
#: wiring is ``fused_pairs._install_fused_pair``'s. The flag and the wiring change
#: together or not at all (``deposit_repair.py:221-226``).
#:
#: IT IS THE WHOLE PRODUCT HERE. All three corpus rows this family owns declare an
#: electric deposit inside the seam -- ``source_field_types == ["D"]`` on every one in
#: the stamped census -- so without the bracket this module would serve ZERO. That is
#: why ``CARRIES_DEPOSIT_REPAIR`` is asserted as a per-cell MEASUREMENT off the census
#: in the host suite rather than read back as a constant.
CARRIES_DEPOSIT_REPAIR = True

#: WHICH repair the two slots install, and on this family it is NOT the default.
#: ``update_E`` with an inactive layer is a PURE OVERWRITE (stepping.py:1019-1022) and
#: the split-field repair inverts a recurrence that does not run here, so declaring the
#: default would make ``deposit_repair.repairable`` refuse every configuration this
#: product exists for. See the module docstring's repair section; the refusal is
#: measured in the probe's ``split_field_declared`` leg.
REPAIR_PATHS: Tuple[str, ...] = (_deposit_repair.PLAIN_PATH,)

FAMILY = "cuda_no_pml_dispersive_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once. FIXED ACROSS BOTH COMPILE-TIME AXES
#: -- the conductive mask and the pole arity vary the SOURCE, not the name, which is
#: ``ARM_KERNEL_NAMES``' convention and what lets one certification block cover the
#: swept corpus.
KERNEL_NAME = "no_pml_dispersive_fused_electric_pair_real"

#: The driver passes ONE launch performs, in driver order (driver.py:3306, :3310,
#: :3313). ``zero_metal_D`` is a driver pass no slot names, so it is DECLARED here
#: rather than inferred from the two slots. The two symmetry fills are absent because
#: this family REFUSES a mirror plane and they are inert on every row it admits.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: The conductive masks this family emits for a digest, and the ones its gate sweeps.
#: ``(F, F, F)`` is the LOSSLESS curl arm -- ``material-dispersion.py``'s cell -- and
#: ``(T, T, T)`` is the conductive one, which is both absorber rows. The mixed masks
#: are swept because ``conductive_targets`` is per COMPONENT and a run may be lossy on
#: one target and not another; the corpus drives none, and that is recorded rather
#: than used as a reason to skip them.
COND_MASKS_SWEPT: Tuple[Tuple[bool, bool, bool], ...] = (
    (False, False, False),
    (True, True, True),
    (True, False, False),
    (False, True, True),
)


def swept_arities() -> Tuple[Tuple[int, int, int], ...]:
    """The pole arities this family emits, taken from ``dispersive_kernels`` rather
    than restated: a triple this family emitted and that family never swept would be a
    body no device has run. Both corpus arities are in it -- ``(5, 5, 5)`` is the two
    absorber rows and ``(2, 2, 2)`` is ``material-dispersion.py``.
    """
    if dispersive_kernels is None:
        raise RuntimeError(
            "dispersive_kernels is not importable on this host, so the swept arities "
            "cannot be read from the family that defines them")
    return tuple(dispersive_kernels.POLE_COUNTS_SWEPT)


#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"what": "the curl's per-target tail",
     "from": "Dx[idx] = Dx[idx] - curl;  /  cond_apply(Dx, cf0, ci0, idx, curl);",
     "to": "d_x = Dx[idx] - curl;  /  d_x = cond_apply_reg(Dx, cf0, ci0, idx, curl);",
     "why": "the stepped displacement is CARRIED in a register instead of being "
            "written and read back; the store happens after the wall clear, which is "
            "where the array path's own order puts it"},
    {"what": "the constitutive half's index decomposition and bounds guard",
     "from": "int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n_elem) "
             "return;",
     "to": "(dropped)",
     "why": "the curl half above already declares idx and its guard has already "
            "returned; a second decomposition would redeclare idx, and n_elem is the "
            "same number nx * ny * nz is on every run the predicate admits"},
    {"what": "the constitutive half's displacement source",
     "from": "float s_x = Dx[idx];  /  float src_x = Dx[idx] * inv_eps_Ex[idx];",
     "to": "float s_x = d_x;  /  float src_x = d_x * inv_eps_Ex[idx];",
     "why": "THE SEAM. The chain starts from the carried register, which is also what "
            "removes this half's only use of a second D binding and so the aliasing "
            "hazard the signature exists to avoid"},
    {"what": "zero_metal_D",
     "from": "(a separate pass, in_seam_passes.zero_metal_D at driver.py:3310)",
     "to": "an inline clear of the register and the store",
     "why": "the pass sits between the two halves and update_E reads what it wrote; "
            "two of the three corpus rows are METALLIC on z, so declining it would "
            "cost two thirds of this product's coverage"},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "COND_MASKS_SWEPT", "FAMILY",
    "KERNEL_NAME", "LIFT_EDITS", "REPAIR_PATHS", "REPLACES", "SLOT",
    "UNCERTIFIED_KERNELS", "assert_disjoint_bindings", "certified_constitutive_body",
    "certified_curl_body", "conductive_bindings",
    "covers_no_pml_dispersive_fused_electric_pair", "device_sources",
    "launch_no_pml_dispersive_fused_electric_pair",
    "no_pml_dispersive_fused_electric_pair_prelude",
    "no_pml_dispersive_fused_electric_pair_signature",
    "no_pml_dispersive_fused_electric_pair_source", "pole_bindings",
    "run_no_pml_dispersive_fused_electric_pair", "swept_arities", "zero_metal_carry",
]

# =============================================================================
# THE DEVICE TEXT
# =============================================================================

_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"

#: The certified conductive template's name for this sub-step's no-absorber kernel.
_CURL_TEMPLATE = "step_D_no_pml_conductive"

#: The three carried registers, per axis, and the D volume each belongs to.
_CARRIED: Tuple[str, ...] = ("x", "y", "z")
_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: ``stepping._zero_metal``'s component table, restated as (walled-axis flag, that
#: axis's coordinate, the two components wiped there). THE OFF-DIAGONAL COMPLEMENT:
#: ``fields.IYEE_SHIFTS`` gives Dx (1,0,0), Dy (0,1,0), Dz (0,0,1), so a D component
#: has Yee shift 0 on the OTHER TWO axes and TWO components sit on each wall. A pair
#: that reused the B family's diagonal would clear ONE wrong component and leave two
#: right ones standing, on every walled run.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("wall_x", "i", ("y", "z")),
    ("wall_y", "j", ("x", "z")),
    ("wall_z", "k", ("x", "y")),
)

#: ``cond_apply`` as a VALUE. The certified helper stores; the weld needs the same
#: arithmetic without the store, because the store happens after the wall clear. The
#: expression is the certified one character for character -- only the store is gone.
_COND_APPLY_REG = r'''
#if COND0 || COND1 || COND2
// conductive_kernels.cond_apply AS A VALUE. Identical arithmetic, no store: the
// weld clears the register on a metallic wall BEFORE writing it, which is where
// stepping's own order puts zero_metal_D (driver.py:3310, between the curl and
// update_E).
__device__ __forceinline__ float cond_apply_reg(
    const float* __restrict__ f, const float* cf,
    const float* ci, int idx, float curl
) {
    return ((f[idx] * cf[idx]) - curl) * ci[idx];
}

#endif
'''

#: The fused entry point. The ONLY hand-written device text in this module, and it is a
#: signature: no arithmetic lives here.
_SIGNATURE = r'''
extern "C" __global__ void __NAME__(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the constitutive
    // half reads it from a register; binding it a second time as a source would be
    // two pointers to one allocation with a __restrict__ promise on one of them,
    // which is UB and which NVRTC miscompiles without a diagnostic.
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    // The curl's operands. SPELLED AS H because the lifted certified body spells
    // them so; without an absorber Fields.get_H returns the B arrays themselves
    // (fields.py:1164-1187) and the launcher binds exactly those.
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    // NOT __restrict__, and lifted from the certified conductive curl rather than
    // decided here: a run whose three targets share one profile hands the same
    // device pointer three times. A LOSSLESS component binds its own target array
    // here, which is step_conductive_curl's own convention and is safe because
    // #if CONDn is 0 and the pointer is never dereferenced.
    const float* cf0, const float* cf1, const float* cf2,
    const float* ci0, const float* ci1, const float* ci2,
    // E, the constitutive target. There is no f_w beside it: without a layer the
    // stored E IS the constitutive product (stepping.py:990-993).
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    // NOT __restrict__ either, and lifted from the certified update_E rather than
    // decided here: an isotropic run hands the same device pointer three times
    // (fields.py:1321-1326).
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
__POLE_PARAMS__    int nx, int ny, int nz, float dtdx,
    int bc_x, int bc_y, int bc_z,
    // zero_metal_D's three per-axis flags (driver.py:3310). A FOLDED metallic axis
    // carries no wall (stepping._zero_metal, :2237-2239) and this family refuses a
    // fold outright, so the two can never disagree here.
    int wall_x, int wall_y, int wall_z
) {
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 990-993->1019-1022


def _certified_halves() -> None:
    """Refuse the emitter BY NAME on a host where the certified text is unreachable.

    THE SPLICE IS THE LIFT, so a missing half is not a degraded emit -- there is
    nothing to emit. Raising here keeps that fact one frame from the caller.
    :func:`covers_no_pml_dispersive_fused_electric_pair` needs neither module's device
    text and still answers.
    """
    missing = [name for name, module in (
        ("conductive_kernels", conductive_kernels),
        ("dispersive_kernels", dispersive_kernels)) if module is None]
    if missing:
        raise RuntimeError(
            f"the certified halves {missing} are not importable on this host (they "
            f"import CuPy at module scope), so there is no certified text to splice. "
            f"covers_no_pml_dispersive_fused_electric_pair needs neither and still "
            f"answers")


def _split_body(source: str, prelude: str, name: str) -> str:
    """One certified kernel string, reduced to its body.

    Raises rather than returning a truncated body: a splice that silently dropped a
    component would produce a kernel that steps two fields and reports success.
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


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    """Substitute exactly one occurrence, or fail by name.

    A missing anchor is certified text that has changed under this family, and splicing
    around it would produce a kernel that compiles and is quietly not the certified
    arithmetic; TWO matches would mean the anchor no longer identifies a single site.
    """
    count = text.count(old)
    if count != 1:
        raise AssertionError(
            f"{what}: the anchor {old!r} appears {count} times in the certified text, "
            f"not once; this family LIFTS that line rather than retyping it and "
            f"cannot splice around its absence")
    return text.replace(old, new, 1)


def _code_lines(body: str) -> List[str]:
    """``body`` without its comment lines, for checks that must read what compiles."""
    return [line for line in body.splitlines()
            if not line.lstrip().startswith("//")]


def no_pml_dispersive_fused_electric_pair_prelude() -> str:
    """Both certified preludes, plus ``cond_apply_reg``.

    The curl's prelude is the CONDUCTIVE template's own head -- it already carries
    ``shift_up``/``shift_dn`` (lifted from ``no_pml_curl``) and ``cond_apply``. The
    constitutive half's prelude is a comment block only (``dispersive_kernels``'
    ``_POLE_CHAIN_NOTE`` and the no-PML note), so it contributes no symbol and cannot
    collide.
    """
    _certified_halves()
    template = conductive_kernels.kernel_template(_CURL_TEMPLATE)
    if template.count(_BODY_ANCHOR) < 1:
        raise AssertionError(
            "the certified conductive template no longer carries a signature "
            "terminator; the prelude has no end")
    prelude = template.split('extern "C" __global__ void', 1)[0]
    if "cond_apply(" not in prelude:
        raise AssertionError(
            "the certified conductive template's prelude no longer declares "
            "cond_apply; the value rewrite this family adds beside it has no sibling "
            "to match")
    return prelude + _COND_APPLY_REG


def no_pml_dispersive_fused_electric_pair_signature(counts: Sequence[int]) -> str:
    """The fused signature at one pole arity."""
    _certified_halves()
    values = dispersive_kernels.normalized_pole_counts(counts)
    source = _SIGNATURE.replace("__NAME__", KERNEL_NAME)
    # THE CERTIFIED EMITTER'S OWN FUNCTION, called rather than retyped: the pole
    # parameter names and their __restrict__ decision are that family's, and a copy
    # here would be a second place for the launcher's binding order to drift from.
    source = source.replace("__POLE_PARAMS__",
                            dispersive_kernels._pole_parameter_block(values))
    if "__" in source.replace("__restrict__", "").replace("__global__", ""):
        raise AssertionError("an unsubstituted placeholder survived the signature")
    return source


def certified_curl_body() -> str:
    """``step_D_no_pml_conductive``'s body, with the three registers captured.

    Not a transcription: this is what ``conductive_kernels.kernel_template`` holds,
    minus its prelude and signature -- and that template is itself built from
    ``no_pml_curl.kernel_source("step_D_no_pml_real")``'s shipped bytes, which is why
    the ``COND=(0,0,0)`` build is the certified lossless curl rather than a second
    copy of it.

    THE DEFINES ARE NOT EMITTED HERE. ``kernel_template`` is the text WITHOUT them and
    :func:`no_pml_dispersive_fused_electric_pair_source` prepends the three
    ``#define`` lines, exactly as ``conductive_kernels.kernel_source`` does.
    """
    _certified_halves()
    template = conductive_kernels.kernel_template(_CURL_TEMPLATE)
    prelude = template.split('extern "C" __global__ void', 1)[0]
    body = _split_body(template, prelude, f"kernel_template({_CURL_TEMPLATE!r})")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the carried "
            "registers have nowhere to be declared")
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END,
        "\n    // The three carried registers. Declared HERE because each certified\n"
        "    // component below is a braced scope, and a value declared inside one\n"
        "    // does not outlive it. They are the stepped displacement, held between\n"
        "    // the curl and update_E instead of being written and read back.\n"
        "    float d_x = 0.0f;\n"
        "    float d_y = 0.0f;\n"
        "    float d_z = 0.0f;\n"
        "\n",
        tail,
    ))
    for index, (axis, target) in enumerate(zip(_CARRIED, _TARGETS)):
        # BOTH BRANCHES OF THE CERTIFIED #if ARE REWRITTEN, and both are checked. The
        # guarded block is conductive_kernels' own text; capturing only one branch
        # would leave the other writing to global memory, where the constitutive half
        # would never read it.
        body = _replace_once(
            body,
            f"        cond_apply({target}, cf{index}, ci{index}, idx, curl);\n",
            f"        d_{axis} = cond_apply_reg({target}, cf{index}, ci{index}, "
            f"idx, curl);\n",
            f"target {target}'s conductive tail")
        body = _replace_once(
            body,
            f"        {target}[idx] = {target}[idx] - curl;\n",
            f"        d_{axis} = {target}[idx] - curl;\n",
            f"target {target}'s plain tail")
    # NO TAIL MAY SURVIVE UNCAPTURED. A call left storing would put the stepped value
    # in global memory where the wall clear below could not reach it, and the
    # constitutive half would read a pre-clear volume.
    code = "\n".join(_code_lines(body))
    if "cond_apply(" in code.replace("cond_apply_reg(", ""):
        raise AssertionError(
            "a call to cond_apply survived the capture rewrite; its value would be "
            "written to global memory and never read into the seam")
    for target in _TARGETS:
        if f"{target}[idx] = " in code:
            raise AssertionError(
                f"a store into {target} survived the capture rewrite; the wall clear "
                f"below writes the register and this store would race it")
    # THE STENCIL IS ASSERTED, NOT EDITED. It is the arithmetic this family reuses
    # unchanged, so its survival is a property of the build rather than a claim.
    stencil = "        float curl = dtdx * ((sf - f1) + (f2 - ss));"
    if body.count(stencil) != 3:
        raise AssertionError(
            f"the certified curl body carries {body.count(stencil)} of the three curl "
            f"stencil lines; this family exists to reuse them unchanged")
    return body


def zero_metal_carry() -> str:
    """``zero_metal_D`` (driver.py:3310), carried between the two halves.

    ``stepping._zero_metal`` writes the integer 0 into stored cell 0 of every component
    whose Yee shift on a walled axis is 0 -- for D that is the OFF-DIAGONAL, TWO
    components per wall (:data:`_ZERO_METAL_ROWS`). The register is cleared BESIDE the
    store, because the constitutive half below reads the register and not the volume.

    STORED CELL 0 IS THE LOW WALL and the high wall is the zero ghost ``shift_dn``
    already supplies, which is the certified pass's own note.

    NO OWNERSHIP GUARD, AND THAT IS A CONSEQUENCE OF THE FOLD REFUSAL rather than a
    simplification. The real twin guards this clear on ``own_*`` because a near ghost
    of a cleared cell belongs to its source thread; this family refuses a mirror plane
    outright, so no cell is imaged, every thread owns its own cell, and the guard would
    be constant-true. A FOLDED METALLIC AXIS CARRIES NO WALL either
    (``stepping._zero_metal``'s ``is_metallic and not is_mirrored``, :2237-2239), so
    the wall flags and the boundary codes cannot disagree on any admitted row.

    THE STORE IS HERE, not in the curl body, and that is the whole point of the carry:
    the array path writes D, clears it, and only then reads it, so the one value the
    two consumers see is the POST-clear one.
    """
    lines = [
        "\n    // stepping._zero_metal (:2206-2245) / in_seam_passes.zero_metal_D,\n"
        "    // CARRIED. The OFF-DIAGONAL: a D component sits ON the wall of each\n"
        "    // axis whose Yee shift is 0, which for D is the other two -- so two\n"
        "    // components are wiped per walled axis. Stored cell 0 is the LOW wall;\n"
        "    // the high wall is the zero ghost shift_dn already supplies.\n"
    ]
    for flag, coordinate, axes in _ZERO_METAL_ROWS:
        sets = " ".join(f"d_{axis} = 0.0f;" for axis in axes)
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {sets} }}\n")
    lines.append(
        "\n    // THE STORE, after the clear and before the constitutive read, which\n"
        "    // is exactly where driver.py:3310 puts zero_metal_D relative to :3313.\n")
    for axis, target in zip(_CARRIED, _TARGETS):
        lines.append(f"    {target}[idx] = d_{axis};\n")
    return "".join(lines)


def certified_constitutive_body(counts: Sequence[int]) -> str:
    """``update_E_no_pml_real_dispersive``'s body, lifted, reading the registers.

    Drops the index decomposition and bounds guard (the curl body above already
    declares ``idx`` and its guard has already returned) and turns each component's
    displacement source into the carried register -- which is also what removes this
    half's only use of a second D binding.
    """
    _certified_halves()
    values = dispersive_kernels.normalized_pole_counts(counts)
    source = dispersive_kernels.dispersive_source("no_pml", values)
    prelude = source.split('extern "C" __global__ void', 1)[0]
    body = _split_body(source, prelude, "dispersive_source('no_pml', ...)")
    body = _replace_once(
        body,
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (idx >= n_elem) return;\n"
        "\n",
        "    // The index decomposition and the bounds guard are the CURL half's,\n"
        "    // above: this half's own copy would redeclare idx, and its bound\n"
        "    // (n_elem) is the same number the curl's (nx * ny * nz) is on every\n"
        "    // run the predicate admits.\n",
        "the constitutive index decomposition and bounds guard")

    # THE SEAM, one edit per component and the shape of it decided by the ARITY --
    # because the certified emitter itself emits two different lines. At a non-zero
    # count it opens the chain with `float s_a = Da[idx];`; at zero it forms no
    # intermediate at all and multiplies D directly (fields.py:1096-1098 returns the D
    # array itself when nothing drives the component). Both are rewritten to read the
    # register, and asking for the wrong one is a named failure rather than a silent
    # miss.
    for axis, target, component, count in zip(_CARRIED, _TARGETS, _ELECTRIC, values):
        if int(count) == 0:
            body = _replace_once(
                body,
                f"    float src_{axis} = {target}[idx] * inv_eps_{component}[idx];",
                f"    float src_{axis} = d_{axis} * inv_eps_{component}[idx];"
                f"   // THE SEAM",
                f"target {target}'s degenerate-arity displacement source")
        else:
            body = _replace_once(
                body,
                f"    float s_{axis} = {target}[idx];",
                f"    float s_{axis} = d_{axis};   // THE SEAM",
                f"target {target}'s displacement source")
    code = "\n".join(_code_lines(body))
    for target in _TARGETS:
        if f"{target}[idx]" in code:
            raise AssertionError(
                f"{target} survived the seam rewrite; the constitutive half would need "
                f"D bound a second time, which is the aliasing hazard this signature "
                f"exists to avoid")
    # THE DROPPED BOUND, CHECKED ON CODE RATHER THAN ON PROSE: the comment spliced in
    # above NAMES n_elem on purpose, so a substring test over the whole body would fire
    # on this module's own explanation.
    survived = [line for line in _code_lines(body) if "n_elem" in line]
    if survived:
        raise AssertionError(
            f"n_elem survived in {survived!r}; the constitutive bound is dropped in "
            f"favour of the curl's and is not a parameter of this kernel")
    return body


def no_pml_dispersive_fused_electric_pair_source(cond: Sequence[bool],
                                                 counts: Sequence[int]) -> str:
    """The whole fused kernel at one conductive mask and one pole arity.

    THE THREE ``#define`` LINES ARE THIS FILE'S ONLY ADDITION TO THE MASK, exactly as
    ``conductive_kernels.kernel_source`` spells it -- and for that function's own
    reason: the conductive tail helper lives INSIDE the emitted text behind
    ``#if COND0 || COND1 || COND2``, so a fully lossless build preprocesses it away AND
    a gate that mutates the emitted source still reaches it.
    """
    mask = tuple(bool(flag) for flag in cond)
    if len(mask) != 3:
        raise ValueError(
            f"cond must be a triple of per-target conductivity flags, got {cond!r}; "
            f"it is conductive_kernels.conductive_targets' own answer and a shorter "
            f"tuple would leave a #define unset")
    defines = "".join(f"#define COND{index} {int(flag)}\n"
                      for index, flag in enumerate(mask))
    source = "".join((
        defines,
        no_pml_dispersive_fused_electric_pair_prelude(),
        no_pml_dispersive_fused_electric_pair_signature(counts),
        certified_curl_body(),
        zero_metal_carry(),
        certified_constitutive_body(counts),
        "}\n",
    ))
    # PURE ASCII IS A COMPILE REQUIREMENT, not a style rule; checked here so a mutation
    # leg that inserts a non-ASCII character is refused at emission with the reason
    # instead of at NVRTC three frames away.
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by ``cond/arity`` -- for a digest."""
    out: Dict[str, str] = {}
    for mask in COND_MASKS_SWEPT:
        for counts in swept_arities():
            key = ("".join("1" if flag else "0" for flag in mask) + "/"
                   + "".join(str(int(v)) for v in counts))
            out[key] = no_pml_dispersive_fused_electric_pair_source(mask, counts)
    return out


#: NVRTC compile options -- CORRECTNESS, not performance, and identical to both
#: certified halves'. Spelled here rather than imported so that loading this file by
#: path (which the bit-identity probe does) cannot pick up a different tuple.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one cell per lane -- what every real sibling landed on.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear()


def _get_kernel(cond: Sequence[bool], counts: Sequence[int],
                name: str = KERNEL_NAME):
    """Compile (or fetch) the kernel at one mask and arity.

    ``compile_cache.kernel_cache_key`` keys the memo ON THE SOURCE, so a gate that
    mutates the emitted text is a cache MISS and reaches NVRTC -- which is what stops a
    mutation leg from scoring against a kernel it never applied.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    code = no_pml_dispersive_fused_electric_pair_source(cond, counts)
    key = compile_cache.kernel_cache_key(name, True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def _by_path(module_name: str, attribute: str):
    """One predicate, importable without a device.

    The kernel modules import CuPy at scope, so on a laptop the ordinary import fails
    and the predicate -- the half whose failure mode is a SILENT wrong answer -- would
    be unreachable. It is loaded by path there, exactly as every sibling product does.
    """
    import importlib.util as _importlib_util  # noqa: PLC0415
    import os as _os  # noqa: PLC0415

    here = _os.path.dirname(_os.path.abspath(__file__))
    spec = _importlib_util.spec_from_file_location(
        f"cuda_kernels_{module_name}_predicate",
        _os.path.join(here, f"{module_name}.py"))
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


def _curl_predicates():
    """``(conductive, lossless)`` -- the two ``step_D`` curl predicates, in order."""
    conductive = (conductive_kernels.covers_conductive_curl
                  if conductive_kernels is not None
                  else _by_path("conductive_kernels", "covers_conductive_curl"))
    lossless = (no_pml_curl.covers_no_pml_curl if no_pml_curl is not None
                else _by_path("no_pml_curl", "covers_no_pml_curl"))
    return conductive, lossless


def _constitutive_predicate():
    if dispersive_kernels is not None:
        return dispersive_kernels.covers_no_pml_dispersive_constitutive
    return _by_path("dispersive_kernels", "covers_no_pml_dispersive_constitutive")



def covers_no_pml_dispersive_fused_electric_pair(
        fields: Any, pml: Any, grid: Any,
        sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``zero_metal_D`` -> plain stored-E ``update_E``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION OVER TWO HALVES, AND NOTHING IS WEAKENED. The constitutive half is
    ``dispersive_kernels.covers_no_pml_dispersive_constitutive``, whole. The curl half
    is a DISJUNCTION over two predicates that CANNOT BOTH HOLD --
    ``conductive_kernels.covers_conductive_curl`` requires at least one target to carry
    a sigma and ``no_pml_curl.covers_no_pml_curl`` refuses any target that does -- so
    it is a lookup between two disjoint arms rather than a widening. Both are asked
    whole and the reasons of both are reported when neither admits.

    THE SEAM CLAUSES ARE THIS PREDICATE'S OWN, because there is no sibling product to
    delegate them to on this cell: the fold refusal, the wall/fold disjointness and the
    in-seam source routing are asked here, by name.
    """
    covered, reason = _constitutive_predicate()(fields, pml, grid)
    if not covered:
        return False, f"dispersive constitutive half: {reason}"

    conductive, lossless = _curl_predicates()
    conductive_ok, conductive_reason = conductive(fields, pml, grid, "step_D")
    if conductive_ok:
        arm = "conductive"
    else:
        lossless_ok, lossless_reason = lossless(fields, pml, grid, "step_D")
        if not lossless_ok:
            return False, (f"neither curl arm admits step_D: the conductive arm says "
                           f"{conductive_reason!r} and the lossless arm says "
                           f"{lossless_reason!r}")
        arm = "no_pml"
    _ = arm

    # THE TWO SYMMETRY PASSES. Both return at their first line without a mirror
    # (stepping.py:1481-1482, :1565-1566). RESTATED HERE rather than inferred from
    # either half's guard: this module's coverage may not be read off another module's,
    # and if either tranche ever admits a fold this weld would silently swallow two
    # passes that had started doing work.
    if bool(getattr(grid, "has_symmetry", bool)()):
        return False, ("a mirror plane is active: stepping.fill_symmetry_bc_D "
                       "(driver.py:3309) and fill_folded_far_ghosts_D (:3311) run "
                       "inside this seam and this weld carries neither")
    for axis in range(3):
        if bool(getattr(grid, "is_mirrored", lambda _a: False)(axis)):
            return False, (f"axis {axis} is folded by a mirror plane: the two symmetry "
                           f"passes image ghost cells inside this seam and this weld "
                           f"carries neither")

    # THE WALL. ``zero_metal_axes`` is the same reading ``in_seam_passes`` takes, so
    # the flags this predicate admits and the ones the launcher binds are one answer.
    try:
        walls = zero_metal_axes(grid)
    except Exception as error:  # an unreadable wall is not an absent one
        return False, (f"the metallic-wall axes cannot be read from the grid "
                       f"({type(error).__name__}: {error}), and this weld carries "
                       f"zero_metal_D inline")
    for axis in range(3):
        if walls[axis] and bool(getattr(grid, "is_mirrored", lambda _a: False)(axis)):
            return False, (f"axis {axis} is reported BOTH walled and folded; "
                           f"stepping._zero_metal carries no wall on a folded axis "
                           f"(:2237-2239) and this weld would clear a plane the array "
                           f"path leaves alone")

    # THE IN-SEAM SOURCE, routed through the repair this family DECLARES. The path
    # matters: ``repairable`` refuses by name any recurrence not among ``paths``, so
    # handing it the default here would refuse every configuration this product exists
    # for -- and handing it the wrong one would MIS-REPAIR rather than refuse.
    # THE WORDING IS THIS SITE'S OWN, which is what ``seam_source_reasons`` asks for:
    # the shared helper decides WHICH sources count as in-seam and whether a deposit
    # is carryable, and each product keeps the text its own gate and tests pin.
    #
    # ``pml`` IS FORWARDED, and on this family it is REQUIRED rather than optional:
    # ``repairable`` refuses a plain-only declaration with no layer BY NAME, because
    # which recurrence ran cannot be established without one.
    reasons = _deposit_repair.seam_source_reasons(
        fields, sources, "D", pml=pml, carries_repair=CARRIES_DEPOSIT_REPAIR,
        repair_paths=REPAIR_PATHS,
        undeclared=("the source list was not declared: this weld spans the driver's "
                    "electric injection (driver.py:3306-3313) and Fields does not "
                    "hold the sources, so an undeclared set is a refusal rather than "
                    "an empty seam"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) deposits into D inside this "
            f"seam and this weld does not carry it"))
    if reasons:
        return False, reasons[0]
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

_FIELD_BINDINGS: Tuple[str, ...] = ("Dx", "Dy", "Dz")

#: The curl's source arrays. Without an absorber ``Fields.get_H`` returns the B arrays
#: themselves (fields.py:1164-1187) and ``stepping.step_D`` differences exactly them,
#: which is also what ``conductive_kernels.step_conductive_curl`` binds on this arm.
_CURL_SOURCES: Tuple[str, ...] = ("Bx", "By", "Bz")


def conductive_bindings(fields: "Fields") -> Tuple[Tuple[Any, ...],
                                                   Tuple[bool, bool, bool]]:
    """``(condfac triple + condinv triple, the COND mask)``, resolved RIGHT NOW.

    ``conductive_kernels.conductive_targets`` is the one reading of the question, and
    it is what the predicate reads too -- the discipline that keeps a clause and a
    compile-time choice from drifting apart.

    A LOSSLESS COMPONENT BINDS ITS OWN TARGET ARRAY in both slots, which is
    ``step_conductive_curl``'s own convention: ``#if CONDn`` is 0 for it, so the
    pointer is never dereferenced, and binding a null would be a different kind of
    hazard.
    """
    if conductive_kernels is None:
        raise RuntimeError(
            "conductive_kernels is not importable on this host, so the per-component "
            "conductivity mask cannot be read from the family that defines it")
    mask = conductive_kernels.conductive_targets(fields, "step_D")
    targets = tuple(getattr(fields, name) for name in _FIELD_BINDINGS)
    condfac = tuple(fields.condfac_for(name) if flag else target
                    for name, flag, target in zip(_FIELD_BINDINGS, mask, targets))
    condinv = tuple(fields.condinv_for(name) if flag else target
                    for name, flag, target in zip(_FIELD_BINDINGS, mask, targets))
    return tuple(condfac) + tuple(condinv), tuple(bool(flag) for flag in mask)


def pole_bindings(fields: "Fields") -> Tuple[Tuple[Any, ...], Tuple[int, int, int]]:
    """``(P pointers in signature order, the arity triple)``, resolved RIGHT NOW.

    ``PolarizationState.update`` rotates the three buffers every step
    (dispersion.py:689-691), so a plan built before ``update_P`` names last step's
    arrays after it -- stale in a way that still computes, which is the worst kind.
    """
    plan = dispersive_kernels.resolve_pole_plan(fields)
    counts = dispersive_kernels.pole_counts_of(plan)
    pointers: List[Any] = []
    for target, _source, _axis in dispersive_kernels.ELECTRIC_TERMS:
        pointers.extend(plan[target])
    return tuple(pointers), counts


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three inverse-epsilon volumes, in signature order."""
    return tuple(fields.inverse_epsilon_for(component) for component in _ELECTRIC)


def assert_disjoint_bindings(fields: "Fields", poles: Sequence[Any],
                             conductivity: Sequence[Any]) -> int:
    """Check the promise every ``__restrict__`` in this signature makes.

    THE RESTRICT GROUP IS: the three D volumes, the three B sources, the three E
    volumes and the pole pointers. The inverse-epsilon and conductivity pointers are
    NOT restrict and may legitimately repeat, so they are checked only AGAINST the
    restrict group and never against each other.

    Returns the number of distinct allocations checked, so a caller can assert that
    something was actually inspected.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []
    for name in _FIELD_BINDINGS + _CURL_SOURCES + _ELECTRIC:
        array = getattr(fields, name)
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{name} and {bound[pointer]} are the same allocation")
            continue
        bound[pointer] = name
    for index, array in enumerate(poles):
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"pole {index} and {bound[pointer]} are the same "
                              f"allocation")
            continue
        bound[pointer] = f"pole {index}"
    # THE NON-RESTRICT GROUPS, ASKED ONE WAY ONLY. A conductivity volume that IS a
    # target array is this launcher's own placeholder convention for a lossless
    # component and is expected; what is not allowed is a non-restrict pointer
    # colliding with a POLE, which is bound restrict.
    for label, group in (("inv_eps", inverse_epsilon_bindings(fields)),
                         ("conductivity", tuple(conductivity))):
        for index, volume in enumerate(group):
            pointer = int(volume.data.ptr)
            existing = bound.get(pointer)
            if existing is not None and existing.startswith("pole "):
                collisions.append(
                    f"{label} {index} and {existing} are the same allocation, and the "
                    f"second is bound __restrict__")
    if collisions:
        raise ValueError(
            "the no-PML dispersive fused electric pair binds every field and pole "
            "argument __restrict__, and these arguments alias, which is undefined "
            "behaviour NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_no_pml_dispersive_fused_electric_pair(
        fields: "Fields", boundary_codes: Sequence[Any], walls: Sequence[int],
        dtdx: float, poles: Sequence[Any], counts: Sequence[int],
        conductivity: Sequence[Any], cond: Sequence[bool],
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch, at this run's mask and arity.

    THE ARGUMENT ORDER IS THE SIGNATURE'S.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is: a gate
    compiles a deliberately broken copy of the shipped source and hands it here. A
    launcher that could not be handed its own kernel could not arm a single mutation.
    """
    counts = dispersive_kernels.normalized_pole_counts(counts)
    if len(tuple(poles)) != sum(counts):
        raise ValueError(
            f"{len(tuple(poles))} pole pointers were resolved against the compiled "
            f"arity {tuple(counts)} (sum {sum(counts)}); the signature binds exactly "
            f"one pointer per contributor and a mismatch would shift every argument "
            f"after the group")
    if len(tuple(conductivity)) != 6:
        raise ValueError(
            f"{len(tuple(conductivity))} conductivity pointers were resolved, not 6; "
            f"the signature binds condfac and condinv per target and a mismatch would "
            f"shift every argument after the group")
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + tuple(
        getattr(fields, name) for name in _CURL_SOURCES
    ) + tuple(conductivity) + tuple(
        getattr(fields, name) for name in _ELECTRIC
    ) + tuple(inverse_epsilon_bindings(fields)) + tuple(poles) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3))
    (kernel or _get_kernel(cond, counts))((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "arity": tuple(int(v) for v in counts),
            "cond": tuple(bool(flag) for flag in cond),
            "poles_bound": len(tuple(poles)),
            "walls": tuple(int(bool(walls[axis])) for axis in range(3))}


def run_no_pml_dispersive_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        sources: Any = None,
        plan: Optional[Dict[str, List[Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``plan`` and ``kernel`` are the gate's doors, keyword-only. ``plan`` is how the
    pole ORDERING mutation is armed: a harness hands in a reversed chain, which a
    launcher that always resolved its own could not be given.
    """
    covered, reason = covers_no_pml_dispersive_fused_electric_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if plan is None:
        poles, counts = pole_bindings(fields)
    else:
        counts = dispersive_kernels.pole_counts_of(plan)
        poles = tuple(array for target, _s, _a in dispersive_kernels.ELECTRIC_TERMS
                      for array in plan[target])
    conductivity, cond = conductive_bindings(fields)
    assert_disjoint_bindings(fields, poles, conductivity)
    from . import step_curl_kernels  # noqa: PLC0415 - a device import
    return launch_no_pml_dispersive_fused_electric_pair(
        fields, step_curl_kernels.real_curl_boundary_codes(grid),
        zero_metal_axes(grid), dtdx, poles, counts, conductivity, cond, kernel)
