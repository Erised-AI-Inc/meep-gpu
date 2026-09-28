"""A CONDUCTIVITY ON A CURL TARGET, for the hand-CUDA track: five kernels, one predicate.

WHAT THIS CLOSES, AND WHY IT IS THE RESIDUE OF TWO OTHER FAMILIES
=================================================================
``stepping._apply_curl`` (stepping.py:489-539) has FOUR tails. The certified pair
owns the second (``_apply_pml_update``, PML and no sigma) and ``no_pml_curl.py``
owns the fourth (``target -= curl``, neither). The other two are read through
``fields.condfac_for(term.target)`` at stepping.py:508 and are this file's::

    if pml_active:
        ...
        if condfac is not None:
            _apply_conductive_pml_update(...)          # :491-505   <- HERE
        else:
            _apply_pml_update(...)                     # :506
    elif condfac is not None:
        _apply_conductive_update(...)                  # :507-508   <- HERE
    else:
        target -= curl                                 # :509-510

``no_pml_curl.py``'s own record names those six no-PML curl slots as "the largest
thing it did not build" (``NO_PML_CURL_ADMISSION['what_it_does_not_license']``),
and the certified curl predicate refuses the seventh by name ("Dx carries a
conductivity: routes to the three-history conductive-PML recurrence",
coverage.py:978-980). MEASURED on the 186-row census
(``results/cuda_predicate_coverage_2026-08-20_closeout``) those are exactly SEVEN
slots on THREE rows, and every one of them is a MEEP ``Absorber`` or a
``D_conductivity`` — the absorber IS a conductivity, not a PML:

* ``absorber-1d.py`` and ``TestAbsorber.test_absorber`` — (1,1,400),
  periodic/periodic/metallic, NO active layer, sigma on BOTH sides, ``stores_E``
  True (5 registered poles). ``step_B`` + ``step_D`` = 4 slots.
* ``TestAbsorber.test_absorber_2d`` — (200,200,1), metallic/metallic/periodic, NO
  active layer, sigma on BOTH sides, ``stores_E`` FALSE. 2 slots, and its
  ``step_B`` is the DERIVED-E arm.
* ``TestAdjointSolver.test_damping`` — (150,150,1), metallic/metallic/periodic,
  PML ACTIVE, a D conductivity and no B conductivity. ``step_D`` only: 1 slot.
  Its ``step_B`` is already served by the certified curl, which is why the
  predicate below must refuse a sub-step whose targets carry no sigma — admitting
  it would be an OVERLAP, and the union census scores an overlap as a widening
  rather than as coverage.

THE PER-COMPONENT QUESTION, DECIDED BY MEASUREMENT RATHER THAN BY READING
========================================================================
``_apply_curl`` reads the conductivity PER TARGET COMPONENT, inside the loop over
that sub-step's three terms. One lossy component and two lossless therefore take
DIFFERENT tails inside ONE sub-step, and the brief this file answers asked whether
one kernel can serve that.

IT CAN, AND THE SUBSTITUTION THAT WOULD MAKE IT UNNECESSARY IS NOT AVAILABLE.
The tempting collapse is to bind ``condfac = 1``/``condinv = 1`` volumes on the
lossless components and run the conductive tail everywhere. That is NOT the same
computation:

* ``((f * 1.0f) - curl) * 1.0f`` is bit-identical to ``f - curl`` in IEEE-754
  arithmetic, but the corpus is stepped under TWO float32 subnormal policies, and
  under ``meep_x86_flush`` a subnormal ``f`` is flushed by the identity multiply
  and is not flushed by the plain subtraction's operand load. The gate sweeps a
  ``subnormal_band`` value class for exactly this reason.
* It also allocates two full float32 volumes per lossless component that the array
  path never allocates (``Fields._set_conductivity_side`` stores ``None`` rather
  than a volume of zeros, fields.py:745-775, precisely so the plain path stays
  bit-identical rather than merely equal to within a multiply by 1.0).

So the choice is carried as a COMPILE-TIME constant per component -- ``COND0``,
``COND1``, ``COND2``, the same three the Triton sibling
(``triton_kernels/conductivity.py``) carries as ``tl.constexpr`` -- and where a
component is lossless the conductive branch is preprocessed out entirely and the
kernel emits the plain family's tail line VERBATIM. That is a construction, not an
argument, and :func:`kernel_source` is where it can be read off.

THE CURL HALF IS NOT RE-WRITTEN. IT IS THE SIBLING'S BYTES.
==========================================================
Conductivity enters ``_apply_curl`` only AFTER the curl is formed, so the term
table, the ghost rule, the stencil grouping, the ownership mask and the derived-E
source are unchanged. Rather than copy them -- which would give two sets of bytes
that are equal only until someone edits one -- each template here is built by
TRANSFORMING the shipped sibling's device text:

* the three no-PML kernels from ``no_pml_curl.kernel_source(...)``;
* the two PML kernels from ``step_curl_kernels``'s two string literals, READ with
  ``ast`` because that module imports ``cupy`` at scope and this one must stay
  importable on the census laptop.

Each transformation is a targeted replacement WITH AN EXACT COUNT CHECK
(:func:`_replace_exactly`), so a rename or a restructure in either sibling fails
loudly here instead of quietly forking the arithmetic. The only lines that differ
from the sibling are the kernel name, the added pointer parameters, and the three
tail lines -- and ``test_conductive_kernels`` re-derives the sibling's text from
this file's mask-0 build to pin that.

THE FOUR-CASE RECURRENCE, TRANSCRIBED
=====================================
Under an active layer the tail is ``stepping._apply_conductive_pml_update``
(stepping.py:2001-2109), which the module calls "the engine's most expensive
curl". The array path evaluates ALL FOUR of MEEP's subchunk cases over the whole
volume and selects with ``xp.copyto(..., where=)``; a kernel evaluates the
expressions and selects per cell. Those are the same bits only because the four
branch expressions are mutually independent -- ``first_only`` is copied off
``field_previous`` BEFORE ``direct`` mutates it in place (stepping.py:2091-2099)
-- which the sibling track measured rather than argued (18/18 over three shapes x
two courants x three targets, with a flattened-grouping control at 0/18;
``triton_kernels/conductivity.py`` module docstring).

The cases, with ``f`` = target, ``u`` = ``fu_<target>``, ``c`` =
``f_cond_<target>``, ``cu`` = the curl (already carrying dtdx), ``cf``/``ci`` =
``condfac``/``condinv``, ``(km1, si1)`` the ``dsig`` axis's pair and
``(km2, si2)`` the ``dsigu`` axis's::

    dsig  = (km1 != 1.0f) || (si1 != 1.0f)        # EXACT, not a tolerance
    dsigu = (km2 != 1.0f) || (si2 != 1.0f)

    A  dsig && dsigu   c <- ((c*cf) - cu) * ci
                       u <- (((u*km1) + c_new) - c) * si1
                       f <- (((f*km2) + u_new) - u) * si2
    B  dsigu only      u <- ((u*cf) - cu) * ci
                       f <- (((f*km2) + u_new) - u) * si2      c UNCHANGED
    C  dsig only       c <- ((c*cf) - cu) * ci
                       f <- (((f*km1) + c_new) - c) * si1      u UNCHANGED
    D  neither         f <- ((f*cf) - cu) * ci                 u AND c UNCHANGED

"UNCHANGED" is a requirement, not a don't-care: the array path restores ``u`` and
``c`` in their inactive regions (stepping.py:2107-2109) so a diagnostic read
cannot mistake unused arithmetic for state. In the kernel that is "do not store",
which is stronger and identical.

DO NOT FLATTEN THOSE PARENTHESES and do not let a multiply contract into an FMA.
``--fmad=false`` is carried for that reason, and the gate runs an UNGUARDED
control at the inexact courant and REPORTS whether it changed an answer rather
than asserting that it must.

THE PARTITION IS EXACT, AND IT IS THE SAME COMPARISON THE ARRAY PATH MAKES.
``dsig_active`` is a bare ``!= 1.0`` (stepping.py:2055-2058), so a kernel using
``>=`` or a tolerance is a defect the real tables cannot expose -- on the sibling
track's real ``2d_cond_pml`` layout, ``kms_x`` has 561/640 entries EXACTLY 1.0 and
561 within 1e-7 of 1.0, i.e. no near-one band. The gate drives that mutation with
a SYNTHETIC table carrying a near-one entry, for the same reason.

WHAT IS NOT HERE
================
* The constitutive sub-steps. ``fields.condfac_for`` is read in ``_apply_curl``
  and NOWHERE ELSE in ``stepping``; ``update_H``, ``update_E`` and
  ``_apply_constitutive_pml`` never mention it. A conductive run's constitutive
  halves are ordinary products and belong to the constitutive families, which
  already admit them (coverage.py:2530-2537 says so in as many words).
* Complex storage, cylindrical coordinates, a Bloch phase, BFAST, special_kz,
  chi2/chi3, a third simultaneous mirror plane -- each refused by a clause
  INHERITED from the certified predicate rather than restated here.
* Any dispatch. Nothing in ``meep_gpu`` imports ``cuda_kernels``, so a True
  verdict licenses a MEASUREMENT and not a production step. What has and has not
  run on a device is :data:`CONDUCTIVE_CURL_ADMISSION`, whose ``host`` is None
  until a device leg has written it.

IMPORTABLE WITHOUT CUPY, DELIBERATELY -- the predicate is consumed by the coverage
census, which runs on a laptop. ``cupy`` is imported INSIDE the launcher, never at
module scope.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import pathlib
from typing import Any, Dict, Optional, Tuple

from . import coverage as _coverage
from . import no_pml_curl as _no_pml_curl
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)

__all__ = (
    "CERTIFIED_KERNELS",
    "CONDUCTIVE_KERNELS",
    "CONDUCTIVE_CURL_ADMISSION",
    "CONDUCTIVE_SUB_STEPS",
    "UNCERTIFIED_KERNELS",
    "conductive_targets",
    "conductive_arm",
    "covers_conductive_curl",
    "kernel_source",
    "kernel_template",
    "set_kernel_template",
    "clear_kernel_cache",
    "step_conductive_curl",
)

#: The two curl sub-steps and the targets each writes, keyed exactly as
#: ``coverage.CURL_SUB_STEPS``. Duplicated here rather than imported so this file
#: reads on its own, and pinned equal to the sibling's by
#: ``test_conductive_kernels.test_sub_step_targets_match_coverage``.
CONDUCTIVE_SUB_STEPS: Dict[str, Tuple[str, str, str]] = {
    "step_B": ("Bx", "By", "Bz"),
    "step_D": ("Dx", "Dy", "Dz"),
}

#: The five kernels, in ``(sub_step, layer, arm)`` order.
CONDUCTIVE_KERNELS: Tuple[str, ...] = (
    "step_B_no_pml_conductive",
    "step_B_no_pml_conductive_derived",
    "step_D_no_pml_conductive",
    "step_B_pml_conductive",
    "step_D_pml_conductive",
)

#: Byte-identical to the array path with a device verdict behind it. The evidence is
#: :data:`CONDUCTIVE_CURL_ADMISSION` and ``certification.json``'s
#: ``cuda_conductive_2026-08-21`` block, which names the same artifact directory; a
#: name here without a record block is the failure ``test_kernel_partition.py``
#: exists to catch.
CERTIFIED_KERNELS = (
    "step_B_no_pml_conductive",
    "step_B_no_pml_conductive_derived",
    "step_D_no_pml_conductive",
    "step_B_pml_conductive",
    "step_D_pml_conductive",
)

#: Shipped but not gated. EMPTY, and the partition requires every shipped kernel to
#: be in exactly one of the two sets, so a kernel added to this file without a gate
#: verdict fails there rather than shipping unmeasured. Spelled as a dict WITHOUT a
#: type annotation for the reason ``constitutive_kernels.py`` records: the partition
#: readers walk the syntax tree so they run where there is no CuPy, and an annotated
#: assignment is an ``ast.AnnAssign`` the plain-assignment readers do not match --
#: annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}

#: Which kernel serves which ``(sub_step, layer, arm)``. ``step_D`` has ONE arm on
#: both layers because it never reads E; ``step_B``'s two no-PML arms are
#: ``no_pml_curl``'s, for its reason (``_read_component`` derives ``D * inv_eps``
#: whenever ``fields.stores_E`` is False, stepping.py:2440-2450). Under an ACTIVE
#: layer there is no derived arm at all: ``enable_pml_storage`` stores E
#: unconditionally (fields.py:678 -> :653 -> ``_stored_E``).
ARM_KERNELS: Dict[Tuple[str, str, str], str] = {
    ("step_B", "none", "stored_E"): "step_B_no_pml_conductive",
    ("step_B", "none", "derived_E"): "step_B_no_pml_conductive_derived",
    ("step_D", "none", "magnetic"): "step_D_no_pml_conductive",
    ("step_B", "active", "stored_E"): "step_B_pml_conductive",
    ("step_D", "active", "magnetic"): "step_D_pml_conductive",
}

#: Which kernels take an ``f_cond`` history. Only the PML pair does: without an
#: absorber ``Fields._ensure_conductive_pml_storage`` returns at its first line
#: (fields.py:731-733) and ``f_cond_*`` is never allocated, which is exactly why
#: ``conductivity.conductive_pml_curl_coverage`` on the sibling track refuses the
#: three no-PML rows -- a TRUE refusal, and the reason this file carries two
#: families rather than one.
_PML_KERNELS: Tuple[str, ...] = ("step_B_pml_conductive",
                                 "step_D_pml_conductive")


# ---------------------------------------------------------------------------
# The device text, built by transforming the siblings' bytes
# ---------------------------------------------------------------------------

def _replace_exactly(source: str, needle: str, replacement: str,
                     expected: int, what: str) -> str:
    """One targeted edit of a sibling's device text, with the count CHECKED.

    A silent zero-count replacement is the whole hazard of building a kernel out
    of another kernel's bytes: the file would still compile, still launch, and
    still be wrong in exactly the way the edit was meant to fix. Raising at IMPORT
    time instead means a rename in ``step_curl_kernels.py`` or ``no_pml_curl.py``
    breaks this module loudly.
    """
    found = source.count(needle)
    if found != expected:
        raise RuntimeError(
            f"building the conductive kernels: expected {expected} occurrence(s) of "
            f"{what} in the sibling's device text, found {found}. The sibling has "
            f"been restructured and this file must be re-derived, not patched.")
    return source.replace(needle, replacement)


def _pml_sibling_source(name: str) -> str:
    """One of ``step_curl_kernels``'s two certified curl strings, READ not imported.

    ``step_curl_kernels.py`` imports ``cupy`` at module scope, so importing it
    would make this file -- predicate and all -- unimportable on the census
    laptop. Parsing the sibling's source for its own string literal has neither
    the import problem nor the copy problem: it is the sibling's bytes by
    construction. This is ``no_pml_curl._sibling_prelude``'s technique, applied to
    a whole kernel rather than to the prelude.
    """
    path = pathlib.Path(__file__).with_name("step_curl_kernels.py")
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Name) and target.id == name
                and isinstance(node.value, ast.BinOp)):
            # ``_REAL_PML_PRELUDE + r'''...'''`` -- the prelude is already carried
            # by ``no_pml_curl``'s reader, so only the right operand is wanted.
            right = node.value.right
            if isinstance(right, ast.Constant) and isinstance(right.value, str):
                return right.value
    raise RuntimeError(
        f"step_curl_kernels.py no longer assigns <prelude> + <string> to {name}; "
        f"the conductive PML pair is derived from that string and cannot be built "
        f"without it")


#: The shared real-field prelude -- ``shift_up``, ``shift_dn``, ``pml_apply`` and
#: the three boundary codes -- taken from the sibling that already reads it out of
#: the certified module, so there is exactly ONE reader in the package.
_REAL_PML_PRELUDE = _no_pml_curl._REAL_PML_PRELUDE

#: ``derived_at`` / ``derived_up``: ``_read_component``'s multiply COMPOSED WITH
#: ``_shift_up``, in that order, taken from the same sibling. The order is the
#: arithmetic -- see ``no_pml_curl``'s header for the two mutations that arm it.
_DERIVED_PRELUDE = _no_pml_curl._DERIVED_PRELUDE

#: The two conductive tails, and nothing else. Both are transcriptions with a line
#: number, and both are written in the array path's OWN order of operations.
_COND_APPLY_PRELUDE = r'''
#if COND0 || COND1 || COND2
// stepping._apply_conductive_update (stepping.py:1938-1951), in its in-place order:
//   field *= condfac;  field -= curl;  field *= condinv
// i.e. f <- ((f * cf) - cu) * ci, MEEP step_generic.cpp:87-97, with
// cf = 1 - sigma*dt/2 and ci = 1/(1 + sigma*dt/2) (fields.py:821-822), both FULL
// VOLUMES of the grid shape. There is no fu and no f_cond on this path.
__device__ __forceinline__ void cond_apply(
    float* __restrict__ f, const float* __restrict__ cf,
    const float* __restrict__ ci, int idx, float curl
) {
    f[idx] = ((f[idx] * cf[idx]) - curl) * ci[idx];
}

#endif
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1938-1951->1985-1998

#: ``cond_pml_apply`` ONLY, and it is carried ONLY by the two PML templates.
#: Compiling it into the no-absorber kernels would be dead code with a name --
#: and the gate's battery arms every one of its expressions, so a mutation on
#: an unreachable copy would score UNCAUGHT for a reason about dead code rather
#: than about the family. Splitting the two preludes is what keeps every armed
#: site reachable from some launch.
_COND_PML_APPLY_PRELUDE = r'''
#if COND0 || COND1 || COND2
// stepping._apply_conductive_pml_update (stepping.py:1954-2062): MEEP's four
// subchunk cases, selected per cell instead of over the whole volume.
//
// THE PREDICATES ARE EXACT COMPARISONS, transcribed from stepping.py:2008-2011
//   dsig_active  = (kms   != 1.0) | (sinv   != 1.0)
//   dsigu_active = (kms_u != 1.0) | (sinv_u != 1.0)
// A tolerance here is a defect the REAL coefficient tables cannot expose, so the
// gate drives it with a synthetic table carrying a near-one entry.
//
// THE STORE PREDICATES ARE THE 'UNCHANGED' COLUMN of the case table: u is written
// only where dsigu, c only where dsig, f always. The array path spells the same
// thing as a restore of the prior value (stepping.py:2060-2062).
//
// DO NOT FLATTEN THESE PARENTHESES. The array path accumulates in place, one
// operation at a time, and each grouping below is one of those steps.
__device__ __forceinline__ void cond_pml_apply(
    float* __restrict__ f, float* __restrict__ u, float* __restrict__ c,
    const float* __restrict__ cf, const float* __restrict__ ci,
    int idx, float curl, float km1, float si1, float km2, float si2
) {
    float fv = f[idx];
    float uv = u[idx];
    float cv = c[idx];
    float cfv = cf[idx];
    float civ = ci[idx];

    bool dsig = (km1 != 1.0f) || (si1 != 1.0f);
    bool dsigu = (km2 != 1.0f) || (si2 != 1.0f);

    float c_new    = ((cv * cfv) - curl) * civ;            // cases A and C
    float u_cond   = ((uv * cfv) - curl) * civ;            // case B
    float u_split  = (((uv * km1) + c_new) - cv) * si1;    // case A
    float u_new    = dsig ? u_split : u_cond;
    float f_split  = (((fv * km2) + u_new) - uv) * si2;    // cases A and B
    float f_first  = (((fv * km1) + c_new) - cv) * si1;    // case C
    float f_direct = ((fv * cfv) - curl) * civ;            // case D

    f[idx] = dsigu ? f_split : (dsig ? f_first : f_direct);
    if (dsigu) u[idx] = u_new;
    if (dsig) c[idx] = c_new;
}
#endif
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1954-2062->2001-2109, 2008-2011->2055-2058, 2060-2062->2107-2109

#: The six (no-PML) or twelve (PML) extra pointers every conductive kernel takes,
#: inserted at the one line every certified curl body shares. ``cf``/``ci`` are the
#: condfac/condinv volumes; where a component is lossless the parameter is a
#: PLACEHOLDER the preprocessor guarantees is never dereferenced.
_ANCHOR = "    int nx, int ny, int nz, float dtdx,\n"

_COND_PARAMS = (
    "    const float* __restrict__ cf0, const float* __restrict__ cf1,\n"
    "    const float* __restrict__ cf2,\n"
    "    const float* __restrict__ ci0, const float* __restrict__ ci1,\n"
    "    const float* __restrict__ ci2,\n"
)

_FCOND_PARAMS = (
    "    float* __restrict__ fc0, float* __restrict__ fc1,\n"
    "    float* __restrict__ fc2,\n"
)


def _no_pml_template(sibling: str, name: str, targets: Tuple[str, str, str]) -> str:
    """One no-PML conductive template, from ``no_pml_curl``'s shipped bytes."""
    # THE HELPER GOES FIRST. Appended, it would be declared after its call
    # sites and the translation unit would not compile -- and the byte-equality
    # test slices from ``extern "C"``, so it would not have noticed.
    source = _COND_APPLY_PRELUDE + _no_pml_curl.kernel_source(sibling)
    source = _replace_exactly(source, f"void {sibling}(", f"void {name}(", 1,
                              "the kernel's name")
    source = _replace_exactly(source, _ANCHOR, _COND_PARAMS + _ANCHOR, 1,
                              "the scalar-parameter line")
    for index, target in enumerate(targets):
        plain = f"        {target}[idx] = {target}[idx] - curl;\n"
        guarded = (f"#if COND{index}\n"
                   f"        cond_apply({target}, cf{index}, ci{index}, idx, curl);\n"
                   f"#else\n"
                   f"{plain}"
                   f"#endif\n")
        source = _replace_exactly(source, plain, guarded, 1,
                                  f"{target}'s plain tail line")
    return source


def _pml_template(literal: str, sibling: str, name: str,
                  targets: Tuple[str, str, str],
                  pairs: Tuple[Tuple[str, str], ...]) -> str:
    """One PML conductive template, from ``step_curl_kernels``'s shipped bytes.

    ``pairs`` is each target's ``(dsig axis letter + index, dsigu axis letter +
    index)`` exactly as the certified body spells its ``pml_apply`` arguments --
    ``("y[j]", "z[k]")`` for the first target on both sides, and so on round
    ``vec.hpp``'s cycle. They are matched as TEXT rather than rebuilt, so a
    half-cell error in the sibling cannot be silently "fixed" here into a
    disagreement with the certified kernel.
    """
    source = (_REAL_PML_PRELUDE + _COND_PML_APPLY_PRELUDE
              + _pml_sibling_source(literal))
    source = _replace_exactly(source, f"void {sibling}(", f"void {name}(", 1,
                              "the kernel's name")
    source = _replace_exactly(source, _ANCHOR,
                              _COND_PARAMS + _FCOND_PARAMS + _ANCHOR, 1,
                              "the scalar-parameter line")
    for index, (target, (first, second)) in enumerate(zip(targets, pairs)):
        coefficients = (f"kms_{first}, sinv_{first}, kms_{second}, sinv_{second}")
        plain = (f"        pml_apply({target}, fu_{target}, idx, curl, "
                 f"{coefficients});\n")
        guarded = (f"#if COND{index}\n"
                   f"        cond_pml_apply({target}, fu_{target}, fc{index}, "
                   f"cf{index}, ci{index}, idx, curl, {coefficients});\n"
                   f"#else\n"
                   f"{plain}"
                   f"#endif\n")
        source = _replace_exactly(source, plain, guarded, 1,
                                  f"{target}'s plain pml_apply tail")
    return source


#: ``vec.hpp``'s cycle, as the certified bodies spell it: target 0 takes (y, z),
#: target 1 (z, x), target 2 (x, y), on BOTH sides. The index letter is the one
#: that axis's coefficient vector is indexed by in the certified text.
_COEFFICIENT_PAIRS: Tuple[Tuple[str, str], ...] = (("y[j]", "z[k]"),
                                                   ("z[k]", "x[i]"),
                                                   ("x[i]", "y[j]"))

#: The device text, by kernel name, WITHOUT the three ``COND`` defines --
#: :func:`kernel_source` prepends those. A MUTABLE MODULE GLOBAL on purpose: the
#: gate's source-mutation battery rewrites entries here and re-launches, and
#: :func:`_get_kernel` re-reads this map on every call so a rewritten string is
#: seen. A map memoized at first call would hand back the pre-mutation bytes
#: forever -- "a leg reporting a pass for a mutation it never applied".
_TEMPLATES: Dict[str, str] = {
    "step_B_no_pml_conductive": _no_pml_template(
        "step_B_no_pml_real", "step_B_no_pml_conductive",
        ("Bx", "By", "Bz")),
    "step_B_no_pml_conductive_derived": _no_pml_template(
        "step_B_no_pml_real_derived",
        "step_B_no_pml_conductive_derived", ("Bx", "By", "Bz")),
    "step_D_no_pml_conductive": _no_pml_template(
        "step_D_no_pml_real", "step_D_no_pml_conductive",
        ("Dx", "Dy", "Dz")),
    "step_B_pml_conductive": _pml_template(
        "_step_B_pml_real_kernel_code", "step_B_pml_real",
        "step_B_pml_conductive", ("Bx", "By", "Bz"), _COEFFICIENT_PAIRS),
    "step_D_pml_conductive": _pml_template(
        "_step_D_pml_real_kernel_code", "step_D_pml_real",
        "step_D_pml_conductive", ("Dx", "Dy", "Dz"), _COEFFICIENT_PAIRS),
}

#: ``--fmad=false`` is CORRECTNESS here, not tuning. The conductive tails are
#: chains of multiply-then-subtract -- ``((f * cf) - curl) * ci`` is a textbook
#: fused-multiply-add candidate at the inner grouping, and the array path rounds
#: the multiply and the subtraction separately. The gate runs an UNGUARDED control
#: at the inexact courant and RECORDS whether it changed an answer rather than
#: asserting that it must.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: Matching ``step_curl_kernels._REAL_PML_THREADS``. Every kernel here guards on
#: ``idx >= nx*ny*nz`` and touches only its own cell's outputs, so a different
#: value changes throughput and no bit.
_THREADS = 256


def kernel_template(name: str) -> str:
    """The device text for one kernel WITHOUT its ``COND`` defines."""
    if name not in _TEMPLATES:
        raise ValueError(
            f"no such conductive kernel {name!r}; have {sorted(_TEMPLATES)}")
    return _TEMPLATES[name]


def set_kernel_template(name: str, source: str) -> None:
    """Replace one kernel's device text -- the gate's mutation seam.

    Paired with :func:`clear_kernel_cache`: the compile memo keys on the SOURCE,
    so a mutated body is a miss and reaches NVRTC without the clear, but the clear
    is what makes the accounting ("how many constructions came from the mutated
    bytes") exact.
    """
    if name not in _TEMPLATES:
        raise ValueError(
            f"no such conductive kernel {name!r}; have {sorted(_TEMPLATES)}")
    _TEMPLATES[name] = source


def kernel_source(name: str, cond: Tuple[bool, bool, bool]) -> str:
    """The device text for one kernel at one per-component conductivity mask.

    ``cond`` is ``(COND0, COND1, COND2)`` -- whether this sub-step's first, second
    and third target carries a sigma, in ``CONDUCTIVE_SUB_STEPS`` order. It comes
    from :func:`conductive_targets`, which is also what the predicate reads, so the
    compile-time choice and the admission cannot disagree.

    THE ONLY THING THIS ADDS TO THE TEMPLATE IS THREE ``#define`` LINES. The two
    conductive tail helpers live IN the template, behind
    ``#if COND0 || COND1 || COND2``, so a fully lossless build still preprocesses
    them away AND a gate that mutates the template reaches them. They were outside
    the template for one device run and every ``cond_pml_apply`` mutation came back
    "matched 0 sites" -- a battery scoring NOT ARMED on the file's most consequential
    arithmetic, which is exactly the accounting the source-in-the-memo-key design
    exists to prevent.
    """
    template = kernel_template(name)
    defines = "".join(f"#define COND{index} {int(bool(flag))}\n"
                      for index, flag in enumerate(cond))
    return defines + template


def clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return _clear_cache()


def _get_kernel(name: str, cond: Tuple[bool, bool, bool]):
    """Compile (or fetch) one kernel. ``cupy`` is imported HERE, never at scope."""
    import cupy as cp  # noqa: PLC0415 - a device-only import in a laptop-importable module

    code = kernel_source(name, cond)
    key = _kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def conductive_targets(fields: Any, sub_step: str) -> Tuple[bool, bool, bool]:
    """Which of this sub-step's three targets carry a sigma. NOT a refusal.

    THE SINGLE PLACE THE QUESTION IS ASKED. The predicate reads this and so does
    :func:`kernel_source`'s mask, which is the discipline that keeps a clause and
    a compile-time choice from drifting apart -- the sibling track spells the same
    rule as ``conductivity.conductive_targets`` for the same reason.

    ``fields.condfac_for(component)`` is what ``stepping._apply_curl`` reads
    (stepping.py:508), so this asks the array path's own question and not a proxy
    for it (``fields.has_conductivity`` is per SIDE and would answer for the wrong
    granularity).
    """
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        raise AttributeError(
            "fields does not expose condfac_for; an unreadable conductivity table "
            "is not an absent one")
    return tuple(reader(component) is not None
                 for component in CONDUCTIVE_SUB_STEPS[sub_step])  # type: ignore[return-value]


def conductive_arm(fields: Any, pml: Any, sub_step: str) -> Tuple[str, str]:
    """``(layer, arm)`` for this run at this sub-step. A classification, never a refusal.

    ``layer`` is ``"active"`` or ``"none"``, taken from the same
    ``pml.is_active`` question ``stepping._pml_is_active`` asks (stepping.py:2498).
    ``arm`` follows ``fields.stores_E``, the exact flag
    ``stepping._read_component`` branches on (stepping.py:2440).
    """
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    layer = "active" if (pml is not None
                         and bool(getattr(pml, "is_active", False))) else "none"
    if sub_step == "step_D":
        return layer, "magnetic"
    if layer == "active":
        # ``enable_pml_storage`` stores E unconditionally, so there is no derived
        # arm under an active layer -- and the predicate refuses a run that claims
        # otherwise rather than launching a kernel that binds a None.
        return layer, "stored_E"
    return layer, "stored_E" if getattr(fields, "stores_E", False) else "derived_E"


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

class _LosslessFieldsProxy:
    """The run's fields, answering ``condfac_for`` None on every component.

    WHY A PROXY AND NOT A STRING FILTER, which is the same answer
    ``no_pml_curl._NoAbsorberFieldsProxy`` gives for a different clause: the
    certified predicate SHORT-CIRCUITS, and its conductivity clause is followed by
    the ``stores_E`` clause, the chi2/chi3 clause, the dimensionality clause and
    the int32 index-range clause. Forgiving the conductivity refusal out of the
    returned string would silently skip all four -- four real refusals turned into
    admissions by a filter that never claimed to touch them. Answering the question
    AT THE INPUT instead lets every later clause run.

    The conductivity itself is then checked HERE, by :func:`_conductivity_problem`,
    which asks the questions a launch actually depends on: that each conductive
    component has a float32 C-contiguous ``condfac`` AND ``condinv`` volume of the
    grid's shape, and -- under an active layer -- an ``f_cond`` history, allocated
    exactly where MEEP allocates one.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def condfac_for(self, component: str) -> Any:  # noqa: ARG002 - deliberately blind
        return None

    def condinv_for(self, component: str) -> Any:  # noqa: ARG002 - deliberately blind
        return None

    @property
    def has_conductivity(self) -> bool:
        return False

    @property
    def has_magnetic_conductivity(self) -> bool:
        return False

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_fields"), item)


class _NoConductiveHistoryProxy(_LosslessFieldsProxy):
    """:class:`_LosslessFieldsProxy` that also hides the ``f_cond`` histories.

    ``no_pml_curl._array_inventory`` refuses a run whose ``fu_*`` is allocated,
    because that family writes no auxiliary. It says nothing about ``f_cond``, and
    it does not need to -- without an absorber ``f_cond_*`` is never allocated
    (fields.py:731-733). Under an ACTIVE layer this proxy is not used at all. It
    exists so the no-PML delegation is asked about a fields object whose ONLY
    difference from the real one is the conductivity, which is what makes the
    delegated verdict mean what it says.
    """

    _HIDDEN = ("f_cond_Bx", "f_cond_By", "f_cond_Bz",
               "f_cond_Dx", "f_cond_Dy", "f_cond_Dz")

    def __getattr__(self, item: str) -> Any:
        if item in self._HIDDEN:
            return None
        return super().__getattr__(item)


def _conductivity_problem(fields: Any, grid: Any, sub_step: str, layer: str,
                          cond: Tuple[bool, bool, bool]) -> Optional[str]:
    """Why this run's conductivity is not one a launch can bind; None if it is.

    THE VOLUMES ARE POINTERS AND THEIR DTYPE IS THE ARITHMETIC. The array path
    forms ``field * condfac`` in ``result_type(field, condfac)``
    (stepping.py:1996), so a float64 ``condfac`` makes the whole tail a float64
    tail and this float32 kernel would be a DIFFERENT COMPUTATION rather than a
    different rounding. ``Fields._set_conductivity_side`` casts both to float32
    (fields.py:821-822), so a run that fails this check has had its tables
    replaced by something other than the setter.
    """
    xp = getattr(grid, "xp", None)
    try:
        shape = tuple(grid.shape)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return f"grid shape could not be read: {type(exc).__name__}: {exc}"

    targets = CONDUCTIVE_SUB_STEPS[sub_step]
    for index, component in enumerate(targets):
        history = f"f_cond_{component}"
        if not cond[index]:
            # A LOSSLESS COMPONENT MUST CARRY NO HISTORY. MEEP allocates
            # ``f_cond[c]`` exactly where ``s->conductivity[c][d]`` exists
            # (fields.py:724-743), so an allocated history on a lossless component
            # means something else built it and this kernel -- whose COND == 0
            # branch never writes it -- would leave it stale.
            if getattr(fields, history, None) is not None:
                return (f"{history} is allocated while {component} carries no "
                        f"conductivity: this kernel's COND == 0 branch writes no "
                        f"history and would leave it stale")
            continue
        for label, reader in (("condfac", "condfac_for"), ("condinv", "condinv_for")):
            call = getattr(fields, reader, None)
            if not callable(call):
                return f"fields does not expose {reader}"
            try:
                volume = call(component)
            except Exception as exc:  # noqa: BLE001
                return (f"{reader}({component!r}) raised "
                        f"{type(exc).__name__}: {exc}")
            problem = _coverage._array_problem(f"{label} for {component}", volume,
                                               xp, shape)
            if problem is not None:
                return problem
        if layer == "active":
            problem = _coverage._array_problem(history,
                                               getattr(fields, history, None),
                                               xp, shape)
            if problem is not None:
                return problem
        elif getattr(fields, history, None) is not None:
            return (f"{history} is allocated with no active layer: the no-PML "
                    f"conductive tail is one expression (stepping.py:1996-1998) "
                    f"and writes no history")
    return None


def covers_conductive_curl(fields: Any, pml: Any, grid: Any,
                           sub_step: str) -> tuple:
    """Whether the conductive curl kernels may serve this run at this sub-step.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. Returns ``(covered, reason)``.

    THE ADMISSION IS THE EXACT INVERSE OF THE TWO INCUMBENTS' CONDUCTIVITY CLAUSE,
    per sub-step, and that is what makes the three a PARTITION rather than a set
    with a hole and an overlap:

    * ``coverage.covers_real_pml_curl`` refuses when ANY target of THIS sub-step
      carries a sigma (coverage.py:972-980).
    * ``no_pml_curl.covers_no_pml_curl`` delegates to it and inherits the same
      clause.
    * This predicate refuses when NO target of this sub-step carries one.

    So ``TestAdjointSolver.test_damping`` -- a D conductivity and no B one -- is
    the certified pair's at ``step_B`` and this family's at ``step_D``, with no
    slot admitted twice. The census checks disjointness rather than assuming it,
    and an overlap there is scored as a widening, not as coverage.

    DELEGATION, NOT A SECOND CLAUSE SET. Every question except the conductivity
    is the certified curl predicate's, asked through it -- directly under an active
    layer, and through ``no_pml_curl``'s own delegation when there is none, so the
    no-absorber storage inventory (no stored H, no ``fu``, a derived E) is that
    file's and not a second copy of it.
    """
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, "
                         f"got {sub_step!r}")

    try:
        cond = conductive_targets(fields, sub_step)
    except AttributeError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"fields could not be asked for a conductivity: "
                       f"{type(exc).__name__}: {exc}")
    if not any(cond):
        targets = CONDUCTIVE_SUB_STEPS[sub_step]
        return False, (f"no target of {sub_step} carries a conductivity "
                       f"({', '.join(targets)} are all lossless): a lossless curl "
                       f"belongs to covers_real_pml_curl or covers_no_pml_curl, "
                       f"and admitting it here would be an overlap")

    layer, _arm = conductive_arm(fields, pml, sub_step)
    problem = _conductivity_problem(fields, grid, sub_step, layer, cond)
    if problem is not None:
        return False, problem

    if layer == "active":
        covered, reason = _coverage.covers_real_pml_curl(
            _LosslessFieldsProxy(fields), pml, grid, sub_step)
        return (True, "") if covered else (False, reason)

    covered, reason = _no_pml_curl.covers_no_pml_curl(
        _NoConductiveHistoryProxy(fields), pml, grid, sub_step)
    return (True, "") if covered else (False, reason)


# ---------------------------------------------------------------------------
# The launcher
# ---------------------------------------------------------------------------

def step_conductive_curl(fields: Any, pml: Any, sub_step: str, boundary_codes,
                         dtdx: float, tables: Optional[Dict[str, Any]] = None,
                         arm: Optional[Tuple[str, str]] = None,
                         inverse_epsilon: Optional[Tuple[Any, Any, Any]] = None,
                         conductivity: Optional[Dict[str, Any]] = None) -> str:
    """Launch the conductive curl for one sub-step. Returns the kernel launched.

    ``boundary_codes`` is the triple ``coverage.real_curl_boundary_codes`` builds
    -- the SAME function the certified pair's launcher calls, so the launcher and
    the predicate cannot disagree about what a folded axis resolves to.

    ``tables`` is required under an ACTIVE layer and is
    ``step_curl_kernels.real_pml_curl_tables(pml, half_integer)`` -- the flattened
    coefficient views, built once per frozen configuration rather than per launch.
    ``half_integer`` is True for ``step_B`` and False for ``step_D``; getting that
    backwards is a half-cell error, not a crash.

    ``arm``, ``inverse_epsilon`` and ``conductivity`` are GATE SEAMS: the run's own
    values are used unless a host-mutation leg substitutes corrupted ones, so the
    defect it plants is on the POINTERS a launch binds rather than on the fields
    object every other leg shares.
    """
    import numpy as np  # noqa: PLC0415 - only the scalar types are needed

    layer, arm_name = arm or conductive_arm(fields, pml, sub_step)
    name = ARM_KERNELS[(sub_step, layer, arm_name)]
    targets_names = CONDUCTIVE_SUB_STEPS[sub_step]
    cond = conductive_targets(fields, sub_step)
    kernel = _get_kernel(name, cond)

    targets = tuple(getattr(fields, component) for component in targets_names)

    if sub_step == "step_B":
        if arm_name == "stored_E":
            sources = (fields.Ex, fields.Ey, fields.Ez)
            derived: Tuple[Any, ...] = ()
        else:
            sources = (fields.Dx, fields.Dy, fields.Dz)
            derived = tuple(inverse_epsilon) if inverse_epsilon is not None else tuple(
                fields.inverse_epsilon_for(component)
                for component in ("Ex", "Ey", "Ez"))
    else:
        if layer == "active":
            # Under an active layer H is STORED (``enable_pml_storage``), and the
            # certified D kernel differences exactly those arrays.
            sources = (fields.Hx, fields.Hy, fields.Hz)
        else:
            # ``get_H``'s no-absorber alias, spelled out: mu = 1, nothing stored,
            # ``fields.Hx`` is None and the array path differences B.
            sources = (fields.Bx, fields.By, fields.Bz)
        derived = ()

    supplied = conductivity or {}
    condfac = tuple(
        supplied.get(f"condfac_{component}",
                     fields.condfac_for(component) if flag else target)
        for component, flag, target in zip(targets_names, cond, targets))
    condinv = tuple(
        supplied.get(f"condinv_{component}",
                     fields.condinv_for(component) if flag else target)
        for component, flag, target in zip(targets_names, cond, targets))

    # THE ARGUMENT ORDER IS THE GATED ONE. Both branches are written out rather
    # than assembled incrementally because the device record was cut against this
    # exact sequence; a reordering here would be a wrong launch that no template
    # digest could catch.
    extra: Tuple[Any, ...] = tuple(condfac) + tuple(condinv)
    if name in _PML_KERNELS:
        histories = tuple(
            getattr(fields, f"f_cond_{component}") if flag else target
            for component, flag, target in zip(targets_names, cond, targets))
        auxiliaries = tuple(getattr(fields, f"fu_{component}")
                            for component in targets_names)
        if tables is None:
            raise ValueError(
                "an active layer needs the six flattened coefficient vectors; pass "
                "step_curl_kernels.real_pml_curl_tables(pml, half_integer) as "
                "`tables` -- half_integer is True for step_B and False for step_D")
        coefficients = tuple(tables[f"{label}_{axis}"]
                             for axis in ("x", "y", "z")
                             for label in ("kms", "sinv"))
        head: Tuple[Any, ...] = targets + auxiliaries + sources
        extra = tuple(condfac) + tuple(condinv) + histories
        tail: Tuple[Any, ...] = coefficients
    else:
        head = targets + sources + derived
        tail = ()

    nx, ny, nz = targets[0].shape
    blocks = (nx * ny * nz + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,), (
        *head,
        *extra,
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
        *tail,
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ))
    return name


#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, AND WHAT HAS NOT.
#:
#: ``host`` is None until a device leg has run. Nothing in this file may be read
#: as a device verdict while it is None -- the same all-or-nothing rule
#: ``no_pml_curl.NO_PML_CURL_ADMISSION`` and
#: ``no_pml_constitutive.NULL_CONSTITUTIVE_CONFIRMATION`` carry, and
#: ``test_conductive_kernels.test_admission_record_is_all_or_nothing`` is what
#: keeps the two states from blurring.
CONDUCTIVE_CURL_ADMISSION: dict = {
    "gate": "parity/meep_gpu/gate_cuda_conductive.py",
    "artifacts": "parity/meep_gpu/results/cuda_conductive_2026-08-21_rename",
    "artifact_sha256": {
        "keep": "86a569715786c414501f65897006c5be09df6cbb487128ca6a954cca2b1b965d",
        "flush": "0ebc364b58ee0d57130f50f9c569ef63b0d91fbf7b5ec3b9a2b6ef5808e2de98",
        "falsify": "0bcdb344a6fcbeda3d345e039beef89a3c5bd859e152ab2d7a8182c4d2b211f9",
    },
    "recorded_utc": "2026-08-21T23:17:19Z",
    "host": "the GPU host",
    "device": ("NVIDIA RTX A6000 (index 3, verified physically empty before each "
               "leg), CuPy 13.5.1, NVRTC 11.6"),
    # PER POLICY, and identical under both. 972 cases are scored per leg; the 944
    # below are the LICENSED substitution (the shipped code resolution), which is
    # the only triple a dispatch could produce. The other 28 are the two
    # misdeclaration controls and are meant to diverge -- and did, 4/4 and 24/24.
    "cases_scored": 972,
    "single_launch_identical": 944,
    "multi_step_identical": 944,
    "multi_step_budget": 60,
    "arms_swept": ("step_B/none/stored_E", "step_B/none/derived_E",
                   "step_D/none/magnetic", "step_B/active/stored_E",
                   "step_D/active/magnetic"),
    "arm_case_counts": {"step_B/none/stored_E": 160, "step_B/none/derived_E": 160,
                        "step_D/none/magnetic": 320,
                        "step_B/active/stored_E": 152,
                        "step_D/active/magnetic": 152},
    # THE MASK IS THE EXPERIMENTAL VARIABLE and every one of the five was swept.
    # A run that swept only 'CCC' would say nothing about the configuration MEEP's
    # own allocation granularity makes ordinary.
    "conductivity_masks_swept": ("CCC", "C--", "-C-", "--C", "C-C"),
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "courants": (0.5, 0.35),
    "value_classes": ("uniform", "subnormal_band"),
    "shapes_swept": ("3-D", "2-D", "1-D"),
    "walls_swept": ("periodic", "metallic", "mixed", "mirror fold at both terminations"),
    "absorber_shapes_swept": ("none", "inert_layer", "active layer"),
    "coefficient_tables_swept": ("shipped", "near_one"),
    # THE TWO POLICIES ARE DIFFERENT BUILDS, not one build named twice: the NVRTC
    # observer recorded 191 calls and 177 distinct binaries on each leg, with
    # ``any_ftz_true_reached_nvrtc`` FALSE under keep and TRUE under flush.
    "nvrtc_calls_per_leg": 191,
    "distinct_binaries_per_leg": 177,
    "ftz_true_reached_nvrtc": {"keep": False, "flush": True},
    # THE TEMPLATE DIGESTS, not the file's: a record written after a gate ran
    # cannot also be inside the bytes that gate stamped, and pinning the whole
    # file would make every comment edit read as an un-gated kernel change.
    "kernel_template_sha256": {
        "step_B_no_pml_conductive":
            "d3886fa2acb398bef772d65988bf5879f8f82baedb764397adab3fb920689824",
        "step_B_no_pml_conductive_derived":
            "6cecd686e34fe707e644e17ca703da3967b16a30311df1f21da35932afa747b4",
        "step_D_no_pml_conductive":
            "211f848e6e2800d3cd51e18f2da68bed81a54768a1844b0aa0d6dff74a78c301",
        "step_B_pml_conductive":
            "ed3c7d923ba9c9f0077f2b393cc4a9d1756a20cc2168ee946cb8fb43cc9e4079",
        "step_D_pml_conductive":
            "79dbb652ba1afe1c983cde3fc4f18a7cff8a8cb9e5ac875d7f1a821548b93d93",
    },
    "source_mutations_caught": 45,
    "source_mutations_null_confirmed": 13,
    "source_mutations_escaped": 0,
    "source_mutations_not_armed": 32,
    "host_mutations_caught": ("swap_condfac_and_condinv_volumes",
                              "alias_every_component_to_the_first"),
    "host_mutations_null_confirmed": ("identity_rebind",),
    # THE FOUR NULLS, and what their silence measures. Each edits the SAME
    # expression as a defect that must be caught, so "inert" is a measurement
    # rather than a claim about a leg nobody showed could fail:
    #  * cond_operand_order      <-> drop_condfac       (IEEE multiply commutes)
    #  * cond_pml_scale_commutes <-> flatten_split_grouping
    #  * commute_dtdx_scale      <-> regroup_stencil
    #  * fortran_order_index_decomposition_on_a_flat_grid <-> the same transform
    #    on the shapes where the two decompositions differ. On (1, 1, 24) they
    #    coincide cell for cell, which is MEEP's stride(d) = 0 for a direction it
    #    does not have showing up in a mutation score.
    "null_mutations_confirmed": ("cond_operand_order", "cond_pml_scale_commutes",
                                 "commute_dtdx_scale",
                                 "fortran_order_index_decomposition_on_a_flat_grid"),
    # THE EXACT-COMPARISON CLAIM, MEASURED AS A CONTRAST rather than asserted.
    # stepping.py:2055-2058 is a bare ``!= 1.0``; a real PML table carries no entry
    # within 1e-7 of 1.0 that is not exactly 1.0, so a tolerance kernel is a defect
    # the shipped tables CANNOT expose. Replacing both predicates with a 1e-7
    # tolerance was therefore UNCAUGHT 0/6 on the shipped table (expected, and
    # evidence of nothing) and CAUGHT 8/8 on a table with one entry set to
    # 0.99999994f -- the largest float32 below 1.0. Both numbers are in the record;
    # the per-table breakdown labels the near-one half "NULL BROKEN" because it is
    # scored through the same helper, and the mutation's own verdict is CAUGHT.
    "exact_predicate_contrast": {
        "shipped_table": "UNCAUGHT 0/6 per PML kernel -- there is no near-one band",
        "near_one_table": "CAUGHT 8/8 per PML kernel",
        "fixture": "one kms entry set to 0.99999994f, installed on the PML object "
                   "so the ORACLE and the KERNEL read the same table",
    },
    # THE FOUR-CASE TAIL WAS NOT TESTED ON ONE BRANCH. Every PML case asserts a
    # nonzero cell count in each of MEEP's four subchunk cases before it is
    # scored, and a fixture that reaches fewer is SKIPPED with the census printed
    # rather than counted as evidence for a branch it never ran. Reached over the
    # licensed sweep, per case, at least:
    "subchunk_cases_reached": {"A": 275, "B": 550, "C": 450, "D": 768},
    "misdeclaration_controls": {
        # A folded PERIODIC axis handed BC_METALLIC: the two codes share a ghost
        # rule, so this changes only the top-plane mask.
        "mirror_as_metallic": {"cases": 4, "diverged": 4, "differing_words": 821},
        # The blunt one: no mask and a wrapping ghost everywhere.
        "all_periodic": {"cases": 24, "diverged": 24, "differing_words": 10365},
    },
    # THE VERDICT WAS SHOWN TO GO RED four ways on the record itself (a licensed
    # case flipped to diverging, a must-be-caught mutation flipped to ESCAPED, a
    # subchunk case flipped to unreached, the misdeclaration control silenced) and
    # once for real.
    "verdict_flips_against_planted_defect": True,
    "falsification_leg": {
        "plant": ("the four-case selection collapsed -- "
                  "``dsigu ? f_split : (dsig ? f_first : f_direct)`` replaced by "
                  "``f_split``, i.e. MEEP's cases C and D served by case A/B's "
                  "expression"),
        "released": False,
        "pml_arms_identical": "0/4 single and 0/4 multi on step_B, 0/4 and 0/4 on step_D",
        "no_absorber_arms_identical": "4/4, 4/4 and 8/8, unaffected",
        "localization": ("the plant is in the two PML templates' digests and in "
                         "neither of the three no-absorber ones, and the "
                         "divergence follows exactly that split"),
    },
    # THE CONTRACTION GUARD IS LOAD-BEARING HERE, and that is the opposite of what
    # the no-absorber sibling measured for its own tail. Every conductive tail is a
    # chain of multiply-then-subtract -- ``(f * cf) - curl``, ``(u * km1) + c_new``
    # -- which the array path rounds one operation at a time. MEASURED at the
    # inexact courant (0.35): of 472 cases where the guarded leg was bit-identical,
    # the UNGUARDED build diverged on 463 under keep and 464 under flush.
    # ``no_pml_curl``'s record reads MEASURED DECORATIVE on the same compiler and
    # the same device; the difference is the tail, not the machine.
    "contraction_guard": {
        "flag": "--fmad=false",
        "comparable_cases": 472,
        "unguarded_diverged": {"keep": 463, "flush": 464},
        "reading": "LOAD-BEARING on this family, measured",
    },
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-20_conductive_v2"),
    # RECOMPUTED as a CONTROLLED comparison rather than a difference of two rounds:
    # the closeout round's analyzer (whose UNION_FAMILIES does not know this
    # family) run over THIS round's data gives 668/759, and the same data with the
    # family added gives 675/759. The two differ in one line of configuration, so
    # +7 slots and +2 rows-covered-at-every-sub-step is this family's contribution
    # and nothing else's. Disjointness was checked, not assumed: no slot is
    # admitted by two families on any of the 186 rows.
    #
    # THE ABSOLUTE FIGURES ARE A SNAPSHOT AND THE DELTA IS NOT. The same controlled
    # pair was taken twice on the same day against two states of a tree several
    # families were landing into: 653 -> 660 at 08:51 and 668 -> 675 at 10:33. The
    # absolutes moved by seven because other families widened; the delta did not
    # move at all, because none of the three rows this family serves carries a
    # chi2/chi3, complex storage or a fold for another family's widening to reach.
    "slots_before": 668,
    "slots_after": 675,
    "rows_covered_at_every_sub_step_before": 139,
    "rows_covered_at_every_sub_step_after": 141,
    "slots_gained": 7,
    "delta_reproduced_on_an_earlier_tree_state": {"before": 653, "after": 660,
                                                  "gained": 7},
    "rows_served": (
        "absorber-1d.py -- step_B + step_D (1-D, mp.Absorber, sigma on both sides)",
        "TestAbsorber.test_absorber -- step_B + step_D (the same configuration)",
        "TestAbsorber.test_absorber_2d -- step_B (DERIVED-E arm) + step_D",
        "TestAdjointSolver.test_damping -- step_D only, under an ACTIVE PML; its "
        "step_B is the certified pair's, because no B conductivity is installed",
    ),
    "what_it_does_not_license": (
        "any dispatch. Nothing in meep_gpu imports cuda_kernels, so a True verdict "
        "licenses a MEASUREMENT and not a production step.",
        "the constitutive sub-steps of a conductive run. fields.condfac_for is read "
        "in stepping._apply_curl and nowhere else in the module, so those are "
        "ordinary products and belong to the constitutive families.",
        "complex storage, cylindrical coordinates, a Bloch phase, BFAST, "
        "special_kz and a third simultaneous mirror plane -- each refused by a "
        "clause INHERITED from the certified curl predicate and untested here.",
        "a conductivity on a 1-D grid UNDER AN ACTIVE LAYER. A 1-D grid has one "
        "non-invariant axis and a target's two ladder axes are different axes, so "
        "MEEP's cases A and B are unreachable there; every such case was SKIPPED "
        "with its census printed rather than counted. The corpus carries no such "
        "row, so this is a gap in the sweep and not in the coverage.",
        "any throughput claim. The gate is a correctness gate and times nothing -- "
        "which matters more here than elsewhere, because the array path's "
        "four-case conductive curl is the engine's most expensive and the "
        "temptation to read a correctness result as a speed result is real.",
    ),
}
