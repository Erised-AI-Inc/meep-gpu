"""The COMPLEX no-absorber THREE-SLOT weld: ``step_D`` -> ``update_E`` -> ``update_P``
in ONE launch per component, the register hand-off carried across BOTH seams.

THE SECOND THREE-SLOT PRODUCT ON THIS TRACK, and the one the first one's record said
it was NOT: ``cuda_three_slot_dispersive_weld_2026-09-03`` names the four
``cuda_no_pml_complex_fused_electric_pair`` rows -- ``tests:TestLoadDump.test_load_
dump_{chunk_layout_file_3d, chunk_layout_sim_3d, structure_3d, structure_sharded_3d}``,
grid (35, 32, 41), complex64 storage, Bloch k = (0.4, -1.3, 0.7), periodic on all
three axes, NO absorber, a D AND a B conductivity, one Lorentzian driving Ex, Ey and
Ez, a MAGNETIC source only -- as "a DIFFERENT shape and not either of these arms",
because "their source is magnetic only, their storage is complex64 and their D->E
product declares no bracket; with nothing between update_E and update_P a register
hand-off is available there". This module is that shape.

=============================================================================
THE COLLISION IT RESOLVES, MEASURED
=============================================================================

On those four rows the composer selects, slot by slot, ``step_D`` = ``complex no-PML
curl``, ``update_E`` = ``complex no-PML stored E``, ``update_P`` = ``complex no-PML
ADE``, and the two fused products that admit them COMPETE for ``update_E``:

    cuda_no_pml_complex_fused_electric_pair        4 rows (D->E)
    cuda_complex_no_pml_fused_polarization_pair    the SAME 4 (E->P)

``fused_pairs._later_seam_claimant`` withholds the D->E product because the trade is
net zero, so the board carries these four instances as ``buildable_not_built`` at
``D_to_E`` (``results/fusion_matrix_cuda_2026-09-03_extended``). A product that owns
all three slots makes no trade and serves both seams -- exactly what the real
three-slot weld does for its ten rows.

=============================================================================
WHY THIS ONE MAY BE A SINGLE LAUNCH WHERE THE REAL WELD COULD NOT
=============================================================================

The real weld runs in TWO device groups with ``deposit_repair.apply`` between them,
because every one of its rows carries an ELECTRIC deposit inside the D seam and the
repair recomputes the constitutive product through
``Fields.displacement_minus_polarization``, which subtracts ``P`` AS IT STANDS. Its
probe measured the single-launch ordering (``p_inside_the_launch``) diverging on 6 of
7 configurations, at the deposit cells.

HERE THERE IS NO DEPOSIT AND NO REPAIR, and that is a predicate fact rather than a
hope: the D->E half this product conjoins declares ``CARRIES_DEPOSIT_REPAIR = False``
and refuses every electric in-seam source by name (all four rows of the cell declare
``source_field_types == ['B']``, injected one seam earlier). Between the ``step_D``
consult and the ``update_P`` consult the driver then runs NOTHING that reaches these
volumes on an admitted row: no electric injection, no mirror fill (the D->E half
refuses a fold), no wall clear (it refuses a wall), and nothing at all between
``update_E`` and ``update_P`` (driver.py, the two consults are adjacent). So the
three sub-steps are, per cell and per component, one dependency chain::

    curl_c  = dtdx * (shifts of B)                   reads B only
    D_c    <- conductive/plain tail (curl_c)          reads and writes D_c
    E_c     = (D_c - sum_i P_i,c) * inv_eps_c         reads D_c (register), P_c
    P_i,c  <- ADE(P_i,c, P_prev_i,c, E_c)             reads E_c (register), P_c, writes scratch

and nothing in it reads another component's D, E or P, or any cell but its own
(the curl's neighbour reads are of B, which no slot in the span writes). The
component-major split with host rotation between component launches is the
certified E->P weld's own shape (``complex_no_pml_fused_polarization_pair``, both
policies, 8/8), and welding the certified curl block for that component ahead of it
adds no read of anything the launch writes. The gate measures it anyway --
``parity/meep_gpu/gate_cuda_complex_three_slot_weld.py``, per complete driver step,
every stored volume and every P / P_prev / scratch buffer as uint32 words, against
the array path, the certified singles, and both two-slot arrangements -- because
"nothing reads what it writes" is the kind of claim this track has been wrong about.

=============================================================================
WHAT IS LIFTED, AND WHAT IS NEW
=============================================================================

NOTHING IS RE-DERIVED. Three certified bodies are spliced, each read off the module
that owns it at emission time so a drift raises here rather than compiling:

* the CURL BLOCK for one component, from :mod:`.no_pml_complex_fused_electric_pair`'s
  ``certified_curl_body`` -- itself the certified ``step_D_no_pml_complex[_conductive]``
  with its tail value-named (``no_pml_apply_reg`` / ``conductive_apply_reg``), so
  ``d<c>`` is the register the curl stored. That module's prelude is taken whole,
  which is what brings ``minus_poles_reg``, both ``_reg`` tails and ``ade_step`` in;
* the CONSTITUTIVE STORE for that component, from the same module's
  ``certified_constitutive_body`` -- ``minus_poles_reg(d<c>, <bank>, idx, np<c>)``
  times ``inv_eps_<c>``, FIELD on the left -- with the ONE edit the E->P weld makes to
  the certified store: the value is NAMED (``cf ev``) before it is stored, so the
  recurrence reads it from a register;
* the ADE RECURRENCE per driving state, :mod:`.complex_no_pml_fused_polarization_pair`'s
  own lifted line under its own per-pole renames and ITS seam: the drive load is
  ``ev``. A complex64 word pair stored to global and reloaded is the identity on the
  bits, which that weld's gate measured as a NULL (``reload_drive_from_global``) and
  this one's measures again.

The only hand-written device text is the signature. :data:`LIFT_EDITS` is the whole
list of edits, as data.

=============================================================================
THE LAUNCH: THREE PER STEP, HOST ROTATION BETWEEN
=============================================================================

One launch per E component, ALWAYS three (a component with no driving state still
has a curl and a constitutive), component-major, with the certified rotation
(dispersion.py:689-691 -- ``P <- scratch``, ``P_prev <- P``, ``scratch <- P_prev``)
performed on the host once per driving state after each launch RETURNS. Every
buffer is re-read after the previous component's rotation, never cached; that is the
E->P weld's launcher, transcribed. The array path performs the same three sub-steps
in 3 + 1 + sum(driven) launches through the certified singles.

=============================================================================
HOW THIS STAYS OUT OF THE TWO PRODUCTS' WAY, AND WHERE THE ARBITRATION IS
=============================================================================

Its predicate is the CONJUNCTION of both shipped predicates, so its admission set is
a subset of each; ``fused_pairs._superseded_by_a_longer_span`` refuses the D->E pair
by name where this admits, and ``_pair_may_absorb`` fails the E->P pair closed out of
the ``update_E`` slot this product already holds -- the same two rules the real
three-slot weld landed with, and the board's ``_loses_the_shared_slot`` withholds
nothing from a column that serves both seams itself.

THE THIRD CONSULT IS ANSWERED, NOT LAUNCHED. ``fused_pairs`` installs this as a
triple (leading half in ``step_D``, ``NoopPlan`` in ``update_E`` because the seam is
empty, trailing half in ``update_P``); the leading half performs all three sub-steps
and the trailing half is a NO-OP that must still answer the consult, because a
``False`` there would make the driver run ``update_P`` on the array path a second
time. That the trailing consult may be empty is the same fact as the single launch
being byte-identical, and the gate's ``p_at_update_P_consult`` arm measures the
converse: advancing P at the trailing consult instead is ALSO byte-identical here,
which is the measurement that nothing sits between the two.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` reaches no hand-CUDA product. This module ships
an emitter, a predicate and a launcher; ``fused_pairs`` can plan it opt-in
(``arms.plan_step(..., fuse=True)``), and no shipped dispatch path launches it.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None

try:
    from . import compile_cache
    from . import complex_emitter
    from . import complex_no_pml_kernels as certified
    from . import complex_no_pml_fused_polarization_pair as ep
    from . import no_pml_complex_fused_electric_pair as de
    from .coverage import _base_address, ade_sigma_is_volume
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.machinery as _machinery
    import importlib.util as _importlib_util
    import sys as _sys

    def _load(stem):
        here = os.path.dirname(os.path.abspath(__file__))
        package = "cuda_kernels_bypath"
        if package not in _sys.modules:
            spec = _machinery.ModuleSpec(package, None, is_package=True)
            shim = _importlib_util.module_from_spec(spec)
            shim.__path__ = [here]
            _sys.modules[package] = shim
        name = f"{package}.{stem}"
        if name in _sys.modules:
            return _sys.modules[name]
        spec = _importlib_util.spec_from_file_location(
            name, os.path.join(here, f"{stem}.py"))
        module = _importlib_util.module_from_spec(spec)
        _sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    compile_cache = _load("compile_cache")
    complex_emitter = _load("complex_emitter")
    certified = _load("complex_no_pml_kernels")
    ep = _load("complex_no_pml_fused_polarization_pair")
    de = _load("no_pml_complex_fused_electric_pair")
    _coverage = _load("coverage")
    _base_address = _coverage._base_address
    ade_sigma_is_volume = _coverage.ade_sigma_is_volume


# =============================================================================
# THE PARTITION -- FLIPPED AHEAD OF THE GATE, 2026-09-04
# =============================================================================
#
# Plain assignments with NO type annotation: the partition readers walk the syntax
# tree without importing the module.

#: RELEASED 2026-09-04 under BOTH float32 subnormal policies by
#: ``parity/meep_gpu/gate_cuda_complex_three_slot_weld.py`` on one RTX A6000: the
#: block is ``cuda_complex_no_pml_three_slot_dispersive_weld_2026-09-04`` in
#: ``certification.json``, and the partition was moved BEFORE the run so the gate
#: measured the bytes that ship (the 2026-09-03 lesson: this module's kernel text is
#: composed at call time, so ``device_identity`` applies the code tier to it and a
#: flip after the run would have drifted the pinned bytes). TWO names because the
#: certified curl half ships two arms -- a plain tail and a conductive one -- and the
#: corpus cell takes the conductive one on every row.
CERTIFIED_KERNELS = (
    "three_slot_no_pml_complex",
    "three_slot_no_pml_complex_conductive",
)
#: EMPTY since 2026-09-04; what emptied it was the run above.
UNCERTIFIED_KERNELS = {}

#: FALSE, AND IT IS THE PRODUCT'S WHOLE SHAPE. All four rows of the cell declare a
#: magnetic source only, so the D seam is empty on 4 of 4 and no bracket is installed;
#: the D->E half this predicate conjoins refuses every electric in-seam source by name
#: (``deposit_repair.seam_source_reasons`` with its own ``carries_repair=False``), so
#: a run that grew one falls back to the array path rather than being served wrongly.
#: That refusal is what LICENSES the single launch: with a deposit inside the span the
#: constitutive and the recurrence would both read a pre-injection field, and the real
#: weld's probe measured that ordering diverging.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_three_slot_complex_no_pml_dispersive_weld"

#: The kernel LABEL, in the bracket spelling the certified curl uses for its two arms.
#: A label and NOT a symbol: the two symbols are :data:`KERNEL_KEYS`' values.
KERNEL_NAME = "three_slot_no_pml_complex[_conductive]"

#: The ``extern "C"`` symbol each curl arm's weld is emitted under, keyed by the arm
#: name ``complex_no_pml_kernels.complex_no_pml_curl_arm`` returns.
KERNEL_KEYS: Dict[str, str] = {
    "plain": "three_slot_no_pml_complex",
    "conductive": "three_slot_no_pml_complex_conductive",
}

#: The module that owns the emitted text and the gate's doors: this one.
KERNEL_OWNER_MODULE = "complex_no_pml_three_slot_dispersive_weld"

#: The driver passes the three per-component launches jointly perform, in driver
#: order. THREE -- no fill and no wall clear, because the D->E half refuses a fold and
#: a wall by name, and nothing sits between ``update_E`` and ``update_P``.
REPLACES: Tuple[str, ...] = ("step_D", "update_E", "update_P")

#: The slot the composer holds this on (the span's first).
SLOT = "step_D"

#: ALL THREE slots this product owns, in driver order.
SLOTS: Tuple[str, str, str] = ("step_D", "update_E", "update_P")

#: The certified bank width, inherited.
MAX_POLES = certified.MAX_POLES

#: (stored E, displacement, bank stem, count name) per component -- the E->P weld's
#: own table, re-read rather than retyped.
_COMPONENTS = ep._COMPONENTS

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "(the certified curl body: three braced component blocks in one launch)",
     "became": "(ONE braced component block per launch, the other two absent)",
     "why": "the launch is per component so the recurrence can read this component's "
            "constitutive product from a register. Each block reads B at its own cell "
            "and two neighbours, writes only its own D, and nothing in the span "
            "writes B; dropping the other two blocks removes no read this block "
            "makes. The block's text, its tail (`d<c> = conductive_apply_reg(...)` / "
            "`no_pml_apply_reg(...)`) and the header that declares d0/d1/d2 are the "
            "D->E weld's own emission, character for character."},
    {"line": "    cf_store(h<c>, idx, mul_field_left(\n"
             "        minus_poles_reg(d<c>, <bank>, idx, np<c>),   // THE SEAM\n"
             "        inv_eps_<c>[idx]));",
     "became": "    cf ev = mul_field_left(\n"
               "        minus_poles_reg(d<c>, <bank>, idx, np<c>),   // THE SEAM\n"
               "        inv_eps_<c>[idx]);\n"
               "    cf_store(h<c>, idx, ev);",
     "why": "same expression tree, same store; the value is additionally NAMED so the "
            "recurrence can read it from a register -- the E->P weld's one edit to the "
            "certified store, applied to the D->E weld's spelling of the same line."},
    {"line": "    cf_store(p_out, idx, ade_step(\n"
             "        cf_load(p_now, idx), cf_load(p_prev, idx), cf_load(drive, idx),\n"
             "        sigma[idx], c_now, c_prev, c_drive));",
     "became": "    cf_store(p_out_i, idx, ade_step(\n"
               "        cf_load(<stem>i, idx), cf_load(p_prev_i, idx), ev,\n"
               "        sigma_i[idx], c_now_i, c_prev_i, c_drive_i));",
     "why": "the E->P weld's per-pole renames and ITS seam, verbatim: the drive load "
            "becomes the register the constitutive just stored (fields.drive_field(c) "
            "IS the stored E without an absorber) and p_now becomes the bank slot "
            "minus_poles_reg already reads through -- the pole buffer bound ONCE."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_KEYS",
    "KERNEL_NAME", "KERNEL_OWNER_MODULE", "LIFT_EDITS", "MAX_POLES", "REPLACES",
    "SLOT", "SLOTS", "UNCERTIFIED_KERNELS", "assert_disjoint_bindings",
    "certified_curl_block", "component_specs",
    "covers_three_slot_complex_no_pml_dispersive_weld", "device_sources",
    "kernel_name", "launch_three_slot_component", "run_three_slot_no_pml_complex",
    "three_slot_no_pml_complex_source",
]


# =============================================================================
# THE LIFT
# =============================================================================

#: The marker each component block of the D->E weld's certified curl body opens with.
_BLOCK_MARKER = "\n    // Target "


def kernel_name(arm: str) -> str:
    """The ``extern "C"`` symbol NVRTC is asked for, by curl arm; a refusal otherwise."""
    if arm not in KERNEL_KEYS:
        raise ValueError(
            f"arm must be one of {sorted(KERNEL_KEYS)}, got {arm!r}; the plain and "
            f"conductive tails are different arithmetic and neither is a default")
    return KERNEL_KEYS[arm]


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(
            f"the certified text carries {text.count(old)} occurrences of {what}, "
            f"not 1; this family LIFTS it rather than retyping it")
    return text.replace(old, new, 1)


def certified_curl_block(arm, conductive: bool, component_index: int
                         ) -> Tuple[str, str]:
    """The D->E weld's certified curl body, reduced to its header and ONE block.

    Returns ``(header, block)``: the header is everything before the first component
    block (the index decomposition, the guard, the strides, the three register
    declarations, the Bloch factors), and the block is component ``c``'s braced scope
    with its capture line -- both character for character the D->E weld's emission.
    """
    if not 0 <= int(component_index) <= 2:
        raise ValueError(f"component_index must be 0..2, got {component_index!r}")
    body = de.certified_curl_body(arm, bool(conductive))
    parts = body.split(_BLOCK_MARKER)
    if len(parts) != 4:
        raise AssertionError(
            f"the certified curl body carries {len(parts) - 1} component blocks, "
            f"not 3; this family lifts exactly one of them and cannot choose")
    header = parts[0]
    if not header.rstrip().endswith("cf pz; pz.re = pzr; pz.im = pzi;"):
        raise AssertionError(
            "the certified curl body's header no longer ends with the three Bloch "
            "factors; the block split would carry text into the wrong half")
    raw = parts[1 + int(component_index)]
    close = raw.index("\n    }\n")
    block = _BLOCK_MARKER + raw[:close + len("\n    }\n")]
    if not block.startswith(f"{_BLOCK_MARKER}{int(component_index)}: "):
        raise AssertionError(
            f"component block {component_index} opens as {block[:40]!r}; the "
            f"certified body's block order has moved")
    capture = f"        d{int(component_index)} = "
    if block.count(capture) != 1:
        raise AssertionError(
            f"component block {component_index} captures its register "
            f"{block.count(capture)} times, not once")
    for other in range(3):
        if other != int(component_index) and f"d{other} = " in block:
            raise AssertionError(
                f"component block {component_index} writes d{other}; the blocks "
                f"are not independent and one may not be lifted alone")
    return header, block


def _certified_constitutive_value(arm, component_index: int) -> str:
    """The D->E weld's constitutive store for one component, value-named."""
    target, _displacement, stem, count_name = _COMPONENTS[component_index]
    slots = ", ".join(f"{stem}{slot}" for slot in range(MAX_POLES))
    line = (f"    cf_store(h{component_index}, idx, mul_field_left(\n"
            f"        minus_poles_reg(d{component_index}, {slots}, idx, "
            f"{count_name}),   // THE SEAM\n"
            f"        inv_eps_{component_index}[idx]));\n")
    body = de.certified_constitutive_body(arm)
    if body.count(line) != 1:
        raise AssertionError(
            f"the D->E weld's constitutive body carries {target}'s store "
            f"{body.count(line)} times, not once; this family LIFTS that line rather "
            f"than retyping it and cannot splice around its absence")
    value = _replace_once(
        line, f"    cf_store(h{component_index}, idx, mul_field_left(",
        "    cf ev = mul_field_left(", "the certified store's opening")
    value = _replace_once(
        value, "[idx]));", f"[idx]);\n    cf_store(h{component_index}, idx, ev);",
        "the certified store's closing")
    return value


def _certified_poles(arm, component_index: int, count: int,
                     kinds: Sequence[bool]) -> str:
    """The E->P weld's per-pole recurrence lines, under ITS renames and ITS seam."""
    _target, _displacement, stem, _count_name = _COMPONENTS[component_index]
    poles: List[str] = []
    for i in range(int(count)):
        line = ep._certified_recurrence_line(bool(kinds[i]), arm)
        line = _replace_once(line, "cf_store(p_out, idx,", f"cf_store(p_out_{i}, idx,",
                             "the recurrence's store")
        line = _replace_once(line, "cf_load(p_now, idx)", f"cf_load({stem}{i}, idx)",
                             "the recurrence's P^n load (the bank slot)")
        line = _replace_once(line, "cf_load(p_prev, idx)", f"cf_load(p_prev_{i}, idx)",
                             "the recurrence's P^(n-1) load")
        line = _replace_once(
            line, "cf_load(drive, idx)", "ev",
            "the certified drive load (THE SEAM: the register the constitutive "
            "just stored)")
        if kinds[i]:
            line = _replace_once(line, "sigma[idx]", f"sigma_{i}[idx]",
                                 "the volume sigma load")
        else:
            line = _replace_once(line, "        sigma,", f"        sigma_{i},",
                                 "the uniform sigma bind")
        line = _replace_once(line, "c_now, c_prev, c_drive",
                             f"c_now_{i}, c_prev_{i}, c_drive_{i}",
                             "the recurrence coefficients")
        poles.append(
            f"\n    // pole {i}: dispersion.PolarizationState.update, the certified\n"
            f"    // complex recurrence under this launch's per-pole names.\n" + line)
    return "".join(poles)


def _signature(arm_name: str, conductive: bool, component_index: int, count: int,
               kinds: Sequence[bool]) -> str:
    """The fused signature for one curl arm, one component, one specialization.

    THE ONLY HAND-WRITTEN DEVICE TEXT IN THIS MODULE, and it is a signature: no
    arithmetic lives here. D appears EXACTLY ONCE (written by the curl, read from a
    register); E appears exactly once (stored); the bank appears once, NOT
    ``__restrict__`` because the spare slots are bound to ``inv_eps`` and never read
    -- the D->E weld's binding, kept because D is ``__restrict__`` here and may not be
    the spare.
    """
    c = int(component_index)
    _target, _displacement, stem, count_name = _COMPONENTS[c]
    lines: List[str] = [f'extern "C" __global__ void {arm_name}(']
    lines.append("    // THE SHARED VOLUME, BOUND ONCE: this component's D, written by")
    lines.append("    // the curl and read from a register by the constitutive.")
    lines.append(f"    float* __restrict__ f{c},")
    lines.append("    // B, the curl's operands (Fields.get_H returns the B arrays")
    lines.append("    // themselves without an absorber). All three are bound; one")
    lines.append("    // component's block reads two of them.")
    lines.append("    const float* __restrict__ g0, const float* __restrict__ g1,")
    lines.append("    const float* __restrict__ g2,")
    if conductive:
        lines.append("    // NOT __restrict__, lifted from the certified conductive curl:")
        lines.append("    // float32 VOLUMES indexed by the COMPLEX CELL index.")
        lines.append(f"    const float* condfac_{c}, const float* condinv_{c},")
    lines.append("    // E, the constitutive target; there is no f_w without a layer.")
    lines.append(f"    float* __restrict__ h{c},")
    lines.append("    // NOT __restrict__, the certified template's own declaration.")
    lines.append(f"    const float* inv_eps_{c},")
    lines.append(f"    // THE POLE BANK, BOUND ONCE ({MAX_POLES} slots, the certified")
    lines.append("    // width); spare slots are bound to inv_eps and never read. NOT")
    lines.append("    // __restrict__: a spare may legitimately be inv_eps's pointer.")
    lines.append("    " + ", ".join(f"const float* {stem}{slot}"
                                     for slot in range(MAX_POLES)) + ",")
    for i in range(int(count)):
        sigma = (f"const float* __restrict__ sigma_{i}," if kinds[i]
                 else f"float sigma_{i},")
        lines.append(f"    float* __restrict__ p_out_{i}, "
                     f"const float* __restrict__ p_prev_{i}, {sigma}")
        lines.append(f"    float c_now_{i}, float c_prev_{i}, float c_drive_{i},")
    lines.append("    // Extents in COMPLEX CELLS; the guard is the curl's nx*ny*nz.")
    lines.append("    int nx, int ny, int nz, float dtdx,")
    lines.append("    int bc_x, int bc_y, int bc_z,")
    lines.append("    // The BACKWARD Bloch table: bloch_phase_arguments(grid, True).")
    lines.append("    int ph_x, int ph_y, int ph_z,")
    lines.append("    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,")
    lines.append(f"    int {count_name}")
    lines.append(") {")
    return "\n".join(lines) + "\n"


def three_slot_no_pml_complex_source(arm, conductive: bool, component_index: int,
                                     count: int, sigma_kinds: Sequence[bool]) -> str:
    """The whole kernel for one curl arm, one component, one specialization.

    ``arm`` is the EXPANSION arm (``complex_emitter``'s); ``conductive`` selects the
    curl tail; ``count``/``sigma_kinds`` are the driving states of this component and
    their compile-time sigma forms, in registration order.
    """
    if not isinstance(conductive, bool):
        raise TypeError(
            f"conductive must be a bool, got {type(conductive).__name__}; the two "
            f"tails are different arithmetic and a truthy value is not a choice")
    c = int(component_index)
    if not 0 <= c <= 2:
        raise ValueError(f"component_index must be 0..2, got {component_index!r}")
    count = int(count)
    if not 0 <= count <= MAX_POLES:
        raise ValueError(
            f"count must be 0..{MAX_POLES} (the certified bank width), got {count}")
    kinds = tuple(bool(kind) for kind in sigma_kinds)
    if len(kinds) != count:
        raise ValueError(
            f"sigma_kinds carries {len(kinds)} flags for {count} poles; one "
            f"compile-time sigma form per driving state, in registration order")
    header, block = certified_curl_block(arm, conductive, c)
    source = "".join((
        de.no_pml_complex_fused_electric_pair_prelude(arm),
        _signature(kernel_name("conductive" if conductive else "plain"),
                   conductive, c, count, kinds),
        header,
        block,
        "\n    // THE CONSTITUTIVE, value-named: stepping.update_E's no-PML store for\n"
        "    // this component, reading D from the register the curl just stored\n"
        "    // (THE FIRST SEAM) and naming the product so the recurrence reads it\n"
        "    // from a register (THE SECOND SEAM). fields.drive_field(c) IS the\n"
        "    // stored E without an absorber (fields.py:1140-1163).\n",
        _certified_constitutive_value(arm, c),
        _certified_poles(arm, c, count, kinds),
        "}\n",
    ))
    if "__" in (source.replace("__restrict__", "").replace("__global__", "")
                .replace("__device__", "").replace("__forceinline__", "")
                .replace("__fmaf_rn", "").replace("blockIdx", "")
                .replace("blockDim", "").replace("threadIdx", "")):
        raise AssertionError("an unsubstituted placeholder survived the splice")
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """Every source a digest pins: both expansion arms, both curl arms, every
    component, the corpus pole count and its neighbours, both sigma forms."""
    out: Dict[str, str] = {}
    for arm_name in sorted(complex_emitter.EXPANSIONS):
        for curl_arm in sorted(KERNEL_KEYS):
            for component in (0, 1, 2):
                for count in (0, 1, 2):
                    kind_sets = {tuple([True] * count), tuple([False] * count)}
                    for kinds in sorted(kind_sets):
                        key = (f"{KERNEL_KEYS[curl_arm]}|arm{arm_name}|c{component}|"
                               f"n{count}|"
                               f"{''.join('v' if kind else 'u' for kind in kinds)}")
                        out[key] = three_slot_no_pml_complex_source(
                            arm_name, curl_arm == "conductive", component, count,
                            kinds)
    return out


# =============================================================================
# COMPILATION AND LAUNCH
# =============================================================================

#: NVRTC compile options -- CORRECTNESS, not performance, identical to every certified
#: half's. Spelled here so loading by path cannot pick up a different tuple.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane.
_FUSED_THREADS = 256

#: THE GATE'S DOOR into the emitted text: ``None`` in every shipped path; a probe
#: assigns a ``(arm, conductive, component, count, kinds, source) -> source`` callable.
SOURCE_TRANSFORM: Optional[Any] = None


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm, conductive: bool, component_index: int, count: int,
                sigma_kinds: Sequence[bool]):
    """Compile one specialization, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL: it is part of the memo key, and a gate mutates
    this family through :data:`SOURCE_TRANSFORM`. The memo key's FIRST element is the
    kernel symbol followed by the specialization axes, so a launch counter keyed on
    the symbol prefix sees every specialization of one arm.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    kinds = tuple(bool(kind) for kind in sigma_kinds)
    name = kernel_name("conductive" if conductive else "plain")
    code = three_slot_no_pml_complex_source(arm, conductive, component_index, count,
                                            kinds)
    if SOURCE_TRANSFORM is not None:
        code = SOURCE_TRANSFORM(arm, conductive, component_index, count, kinds, code)
    key = compile_cache.kernel_cache_key(
        f"{name}:{complex_emitter.normalized_expansion(arm)}:{component_index}:"
        f"{count}:" + "".join("v" if kind else "u" for kind in kinds),
        True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def component_specs(fields: Any) -> Dict[str, Dict[str, Any]]:
    """Per component: the driving states, pole count and sigma kinds, RIGHT NOW --
    the E->P weld's own resolver, never cached (the rotation moves the buffers)."""
    return ep.component_specs(fields)


def assert_disjoint_bindings(fields: Any, target: str, states: Sequence[Any]) -> int:
    """Every ``__restrict__`` pointer of ONE component launch distinct from every other
    live pointer, checked per launch AFTER the previous component's rotation.

    The restrict group is D, the three B volumes, E, each ``p_prev`` and each
    ``p_out``; the non-restrict group (bank slots, ``inv_eps``, the conductivity
    volumes) must not alias a restrict one either -- a bank slot IS a P volume, and a
    P volume bound as ``p_out`` too would be written through a restrict pointer while
    read through a plain one.
    """
    displacement = "D" + target[1]
    named: List[Tuple[str, Any]] = [
        (displacement, getattr(fields, displacement)),
        ("Bx", fields.Bx), ("By", fields.By), ("Bz", fields.Bz),
        (target, getattr(fields, target))]
    for i, state in enumerate(states):
        named.append((f"p_prev_{i}", state.P_prev[target]))
        named.append((f"p_out_{i}", state._scratch))
        if ade_sigma_is_volume(state, target):
            named.append((f"sigma_{i}", state.sigma[target]))
    plain: List[Tuple[str, Any]] = [(f"bank_{i}", state.P[target])
                                    for i, state in enumerate(states)]
    seen: Dict[int, str] = {}
    for label, array in named:
        address = _base_address(array)
        if address is None:
            raise ValueError(
                f"{label} exposes no readable base address; the restrict promises in "
                f"the fused signature cannot be shown to hold")
        if address in seen:
            raise ValueError(
                f"{label} aliases {seen[address]} in the {target} launch; every live "
                f"pointer in this group is __restrict__, and binding one allocation "
                f"twice is undefined behaviour NVRTC miscompiles without a diagnostic")
        seen[address] = label
    for label, array in plain:
        address = _base_address(array)
        if address is not None and address in seen:
            raise ValueError(
                f"{label} (a P volume) aliases the restrict pointer {seen[address]} "
                f"in the {target} launch")
    return len(seen)


def launch_three_slot_component(fields: Any, expansion, curl_arm: str, target: str,
                                states: Sequence[Any], boundary_codes: Sequence[Any],
                                phase_flags: Sequence[Any],
                                phase_values: Sequence[Any], dtdx: float,
                                kernel: Optional[Any] = None) -> Dict[str, Any]:
    """ONE component's launch: its curl, its constitutive and its recurrences.

    ``kernel`` is the gate's door, keyword-optional: a gate compiles a deliberately
    broken copy of the shipped source and hands it here.
    """
    if curl_arm not in KERNEL_KEYS:
        raise ValueError(f"curl_arm must be one of {sorted(KERNEL_KEYS)}, got {curl_arm!r}")
    spec = {t: i for i, (t, _d, _s, _c) in enumerate(_COMPONENTS)}
    if target not in spec:
        raise ValueError(f"target must be one of {sorted(spec)}, got {target!r}")
    component_index = spec[target]
    count = len(states)
    if count > MAX_POLES:
        raise ValueError(
            f"{count} driving states exceed the certified bank width {MAX_POLES}")
    kinds = tuple(bool(ade_sigma_is_volume(state, target)) for state in states)
    problem = ep._drive_identity_problem(fields, target)
    if problem is not None:
        raise ValueError(problem)
    assert_disjoint_bindings(fields, target, states)
    word_view = certified._word_view
    displacement = "D" + target[1]
    stored = getattr(fields, target)
    nx, ny, nz = (int(n) for n in stored.shape)
    n_elem = nx * ny * nz
    inverse = fields.inverse_epsilon_for(target)
    arguments: List[Any] = [word_view(getattr(fields, displacement)),
                            word_view(fields.Bx), word_view(fields.By),
                            word_view(fields.Bz)]
    if curl_arm == "conductive":
        arguments += [fields.condfac_for(displacement),
                      fields.condinv_for(displacement)]
    arguments += [word_view(stored), inverse]
    for i in range(MAX_POLES):
        # A used slot holds a complex64 P volume, word-viewed; a SPARE slot holds
        # inv_eps -- float32 already, never re-viewed, never dereferenced.
        arguments.append(word_view(states[i].P[target]) if i < count else inverse)
    for i, state in enumerate(states):
        scratch = state._scratch
        if int(scratch.size) != n_elem:
            raise ValueError(
                f"state {i}'s scratch holds {int(scratch.size)} cells and the launch "
                f"walks {n_elem}; the certified update_P guard this weld drops was "
                f"that size")
        coefficients = tuple(state._coefficients)
        if len(coefficients) != 3:
            raise ValueError(
                f"a polarization carries {len(coefficients)} recurrence "
                f"coefficients, not the (c_now, c_prev, c_drive) triple")
        arguments.append(word_view(scratch))
        arguments.append(word_view(state.P_prev[target]))
        sigma = state.sigma[target]
        arguments.append(sigma if kinds[i] else np.float32(sigma))
        arguments.extend(np.float32(value) for value in coefficients)
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx)]
    arguments += [np.int32(code) for code in boundary_codes]
    arguments += [np.int32(flag) for flag in phase_flags]
    arguments += [np.float32(value) for value in phase_values]
    arguments.append(np.int32(count))
    blocks = (n_elem + _FUSED_THREADS - 1) // _FUSED_THREADS
    (kernel or _get_kernel(expansion, curl_arm == "conductive", component_index,
                           count, kinds))(
        (blocks,), (_FUSED_THREADS,), tuple(arguments))
    return {"component": target, "poles": count, "kinds": list(kinds),
            "blocks": blocks, "threads": _FUSED_THREADS, "elements": n_elem,
            "kernel": kernel_name(curl_arm)}


def run_three_slot_no_pml_complex(fields: Any, grid: Any, expansion, *,
                                  curl_arm: Optional[str] = None,
                                  kernel: Optional[Any] = None,
                                  boundary_codes: Optional[Sequence[Any]] = None,
                                  phase_flags: Optional[Sequence[Any]] = None,
                                  phase_values: Optional[Sequence[Any]] = None
                                  ) -> Dict[str, Any]:
    """All three of :data:`REPLACES`: three launches, host rotation between.

    THE ROTATION IS TRANSCRIBED from dispersion.py:689-691 and run once per driving
    state after each component launch RETURNS, exactly as the certified launchers
    and the E->P weld run it. Every slot is re-read after the previous component's
    rotation through :func:`component_specs`.

    ``curl_arm`` defaults to the certified classifier's answer; the overrides are the
    gate's doors (a wrong tail, an unconjugated phase table).
    """
    from .complex_pml_kernels import (bloch_phase_arguments,  # noqa: PLC0415
                                      complex_boundary_codes)

    if curl_arm is None:
        curl_arm = de.no_pml_complex_curl_arm(fields, "step_D")
    if boundary_codes is None:
        boundary_codes = complex_boundary_codes(grid)
    if phase_flags is None or phase_values is None:
        # THE BACKWARD TABLE: complex_emitter.KERNELS['step_D'][1] is True.
        derived_flags, derived_values = bloch_phase_arguments(grid, True)
        phase_flags = derived_flags if phase_flags is None else phase_flags
        phase_values = derived_values if phase_values is None else phase_values
    dtdx = float(grid.dt / grid.dx)
    launches = 0
    recurrences = 0
    per_component: List[Dict[str, Any]] = []
    for target, _displacement, _stem, _count in _COMPONENTS:
        states = component_specs(fields)[target]["states"]
        per_component.append(launch_three_slot_component(
            fields, expansion, curl_arm, target, states, boundary_codes,
            phase_flags, phase_values, dtdx, kernel))
        launches += 1
        recurrences += len(states)
        for state in states:
            # dispersion.py:689-691, verbatim, once the launch API returned.
            p = state.P[target]
            p_prev = state.P_prev[target]
            scratch = state._scratch
            state.P[target] = scratch
            state.P_prev[target] = p
            state._scratch = p_prev
    # THE GROUP'S LAUNCH GRID, AND WHY A COMPONENT-MAJOR AGGREGATE HAS TO SPELL ONE.
    # ``fastpath._launch_grid_of`` reads ``blocks`` off this report through
    # ``_TripleHalfPlan.launch_grid``, and ``FastPathPlan.launch_counters`` turns it
    # into ``programs_per_dispatch`` -- the counter that separates "a kernel
    # launched" from "a kernel computed over data", which a warm launch at
    # ``grid=(0,)`` passes in every other counter.
    #
    # THIS PRODUCT IS THE FIRST TO PUT A MULTI-LAUNCH GROUP ON A *LEADING* SLOT. The
    # real weld's component-major aggregate
    # (``three_slot_dispersive_weld.run_three_slot_polarization``) spells no
    # ``blocks`` either, but it sits on ``update_P``, and the route gate probes
    # programs on ``step_B``/``step_D`` only -- so its ``None`` was never read and
    # the gap stayed invisible. MEASURED on the GPU host 2026-09-14: complex_no_pml_3d
    # reported ``step_D programs_per_dispatch: null`` against 600 dispatches and
    # ``plan_launches`` 600, and the route refused the case as VACUOUS-PASS.
    #
    # THE TOTAL, NOT ONE COMPONENT'S, because the counter is programs per DISPATCH
    # and one dispatch of this slot launches every component. It is the same
    # summation ``launch_counters`` already performs over a list-valued slot, where
    # ``update_P``'s per-susceptibility plans are reported as the list's total.
    blocks = sum(int(report["blocks"]) for report in per_component)
    return {"launched": True, "launches": launches, "recurrences": recurrences,
            "replaces": REPLACES, "kernel": kernel_name(curl_arm),
            "curl_arm": curl_arm, "blocks": blocks, "components": per_component}


# =============================================================================
# COVERAGE
# =============================================================================

def covers_three_slot_complex_no_pml_dispersive_weld(
        fields: Any, pml: Any, grid: Any, sources: Any = None, license: Any = None,
        subnormal_policy: Any = None) -> Tuple[bool, str]:
    """May three per-component launches span ``step_D`` -> ``update_E`` -> ``update_P``?

    A CONJUNCTION OF TWO SHIPPED PREDICATES, AND NOTHING IS WEAKENED. The D->E half is
    ``no_pml_complex_fused_electric_pair.covers_no_pml_complex_fused_electric_pair``
    whole -- the SOURCE question is asked there, through
    ``deposit_repair.seam_source_reasons`` with that module's ``carries_repair=False``,
    which is what makes this product's single launch sound: an electric in-seam
    source is REFUSED, never carried. The E->P half is
    ``complex_no_pml_fused_polarization_pair.covers_complex_no_pml_fused_polarization_
    pair`` whole, which is where the register hand-off's drive identity and the
    dropped guard's extent identity are asked. Each refusal is prefixed with the side
    that said it.

    THE ADMISSION SET IS A SUBSET OF EACH HALF'S, which is what
    ``fused_pairs._superseded_by_a_longer_span`` needs. Disjoint from both real
    three-slot welds on storage alone (their halves refuse complex64 by name).
    """
    covered, reason = de.covers_no_pml_complex_fused_electric_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return False, f"step_D->update_E half: {reason}"
    covered, reason = ep.covers_complex_no_pml_fused_polarization_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return False, f"update_E->update_P half: {reason}"
    return True, "covered"
