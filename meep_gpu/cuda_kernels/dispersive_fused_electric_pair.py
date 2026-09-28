"""The DISPERSIVE hand-CUDA fused pair: ``step_D`` -> the two fills -> pole-aware ``update_E``.

THE LARGEST SINGLE PRODUCT LEFT ON ANY BACKEND when it was built. The board's
``D_to_E (cuda_curl/PML, cuda_dispersive/dispersive)`` cell carries SEVEN corpus rows
and no product occupied it: three ``stochastic_emitter*`` examples and the four
``TestLoadDump.*_2d`` tests
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-02_tierfix``,
``taxonomy.instances.buildable_not_built``). Both halves were already CERTIFIED --
``fused_step_D_pml_real`` and ``fused_update_E_pml_real_dispersive`` -- and the cell's
verdict was POINTWISE-BUILDABLE, so what was missing was the weld and nothing else.

This module is the twin of :mod:`.fused_electric_pair` with ONE half exchanged. Every
geometric emitter it needs -- the ownership inversion, the pre-clear copy, the wall
carry, the ghost destination closure -- is IMPORTED FROM THAT MODULE rather than
copied, so the two families cannot drift on the carry, and the only text this file
writes is the pole chain and the two anchored edits that splice it in.

=============================================================================
BUILT AND RELEASED, AND NOT INSTALLED ON THIS CORPUS -- ADDED 2026-09-02
=============================================================================

Added by the CONDUCTIVE round, which hit the same collision on its own cells and
measured this one while it was there. NOTHING BELOW IS RETRACTED: the arithmetic is
gated and RELEASED under both float32 subnormal policies
(``certification.json:cuda_dispersive_fused_electric_pair_2026-09-02``) and the
predicate still admits all seven rows.

WHAT CHANGED IS WHO GETS THE SLOT. This product owns ``step_D`` AND ``update_E``;
``cuda_fused_polarization_pair`` owns ``update_E`` AND ``update_P``. All SEVEN rows of
this family's cell are EXACTLY the seven that pair serves at E->P -- measured on the
stamped census (every row's ``cuda_ade`` covering ``update_P``) and on the board cut
BEFORE this product existed (``results/fusion_matrix_cuda_2026-09-02_tierfix``, which
records that pair at 7). So installing here is +7 at D_to_E and -7 at E_to_P: NET
ZERO, paid for by displacing a released product -- and until it was noticed it also
made ``test_fused_polarization_pair.py``'s PML seam test read red.

``fused_pairs._later_seam_claimant`` now leaves the slot with the product that already
served it. Lifting that needs a THREE-SLOT weld (``step_D`` -> ``update_E`` ->
``update_P``, serving both seams instead of trading one for the other) or a ruling on
the seam order -- either a release decision, neither a change to this file.

=============================================================================
WHAT THIS DOES THAT NEITHER SIBLING BACKEND DOES, AND WHY IT HAD TO
=============================================================================

Metal and Triton both ship a dispersive electric pair. BOTH REFUSE A FOLDED GRID:

* ``metal_kernels/fused_dispersive_pair.py`` refuses the two mirror fills through the
  E half's own "a mirror plane is active" clause and ships the fold as a SEPARATE
  product (``metal_kernels/folded_fused_dispersive_pair.py``);
* ``triton_kernels/dispersive_fused_pair.py`` refuses it the same way, through
  ``dispersive_update_e.dispersive_constitutive_coverage``.

CUDA's ``covers_real_pml_dispersive_constitutive`` ADMITS a fold up to
``ADE_FOLD_PLANES_SWEPT``, and FOUR OF THE SEVEN ROWS ARE FOLDED -- the four
``TestLoadDump.*_2d`` rows are ``metallic`` on x, ``mirror`` on y and ``periodic`` on
z (the stamped census's ``configuration.cuda_boundary_kinds``). A straight port of
either sibling would serve three of seven. So this product CARRIES both fills, the way
:mod:`.fused_electric_pair` does, and inherits that module's whole carry.

THE ONE ARITHMETIC FACT THAT CARRY ADDS, AND IT IS THE POINT OF THIS FILE. The array
path's ``update_E`` runs over the WHOLE volume and evaluates
``Fields.displacement_minus_polarization`` cell by cell (fields.py:1096-1105), so at an
imaged ghost it subtracts THAT CELL'S ``P``. The source thread that writes the ghost
holds its OWN pole registers, and re-using them is the obvious port and the cheap one.
It is also wrong on every imaged plane. So each ghost's chain is re-formed FROM THE
DESTINATION'S POLE WORDS, at the destination index, exactly as its inverse epsilon and
its coefficient pair already were.

MEASURED BEFORE A DEVICE WAS BOOKED.
``parity/meep_gpu/probe_cuda_dispersive_electric_pair.py`` runs this closed form
against the driver's own five passes over complete steps on eight configurations --
including both corpus shapes -- and scores six armed mutations. ``poles_at_source``,
the port above, diverges on 6 of the 8; ``inv_eps_at_source`` on 6; the pole ORDER on
8; the pre-clear ordering on 3; the near-fill ordering on 1 (the cross term).

=============================================================================
WHERE THE ORDERED ``D - sum P`` SUBTRACTION SITS, MEASURED ON BOTH SIBLINGS
=============================================================================

Both siblings put the WHOLE chain strictly AFTER the metallic cleanup, consuming the
post-clear register -- ``metal_kernels/fused_dispersive_pair.py``'s
``fused_dispersive_pair_source`` emits ``zero_metal_mask(axis, ...)`` and then
``float source = v{axis};``, and ``triton_kernels/dispersive_fused_pair.py``'s
``fused_curl_dispersive_E`` applies its ``ZM_X/ZM_Y/ZM_Z`` block to ``v0/v1/v2`` and
then opens the E half with ``s0 = v0``. They AGREE, and the probe's
``subtract_before_clear`` mutation shows the agreement is load-bearing rather than
incidental: it diverges on 3 of 8 configurations.

WHAT CUDA DOES DIFFERENTLY IS NOT THAT ORDER BUT ITS GRANULARITY. Both siblings
interleave per component -- Metal because it dispatches one component per launch
(a 31-buffer binding ceiling this platform does not have), Triton because it writes
three sequential E blocks. The CUDA constitutive half is LIFTED, not transcribed, and
the certified ``update_E_pml_real_dispersive`` forms ALL THREE ``D - sum P`` chains and
all three inverse-epsilon products BEFORE any ``constitutive_apply`` store, with its
own note saying why that reordering is exact (the sub-step reads D, the poles and
inv_eps and writes E and f_w_E -- disjoint sets). Adopting either sibling's interleave
would silently retype that body; this file keeps it and splices into it.

=============================================================================
THE ARITY IS A COMPILE-TIME AXIS, WHICH THE REAL TWIN'S IS NOT
=============================================================================

:mod:`.fused_electric_pair` emits ONE device string. This family emits one per
``(n_Ex, n_Ey, n_Ez)`` triple, because ``dispersive_kernels`` does: the chain length is
baked, a component with no contributor forms NO subtraction at all (fields.py:1096-1098
returns the D array itself), and at arity ``(0, 0, 0)`` the emitted constitutive lines
are character-identical to the non-dispersive family's. The ENTRY POINT NAME is fixed
and only the parameter list and the chain vary, which is ``ARM_KERNEL_NAMES``' own
convention; ``compile_cache.kernel_cache_key`` keys the memo on the source.

:data:`POLE_COUNTS_SWEPT` is ``dispersive_kernels``' own list and already names both
corpus shapes: ``(5, 5, 5)`` is the four ``TestLoadDump.*_2d`` rows and ``(6, 6, 6)``
the three ``stochastic_emitter`` ones.

=============================================================================
THE ALIASING HAZARD, WITH ONE MORE GROUP THAN THE REAL TWIN
=============================================================================

D appears EXACTLY ONCE in the signature, for :mod:`.fused_electric_pair`'s reason: the
curl writes it in place and the constitutive half reads it from a register, so binding
it a second time as ``const __restrict__`` would be two restrict pointers to one
allocation, which NVRTC miscompiles without a diagnostic.

THE POLE POINTERS ARE ``__restrict__`` and the three inverse-epsilon ones are not, both
lifted from ``dispersive_kernels.dispersive_source`` rather than decided here: two
susceptibilities never share a ``P`` buffer (``PolarizationState.__init__`` allocates
its own, dispersion.py:645-647) while an isotropic run hands ONE inverse-epsilon
pointer three times (fields.py:1321-1326). :func:`assert_disjoint_bindings` therefore
checks the pole group against every other restrict argument, and checks separately that
no inv_eps volume collides with one.

THE POLE POINTERS ROTATE. ``PolarizationState.update`` moves ``P``/``P_prev``/scratch
every step (dispersion.py:689-691), so the launcher resolves ``state.P[c]`` immediately
before each launch through ``dispersive_kernels.resolve_pole_plan`` and never caches a
view. A plan built before ``update_P`` names last step's arrays after it -- stale in a
way that still computes, which is the worst kind.

=============================================================================
DISJOINT FROM THE REAL TWIN BY A SINGLE BOOLEAN
=============================================================================

``install_fused_pairs`` leaves a seam UNFUSED when two products admit it, so a new
product that overlapped :mod:`.fused_electric_pair` would COST the 79 rows that one
serves rather than adding seven. The two partition exactly:
``coverage.covers_real_pml_constitutive(side='E')`` refuses ``fields.polarizations``
truthy and ``dispersive_kernels.covers_real_pml_dispersive_constitutive`` REQUIRES it,
which is that predicate's own stated disjointness clause. Nothing else is needed and
nothing here restates it -- both predicates are conjoined whole.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

# THE REAL TWIN. Every geometric emitter and every launch helper this family needs is
# taken from it rather than copied: the carry is one object, and two copies of the
# ownership inversion is two places for the fold to be wrong.
from . import fused_electric_pair as real

# THE PREDICATES. ``in_seam_coverage`` is stdlib-only by construction, so the predicate
# below answers on a laptop -- which is what lets the backend-free census ask it on
# every corpus row.
try:
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

    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")

# THE CERTIFIED DEVICE TEXT AND THE DISPERSIVE PREDICATE. ``dispersive_kernels``
# imports CuPy at module scope, so it is taken DEFENSIVELY -- but its PREDICATE is the
# part whose failure mode is a silent wrong answer, so it is re-imported lazily inside
# :func:`covers_dispersive_fused_electric_pair` where a device-free host still reaches
# it through the by-path loader.
try:
    from . import dispersive_kernels
except ImportError:  # a host with no CuPy
    dispersive_kernels = None
try:
    from . import constitutive_kernels
except ImportError:  # a host with no CuPy
    constitutive_kernels = None

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
# Plain assignments with NO type annotation, for the reason every sibling in this
# directory spells them that way: the partition readers walk the syntax tree WITHOUT
# importing the module (these modules import cupy) and an annotated assignment is an
# ``ast.AnnAssign`` those readers do not match.

#: RELEASED 2026-09-02 under BOTH float32 subnormal policies. What moved this kernel
#: across the partition is the run, not an argument: ``gate_cuda_dispersive_fused_
#: electric_pair.py`` on one RTX A6000, 18 of 18 cases bit-identical over 60 COMPLETE
#: driver steps at six pole arities including both corpus ones, every stored field
#: volume AND every pole buffer compared as uint32 words, 119 of 143 mutation legs
#: caught with three confirmed nulls and none unarmed, the deposit legs identical with
#: the unbracketed control diverging, and the block written into
#: ``certification.json`` as ``cuda_dispersive_fused_electric_pair_2026-09-02``.
CERTIFIED_KERNELS = (
    "dispersive_fused_electric_pair_pml_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its fused launch with the deposit repair? TRUE, and the
#: wiring is ``fused_pairs._install_fused_pair``'s -- reached because this family has a
#: row in ``fused_pairs.FUSED_PRODUCTS`` and because ``fused_pairs.FUSED_PAIR_SEAMS``
#: carries the ``step_D`` row. The flag and the wiring change together or not at all
#: (``deposit_repair.py:221-226``).
#:
#: IT IS THE WHOLE PRODUCT HERE. Every one of the seven rows this cell owns carries an
#: electric deposit inside the seam -- the stamped census reports
#: ``source_field_types`` of ``["D"]`` on six of them and ten D sources on
#: ``stochastic_emitter.py`` -- so without the bracket this module would serve ZERO.
#: That is not the real twin's situation, where four of 79 clear the seam unaided, and
#: it is why ``CARRIES_DEPOSIT_REPAIR`` is asserted as a per-cell MEASUREMENT off the
#: census in the host suite rather than read as a constant.
#:
#: WHAT THE REPAIR HAS TO RECONSTRUCT ON A DISPERSIVE HALF. ``deposit_repair.apply``
#: recomputes ``(f + kps*fw_fresh) - kms*fw_prev`` with ``fw_fresh`` from
#: ``Fields.displacement_minus_polarization`` (deposit_repair.py:150-152), which IS a
#: dispersive ``update_E``'s source -- and it is entitled to read the pole arrays it
#: reads, because ``update_P`` runs AFTER ``update_E`` (driver.py:3315), so the
#: polarizations at the repair are the ones the launch consumed, unchanged. The two
#: D-only clauses in ``deposit_repair.repairable`` (an off-diagonal chi1inv, an
#: instantaneous chi2/chi3) are refused by the dispersive constitutive half's own
#: predicate before this predicate reaches the seam clause.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_dispersive_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once. FIXED ACROSS ARITIES -- the pole
#: count varies the SOURCE, not the name, which is ``ARM_KERNEL_NAMES``' convention and
#: what lets one certification block cover the swept corpus.
KERNEL_NAME = "dispersive_fused_electric_pair_pml_real"

#: The driver passes ONE launch performs, in driver order (driver.py:3302, :3309,
#: :3310, :3311, :3313). Declared rather than inferred from the two slots --
#: ``zero_metal_D`` and the two fills are driver passes no slot names.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: The arities this family emits for a digest, and the ones its gate sweeps. Taken
#: from ``dispersive_kernels`` rather than restated: a triple this family emitted and
#: that family never swept would be a body no device has run.
def _swept() -> Tuple[Tuple[int, int, int], ...]:
    if dispersive_kernels is None:
        return ()
    return tuple(dispersive_kernels.POLE_COUNTS_SWEPT)


#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring.
#: The first six rows are :mod:`.fused_electric_pair`'s, inherited whole because the
#: curl half, the ownership inversion, the pre-clear copy and the wall carry are that
#: module's own emitters called from here; only the last two are this file's.
LIFT_EDITS: Tuple[Dict[str, str], ...] = real.LIFT_EDITS + (
    {"line": "    float s_x = Dx[idx];   "
             "(dispersive_kernels._source_lines, count > 0)",
     "became": "    float s_x = d_x;",
     "why": "THE SEAM, at every arity above zero. The certified dispersive body opens "
            "each driven component by reloading the flux density the curl just "
            "stored; this reads the register instead. A float32 stored to global and "
            "reloaded is the identity on the bits. It is also what removes the "
            "constitutive half's only use of Dx/Dy/Dz, which is what lets D be bound "
            "exactly once. Every subtraction line after it -- the ORDER of which is "
            "the arithmetic -- is untouched."},
    {"line": "    float src_x = Dx[idx] * inv_eps_Ex[idx];   "
             "(dispersive_kernels._source_lines, count == 0)",
     "became": "    float src_x = d_x * inv_eps_Ex[idx];",
     "why": "THE SEAM at the DEGENERATE arity, where the emitter forms no chain at "
            "all because fields.py:1096-1098 returns the D array itself. The same one "
            "token moves; the multiply, its operand order (D on the left, "
            "stepping.py:1011) and the trailing annotation are the certified line's."},
    {"line": "        float gx_y_s = gx_y_v * inv_eps_Ex[gx_y_i];   "
             "(fused_electric_pair.fill_carry_blocks)",
     "became": "        float gx_y_p = gx_y_v;\n"
               "        gx_y_p = gx_y_p - P_Ex_0[gx_y_i];  ...\n"
               "        float gx_y_s = gx_y_p * inv_eps_Ex[gx_y_i];",
     "why": "THE GHOST CHAIN, and it is the one thing neither sibling backend has to "
            "do. update_E is element-wise over the WHOLE volume, so an imaged ghost "
            "subtracts THAT CELL'S poles; the source thread's own registers are a "
            "different field on every imaged plane. Same chain, same left-to-right "
            "order, same operand order into inv_eps -- the only change is the INDEX, "
            "which is the ghost's, exactly as inv_eps and the kps/kms pair already "
            "were. At arity 0 the line is left character-identical."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_dispersive_constitutive_body",
    "covers_dispersive_fused_electric_pair", "device_sources",
    "dispersive_fill_carry_blocks", "dispersive_fused_electric_pair_prelude",
    "dispersive_fused_electric_pair_signature",
    "dispersive_fused_electric_pair_source", "launch_dispersive_fused_electric_pair",
    "pole_bindings", "run_dispersive_fused_electric_pair", "swept_arities",
]


def swept_arities() -> Tuple[Tuple[int, int, int], ...]:
    """``dispersive_kernels.POLE_COUNTS_SWEPT``, or ``()`` on a host without it."""
    return _swept()


# =============================================================================
# THE LIFT
# =============================================================================

#: The entry point the real twin's signature declares, and this family's own. Renamed
#: rather than re-typed so the parameter list stays ONE string in the tree.
_REAL_ENTRY = "extern \"C\" __global__ void fused_electric_pair_pml_real("
_ENTRY = f'extern "C" __global__ void {KERNEL_NAME}('

#: Where the pole pointer block goes: immediately after the three inverse-epsilon
#: parameters, which is where ``dispersive_kernels.dispersive_source`` puts it in the
#: stand-alone arm's signature.
_INV_EPS_LINE = ("    const float* inv_eps_Ex, const float* inv_eps_Ey, "
                 "const float* inv_eps_Ez,\n")


def _certified_halves() -> None:
    """Refuse the emitter BY NAME on a host where the certified text is unreachable."""
    missing = [name for name, module in (
        ("dispersive_kernels", dispersive_kernels),
        ("constitutive_kernels", constitutive_kernels)) if module is None]
    if missing:
        raise RuntimeError(
            f"the certified halves {missing} are not importable on this host (they "
            f"import CuPy at module scope), so there is no certified text to splice. "
            f"covers_dispersive_fused_electric_pair needs neither and still answers")
    real._certified_halves()


def dispersive_fused_electric_pair_prelude() -> str:
    """The real twin's curl prelude, with the DISPERSIVE family's constitutive one.

    :func:`.fused_electric_pair.fused_electric_pair_prelude` returns the certified
    curl prelude with ``pml_apply`` rewritten into a value, followed by
    ``constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE``. This family splices the
    DISPERSIVE constitutive body, so it takes the DISPERSIVE family's prelude in that
    slot -- ``dispersive_kernels`` keeps its OWN copy of ``constitutive_apply``
    deliberately, so that a bit-identity probe rewriting one family's string does not
    silently mutate the other's, and a product that borrowed the wrong copy would be
    outside every mutation its own gate arms.

    The exchange is an ANCHORED edit on the real twin's own output rather than a second
    assembly of the curl half, so the ``pml_apply_reg`` rewrite has one home.
    """
    _certified_halves()
    prelude = real.fused_electric_pair_prelude()
    ordinary = constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE
    if not prelude.endswith(ordinary):
        raise AssertionError(
            "fused_electric_pair_prelude no longer ends with the ORDINARY "
            "constitutive prelude, so the dispersive one cannot be swapped in and "
            "this family would emit constitutive_apply from a family whose gate does "
            "not mutate it")
    dispersive = (dispersive_kernels._POLE_CHAIN_NOTE
                  + dispersive_kernels._PML_ACCUMULATION_NOTE)
    if dispersive.count("void constitutive_apply(") != 1:
        raise AssertionError(
            f"the dispersive prelude declares constitutive_apply "
            f"{dispersive.count('void constitutive_apply(')} times, not once; the "
            f"spliced body calls it and a missing or doubled definition is a compile "
            f"failure at first launch rather than at import")
    return prelude[: -len(ordinary)] + dispersive


def dispersive_fused_electric_pair_signature(counts: Sequence[int]) -> str:
    """The real twin's signature, renamed, with this arity's pole pointers inserted.

    TWO ANCHORED EDITS ON ONE LIFTED STRING. The parameter list -- the shared flux
    density bound once, the non-restrict inverse-epsilon group, the curl's integer
    sub-lattice against the constitutive's renamed half-integer one, the wall flags and
    the two fills' runtime plan -- is :data:`.fused_electric_pair._SIGNATURE` itself,
    so a change there reaches this family instead of being transcribed twice.
    """
    _certified_halves()
    counts = dispersive_kernels.normalized_pole_counts(counts)
    signature = real._SIGNATURE
    if signature.count(_REAL_ENTRY) != 1:
        raise AssertionError(
            f"the real twin's signature declares its entry point "
            f"{signature.count(_REAL_ENTRY)} times, not once; the rename has no anchor")
    if signature.count(_INV_EPS_LINE) != 1:
        raise AssertionError(
            "the real twin's signature no longer carries the three inverse-epsilon "
            "parameters on one line; the pole block has nowhere to be inserted and a "
            "kernel emitted without it would not compile")
    signature = signature.replace(_REAL_ENTRY, _ENTRY, 1)
    poles = dispersive_kernels._pole_parameter_block(counts)
    return signature.replace(
        _INV_EPS_LINE,
        _INV_EPS_LINE
        + ("    // THE POLE POINTERS, __restrict__ and lifted from\n"
           "    // dispersive_kernels.dispersive_source: two susceptibilities never\n"
           "    // share a P buffer (dispersion.py:645-647), and they ROTATE every\n"
           "    // step, so the launcher resolves them immediately before each launch.\n"
           if poles else "")
        + poles, 1)


def dispersive_fill_carry_blocks(target: int, count: int,
                                 indent: str = "        ") -> List[str]:
    """The real twin's ghost carry, with this component's pole chain re-formed AT THE
    DESTINATION.

    ONE ANCHORED EDIT PER GHOST BLOCK. :func:`.fused_electric_pair.fill_carry_blocks`
    emits, for every destination this thread owns, the imaged displacement, the store
    to D, the inverse-epsilon product and the ``constitutive_apply`` at the ghost's own
    coefficient row. All of that is lifted; what is inserted is the chain between the
    displacement and the product.

    THE INDEX IS THE DESTINATION'S AND THAT IS THE WHOLE FINDING. ``update_E`` is
    element-wise over the whole volume, so the ghost cell subtracts ITS OWN ``P``
    words; the source thread's registers hold a different cell's. ``P`` is READ-ONLY in
    this launch -- ``update_P`` is a later sub-step (driver.py:3315) -- so a load off
    another thread's cell is defined, exactly as the ghost's ``inv_eps`` load already
    is, and needs no barrier.

    AT ARITY ZERO NOTHING IS INSERTED and the block is character-identical to the real
    twin's, which is what makes the degenerate configuration reduce to that family's
    certified text by construction rather than by rounding.
    """
    letter = real._CARRIED[target]
    component = "E" + letter
    lines = real.fill_carry_blocks(target, indent)
    if count == 0:
        return lines
    inner = indent + "    "
    out: List[str] = []
    edited = 0
    for line in lines:
        # THE ANCHOR IS THE WHOLE STATEMENT, not a substring: a partial match would
        # attach the chain to a line this module has not read. The tag is recovered
        # from the line and the reconstruction is compared back against it.
        prefix = f"{inner}float g{letter}"
        if (line.startswith(prefix) and "_s = g" in line
                and f"inv_eps_{component}[" in line):
            # `        float gx_y_s = gx_y_v * inv_eps_Ex[gx_y_i];`
            tag = line[len(inner) + len("float "):].split("_s = ", 1)[0]
            expected = (f"{inner}float {tag}_s = {tag}_v * "
                        f"inv_eps_{component}[{tag}_i];")
            if line != expected:
                raise AssertionError(
                    f"the real twin's ghost product is {line!r}, not the "
                    f"{expected!r} this family splices into; the pole chain would be "
                    f"inserted around a statement this module has not read")
            out.append(f"{inner}// THE GHOST CHAIN. update_E is element-wise over the "
                       f"whole volume, so")
            out.append(f"{inner}// this imaged cell subtracts ITS OWN poles -- the "
                       f"DESTINATION index, not")
            out.append(f"{inner}// this thread's. P is read-only in this launch "
                       f"(update_P is a later")
            out.append(f"{inner}// sub-step, driver.py:3315), so the load is defined "
                       f"with no barrier.")
            out.append(f"{inner}float {tag}_p = {tag}_v;")
            for index in range(int(count)):
                out.append(f"{inner}{tag}_p = {tag}_p - "
                           f"P_{component}_{index}[{tag}_i];")
            out.append(f"{inner}float {tag}_s = {tag}_p * "
                       f"inv_eps_{component}[{tag}_i];")
            edited += 1
            continue
        out.append(line)
    if not edited:
        raise AssertionError(
            f"the real twin emitted no inverse-epsilon product for {component} at any "
            f"ghost destination, so this family had nowhere to splice the pole chain; "
            f"every D component is imaged by at least one fill (IYEE_SHIFTS gives it "
            f"one shift-1 axis and two shift-0 axes)")
    return out


def certified_dispersive_constitutive_body(counts: Sequence[int]) -> str:
    """``update_E_pml_real_dispersive``'s body at one arity, lifted, reading the registers.

    Drops the second index decomposition (the curl body above already declares ``idx``,
    ``i``, ``j`` and ``k`` and its bounds guard has already returned), turns each
    component's flux-density read into the seam, resolves the sub-lattice name
    collision, and wraps each store in the ownership guard with its ghost blocks.

    THE CHAIN ITSELF IS UNTOUCHED. Every ``s_a = s_a - P_Ea_n[idx];`` line is the
    certified emitter's own text in the certified order, which is the arithmetic here:
    float32 addition is not associative, so ``((D - P0) - P1)`` and ``D - (P0 + P1)``
    are different numbers at two poles and above.
    """
    _certified_halves()
    counts = dispersive_kernels.normalized_pole_counts(counts)
    prelude = (dispersive_kernels._POLE_CHAIN_NOTE
               + dispersive_kernels._PML_ACCUMULATION_NOTE)
    body = real._split_body(dispersive_kernels.dispersive_source("pml", counts),
                            prelude, "dispersive_source('pml', ...)")
    if real._DECODE_END not in body:
        raise AssertionError(
            "the certified dispersive body no longer decodes i on its own line; the "
            "duplicate decomposition cannot be identified")
    tail = body.split(real._DECODE_END, 1)[1]

    # THE SEAM, per component and per arity. ONE TOKEN PER LINE is substituted and the
    # rest of the certified statement -- the operand order, the inv_eps index, the
    # trailing annotation and every subtraction line after it -- is left as it stands.
    for target, (letter, name) in enumerate(zip(real._CARRIED, real._TARGETS)):
        if counts[target] > 0:
            anchor = f"    float s_{letter} = {name}[idx];"
        else:
            anchor = (f"    float src_{letter} = {name}[idx] * "
                      f"inv_eps_E{letter}[idx];")
        line = real._line_starting(tail, anchor,
                                   f"{name}'s dispersive source at arity "
                                   f"{counts[target]}")
        if line.count(f"{name}[idx]") != 1:
            raise AssertionError(
                f"{name}[idx] appears {line.count(f'{name}[idx]')} times in {line!r}; "
                f"the seam substitution has no unambiguous token")
        tail = tail.replace(line, line.replace(f"{name}[idx]", f"d_{letter}", 1), 1)
    first_letter = real._CARRIED[0]
    opening = (f"    float s_{first_letter} = d_{first_letter};" if counts[0] > 0
               else f"    float src_{first_letter} = d_{first_letter} * "
                    f"inv_eps_E{first_letter}[idx];")
    first = real._line_starting(tail, opening, "the rewritten first dispersive source")
    tail = tail.replace(
        first,
        "    // THE SEAM. The certified body reads D[idx] here -- the very word the\n"
        "    // curl half stored a few lines above -- and this reads the register\n"
        "    // instead. A float32 stored to global and reloaded is the identity on\n"
        "    // the bits, so the substitution is exact rather than close; it is also\n"
        "    // what removes this half's only use of Dx/Dy/Dz, which is what lets D\n"
        "    // be bound exactly once. The subtraction chain below is the certified\n"
        "    // emitter's own text in the certified ORDER, which is the arithmetic:\n"
        "    // float32 addition is not associative.\n" + first, 1)
    for name in real._TARGETS:
        if f"{name}[idx]" in tail:
            raise AssertionError(
                f"a read of {name} survived the seam rewrite; the constitutive half "
                f"would need D bound a second time, which is the aliasing hazard this "
                f"signature exists to avoid")

    # THE ONE RENAME, THE OWNERSHIP GUARD AND THE GHOST CARRY, on one line each.
    for target, (letter, coordinate) in enumerate(zip(real._CARRIED,
                                                      real._COORDINATE)):
        prefix = f"    constitutive_apply(E{letter}, f_w_E{letter}, idx, src_{letter}, "
        call = real._line_starting(tail, prefix, f"E{letter}'s constitutive_apply")
        expected = (f"{prefix}kps_{letter}[{coordinate}], "
                    f"kms_{letter}[{coordinate}]);")
        if call != expected:
            raise AssertionError(
                f"E{letter}'s constitutive_apply is {call!r}, not the certified "
                f"{expected!r}; the sub-lattice rename would be applied to an argument "
                f"list this module has not read")
        renamed = call.replace(f"kms_{letter}[", f"kms_half_{letter}[", 1)
        tail = tail.replace(call, "\n".join(
            [f"    if (own_{letter}) {{", f"        {renamed.strip()}"]
            + dispersive_fill_carry_blocks(target, counts[target])
            + ["    }"]), 1)
    for letter, coordinate in zip(real._CARRIED, real._COORDINATE):
        if f"kms_{letter}[{coordinate}])" in tail:
            raise AssertionError(
                f"kms_{letter} survived the sub-lattice rename; it is the INTEGER "
                f"vector in the curl body and the HALF-INTEGER one here, and the two "
                f"would collide on the bare name")
    return tail


def dispersive_fused_electric_pair_source(counts: Sequence[int]) -> str:
    """The whole fused kernel at one arity: two lifted preludes, two lifted bodies.

    PURE ASCII, and that is a COMPILE REQUIREMENT rather than a style rule:
    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source with a bare
    ``open(..., 'w')``, so the bytes go through the interpreter's locale encoding.
    """
    # THE CERTIFIED HALVES ARE ASKED FOR FIRST, before the arity is even normalised: a
    # missing half is not a degraded emit, and the refusal has to name the modules one
    # frame from the caller rather than surface as an AttributeError on whichever
    # helper happened to be reached first.
    _certified_halves()
    counts = dispersive_kernels.normalized_pole_counts(counts)
    source = "".join((
        dispersive_fused_electric_pair_prelude(),
        dispersive_fused_electric_pair_signature(counts),
        real.certified_curl_body(),
        real.pre_clear_registers(),
        real.zero_metal_carry(),
        "\n    // --- pole-aware update_E (stepping.update_E:968-989 / "
        "_apply_constitutive_pml:2065) --\n"
        "    // Its three sources are the registers above, not a reload of D, and\n"
        "    // each runs again at every ghost cell this thread owns -- there with\n"
        "    // the DESTINATION's pole words, because update_E is element-wise over\n"
        "    // the whole volume and an imaged cell subtracts its own P.\n",
        certified_dispersive_constitutive_body(counts),
        "}\n",
    ))
    try:
        source.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AssertionError(
            f"the fused device source is not pure ASCII ({exc}); NVRTC's source file "
            f"is written through the interpreter's locale encoding and this would "
            f"fail at first launch, not at import") from exc
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by ``name|arity`` -- for a digest.

    ONE ENTRY PER SWEPT ARITY, not one per family: the arity is a compile-time axis
    here, so a digest over a single body would pin one of six strings and let the other
    five drift. The key carries the arity for the same reason
    ``dispersive_kernels.corpus_digest`` puts it in the digest's own text.
    """
    return {f"{KERNEL_NAME}|{'-'.join(str(v) for v in counts)}":
            dispersive_fused_electric_pair_source(counts)
            for counts in swept_arities()}


#: NVRTC compile options -- CORRECTNESS, not performance, and identical to both
#: certified halves'. Spelled here rather than imported so that loading this file by
#: path (which the bit-identity probe does) cannot pick up a different tuple.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one cell per lane -- what every real sibling landed on.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(counts: Sequence[int], name: str = KERNEL_NAME):
    """Compile the fused kernel at one arity, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING: it is part of the memo
    key, so it has to be built before there is a key to miss on, and a gate mutates
    this family by monkeypatching the certified halves. A source memoized at first call
    would hand back the pre-mutation string forever -- a leg reporting a pass for a
    mutation it never applied.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    code = dispersive_fused_electric_pair_source(counts)
    key = compile_cache.kernel_cache_key(name, True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def _dispersive_predicate():
    """``covers_real_pml_dispersive_constitutive``, importable without a device.

    ``dispersive_kernels`` imports CuPy at module scope, so on a laptop the ordinary
    import fails and the predicate -- the half whose failure mode is a SILENT wrong
    answer -- would be unreachable. It is loaded by path there, exactly as the real
    twin loads ``coverage``.
    """
    if dispersive_kernels is not None:
        return dispersive_kernels.covers_real_pml_dispersive_constitutive
    import importlib.util as _importlib_util  # noqa: PLC0415
    import os as _os  # noqa: PLC0415

    here = _os.path.dirname(_os.path.abspath(__file__))
    spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_dispersive_kernels_predicate",
        _os.path.join(here, "dispersive_kernels.py"))
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.covers_real_pml_dispersive_constitutive



def covers_dispersive_fused_electric_pair(fields: Any, pml: Any, grid: Any,
                                          sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> the two fills -> pole-aware ``update_E``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. The curl half is the real twin's own
    predicate; the constitutive half is
    ``dispersive_kernels.covers_real_pml_dispersive_constitutive``, whole. Every seam
    clause the real twin adds -- the two fills, the injection route, the wall reading,
    the fold/wall disjointness, the stored-extent agreement, the near-fill source row,
    the inverse-epsilon extents -- is asked THROUGH that predicate rather than restated
    here, so a clause added there reaches this family without a second edit.

    WHAT THIS PREDICATE ADDS OVER THE CONJUNCTION IS ONE FACT: the pole volumes are
    read AT AN IMAGED GHOST'S INDEX, which the stand-alone dispersive arm never does
    (it reads them at ``idx`` alone). The dispersive predicate already requires every
    ``P`` to carry the launch's shape, so what is left is that the ghost index is
    inside it -- which is the real twin's stored-extent clause, and is why that clause
    is asked here rather than assumed.
    """
    # THE REAL TWIN'S PREDICATE, MINUS ITS CONSTITUTIVE HALF. It is asked in two
    # pieces rather than as a whole because its ordinary constitutive clause refuses
    # exactly the runs this family exists for -- ``fields.polarizations`` truthy -- so
    # calling it whole would refuse every configuration and this product would serve
    # nothing while reporting a conjunction.
    covered, reason = real.covers_real_pml_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _dispersive_predicate()(fields, pml, grid)
    if not covered:
        return False, f"dispersive constitutive half: {reason}"

    # THE SEAM CLAUSES, DELEGATED WHOLE to the real twin's predicate by handing it a
    # view of this run with NO polarizations -- the one fact its ordinary constitutive
    # clause refuses on, and the one this family has already answered above through the
    # dispersive predicate. Everything else it asks is about the CURL, the two FILLS,
    # the injection route, the walls and the extents, all of which are identical
    # questions for the two families, and restating them here would be a second copy
    # free to drift from the first.
    covered, reason = real.covers_fused_electric_pair(
        _WithoutPolarizations(fields), pml, grid, sources)
    if not covered:
        return False, reason

    return True, "covered"


class _WithoutPolarizations:
    """``fields`` with an empty ``polarizations``, for the delegated seam clauses.

    NOT A WEAKENING AND NOT A MOCK. The real twin's predicate is asked for its SEAM
    clauses -- the two fills, the injection route, the wall reading, the extents -- and
    its ordinary constitutive clause would refuse before reaching any of them on the
    single fact this family has already answered with the DISPERSIVE predicate above.
    Every other attribute is the live object's, so every clause reads the real run.

    A PROXY RATHER THAN A COPY, because ``Fields`` owns device arrays: copying it would
    duplicate volumes on a laptop and, worse, would let a clause read an array that is
    not the one the launch will bind.
    """

    __slots__ = ("_fields",)

    #: The empty chain the delegated call must see, and the ONLY attribute this proxy
    #: answers itself.
    polarizations: Tuple[Any, ...] = ()

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_fields"), name)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_WithoutPolarizations({object.__getattribute__(self, '_fields')!r})"


# =============================================================================
# THE LAUNCH
# =============================================================================

def pole_bindings(fields: "Fields") -> Tuple[Tuple[Any, ...], Tuple[int, int, int]]:
    """``(P pointers in signature order, the arity triple)``, resolved RIGHT NOW.

    ``dispersive_kernels.resolve_pole_plan`` is the one reading of
    ``Fields.displacement_minus_polarization``'s own filter, and it is resolved per
    launch and never cached: ``PolarizationState.update`` rotates the three buffers
    every step (dispersion.py:689-691), so a plan built before ``update_P`` names last
    step's arrays after it -- stale in a way that still computes.
    """
    plan = dispersive_kernels.resolve_pole_plan(fields)
    counts = dispersive_kernels.pole_counts_of(plan)
    pointers: List[Any] = []
    for target, _source, _axis in dispersive_kernels.ELECTRIC_TERMS:
        pointers.extend(plan[target])
    return tuple(pointers), counts


def assert_disjoint_bindings(fields: "Fields", tables: Dict[str, Dict[str, Any]],
                             poles: Sequence[Any]) -> int:
    """Check the promise every ``__restrict__`` in this signature makes.

    THE REAL TWIN'S CHECK PLUS THE POLE GROUP. Its own check covers the fifteen field
    volumes, both coefficient groups and the three non-restrict inverse-epsilon ones;
    this adds the pole pointers, which ARE ``__restrict__`` here, and asks them against
    everything the twin already collected as well as against each other.

    Returns the number of distinct allocations checked, so a caller can assert that
    something was actually inspected.
    """
    checked = real.assert_disjoint_bindings(fields, tables)
    bound: Dict[int, str] = {}
    for name in real._FIELD_BINDINGS:
        bound[int(getattr(fields, name).data.ptr)] = name
    for key in real._CURL_TABLE_KEYS:
        bound[int(tables["curl"][key].data.ptr)] = f"curl:{key}"
    for key in real._CONSTITUTIVE_TABLE_KEYS:
        bound[int(tables["constitutive"][key].data.ptr)] = f"constitutive:{key}"
    collisions: List[str] = []
    for index, array in enumerate(poles):
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"pole {index} and {bound[pointer]} are the same "
                              f"allocation")
            continue
        bound[pointer] = f"pole {index}"
    for component, volume in zip(real._INVERSE_EPSILON_COMPONENTS,
                                 real.inverse_epsilon_bindings(fields)):
        pointer = int(volume.data.ptr)
        label = bound.get(pointer)
        if label is not None and label.startswith("pole "):
            collisions.append(
                f"inv_eps_{component} and {label} are the same allocation, and the "
                f"second is bound __restrict__")
    if collisions:
        raise ValueError(
            "the dispersive fused electric pair binds every field, table and pole "
            "argument __restrict__, and these arguments alias, which is undefined "
            "behaviour NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return checked + len(tuple(poles))


def launch_dispersive_fused_electric_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], walls: Sequence[int],
        fills: Dict[str, Tuple[Any, ...]], dtdx: float,
        poles: Sequence[Any], counts: Sequence[int],
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch, at this run's pole arity.

    THE ARGUMENT ORDER IS THE SIGNATURE'S and the pole group sits immediately after the
    three inverse-epsilon volumes, which is where
    :func:`dispersive_fused_electric_pair_signature` inserts it.

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
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(getattr(fields, name) for name in real._FIELD_BINDINGS) + tuple(
        real.inverse_epsilon_bindings(fields)
    ) + tuple(poles) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in real._CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in real._CONSTITUTIVE_TABLE_KEYS
    ) + (
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3)) + tuple(
        np.int32(int(fills["near"][axis])) for axis in range(3)
    ) + tuple(
        np.int32(int(fills["reflect"][axis])) for axis in range(3)
    ) + tuple(
        np.float32(float(fills["phase"][axis])) for axis in range(3))
    (kernel or _get_kernel(counts))((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "arity": tuple(int(v) for v in counts),
            "poles_bound": len(tuple(poles)),
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "fills": {key: tuple(fills[key]) for key in ("near", "reflect", "phase")}}


def run_dispersive_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        plan: Optional[Dict[str, List[Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``tables``, ``plan`` and ``kernel`` are the gate's doors, keyword-only. ``plan`` is
    how the pole ORDERING mutation is armed: a harness hands in a reversed chain, which
    a launcher that always resolved its own could not be given.
    """
    covered, reason = covers_dispersive_fused_electric_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = real.fused_electric_pair_tables(pml)
    if plan is None:
        poles, counts = pole_bindings(fields)
    else:
        counts = dispersive_kernels.pole_counts_of(plan)
        poles = tuple(array for target, _s, _a in dispersive_kernels.ELECTRIC_TERMS
                      for array in plan[target])
    assert_disjoint_bindings(fields, tables, poles)
    from . import step_curl_kernels  # noqa: PLC0415 - a device import
    return launch_dispersive_fused_electric_pair(
        fields, tables, step_curl_kernels.real_curl_boundary_codes(grid),
        zero_metal_axes(grid), real.fused_electric_pair_fills(grid), dtdx,
        poles, counts, kernel)
