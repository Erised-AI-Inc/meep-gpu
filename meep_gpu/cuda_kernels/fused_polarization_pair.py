"""The hand-CUDA E->P fused pair: ``update_E`` -> ``update_P``, welded per component.

THE BOARD'S TWO CELLS. ``E_to_P (cuda_dispersive/dispersive, cuda_ade/ADE)``, 7
seam-instances, and ``E_to_P (cuda_no_pml_dispersive/no-PML dispersive store,
cuda_no_pml_ade/no-PML ADE)``, 3 -- all ten clear the source seam on the driver
fact alone and carry NOTHING between the halves
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-08-31_nopmlweld/``, the two
POINTWISE-BUILDABLE ``E_to_P`` cells). The driver advances the polarizations
immediately after ``update_E`` (driver.py:3313 then :3315) with nothing between
the two consults: no injection slot, no fill, no wall pass. So this pair carries
no deposit repair, no boundary code, no phase table and no in-seam register --
what it carries is a SHAPE CHANGE, and that is the whole module.

=============================================================================
THE SHAPE MISMATCH, MEASURED, AND WHAT THE WELD DOES ABOUT IT
=============================================================================

``update_E`` runs in ONE launch for all three components
(``dispersive_kernels.update_E_fused_pml_real_dispersive``), while ``update_P``
runs ONE LAUNCH PER (state, component), because each state's three buffers rotate
per component INSIDE one call and "the retired history becomes the NEXT
component's scratch" (ade_kernels.py:414-424, transcribing dispersion.py:689-691).
The launcher re-reads every slot after each rotation; a launcher that cached the
views "would be stale from the second component of the first step, and stale in a
way that still computes". A single launch spanning both sub-steps whole is
therefore structurally refused by buffer count: advancing all three components of
one state at once needs three outputs distinct from six live inputs, and a state
owns seven buffers.

So the weld SPLITS ``update_E`` INTO THREE PER-COMPONENT LAUNCHES and fuses each
with the ``update_P`` recurrences of its own component: 3 launches replacing
``1 + sum over states of len(driven())``, with the HOST still rotating between
component launches exactly as ``ade_kernels.update_P_fused_pml_real`` rotates
between its own. The rotation code here is that function's, line for line, run
once per driving state after each component launch returns.

WHY THE SPLIT IS BIT-EXACT AGAINST THE DRIVER ORDER, word for word:

* the driver order is ``update_E`` (:3313) then ``update_P`` (:3315), with
  nothing between, "and the equivalence holds only while nothing reads P between
  them" (driver.py:3196-3219, pinned by
  ``test_the_two_orders_of_update_E_and_update_P_agree``);
* the DIAGONAL dispersive ``update_E`` reads P at ITS OWN component only: the
  emitted chain for component c subtracts ``P_<c>_<i>[idx]`` and nothing else
  (``dispersive_kernels._source_lines``), so component c's launch reads nothing
  any earlier component's launch or rotation wrote -- launch x writes ``Ex``,
  ``f_w_Ex`` and state scratches that the rotation moves into the P[x] slots,
  none of which the y or z chain reads;
* ``update_P``'s drive for component c is ``fields.drive_field(c)`` -- ``f_w_<c>``
  under an active layer, the stored ``E<c>`` without one (fields.py:1140-1163) --
  and BOTH are exactly the value the ``update_E`` half of this same thread just
  stored from its ``src`` register. A float32 word stored to global and reloaded
  is the identity on the bits, so handing the register across the seam is exact,
  not close. That identity is a PREDICATE CLAUSE here, checked against the arrays
  this launch binds, never assumed;
* the P recurrence for (state, c) reads that state's own ``P[c]``/``P_prev[c]``
  and writes that state's own ``_scratch`` -- disjoint across states, untouched
  by the E half (whose only writes are ``E<c>``/``f_w_<c>``) -- so fusing all
  driving states of one component into one launch reorders nothing that shares
  data.

WHAT THE SPLIT REFUSES, AND THE ONE SLOT IT COSTS. The OFF-DIAGONAL dispersive
``update_E`` (``dispersive_offdiag_update_e``) forms a row product that reads the
OTHER components' ``D - sum P`` volumes at neighbouring cells
(stepping.py:1196-1251). Split per component, launch y would read ``P[x]`` AFTER
launch x's rotation moved the advanced buffer into that slot -- the driver order
reads the OLD one, so the weld would diverge, and diverge in a way that still
computes. The cell ``E_to_P (cuda_dispersive_offdiag/dispersive off-diagonal,
cuda_ade/ADE)`` -- 1 instance, ``absorbed_power_density.py`` -- is therefore
REFUSED by both predicates below through their update_E halves, which refuse an
off-diagonal row by name. Serving it would take a different weld shape (all three
components in one launch, which the buffer count above refuses), not a wider
predicate.

=============================================================================
THE ALIASING HAZARD, AND WHERE IT LIVES ON THIS SEAM
=============================================================================

On the D/E welds the shared volume is the flux density. Here it is THE POLE
BUFFER: the E half's chain reads ``state.P[c]`` and the P half's recurrence reads
the SAME array as its ``p_now``. Binding one allocation to two ``__restrict__``
parameters is undefined behaviour NVRTC miscompiles without a diagnostic, so each
``P_<c>_<i>`` appears EXACTLY ONCE in the emitted signature and BOTH halves read
through that one parameter. The certified ADE kernel's ``p_now`` parameter is
therefore not in this signature at all -- see :data:`LIFT_EDITS`.

The recurrence's other pointers -- ``p_out_<i>`` (the state's scratch),
``p_prev_<i>``, a volume ``sigma_<i>`` -- are ``__restrict__`` as in the
certified kernel, and :func:`assert_disjoint_bindings` checks every launch's
bindings pairwise, per component, AFTER the rotation that decided them, for the
certified launcher's reason: the rotation moves three names between launches, so
what the predicate checked is one permutation and this is the one being run.
``inv_eps_<c>`` keeps the certified dispersive kernel's declaration -- NOT
``__restrict__`` -- lifted rather than re-decided.

=============================================================================
TWO PRODUCTS, ONE MODULE, DISJOINT BY THE ABSORBER
=============================================================================

``cuda_fused_polarization_pair`` welds arm 1 (``update_E_pml_real_dispersive``,
stepping.py:1014-1018) to the PML-ADE ``update_P``; its predicate conjoins
``covers_real_pml_dispersive_constitutive`` and ``covers_real_pml_ade_update_p``,
which REQUIRE an active layer. ``cuda_no_pml_fused_polarization_pair`` welds arm
2 (``update_E_no_pml_real_dispersive``, :990-993) to the same certified ADE
kernels under ``covers_no_pml_dispersive_constitutive`` and
``covers_no_pml_ade_update_p``, which refuse one by name. The two candidates on
the ``update_E`` seam partition on that single boolean, exactly as the four
``step_D`` products partition -- two admitters would leave the seam UNFUSED
naming both (``fused_pairs.install_fused_pairs``), so the disjointness is what
buys the slots.

They live in one module because ``dispersive_kernels`` already hosts both
update_E arms and this weld lifts from it either way; the emitters share every
anchor and differ where the certified arms differ.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch. This
module ships two predicates, an emitter and a launcher; ``fused_pairs`` can plan
it opt-in (``arms.plan_step(..., fuse=True)``), and no shipped code path launches
it.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # a host with no CuPy: the emitter and both predicates still run
    import cupy as cp
except ImportError:
    cp = None

# THE CERTIFIED TEXT. ``dispersive_kernels`` imports no CuPy at module scope, so
# the E half's emitter is importable on a laptop. ``ade_kernels`` imports CuPy at
# scope, so its device strings are read OFF THE SYNTAX TREE instead -- the same
# route ``test_ade_update_p.py`` takes, and the reason
# :func:`_certified_ade_strings` exists.
try:
    from . import compile_cache
    from . import dispersive_kernels as certified_e
    from .coverage import (_base_address, ade_sigma_is_volume,
                           covers_real_pml_ade_update_p)
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util

    def _load(stem):
        here = os.path.dirname(os.path.abspath(__file__))
        spec = _importlib_util.spec_from_file_location(
            f"cuda_kernels_{stem}", os.path.join(here, f"{stem}.py"))
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    compile_cache = _load("compile_cache")
    certified_e = _load("dispersive_kernels")
    _coverage = _load("coverage")
    _base_address = _coverage._base_address
    ade_sigma_is_volume = _coverage.ade_sigma_is_volume
    covers_real_pml_ade_update_p = _coverage.covers_real_pml_ade_update_p


# =============================================================================
# THE PARTITION -- EMPTY ON THE CERTIFIED SIDE UNTIL A GATE MOVES IT
# =============================================================================
#
# Plain assignments with NO type annotation: the partition readers walk the syntax
# tree without importing the module, and an annotated assignment is an
# ``ast.AnnAssign`` those readers do not match.

#: RECORDED 2026-09-01 as ``cuda_fused_polarization_pair_2026-09-01`` and
#: ``cuda_no_pml_fused_polarization_pair_2026-09-01`` in ``certification.json``,
#: one block per family, cut in one campaign on the GPU host (RTX A6000, cc 8.6,
#: CuPy 13.5.1, NVRTC 11.6) by
#: ``parity/meep_gpu/gate_cuda_fused_polarization_pair.py``. All four legs
#: RELEASED under BOTH float32 subnormal policies: 180/180 (PML arm) and 90/90
#: (no-absorber arm) cases per policy bit-identical PER COMPLETE SEAM STEP to
#: ``stepping.update_E`` -> ``stepping.update_P`` AND to the separately
#: certified kernels the weld replaces, pole volumes and retired scratch in the
#: comparison, over 60 consecutive steps, with 3 fused launches per step counted
#: two independent ways. THE SEAM NULL is what carries the register hand-off:
#: ``reload_drive_from_global`` -- the register rewritten back into the certified
#: global reload -- came back NULL CONFIRMED on every leg, and its control
#: ``wrong_drive_register`` (D handed across instead) CAUGHT on every leg. The
#: three rotation defects (``stale_pole_pointers``, ``rotate_before_launch``,
#: ``freeze_one_states_rotation``) were each CAUGHT against a local copy of the
#: shipped loop whose no-defect twin was required UNCAUGHT first. The SUBJECT
#: module was checked unchanged since those runs before this record was landed.
CERTIFIED_KERNELS = (
    "fused_polarization_pair_pml_real",
    "fused_polarization_pair_no_pml_real",
)
#: Shipped without a gate verdict. EMPTY since 2026-09-01, and what emptied it
#: was a run rather than an argument: a fused product's bit-identity is a claim
#: about the WELD, which no verdict on either half establishes -- the halves were
#: certified individually (``cuda_dispersive_2026-08-27`` arms 1-3,
#: ``ade_2026-08-19``) and none of those is a verdict about the two composed per
#: component with the drive handed across in a register; the fused gate is.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its launches with the deposit repair? FALSE, AND IT
#: IS A DRIVER FACT RATHER THAN A MEASUREMENT OF THIS CELL'S ROWS: the driver
#: injects nothing between ``update_E`` (driver.py:3313) and ``update_P``
#: (:3315) on ANY run -- there is no source slot in this seam for a repair to
#: carry, unlike the B and D seams, where the flag is measured per cell. The
#: seam row in ``fused_pairs.FUSED_PAIR_SEAMS`` records the same fact as a
#: ``None`` deposit-list name, and ``test_fused_pairs.py`` pins the two together.
CARRIES_DEPOSIT_REPAIR = False

#: The two families, and the arm each welds. Keys are the ``arm`` argument every
#: function here takes; the labels under ``registry._TABLE`` are what
#: ``fused_pairs.FUSED_PAIR_ARMS`` binds per family.
FAMILY_PML = "cuda_fused_polarization_pair"
FAMILY_NO_PML = "cuda_no_pml_fused_polarization_pair"
FAMILIES: Dict[str, str] = {"pml": FAMILY_PML, "no_pml": FAMILY_NO_PML}

#: The ``extern "C"`` symbol each arm's weld is emitted under. The SOURCE varies
#: with the component, its pole count and its sigma kinds; the name does not,
#: which is ``dispersive_kernels``' own convention for a compile-time axis.
KERNEL_KEYS: Dict[str, str] = {
    "pml": "fused_polarization_pair_pml_real",
    "no_pml": "fused_polarization_pair_no_pml_real",
}

#: The kernel LABEL per family, in the spelling the board and census carry.
KERNEL_NAME = "fused_polarization_pair_(no_)pml_real"

#: The driver passes ONE ``run`` of this product performs (three launches, host
#: rotations between), in driver order (driver.py:3313, :3315). Declared rather
#: than inferred from the two slots. THERE IS NO PASS BETWEEN THEM TO CARRY OR
#: REFUSE: the E->P seam has no injection, no fill and no wall clear -- the two
#: consults are adjacent statements.
REPLACES: Tuple[str, ...] = ("update_E", "update_P")

#: The sub-step slot the planner holds this on (the seam's first slot).
SLOT = "update_E"

#: The three E components with their sources and axes, re-read from the certified
#: family rather than retyped -- component c indexes AXIS c's coefficient vector.
ELECTRIC_TERMS = certified_e.ELECTRIC_TERMS

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    float w = drive[idx];",
     "became": "    float w = src_<axis>;",
     "why": "THE SEAM. The certified update_P loads the drive from the array "
            "fields.drive_field(component) names -- f_w_<c> under an active "
            "layer, the stored E<c> without one -- and BOTH hold exactly the "
            "value the update_E half of this same thread just stored from its "
            "src register (constitutive_apply stores fw[idx] = src on the PML "
            "arm; the no-PML arm stores E[idx] = src). A float32 word stored to "
            "global and reloaded is the identity on the bits. The predicate "
            "checks the drive identity against the arrays this launch binds; "
            "the load is emitted ONCE per component rather than once per pole, "
            "because every pole of a component reads the same array and nothing "
            "in this seam writes it between the certified launches."},
    {"line": "    const float* __restrict__ p_now,",
     "became": "(not a parameter: the pole pointer P_<c>_<i> is bound once)",
     "why": "the E half's chain reads state.P[c] and the recurrence's p_now IS "
            "state.P[c] -- one allocation. Binding it twice with __restrict__ "
            "on both is undefined behaviour, so the signature binds it once and "
            "both halves read through that parameter. The recurrence's load and "
            "arithmetic are untouched; only the pointer's NAME changed."},
    {"line": "    float p = p_now[idx]; (and q, s; c_now, c_prev, c_drive)",
     "became": "    float p_<i> = P_<c>_<i>[idx]; (suffixed per pole)",
     "why": "one launch now carries every driving state's recurrence for one "
            "component, so each pole's locals and scalar coefficients carry the "
            "pole's index. Names only: the loads, the grouping "
            "((p*c_now) + (c_prev*q)) + (c_drive*(s*w)) and the store are the "
            "certified line under a checked substitution that raises if the "
            "certified string drifts."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= n_elem) return; (the ADE kernel's own guard)",
     "became": "(dropped: the E half's guard has already returned)",
     "why": "the update_E body above already declares idx and guarded the "
            "extent; splicing the ADE guard would redeclare idx. On the PML arm "
            "n_elem is therefore not a parameter at all, and the predicate "
            "requires every P/P_prev/scratch/sigma volume to carry exactly the "
            "stored extent the E half walks, so the dropped bound is the same "
            "number the certified kernel would have used."},
    {"line": "(the certified update_E, one launch for all three components)",
     "became": "(three per-component launches, host rotation between)",
     "why": "THE SPLIT. update_P's buffers rotate per component on the host, so "
            "a whole-grid update_E cannot fuse with it -- see the module "
            "docstring for the buffer count and the word-for-word equivalence "
            "argument. Each per-component kernel's chain, inverse-epsilon "
            "multiply and store are the certified emitter's own lines for that "
            "component, lifted by exact anchor from the full emission; the "
            "other two components' lines are in the other two launches."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILIES", "FAMILY_NO_PML",
    "FAMILY_PML", "KERNEL_KEYS", "KERNEL_NAME", "LIFT_EDITS", "REPLACES", "SLOT",
    "UNCERTIFIED_KERNELS", "assert_disjoint_bindings", "component_chain",
    "component_specs", "covers_fused_polarization_pair",
    "covers_no_pml_fused_polarization_pair", "device_sources",
    "fused_polarization_pair_source", "kernel_name",
    "launch_fused_polarization_component", "run_fused_polarization_pair",
]


# =============================================================================
# THE LIFT -- E HALF (dispersive_kernels' own emitter, restricted per component)
# =============================================================================

def kernel_name(arm: str) -> str:
    """The ``extern "C"`` symbol NVRTC is asked for, by arm. A refusal, not a default."""
    if arm not in KERNEL_KEYS:
        raise ValueError(
            f"arm must be one of {sorted(KERNEL_KEYS)}, got {arm!r}; the PML "
            f"accumulation and the no-PML store are different arithmetic and "
            f"neither is a default")
    return KERNEL_KEYS[arm]


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    """Substitute ``old`` exactly once, or raise naming what was being lifted."""
    if text.count(old) != 1:
        raise AssertionError(
            f"the certified text carries {text.count(old)} occurrences of {what}, "
            f"not 1; this family LIFTS it rather than retyping it and cannot "
            f"splice around an anchor that no longer identifies a single site")
    return text.replace(old, new, 1)


def _counts_with(component_index: int, count: int) -> Tuple[int, int, int]:
    """A pole triple with ``count`` at one component and zero elsewhere."""
    values = [0, 0, 0]
    values[component_index] = int(count)
    return tuple(values)  # type: ignore[return-value]


#: The zero-count line each component emits when nothing drives it, exactly as
#: ``dispersive_kernels._source_lines`` writes it (the x line carries the
#: certified trailing comment). Used as REMOVAL anchors when another component's
#: block is being lifted, and re-checked against the emitter on every call.
def _zero_line(component_index: int) -> str:
    target, source, axis = ELECTRIC_TERMS[component_index]
    line = f"    float src_{axis} = {source}[idx] * inv_eps_{target}[idx];"
    if axis == "x":
        line += "   // stepping.py:982: source * inv_eps"  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011
    return line


def component_chain(component_index: int, count: int) -> str:
    """The certified ``D - sum P`` chain and inverse-epsilon multiply for ONE component.

    LIFTED from ``dispersive_kernels._source_lines`` -- the same function whose
    output the certified kernels compile -- by emitting a triple with this
    component's count and zero elsewhere, then removing the other two components'
    zero-count lines by exact anchor. What remains must mention only this
    component's identifiers, and that is asserted rather than trusted.
    """
    emitted = certified_e._source_lines(_counts_with(component_index, count))
    for other in range(3):
        if other == component_index:
            continue
        emitted = _replace_once(
            emitted, _zero_line(other),
            "", f"component {ELECTRIC_TERMS[other][0]}'s zero-count line")
    block = "\n".join(line for line in emitted.splitlines() if line.strip())
    foreign = [term for index, term in enumerate(ELECTRIC_TERMS)
               if index != component_index
               for name in (term[0], term[1], f"src_{term[2]}", f"s_{term[2]}")
               if name in block]
    if foreign:
        raise AssertionError(
            f"the lifted chain for {ELECTRIC_TERMS[component_index][0]} still "
            f"mentions {foreign}; the per-component restriction would bind "
            f"volumes this kernel does not declare")
    return block


#: The certified PML index block and per-component tail, anchored against the
#: emitter's own output in :func:`_certified_e_pieces` rather than trusted.
_PML_INDEX_BLOCK = (
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (idx >= nx * ny * nz) return;\n"
    "\n"
    "    int k = idx % nz;\n"
    "    int j = (idx / nz) % ny;\n"
    "    int i = idx / (ny * nz);\n")
_NO_PML_INDEX_BLOCK = (
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (idx >= n_elem) return;\n")
_AXIS_INDEX = {"x": "i", "y": "j", "z": "k"}


def _pml_tail_line(component_index: int) -> str:
    target, _source, axis = ELECTRIC_TERMS[component_index]
    letter = _AXIS_INDEX[axis]
    return (f"    constitutive_apply({target}, f_w_{target}, idx, src_{axis}, "
            f"kps_{axis}[{letter}], kms_{axis}[{letter}]);")


def _no_pml_tail_line(component_index: int) -> str:
    target, _source, axis = ELECTRIC_TERMS[component_index]
    return f"    {target}[idx] = src_{axis};"


def _certified_e_pieces(arm: str, component_index: int, count: int
                        ) -> Tuple[str, str, str]:
    """(prelude, index block, tail line) for one arm, each verified against the
    certified emitter's own full emission at this component's count.

    The full emission is built at the reference triple and every piece this
    module splices is required to appear in it verbatim -- so a drift in the
    certified emitter fails here, on a laptop, rather than compiling a weld that
    is quietly not the certified arithmetic.
    """
    reference = certified_e.dispersive_source(
        arm, _counts_with(component_index, count))
    prelude, _sig, _rest = reference.partition('extern "C" __global__ void')
    if not _sig:
        raise AssertionError(
            "the certified dispersive emission no longer carries an extern-C "
            "signature; there is no prelude to lift")
    index_block = _PML_INDEX_BLOCK if arm == "pml" else _NO_PML_INDEX_BLOCK
    if index_block not in reference:
        raise AssertionError(
            f"the certified {arm} emission no longer carries the index block "
            f"this weld lifts; the guard would be retyped rather than lifted")
    tail = (_pml_tail_line(component_index) if arm == "pml"
            else _no_pml_tail_line(component_index))
    if tail not in reference:
        raise AssertionError(
            f"the certified {arm} emission no longer carries "
            f"{tail.strip()!r}; the constitutive tail would be retyped rather "
            f"than lifted")
    chain = component_chain(component_index, count)
    for line in chain.splitlines():
        if line not in reference:
            raise AssertionError(
                f"lifted chain line {line!r} is not in the certified emission")
    return prelude, index_block, tail


# =============================================================================
# THE LIFT -- P HALF (ade_kernels' device strings, read off the syntax tree)
# =============================================================================
#
# ``ade_kernels`` imports CuPy at module scope, so its strings are recovered the
# way ``test_ade_update_p.py`` reads them: parse the file, fold the string
# concatenations. The result is the exact text NVRTC compiles on a device host.

_ADE_STRING_NAMES = ("_ADE_UPDATE_P_PROLOGUE",
                     "_update_P_pml_real_kernel_code",
                     "_update_P_pml_real_uniform_kernel_code")


def _fold_strings(node: ast.AST, resolved: Dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in resolved:
        return resolved[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (_fold_strings(node.left, resolved)
                + _fold_strings(node.right, resolved))
    raise AssertionError(
        f"ade_kernels device text is no longer a foldable string expression "
        f"({ast.dump(node)[:80]}...); the P half cannot be lifted")


#: ``{ade_kernels.py text: its three folded device strings}`` -- the parse memo
#: behind :func:`_certified_ade_strings`. Keyed on the WHOLE TEXT and written only
#: after a parse returned, so it holds successes alone; one entry per distinct text
#: this process has read, which is one on every shipped path.
_ADE_STRINGS_BY_TEXT: Dict[str, Dict[str, str]] = {}


def _read_ade_kernels_text() -> str:
    """``ade_kernels.py``'s text, read off the disk on EVERY call."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ade_kernels.py")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _parse_ade_strings(text: str) -> Dict[str, str]:
    """The three ADE device strings folded out of ``text``, parsed once per text.

    Returns a FRESH dict on every call, hit or miss, so a caller that edits what
    it was handed cannot edit the memo. A text that does not fold, or that no
    longer assigns every name in :data:`_ADE_STRING_NAMES`, raises here and is
    NOT stored: the memo is written after the checks return, so a drifted text
    raises on every call rather than once.
    """
    cached = _ADE_STRINGS_BY_TEXT.get(text)
    if cached is None:
        tree = ast.parse(text)
        resolved: Dict[str, str] = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in _ADE_STRING_NAMES):
                resolved[node.targets[0].id] = _fold_strings(node.value,
                                                             resolved)
        missing = [name for name in _ADE_STRING_NAMES if name not in resolved]
        if missing:
            raise AssertionError(
                f"ade_kernels.py no longer assigns {missing}; the P half cannot "
                f"be lifted")
        _ADE_STRINGS_BY_TEXT[text] = resolved
        cached = resolved
    return dict(cached)


def _certified_ade_strings() -> Dict[str, str]:
    """The three ADE device strings, exactly as the certified module assigns them.

    READ ON EVERY CALL, PARSED ONCE PER DISTINCT TEXT (2026-09-19). Every emission
    of a fused polarization kernel asks for these strings twice -- once through
    :func:`_verified_ade_bodies`, once through :func:`_ade_prologue` -- and each
    ask used to open ``ade_kernels.py`` and run ``ast.parse`` plus
    :func:`_fold_strings` over the whole file. MEASURED 2026-09-19 on the laptop
    (Apple M1 Max, no device, 100 calls per row after one warm-up, the pre-edit
    file loaded beside this one): this function 855 -> 33 us per call; one
    emission of :func:`fused_polarization_pair_source` at one pole 1744 -> 142 us
    and at six 1765 -> 200 us; ``three_slot_polarization_source`` 1610 -> 93 us
    and 1788 -> 150 us; the off-diagonal weld's ``kernel_source`` at the corpus
    point 1912 -> 248 us. The emitted text is byte-identical over every key the
    three emitters admit (762, 378 and 42 336 keys). What a hit still costs is
    the read, one hash and one comparison of the fresh 26 KB string.

    THE KEY IS THE TEXT, NOT A STAT OF THE FILE. The file is still opened and read
    on every call, exactly as before, and only the parse of a text already seen is
    skipped. An mtime or size key is the stale-``.pyc`` class: a same-second,
    size-preserving edit of ``ade_kernels.py`` would be served the pre-edit
    strings. Keyed on the text, an edited file is honoured on the next call, and a
    probe that monkeypatches this function (``test_fused_polarization_pair.py``)
    still reaches both callers, which look it up through the module at call time.

    THE EMITTED SOURCE IS STILL NOT MEMOIZED, and that is not an oversight: each
    ``_get_kernel`` on this track emits per call because its ``SOURCE_TRANSFORM``
    is applied to the emitted text and the text is part of the compile memo key
    (:func:`_get_kernel`). This memo sits one level below, on the certified INPUT,
    which no probe door rewrites.

    THREE MODULES SHARE THIS READER, through :func:`_verified_ade_bodies` and
    :func:`_ade_prologue`: this one, ``three_slot_dispersive_weld`` (and its
    no-absorber twin, which re-exports that emitter) and
    ``dispersive_offdiag_fused_polarization_pair``. A copy of this file loaded by
    path (the probes' ``_load``) is a second module object with its own memo --
    two memos, each correct.
    """
    return _parse_ade_strings(_read_ade_kernels_text())


#: The certified recurrence bodies, one per sigma form, each the 5/4-line block
#: between the ADE kernel's guard and its closing brace.
_ADE_GUARD = ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
              "    if (idx >= n_elem) return;\n\n")
_ADE_VOLUME_BODY = ("    float p = p_now[idx];\n"
                    "    float q = p_prev[idx];\n"
                    "    float w = drive[idx];\n"
                    "    float s = sigma[idx];\n"
                    "    p_out[idx] = ((p * c_now) + (c_prev * q)) + "
                    "(c_drive * (s * w));\n")
_ADE_UNIFORM_BODY = _ADE_VOLUME_BODY.replace("    float s = sigma[idx];\n",
                                             "    float s = sigma;\n")


def _verified_ade_bodies() -> Dict[bool, str]:
    """{sigma_is_volume: certified body}, verified against ade_kernels' own text."""
    strings = _certified_ade_strings()
    out: Dict[bool, str] = {}
    for volume, name, body in (
            (True, "_update_P_pml_real_kernel_code", _ADE_VOLUME_BODY),
            (False, "_update_P_pml_real_uniform_kernel_code", _ADE_UNIFORM_BODY)):
        source = strings[name]
        if _ADE_GUARD + body not in source:
            raise AssertionError(
                f"ade_kernels.{name} no longer carries the recurrence block this "
                f"weld lifts; the P half would be retyped rather than lifted")
        out[volume] = body
    return out


#: The certified ADE provenance comment, spliced above the fused signature so the
#: emitted text carries the recurrence's own citation. Verified in
#: :func:`_verified_ade_bodies`' caller against the same parse.
def _ade_prologue() -> str:
    return _certified_ade_strings()["_ADE_UPDATE_P_PROLOGUE"]


def _pole_block(component_index: int, pole_index: int, volume: bool,
                bodies: Dict[bool, str]) -> str:
    """One pole's recurrence, built from the certified body by checked renames.

    Every substitution is applied with :func:`_replace_once`, so a certified
    string that drifts raises here instead of emitting a block that is quietly
    not the certified arithmetic. The drive load is NOT in this block -- it is
    the seam, emitted once per component by the caller.
    """
    target = ELECTRIC_TERMS[component_index][0]
    i = pole_index
    body = bodies[volume]
    body = _replace_once(body, "    float w = drive[idx];\n", "",
                         "the certified drive load (moved to the seam)")
    body = _replace_once(body, "p_now[idx]", f"P_{target}_{i}[idx]",
                         "the recurrence's P^n load")
    body = _replace_once(body, "p_prev[idx]", f"p_prev_{i}[idx]",
                         "the recurrence's P^(n-1) load")
    if volume:
        body = _replace_once(body, "sigma[idx]", f"sigma_{i}[idx]",
                             "the volume sigma load")
    else:
        body = _replace_once(body, "float s = sigma;", f"float s = sigma_{i};",
                             "the uniform sigma bind")
    body = _replace_once(body, "p_out[idx]", f"p_out_{i}[idx]",
                         "the recurrence's store")
    for name in ("p", "q", "s"):
        body = _replace_once(body, f"float {name} ", f"float {name}_{i} ",
                             f"the {name} declaration")
    for name, suffixed in (("p", f"p_{i}"), ("q", f"q_{i}"), ("s", f"s_{i}")):
        body = _replace_once(body, f"({name} * c_now)" if name == "p"
                             else f"(c_prev * {name})" if name == "q"
                             else f"({name} * w)",
                             f"({suffixed} * c_now_{i})" if name == "p"
                             else f"(c_prev_{i} * {suffixed})" if name == "q"
                             else f"({suffixed} * w)",
                             f"the {name} use in the recurrence")
    body = _replace_once(body, "(c_drive * ", f"(c_drive_{i} * ",
                         "the drive coefficient")
    return body


# =============================================================================
# THE FUSED SIGNATURE AND SOURCE
# =============================================================================

def _signature(arm: str, component_index: int, count: int,
               sigma_kinds: Sequence[bool]) -> str:
    """The fused signature for one arm, one component, one specialization.

    The ONLY hand-written device text in this module, and it is a signature: no
    arithmetic lives here. Each pole pointer appears EXACTLY ONCE -- see the
    module docstring's aliasing section.
    """
    target, source, axis = ELECTRIC_TERMS[component_index]
    lines: List[str] = [f'extern "C" __global__ void {kernel_name(arm)}(']
    if arm == "pml":
        lines.append(f"    float* __restrict__ {target}, "
                     f"float* __restrict__ f_w_{target},")
    else:
        lines.append(f"    float* __restrict__ {target},")
    lines.append(f"    const float* __restrict__ {source},")
    # NOT __restrict__, lifted from the certified dispersive signature rather
    # than re-decided: an isotropic run hands one allocation three times there,
    # and one binding here keeps the same declared promise.
    lines.append(f"    const float* inv_eps_{target},")
    for i in range(count):
        # THE SHARED VOLUME, BOUND ONCE: the E half's chain and the P half's
        # recurrence both read this parameter.
        lines.append(f"    const float* __restrict__ P_{target}_{i},")
    for i in range(count):
        sigma = (f"const float* __restrict__ sigma_{i}," if sigma_kinds[i]
                 else f"float sigma_{i},")
        lines.append(f"    float* __restrict__ p_out_{i}, "
                     f"const float* __restrict__ p_prev_{i}, {sigma}")
        lines.append(f"    float c_now_{i}, float c_prev_{i}, "
                     f"float c_drive_{i},")
    if arm == "pml":
        lines.append("    int nx, int ny, int nz,")
        lines.append(f"    const float* __restrict__ kps_{axis}, "
                     f"const float* __restrict__ kms_{axis}")
    else:
        lines.append("    int n_elem")
    lines.append(") {")
    return "\n".join(lines) + "\n"


def fused_polarization_pair_source(arm: str, component_index: int, count: int,
                                   sigma_kinds: Sequence[bool]) -> str:
    """The whole fused kernel for one arm, one component, one specialization.

    ``sigma_kinds[i]`` is ``ade_sigma_is_volume``'s answer for the i-th driving
    state -- a compile-time axis exactly as it is in the certified ADE family,
    here one flag per pole because one launch carries every driving state.
    """
    if arm not in KERNEL_KEYS:
        raise ValueError(f"arm must be one of {sorted(KERNEL_KEYS)}, got {arm!r}")
    if not 0 <= int(component_index) <= 2:
        raise ValueError(f"component_index must be 0..2, got {component_index!r}")
    count = int(count)
    # The certified cap is a MEASURED ceiling; reuse its refusal verbatim.
    certified_e.normalized_pole_counts(_counts_with(component_index, count))
    kinds = tuple(bool(kind) for kind in sigma_kinds)
    if len(kinds) != count:
        raise ValueError(
            f"sigma_kinds carries {len(kinds)} flags for {count} poles; one "
            f"compile-time sigma form per driving state, in registration order")

    target, _source, axis = ELECTRIC_TERMS[component_index]
    prelude, index_block, tail = _certified_e_pieces(arm, component_index, count)
    bodies = _verified_ade_bodies()
    seam = (
        "\n"
        "    // THE SEAM. stepping.update_P binds fields.drive_field(component)\n"
        "    // as w -- " + ("f_w_" + target + " under this active layer, the "
                             "array the\n    // constitutive_apply above just "
                             "stored src_" + axis + " into" if arm == "pml"
                             else "the stored " + target + " without a layer, "
                             "the array the\n    // store above just wrote from "
                             "src_" + axis) + ". A float32 word\n"
        "    // stored and reloaded is the identity on the bits, so the register\n"
        "    // IS the certified load. One load per component, not per pole: every\n"
        "    // pole of a component reads the same array, and nothing between the\n"
        "    // certified launches writes it.\n"
        f"    float w = src_{axis};\n")
    poles = "".join(
        "\n"
        f"    // pole {i}: dispersion.PolarizationState.update, the certified\n"
        f"    // recurrence under this launch's per-pole names.\n"
        + _pole_block(component_index, i, kinds[i], bodies)
        for i in range(count))
    source = "".join((
        prelude,
        # The ADE provenance rides only a kernel that carries a recurrence: a
        # zero-arity component's kernel is the E half alone.
        _ade_prologue() if count else "",
        _signature(arm, component_index, count, kinds),
        index_block,
        "\n",
        component_chain(component_index, count),
        "\n",
        tail + "\n" if count == 0 else tail + "\n" + seam + poles,
        "}\n",
    ))
    if "__" in (source.replace("__restrict__", "").replace("__global__", "")
                .replace("__device__", "").replace("__forceinline__", "")
                .replace("blockIdx", "").replace("blockDim", "")
                .replace("threadIdx", "")):
        raise AssertionError("an unsubstituted placeholder survived the splice")
    # PURE ASCII IS A COMPILE REQUIREMENT: NVRTC receives these bytes through the
    # interpreter's locale encoding.
    source.encode("ascii")
    return source


def _digest_specs() -> Tuple[Tuple[str, int, int, Tuple[bool, ...]], ...]:
    """The specializations a digest covers: both arms, the corpus-swept counts,
    both sigma forms and a mixed one at every count that can hold one."""
    specs: List[Tuple[str, int, int, Tuple[bool, ...]]] = []
    for arm in ("pml", "no_pml"):
        for component in (0, 1, 2):
            for count in (0, 1, 2, 5, 6):
                kind_sets = {tuple([True] * count), tuple([False] * count),
                             tuple(bool(i % 2) for i in range(count))}
                for kinds in sorted(kind_sets):
                    specs.append((arm, component, count, kinds))
    return tuple(specs)


def device_sources() -> Dict[str, str]:
    """Every source a digest pins, keyed ``arm/component/count/kinds``."""
    out: Dict[str, str] = {}
    for arm, component, count, kinds in _digest_specs():
        key = (f"{arm}/{ELECTRIC_TERMS[component][0]}/{count}/"
               + "".join("v" if kind else "u" for kind in kinds))
        out[key] = fused_polarization_pair_source(arm, component, count, kinds)
    return out


# =============================================================================
# COMPILATION AND LAUNCH
# =============================================================================

#: NVRTC compile options -- CORRECTNESS, not performance, identical to both
#: certified halves'. Spelled here rather than imported so loading this file by
#: path cannot pick up a different tuple than the one a gate compiled.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one cell per lane -- both halves' geometry.
_FUSED_THREADS = 256

#: THE GATE'S DOOR into the emitted text. ``None`` in every shipped path; a
#: bit-identity probe assigns an ``(arm, component_index, count, kinds, source)
#: -> source`` callable to plant a defect in a body that does not exist until it
#: is asked for -- ``dispersive_kernels.SOURCE_TRANSFORM``'s shape, one argument
#: wider because the component is a compile-time axis here.
SOURCE_TRANSFORM: Optional[Any] = None


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm: str, component_index: int, count: int,
                sigma_kinds: Sequence[bool]):
    """Compile on first use, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING: the source is part
    of the memo key, and a probe mutates this family through
    :data:`SOURCE_TRANSFORM` -- a body memoized at first call would hand back
    the pre-mutation string forever.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be "
            "compiled; the predicates and the emitter need no device and still "
            "run")
    kinds = tuple(bool(kind) for kind in sigma_kinds)
    code = fused_polarization_pair_source(arm, component_index, count, kinds)
    if SOURCE_TRANSFORM is not None:
        code = SOURCE_TRANSFORM(arm, component_index, count, kinds, code)
    name = kernel_name(arm)
    key = compile_cache.kernel_cache_key(
        f"{name}:{component_index}:{count}:"
        + "".join("v" if kind else "u" for kind in kinds),
        False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def _driving_states(fields: Any, target: str) -> Tuple[Any, ...]:
    """The states driving one component, in ``fields.polarizations`` order.

    Transcribed from ``Fields.displacement_minus_polarization`` (fields.py:1096)
    -- the SAME order the chain subtracts in, which is also the order
    ``stepping.update_P`` advances in. One resolver so the two halves of the
    weld cannot disagree about the bank.
    """
    return tuple(state for state in tuple(getattr(fields, "polarizations", ()) or ())
                 if callable(getattr(state, "drives", None)) and state.drives(target))


def component_specs(fields: Any) -> Dict[str, Dict[str, Any]]:
    """Per component: the driving states, pole count and sigma kinds, RIGHT NOW.

    RESOLVED PER CALL AND NEVER CACHED: the rotation moves the buffers between
    component launches, which is the certified launchers' standing hazard.
    """
    specs: Dict[str, Dict[str, Any]] = {}
    for index, (target, _source, _axis) in enumerate(ELECTRIC_TERMS):
        states = _driving_states(fields, target)
        specs[target] = {
            "index": index,
            "states": states,
            "count": len(states),
            "kinds": tuple(bool(ade_sigma_is_volume(state, target))
                           for state in states),
        }
    return specs


def assert_disjoint_bindings(fields: Any, arm: str, target: str,
                             states: Sequence[Any]) -> int:
    """Every ``__restrict__`` pointer of ONE component launch distinct from every other.

    CHECKED PER LAUNCH, AFTER THE PREVIOUS COMPONENT'S ROTATION, for the
    certified ADE launcher's reason: the rotation moves three names between
    launches, so what a predicate checked is one permutation and this is the one
    being run. Base addresses only -- overlapping views with different bases pass
    unseen, the accepted limitation every launcher on this track shares.
    ``inv_eps`` is excluded exactly as the certified dispersive launcher excludes
    it: its parameter carries no ``restrict``.

    Returns the number of distinct allocations checked, so a caller can assert
    something was inspected.
    """
    named: List[Tuple[str, Any]] = [(target, getattr(fields, target)),
                                    ("D" + target[1],
                                     getattr(fields, "D" + target[1]))]
    if arm == "pml":
        named.append((f"f_w_{target}", getattr(fields, f"f_w_{target}")))
    for i, state in enumerate(states):
        named.append((f"P_{target}_{i}", state.P[target]))
        named.append((f"p_prev_{i}", state.P_prev[target]))
        named.append((f"p_out_{i}", state._scratch))
        if ade_sigma_is_volume(state, target):
            named.append((f"sigma_{i}", state.sigma[target]))
    seen: Dict[int, str] = {}
    for label, array in named:
        address = _base_address(array)
        if address is None:
            raise ValueError(
                f"{label} exposes no readable base address; the restrict "
                f"promises in the fused signature cannot be shown to hold")
        if address in seen:
            raise ValueError(
                f"{label} aliases {seen[address]} in the {target} launch; every "
                f"field and pole pointer in this signature is __restrict__, and "
                f"the weld's whole shape exists to bind each allocation once")
        seen[address] = label
    return len(seen)


def _drive_identity_problem(fields: Any, arm: str, target: str) -> Optional[str]:
    """Why the register hand-off would not be the certified drive; None if it is.

    ``stepping.update_P`` binds ``fields.drive_field(component)``. The fused
    kernel binds NOTHING for it -- the register carries the value -- so the
    array that method names must BE the array the E half writes: ``f_w_<c>``
    under the PML arm, the stored ``E<c>`` under the no-PML one. Checked by base
    address against the arrays this launch binds.
    """
    drive = getattr(fields, "drive_field", None)
    if not callable(drive):
        return ("fields.drive_field is not callable; the drive identity the "
                "register hand-off rests on cannot be checked")
    try:
        named = drive(target)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return f"fields.drive_field({target!r}) raised {type(exc).__name__}: {exc}"
    expected_name = f"f_w_{target}" if arm == "pml" else target
    expected = getattr(fields, expected_name, None)
    if expected is None:
        return f"fields.{expected_name} is not allocated"
    named_address = _base_address(named)
    expected_address = _base_address(expected)
    if named_address is None or named_address != expected_address:
        return (f"fields.drive_field({target!r}) is not fields.{expected_name}: "
                f"the update_P half would read a drive the update_E half of this "
                f"launch did not write, and the register hand-off would be a "
                f"different number, not a faster one")
    return None


def launch_fused_polarization_component(
        fields: Any, arm: str, target: str, states: Sequence[Any],
        tables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """ONE component's fused launch. The caller rotates afterwards.

    ``tables`` is required on the PML arm: the HALF-INTEGER kps/kms views from
    ``dispersive_kernels.dispersive_tables`` (stepping.py:1015), of which this
    launch binds only this component's axis pair.
    """
    spec = {t: i for i, (t, _s, _a) in enumerate(ELECTRIC_TERMS)}
    if target not in spec:
        raise ValueError(f"target must be one of {sorted(spec)}, got {target!r}")
    component_index = spec[target]
    axis = ELECTRIC_TERMS[component_index][2]
    count = len(states)
    kinds = tuple(bool(ade_sigma_is_volume(state, target)) for state in states)
    problem = _drive_identity_problem(fields, arm, target)
    if problem is not None:
        raise ValueError(problem)
    assert_disjoint_bindings(fields, arm, target, states)

    stored = getattr(fields, target)
    n_elem = int(stored.size)
    arguments: List[Any] = [stored]
    if arm == "pml":
        arguments.append(getattr(fields, f"f_w_{target}"))
    arguments.append(getattr(fields, "D" + target[1]))
    arguments.append(fields.inverse_epsilon_for(target))
    for state in states:
        arguments.append(state.P[target])
    for i, state in enumerate(states):
        scratch = state._scratch
        if int(scratch.size) != n_elem:
            raise ValueError(
                f"state {i}'s scratch holds {int(scratch.size)} cells and the "
                f"launch walks {n_elem}; the certified update_P guard this weld "
                f"drops was that size, and dropping it is licensed only where "
                f"the two are one number")
        coefficients = tuple(state._coefficients)
        if len(coefficients) != 3:
            raise ValueError(
                f"a polarization carries {len(coefficients)} recurrence "
                f"coefficients, not the (c_now, c_prev, c_drive) triple this "
                f"kernel bakes as scalars")
        arguments.append(scratch)
        arguments.append(state.P_prev[target])
        sigma = state.sigma[target]
        arguments.append(sigma if kinds[i] else np.float32(sigma))
        arguments.extend(np.float32(value) for value in coefficients)
    if arm == "pml":
        if tables is None:
            raise ValueError(
                "the PML arm indexes the half-integer kps/kms views and was "
                "handed no tables; dispersive_kernels.dispersive_tables(pml) is "
                "the certified resolver")
        nx, ny, nz = stored.shape
        arguments.extend([np.int32(nx), np.int32(ny), np.int32(nz),
                          tables[f"kps_{axis}"], tables[f"kms_{axis}"]])
    else:
        arguments.append(np.int32(n_elem))

    blocks = (n_elem + _FUSED_THREADS - 1) // _FUSED_THREADS
    _get_kernel(arm, component_index, count, kinds)(
        (blocks,), (_FUSED_THREADS,), tuple(arguments))
    return {"component": target, "poles": count, "kinds": list(kinds),
            "blocks": blocks, "threads": _FUSED_THREADS, "elements": n_elem}


def run_fused_polarization_pair(fields: Any, pml: Any, arm: str, *,
                                tables: Optional[Dict[str, Any]] = None
                                ) -> Dict[str, Any]:
    """Both of :data:`REPLACES` in THREE launches, host rotation between.

    THE ROTATION IS TRANSCRIBED, NOT REIMPLEMENTED, from
    ``ade_kernels.update_P_fused_pml_real`` (itself from dispersion.py:689-691)::

        state.P[component]      = scratch     # this step's result
        state.P_prev[component] = p           # what P held
        state._scratch          = p_prev      # the retired history

    run once per driving state after each component launch RETURNS, never
    before, so a raised launch leaves the state exactly as it found it. The
    component order is ``ELECTRIC_TERMS``' -- the same order the certified
    per-(state, component) launcher visits each state's components, which is
    what makes the buffer sequence identical: rotations of different states
    commute (each state owns its seven buffers), and within a state the x, y, z
    sequence is preserved.

    EVERY SLOT IS RE-READ AFTER THE PREVIOUS COMPONENT'S ROTATION, through
    :func:`component_specs` inside the loop -- the certified launchers' standing
    hazard, restated because this launcher is the one that would enjoy caching.
    """
    if arm not in KERNEL_KEYS:
        raise ValueError(f"arm must be one of {sorted(KERNEL_KEYS)}, got {arm!r}")
    if arm == "pml" and tables is None:
        if pml is None:
            raise ValueError(
                "supply either pml (and the HALF-INTEGER sub-lattice is chosen "
                "by dispersive_kernels.dispersive_tables, stepping.py:1015) or "
                "tables; passing neither leaves the kernel with no absorber "
                "profile to index")
        tables = certified_e.dispersive_tables(pml)

    launches = 0
    recurrences = 0
    per_component: List[Dict[str, Any]] = []
    for target, _source, _axis in ELECTRIC_TERMS:
        spec = component_specs(fields)[target]
        states = spec["states"]
        result = launch_fused_polarization_component(
            fields, arm, target, states, tables)
        launches += 1
        recurrences += len(states)
        per_component.append(result)
        for state in states:
            # dispersion.py:689-691, verbatim, once the launch API returned.
            p = state.P[target]
            p_prev = state.P_prev[target]
            scratch = state._scratch
            state.P[target] = scratch
            state.P_prev[target] = p
            state._scratch = p_prev
    return {"launched": True, "arm": arm, "launches": launches,
            "recurrences": recurrences, "replaces": REPLACES,
            "kernel": kernel_name(arm), "components": per_component}


# =============================================================================
# COVERAGE
# =============================================================================

def _seam_clauses(fields: Any, arm: str) -> Optional[str]:
    """The clauses this weld ADDS over its two halves; None when they hold.

    The halves' own predicates have already run when this is asked, so the
    polarization list, the buffers, the boundary kinds and the feature refusals
    are theirs. What is added is exactly what the weld changes: the drive
    identity behind the register hand-off, and the extent identity behind the
    dropped update_P guard.
    """
    stored = getattr(fields, "Ex", None)
    if stored is None:
        return "fields.Ex is not allocated"
    n_elem = int(getattr(stored, "size", 0))
    for component_index, (target, _source, _axis) in enumerate(ELECTRIC_TERMS):
        states = _driving_states(fields, target)
        if states:
            problem = _drive_identity_problem(fields, arm, target)
            if problem is not None:
                return problem
        try:
            certified_e.normalized_pole_counts(
                _counts_with(component_index, len(states)))
        except ValueError as exc:
            return str(exc)
        for index, state in enumerate(states):
            for label, array in ((f"P[{target!r}]", state.P.get(target)),
                                 (f"P_prev[{target!r}]",
                                  state.P_prev.get(target)),
                                 ("_scratch", getattr(state, "_scratch", None))):
                size = int(getattr(array, "size", -1)) if array is not None else -1
                if size != n_elem:
                    return (f"polarization {index} {label} holds {size} cells "
                            f"and the fused launch walks {n_elem}; this weld "
                            f"drops the certified update_P guard and may only "
                            f"do so where the two bounds are one number")
    return None


def covers_fused_polarization_pair(fields: Any, pml: Any, grid: Any,
                                   sources: Any = None) -> Tuple[bool, str]:
    """May three per-component launches span ``update_E`` -> ``update_P`` under a PML?

    A CONJUNCTION, AND NOTHING IS WEAKENED: a configuration either half's own
    certified predicate refuses is refused here with that half's reason,
    prefixed so a reader can tell which side said it. The off-diagonal cell is
    refused through the update_E half, and the per-component split is why that
    refusal is also THIS product's (module docstring).

    ``sources`` IS ACCEPTED AND INERT, and that is a driver fact rather than a
    leniency: nothing is injected between driver.py:3313 and :3315 on any run,
    so no source list -- declared, undeclared or empty -- can put a deposit in
    this seam. The B and D products must refuse an undeclared list because their
    seams have an injection slot; this seam has none to be ignorant about.
    """
    covered, reason = certified_e.covers_real_pml_dispersive_constitutive(
        fields, pml, grid)
    if not covered:
        return False, f"update_E half: {reason}"
    covered, reason = covers_real_pml_ade_update_p(fields, pml, grid)
    if not covered:
        return False, f"update_P half: {reason}"
    problem = _seam_clauses(fields, "pml")
    if problem is not None:
        return False, problem
    return True, "covered"


def covers_no_pml_fused_polarization_pair(fields: Any, pml: Any, grid: Any,
                                          sources: Any = None
                                          ) -> Tuple[bool, str]:
    """The no-absorber twin: arm 2's store welded to the layerless ADE.

    Disjoint from :func:`covers_fused_polarization_pair` on the absorber alone:
    both of this one's halves refuse an active layer by name and both of that
    one's require it.
    """
    covered, reason = certified_e.covers_no_pml_dispersive_constitutive(
        fields, pml, grid)
    if not covered:
        return False, f"update_E half: {reason}"
    covered, reason = certified_e.covers_no_pml_ade_update_p(fields, pml, grid)
    if not covered:
        return False, f"update_P half: {reason}"
    problem = _seam_clauses(fields, "no_pml")
    if problem is not None:
        return False, problem
    return True, "covered"
