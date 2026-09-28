"""The THREE-SLOT weld: ``step_D`` -> ``update_E`` -> ``update_P`` in one product.

THE FIRST PRODUCT ON ANY BACKEND THAT OWNS THREE DRIVER SLOTS, and the reason it
exists is a MEASURED COLLISION rather than an ambition. ``update_E`` is the SECOND
consult of the ``D_to_E`` seam and the FIRST of ``E_to_P`` -- the board's own seam
locator prints ``D_to_E`` ending at driver.py:3332 and ``E_to_P`` beginning at
driver.py:3332 -- so a two-slot D->E weld and a two-slot E->P weld COMPETE for one
slot and ``install_fused_pairs`` can give it to only one. On this corpus that trade
is exactly net zero:

    cuda_dispersive_fused_electric_pair          7 rows
    cuda_fused_polarization_pair                 the SAME 7
    cuda_no_pml_dispersive_fused_electric_pair   3 rows
    cuda_no_pml_fused_polarization_pair          the SAME 3

so ``fused_pairs._later_seam_claimant`` withholds ``update_E`` from the D->E side and
its own docstring names the resolution: "What would make the D->E weld strictly
better is a THREE-SLOT product spanning ``step_D`` -> ``update_E`` -> ``update_P``,
which serves BOTH seams instead of trading one for the other". This is that product,
for the two REAL dispersive cells -- 10 of the 14 collision rows, +10 of the +14
seam-instances (``D_to_E`` 10/10 and ``E_to_P`` 10/10 where the board carries 0/10
and 10/10 today).

=============================================================================
WHAT IS LIFTED, AND WHAT LITTLE IS NEW
=============================================================================

NEITHER HALF IS RE-DERIVED. The ``step_D`` -> ``update_E`` half is the SHIPPED,
RELEASED kernel, launched through its own launcher:

* :mod:`.dispersive_fused_electric_pair` (``cuda_dispersive_fused_electric_pair_
  2026-09-02``) on the PML arm, and
* :mod:`.no_pml_dispersive_fused_electric_pair`
  (``cuda_no_pml_dispersive_fused_electric_pair_2026-09-02``) on the no-absorber arm,
  through :mod:`.no_pml_three_slot_dispersive_weld`.

Not one byte of either is copied here; this module calls them.

THE ONLY NEW DEVICE TEXT IS ONE KERNEL FAMILY, :data:`KERNEL_NAME`, and it is a
STRICTLY SMALLER LIFT than the shipped E->P weld's. That weld
(:mod:`.fused_polarization_pair`) fused ``update_E`` INTO the per-component ADE
launch and paid for it with two risky edits; this one removes the ``update_E`` half
entirely, so BOTH of those edits disappear:

* ``float w = drive[idx];`` COMES BACK. There is no register to hand across a seam
  here -- the constitutive half ran in a different launch -- so the certified ADE
  drive load is restored UNEDITED, and with it the whole drive-identity argument the
  E->P weld had to make about a float32 word stored and reloaded.
* The ADE guard ``int idx = ...; if (idx >= n_elem) return;`` COMES BACK. No
  ``update_E`` body above has declared ``idx``, so the certified guard is spliced
  whole and ``n_elem`` is a parameter again, exactly as
  ``ade_kernels.update_P_pml_real`` declares it.

What survives from :mod:`.fused_polarization_pair` is only its per-pole renaming
machinery (:func:`~.fused_polarization_pair._pole_block`, imported rather than
copied) and its COMPONENT-MAJOR host loop. :data:`LIFT_EDITS` is three rows where
that module's is five.

=============================================================================
THE LAUNCH STRUCTURE: 1 + 3, NOT 4, AND NOT 1
=============================================================================

One ``run`` of this product is FOUR launches in TWO GROUPS with the deposit bracket
BETWEEN them::

    step_D consult    LeadingRepairPlan: deposit_repair.save(..., paths=REPAIR_PATHS)
                      LAUNCH 1 -- the shipped D->E kernel over the whole grid, on an
                      UNINJECTED displacement, carrying step_D (driver.py:3316),
                      fill_symmetry_bc_D (:3326), zero_metal_D (:3327),
                      fill_folded_far_ghosts_D (:3330) and update_E (:3332)

    (the driver)      electric inject (:3319/:3322), then the three passes again

    update_E consult  TrailingRepairPlan: deposit_repair.apply(...). NO LAUNCH OF
                      OURS. P is still P^n here, and that is load-bearing.

    update_P consult  LAUNCHES 2, 3, 4 -- one per DRIVEN component, each carrying
                      EVERY driving state's ADE recurrence for that component,
                      reading the drive from GLOBAL. Between them, on the host, per
                      state, only after the launch RETURNS:
                          P[c] <- scratch ; P_prev[c] <- p ; _scratch <- p_prev

THE ONE THING THAT DOES NOT SURVIVE IS THE MEGA-LAUNCH, and it is refused BY
MEASUREMENT rather than by taste. A single kernel spanning ``step_D`` ->
``update_E`` -> ``update_P`` computes the ADE recurrence from a PRE-INJECTION drive
AND advances P before the repair reads it.
``parity/meep_gpu/probe_cuda_three_slot_weld.py``'s ``p_inside_the_launch`` ordering
diverges on 6 of its 7 configurations (36, 60, 26, 7, 14 and 12 differing words over
6 complete driver steps), null only at arity ``(0, 0, 0)`` with the reason recorded.

WHY THE REPAIR MUST SIT BETWEEN THE TWO GROUPS, and why this is the first product on
this track whose OWN host work sits between its OWN two device groups.
``deposit_repair.apply`` recomputes the constitutive product through
``Fields.displacement_minus_polarization`` on both of its paths
(deposit_repair.py:781-782 and :922-923), and that method subtracts ``state.P[c]``
AS IT STANDS (fields.py:1096-1105). So the repair is correct only while P is still
P^n. The probe's ``repair_after_p`` ordering -- the polarization advance moved one
group earlier -- diverges on the same 6 of 7 configurations. Both wrong orders
compute, converge, and are wrong only at the deposit cells, which is why they are
armed rather than reasoned about.

=============================================================================
THE ROTATION IS NOT TOUCHED
=============================================================================

It stays exactly where ``ade_kernels.update_P_fused_pml_real`` and
``fused_polarization_pair.run_fused_polarization_pair`` put it: on the HOST, once
per driving state, after each per-component launch RETURNS, with every slot (``P``,
``P_prev``, ``_scratch``, ``sigma`` and the drive) RE-READ inside the loop and never
cached. The only change is that the drive now arrives from global memory instead of
a register. ``stale_pole_pointers``, ``rotate_before_launch`` and
``shared_scratch_all_components`` are armed in the probe and diverge by 27720,
41580 and 41580 words on the six-pole corpus shape.

THE COMPONENT-MAJOR ORDER IS A MEASUREMENT, NOT A PREFERENCE. The certified ADE
launcher is STATE-MAJOR (``for state: for component in state.driven()``) and this
one is component-major; the probe records ``state_order_reversed`` and
``component_order_reversed`` as PREDICTED NULLS confirmed on all 7 configurations,
which is the measurement that every driving state of one component may be fused into
one launch and that the three component launches may be issued in any order.

=============================================================================
TWO ARMS, DISJOINT ON THE ABSORBER, IN TWO MODULES
=============================================================================

``cuda_three_slot_dispersive_weld`` (this module) welds the PML D->E product to the
PML-ADE recurrence; ``cuda_three_slot_no_pml_dispersive_weld``
(:mod:`.no_pml_three_slot_dispersive_weld`) welds the no-absorber one. They are TWO
MODULES rather than two families in one because they declare DIFFERENT
:data:`REPAIR_PATHS` -- ``update_E`` under an inactive layer is a pure overwrite
(stepping.py:1019-1022) and the split-field repair inverts a recurrence that does not
run there -- and ``fused_pairs._repair_paths_of`` reads that declaration off the
MODULE. One module could not carry two answers.

THE DEVICE TEXT IS ARM-INDEPENDENT and that is a fact rather than a shortcut: the
``update_E`` half is gone, so what is left is the certified ADE recurrence, which
``covers_real_pml_ade_update_p`` and ``covers_no_pml_ade_update_p`` admit through the
SAME two kernels (``ade_kernels``' volume and uniform sigma forms). The absorber
reaches the polarization through the DRIVE and nowhere else (stepping.py:1424), and
the drive is a bound argument. So :func:`three_slot_polarization_source` takes no
arm, and the no-absorber module imports this one's emitter whole.

=============================================================================
HOW THIS PRODUCT AND THE TWO IT SUPERSEDES STAY OUT OF EACH OTHER'S WAY
=============================================================================

Its predicate is the CONJUNCTION of both shipped predicates, so its admission set is
a SUBSET of each. On ``step_D`` that would make it and
``cuda_dispersive_fused_electric_pair`` two admitters on one seam, which
``install_fused_pairs`` leaves UNFUSED naming both -- a coverage loss. The resolution
is in the composer and not in either predicate:
``fused_pairs._superseded_by_a_longer_span`` refuses the SHORTER product by name
where a longer span admits the same run. That is slot arbitration of exactly the kind
``_later_seam_claimant`` already performs, and it is strictly non-lossy: this
product's admission implies that one's, so the shorter one installs unchanged
everywhere this one is refused.

On ``update_E``, ``cuda_fused_polarization_pair`` is failed closed OUT by the
EXISTING mechanism and no new guard: by the time the seam loop reaches ``update_E``
this product has already written ``selected['update_E']``, and
``_pair_may_absorb`` refuses BY NAME any pair whose slot went to a different arm.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` composes ``triton_kernels.plan_step`` and
reaches no hand-CUDA product. This module ships an emitter, two predicates and two
launchers; ``fused_pairs`` can plan it opt-in (``arms.plan_step(..., fuse=True)``),
and no shipped dispatch path launches it.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # a host with no CuPy: the emitter and both predicates still run
    import cupy as cp
except ImportError:
    cp = None

# THE E->P WELD, IMPORTED FOR ITS CERTIFIED LIFT MACHINERY. ``_pole_block``,
# ``_verified_ade_bodies`` and ``_ade_prologue`` read ``ade_kernels``' device strings
# off the syntax tree and rename them under checked substitutions; they are IMPORTED
# rather than copied so a drift in the certified ADE text raises in one place.
try:
    from . import compile_cache
    from . import fused_polarization_pair as ep
    from .coverage import _base_address, ade_sigma_is_volume
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
    ep = _load("fused_polarization_pair")
    _coverage = _load("coverage")
    _base_address = _coverage._base_address
    ade_sigma_is_volume = _coverage.ade_sigma_is_volume

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
# THE PARTITION -- MOVED BY THE GATE, 2026-09-03
# =============================================================================
#
# Plain assignments with NO type annotation: the partition readers walk the syntax
# tree without importing the module, and an annotated assignment is an
# ``ast.AnnAssign`` those readers do not match.

#: RELEASED 2026-09-03 under BOTH float32 subnormal policies. A fused product's
#: bit-identity is a claim about the WELD, and no verdict on either half establishes
#: it: the D->E half is certified (``cuda_dispersive_fused_electric_pair_2026-09-02``,
#: ``cuda_no_pml_dispersive_fused_electric_pair_2026-09-02``) and the ADE recurrence
#: is certified (``ade_2026-08-19``), and NEITHER is a verdict about a composition
#: whose own host repair sits between its two device groups. What moved this kernel
#: across the partition is the run, not an argument: ``gate_cuda_three_slot_weld.py``
#: on one RTX A6000, four legs (both policies x both arms), 18 of 18 PML and 14 of 14
#: no-PML cases bit-identical over 60 COMPLETE driver steps at seven pole arities
#: including the corpus ones, every stored volume AND every P / P_prev buffer
#: compared as uint32 words, the launch-structure fixtures identical to FOUR
#: compositions from one seed (the array path, the certified singles, the D->E weld
#: plus the certified ADE, step_D plus the E->P weld) with the launch count measured
#: by two instruments, the deposit legs identical with the unbracketed and
#: P-before-repair controls diverging and the wrong REPAIR_PATHS refused by name, 9 +
#: 9 and 8 + 8 mutation arms caught with every null predicted and none unarmed, and
#: the block written into ``certification.json`` as
#: ``cuda_three_slot_dispersive_weld_2026-09-03``. The name is owned HERE for both
#: arms: ``no_pml_three_slot_dispersive_weld`` launches this same text under
#: different bindings and declares no kernel of its own.
CERTIFIED_KERNELS = (
    "three_slot_polarization_real",
)
#: EMPTY since 2026-09-03; what emptied it was the run above.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its launches with the deposit repair? TRUE, AND IT IS
#: THE WHOLE PRODUCT. Every one of the ten rows both cells own carries an ELECTRIC
#: deposit inside the ``step_D`` -> ``update_E`` span (the stamped census's
#: ``source_field_types``), so without the bracket this weld would serve ZERO. The
#: flag and the wiring change together or not at all (deposit_repair.py:921-926):
#: ``fused_pairs._install_fused_triple`` is what installs the bracket, and this
#: module's predicate passes the flag to ``deposit_repair.seam_source_reasons``
#: through the D->E half it conjoins.
CARRIES_DEPOSIT_REPAIR = True

#: WHICH repair the bracket installs. The SPLIT-FIELD one, inherited from the D->E
#: half this arm welds rather than re-decided: an ACTIVE layer makes ``update_E``
#: the split-field accumulation (stepping.py:2130-2134). The no-absorber arm's module
#: declares the plain one. ``deposit_repair.repairable`` refuses BY NAME any
#: configuration whose recurrence is not among the declared paths, so a wrong
#: declaration is a refusal and never a mis-repair -- measured 7 of 7 in the probe's
#: ``wrong_repair_path`` row.
REPAIR_PATHS: Tuple[str, ...] = (_deposit_repair.SPLIT_FIELD_PATH,)

FAMILY = "cuda_three_slot_dispersive_weld"

#: The ``extern "C"`` symbol the NEW kernel is emitted under. FIXED across the
#: component, the pole count and the sigma forms -- those vary the SOURCE, not the
#: name, which is ``dispersive_kernels``' own convention for a compile-time axis --
#: and fixed across the ARM, because with the ``update_E`` half removed the two arms'
#: device text is character-identical (module docstring).
KERNEL_NAME = "three_slot_polarization_real"

#: WHICH MODULE OWNS THE EMITTED TEXT AND ITS GATE DOORS -- this one. Both arms
#: launch the SAME device text (module docstring), so :data:`SOURCE_TRANSFORM`,
#: :func:`_get_kernel` and :func:`_clear_kernel_cache` are module-level state that
#: exists in exactly ONE place; a gate that assigned a transform to the no-absorber
#: module would set an attribute nothing reads and score every device mutation as
#: silently unarmed. Declared rather than inferred so the gate reads the answer
#: instead of hardcoding which arm is the twin.
KERNEL_OWNER_MODULE = "three_slot_dispersive_weld"

#: The driver passes this product performs, in driver order (driver.py:3316, :3326,
#: :3327, :3330, :3332, :3334). SIX, which is every pass in the span: the first five
#: are the D->E half's own ``REPLACES`` and the sixth is the polarization advance.
#: Declared rather than inferred from the three slots -- ``zero_metal_D`` and the two
#: fills are driver passes no slot names.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E", "update_P")

#: The slot the composer holds this on (the span's first).
SLOT = "step_D"

#: ALL THREE slots this product owns, in driver order. Read by
#: ``fused_pairs._install_fused_triple`` and by ``_superseded_by_a_longer_span``,
#: which is what makes "this span strictly contains that one" a lookup rather than a
#: judgement.
SLOTS: Tuple[str, str, str] = ("step_D", "update_E", "update_P")

#: The three E components with their sources and axes, re-read from the certified
#: family through the E->P weld rather than retyped.
ELECTRIC_TERMS = ep.ELECTRIC_TERMS

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate can assert the list rather than a docstring.
#:
#: THREE ROWS, WHERE :data:`fused_polarization_pair.LIFT_EDITS` HAS FIVE. The two
#: that are gone are the two that carried risk -- the drive register hand-off and the
#: dropped ADE guard -- and they are gone because the ``update_E`` half is not in this
#: kernel. Every row below is a NAME or the fusion itself; no arithmetic is edited.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    const float* __restrict__ p_now,",
     "became": "    const float* __restrict__ P_<c>_<i>,",
     "why": "one launch now carries every driving state's recurrence for one "
            "component, so the certified kernel's single p_now parameter becomes one "
            "per pole and each carries the pole's index. The NAME is the E->P weld's "
            "(fused_polarization_pair._pole_block is imported, not copied, so the "
            "two families cannot rename differently); the declaration keeps the "
            "certified __restrict__, which is sound here for a reason it was NOT "
            "sound there: the update_E half that also read state.P[c] is in a "
            "different launch, so this signature binds the allocation exactly once "
            "and nothing else in the kernel reads it"},
    {"line": "    float p = p_now[idx]; (and q, s; c_now, c_prev, c_drive)",
     "became": "    float p_<i> = P_<c>_<i>[idx]; (suffixed per pole)",
     "why": "the same per-pole suffixing, applied by the same imported function. "
            "Names only: the loads, the grouping "
            "((p * c_now) + (c_prev * q)) + (c_drive * (s * w)) and the store are the "
            "certified line under checked substitutions that raise if the certified "
            "string drifts. THE DRIVE LOAD IS NOT AMONG THEM -- float w = drive[idx] "
            "is emitted ONCE PER COMPONENT, verbatim, because every pole of a "
            "component reads the same array and nothing in this launch writes it"},
    {"line": "(the certified update_P, one launch per (state, component))",
     "became": "(one launch per COMPONENT, every driving state fused into it, host "
               "rotation between)",
     "why": "THE FUSION, and the only structural edit. The recurrence for (state, c) "
            "reads that state's own P[c]/P_prev[c] plus the shared drive and writes "
            "that state's own _scratch -- disjoint across states -- so fusing all "
            "driving states of one component into one launch reorders nothing that "
            "shares data. The certified launcher is STATE-major and this one is "
            "COMPONENT-major; both reorderings are PREDICTED NULLS confirmed on all "
            "7 of the probe's configurations (state_order_reversed, "
            "component_order_reversed). The host rotation is unmoved: still once per "
            "driving state, still only after the launch returns"},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "KERNEL_OWNER_MODULE", "LIFT_EDITS", "REPAIR_PATHS", "REPLACES", "SLOT",
    "SLOTS", "UNCERTIFIED_KERNELS", "assert_disjoint_bindings", "component_specs",
    "covers_three_slot_dispersive_weld", "device_sources",
    "launch_three_slot_polarization_component", "run_three_slot_polarization",
    "three_slot_polarization_source",
]


# =============================================================================
# THE EMITTER -- the certified ADE recurrence, N poles to a launch
# =============================================================================

def _signature(component_index: int, count: int,
               sigma_kinds: Sequence[bool]) -> str:
    """The fused signature for one component and one specialization.

    THE ONLY HAND-WRITTEN DEVICE TEXT IN THIS MODULE, and it is a signature: no
    arithmetic lives here. The parameter ORDER is the certified ADE kernel's --
    ``p_out, p_now, p_prev, sigma, drive, c_now, c_prev, c_drive, n_elem`` -- with
    the drive hoisted ahead of the per-pole groups because ONE launch now carries
    several poles that all read it, and ``n_elem`` kept last exactly as
    ``ade_kernels`` declares it.
    """
    target = ELECTRIC_TERMS[component_index][0]
    lines: List[str] = [f'extern "C" __global__ void {KERNEL_NAME}(']
    # THE DRIVE, BOUND ONCE. ``Fields.drive_field(c)`` -- f_w_<c> under an active
    # layer, the stored E<c> without one (fields.py:1140-1163) -- and every pole of
    # this component reads it.
    lines.append("    const float* __restrict__ drive,")
    for i in range(count):
        lines.append(f"    const float* __restrict__ P_{target}_{i},")
    for i in range(count):
        sigma = (f"const float* __restrict__ sigma_{i}," if sigma_kinds[i]
                 else f"float sigma_{i},")
        lines.append(f"    float* __restrict__ p_out_{i}, "
                     f"const float* __restrict__ p_prev_{i}, {sigma}")
        lines.append(f"    float c_now_{i}, float c_prev_{i}, "
                     f"float c_drive_{i},")
    lines.append("    int n_elem")
    lines.append(") {")
    return "\n".join(lines) + "\n"


def three_slot_polarization_source(component_index: int, count: int,
                                   sigma_kinds: Sequence[bool]) -> str:
    """The whole kernel for ONE component at one specialization.

    ``sigma_kinds[i]`` is ``coverage.ade_sigma_is_volume``'s answer for the i-th
    driving state -- a compile-time axis exactly as it is in the certified ADE
    family, here one flag per pole because one launch carries every driving state.

    A ZERO-POLE COMPONENT EMITS NOTHING AND IS NEVER LAUNCHED: the caller skips it,
    the way ``ade_kernels.update_P_fused_pml_real`` skips a state that drives
    nothing. Asking for one is a refusal rather than an empty kernel, because an
    empty kernel is a launch that did no work while the counter says one ran.
    """
    if not 0 <= int(component_index) <= 2:
        raise ValueError(f"component_index must be 0..2, got {component_index!r}")
    count = int(count)
    if count <= 0:
        raise ValueError(
            "a component with no driving state has no recurrence to emit; the "
            "launcher skips it rather than issuing a kernel that writes nothing")
    # The certified cap is a MEASURED ceiling; reuse the dispersive family's refusal
    # verbatim rather than restating a number.
    ep.certified_e.normalized_pole_counts(ep._counts_with(component_index, count))
    kinds = tuple(bool(kind) for kind in sigma_kinds)
    if len(kinds) != count:
        raise ValueError(
            f"sigma_kinds carries {len(kinds)} flags for {count} poles; one "
            f"compile-time sigma form per driving state, in registration order")

    bodies = ep._verified_ade_bodies()
    # THE CERTIFIED GUARD, RESTORED WHOLE. The E->P weld had to drop it (an update_E
    # body above had already declared ``idx``); nothing here has, so this is
    # ade_kernels' own two lines and ``n_elem`` is a real parameter again.
    guard = ep._ADE_GUARD
    drive = (
        "    // THE DRIVE, LOADED FROM GLOBAL EXACTLY AS THE CERTIFIED KERNEL LOADS\n"
        "    // IT. stepping.update_P binds fields.drive_field(component) as w; this\n"
        "    // launch binds the same array. One load per component rather than one\n"
        "    // per pole: every pole of a component reads the same array, and nothing\n"
        "    // in this launch writes it.\n"
        "    float w = drive[idx];\n")
    poles = "".join(
        "\n"
        f"    // pole {i}: dispersion.PolarizationState.update, the certified\n"
        f"    // recurrence under this launch's per-pole names.\n"
        + ep._pole_block(component_index, i, kinds[i], bodies)
        for i in range(count))
    source = "".join((
        ep._ade_prologue(),
        _signature(component_index, count, kinds),
        guard,
        drive,
        poles,
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


def _digest_specs() -> Tuple[Tuple[int, int, Tuple[bool, ...]], ...]:
    """The specializations a digest covers: every component, the corpus-swept pole
    counts, both sigma forms and a mixed one at every count that can hold one."""
    specs: List[Tuple[int, int, Tuple[bool, ...]]] = []
    for component in (0, 1, 2):
        for count in (1, 2, 3, 5, 6):
            kind_sets = {tuple([True] * count), tuple([False] * count),
                         tuple(bool(i % 2) for i in range(count))}
            for kinds in sorted(kind_sets):
                specs.append((component, count, kinds))
    return tuple(specs)


def device_sources() -> Dict[str, str]:
    """Every source a digest pins, keyed ``component/count/kinds``."""
    return {
        f"{ELECTRIC_TERMS[component][0]}/{count}/"
        + "".join("v" if kind else "u" for kind in kinds):
        three_slot_polarization_source(component, count, kinds)
        for component, count, kinds in _digest_specs()}


# =============================================================================
# COMPILATION AND LAUNCH
# =============================================================================

#: NVRTC compile options -- CORRECTNESS, not performance, and identical to every
#: certified half's. Spelled here rather than imported so loading this file by path
#: cannot pick up a different tuple than the one a gate compiled.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one cell per lane -- the certified ADE geometry.
_FUSED_THREADS = 256

#: THE GATE'S DOOR into the emitted text. ``None`` in every shipped path; a
#: bit-identity probe assigns a ``(component_index, count, kinds, source) -> source``
#: callable to plant a defect in a body that does not exist until it is asked for.
SOURCE_TRANSFORM: Optional[Any] = None


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(component_index: int, count: int, sigma_kinds: Sequence[bool]):
    """Compile on first use, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING: it is part of the memo
    key, and a probe mutates this family through :data:`SOURCE_TRANSFORM` -- a body
    memoized at first call would hand back the pre-mutation string forever.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicates and the emitter need no device and still run")
    kinds = tuple(bool(kind) for kind in sigma_kinds)
    code = three_slot_polarization_source(component_index, count, kinds)
    if SOURCE_TRANSFORM is not None:
        code = SOURCE_TRANSFORM(component_index, count, kinds, code)
    key = compile_cache.kernel_cache_key(
        f"{KERNEL_NAME}:{component_index}:{count}:"
        + "".join("v" if kind else "u" for kind in kinds),
        False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def component_specs(fields: Any) -> Dict[str, Dict[str, Any]]:
    """Per component: the driving states, pole count and sigma kinds, RIGHT NOW.

    THE E->P WELD'S OWN RESOLVER, imported rather than copied, so the two products
    cannot disagree about the bank or its order. RESOLVED PER CALL AND NEVER CACHED:
    the rotation moves the buffers between component launches, which is the certified
    launchers' standing hazard.
    """
    return ep.component_specs(fields)


def assert_disjoint_bindings(fields: Any, arm: str, target: str,
                             states: Sequence[Any]) -> int:
    """Every ``__restrict__`` pointer of ONE component launch distinct from every other.

    CHECKED PER LAUNCH, AFTER THE PREVIOUS COMPONENT'S ROTATION, for the certified
    ADE launcher's reason: the rotation moves three names between launches, so what a
    predicate checked is one permutation and this is the one being run.

    THE GROUP IS SMALLER THAN THE E->P WELD'S BY EXACTLY THE update_E HALF'S
    BINDINGS: no ``E<c>``, no ``D<c>``, no ``f_w_<c>`` as an output and no
    ``inv_eps``, because this kernel is the recurrence alone. What is left is the
    DRIVE -- which IS ``f_w_<c>`` on this arm and the stored ``E<c>`` on the other,
    and is ``const __restrict__`` here -- and the four per-pole pointers.

    Base addresses only: overlapping views with different bases pass unseen, the
    accepted limitation every launcher on this track shares. Returns the number of
    distinct allocations checked, so a caller can assert something was inspected.
    """
    if arm not in ("pml", "no_pml"):
        raise ValueError(f"arm must be 'pml' or 'no_pml', got {arm!r}")
    named: List[Tuple[str, Any]] = [("drive", _drive_array(fields, arm, target))]
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
                f"{label} exposes no readable base address; the restrict promises in "
                f"the fused signature cannot be shown to hold")
        if address in seen:
            raise ValueError(
                f"{label} aliases {seen[address]} in the {target} launch; every "
                f"pointer in this signature is __restrict__, and binding one "
                f"allocation twice is undefined behaviour NVRTC miscompiles without "
                f"a diagnostic")
        seen[address] = label
    return len(seen)


def _drive_array(fields: Any, arm: str, target: str) -> Any:
    """``stepping.update_P``'s own drive for one component, resolved the same way.

    ``Fields.drive_field`` is asked rather than a field reached for directly, exactly
    as ``ade_kernels.update_P_fused_pml_real`` asks it (stepping.py:1428). The arm
    only says which array that method MUST hand back, which
    :func:`_drive_identity_problem` is what checks.
    """
    drive = getattr(fields, "drive_field", None)
    if not callable(drive):
        raise ValueError(
            "fields.drive_field is not callable; this launcher never reaches for a "
            "field itself and has nothing to bind as the recurrence's w")
    return drive(target)


def _drive_identity_problem(fields: Any, arm: str, target: str) -> Optional[str]:
    """Why the bound drive would not be the certified one; ``None`` if it is.

    Delegated to the E->P weld's own clause, which asks the same question of the same
    two arrays. NOT LOAD-BEARING HERE THE WAY IT IS THERE, and that is worth saying:
    that product hands the drive across a seam in a REGISTER, so the identity is what
    makes its arithmetic the certified arithmetic. This one binds
    ``fields.drive_field(c)`` as a pointer, so the certified load happens whatever
    that method returns. The clause is kept because it is inherited whole with the
    predicate and because a run where it failed would be one where ``update_E`` and
    ``update_P`` disagreed about the layer.
    """
    return ep._drive_identity_problem(fields, arm, target)


def launch_three_slot_polarization_component(
        fields: Any, arm: str, target: str, states: Sequence[Any],
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """ONE component's recurrence launch. The caller rotates afterwards.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional: a gate compiles a deliberately
    broken copy of the shipped source and hands it here. A launcher that could not be
    handed its own kernel could not arm a single mutation.
    """
    spec = {t: i for i, (t, _s, _a) in enumerate(ELECTRIC_TERMS)}
    if target not in spec:
        raise ValueError(f"target must be one of {sorted(spec)}, got {target!r}")
    component_index = spec[target]
    count = len(states)
    if not count:
        raise ValueError(
            f"{target} is driven by no state; the caller skips such a component "
            f"rather than issuing a launch that writes nothing")
    kinds = tuple(bool(ade_sigma_is_volume(state, target)) for state in states)
    problem = _drive_identity_problem(fields, arm, target)
    if problem is not None:
        raise ValueError(problem)
    assert_disjoint_bindings(fields, arm, target, states)

    drive = _drive_array(fields, arm, target)
    n_elem = int(drive.size)
    arguments: List[Any] = [drive]
    for state in states:
        arguments.append(state.P[target])
    for i, state in enumerate(states):
        scratch = state._scratch
        if int(scratch.size) != n_elem:
            raise ValueError(
                f"state {i}'s scratch holds {int(scratch.size)} cells and the launch "
                f"walks {n_elem}; the certified guard this kernel keeps is that "
                f"bound, and a mismatch would leave cells unwritten")
        coefficients = tuple(state._coefficients)
        if len(coefficients) != 3:
            raise ValueError(
                f"a polarization carries {len(coefficients)} recurrence "
                f"coefficients, not the (c_now, c_prev, c_drive) triple this kernel "
                f"bakes as scalars")
        arguments.append(scratch)
        arguments.append(state.P_prev[target])
        sigma = state.sigma[target]
        arguments.append(sigma if kinds[i] else np.float32(sigma))
        arguments.extend(np.float32(value) for value in coefficients)
    arguments.append(np.int32(n_elem))

    blocks = (n_elem + _FUSED_THREADS - 1) // _FUSED_THREADS
    (kernel or _get_kernel(component_index, count, kinds))(
        (blocks,), (_FUSED_THREADS,), tuple(arguments))
    return {"component": target, "poles": count, "kinds": list(kinds),
            "blocks": blocks, "threads": _FUSED_THREADS, "elements": n_elem}


def run_three_slot_polarization(fields: Any, arm: str,
                                kernel: Optional[Any] = None) -> Dict[str, Any]:
    """THE THIRD SLOT: one launch per DRIVEN component, host rotation between.

    THE ROTATION IS TRANSCRIBED, NOT REIMPLEMENTED, from
    ``ade_kernels.update_P_fused_pml_real`` (itself from dispersion.py:689-691)::

        state.P[component]      = scratch     # this step's result
        state.P_prev[component] = p           # what P held
        state._scratch          = p_prev      # the retired history

    run once per driving state after each component launch RETURNS, never before, so
    a raised launch leaves the state exactly as it found it.

    EVERY SLOT IS RE-READ AFTER THE PREVIOUS COMPONENT'S ROTATION, through
    :func:`component_specs` inside the loop.

    A COMPONENT WITH NO DRIVING STATE IS SKIPPED RATHER THAN LAUNCHED EMPTY, exactly
    as the certified launcher skips a state that drives nothing -- so the launch
    count is the number of DRIVEN components, which is what the arity says and what
    the gate counts.
    """
    if arm not in ("pml", "no_pml"):
        raise ValueError(f"arm must be 'pml' or 'no_pml', got {arm!r}")
    launches = 0
    recurrences = 0
    per_component: List[Dict[str, Any]] = []
    for target, _source, _axis in ELECTRIC_TERMS:
        states = component_specs(fields)[target]["states"]
        if not states:
            continue
        per_component.append(launch_three_slot_polarization_component(
            fields, arm, target, states, kernel))
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
    return {"launched": True, "arm": arm, "launches": launches,
            "recurrences": recurrences, "replaces": ("update_P",),
            "kernel": KERNEL_NAME, "components": per_component}


# =============================================================================
# COVERAGE
# =============================================================================

def covers_three_slot_dispersive_weld(fields: Any, pml: Any, grid: Any,
                                      sources: Any = None) -> Tuple[bool, str]:
    """May ONE product span ``step_D`` -> ``update_E`` -> ``update_P`` under a PML?

    A CONJUNCTION OF TWO SHIPPED PREDICATES, AND NOTHING IS WEAKENED. The D->E half
    is ``dispersive_fused_electric_pair.covers_dispersive_fused_electric_pair``
    whole -- which is where the SOURCE question is asked, through
    ``deposit_repair.seam_source_reasons`` with that module's own
    ``CARRIES_DEPOSIT_REPAIR`` and its split-field repair, the same declaration this
    module makes. The E->P half is
    ``fused_polarization_pair.covers_fused_polarization_pair`` whole. Each refusal is
    prefixed so a reader can tell which side said it.

    THE ADMISSION SET IS THEREFORE A SUBSET OF EACH HALF'S, which is exactly what
    ``fused_pairs._superseded_by_a_longer_span`` needs: where this product admits,
    the two-slot ones do too, so refusing the shorter span costs nothing and gains
    the second seam.

    NOTHING IS ADDED, AND THAT IS A MEASUREMENT RATHER THAN AN OMISSION. A weld's
    own clauses are the facts its composition needs that neither half asks; this one
    was written with a coefficient-triple clause and the clause was REMOVED when the
    two halves turned out to ask it already -- ``coverage._ade_coefficient_problem``
    is reached from ``covers_real_pml_ade_update_p`` (coverage.py:2142) and from its
    no-absorber twin (dispersive_kernels.py:1272). Shipping it anyway would have put
    a refusal in this file that claims to be load-bearing and is not.

    THE ONE FACT THIS PRODUCT NEEDS THAT NO PREDICATE CAN EXPRESS is the ORDER of its
    two device groups relative to the repair: ``update_P`` must run AFTER
    ``deposit_repair.apply``, because that function subtracts ``state.P[c]`` as it
    stands. That is a property of the WIRING, not of the configuration -- there is no
    ``fields`` on which it is true or false -- so it is enforced in
    ``fused_pairs._install_fused_triple``, pinned by
    ``test_three_slot_dispersive_weld.py`` and measured by the gate's deposit legs.
    """
    from . import dispersive_fused_electric_pair as de  # noqa: PLC0415

    covered, reason = de.covers_dispersive_fused_electric_pair(
        fields, pml, grid, sources)
    if not covered:
        return False, f"step_D->update_E half: {reason}"
    covered, reason = ep.covers_fused_polarization_pair(fields, pml, grid, sources)
    if not covered:
        return False, f"update_E->update_P half: {reason}"
    return True, "covered"
