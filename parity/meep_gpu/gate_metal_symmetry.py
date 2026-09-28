"""Byte gate for the Metal folded (mirror-symmetry) real-field family.

THE CLAIM, and the only one: **byte-identity to ``stepping.py`` on this host,
subject to a declared and CHECKED subnormal-free precondition, and that precondition
is a WINDOW rather than a scalar.** Not a stated tolerance. Leg 7 is the check and it
is required to FIRE — a precondition never demonstrated to fire is decorative.

THE ORACLE IS IN PROCESS. ``stepping.step_B`` / ``step_D``, ``fill_symmetry_bc_*``,
``fill_folded_far_ghosts_*``, ``update_H`` / ``update_E`` and the Metal launches run
in ONE process against ONE seed, on real ``Grid``/``Fields``/``PML`` objects built by
``metal_composition_matrix.folded`` — the SAME builder the composition sweep uses, so
the configuration this gate certifies is the configuration that sweep proved disjoint.
There is no in-gate transcription of the arithmetic and there deliberately is none:
this family emits no new arithmetic. Its curl is the CERTIFIED emitters' output fed
reduced codes plus one mask block, and its constitutive slot is the CERTIFIED kernel
under a restated predicate. What is new is the FOLD — a ghost rule, a mask, two fill
passes and a plane written by one sub-step and read by the next — and a transcription
would only give the fold a second chance to be wrong the same way twice.

WHAT IS COMPARED: uint32 word equality on float32 storage, NEVER ``allclose``. The
compared set is the sub-step's TARGETS and its PML AUXILIARIES (``fu_*``) for the
curl, the family's three targets for the fill, and targets plus ``f_w_*`` for the
constitutive pair — and in the whole-step leg the FULL stored inventory after every
complete driver step, because ``fu_*`` and ``f_w_*`` are STATE and a kernel right for
one launch and wrong forever after is identical in a single-launch gate.

THE LEGS:

  0  execution      proof the bytes came off the GPU: mirror residency, the entry
                    point's type, the launch counter, and an UNLAUNCHED control
  1  curl           the folded curl vs ``step_B``/``step_D``, SUB-STEP granularity,
                    which is the only granularity the two ownership masks are
                    visible at (see below)
  2  fill           the two ghost passes vs ``fill_symmetry_bc_*`` and
                    ``fill_folded_far_ghosts_*``, near and far separately AND in
                    the driver's order
  3  constitutive   the re-admitted CERTIFIED body vs ``update_H``/``update_E``
  4  whole_step     the driver's five passes per half, per COMPLETE STEP over a
                    stated budget, reporting the FIRST DIVERGENT STEP, with every
                    slot's launch counter asserted
  5  reduction      the folded curl on an UNFOLDED grid vs the CERTIFIED
                    ``pml_curl_step`` — the structural claim, measured
  6  signed_zero    a +-0 lattice through every fill plane, census-floored. THE
                    FILL'S PARITY IS A SIGN-BIT OPERATION, so this is the seeding
                    that separates the shipped spelling from the refuted one
  7  precondition   THE WINDOW: three driven runs censused at EVERY step, each
                    recording first_fired / last_fired; plus the scaled ladder that
                    measures the fill/curl SPLIT
  8  mutations      armed, launch-counted, three-valued, and FOLD-SPECIFIC
  9  composition    the six slots, the foreign-arm check, and the named refusal

WHY BOTH A SUB-STEP LEG AND A WHOLE-STEP LEG, and neither is redundant:

* THE TWO OWNERSHIP MASKS ARE INVISIBLE AT WHOLE-STEP GRANULARITY. The driver's fill
  passes overwrite exactly the planes the masks protect, so a kernel carrying NEITHER
  mask is bytewise-identical after a complete step. Only leg 1 can fail on a mask —
  measured here: ``drop_top_plane_mask`` moves 384-1,080 words at sub-step
  granularity and the whole-step leg cannot see it at all;
* THREE FAILURE CLASSES LIVE ONLY IN A COMPLETE STEP. A STALE MIRROR (the engine
  holds NumPy, so a sub-step left on the array path writes the HOST array and a
  device mirror held across it is a smooth, plausible, WRONG field); a SEAM
  (``zero_metal_B``/``zero_metal_D`` clear stored cell 0 between the curl and the
  constitutive sub-step); and an ACCUMULATING AUXILIARY. THE FOLD ADDS A FOURTH: the
  folded axis's ghost plane is WRITTEN BY ONE PASS AND READ BY THE NEXT, so a fold
  bug can be byte-perfect per sub-step and wrong per step. Leg 8's
  ``ghost_consumed_before_it_is_written`` is that class, and it is caught at step 0
  with 1,152-5,424 words on every case.

THE PRECONDITION IS A WINDOW AND THIS GATE MEASURED WHY. A census that reports a
scalar answers "how many", which licenses nothing about a run whose band entry is
transient. Measured here on a folded 3-D grid, censusing all 24 stored volumes after
every complete step:

    Q ~ 20 turn-on (df=0.05, envelope(0) = 3.7e-06)   500 steps   NEVER fires
    Q ~ 50 turn-on (df=0.02, envelope(0) = 1.9e-22)   700 steps   window [10, 21],
                                                                  then 678 CLEAN
    Q ~100 turn-on (df=0.01, envelope(0) = 5.4e-32)   900 steps   window [4, 899]

The middle row is the whole argument. A gate that censused at step 0, or at step 100,
or that reported "11 steps fired" out of 700, would be reporting a true number that
answers the wrong question. The window says WHERE, so a reader can tell that the
Q~50 run is refusable at its leading edge and certifiable after step 21 — and the
gate REFUSES the two firing runs BY NAME, recorded as a coverage refusal rather than
as a failure.

THE FILL AND THE CURL DO NOT SHARE THE PRECONDITION, and stating it per sub-step is
what makes it a measurement. The fill performs NO ARITHMETIC — its parity reduces to
a sign-bit operation or a plain copy, and a subnormal can live in a Metal buffer
because it is the arithmetic that flushes, not the storage. Measured in leg 7 with
27,648 subnormal operand words in play: THE FILL is identical at 1e-38 and at 1e-40
(0 differing, 576 moved) while THE CURL diverges in 13,248 words on the same grid at
the same scale. A family-wide precondition would understate what the fill delivers.

THE SIGNED-ZERO LEG IS NOT DECORATION HERE. On this backend the fold's parity is a
COMPILE-TIME specialisation — ``-x`` for odd, a PLAIN COPY for even — because a
runtime weight flushes every subnormal at BOTH signs. The refuted spelling
``0.0f - x`` is byte-identical to ``-x`` on the physical band and MEASURABLY WRONG on
signed zeros: leg 8's ``parity_spelled_as_zero_minus_x`` is a MEASURED NULL on the
physical seeding (0 words on every case) and CAUGHT under the +-0 lattice (64-140
words). That pair is what turns the platform fact into an enforced one.

STATED WEAKNESS, WHICH IS THIS CERTIFICATION'S ONE GAP AGAINST THE TRITON TWIN:
THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND. ``torch.mps.compile_shader``
exposes no disassembly, so unlike the Triton track this gate cannot refuse a compile
whose emitted code violates the policy, and it cannot establish that the contraction
guard was OBEYED — only that removing it changes the answer (leg 8's ``contract_on``,
220-433 words on every case). The byte gate and the mutation legs are the only
arbiters here and they are BEHAVIOURAL: they catch a wrong answer, not a wrong
instruction.

CASE DISCIPLINE, inherited and non-negotiable: every leg proves the step MOVED STATE
before claiming the bytes agreed, because zero-init is a FIXED POINT of the
constitutive sub-step and a no-op agreeing with a no-op is trivially identical; the
signed-zero census carries a FLOOR; armed mutations are launch-counted and a mutant
that never launched is DISARMED and FAILS; a needle that matched nothing is
NEEDLE-MISSED and FAILS; and the top-level verdict reads BOTH the compare COUNT and
the OUTCOME, the outcome re-read from the recorded rows rather than inferred from
having reached the summary.

NaN NEEDLES ARE EXCLUDED, on this track as on the Triton one. Measured while this
family was built: ``-x`` and ``x * -1.0f`` are byte-identical to each other on every
value class and BOTH differ from NumPy on NaN 6/6 (NumPy leaves the sign bit, Metal
flips it). That is a property of the NumPy reference, it is unreachable — a NaN in a
folded field is a dead run, not a parity question — and a needle there would measure
the toolchain's mood.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The MPS executor delivers `flush` natively and cannot deliver `keep`, and this
# arm64 host's DEFAULT resolves to keep (MEEP's set_zero_subnormals is a no-op under
# `#if HAVE_IMMINTRIN_H`, so match_meep measures "keep"). The gate requests flush
# EXPLICITLY, before anything resolves a policy, and stamps the resolution into the
# artifact — the claim is only as good as the precondition it was certified under.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import device, launch, preconditions  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal, symmetry  # noqa: E402
from meep_gpu.metal_kernels.device import compile_source  # noqa: E402
from meep_gpu.triton_kernels import symmetry as triton_symmetry  # noqa: E402
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: E402
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402
import metal_gate_kit as kit  # noqa: E402

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words
needle = kit.needle

MP = symmetry.CODE_MIRROR_PERIODIC
MM = symmetry.CODE_MIRROR_METALLIC
PER = symmetry.CODE_PERIODIC

#: Everything the engine stores that a folded step can touch. The whole-step leg
#: compares ALL of it after every complete step: the ``fu_*`` PML auxiliaries and
#: the ``f_w_*`` constitutive workspaces are STATE, and a kernel right for one
#: launch and wrong forever after diverges only once they accumulate.
STORED = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
          "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
          "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The driver's own order, five passes per half (driver.py:3282-3287, :3293-3302).
#: Spelled once, so the reference walk and the device walk cannot drift apart.
DRIVER_ORDER: Tuple[str, ...] = (
    "step_B", "fill_B_near", "fill_B_far", "update_H",
    "step_D", "fill_D_near", "fill_D_far", "update_E")


# ---------------------------------------------------------------------------
# The case matrix — every axis carries a mutation that is reachable on one value
# and a measured null on the other
# ---------------------------------------------------------------------------
#
#   TERMINATION      MIRROR_PERIODIC carries the top-plane mask AND the far ghost
#                    pass; MIRROR_METALLIC carries NEITHER. Measured:
#                    `drop_top_plane_mask` fires on the first and is 0 on the
#                    second, and `far_reads_one_row_over` has no far pass to plant
#                    in at all on the second;
#   PHASE            the parity is a SOURCE specialisation on this backend, so an
#                    odd plane is a DIFFERENT COMPILED FILL, not a different scalar;
#   FULL-COUNT       `reflect_row` is `stored - 2` at an even full count and
#                    `stored - 3` at an odd one, so at the default extent the wrong
#                    `n - 2` formula HAPPENS TO BE RIGHT (measured: 0 words at 2.0,
#                    384 at 2.1) and a matrix carrying only 2.0 measures nothing
#                    about the reflect row;
#   FOLDED AXES      a corner unowned on two planes must carry the PRODUCT of both
#                    parities, which one folded axis cannot exercise at all;
#   DIMENSIONALITY   the 2-D row is the sweep's own shape and the corpus's most
#                    common; the 3-D rows exist because a claim is only as strong as
#                    the words it moved and a 2-D ghost plane is 16 words against
#                    192 in 3-D.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("periodic_even_2d", dict()),
    ("periodic_even_3d", dict(depth=1.2)),
    ("periodic_odd_3d", dict(phase=-1, depth=1.2)),
    ("metallic_even_3d", dict(boundaries={"y": "metallic"}, depth=1.2)),
    ("metallic_odd_3d", dict(phase=-1, boundaries={"y": "metallic"}, depth=1.2)),
    ("x_fold_3d", dict(axis="X", depth=1.2)),
    ("odd_full_count_3d", dict(extent=2.1, depth=1.2)),
    ("two_axis_mixed_3d", dict(axis="XY", phase=(1, -1), depth=1.2)),
    ("two_axis_mixed_odd_3d", dict(axis="XY", phase=(-1, 1), extent=2.1,
                                   depth=1.2)),
)


def build(label: str) -> Tuple[Any, Any]:
    return matrix.folded(**dict(CASES)[label])


def tags(fields: Any, pml: Any) -> Tuple[str, ...]:
    """What a case IS, derived from the engine rather than typed beside the row.

    A mutation scopes itself with these. Typing the tags into the matrix would put
    the fold's single point of failure — the MIRROR_METALLIC / MIRROR_PERIODIC
    split — in a second place, and a scoping that disagreed with the grid would
    silently turn a reachable defect into a NEEDLE-MISSED row.
    """
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    codes = tuple(int(code) for code in (codes or ()))
    found: List[str] = []
    if MP in codes:
        found.append("far")           # a far ghost pass and a top-plane mask exist
    else:
        found.append("no_far")
    if sum(1 for code in codes if code in symmetry.MIRROR_CODES) >= 2:
        found.append("two_axis")
    else:
        found.append("one_axis")
    rows = triton_symmetry._far_reflect_rows(fields.grid) or (None, None, None)
    for axis, code in enumerate(codes):
        if code == MP and rows[axis] is not None:
            if int(fields.grid.shape[axis]) - int(rows[axis]) == 3:
                found.append("odd_full_count")
            else:
                found.append("even_full_count")
    if int(fields.grid.shape[2]) > 1:
        found.append("three_d")
    else:
        found.append("two_d")
    # DOES ANY EMITTED FILL ON THIS GRID CARRY A SIGN FLIP? Asked of the EMITTER,
    # not re-derived from the phase, because the answer is not the phase: the near
    # pass's weight is `+phase` and the far pass's is `-phase`, so an EVEN plane on
    # a folded METALLIC axis (which has no far pass at all) emits no `-x` anywhere
    # and a needle on the sign is structurally absent there. Measured: that is
    # exactly the case where `drop_the_sign_flip` plants nothing, and scoping it by
    # hand would have reported a structurally absent defect as a missed needle.
    if any("= -f" in symmetry.mirror_ghost_fill_source(
               entry["axis"], entry["phase"], entry["pass"], entry["shifts"])
           for family in triton_symmetry.GHOST_FILL_FAMILIES
           for pass_name in symmetry.FILL_PASSES
           for entry in symmetry.ghost_fill_axis_entries(fields.grid, family,
                                                         pass_name)):
        found.append("sign_flip")
    return tuple(found)


# ---------------------------------------------------------------------------
# Snapshots and oracles
# ---------------------------------------------------------------------------

def snapshot(fields: Any, names: Sequence[str] = STORED) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in names
            if getattr(fields, name, None) is not None}


def restore(fields: Any, state: Dict[str, Any]) -> None:
    for name, array in state.items():
        getattr(fields, name)[...] = array


def curl_names(sub_step: str) -> Tuple[str, ...]:
    spec = launch.SUB_STEPS[sub_step]
    return tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])


def fill_names(family: str) -> Tuple[str, ...]:
    return tuple(triton_symmetry.GHOST_FILL_FAMILIES[family]["targets"])


def constitutive_names(side: str) -> Tuple[str, ...]:
    spec = CONSTITUTIVE_SIDES[side]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def oracle(fields: Any, apply: Callable[[], None],
           names: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the array path, capture before/after, and PUT THE STATE BACK.

    Returning the fields to ``before`` is what lets the device launch start from the
    same bytes the oracle started from. A leg that forgot it would compare the
    kernel's output against a state the kernel never saw.
    """
    before = snapshot(fields, names)
    apply()
    after = snapshot(fields, names)
    restore(fields, before)
    return before, after


def moved_words(before: Dict[str, Any], after: Dict[str, Any]) -> int:
    return sum(differing(before[name], after[name]) for name in before)


def compared_words(state: Dict[str, Any]) -> int:
    return sum(int(words(array).size) for array in state.values())


def divergence(fields: Any, after: Dict[str, Any]) -> Dict[str, int]:
    return {name: differing(getattr(fields, name), value)
            for name, value in after.items()
            if differing(getattr(fields, name), value)}


# ---------------------------------------------------------------------------
# Device drivers, each with the mutation seam the legs need
# ---------------------------------------------------------------------------

def run_curl(fields: Any, pml: Any, sub_step: str,
             source: Optional[str] = None,
             counters: Optional[List[kit.Counter]] = None) -> Any:
    """Launch the folded curl through the ENGINE ROUTE, or a mutant in its place.

    ``source`` replaces the compiled entry point and nothing else, so a mutation
    leg exercises the shipped plan, the shipped bindings and the shipped dispatch
    with one defect planted in the device code. Dropping the seam would not slow the
    leg down, it would DISARM it: every mutation would launch the shipped kernel and
    report its defect as uncaught.
    """
    residency = device.Residency()
    plan = symmetry.plan_folded_pml_curl(fields, pml, sub_step, residency)
    if plan is None:
        raise AssertionError(symmetry.folded_composition_curl_coverage(
            fields, pml, sub_step, residency).reasons)
    if source is not None:
        function = kit.Counter(compile_source(source).folded_pml_curl_step)
        if counters is not None:
            counters.append(function)
        plan._functions = {shaders.CONTRACT_OFF: function}
    elif counters is not None:
        function = kit.Counter(plan._functions[shaders.CONTRACT_OFF])
        counters.append(function)
        plan._functions = {shaders.CONTRACT_OFF: function}
    residency.sync_in()
    plan.run()
    residency.sync_out()
    return plan


def build_fill(fields: Any, family: str, residency: Any,
               entries: Optional[Callable[[Any, str], Tuple[Any, Any]]] = None,
               ) -> Any:
    """The fill plan, from the engine's objects or from deliberately wrong entries.

    ``entries`` is the HOST mutation seam that ``plan_mirror_ghost_fill_from_arrays``
    exists for: a wrong reflect row, a flipped phase, a reversed axis order or a
    dropped far pass are plan-level defects, not device-code ones, and planting them
    in the source would test a different thing.
    """
    if entries is None:
        plan = symmetry.plan_mirror_ghost_fill(fields, family, "fill_" + family,
                                               residency)
        if plan is None:
            raise AssertionError(symmetry.mirror_ghost_fill_coverage(
                fields, family, residency).reasons)
        return plan
    near, far = entries(fields.grid, family)
    names = fill_names(family)
    return symmetry.plan_mirror_ghost_fill_from_arrays(
        family, "fill_" + family, {n: getattr(fields, n) for n in names},
        near, far, residency)


def apply_fill_source(plan: Any, text: Callable[[str, Dict[str, Any]], Optional[str]],
                      counters: Optional[List[kit.Counter]] = None) -> int:
    """Recompile each launch's entry point through ``text``. Returns how many took.

    THE PLANT COUNT IS THE RETURN VALUE AND IT IS LOAD-BEARING. This plan holds one
    launch per (pass, folded axis) and a needle naturally lives in only one of them
    — ``base + 2 * stride`` is the NEAR pass's and ``reflect_row`` is the FAR
    pass's. A harness that demanded every entry match would report a structurally
    scoped needle as NEEDLE-MISSED; one that demanded none would let a mutation that
    planted nothing report itself CAUGHT. So the caller scopes the mutation to the
    cases where the pass EXISTS and this returns zero only when it genuinely
    matched nothing.
    """
    planted = 0
    table: Dict[str, Any] = {}
    for entry in tuple(plan.near) + tuple(plan.far):
        shipped = symmetry.mirror_ghost_fill_source(
            entry["axis"], entry["phase"], entry["pass"], entry["shifts"])
        mutated = text(shipped, entry)
        chosen = shipped
        if mutated is not None and mutated != shipped:
            planted += 1
            chosen = mutated
        function = kit.Counter(compile_source(chosen).mirror_ghost_fill)
        if counters is not None:
            counters.append(function)
        table[entry["key"]] = function
    plan._functions = {shaders.CONTRACT_OFF: table}
    return planted


def run_fill(fields: Any, family: str,
             text: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None,
             entries: Optional[Callable[[Any, str], Tuple[Any, Any]]] = None,
             counters: Optional[List[kit.Counter]] = None) -> Tuple[Any, int]:
    residency = device.Residency()
    plan = build_fill(fields, family, residency, entries)
    planted = -1
    if text is not None:
        planted = apply_fill_source(plan, text, counters)
        if not planted:
            return plan, 0
    elif counters is not None:
        planted = apply_fill_source(plan, lambda source, entry: None, counters)
    residency.sync_in()
    plan.run_near()
    plan.run_far()
    residency.sync_out()
    return plan, planted


def run_constitutive(fields: Any, pml: Any, side: str,
                     counters: Optional[List[kit.Counter]] = None) -> Any:
    residency = device.Residency()
    plan = symmetry.plan_folded_constitutive(fields, pml, side, residency)
    if plan is None:
        raise AssertionError(symmetry.folded_constitutive_coverage(
            fields, pml, side, residency).reasons)
    if counters is not None:
        function = kit.Counter(plan._functions[shaders.CONTRACT_OFF])
        counters.append(function)
        plan._functions = {shaders.CONTRACT_OFF: function}
    residency.sync_in()
    plan.run()
    residency.sync_out()
    return plan


def reference_step(fields: Any, pml: Any) -> None:
    """One complete driver step on the array path — the whole-step leg's oracle."""
    stepping.step_B(fields, pml)
    stepping.fill_symmetry_bc_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.fill_symmetry_bc_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)


def composed_plans(fields: Any, pml: Any, residency: Any) -> Dict[str, Any]:
    plans = {
        "step_B": symmetry.plan_folded_pml_curl(fields, pml, "step_B", residency),
        "fill_B": symmetry.plan_mirror_ghost_fill(fields, "B", "fill_B", residency),
        "update_H": symmetry.plan_folded_constitutive(fields, pml, "H", residency),
        "step_D": symmetry.plan_folded_pml_curl(fields, pml, "step_D", residency),
        "fill_D": symmetry.plan_mirror_ghost_fill(fields, "D", "fill_D", residency),
        "update_E": symmetry.plan_folded_constitutive(fields, pml, "E", residency),
    }
    missing = [name for name, plan in plans.items() if plan is None]
    if missing:
        raise AssertionError(f"the composed folded step is short {missing}")
    return plans


def walk(plans: Dict[str, Any], order: Sequence[str] = DRIVER_ORDER) -> None:
    for item in order:
        if item.endswith("_near"):
            plans[item[:-5]].run_near()
        elif item.endswith("_far"):
            plans[item[:-4]].run_far()
        else:
            plans[item].run()


# ---------------------------------------------------------------------------
# LEG 0 — proof the bytes came off the GPU
# ---------------------------------------------------------------------------

def leg_execution(payload: Dict[str, Any], out: str) -> None:
    """A byte gate whose kernel never ran certifies the host's own arrays.

    Four facts, each of which would be true of a gate that had silently fallen back
    to NumPy and each of which is checked rather than assumed: the mirror is a torch
    tensor ON the MPS device; the entry point is a compiled Metal function and not a
    Python callable; the launch counter moved; and an UNLAUNCHED control leaves the
    target untouched, which is what makes "the launch changed the array" a statement
    about the launch.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    import torch  # noqa: PLC0415

    for label, _ in CASES[:2]:
        fields, pml = build(label)
        residency = device.Residency()
        plan = symmetry.plan_folded_pml_curl(fields, pml, "step_B", residency)
        assert plan is not None, label
        tensor = residency.tensor("Bx")
        entry = plan._functions[shaders.CONTRACT_OFF]

        before = snapshot(fields, curl_names("step_B"))
        residency.sync_in()
        unlaunched = differing(residency.tensor("Bx").cpu().numpy(),
                               np.ascontiguousarray(fields.Bx).reshape(-1))
        plan.run()
        residency.sync_out()
        after = snapshot(fields, curl_names("step_B"))

        row = {
            "case": label,
            "mirror_device": str(tensor.device),
            "mirror_dtype": str(tensor.dtype),
            "entry_point_type": type(entry).__name__,
            "entry_point_is_python": callable(getattr(entry, "__code__", None)),
            "launches": plan.launches,
            "unlaunched_control_differing": unlaunched,
            "moved": moved_words(before, after),
            "compared": compared_words(after),
            "differing": {},
        }
        rows.append(row)
        log(f"[leg0 execution] {label:<22} device={row['mirror_device']} "
            f"entry={row['entry_point_type']} launches={plan.launches} "
            f"moved={row['moved']} ({time.time() - started:.1f}s)")
        payload["legs"]["execution"] = rows
        save(payload, out)

        assert tensor.device.type == "mps", row
        assert tensor.dtype == torch.float32, row
        assert getattr(entry, "__code__", None) is None, (
            "the entry point is a PYTHON callable, so this gate would be certifying "
            "the host's own arithmetic rather than the device's")
        assert plan.launches == 1, row
        assert unlaunched == 0, (
            "sync_in did not reproduce the host bytes on the device, so nothing "
            "downstream can attribute a divergence to the kernel")
        kit.assert_moved(row["moved"], f"{label} execution", floor=100)


# ---------------------------------------------------------------------------
# LEG 1 — the folded curl, at SUB-STEP granularity
# ---------------------------------------------------------------------------

def leg_curl(payload: Dict[str, Any], out: str) -> None:
    """``folded_pml_curl_step`` vs ``stepping.step_B`` / ``step_D``, every case.

    SUB-STEP GRANULARITY IS THE POINT. The driver's fill passes overwrite exactly
    the planes the cell-0 and top-plane masks protect, so a mask-less kernel is
    bytewise-identical after a complete step; this is the only leg that can fail on
    a mask, and leg 8 proves it can.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _ in CASES:
        for sub_step in ("step_B", "step_D"):
            fields, pml = build(label)
            names = curl_names(sub_step)
            before, after = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            plan = run_curl(fields, pml, sub_step)
            bad = divergence(fields, after)
            row = {"case": label, "sub_step": sub_step, "tags": list(tags(fields, pml)),
                   "shape": list(map(int, fields.grid.shape)),
                   "codes": [int(c) for c in symmetry.folded_axis_kinds(
                       fields.grid, pml)[0]],
                   "compared": compared_words(after),
                   "moved": moved_words(before, after),
                   "launches": plan.launches,
                   "digest": kit.state_digest(after),
                   "differing": bad}
            rows.append(row)
            log(f"[leg1 curl] {label:<22} {sub_step} compared={row['compared']} "
                f"moved={row['moved']} differing={sum(bad.values())} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["curl"] = rows
            save(payload, out)
            assert plan.launches == 1, row
            kit.assert_moved(row["moved"], f"{label}/{sub_step} curl", floor=500)
            assert not bad, row


# ---------------------------------------------------------------------------
# LEG 2 — the two ghost passes
# ---------------------------------------------------------------------------

def leg_fill(payload: Dict[str, Any], out: str) -> None:
    """``mirror_ghost_fill`` vs BOTH array-path passes, separately and together.

    SEPARATELY MATTERS. ``zero_metal_*`` runs BETWEEN them in the driver
    (driver.py:3286 / :3301), so the plan holds them apart and a leg that only ever
    ran the fused ``run()`` would certify an object the driver cannot always use.
    Each pass is compared against its OWN array-path function first, then the pair
    against the pair.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _ in CASES:
        for family in ("B", "D"):
            near_fn = getattr(stepping, "fill_symmetry_bc_" + family)
            far_fn = getattr(stepping, "fill_folded_far_ghosts_" + family)
            names = fill_names(family)

            for pass_name, reference, launcher in (
                    ("near", near_fn, "run_near"),
                    ("far", far_fn, "run_far"),
                    ("both", None, None)):
                fields, pml = build(label)
                if reference is None:
                    before, after = oracle(
                        fields, lambda: (near_fn(fields), far_fn(fields)), names)
                else:
                    before, after = oracle(
                        fields, lambda: reference(fields), names)
                residency = device.Residency()
                plan = build_fill(fields, family, residency)
                residency.sync_in()
                if launcher is None:
                    plan.run_near()
                    plan.run_far()
                else:
                    getattr(plan, launcher)()
                residency.sync_out()
                bad = divergence(fields, after)
                expected = len(plan.near) if pass_name == "near" else (
                    len(plan.far) if pass_name == "far"
                    else len(plan.near) + len(plan.far))
                row = {"case": label, "family": family, "pass": pass_name,
                       "tags": list(tags(fields, pml)),
                       "near_axes": [e["axis"] for e in plan.near],
                       "far_axes": [e["axis"] for e in plan.far],
                       "reflect_rows": [e["reflect_row"] for e in plan.far],
                       "replaces_sub_steps": list(plan.replaces_sub_steps),
                       "compared": compared_words(after),
                       "moved": moved_words(before, after),
                       "launches": plan.launches,
                       "differing": bad}
                rows.append(row)
                log(f"[leg2 fill] {label:<22} {family} {pass_name:<5} "
                    f"compared={row['compared']} moved={row['moved']} "
                    f"launches={plan.launches} differing={sum(bad.values())} "
                    f"({time.time() - started:.1f}s)")
                payload["legs"]["fill"] = rows
                save(payload, out)
                assert plan.launches == expected, row
                # A folded METALLIC axis HAS no far pass, so the far row there is a
                # PREDICTED NULL and is recorded as one rather than dropped: a
                # silently skipped case is indistinguishable from a forgotten one.
                if pass_name == "far" and not plan.far:
                    kit.predicted_null(row, "this grid carries no folded PERIODIC "
                                            "axis, so the far ghost pass does not "
                                            "exist and writes nothing")
                    assert row["moved"] == 0, row
                else:
                    kit.assert_moved(row["moved"], f"{label}/{family}/{pass_name}",
                                     floor=16)
                assert not bad, row


# ---------------------------------------------------------------------------
# LEG 3 — the re-admitted CERTIFIED constitutive body
# ---------------------------------------------------------------------------

def leg_constitutive(payload: Dict[str, Any], out: str) -> None:
    """``update_H`` / ``update_E`` on a folded grid.

    THIS FAMILY CONTRIBUTES ONLY THE ADMISSION on these two slots — the arithmetic,
    the source and the plan class are the certified ones, reached through the
    certified builder. So a divergence here could only come from the predicate, and
    that is exactly why the leg exists: a predicate that admitted a folded grid the
    certified body cannot step would be invisible to the curl and fill legs.

    ZERO-INIT IS A FIXED POINT OF THIS SUB-STEP, which is why the builder fills
    every volume with physical-band values and why the vacuity floor is checked
    before the identity is claimed.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _ in CASES:
        for side in ("H", "E"):
            fields, pml = build(label)
            names = constitutive_names(side)
            reference = getattr(stepping, "update_" + side)
            before, after = oracle(fields, lambda: reference(fields, pml), names)
            plan = run_constitutive(fields, pml, side)
            bad = divergence(fields, after)
            row = {"case": label, "side": side, "tags": list(tags(fields, pml)),
                   "plan_class": type(plan).__name__,
                   "compared": compared_words(after),
                   "moved": moved_words(before, after),
                   "launches": plan.launches, "differing": bad}
            rows.append(row)
            log(f"[leg3 constitutive] {label:<22} update_{side} "
                f"compared={row['compared']} moved={row['moved']} "
                f"differing={sum(bad.values())} ({time.time() - started:.1f}s)")
            payload["legs"]["constitutive"] = rows
            save(payload, out)
            assert plan.launches == 1, row
            assert type(plan).__name__ == "ConstitutivePlan", (
                "this slot must carry the CERTIFIED plan class; a family-private "
                "copy would make 'same kernel' a claim rather than a fact")
            kit.assert_moved(row["moved"], f"{label}/update_{side}", floor=500)
            assert not bad, row


# ---------------------------------------------------------------------------
# LEG 4 — the whole step, per COMPLETE STEP, first divergence reported
# ---------------------------------------------------------------------------

WHOLE_STEP_BUDGET = 12


def leg_whole_step(payload: Dict[str, Any], out: str) -> None:
    """The object the engine would actually run, over a stated budget.

    THE REAL ARBITER. Six green per-sub-step legs say nothing about a composed step:
    a STALE MIRROR, a SEAM and an ACCUMULATING AUXILIARY are invisible in a
    single-launch comparison, and the FOLD adds a fourth — the mirror plane is
    written by one pass and READ by the next, so a fold bug can be byte-perfect per
    sub-step and wrong per step.

    THE FIRST DIVERGENT STEP IS REPORTED, not just "it diverged": with an
    accumulating auxiliary the step number is the diagnosis. Every slot's launch
    counter is asserted, because a slot that passes by NOT EXECUTING is the hollow
    pass this whole discipline exists to make impossible.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _ in CASES:
        fields, pml = build(label)
        reference_fields, reference_pml = build(label)
        residency = device.Residency()
        plans = composed_plans(fields, pml, residency)
        residency.sync_in()

        first_divergent: Optional[Dict[str, Any]] = None
        per_step: List[int] = []
        compared = 0
        for step in range(WHOLE_STEP_BUDGET):
            reference_step(reference_fields, reference_pml)
            walk(plans)
            residency.sync_out()
            state = snapshot(reference_fields)
            compared += compared_words(state)
            bad = divergence(fields, state)
            per_step.append(sum(bad.values()))
            if bad and first_divergent is None:
                first_divergent = {"step": step, "differing": bad}

        evolved = moved_words({n: np.zeros_like(v) for n, v in state.items()}, state)
        expected_fill = WHOLE_STEP_BUDGET * (len(plans["fill_B"].near)
                                             + len(plans["fill_B"].far))
        row = {"case": label, "budget": WHOLE_STEP_BUDGET,
               "tags": list(tags(fields, pml)),
               "compared": compared, "per_step_differing": per_step,
               "first_divergent_step": first_divergent,
               "state_evolved_words": evolved,
               "launches": {name: plan.launches for name, plan in plans.items()},
               "digest": kit.state_digest(state),
               "differing": first_divergent["differing"] if first_divergent else {}}
        rows.append(row)
        log(f"[leg4 whole_step] {label:<22} compared={compared} "
            f"first_divergent={first_divergent['step'] if first_divergent else None} "
            f"evolved={evolved} ({time.time() - started:.1f}s)")
        payload["legs"]["whole_step"] = rows
        save(payload, out)

        assert first_divergent is None, row
        kit.assert_moved(evolved, f"{label} whole step", floor=1000)
        assert plans["step_B"].launches == WHOLE_STEP_BUDGET, row
        assert plans["step_D"].launches == WHOLE_STEP_BUDGET, row
        assert plans["update_H"].launches == WHOLE_STEP_BUDGET, row
        assert plans["update_E"].launches == WHOLE_STEP_BUDGET, row
        assert plans["fill_B"].launches == expected_fill, row
        assert plans["fill_D"].launches > 0, row


# ---------------------------------------------------------------------------
# LEG 5 — the reduction to the certified curl
# ---------------------------------------------------------------------------

def leg_reduction(payload: Dict[str, Any], out: str) -> None:
    """On an UNFOLDED grid the folded curl IS the certified curl, measured.

    The family's structural claim is that its only device-code delta over
    ``shaders._CURL_TEMPLATE`` is the top-plane mask — the ghost gather and the
    cell-0 mask being the CERTIFIED emitters' own output reached through reduced
    codes. ``test_metal_symmetry`` pins that by CHARACTER; this pins it by BYTES, on
    a grid where the mask emits nothing at all, so the two kernels must produce
    identical words from identical inputs.

    IT RUNS THROUGH ``plan_folded_pml_curl_from_arrays`` DELIBERATELY. The routing
    predicate REFUSES an unfolded grid (that is the inverted clause that keeps this
    family disjoint from the plain one), so the engine route cannot reach this
    configuration and the array route is the only way to measure the reduction. Leg
    9 checks the refusal itself.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for sub_step in ("step_B", "step_D"):
        fields, pml = matrix.cart()
        codes = [1 if kind == "metallic" else 0
                 for kind in stepping._boundary_kinds(fields.grid, pml)]
        names = curl_names(sub_step)
        spec = launch.SUB_STEPS[sub_step]

        before = snapshot(fields, names)
        residency = device.Residency()
        certified = launch.plan_pml_curl(fields, pml, sub_step, residency)
        assert certified is not None
        residency.sync_in()
        certified.run()
        residency.sync_out()
        after = snapshot(fields, names)
        restore(fields, before)

        residency = device.Residency()
        folded = symmetry.plan_folded_pml_curl_from_arrays(
            sub_step,
            {name: getattr(fields, name) for name in STORED
             if getattr(fields, name, None) is not None},
            {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{spec['suffix']}")
             for axis in "xyz" for stem in ("kms", "sinv")},
            codes, float(fields.grid.dt / fields.grid.dx), residency)
        residency.sync_in()
        folded.run()
        residency.sync_out()

        bad = divergence(fields, after)
        mask = symmetry.folded_top_plane_mask(codes, spec["backward"])
        row = {"sub_step": sub_step, "codes": codes,
               "top_plane_mask_emits_code": "curl" in mask,
               "wide_predicate_admits": symmetry.folded_pml_curl_coverage(
                   fields, pml, sub_step, residency).covered,
               "routing_predicate_admits": symmetry.folded_composition_curl_coverage(
                   fields, pml, sub_step, residency).covered,
               "compared": compared_words(after),
               "moved": moved_words(before, after),
               "launches": folded.launches, "differing": bad}
        rows.append(row)
        log(f"[leg5 reduction] {sub_step} compared={row['compared']} "
            f"moved={row['moved']} differing={sum(bad.values())} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["reduction"] = rows
        save(payload, out)
        assert not row["top_plane_mask_emits_code"], (
            "an unfolded grid emitted top-plane mask code, so this leg is not "
            "measuring the reduction it claims to")
        assert row["wide_predicate_admits"] and not row["routing_predicate_admits"], (
            "the standalone predicate must ADMIT an unfolded grid (so this leg can "
            "run) and the routing one must REFUSE it (so no unfolded row is "
            "ambiguous); the two verdicts disagreeing the other way is a "
            "composition defect", row)
        kit.assert_moved(row["moved"], f"reduction/{sub_step}", floor=500)
        assert not bad, row

    # --- why m3's scope is a property of the GRID, not of the harness -------
    #
    # `m3_fold_applied_to_the_wrong_axis` swaps the mirror code with the axis beside
    # it and is scoped to ONE-AXIS folds. On a grid folded on both x and y that swap
    # is the IDENTITY and emits a CHARACTER-IDENTICAL source — there is no mutant to
    # launch, so the mutation harness cannot record it as a null (a transform that
    # changed nothing is NEEDLE-MISSED there, and rightly). It is asserted here
    # instead, so the scope is a recorded measurement rather than an unexplained
    # `tags` entry a reader has to take on trust.
    fields, pml = build("two_axis_mixed_3d")
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    codes = tuple(int(code) for code in codes)
    swapped = (codes[1], codes[0], codes[2])
    row = {"sub_step": "both", "case": "two_axis_mixed_3d", "codes": list(codes),
           "swapped_codes": list(swapped),
           "sources_identical": all(
               symmetry.folded_curl_source(codes, backward)
               == symmetry.folded_curl_source(swapped, backward)
               for backward in (False, True)),
           "note": "m3's transform is the identity here, which is why m3 is scoped "
                   "to one-axis folds rather than armed and reported as a null",
           "compared": 0, "differing": {}}
    rows.append(row)
    payload["legs"]["reduction"] = rows
    save(payload, out)
    assert codes[0] == codes[1] and row["sources_identical"], row


# ---------------------------------------------------------------------------
# LEG 6 — signed zeros through every fill plane
# ---------------------------------------------------------------------------

def seed_signed_zeros(fields: Any, seed: int = 20260816) -> Dict[str, int]:
    """A +-0 lattice in every field volume, over a physical-band background.

    NOT A DECORATION ON THIS FAMILY. The fill's parity is a SIGN-BIT OPERATION, so
    a signed zero is the ONLY value class that separates the shipped spelling
    (``-x``) from the refuted one (``0.0f - x``): ``-(+0.0)`` is ``-0.0`` and
    ``0.0f - (+0.0f)`` is ``+0.0``, while on every normal operand the two agree in
    bits. Leg 8 measures exactly that — the refuted spelling is 0 words on the
    physical seeding and 64-152 under this one.

    THE LATTICE IS INDEXED IN 3-D, NOT FLAT, AND THAT IS A MEASURED CORRECTION.
    This function first strided the FLATTENED array by 3 (``values[0::3] = +0.0``,
    ``values[1::3] = -0.0``), and on the 2-D case that ALIASED WITH THE GRID STRIDE:
    with ``nz = 1`` the far pass's read plane is flat index ``12*i + 10``, which is
    congruent to 1 modulo 3 for EVERY ``i``, so the whole plane carried ``-0.0`` and
    nothing else. Both spellings agree on ``-0.0`` (``-(-0.0)`` and
    ``0.0f - (-0.0f)`` are both ``+0.0``), so ``m14`` came back CAUGHT 14/16 with the
    two misses being that one case — a signed-zero leg silently disarmed by a stride
    coincidence, which is the most dangerous shape a seeding defect can take because
    the census still reported thousands of zeros. Keying on ``(i + j + k) % 3``
    depends on no stride at all, so every axis-aligned plane of extent >= 3 along any
    varying axis carries BOTH signs. Measured after the fix: 16/16.
    """
    rng = np.random.default_rng(seed)
    counts = {"negative_zero": 0, "positive_zero": 0}
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        shape = tuple(int(n) for n in array.shape)
        lattice = sum(np.indices(shape)) % 3
        values = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
        values[lattice == 0] = np.float32(0.0)
        values[lattice == 1] = np.float32(-0.0)
        array[...] = values
        found = preconditions.signed_zero_census(array)
        counts["negative_zero"] += found["negative_zero"]
        counts["positive_zero"] += found["positive_zero"]
    return counts


def leg_signed_zero(payload: Dict[str, Any], out: str) -> None:
    """Curl and fill under a +-0 lattice, with a CENSUS FLOOR on both zeros."""
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _ in CASES:
        fields, pml = build(label)
        counts = seed_signed_zeros(fields)
        kit.assert_census_floor(counts["negative_zero"], f"{label} -0.0", floor=64)
        kit.assert_census_floor(counts["positive_zero"], f"{label} +0.0", floor=64)

        outcomes: Dict[str, Any] = {}
        bad_total: Dict[str, int] = {}
        for family in ("B", "D"):
            names = fill_names(family)
            near_fn = getattr(stepping, "fill_symmetry_bc_" + family)
            far_fn = getattr(stepping, "fill_folded_far_ghosts_" + family)
            before, after = oracle(
                fields, lambda: (near_fn(fields), far_fn(fields)), names)
            plan, _ = run_fill(fields, family)
            bad = divergence(fields, after)
            outcomes["fill_" + family] = {"moved": moved_words(before, after),
                                          "compared": compared_words(after)}
            bad_total.update({f"fill_{family}:{k}": v for k, v in bad.items()})
            kit.assert_moved(outcomes["fill_" + family]["moved"],
                             f"{label} signed-zero fill {family}", floor=16)
        for sub_step in ("step_B", "step_D"):
            names = curl_names(sub_step)
            before, after = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            run_curl(fields, pml, sub_step)
            bad = divergence(fields, after)
            outcomes[sub_step] = {"moved": moved_words(before, after),
                                  "compared": compared_words(after)}
            bad_total.update({f"{sub_step}:{k}": v for k, v in bad.items()})
            kit.assert_moved(outcomes[sub_step]["moved"],
                             f"{label} signed-zero {sub_step}", floor=500)

        row = {"case": label, "census": counts, "outcomes": outcomes,
               "compared": sum(o["compared"] for o in outcomes.values()),
               "differing": bad_total}
        rows.append(row)
        log(f"[leg6 signed_zero] {label:<22} -0={counts['negative_zero']} "
            f"+0={counts['positive_zero']} compared={row['compared']} "
            f"differing={sum(bad_total.values())} ({time.time() - started:.1f}s)")
        payload["legs"]["signed_zero"] = rows
        save(payload, out)
        assert not bad_total, row


# ---------------------------------------------------------------------------
# LEG 7 — the precondition, as a WINDOW
# ---------------------------------------------------------------------------

#: Three driven runs, chosen so the answer is not the same for all three. ``df`` is
#: the source bandwidth and ``cut`` how many widths before the peak the run starts,
#: so ``envelope(0) = exp(-cut^2/2)`` is the LEADING-EDGE magnitude the fields are
#: born at. THE MEASURED RESULTS, on a folded 3-D grid on this host 2026-08-16:
#:
#:   Q ~ 20   envelope(0) = 3.7e-06   500 steps   NEVER fires
#:   Q ~ 50   envelope(0) = 1.9e-22   700 steps   window [10, 21], then 678 CLEAN
#:   Q ~100   envelope(0) = 5.4e-32   900 steps   window [4, 899]
#:
#: The middle row is why this leg reports a WINDOW. "11 of 700 steps fired" is a
#: true scalar that answers the wrong question; "[10, 21] then clean" tells a reader
#: the leading edge is refusable and the rest of the run is not.
DRIVEN_RUNS: Tuple[Tuple[str, float, float, int, bool], ...] = (
    ("physical_Q20", 0.05, 5.0, 500, True),
    ("narrow_band_Q50", 0.02, 10.0, 700, False),
    ("narrow_band_Q100", 0.01, 12.0, 900, False),
)

#: The scaled ladder. ``expect_clean`` is what the CENSUS must say; the fill and the
#: curl are then asked separately, because they do NOT share this precondition.
SCALED_LADDER: Tuple[Tuple[str, float, bool], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-20, True),
    ("subnormal_band", 1e-38, False),
    ("deep", 1e-40, False),
)


def drive(fields: Any, pml: Any, df: float, cut: float, steps: int,
          progress: str) -> Dict[str, Any]:
    """Run a pulsed source and census EVERY stored volume after EVERY step.

    The window is taken over the whole stored inventory rather than over a sampled
    plane, and per step rather than once, because band entry is a RUN-and-WINDOW
    fact: measured here, a Q~50 turn-on is in the band for steps 10 through 21 and
    clean for the 678 that follow.
    """
    for name in STORED:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = 0.0
    shape = tuple(int(n) for n in fields.grid.shape)
    cell = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    t0, width = cut / df, 1.0 / df
    dt = float(fields.grid.dt)
    window = preconditions.SubnormalWindow(first_step=0, last_step=steps - 1,
                                           per_array_words=64)
    first = last = None
    fired_steps = 0
    started = time.time()
    for step in range(steps):
        t = step * dt
        fields.Ez[cell] += np.float32(
            math.exp(-((t - t0) ** 2) / (2.0 * width * width))
            * math.cos(2.0 * math.pi * t))
        reference_step(fields, pml)
        found = 0
        for name in STORED:
            array = getattr(fields, name, None)
            if array is not None:
                found += window.observe(name, array, step=step)
        if found:
            fired_steps += 1
            first = step if first is None else first
            last = step
        if step % 100 == 0:
            log(f"    [{progress}] step {step}/{steps} fired_steps={fired_steps} "
                f"window={[first, last]} ({time.time() - started:.1f}s)")
    report = window.report()
    report["first_fired"] = first
    report["last_fired"] = last
    report["fired_steps"] = fired_steps
    report["steps"] = steps
    report["clean_steps_after_last"] = (steps - 1 - last) if last is not None else steps
    return report


def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The window, its refusals, and the fill/curl split.

    THREE PARTS:

    1. THE DRIVEN RUNS. Each records ``first_fired`` / ``last_fired``. A run that
       stays clean CERTIFIES its window; a run that enters the band is REFUSED BY
       NAME and the refusal is recorded as a COVERAGE REFUSAL, not a failure — the
       claim simply does not extend to those steps, and saying so is the honest
       outcome on a backend whose flush has no lever;
    2. THE SCALED LADDER, which demonstrates the census FIRING. A precondition never
       shown to fire is decorative;
    3. THE SPLIT. At the same scale, on the same grid, the FILL is byte-identical
       inside the band and the CURL is not. The precondition bounds the arithmetic
       half of this family and not the fill, and that is a measurement rather than
       an argument.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()

    for name, df, cut, steps, expect_clean in DRIVEN_RUNS:
        fields, pml = build("periodic_even_3d")
        report = drive(fields, pml, df, cut, steps, name)
        refused = not report["clean"]
        row = {"part": "driven_window", "run": name, "df": df,
               "leading_edge_envelope": math.exp(-(cut ** 2) / 2.0),
               "window": [report["first_fired"], report["last_fired"]],
               "first_fired": report["first_fired"],
               "last_fired": report["last_fired"],
               "fired_steps": report["fired_steps"], "steps": steps,
               "clean_steps_after_last": report["clean_steps_after_last"],
               "subnormal_words": report["subnormal_words"],
               "observed_words": report["observed_words"],
               "vacuous": report["vacuous"],
               "verdict": ("REFUSED (subnormal precondition)" if refused
                           else "CERTIFIED over the stated window"),
               "coverage_refusal": refused,
               "expected_clean": expect_clean,
               "differing": {}}
        rows.append(row)
        log(f"[leg7 window] {name:<18} window={row['window']} "
            f"fired_steps={row['fired_steps']}/{steps} "
            f"verdict={row['verdict']} ({time.time() - started:.1f}s)")
        payload["legs"]["precondition"] = rows
        save(payload, out)
        assert not report["vacuous"], report["vacuity_reasons"]
        if expect_clean:
            assert report["clean"], (
                f"{name}: the physical-band driven run entered the subnormal band, "
                f"so the window this family is certified over is not the one this "
                f"gate assumed", row)
        else:
            assert not report["clean"], (
                f"{name}: the narrow-band control did NOT reach the band, so the "
                f"window this leg exists to demonstrate was never constructed", row)
            assert report["first_fired"] is not None, row

    # --- part 2 + 3: the ladder, and the fill/curl split --------------------
    for name, scale, expect_clean in SCALED_LADDER:
        for label in ("periodic_even_3d", "periodic_odd_3d"):
            fields, pml = build(label)
            for volume in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                           "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                array = getattr(fields, volume, None)
                if array is not None:
                    array[...] = (array.astype(np.float64)
                                  * scale).astype(np.float32)
            census = sum(subnormal.census(getattr(fields, n)) for n in STORED
                         if getattr(fields, n, None) is not None)

            names = fill_names("B")
            near_fn, far_fn = (stepping.fill_symmetry_bc_B,
                               stepping.fill_folded_far_ghosts_B)
            before, after = oracle(
                fields, lambda: (near_fn(fields), far_fn(fields)), names)
            run_fill(fields, "B")
            fill_bad = sum(divergence(fields, after).values())
            fill_moved = moved_words(before, after)
            restore(fields, before)

            names = curl_names("step_B")
            before, after = oracle(
                fields, lambda: stepping.step_B(fields, pml), names)
            run_curl(fields, pml, "step_B")
            curl_bad = sum(divergence(fields, after).values())
            restore(fields, before)

            row = {"part": "scaled_ladder", "scale": name, "factor": scale,
                   "case": label, "operand_subnormal_words": census,
                   "census_fired": census > 0,
                   "fill_differing": fill_bad, "fill_moved": fill_moved,
                   "curl_differing": curl_bad,
                   "verdict": ("REFUSED (precondition) for the CURL; the FILL is "
                               "band-safe" if census else "identical"),
                   "compared": compared_words(after) + compared_words(before),
                   # The band rows' divergence is EXPECTED, so it is reported under
                   # its own key and `differing` stays empty there; a divergence on
                   # a band that must agree lands in `differing` and reaches the
                   # summary's independent second read. ONLY NONZERO ENTRIES GO IN,
                   # which is the convention `divergence()` sets for every other
                   # leg — an all-zero dict is TRUTHY and this row carried one, so
                   # the summary's second read reported four clean ladder rows as
                   # failures. Caught by that read doing its job, and fixed at the
                   # source as well as in the reader.
                   "differing": ({} if not expect_clean else
                                 {k: v for k, v in (("fill", fill_bad),
                                                    ("curl", curl_bad)) if v}),
                   "expected_to_diverge": not expect_clean}
            rows.append(row)
            log(f"[leg7 ladder] {name:<14} {label:<18} subnormal={census} "
                f"fill={fill_bad} curl={curl_bad} ({time.time() - started:.1f}s)")
            payload["legs"]["precondition"] = rows
            save(payload, out)

            kit.assert_moved(fill_moved, f"{name}/{label} ladder fill", floor=16)
            if expect_clean:
                assert census == 0, row
                assert fill_bad == 0 and curl_bad == 0, row
            else:
                kit.assert_census_floor(census, f"{name}/{label}", floor=1000)
                assert curl_bad > 0, (
                    "the CURL was expected to diverge in the band; if it does not, "
                    "the cliff the whole precondition rests on was not reproduced "
                    "and the precondition is decorative", row)
                assert fill_bad == 0, (
                    "the FILL was measured band-safe when this family was built "
                    "(its parity is a sign-bit operation, and a subnormal can live "
                    "in a Metal buffer because it is the arithmetic that flushes, "
                    "not the storage); a divergence here retires that claim", row)


# ---------------------------------------------------------------------------
# LEG 8 — mutations, armed, launch-counted, FOLD-SPECIFIC
# ---------------------------------------------------------------------------

def replace(source: str, old: str, new: str) -> Optional[str]:
    """A scoped needle: ``None`` when this specialisation does not carry the text.

    DIFFERENT FROM ``kit.needle``, DELIBERATELY. That one raises so a needle that
    matched NOTHING becomes NEEDLE-MISSED. Here a single mutation is applied to
    several launches of one plan and a needle legitimately lives in only one of them
    (``base + 2 * stride`` is the NEAR pass's text and ``reflect_row`` the FAR
    pass's), so per-launch absence is normal and the MISS is decided at the case
    level: :func:`apply_fill_source` returns the plant count and a case that planted
    NOTHING is what the harness records as a miss.
    """
    return source.replace(old, new) if old in source else None


def rebuild_curl(codes: Sequence[int], backward: bool,
                 transform: Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]],
                 ) -> Optional[str]:
    """Re-emit the curl from a DIFFERENT code triple — the fold-placement needles.

    A defect that puts the fold on the wrong axis, or gives a mirror axis the
    periodic ghost rule, is not a text edit inside the body: it is the whole
    specialisation choosing wrong. Re-emitting through the shipped emitter is what
    makes the mutant a kernel the family could actually have produced.
    """
    wanted = transform(tuple(int(code) for code in codes))
    if wanted is None or tuple(wanted) == tuple(int(c) for c in codes):
        return None
    return symmetry.folded_curl_source(wanted, backward)


#: THE CURL MUTATIONS. ``tags`` scopes a mutation to the cases where its defect is
#: STRUCTURALLY REACHABLE; ``must_catch`` is three-valued, and the two conditional
#: pairs below are the gate's own demonstration that the two terminations are
#: different kernels rather than one kernel with a flag.
CURL_MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"label": "m1_drop_top_plane_mask",
     "must_catch": True, "tags": ("far",),
     "why": "the top-plane arm of `_mask_non_owned_cells` (stepping.py:1898-1902) "  # stepping.py live lines for the frozen device-text citation(s) in this string: 1898-1902->1945-1949
            "deleted. It is the ONE block this family emits that the certified "
            "emitters do not, and it fires only where `_stored_past_owned` is True "
            "— a folded axis whose outer declaration is periodic. Measured "
            "384-1,080 words",
     "apply": lambda src, codes, backward: replace(
         src, symmetry.folded_top_plane_mask(codes, backward),
         "    // MUTANT: the top-plane mask is dropped")},
    {"label": "m2_top_plane_mask_null_without_a_periodic_fold",
     "must_catch": False, "tags": ("no_far",),
     "why": "THE OTHER HALF OF m1, and the reason this gate runs both terminations. "
            "On a folded METALLIC axis the top plane is OWNED and STEPPED, the "
            "emitter produces no mask lines at all, and dropping the (absent) mask "
            "must be a MEASURED NULL. A leg carrying only the periodic case would "
            "report m1 as a property of folds rather than of one termination",
     "apply": lambda src, codes, backward: replace(
         src, symmetry.folded_top_plane_mask(codes, backward),
         "    // MUTANT: the top-plane mask is dropped")},
    {"label": "m3_fold_applied_to_the_wrong_axis",
     "must_catch": True, "tags": ("one_axis",),
     "why": "the fold moved to a NEIGHBOURING AXIS — the mirror code swapped with "
            "the axis beside it. Both the ghost rule and both masks follow the "
            "code, so this is the single defect that most looks like a working "
            "kernel: it steps, it is stable, and a whole plane is wrong. Measured "
            "1,248-1,944 words. SCOPED TO ONE-AXIS FOLDS because on a grid folded "
            "on BOTH x and y the swap is the IDENTITY and emits a "
            "character-identical source; that is not a null this harness can "
            "record (there is no mutant to launch), so it is asserted as a source "
            "fact in leg 5 instead",
     "apply": lambda src, codes, backward: rebuild_curl(
         codes, backward, lambda c: (c[1], c[0], c[2]))},
    {"label": "m5_mirror_axis_takes_the_periodic_ghost",
     "must_catch": True, "tags": (),
     "why": "the fold's ghost rule WRAPS instead of terminating. `_reduced_codes` "
            "maps both mirror codes to METALLIC so the certified emitter masks the "
            "out-of-range neighbour and serves 0.0 (stepping.py:1788-1791); mapping "
            "them to PERIODIC makes the near face read the far face. This is the "
            "defect that porting a periodic kernel and 'adding symmetry' produces. "
            "Measured 768-1,656 words",
     "apply": lambda src, codes, backward: rebuild_curl(
         codes, backward,
         lambda c: tuple(PER if x in symmetry.MIRROR_CODES else x for x in c))},
    {"label": "m6_cell_zero_mask_dropped",
     "must_catch": True, "tags": (),
     "why": "the cell-0 arm of the ownership mask deleted. On a folded axis this is "
            "the mask that WIDENS from `== METALLIC` to `!= PERIODIC`, so it is "
            "load-bearing on exactly the cells the fold owns. Measured 384-1,080 "
            "words",
     "apply": lambda src, codes, backward: replace(
         src, symmetry.folded_cell_zero_mask(codes, backward),
         "    // MUTANT: the cell-0 mask is dropped")},
    {"label": "m7_contract_on",
     "must_catch": True, "tags": (),
     "why": "the ONE compile option removed. Byte-identity on this backend is "
            "attainable only with `#pragma clang fp contract(off)`; without it the "
            "compiler contracts the curl's multiply-adds into fmas and the gate "
            "fails. THIS IS THE ONLY EVIDENCE AVAILABLE HERE that the guard is "
            "load-bearing — there is no disassembly to read, so the gate can show "
            "that removing it CHANGES the answer but cannot show that leaving it in "
            "was OBEYED. Measured 220-433 words",
     "apply": lambda src, codes, backward: replace(
         src, shaders.contraction_pragma("off"),
         shaders.contraction_pragma("fast"))},
)

#: THE FILL MUTATIONS. These are the fold's own — the plane, the half, the sign and
#: the row — and they are the reason this gate exists separately from the certified
#: curl's.
FILL_MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"label": "m8_mirror_the_wrong_half",
     "must_catch": True, "tags": (), "seeding": "physical",
     "why": "THE NEAR PASS WRITES THE FAR HALF. `_write_mirror_ghost` "
            "(stepping.py:1498) sets stored cell 0 from cell 2; writing the LAST "
            "stored slot instead leaves the reconstructed half stale and clobbers "
            "the plane the far pass owns. Measured 384-1,104 words",
     "apply": lambda src, entry: replace(src, "[base] =",
                                         "[base + last * stride] =")},
    {"label": "m9_drop_the_sign_flip",
     "must_catch": True, "tags": ("sign_flip",), "seeding": "physical",
     "why": "the parity dropped on a mirrored component — `-x` spelled as a plain "
            "copy. A mirror plane reconstructs the discarded half with the sign "
            "INVERTED wherever the weight is -1, and dropping it is a smooth, "
            "plausible, wrong field rather than a crash. SCOPED BY THE EMITTER: an "
            "even plane on a folded METALLIC axis has no far pass and a `+phase` "
            "near weight, so it emits no `-x` at all and there is nothing to "
            "plant. Measured 192-408 words",
     "apply": lambda src, entry: replace(src, "= -f", "= f")},
    {"label": "m10_mirror_plane_shifted_one_cell",
     "must_catch": True, "tags": (), "seeding": "physical",
     "why": "the near pass reads cell 3 instead of `MIRROR_SOURCE_INDEX = 2`. The "
            "source index is MEEP's halved-origin offset (`io = -2`), and being one "
            "cell out is a half-cell error in the reconstructed half — converged, "
            "smooth and wrong. Measured 192-564 words",
     "apply": lambda src, entry: replace(src, "base + 2 * stride",
                                         "base + 3 * stride")},
    {"label": "m11_far_pass_reads_one_row_over",
     "must_catch": True, "tags": ("far",), "seeding": "physical",
     "why": "the SAME defect on the far plane: `reflect_row + 1`. Scoped to the "
            "cases that HAVE a far pass, because a folded METALLIC axis carries "
            "none and planting there would report a structurally absent pass as a "
            "missed needle. Measured 192-564 words",
     "apply": lambda src, entry: replace(
         src, "base + reflect_row * stride", "base + (reflect_row + 1) * stride")},
    {"label": "m12_plane_decode_from_the_wrong_axis",
     "must_catch": True, "tags": (), "seeding": "physical",
     "why": "THE FOLD APPLIED TO THE WRONG AXIS, on the fill side: the flat-index "
            "decode re-emitted for the next axis while the entry still names this "
            "one. The launch then walks a plane of the wrong orientation with the "
            "right stride count. Measured 352-1,433 words",
     "apply": lambda src, entry: symmetry.mirror_ghost_fill_source(
         (int(entry["axis"]) + 1) % 3, entry["phase"], entry["pass"],
         entry["shifts"])},
    {"label": "m13_parity_spelled_as_zero_minus_x_physical_band",
     "must_catch": False, "tags": ("sign_flip",), "seeding": "physical",
     "why": "THE REFUTED SPELLING, on the band where it is INDISTINGUISHABLE. "
            "`0.0f - x` agrees with `-x` in bits on every normal operand, so this "
            "row must measure NOTHING — 0 words on every case. Recorded rather than "
            "skipped because it is the control for m14: without it, m14 would show "
            "only that the spelling matters SOMEWHERE",
     "apply": lambda src, entry: replace(src, "= -f", "= 0.0f - f")},
    {"label": "m14_parity_spelled_as_zero_minus_x_signed_zeros",
     "must_catch": True, "tags": ("sign_flip",), "seeding": "signed_zero",
     "why": "THE SAME MUTANT UNDER THE +-0 LATTICE, where it is caught. "
            "`-(+0.0)` is `-0.0` and `0.0f - (+0.0f)` is `+0.0`, and the array path "
            "computes `parity * x` in NumPy, which keeps the sign. This pair is why "
            "the parity is a COMPILE-TIME specialisation on this backend rather "
            "than the runtime weight the Triton twin uses. Measured 64-140 words "
            "against m13's 0",
     "apply": lambda src, entry: replace(src, "= -f", "= 0.0f - f")},
)

#: THE HOST MUTATIONS — defects in the PLAN's own choices, not in device code. Each
#: is launch-counted for the reason the certified gates spell out: the shipped
#: kernel runs unmodified, so if it did not run at all the returned arrays would be
#: the untouched input, which differs from the oracle exactly as a caught defect
#: does. The counter is what turns that into DISARMED.
def entries_reflect_row_n_minus_two(grid: Any, family: str) -> Tuple[Any, Any]:
    near = list(symmetry.ghost_fill_axis_entries(grid, family, "near"))
    far = [dict(entry) for entry in
           symmetry.ghost_fill_axis_entries(grid, family, "far")]
    for entry in far:
        entry["reflect_row"] = int(grid.shape[entry["axis"]]) - 2
    return near, far


def entries_phase_flipped(grid: Any, family: str) -> Tuple[Any, Any]:
    near = [dict(e) for e in symmetry.ghost_fill_axis_entries(grid, family, "near")]
    far = [dict(e) for e in symmetry.ghost_fill_axis_entries(grid, family, "far")]
    for entry in near + far:
        entry["phase"] = -int(entry["phase"])
    return near, far


def entries_axis_order_reversed(grid: Any, family: str) -> Tuple[Any, Any]:
    return (list(reversed(symmetry.ghost_fill_axis_entries(grid, family, "near"))),
            list(reversed(symmetry.ghost_fill_axis_entries(grid, family, "far"))))


def entries_far_pass_dropped(grid: Any, family: str) -> Tuple[Any, Any]:
    return list(symmetry.ghost_fill_axis_entries(grid, family, "near")), []


HOST_MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"label": "m15_reflect_row_is_n_minus_two_odd_count",
     "must_catch": True, "tags": ("far", "odd_full_count"),
     "why": "`reflect_row` hard-coded to `stored - 2`. `_far_reflect_rows` is "
            "`n_full - stored + 2`, which IS `stored - 2` at an even full count and "
            "`stored - 3` at an odd one, so the wrong formula is a WHOLE CELL out "
            "on every odd-count run. Measured 192-384 words",
     "entries": entries_reflect_row_n_minus_two},
    {"label": "m16_reflect_row_is_n_minus_two_even_count",
     "must_catch": False, "tags": ("far", "even_full_count"),
     "why": "THE OTHER HALF OF m15 and the reason the matrix carries extent 2.1 at "
            "all: at an even full count the wrong formula HAPPENS TO BE RIGHT and "
            "measures 0. A gate carrying only the default extent would have armed "
            "m15, seen nothing, and reported the reflect row as untested-looking-"
            "tested",
     "entries": entries_reflect_row_n_minus_two},
    {"label": "m17_mirror_phase_flipped",
     "must_catch": True, "tags": (),
     "why": "the plane's declared parity inverted. On this backend that is a "
            "DIFFERENT COMPILED FILL rather than a different scalar, so the "
            "mutation exercises the specialisation seam itself. Measured 192-792 "
            "words",
     "entries": entries_phase_flipped},
    {"label": "m18_far_ghost_pass_dropped",
     "must_catch": True, "tags": ("far",),
     "why": "the driver's FIFTH pass per half deleted. It is the pass the residency "
            "model had no entry for until this family was built, and dropping it "
            "leaves the last stored slot carrying whatever the curl left there. "
            "Measured 192-564 words",
     "entries": entries_far_pass_dropped},
    {"label": "m19_fill_axis_order_reversed",
     "must_catch": False, "tags": ("two_axis",),
     "why": "A MEASURED NULL, WITH ITS REACHABILITY ASSERTED. The array path applies "
            "the axes in X, Y, Z order so a corner unowned on two planes carries the "
            "PRODUCT of both parities; reversing the order must therefore matter — "
            "and measured on the real fold it does NOT, because each axis's launch "
            "writes and reads only its own columns. Recorded as a null rather than "
            "dropped, and the case it runs on is checked to be one where the "
            "ordering is genuinely exercised: two folded axes, DIFFERING phases, "
            "and a corner written by both launches",
     "entries": entries_axis_order_reversed},
)

#: THE STEP-ORDER MUTATIONS — the class that lives ONLY in a complete step, which
#: is the fold's fourth risk: the mirror plane is written by one pass and read by
#: the next.
ORDER_MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"label": "m20_ghost_consumed_before_it_is_written",
     "must_catch": True,
     "why": "THE FOLD'S FOURTH RISK, planted. `update_H` moved AHEAD of the ghost "
            "fill, so the constitutive sub-step consumes the mirror plane one step "
            "stale. Every sub-step kernel is byte-perfect in this run and the step "
            "is still wrong, which is exactly what a per-sub-step gate cannot see. "
            "Measured caught at STEP 0 with 1,152-5,424 words",
     "order": ("step_B", "update_H", "fill_B_near", "fill_B_far",
               "step_D", "update_E", "fill_D_near", "fill_D_far")},
    {"label": "m21_far_pass_before_near_pass",
     "must_catch": False,
     "why": "A MEASURED NULL. The near pass writes stored cell 0 of the SHIFT-0 "
            "components and the far pass writes the last slot of the SHIFT-1 ones, "
            "so on any single axis the two touch DISJOINT component sets and their "
            "order cannot matter. Recorded because the driver nevertheless keeps "
            "them apart for a different reason — `zero_metal_*` runs BETWEEN them "
            "(driver.py:3286/:3301) — and a reader who saw the passes reordered "
            "freely here might conclude the separation is decorative. It is not; "
            "the wall seam is refused by name, see leg 9",
     "order": ("step_B", "fill_B_far", "fill_B_near", "update_H",
               "step_D", "fill_D_far", "fill_D_near", "update_E")},
)


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Every mutation armed, launch-counted and three-valued."""
    harness = kit.MutationHarness(payload, out, key="mutations")

    # --- curl source mutations ---------------------------------------------
    for entry in CURL_MUTATIONS:
        caught = ran = 0
        missed = False
        counters: List[kit.Counter] = []
        for label, _ in CASES:
            probe_fields, probe_pml = build(label)
            if not set(entry["tags"]).issubset(set(tags(probe_fields, probe_pml))):
                continue
            for sub_step in ("step_B", "step_D"):
                # A FRESH GRID PER (case, sub_step). Reusing one would leave the
                # second sub-step starting from the FIRST mutant's output — both
                # sides would still see the same input, so the comparison would
                # stay valid, but the second row's inputs would be a mutant's
                # residue and a reader could not attribute what it measured.
                fields, pml = build(label)
                codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
                spec = launch.SUB_STEPS[sub_step]
                shipped = symmetry.folded_curl_source(codes, spec["backward"])
                mutated = entry["apply"](shipped, codes, spec["backward"])
                if mutated is None or mutated == shipped:
                    missed = True
                    continue
                names = curl_names(sub_step)
                before, after = oracle(
                    fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
                kit.assert_moved(moved_words(before, after),
                                 f"{entry['label']}/{label}/{sub_step}", floor=500)
                run_curl(fields, pml, sub_step, source=mutated, counters=counters)
                ran += 1
                caught += int(bool(divergence(fields, after)))
        launches = sum(counter.launches for counter in counters)
        harness.record(entry["label"],
                       harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, entry["must_catch"], entry["why"],
                       {"kind": "curl-source", "tags": list(entry["tags"])})

    # --- fill source mutations ---------------------------------------------
    for entry in FILL_MUTATIONS:
        caught = ran = 0
        missed = False
        counters = []
        for label, _ in CASES:
            fields, pml = build(label)
            if not set(entry["tags"]).issubset(set(tags(fields, pml))):
                continue
            if entry.get("seeding") == "signed_zero":
                counts = seed_signed_zeros(fields)
                kit.assert_census_floor(counts["negative_zero"],
                                        f"{entry['label']}/{label}", floor=64)
            for family in ("B", "D"):
                names = fill_names(family)
                near_fn = getattr(stepping, "fill_symmetry_bc_" + family)
                far_fn = getattr(stepping, "fill_folded_far_ghosts_" + family)
                before, after = oracle(
                    fields, lambda: (near_fn(fields), far_fn(fields)), names)
                kit.assert_moved(moved_words(before, after),
                                 f"{entry['label']}/{label}/{family}", floor=16)
                _, planted = run_fill(fields, family, text=entry["apply"],
                                      counters=counters)
                if not planted:
                    missed = True
                    continue
                ran += 1
                caught += int(bool(divergence(fields, after)))
        launches = sum(counter.launches for counter in counters)
        harness.record(entry["label"],
                       harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, entry["must_catch"], entry["why"],
                       {"kind": "fill-source", "tags": list(entry["tags"]),
                        "seeding": entry.get("seeding", "physical")})

    # --- host mutations ----------------------------------------------------
    for entry in HOST_MUTATIONS:
        caught = ran = 0
        counters = []
        # ONE RECORD PER CASE, not one for the family. The reachability check runs
        # on every case the mutation is scoped to, and keeping only the last one
        # would leave the artifact unable to show that the OTHER case was checked.
        reachability: List[Dict[str, Any]] = []
        for label, _ in CASES:
            fields, pml = build(label)
            if not set(entry["tags"]).issubset(set(tags(fields, pml))):
                continue
            if entry["label"] == "m19_fill_axis_order_reversed":
                reached = dict(axis_order_reachability(fields, "B"), case=label)
                reachability.append(reached)
                assert reached["folded_axes"] >= 2, reached
                assert reached["distinct_phases"], reached
                assert reached["corner_written_twice"], (
                    "the axis-order null is only a measurement if a cell is "
                    "actually written by BOTH launches; on this case it is not, so "
                    "the null would be an artefact of the case rather than of the "
                    "fill", reached)
            for family in ("B", "D"):
                names = fill_names(family)
                near_fn = getattr(stepping, "fill_symmetry_bc_" + family)
                far_fn = getattr(stepping, "fill_folded_far_ghosts_" + family)
                before, after = oracle(
                    fields, lambda: (near_fn(fields), far_fn(fields)), names)
                kit.assert_moved(moved_words(before, after),
                                 f"{entry['label']}/{label}/{family}", floor=16)
                run_fill(fields, family, entries=entry["entries"],
                         counters=counters)
                ran += 1
                caught += int(bool(divergence(fields, after)))
        launches = sum(counter.launches for counter in counters)
        harness.record(entry["label"],
                       harness.verdict(False, ran, launches, caught),
                       launches, caught, ran, entry["must_catch"], entry["why"],
                       {"kind": "host", "tags": list(entry["tags"]),
                        "reachability": reachability})

    # --- step-order mutations ----------------------------------------------
    for entry in ORDER_MUTATIONS:
        caught = ran = 0
        first_steps: List[Optional[int]] = []
        launches = 0
        for label, _ in CASES:
            fields, pml = build(label)
            reference_fields, reference_pml = build(label)
            residency = device.Residency()
            plans = composed_plans(fields, pml, residency)
            residency.sync_in()
            first: Optional[int] = None
            for step in range(4):
                reference_step(reference_fields, reference_pml)
                walk(plans, entry["order"])
                residency.sync_out()
                bad = divergence(fields, snapshot(reference_fields))
                if bad and first is None:
                    first = step
            ran += 1
            caught += int(first is not None)
            first_steps.append(first)
            launches += sum(plan.launches for plan in plans.values())
        harness.record(entry["label"],
                       harness.verdict(False, ran, launches, caught),
                       launches, caught, ran, entry["must_catch"], entry["why"],
                       {"kind": "step-order", "order": list(entry["order"]),
                        "first_divergent_step_per_case": first_steps})


def axis_order_reachability(fields: Any, family: str) -> Dict[str, Any]:
    """Is the fill's X, Y, Z order EXERCISED on this grid?

    A MEASURED NULL IS ONLY A MEASUREMENT IF THE THING IT MEASURES WAS REACHABLE, and
    "reversing the axis order changed nothing" is the easiest null in this family to
    obtain by accident — one folded axis makes every ordering question trivially
    empty. Three conditions make the order observable at all and all three are
    checked: two axes folded; DIFFERENT phases on them (so a product of parities
    could not collapse to one); and at least one CELL written by two different
    launches, counted here by replaying each launch's write plane onto a per-target
    counter.
    """
    codes, _ = symmetry.folded_axis_kinds(fields.grid, None)
    codes = tuple(int(code) for code in (codes or ()))
    folded_axes = [axis for axis, code in enumerate(codes)
                   if code in symmetry.MIRROR_CODES]
    phases = [int(fields.grid.mirror_phase(axis)) for axis in folded_axes]
    shape = tuple(int(n) for n in fields.grid.shape)
    counters = [np.zeros(shape, np.int32) for _ in fill_names(family)]
    for pass_name in symmetry.FILL_PASSES:
        wanted = 0 if pass_name == "near" else 1
        for entry in symmetry.ghost_fill_axis_entries(fields.grid, family,
                                                      pass_name):
            axis = int(entry["axis"])
            index: List[Any] = [slice(None)] * 3
            index[axis] = 0 if pass_name == "near" else shape[axis] - 1
            for slot, shift in enumerate(entry["shifts"]):
                if int(shift) == wanted:
                    counters[slot][tuple(index)] += 1
    doubled = int(sum(int(np.count_nonzero(counter > 1)) for counter in counters))
    return {"folded_axes": len(folded_axes), "phases": phases,
            "distinct_phases": len(set(phases)) > 1,
            "cells_written_twice": doubled,
            "corner_written_twice": doubled > 0,
            "targets": list(fill_names(family))}


# ---------------------------------------------------------------------------
# LEG 9 — composition: the six slots, the foreign arms, and the named refusal
# ---------------------------------------------------------------------------

FOLDED_SLOTS = ("step_B", "step_D", "update_H", "update_E", "fill_B", "fill_D")


def leg_composition(payload: Dict[str, Any], out: str) -> None:
    """What the SHIPPED registry does on a folded grid, and what it refuses.

    THREE CHECKS:

    1. every registered arm is asked about every folded slot, and on each slot
       EXACTLY ONE admits — this family's. Two admitters is not a tie the composer
       breaks, it is an UNSELECTED slot that falls to the array path, so a foreign
       arm admitting here would be a silent coverage LOSS as well as a disjointness
       defect;
    2. the family is VISIBLE to the composer. A family absent from
       ``registry.FAMILY_MODULES`` is invisible to ``plan_step`` — a silent coverage
       loss with no error anywhere — so the module list is asserted against the
       package's own contents;
    3. THE NAMED REFUSAL. A folded PERIODIC grid with a live ``zero_metal`` axis is
       refused BY NAME rather than mis-stepped: ``zero_metal_*`` runs BETWEEN the
       two fill passes and the far pass reads a plane the wall clear touches. The
       refusal is scoped — the curl and constitutive slots still compose — and it is
       checked here in both halves, because a refusal that also killed the curl
       would be a coverage loss wearing a refusal's clothes.
    """
    from meep_gpu.metal_kernels import arms as arm_table  # noqa: PLC0415
    from meep_gpu.metal_kernels import registry  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    started = time.time()

    for label, _ in CASES:
        fields, pml = build(label)
        residency = device.Residency()
        admitted: Dict[str, List[str]] = {slot: [] for slot in FOLDED_SLOTS}
        gated_out: Dict[str, List[str]] = {slot: [] for slot in FOLDED_SLOTS}
        for slot in FOLDED_SLOTS:
            for spec in arm_table.registered(slot):
                context = arm_table.StepContext(fields=fields, pml=pml,
                                                residency=residency)
                if spec.gate is not None and not spec.gate(context):
                    gated_out[slot].append(spec.family)
                    continue
                try:
                    verdict = spec.coverage(context, slot)
                except Exception as exc:  # noqa: BLE001 - a raise IS the finding
                    raise AssertionError(
                        f"{spec.family} raised on {slot}: a predicate must return "
                        f"a NAMED REFUSAL, never raise into the composer") from exc
                if verdict.covered and spec.wired:
                    admitted[slot].append(spec.family)
        row = {"case": label, "tags": list(tags(fields, pml)),
               "admitted_by_slot": admitted, "gated_out_by_slot": gated_out,
               "arms_consulted": {slot: len(arm_table.registered(slot))
                                  for slot in FOLDED_SLOTS},
               "foreign_admitters": {slot: [f for f in families
                                            if f != symmetry.FAMILY]
                                     for slot, families in admitted.items()},
               "differing": {}}
        rows.append(row)
        log(f"[leg9 composition] {label:<22} admitted={admitted} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["composition"] = rows
        save(payload, out)
        for slot, families in admitted.items():
            assert families == [symmetry.FAMILY], (
                f"{label}/{slot}: exactly one arm must admit and it must be this "
                f"family; {families} leaves the slot UNSELECTED and it falls to the "
                f"array path", row)

    assert symmetry.__name__.rsplit(".", 1)[-1] in registry.FAMILY_MODULES, (
        "this family is absent from registry.FAMILY_MODULES, so plan_step cannot "
        "see it at all — a silent coverage loss with no error anywhere")

    # --- the named refusal --------------------------------------------------
    fields, pml = matrix.folded(boundaries={"x": "metallic"}, depth=1.2)
    residency = device.Residency()
    walls = zero_metal_axes(fields.grid)
    verdict = symmetry.mirror_ghost_fill_coverage(fields, "B", residency)
    curl = symmetry.plan_folded_pml_curl(fields, pml, "step_B", residency)
    constitutive = symmetry.plan_folded_constitutive(fields, pml, "H", residency)
    row = {"case": "folded_periodic_with_a_live_wall",
           "zero_metal_axes": [bool(w) for w in walls],
           "codes": [int(c) for c in symmetry.folded_axis_kinds(fields.grid, pml)[0]],
           "fill_admitted": verdict.covered,
           "fill_refusal_reasons": list(verdict.reasons),
           "curl_still_composes": curl is not None,
           "constitutive_still_composes": constitutive is not None,
           "differing": {}}
    rows.append(row)
    log(f"[leg9 refusal] fill_admitted={verdict.covered} "
        f"curl={row['curl_still_composes']} "
        f"constitutive={row['constitutive_still_composes']} "
        f"({time.time() - started:.1f}s)")
    payload["legs"]["composition"] = rows
    save(payload, out)
    assert any(walls), "this case was meant to carry a live zero_metal wall"
    assert not verdict.covered, row
    assert verdict.reasons and "zero_metal" in " ".join(verdict.reasons), (
        "the refusal must NAME the seam it refuses; an unexplained False is "
        "indistinguishable from a predicate bug", row)
    assert curl is not None and constitutive is not None, (
        "the wall-seam refusal must be SCOPED to the fill; killing the curl and the "
        "constitutive slots too would be a coverage loss wearing a refusal's "
        "clothes", row)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

LEGS: Tuple[Tuple[str, Callable[[Dict[str, Any], str], None]], ...] = (
    ("execution", leg_execution),
    ("curl", leg_curl),
    ("fill", leg_fill),
    ("constitutive", leg_constitutive),
    ("whole_step", leg_whole_step),
    ("reduction", leg_reduction),
    ("signed_zero", leg_signed_zero),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
    ("composition", leg_composition),
)

#: Defects this gate CANNOT hold because they are byte-invisible in float32, named
#: with the reason and with what pins each instead. There is no generated-code audit
#: on this executor, so a predicted null here is a genuine limit rather than a
#: formality.
PREDICTED_NULLS = (
    ("parity_spelled_as_x_times_minus_one",
     "`-x` and `x * -1.0f` are byte-identical on this backend for every value class "
     "including NaN (measured 0/12 finite, 0/10 subnormal, 0/2 infinite, and on NaN "
     "both differ from NumPy identically), so no seeding can distinguish the two "
     "spellings",
     "test_metal_symmetry.test_the_fill_spells_the_sign_fields_mirror_parity_says "
     "pins the emitted text by character"),
    ("fill_pass_order_within_one_axis",
     "the near pass writes the shift-0 components' cell 0 and the far pass the "
     "shift-1 components' last slot, so on one axis they touch disjoint arrays; "
     "m21 records the measured null and the driver's own reason for separating "
     "them is the zero_metal seam, refused by name in leg 9",
     "leg 9's named refusal, and test_metal_symmetry."
     "test_the_wall_seam_pairing_is_refused_by_name"),
    ("the_contraction_guard_was_obeyed",
     "compile_shader exposes no disassembly, so this gate can show that REMOVING "
     "the guard changes the answer (m7) but cannot show that leaving it in was "
     "honoured by the compiler. On the Triton track the analogous claim is settled "
     "with a PTX read; here there is nothing to read",
     "NOTHING PINS IT. This is the stated weakness of the Metal certification "
     "against the Triton one and it belongs in any claim made from this artifact"),
)


def certified_from_rows(legs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """A SECOND, INDEPENDENT read of the outcome, off the rows that were WRITTEN.

    The per-case assertions live inside the legs; this reads the artifact. A leg
    that recorded a nonempty ``differing`` and then failed to assert on it — the
    shape of the Triton round that stamped ``passed: true`` while every compare
    failed — is caught here rather than by the code that produced it.
    """
    def diverged(value: Any) -> bool:
        """Truthiness is NOT the question; the WORD COUNT is.

        A row that recorded ``{'fill': 0, 'curl': 0}`` is a CLEAN row, and an
        all-zero dict is truthy. Four clean ladder rows were reported as failures by
        the truthiness spelling before this — the reader doing its job on a row that
        was fine. Both ends are fixed: the leg records only nonzero entries, and this
        sums rather than tests.
        """
        if isinstance(value, dict):
            return any(int(count) > 0 for count in value.values())
        return bool(value)

    problems: List[str] = []
    for leg, rows in sorted(legs.items()):
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if diverged(row.get("differing")):
                problems.append(f"{leg}[{index}]: differing={row['differing']}")
            if row.get("first_divergent_step") is not None:
                problems.append(f"{leg}[{index}]: first divergent step "
                                f"{row['first_divergent_step']}")
            verdict = row.get("verdict")
            if verdict in ("DISARMED", "NEEDLE-MISSED"):
                problems.append(f"{leg}[{index}]: {row.get('mutation')} {verdict}")
            if row.get("must_catch") is True and row.get("caught") != row.get("ran"):
                problems.append(f"{leg}[{index}]: {row.get('mutation')} caught "
                                f"{row.get('caught')}/{row.get('ran')}")
            if row.get("must_catch") is False and row.get("caught"):
                problems.append(f"{leg}[{index}]: {row.get('mutation')} was caught "
                                f"but is a declared null")
    return (not problems), problems


def count_comparisons(legs: Dict[str, Any]) -> int:
    total = 0
    for rows in legs.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                total += int(row.get("compared", 0) or 0)
    return total


def main() -> int:
    parser = kit.argument_parser(__doc__ or "")
    arguments = parser.parse_args()
    out = arguments.out
    out_dir = os.path.dirname(os.path.abspath(out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    payload: Dict[str, Any] = {
        "gate": "metal_folded_real",
        "family": symmetry.FAMILY,
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "cases": [name for name, _ in CASES],
        "predicted_nulls": [{"defect": d, "why": w, "pinned_by": p}
                            for d, w, p in PREDICTED_NULLS],
        "legs": {},
    }

    try:
        import torch  # noqa: PLC0415

        if not torch.backends.mps.is_available():
            return kit.cannot_certify(payload, out, ["no MPS device on this host"])
    except Exception as exc:  # noqa: BLE001
        return kit.cannot_certify(payload, out, [f"torch is not importable ({exc!r})"])

    policy = payload["subnormal_policy"]
    if not policy.get("admitted"):
        return kit.cannot_certify(payload, out, policy.get("reasons", []))

    payload["provenance"] = kit.provenance(
        out_dir,
        {"stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
         "metal_kernels/symmetry.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "symmetry.py"),
         "metal_kernels/shaders.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "shaders.py"),
         "metal_kernels/templates.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "templates.py"),
         "metal_kernels/coverage.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "coverage.py"),
         "metal_kernels/launch.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "launch.py"),
         "metal_kernels/registry.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "registry.py"),
         "metal_kernels/device.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "device.py"),
         "metal_kernels/plans.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "plans.py"),
         "metal_kernels/preconditions.py": os.path.join(
             API_ROOT, "meep_gpu", "metal_kernels", "preconditions.py"),
         "triton_kernels/symmetry.py": os.path.join(
             API_ROOT, "meep_gpu", "triton_kernels", "symmetry.py"),
         "parity/metal_composition_matrix.py": os.path.join(
             HERE, "metal_composition_matrix.py"),
         "parity/gate_metal_symmetry.py": os.path.abspath(__file__)},
        kernel_sources=symmetry.enumerate_folded_sources(),
        name="provenance_gate_metal_symmetry.json")
    save(payload, out)

    legs_run = kit.run_legs(LEGS, payload, out, kit.wanted_legs(arguments.legs))
    certified, problems = certified_from_rows(payload["legs"])
    payload["problems"] = problems

    return kit.summarize(
        payload, out,
        claim=("byte-identity to stepping.py for the folded real-field curl, the "
               "two mirror ghost passes and the re-admitted certified constitutive "
               "body, on this host, subject to a CHECKED subnormal-free "
               "precondition reported as a WINDOW"),
        scope=("folded (mirror-symmetry) real float32 storage; MIRROR_PERIODIC and "
               "MIRROR_METALLIC terminations; one and two folded axes; both "
               "parities; both full-count parities; 2-D and 3-D. NOT complex "
               "storage, NOT beta, NOT off-diagonal epsilon, NOT cylindrical, NOT "
               "BFAST, NOT a registered susceptibility — each refused by name"),
        stated_weakness=(
            "THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND. compile_shader "
            "exposes no disassembly, so this gate cannot refuse a compile whose "
            "emitted code violates the policy and cannot establish that the "
            "contraction guard was OBEYED — only that removing it changes the "
            "answer. The byte legs and the mutation legs are BEHAVIOURAL: they "
            "catch a wrong answer, not a wrong instruction"),
        started=started, legs_run=legs_run,
        compared=count_comparisons(payload["legs"]),
        certified=certified,
        extra={"problems": problems,
               "whole_step_budget": WHOLE_STEP_BUDGET,
               "predicted_nulls": [d for d, _, _ in PREDICTED_NULLS]})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
