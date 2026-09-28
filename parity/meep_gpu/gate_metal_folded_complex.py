"""BYTE GATE — the mirror fold under COMPLEX64 storage, on Metal.

WHAT THIS CERTIFIES: that ``metal_kernels.folded_complex``'s three products
reproduce ``stepping.py`` WORD FOR WORD on a folded complex64 grid — the curl on
both sub-steps, the two ghost-fill passes, and the certified complex constitutive
pair re-admitted over a folded stored extent — per SUB-STEP and per COMPLETE STEP.

WHAT MAKES THIS GATE DIFFERENT FROM ITS TWO PARENTS' GATES, and it is the only
reason the family exists: the fold's parity STOPS BEING A SIGN FLIP under complex
storage. ``symmetry.mirror_ghost_fill`` spells it ``-x`` or a plain copy, exact on
this host over every value class; the array path spells ``phase * plane`` with a
PYTHON INT against a complex64 array, which numpy carries only as the full
``'FF->F'`` multiply by ``(+/-1.0, +0.0)`` WITH its zero cross terms. Leg ``parity``
measures that difference on a live folded state and REQUIRES it to be nonzero: if it
were ever zero this family should be deleted and the real fold's fill admitted on
complex storage instead, and a gate that did not say so would be certifying a family
whose reason to exist had evaporated.

THE SUBNORMAL PRECONDITION IS LOAD-BEARING HERE IN A WAY IT IS NOT FOR THE REAL
FOLD. That family records its FILL as band-safe because the fill performs no
arithmetic; making the parity a complex product makes it arithmetic, and arithmetic
flushes on MPS. The precondition is reported as a WINDOW on TWO axes, because a
census that reports a scalar licenses nothing about a run whose leading edge sweeps
the band:

* leg ``band`` sweeps the VALUE SCALE and reports which scales flush;
* leg ``whole_step`` censuses EVERY STEP and reports ``[first_step, last_step]`` per
  case, refusing any case that enters the band BY NAME — a coverage refusal, not a
  failure, whose comparisons are withheld from the certified total;
* leg ``precondition`` drives that same per-step census over a state scaled into the
  band and REQUIRES it to fire. Without it, the empty windows next door would be
  indistinguishable from a census that cannot detect anything.

THE FOLD IS MUTATED AS A FOLD, not merely as a kernel. The five defects a mirror is
prone to — mirroring the wrong half, dropping the sign flip, shifting the mirror
plane by a cell, reading the ghost before it is written, and folding the wrong axis —
each have their own arm, and the two that are MEASURED NULLS say so with the reason
that makes them null rather than being reported as catches.

THE STATED WEAKNESS, and it is this certification's one gap against the Triton
twin's: ``torch.mps.compile_shader`` exposes NO DISASSEMBLY. This gate cannot refuse
a compile whose emitted code violates the policy and cannot establish that the
contraction guard was obeyed. Every leg here is BEHAVIOURAL — it catches a wrong
answer, not a wrong instruction.

    python -u gate_metal_folded_complex.py --out results/.../gate.json
"""

from __future__ import annotations

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

# THE POLICY IS SET BEFORE ANY `meep_gpu` MODULE IS REACHED. On MPS the float32
# subnormal flush is native and has no lever, so the resolved default `keep` is NOT
# OFFERABLE and every Metal predicate refuses by name. Setting `flush` is what puts
# the claim under the CHECKED precondition rather than under a pretence.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import metal_composition_matrix as matrix  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_fields, device, folded_complex, launch, shaders, subnormal, templates,
)
from meep_gpu.metal_kernels.device import compile_source  # noqa: E402
from meep_gpu.triton_kernels import symmetry as triton_symmetry  # noqa: E402
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words
needle = kit.needle

MP = folded_complex.CODE_MIRROR_PERIODIC
MM = folded_complex.CODE_MIRROR_METALLIC

#: Everything the engine stores that a folded complex step can touch. The whole-step
#: leg compares ALL of it after every complete step: ``fu_*`` and ``f_w_*`` are STATE
#: and a kernel right for one launch and wrong forever after diverges only once they
#: accumulate.
STORED = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
          "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
          "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The driver's own order, five passes per half (driver.py:3282-3287, :3293-3302).
#: Spelled once, so the reference walk and the device walk cannot drift apart.
DRIVER_ORDER: Tuple[str, ...] = (
    "step_B", "fill_B_near", "fill_B_far", "update_H",
    "step_D", "fill_D_near", "fill_D_far", "update_E")

#: THE CASE MATRIX. Every axis is here because a mutation is reachable on one value
#: and a MEASURED NULL on the other:
#:
#:   TERMINATION    MIRROR_PERIODIC carries the top-plane mask AND the far ghost
#:                  pass; MIRROR_METALLIC carries NEITHER;
#:   PHASE          under complex storage an odd plane is the SAME compiled body
#:                  with a different PASSED coefficient word — the inverse of the
#:                  real fold, where the parity is a source constant — so the
#:                  flipped-coefficient mutation is reachable on every row;
#:   FULL COUNT     ``reflect_row`` is ``stored - 2`` at an even full count and
#:                  ``stored - 3`` at an odd one, so at the default extent the wrong
#:                  ``n - 2`` formula HAPPENS TO BE RIGHT;
#:   FOLDED AXES    a corner unowned on two planes carries the PRODUCT of both
#:                  parities, and only a MIXED-PHASE pair makes the order question
#:                  exist at all;
#:   BLOCH          THIS COMPOSITION'S OWN CASE. Three of the five non-beta folded
#:                  complex corpus rows carry a k on an UNFOLDED axis, and without
#:                  it no leg here would ever launch a folded curl that also
#:                  ROTATES. The phase is on x and the fold on y, which is the only
#:                  pairing the engine will build;
#:   DIMENSIONALITY a claim is only as strong as the words it moved, and a 2-D ghost
#:                  plane is a thin thing to certify a family on.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("periodic_even_2d", dict()),
    ("periodic_even_3d", dict(depth=1.2)),
    ("periodic_odd_3d", dict(phase=-1, depth=1.2)),
    ("metallic_even_3d", dict(boundaries={"y": "metallic"}, depth=1.2)),
    ("metallic_odd_3d", dict(phase=-1, boundaries={"y": "metallic"}, depth=1.2)),
    ("x_fold_3d", dict(axis="X", depth=1.2)),
    ("odd_full_count_3d", dict(extent=2.1, depth=1.2)),
    ("two_axis_mixed_3d", dict(axis="XY", phase=(1, -1), depth=1.2)),
    ("bloch_off_axis_3d", dict(k_point=(0.3, 0.0, 0.0), depth=1.2)),
    ("bloch_metallic_fold_3d", dict(k_point=(0.3, 0.0, 0.0),
                                    boundaries={"y": "metallic"}, depth=1.2)),
)

PROBE = folded_complex.load_expansion_probe()
EXPANSION = folded_complex.expansion_from_probe(PROBE) if PROBE else None


def build(label: str) -> Tuple[Any, Any]:
    return matrix.folded(complex_storage=True, **dict(CASES)[label])


def tags(fields: Any, pml: Any) -> Tuple[str, ...]:
    """What a case IS, derived from the ENGINE rather than typed beside the row.

    A mutation scopes itself with these. Typing them into the matrix would put the
    fold's single point of failure — the MIRROR_METALLIC / MIRROR_PERIODIC split —
    in a second place, and a scoping that disagreed with the grid would silently
    turn a reachable defect into a NEEDLE-MISSED row.
    """
    codes, _ = folded_complex.folded_axis_kinds(fields.grid, pml)
    codes = tuple(int(code) for code in (codes or ()))
    found: List[str] = ["far" if MP in codes else "no_far"]
    folded_axes = [i for i, c in enumerate(codes) if c in folded_complex.MIRROR_CODES]
    found.append("two_axis" if len(folded_axes) >= 2 else "one_axis")
    if len({int(fields.grid.mirror_phase(a)) for a in folded_axes}) > 1:
        found.append("mixed_phase")
    rows = folded_complex._far_reflect_rows(fields.grid) or (None, None, None)
    for axis, code in enumerate(codes):
        if code == MP and rows[axis] is not None:
            found.append("odd_full_count"
                         if int(fields.grid.shape[axis]) - int(rows[axis]) == 3
                         else "even_full_count")
    found.append("three_d" if int(fields.grid.shape[2]) > 1 else "two_d")
    if getattr(fields.grid, "has_bloch", False):
        found.append("bloch")
    # DOES ANY COEFFICIENT THIS CASE PASSES ACTUALLY CARRY A NEGATION? The near word
    # is `+phase` and the far word is `-phase`, so an EVEN fold with NO far pass
    # passes `+1` everywhere and "drop the sign flip" is the IDENTITY there — not a
    # defect that went uncaught. Measured: without this tag that mutation reports
    # CAUGHT 8/10, and the two misses are exactly `metallic_even_3d` and
    # `bloch_metallic_fold_3d`. Scoping on the tag turns a misleading 8/10 into an
    # honest 8/8 over the cases where the defect EXISTS.
    if any(entry["coefficient"][0] < 0
           for family in ("B", "D") for pass_name in ("near", "far")
           for entry in folded_complex.fill_axis_entries(
               fields.grid, family, pass_name)):
        found.append("signed_parity")
    return tuple(found)


def state_census(fields: Any) -> int:
    """How many stored words sit in the subnormal band, over EVERY stored volume.

    Read off the BITS by :func:`subnormal.census` — a value comparison would be
    answered by the very flushing this counts.
    """
    return sum(subnormal.census(getattr(fields, name))
               for name in STORED if getattr(fields, name, None) is not None)


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
           names: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """Run the ARRAY PATH, capture what it wrote, and put the state back.

    Returns (before, after, moved). ``moved`` is the vacuity floor every leg
    asserts: zero-init is a fixed point of half of this family's work, and a no-op
    agreeing with a no-op is trivially identical.
    """
    before = snapshot(fields, names)
    apply()
    after = snapshot(fields, names)
    restore(fields, before)
    moved = sum(differing(before[n], after[n]) for n in names)
    return before, after, moved


def compared_words(state: Dict[str, Any]) -> int:
    return sum(int(words(array).size) for array in state.values())


def divergence(fields: Any, after: Dict[str, Any]) -> Dict[str, int]:
    return {name: differing(getattr(fields, name), array)
            for name, array in after.items()}


# ---------------------------------------------------------------------------
# Device runners — ONE route, so the bytes the gate certifies are the engine's
# ---------------------------------------------------------------------------

def run_curl(fields: Any, pml: Any, sub_step: str,
             functions: Optional[Dict[str, Any]] = None) -> Any:
    """Launch the folded complex curl through the ENGINE route (or a mutant)."""
    residency = device.Residency()
    if functions is None:
        plan = folded_complex.plan_folded_complex_pml_curl(
            fields, pml, sub_step, residency, probe=PROBE)
        if plan is None:
            raise AssertionError(
                folded_complex.folded_complex_composition_curl_coverage(
                    fields, pml, sub_step, residency, PROBE).reasons)
    else:
        codes, _ = folded_complex.folded_axis_kinds(fields.grid, pml)
        kinds = stepping._boundary_kinds(fields.grid, pml)
        phases = complex_fields.bloch_phase_table(fields.grid, kinds)
        arrays = {n: getattr(fields, n) for n in STORED
                  if getattr(fields, n, None) is not None}
        flat = {f"{stem}_{axis}": getattr(
            pml, f"{stem}_{axis}{launch.SUB_STEPS[sub_step]['suffix']}")
            for axis in "xyz" for stem in ("kms", "sinv")}
        plan = folded_complex.plan_folded_complex_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, phases,
            fields.grid.dt / fields.grid.dx, EXPANSION, residency,
            functions=functions)
    residency.sync_in()
    plan.run()
    residency.sync_out()
    return plan


def build_fill(fields: Any, family: str, residency: Any,
               transform: Optional[Callable[[str, List[Dict[str, Any]],
                                             List[Dict[str, Any]]], None]] = None,
               source_text: Optional[Callable[[str, Dict[str, Any]],
                                              Optional[str]]] = None) -> Any:
    """Build a fill plan, optionally with a mutated ENTRY LIST or a mutated SOURCE.

    Two mutation seams because the family has two kinds of defect: the HOST kind
    (a wrong reflect row, a flipped coefficient word, a reversed axis order) lives
    in the entry list, and the KERNEL kind (a plane-wise sign flip, a synthesised
    coefficient) lives in the emitted text. Both go through the SAME plan builder,
    so a mutant is launched exactly the way the shipped plan is.
    """
    near = [dict(e) for e in folded_complex.fill_axis_entries(fields.grid, family,
                                                              "near")]
    far = [dict(e) for e in folded_complex.fill_axis_entries(fields.grid, family,
                                                             "far")]
    if transform is not None:
        transform(family, near, far)
    functions = None
    if source_text is not None:
        functions = {shaders.CONTRACT_OFF: {}}
        for entry in near + far:
            key = (f"axis{entry['axis']}/{entry['pass']}/"
                   f"{''.join(str(s) for s in entry['shifts'])}")
            if key in functions[shaders.CONTRACT_OFF]:
                continue
            shipped = folded_complex.folded_mirror_fill_complex_source(
                entry["axis"], entry["pass"], entry["shifts"], EXPANSION)
            text = source_text(key, entry)
            functions[shaders.CONTRACT_OFF][key] = compile_source(
                text if text is not None else shipped).folded_mirror_fill_complex
        # NO SEPARATE LAUNCH COUNTER HERE: `FoldedComplexFillPlan` counts its own
        # launches and the harness reads THAT, so a mutant that never ran is still
        # classified DISARMED rather than passing as uncaught.
        return folded_complex.plan_folded_complex_fill_from_arrays(
            family, "fill_" + family,
            {n: getattr(fields, n) for n in fill_names(family)}, near, far,
            EXPANSION, residency, functions=functions)
    return folded_complex.plan_folded_complex_fill_from_arrays(
        family, "fill_" + family,
        {n: getattr(fields, n) for n in fill_names(family)}, near, far,
        EXPANSION, residency)


def run_fill(fields: Any, family: str, **kwargs: Any) -> Any:
    residency = device.Residency()
    plan = build_fill(fields, family, residency, **kwargs)
    residency.sync_in()
    plan.run_near()
    plan.run_far()
    residency.sync_out()
    return plan


def run_constitutive(fields: Any, pml: Any, side: str) -> Any:
    residency = device.Residency()
    plan = folded_complex.plan_folded_complex_constitutive(
        fields, pml, side, residency, probe=PROBE)
    if plan is None:
        raise AssertionError(folded_complex.folded_complex_constitutive_coverage(
            fields, pml, side, residency, PROBE).reasons)
    residency.sync_in()
    plan.run()
    residency.sync_out()
    return plan


def reference_step(fields: Any, pml: Any) -> None:
    """ONE COMPLETE DRIVER STEP on the array path — FIVE passes per half."""
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
        "step_B": folded_complex.plan_folded_complex_pml_curl(
            fields, pml, "step_B", residency, probe=PROBE),
        "fill_B": folded_complex.plan_folded_complex_fill(
            fields, "B", "fill_B", residency, probe=PROBE),
        "update_H": folded_complex.plan_folded_complex_constitutive(
            fields, pml, "H", residency, probe=PROBE),
        "step_D": folded_complex.plan_folded_complex_pml_curl(
            fields, pml, "step_D", residency, probe=PROBE),
        "fill_D": folded_complex.plan_folded_complex_fill(
            fields, "D", "fill_D", residency, probe=PROBE),
        "update_E": folded_complex.plan_folded_complex_constitutive(
            fields, pml, "E", residency, probe=PROBE),
    }
    missing = [name for name, plan in plans.items() if plan is None]
    assert not missing, f"no plan for {missing}"
    return plans


def walk(plans: Dict[str, Any]) -> None:
    for name in DRIVER_ORDER:
        if name.endswith("_near"):
            plans[name[:-5]].run_near()
        elif name.endswith("_far"):
            plans[name[:-4]].run_far()
        else:
            plans[name].run()


# ---------------------------------------------------------------------------
# LEG execution — provenance, the bound arm, and the compile sweep
# ---------------------------------------------------------------------------

def leg_execution(payload: Dict[str, Any], out: str) -> None:
    """What was hashed, what arm was bound, and does every specialisation BUILD.

    A specialisation that fails to COMPILE is a crash at plan time on a
    configuration nobody swept, so the whole REACHABLE product is built here — the
    curl restricted to the phase flags each code quadruple can legally carry (a
    metallic or folded axis cannot phase), plus the fill.
    """
    sources = folded_complex.enumerate_folded_complex_sources(EXPANSION)
    kit.provenance(os.path.dirname(out), {
        "folded_complex.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/folded_complex.py"),
        "complex_fields.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/complex_fields.py"),
        "symmetry.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/symmetry.py"),
        "templates.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/templates.py"),
        "shaders.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/shaders.py"),
        "stepping.py": os.path.join(API_ROOT, "meep_gpu/stepping.py"),
        "gate": os.path.abspath(__file__),
    }, kernel_sources=sources, name="provenance_gate.json")

    built = 0
    for mode in shaders.CONTRACT_MODES:
        for source in folded_complex.enumerate_folded_complex_sources(
                EXPANSION, mode).values():
            compile_source(source)
            built += 1
    kit.assert_moved(built, "no specialisation compiled", floor=2 * len(sources))

    payload["legs"]["execution"] = {
        "expansion": EXPANSION,
        "probe_patterns": (PROBE or {}).get("patterns"),
        "specialisations": len(sources),
        "compiled": built,
        "environment": ENVIRONMENT,
        "cases": [label for label, _ in CASES],
        "no_generated_code_audit": (
            "torch.mps.compile_shader exposes no disassembly: this gate cannot "
            "refuse a compile whose emitted code violates the policy, nor establish "
            "that the contraction guard was obeyed. Every leg is behavioural."),
    }
    save(payload, out)
    log(f"[execution] arm={EXPANSION} specialisations={len(sources)} "
        f"compiled={built}")


# ---------------------------------------------------------------------------
# LEG curl — per sub-step, which is where the two ownership masks live
# ---------------------------------------------------------------------------

def leg_curl(payload: Dict[str, Any], out: str) -> None:
    """SUB-STEP granularity. The masks are INVISIBLE at whole-step granularity.

    The driver's fill passes overwrite exactly the planes the two ownership masks
    protect, so a whole-step check certifies a mask-less kernel as correct. This is
    the leg that can fail on a mask.
    """
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        for sub_step in ("step_B", "step_D"):
            names = curl_names(sub_step)
            before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{sub_step} reference barely moved",
                             floor=64)
            plan = run_curl(fields, pml, sub_step)
            assert plan.launches == 1, (label, sub_step, plan.launches)
            per = divergence(fields, after)
            rows.append({"case": label, "sub_step": sub_step, "tags": tags(fields, pml),
                         "bc": list(plan.bc), "phased": list(plan.phased),
                         "moved": moved, "compared": compared_words(after),
                         "differing": sum(per.values()), "per_target": per})
            log(f"[curl] {label:<24} {sub_step} bc={plan.bc} phased={plan.phased} "
                f"moved={moved} differing={sum(per.values())}")
            restore(fields, before)
    payload["legs"]["curl"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG fill — the mirror plane itself, and THE ONE ARITHMETIC DELTA
# ---------------------------------------------------------------------------

def leg_fill(payload: Dict[str, Any], out: str) -> None:
    """Both passes, in the driver's own X, Y, Z order, against both array passes."""
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        for family, near_fill, far_fill in (
                ("B", stepping.fill_symmetry_bc_B,
                 stepping.fill_folded_far_ghosts_B),
                ("D", stepping.fill_symmetry_bc_D,
                 stepping.fill_folded_far_ghosts_D)):
            names = fill_names(family)

            def apply() -> None:
                near_fill(fields)
                far_fill(fields)

            before, after, moved = oracle(fields, apply, names)
            kit.assert_moved(moved, f"{label}/{family} the fill wrote nothing")
            plan = run_fill(fields, family)
            expected = len(plan.near) + len(plan.far)
            assert plan.launches == expected > 0, (label, family, plan.launches)
            per = divergence(fields, after)
            rows.append({"case": label, "family": family, "tags": tags(fields, pml),
                         "near_axes": [e["axis"] for e in plan.near],
                         "far_axes": [e["axis"] for e in plan.far],
                         "replaces": list(plan.replaces_sub_steps),
                         "moved": moved, "compared": compared_words(after),
                         "differing": sum(per.values()), "per_target": per})
            log(f"[fill] {label:<24} {family} near={[e['axis'] for e in plan.near]} "
                f"far={[e['axis'] for e in plan.far]} moved={moved} "
                f"differing={sum(per.values())}")
            restore(fields, before)
    payload["legs"]["fill"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG constitutive — no new device code, so the claim is the ADMISSION's
# ---------------------------------------------------------------------------

def leg_constitutive(payload: Dict[str, Any], out: str) -> None:
    """The CERTIFIED complex body over a folded stored extent.

    This family emits NO constitutive source at all, and that is asserted rather
    than described: ``enumerate_folded_complex_sources`` carries no constitutive
    label, so a future edit that added a body here would be visible in the count
    the execution leg records.
    """
    assert not any("constitutive" in label for label in
                   folded_complex.enumerate_folded_complex_sources(EXPANSION)), (
        "this family emitted a constitutive source: it is supposed to re-admit "
        "complex_fields.bloch_constitutive_step, not replace it")
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        for side, sub_step in (("H", "update_H"), ("E", "update_E")):
            names = constitutive_names(side)
            before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{side} reference barely moved", floor=64)
            plan = run_constitutive(fields, pml, side)
            assert plan.launches == 1, (label, side, plan.launches)
            assert isinstance(plan, complex_fields.ComplexConstitutivePlan), type(plan)
            per = divergence(fields, after)
            rows.append({"case": label, "side": side, "moved": moved,
                         "compared": compared_words(after),
                         "differing": sum(per.values()), "per_target": per,
                         "plan_class": type(plan).__name__})
            log(f"[constitutive] {label:<24} {side} moved={moved} "
                f"differing={sum(per.values())}")
            restore(fields, before)
    payload["legs"]["constitutive"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG whole_step — the only one that can see the fold's fourth risk
# ---------------------------------------------------------------------------

def leg_whole_step(payload: Dict[str, Any], out: str, budget: int = 6) -> None:
    """Per COMPLETE STEP, reporting the FIRST DIVERGENT STEP.

    Three failure classes live ONLY here — a STALE MIRROR, a SEAM, and an
    ACCUMULATING AUXILIARY — and the fold adds a fourth: its ghost plane is written
    by one pass and READ by the next, so a fold bug can be byte-perfect per sub-step
    and wrong per step. Every slot's launch counter is asserted, because a slot that
    passes by NOT EXECUTING is the hollow pass this discipline exists to prevent.

    THE SUBNORMAL CENSUS IS TAKEN PER STEP AND REPORTED AS A WINDOW, and that shape
    is not cosmetic. Band entry is a RUN-and-WINDOW fact rather than a family fact:
    measured on the chi3 round, a Q~20 narrow-band Gaussian turn-on drags the leading
    edge through the whole band — FIRST subnormal at step 55, LAST at step 3,726,
    then clean for 16,274 more steps. A row recording a scalar count cannot be read
    for what it covers. A case whose census fires is REFUSED BY NAME and its
    comparisons are withheld from the certified total: that is a COVERAGE REFUSAL,
    not a failure, and :func:`leg_precondition` is what proves the detector fires.
    """
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        reference_fields, reference_pml = build(label)
        residency = device.Residency()
        plans = composed_plans(fields, pml, residency)
        residency.sync_in()

        first_divergent = None
        per_step: List[int] = []
        census_per_step: List[int] = []
        fired: List[int] = []
        for step in range(budget):
            reference_step(reference_fields, reference_pml)
            walk(plans)
            residency.sync_out()
            per = {name: differing(getattr(fields, name),
                                   getattr(reference_fields, name))
                   for name in STORED if getattr(fields, name, None) is not None}
            total = sum(per.values())
            per_step.append(total)
            count = state_census(reference_fields)
            census_per_step.append(count)
            if count:
                fired.append(step)
            if total and first_divergent is None:
                first_divergent = {"step": step,
                                   "targets": {k: v for k, v in per.items() if v}}

        evolved = sum(differing(np.zeros_like(getattr(reference_fields, n)),
                                getattr(reference_fields, n))
                      for n in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"))
        kit.assert_moved(evolved, f"{label} whole-step state never evolved", floor=64)
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            assert plans[slot].launches == budget, (label, slot)
        for slot in ("fill_B", "fill_D"):
            plan = plans[slot]
            assert plan.launches == budget * (len(plan.near) + len(plan.far))
            assert len(plan.near) > 0, (label, slot)
        refused = bool(fired)
        # A REFUSAL IS NOT A PASS BY ANOTHER NAME. The byte claim is only withheld
        # where the precondition failed; where it held, a divergence is still a
        # failure and still stops the leg here.
        if not refused:
            assert first_divergent is None, (label, first_divergent, per_step)
        window = {"first_step": fired[0] if fired else None,
                  "last_step": fired[-1] if fired else None,
                  "steps_censused": budget,
                  "subnormal_words_per_step": census_per_step}
        rows.append({"case": label, "tags": tags(fields, pml), "budget": budget,
                     "per_step_differing": per_step,
                     "first_divergent": first_divergent,
                     "evolved_words": evolved,
                     "subnormal_window": window,
                     "refused": refused,
                     "refusal": (f"{label}: REFUSED (subnormal precondition) — the "
                                 f"census fired on steps {fired[0]}..{fired[-1]} of "
                                 f"{budget}; every arithmetic claim on this backend "
                                 f"rides a CHECKED subnormal-free precondition and "
                                 f"this case does not meet it") if refused else None,
                     "compared": 0 if refused else
                                 compared_words(snapshot(reference_fields)) * budget,
                     "launches": {k: v.launches for k, v in plans.items()}})
        log(f"[whole_step] {label:<24} {budget} steps, "
            f"first_divergent={None if first_divergent is None else first_divergent['step']}, "
            f"evolved={evolved} subnormal_window="
            f"[{window['first_step']}, {window['last_step']}]"
            f"{' REFUSED' if refused else ''}")
    payload["legs"]["whole_step"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG parity — THE COMPOSITION'S OWN DELTA, and it must be nonzero
# ---------------------------------------------------------------------------

SIGN_ZERO_PLANTS = np.array(
    [complex(0.0, 0.0), complex(-0.0, -0.0), complex(0.0, -0.0), complex(-0.0, 0.0),
     complex(1.5, -0.0), complex(-0.0, 1.5)], dtype=np.complex64)


def plant_signed_zeros(fields: Any, names: Sequence[str]) -> int:
    """SPRINKLE all four signed-zero sign combinations ON TOP of the random state.

    TWO PROPERTIES ARE BOTH REQUIRED AND THE FIRST VERSION OF THIS HELPER HAD ONLY
    ONE. The class must be present — the divergence this family exists for lives
    entirely in signed zeros — AND NEIGHBOURING ROWS MUST STILL DIFFER, because the
    reflect-row mutation is "read one row over" and a plant that made rows identical
    turns a real defect into a measured null. Measured while writing this gate: a
    whole-row pattern of period 6 rolled by one row per step moved 12 elements on a
    grid with ``nz = 12``, which is 2 full periods — rows 10 and 11 came out
    BITWISE IDENTICAL (0 of 384 words differing) and ``reflect_row_n_minus_two``
    reported CAUGHT 0/1 against a kernel that was reading the wrong plane.

    So this writes a SPARSE, row-dependent scatter over the untouched random volume:
    the class is reachable and the row-to-row variation the other needles need
    survives.
    """
    planted = 0
    for offset, name in enumerate(names):
        volume = getattr(fields, name)
        nx, ny, nz = volume.shape
        for row in range(ny):
            for index, value in enumerate(SIGN_ZERO_PLANTS):
                volume[(row * 3 + index * 5 + offset) % nx, row,
                       (row * 2 + index * 7 + offset) % nz] = value
                planted += 1
    return planted


def sign_bit_spelling(plane: Any, parity: int) -> Any:
    """The REAL fold's certified spelling, by WORD: ``-x`` or a plain copy."""
    raw = np.ascontiguousarray(plane, dtype=np.complex64).view(np.uint32).copy()
    if int(parity) == -1:
        raw = raw ^ np.uint32(0x80000000)
    return raw.view(np.float32).view(np.complex64).reshape(plane.shape)


def leg_parity(payload: Dict[str, Any], out: str) -> None:
    """THE WHOLE REASON THIS FAMILY EXISTS, measured and REQUIRED to be nonzero.

    Apply the real fold's sign-bit spelling to the same fill under complex64 and
    count the words it misses. If this were ever zero, ``folded_complex`` should be
    deleted and ``symmetry.mirror_ghost_fill`` admitted on complex storage instead.
    """
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        names = fill_names("B")
        plant_signed_zeros(fields, names)
        before = snapshot(fields, names)
        stepping.fill_symmetry_bc_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        after = snapshot(fields, names)
        restore(fields, before)
        moved = sum(differing(before[n], after[n]) for n in names)
        kit.assert_moved(moved, f"{label} the array path's fill wrote nothing")

        codes, _ = folded_complex.folded_axis_kinds(fields.grid, None)
        reflect = folded_complex._far_reflect_rows(fields.grid) or (None, None, None)
        candidate = {n: np.array(before[n], copy=True) for n in names}
        for axis, code in enumerate(int(c) for c in codes):
            if code not in folded_complex.MIRROR_CODES:
                continue
            declared = int(fields.grid.mirror_phase(axis))
            for name in names:
                shift = triton_symmetry.TARGET_IYEE[name][axis]
                volume = candidate[name]
                index: List[Any] = [slice(None)] * 3
                source = list(index)
                if shift == 0:
                    source[axis] = folded_complex.MIRROR_SOURCE_INDEX
                    index[axis] = 0
                    volume[tuple(index)] = sign_bit_spelling(
                        volume[tuple(source)], declared)
                elif code == MP:
                    source[axis] = int(reflect[axis])
                    index[axis] = -1
                    volume[tuple(index)] = sign_bit_spelling(
                        volume[tuple(source)], -declared)
        misses = sum(differing(candidate[n], after[n]) for n in names)
        assert misses > 0, (
            label, "the real fold's sign-bit spelling reproduced the array path on "
            "every word: this family's whole reason to exist is unmeasured here")
        rows.append({"case": label, "tags": tags(fields, pml), "moved": moved,
                     "sign_bit_spelling_misses": misses})
        log(f"[parity] {label:<24} the real fold's spelling misses {misses} words "
            f"(array path moved {moved})")
    payload["legs"]["parity"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG band — the precondition, reported as a WINDOW rather than a scalar
# ---------------------------------------------------------------------------

def leg_band(payload: Dict[str, Any], out: str) -> None:
    """The subnormal reach, per scale, with the FIRST and LAST case that fired.

    ``symmetry.py`` records the REAL fold's fill as band-safe because it performs no
    arithmetic. That does not survive the composition. A census that reported a
    scalar would licence nothing about a run whose leading edge sweeps the band, so
    the window is what is recorded.
    """
    module = compile_source(folded_complex.folded_mirror_fill_complex_source(
        1, "near", (0, 1, 0), EXPANSION))
    import torch  # noqa: PLC0415

    rng = np.random.default_rng(20260816)
    shape = (4, 6, 5)
    rows: List[Dict[str, Any]] = []
    first_fired = last_fired = None
    for index, scale in enumerate((1e-30, 1e-34, 1e-38, 1e-40, 1e-44)):
        plane = ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                 * scale).astype(np.complex64)
        # THE CENSUS IS ON THE OPERAND WORDS THE FILL ACTUALLY READS, not on the
        # whole volume, so it is COMMENSURABLE with the divergence count below. A
        # whole-volume census would report a larger number for the same measurement
        # and invite the reader to compare two different things.
        raw = words(plane[:, folded_complex.MIRROR_SOURCE_INDEX, :])
        band = 2 * int(np.count_nonzero(((raw >> 23) & 0xFF == 0)
                                        & ((raw & 0x7FFFFFFF) != 0)))
        volumes = [np.ascontiguousarray(plane.copy()) for _ in range(3)]
        reference = [v.copy() for v in volumes]
        for target in (0, 2):                       # shifts (0, 1, 0)
            reference[target][:, 0, :] = np.complex64(1) * reference[target][:, 2, :]
        tensors = [torch.from_numpy(v.view(np.float32).reshape(-1, 2).copy()).to("mps")
                   for v in volumes]
        module.folded_mirror_fill_complex(
            *tensors, shape[0], shape[1], shape[2], shape[0] * shape[2], -1,
            (1.0, 0.0))
        torch.mps.synchronize()
        got = [t.cpu().numpy().reshape(-1).view(np.complex64).reshape(shape)
               for t in tensors]
        differ = sum(differing(got[i][:, 0, :], reference[i][:, 0, :])
                     for i in (0, 2))
        if differ:
            first_fired = index if first_fired is None else first_fired
            last_fired = index
        rows.append({"scale": scale, "subnormal_operand_words": band,
                     "differing": differ})
        log(f"[band] scale={scale:g} subnormal_words={band} differing={differ}")
    assert rows[0]["differing"] == 0, rows
    assert any(row["differing"] for row in rows), (
        "no scale reached the band: this leg measured nothing")
    payload["legs"]["band"] = {
        "per_scale": rows,
        "window": {"first_fired_index": first_fired, "last_fired_index": last_fired,
                   "first_fired_scale": rows[first_fired]["scale"],
                   "last_fired_scale": rows[last_fired]["scale"]},
        "note": ("subnormal_operand_words counts the words on the ROW THE FILL "
                 "READS, doubled for the two shift-0 targets that read it, so it is "
                 "commensurable with `differing`. The two match at every scale: "
                 "every subnormal operand word flushes and nothing else moves. The "
                 "FILL is band-safe on the REAL fold and is NOT here, because the "
                 "composition made the parity arithmetic"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG precondition — the control that makes the per-step census FIRE
# ---------------------------------------------------------------------------

#: Scale, and whether the per-step census MUST stay clean at it. The clean rows are
#: what make the firing row mean something: a detector that fired on everything
#: would refuse the physical band too and certify nothing.
PRECONDITION_SCALES: Tuple[Tuple[str, float, bool], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-30, True),
    ("subnormal_band", 1e-38, False),
    ("deep_subnormal", 1e-40, False),
)


def leg_precondition(payload: Dict[str, Any], out: str, budget: int = 6) -> None:
    """A precondition never demonstrated to FIRE is decoration.

    :func:`leg_whole_step` censuses every case per step and reports a window; on the
    physical band that window is empty, and an empty window is exactly what a broken
    census also produces. This leg drives the SAME per-step census over a state
    scaled INTO the band and requires it to fire, so the empty windows next door are
    a measurement rather than a silence.

    THE WINDOW IS THE UNIT, not a count: the row records the first and last step the
    census fired on, and the scaled cases are REFUSED BY NAME rather than compared.
    """
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in PRECONDITION_SCALES:
        reference_fields, reference_pml = build("periodic_even_3d")
        if scale != 1.0:
            for name in STORED:
                array = getattr(reference_fields, name, None)
                if array is not None:
                    array *= np.array(scale, dtype=np.float32)
        fired: List[int] = []
        per_step: List[int] = []
        for step in range(budget):
            reference_step(reference_fields, reference_pml)
            count = state_census(reference_fields)
            per_step.append(count)
            if count:
                fired.append(step)
        row = {"scale": label, "factor": scale, "steps_censused": budget,
               "subnormal_words_per_step": per_step,
               "first_step": fired[0] if fired else None,
               "last_step": fired[-1] if fired else None,
               "census_fired": bool(fired),
               "verdict": ("REFUSED (subnormal precondition)" if fired
                           else "precondition holds")}
        rows.append(row)
        log(f"[precondition] {label:<16} scale={scale:g} window="
            f"[{row['first_step']}, {row['last_step']}] fired={bool(fired)} "
            f"words_per_step={per_step}")
        payload["legs"]["precondition"] = rows
        save(payload, out)
        if expect_clean:
            assert not fired, (
                f"{label}: the census fired on a band it must not — a detector that "
                f"refuses the physical band certifies nothing")
        else:
            assert fired, (
                f"{label}: the census DID NOT FIRE on the scaled control. Every "
                f"empty window this gate reports would then be a silence rather "
                f"than a measurement")
            kit.assert_census_floor(max(per_step), f"{label} control")


# ---------------------------------------------------------------------------
# LEG mutations — a leg that cannot fail certifies nothing
# ---------------------------------------------------------------------------

def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    harness = kit.MutationHarness(payload, out)

    def fill_mutation(label: str, must_catch: Optional[bool], why: str,
                      scope: Optional[str] = None, plant: bool = True,
                      transform: Any = None, source_text: Any = None,
                      per_case: Any = None) -> None:
        """One defect over the whole case matrix, on a state that can SHOW it.

        ``plant`` SEEDS THE SIGNED-ZERO CLASS ON THE ROWS THE FILL READS, and it is
        on by default because of what was MEASURED here: with the matrix's
        random-uniform volumes the plane-wise-sign-flip mutant is caught on only
        8 of 10 cases, and the two misses are exactly the rows whose random state
        happens to carry no signed zero where the fill reads. THE DIVERGENCE THIS
        FAMILY EXISTS FOR LIVES ENTIRELY IN SIGNED ZEROS — the complex product
        canonicalizes them and a sign-bit copy preserves them — so a state without
        them makes a real defect look uncaught. Planting is what makes the needle
        reachable, not what makes it pass: every other mutation here (a flipped
        coefficient, a shifted reflect row) is visible on both states.
        """
        missed = False
        ran = caught = launches = 0
        skipped: List[str] = []
        for case, _ in CASES:
            fields, pml = build(case)
            if scope is not None and scope not in tags(fields, pml):
                continue
            case_transform = transform
            if per_case is not None:
                # A PER-CASE TRANSFORM MAY DECLINE, and declining is NOT a needle
                # miss: "this grid has no spare axis to misapply the fold to" is a
                # structural fact about the grid, not a transform that matched
                # nothing. The two are recorded separately so a reader can tell a
                # defect that could not be planted from one that was planted and
                # survived.
                case_transform = per_case(fields)
                if case_transform is None:
                    skipped.append(case)
                    continue
            names = fill_names("D")
            if plant:
                plant_signed_zeros(fields, names)
            before = snapshot(fields, names)
            stepping.fill_symmetry_bc_D(fields)
            stepping.fill_folded_far_ghosts_D(fields)
            after = snapshot(fields, names)
            restore(fields, before)
            try:
                plan = run_fill(fields, "D", transform=case_transform,
                                source_text=source_text)
            except LookupError:
                missed = True
                restore(fields, before)
                continue
            launches += plan.launches
            ran += 1
            if sum(divergence(fields, after).values()):
                caught += 1
            restore(fields, before)
        harness.record(label, harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, must_catch, why,
                       extra={"scope": scope, "signed_zeros_planted": plant,
                              "skipped_cases": skipped})

    # 1. THE HEADLINE. A plane-wise sign flip is the REAL fold's certified spelling
    #    and is WRONG here at BOTH parities, including the even mirror.
    fill_mutation(
        "planewise_sign_flip_instead_of_complex_product", True,
        "the real fold's spelling: {-re, -im} for odd, a plain copy for even. Under "
        "complex64 the array path is a FULL multiply by (+/-1, +0) with its zero "
        "cross terms, and the even mirror is not the identity",
        source_text=lambda key, entry: needle(
            folded_complex.folded_mirror_fill_complex_source(
                entry["axis"], entry["pass"], entry["shifts"], EXPANSION),
            "c_mul(c, f", "c_planewise(c, f").replace(
                "kernel void folded_mirror_fill_complex(",
                "static inline float2 c_planewise(float2 c, float2 z) "
                "{ return float2(z.x * c.x, z.y * c.x); }\n\n"
                "kernel void folded_mirror_fill_complex("))

    # 2. THE COEFFICIENT IS PASSED, never synthesised: a flipped word must be caught.
    def flip(family: str, near: List[Dict[str, Any]],
             far: List[Dict[str, Any]]) -> None:
        for entry in near + far:
            re, im = entry["coefficient"]
            entry["coefficient"] = (-re, im)

    fill_mutation("flipped_parity_coefficient", True,
                  "the parity enters ONLY as a host-rounded passed word, which is "
                  "what makes this reachable at all", transform=flip)

    # 3. THE REFLECT ROW. `stored - 2` at an even full count, `stored - 3` at an odd
    #    one — so at an EVEN count the wrong formula happens to be right and this is
    #    a MEASURED NULL there, which is why the matrix carries both.
    def bake_n_minus_two(family: str, near: List[Dict[str, Any]],
                         far: List[Dict[str, Any]]) -> None:
        if not far:
            raise LookupError("no far pass on this grid")
        for entry in far:
            entry["reflect_row"] = int(entry["reflect_row"]) + 1

    fill_mutation("reflect_row_n_minus_two", True,
                  "baking n - 2 reflects about the window top instead of about the "
                  "second mirror: a whole cell wrong at an ODD full count",
                  scope="odd_full_count", transform=bake_n_minus_two)

    # 4. THE ORIENTATION. A MEASURED EQUIVALENCE for THIS coefficient (c_im is
    #    bitwise +0.0), and named as one rather than left untested.
    fill_mutation(
        "coefficient_orientation_swapped", False,
        "phase * plane vs plane * phase. For a GENERAL complex coefficient this is "
        "load-bearing; with c_im an exact +0.0 the two fma addends are the same "
        "exact zero and float addition is sign-commutative",
        source_text=lambda key, entry: needle(
            folded_complex.folded_mirror_fill_complex_source(
                entry["axis"], entry["pass"], entry["shifts"], EXPANSION),
            "c_mul(c, f", "c_swapped(c, f").replace(
                "kernel void folded_mirror_fill_complex(",
                "static inline float2 c_swapped(float2 c, float2 z) "
                "{ return c_mul(z, c); }\n\n"
                "kernel void folded_mirror_fill_complex("))

    # 5. MIRROR THE WRONG HALF. The near pass writes the GHOST cell 0 from the OWNED
    #    cell 2; the inverse writes owned data from the ghost. This is the defect
    #    that destroys the half the fold is supposed to reconstruct, and it is the
    #    one a per-sub-step comparison of the CURL alone would never see.
    def near_body_line(slot: int) -> str:
        return (f"    f{slot}[base] = c_mul(c, "
                f"f{slot}[base + {folded_complex.MIRROR_SOURCE_INDEX} * stride]);")

    def mirror_wrong_half(key: str, entry: Dict[str, Any]) -> Optional[str]:
        if entry["pass"] != "near":
            return None            # the far pass carries no `base` write to invert
        text = folded_complex.folded_mirror_fill_complex_source(
            entry["axis"], entry["pass"], entry["shifts"], EXPANSION)
        for slot, shift in enumerate(entry["shifts"]):
            if shift != 0:
                continue
            text = needle(
                text, near_body_line(slot),
                f"    f{slot}[base + {folded_complex.MIRROR_SOURCE_INDEX} * stride]"
                f" = c_mul(c, f{slot}[base]);")
        return text

    fill_mutation(
        "mirror_the_wrong_half", True,
        "the near pass images the OWNED cell into the GHOST; inverting it overwrites "
        "owned interior data from the ghost plane, which is the fold's most "
        "destructive spelling error and is invisible to a curl-only comparison",
        source_text=mirror_wrong_half)

    # 6. SHIFT THE MIRROR PLANE BY ONE CELL, on the NEAR pass. The far pass's own
    #    version of this is `reflect_row_n_minus_two`, which is REACHABLE ON ONE CASE
    #    (it is a measured null at an even full count). The near plane is a compiled
    #    constant on every case, so this is the broad-coverage twin of that needle.
    def shift_mirror_plane(key: str, entry: Dict[str, Any]) -> Optional[str]:
        if entry["pass"] != "near":
            return None
        return needle(
            folded_complex.folded_mirror_fill_complex_source(
                entry["axis"], entry["pass"], entry["shifts"], EXPANSION),
            f"+ {folded_complex.MIRROR_SOURCE_INDEX} * stride]",
            f"+ {folded_complex.MIRROR_SOURCE_INDEX + 1} * stride]")

    fill_mutation(
        "near_mirror_plane_off_by_one", True,
        "MEEP's little_owned_corner0 puts the near image at stored cell 2 exactly; "
        "cell 3 is a whole cell of the wrong field and is a compiled constant, so "
        "unlike the far reflect row it is reachable on every case",
        source_text=shift_mirror_plane)

    # 7. DROP THE SIGN FLIP on a mirrored component — force every parity word to
    #    +1. SCOPED, because on an EVEN fold with NO far pass every coefficient
    #    ALREADY IS +1 and the mutation is the identity: measured unscoped it reports
    #    CAUGHT 8/10, and the two misses are `metallic_even_3d` and
    #    `bloch_metallic_fold_3d`. Reporting 8/10 for a defect that exists on 8 cases
    #    would be rounding a structural null into a weakness.
    def drop_sign_flip(family: str, near: List[Dict[str, Any]],
                       far: List[Dict[str, Any]]) -> None:
        for entry in near + far:
            re, im = entry["coefficient"]
            entry["coefficient"] = (abs(re), im)

    fill_mutation(
        "drop_the_sign_flip", True,
        "an odd mirror plane negates and an even one does not; forcing +1 drops the "
        "negation the declared parity carries. The near word is +phase and the far "
        "word is -phase, so this is reachable wherever EITHER is negative",
        scope="signed_parity", transform=drop_sign_flip)

    # 8. APPLY THE FOLD TO THE WRONG AXIS. The axis picks the plane decomposition,
    #    the compiled specialisation AND the launch width, so this is the mutation
    #    that proves those three agree with each other rather than merely compiling.
    def wrong_axis(fields: Any) -> Optional[Callable[..., None]]:
        codes, _ = folded_complex.folded_axis_kinds(fields.grid, None)
        folded_axes = {i for i, c in enumerate(int(x) for x in codes)
                       if c in folded_complex.MIRROR_CODES}
        shape = tuple(int(n) for n in fields.grid.shape)
        # The target must be a real axis with room for the image row, or the mutant
        # would read out of bounds and the leg would be measuring a crash.
        spare = [a for a in range(3) if a not in folded_axes
                 and shape[a] > folded_complex.MIRROR_SOURCE_INDEX + 1]
        if not spare:
            return None
        target = spare[0]

        def transform(family: str, near: List[Dict[str, Any]],
                      far: List[Dict[str, Any]]) -> None:
            for entry in near + far:
                entry["axis"] = target
        return transform

    fill_mutation(
        "fold_applied_to_wrong_axis", True,
        "the axis selects the plane decomposition, the compiled specialisation and "
        "the launch width together; misapplying the fold to an UNFOLDED axis mirrors "
        "a boundary that has no mirror",
        per_case=wrong_axis)

    # 9. REVERSE THE X, Y, Z FILL ORDER. A MEASURED NULL on one folded axis and on
    #    matched phases — complex multiplication is not associative, so the order is
    #    load-bearing exactly where a corner is unowned on two planes whose parities
    #    DIFFER. Measured: 5 words on `two_axis_mixed_3d`, 0 on all nine others,
    #    which is why the scope is the tag rather than the whole matrix.
    def reverse_axis_order(family: str, near: List[Dict[str, Any]],
                           far: List[Dict[str, Any]]) -> None:
        near.reverse()
        far.reverse()

    fill_mutation(
        "fill_axis_order_reversed", True,
        "the array path applies the folded axes in X, Y, Z order; a doubly-unowned "
        "corner carries the PRODUCT of both parities and complex multiplication does "
        "not associate, so the order is only measurable at MIXED phase",
        scope="mixed_phase", transform=reverse_axis_order)

    # 10. READ THE GHOST BEFORE IT IS WRITTEN, by swapping the near and far passes.
    #     A MEASURED NULL on every admitted grid, and that is the POINT: the
    #     predicate refuses any grid where the far read row is a row the near launch
    #     already wrote (folded_complex.py:1252). This arm is what turns that refusal
    #     from a claim into a check — if a future grid were admitted where the two
    #     alias, this flips to CAUGHT and the gate fails.
    missed = False
    ran = caught = launches = 0
    for case, _ in CASES:
        fields, pml = build(case)
        names = fill_names("D")
        plant_signed_zeros(fields, names)
        before = snapshot(fields, names)
        stepping.fill_symmetry_bc_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        after = snapshot(fields, names)
        restore(fields, before)
        residency = device.Residency()
        plan = build_fill(fields, "D", residency)
        residency.sync_in()
        plan.run_far()                       # FAR FIRST — the inverted pass order
        plan.run_near()
        residency.sync_out()
        launches += plan.launches
        ran += 1
        if sum(divergence(fields, after).values()):
            caught += 1
        restore(fields, before)
    harness.record("near_far_pass_order_swapped",
                   harness.verdict(missed, ran, launches, caught),
                   launches, caught, ran, False,
                   "reading a ghost the other pass has not written yet. The near "
                   "pass reads cell 2 and writes cell 0; the far pass reads "
                   "reflect_row and writes the last cell. The predicate REFUSES any "
                   "grid where those alias, so this is a null BECAUSE of a guard "
                   "rather than by luck, and this arm is the guard's check")

    # 11. THE TOP-PLANE MASK, on the CURL. Reachable on MIRROR_PERIODIC and a
    #    structural null on MIRROR_METALLIC, where the emitter writes no mask at all.
    missed = False
    ran = caught = launches = 0
    for case, _ in CASES:
        fields, pml = build(case)
        if "far" not in tags(fields, pml):
            continue
        names = curl_names("step_D")
        before, after, moved = oracle(
            fields, lambda: stepping.step_D(fields, pml), names)
        kit.assert_moved(moved, f"{case} step_D reference barely moved", floor=64)
        codes, _ = folded_complex.folded_axis_kinds(fields.grid, pml)
        # THE MUTANT'S PHASE FLAGS ARE THE CASE'S OWN, not a hard-coded (0, 0, 0),
        # and getting that wrong would have CONFOUNDED this mutation on the two
        # Bloch rows: a mutant built unphased against a phased reference diverges
        # because it dropped the ROTATION, and the row would have reported CAUGHT
        # for a defect it never planted.
        flags, _values = complex_fields.phase_arguments(
            complex_fields.bloch_phase_table(
                fields.grid, stepping._boundary_kinds(fields.grid, pml)),
            backward=True)
        shipped = folded_complex.folded_bloch_curl_source(
            codes, True, flags, EXPANSION)
        mask = folded_complex.folded_top_plane_mask(
            codes, True, zero=templates.COMPLEX_ZERO)
        try:
            mutant = needle(shipped, mask, "    // MUTANT: top-plane mask dropped")
        except LookupError:
            missed = True
            continue
        counter = kit.Counter(compile_source(mutant).bloch_pml_curl_step)
        run_curl(fields, pml, "step_D", functions={shaders.CONTRACT_OFF: counter})
        launches += counter.launches
        ran += 1
        if sum(divergence(fields, after).values()):
            caught += 1
        restore(fields, before)
    harness.record("drop_top_plane_mask",
                   harness.verdict(missed, ran, launches, caught),
                   launches, caught, ran, True,
                   "the folded PERIODIC top plane sits past MEEP's big_corner and "
                   "the FILL writes it, not the curl; masking it is what makes the "
                   "two agree", extra={"scope": "far"})

    # 12. READ THE GHOST BEFORE IT IS WRITTEN, at the only granularity where the
    #     defect EXISTS — the complete step. Every sub-step kernel here is the
    #     shipped one and every one of them is byte-perfect in isolation; all that
    #     moves is WHEN the fill runs relative to the constitutive pass that reads
    #     its ghost plane. This is the fold's fourth risk stated as a needle, and it
    #     is what makes `leg_whole_step`'s `first_divergent=None` a result rather
    #     than a property of a leg that cannot fail.
    STALE_ORDER = ("step_B", "update_H", "fill_B_near", "fill_B_far",
                   "step_D", "update_E", "fill_D_near", "fill_D_far")
    assert sorted(STALE_ORDER) == sorted(DRIVER_ORDER), (
        "the stale order must be a PERMUTATION of the driver's own: a needle that "
        "also dropped or duplicated a pass would be measuring something else")
    ran = caught = launches = 0
    first_steps: Dict[str, Optional[int]] = {}
    for case, _ in CASES:
        fields, pml = build(case)
        reference_fields, reference_pml = build(case)
        residency = device.Residency()
        plans = composed_plans(fields, pml, residency)
        residency.sync_in()
        first: Optional[int] = None
        for step in range(6):
            reference_step(reference_fields, reference_pml)
            for name in STALE_ORDER:
                if name.endswith("_near"):
                    plans[name[:-5]].run_near()
                elif name.endswith("_far"):
                    plans[name[:-4]].run_far()
                else:
                    plans[name].run()
            residency.sync_out()
            total = sum(differing(getattr(fields, n), getattr(reference_fields, n))
                        for n in STORED if getattr(fields, n, None) is not None)
            if total and first is None:
                first = step
        launches += sum(plan.launches for plan in plans.values())
        ran += 1
        first_steps[case] = first
        if first is not None:
            caught += 1
    harness.record("read_the_ghost_before_it_is_written",
                   harness.verdict(False, ran, launches, caught),
                   launches, caught, ran, True,
                   "the constitutive pass reads the ghost plane the fill writes. "
                   "Every kernel here is the SHIPPED one and each is byte-perfect "
                   "per sub-step; only the ORDER moves, so this defect exists at "
                   "whole-step granularity and nowhere else",
                   extra={"first_divergent_step_per_case": first_steps,
                          "budget": 6})

    # 13. THE PHASED-AXIS GUARD. A mirror code carrying a Bloch flag must be refused
    #    at EMISSION, not merely by the predicate: a mis-baked flag is a plane of
    #    wrong values, not a crash. Recorded as a structural refusal rather than a
    #    byte comparison, because the source cannot be built at all.
    refusals = 0
    for codes in ((folded_complex.CODE_PERIODIC, MP, folded_complex.CODE_PERIODIC),
                  (MM, folded_complex.CODE_PERIODIC, folded_complex.CODE_PERIODIC)):
        axis = 1 if codes[1] in folded_complex.MIRROR_CODES else 0
        flags = tuple(1 if i == axis else 0 for i in range(3))
        try:
            folded_complex.folded_bloch_curl_source(codes, False, flags, EXPANSION)
        except ValueError:
            refusals += 1
    assert refusals == 2, refusals
    payload["legs"]["emission_refusals"] = {
        "phased_folded_axis_refused": refusals,
        "why": ("clause 9 held at the LAST place it can be seen: a gate hands codes "
                "and flags straight in, so the emitter refuses too"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LEGS: Tuple[Tuple[str, Callable[[Dict[str, Any], str], None]], ...] = (
    ("execution", leg_execution),
    ("curl", leg_curl),
    ("fill", leg_fill),
    ("constitutive", leg_constitutive),
    ("whole_step", leg_whole_step),
    ("parity", leg_parity),
    ("band", leg_band),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
)


def main() -> int:
    parser = kit.argument_parser(__doc__)
    arguments = parser.parse_args()
    started = time.time()
    out = arguments.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    payload: Dict[str, Any] = {"legs": {}, "environment": kit.environment_stamp()}

    reasons: List[str] = []
    try:
        import torch

        if not torch.backends.mps.is_available():
            reasons.append("no MPS device on this host")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"torch unavailable: {exc!r}")
    if EXPANSION is None:
        reasons.append(
            "no folded-complex expansion probe artifact: this family needs the "
            f"{folded_complex.PARITY_PROBE_PATTERN!r} orientation, which no earlier "
            "artifact carries. Cut one with "
            "probe_metal_folded_complex_expansion.py")
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    ran = kit.run_legs(LEGS, payload, out, kit.wanted_legs(arguments.legs))

    compared = 0
    certified = True
    for key in ("curl", "fill", "constitutive"):
        for row in payload["legs"].get(key, ()):
            compared += int(row["compared"])
            certified = certified and row["differing"] == 0
    # A CASE REFUSED ON THE PRECONDITION CONTRIBUTES NO COMPARISONS AND NO VERDICT.
    # Folding it into `certified` either way would be wrong in both directions: as a
    # pass it would certify bytes produced under a condition the claim excludes, and
    # as a failure it would report a coverage boundary as a defect.
    refused = [row["case"] for row in payload["legs"].get("whole_step", ())
               if row.get("refused")]
    for row in payload["legs"].get("whole_step", ()):
        if row.get("refused"):
            continue
        compared += int(row["compared"])
        certified = certified and row["first_divergent"] is None

    return kit.summarize(
        payload, out,
        claim=("metal_kernels.folded_complex reproduces stepping.py word for word "
               "on a folded complex64 grid, per sub-step and per complete step"),
        scope=("the case matrix in CASES: both fold terminations, both declared "
               "phases, one and two folded axes at mixed phase, both full-count "
               "parities, a Bloch phase on an unfolded axis, 2-D and 3-D; beta, "
               "off-diagonal rows, dispersion, conductivity and cylindrical "
               "coordinates are OUT OF SCOPE and refused by name"),
        stated_weakness=(
            "no generated-code audit exists on this backend: torch.mps.compile_shader "
            "exposes no disassembly, so this gate cannot refuse a compile whose "
            "emitted code violates the policy nor establish that the contraction "
            "guard was obeyed. Every leg is BEHAVIOURAL. The arithmetic claims also "
            "ride a CHECKED subnormal-free precondition, which for THIS family "
            "bounds the FILL as well as the curl (see leg band)"),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"cases_refused_on_precondition": refused,
               "subnormal_windows": {
                   row["case"]: [row["subnormal_window"]["first_step"],
                                 row["subnormal_window"]["last_step"]]
                   for row in payload["legs"].get("whole_step", ())}})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
