"""The Metal WHOLE STEP against the array path, byte for byte, per complete step.

WHAT THIS ARBITRATES THAT NO FAMILY GATE CAN. Each family gate certifies ONE
sub-step: it seeds a state, launches one kernel, and compares words. A shelf of
green family gates says nothing about the object the engine would actually run,
which is a COMPLETE DRIVER STEP composed of four sub-steps, a wall pass on a walled
grid, and whatever the composer left on the array path in between. Five failure
classes live only there and every one of them is invisible to a per-sub-step gate:

1. **A stale mirror.** The mirrors are the Metal port's own structure — the engine
   holds NumPy, so a plan owns device copies. A sub-step the composer left on the
   array path writes the HOST array; a mirror held across it is stale, and a stale
   mirror is a smooth, plausible, wrong field rather than an error. The residency
   verdict is supposed to prevent that. This gate runs the composition the verdict
   licenses and checks the bytes.

2. **A seam.** ``zero_metal_B``/``zero_metal_D`` clear stored cell 0 between the
   curl and the constitutive sub-step on a walled run (stepping.py:2211/:2238,
   driver.py:3167 vs :3169). A composition that consumed the pre-seam value is
   byte-perfect per sub-step and wrong per step.

3. **An accumulating auxiliary.** ``fu_*``, ``f_w_*`` and ``f_bfast_*`` are STATE. A
   kernel right for one launch and wrong forever after is identical in a
   single-launch gate and diverges only across steps. So the comparison is per
   COMPLETE STEP over a stated budget, and the first divergent step is reported
   rather than a final pass/fail.

4. **A ghost plane produced and consumed a pass apart** — the FOLD's own class, and
   the reason a folded step is TEN passes rather than six. ``fill_symmetry_bc_*``
   writes cell 0 of every unowned plane and the NEXT sub-step reads it; on a folded
   PERIODIC axis ``fill_folded_far_ghosts_*`` writes another plane AFTER the wall
   clear, which is why the two fills are separate slots and why a plan may not fuse
   across the wall. Every kernel involved can be byte-perfect on its own launch
   while the step is wrong, because what changed is which value was in the plane
   when the reader ran. Leg ``armed`` arms this directly: it moves only the ORDER,
   leaving every shipped kernel exactly as it is, and asserts its reordering is a
   PERMUTATION of the driver's own so it cannot be measuring a dropped pass instead.

5. **A PLAN THAT ITSELF TOUCHES THE HOST MID-COMPOSITION** — tranche 4's own class,
   and the reason the two cylindrical families belong here rather than only in their
   family gates. The complex cylindrical curl's ``run`` syncs its prefix SOURCE out,
   scans it on the host, and syncs the prefix mirror back in BEFORE launching; the
   real m = 0 curl's ``run`` dispatches TWICE, and the second launch reads what the
   first wrote. Both are correct exactly once per sub-step per step and in one
   order, and both are indistinguishable from their broken twins in any gate that
   launches once. Leg ``armed`` arms each directly (``stale_cylindrical_prefix``,
   ``cylindrical_scan_after_curl``) — the second keeps the launch COUNT exactly
   right, which is why a counter cannot catch it.

THE CLAIM SHAPE, unchanged from every other Metal claim and restated because it is
a real condition rather than a formality: **byte-identity subject to a CHECKED
subnormal-free precondition**. On MPS the float32 subnormal flush is native and has
no lever, so a run whose operands, results or intermediates enter the band is not
covered by this gate's claim. Leg ``subnormal_window`` censuses every state array on
every step and REPORTS THE WINDOW — first and last step at which any word landed in
the band — rather than a scalar verdict, because the chi3 round measured a run whose
band entry was a transient: first at step 55, last at step 3,726, clean for the
remaining 16,274. A census that fires once and a census that never fires are
different claims and this gate distinguishes them.

WHAT THIS IS NOT. It is not throughput: nothing here is timed and no throughput has
ever been measured on this backend. It is not dispatch: ``fastpath.plan_fast_path``
still returns ``None`` on every branch, and this gate builds its composition by
calling ``plan_step`` directly, which is what makes it a measurement of what WOULD
run rather than a change to what does.

LEGS
  0  arms          the arm table and the environment this ran under, recorded
  1  whole_step    complete steps, per-step byte compare, per-slot launch counters
  2  mixed         compositions where the composer left slots on the array path
  3  armed         a mutated plan MUST diverge — the anti-hollow-pass leg
  4  subnormal     the checked precondition, as a WINDOW with its floors
  5  resident_pole_pack  folded tensor dispersion without an inter-step host sync

Progress reporting: a flushed line per case, the artifact rewritten as each case lands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metal_composition_matrix as matrix  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms, coverage as metal_coverage, device, launch, preconditions, subnormal,
)

#: Every statically named volume one complete step can touch, in a fixed order.
#: Derived from the composer's OWN residency table rather than listed by hand,
#: plus the BFAST IIR state, which is not a sub-step volume but is carried between
#: steps and is the one buffer whose staleness never decays (its homogeneous mode
#: is (-1)^n).  Polarization arrays are intentionally added at runtime by
#: :func:`state_of`: their physical identities rotate after every ``update_P`` and
#: no static string table can name them without losing the semantic P/P_prev roles.
STATE_NAMES: Tuple[str, ...] = tuple(sorted(
    {name
     for slot in metal_coverage.RESIDENCY_ORDER
     for name in metal_coverage.sub_step_volumes(slot)}
    | {f"f_bfast_{side}{axis}" for side in "BD" for axis in "xyz"}))

#: The budget every multi-step case runs, STATED rather than implied. Tranche 1's
#: cycle leg used four; twelve is used here because this gate's whole point is the
#: classes that only compound — `fu_*`, `f_w_*` and the marginally stable
#: `f_bfast_*`, whose homogeneous mode is (-1)^n and never decays — and a defect
#: that needs three steps to become visible in the low bits is exactly the kind a
#: four-step budget would report as green. The comparison is per COMPLETE STEP, so
#: the budget also bounds `first_divergence`: a case that passes here is identical
#: at every one of its twelve steps, not merely at the last.
CYCLES = 12

#: The seam order one complete driver step runs in — FIVE PASSES PER HALF followed
#: by an optional polarization recurrence, taken
#: from the composer's own residency table rather than re-listed here, so the walk
#: and the residency model cannot disagree about what a step is.
#:
#: The order is driver.py:3281-3287 (B half) and :3292-3302 (D half):
#: ``step_B`` -> magnetic sources -> ``fill_symmetry_bc_B`` -> ``zero_metal_B`` ->
#: ``fill_folded_far_ghosts_B`` -> ``update_H``, and the same five on the D side.
#: THE WALL PASS SITS BETWEEN THE TWO FILL PASSES, which is not a detail: it is the
#: reason the folded families refuse both fill slots on a walled run by name, and
#: the reason a fold bug can be byte-perfect per sub-step and wrong per step.
SEAM_ORDER: Tuple[str, ...] = tuple(metal_coverage.RESIDENCY_ORDER)

#: Which array-path function each seam slot is. ``fill_B``/``fill_D`` are the NEAR
#: symmetry fill (stepping.py:1431/:1443) and the far ghost passes are their own
#: slots (stepping.py:1521/:1533), because the wall pass runs between them.
ARRAY_PATH = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
    "update_P": lambda f, p: stepping.update_P(f, p),
}

#: The two composed fill slots and the far pass each one may also cover. A fill
#: plan DECLARES which passes it replaces (``replaces_sub_steps``), and this gate
#: reads that declaration rather than assuming a plan covers both: on a folded
#: METALLIC axis there is no far pass at all and the plan says so.
FILL_SLOT_FAR = {"fill_B": "fill_folded_far_ghosts_B",
                 "fill_D": "fill_folded_far_ghosts_D"}


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def words(array: Any) -> Any:
    """One array as uint32 WORDS. Byte compares, never allclose."""
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields: Any) -> Dict[str, Any]:
    """Every semantically live state array, including rotating ADE roles.

    ``MetalAdeUpdatePPlan`` rotates ``P``, ``P_prev`` and ``_scratch`` pointers
    after each recurrence.  Comparing only the static field table would therefore
    let a defect in update_P pass a full-step byte comparison.  These keys name the
    roles rather than the changing physical allocations, matching the recurrence
    the array-path oracle executes.
    """
    state = {name: getattr(fields, name) for name in STATE_NAMES
             if getattr(fields, name, None) is not None}
    for index, polarization in enumerate(tuple(getattr(fields, "polarizations", ())
                                               or ())):
        driven = tuple(polarization.driven())
        for component in driven:
            state[f"polarization:{index}:P:{component}"] = polarization.P[component]
            state[f"polarization:{index}:P_prev:{component}"] = (
                polarization.P_prev[component])
        state[f"polarization:{index}:scratch"] = polarization._scratch
    return state


def compare(left: Any, right: Any) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), (sorted(set(a) ^ set(b)),)
    return {name: n for name in a if (n := differing(a[name], b[name]))}


def moved(before: Dict[str, Any], fields: Any) -> int:
    after = state_of(fields)
    return sum(differing(before[name], after[name]) for name in before)


def frozen(fields: Any) -> Dict[str, Any]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


# ---------------------------------------------------------------------------
# Composition and stepping
# ---------------------------------------------------------------------------

def live_slots(fields: Any, pml: Any) -> Tuple[str, ...]:
    """Which seam slots this run actually executes. Sources are ``()`` here.

    THE COMPOSER'S OWN MODEL, not a second one. This used to derive the answer from
    ``zero_metal_axes`` alone, which was correct while nothing folded and silently
    wrong the moment something did: a folded run executes ``fill_symmetry_bc_*`` on
    every step with no source at all (stepping.py:1482-1483 returns immediately only
    when the grid has NO symmetry), and a folded PERIODIC axis executes the far
    ghost pass on top of that. Asking :func:`launch.live_sub_steps` means the walk,
    the sync declaration and the residency verdict are all reading one function.

    An UNREADABLE live set is a refusal here rather than an empty tuple: a gate that
    stepped nothing would compare a no-op with a no-op and pass.
    """
    live = launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live sub-step set is unreadable for this configuration; the walk "
        "would silently step a subset and the comparison would certify it")
    return tuple(name for name in SEAM_ORDER if name in set(live))


def compose(fields: Any, pml: Any, probes: Dict[str, Any]) -> Tuple[Any, Any,
                                                                    Tuple[str, ...]]:
    """Plan the step TWICE, and the second time is not redundant.

    The residency verdict needs the SYNCED set — which array-path sub-steps the
    caller brackets with an explicit sync out and back — and the caller cannot know
    that set until it knows which slots the composer filled. So the first call is a
    throwaway on a throwaway residency that answers "which slots are mine", and the
    second is the composition this gate actually runs, with the sync declaration in
    hand. Declaring the syncs is what makes the verdict a claim about THIS
    composition rather than about an ideal one.
    """
    live = live_slots(fields, pml)

    scout = launch.plan_step(fields, pml, residency=device.Residency(), sources=(),
                             **probes)
    on_device = covered_passes(scout)
    synced = tuple(name for name in live if name not in on_device)

    residency = device.Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=(),
                            synced=synced, **probes)
    return plan, residency, synced


def covered_passes(plan: Any) -> set:
    """Every seam pass this composition takes off the array path.

    NOT ``set(plan.plans)``. A fill plan occupies ONE slot and may replace TWO
    passes — ``fill_B`` and ``fill_folded_far_ghosts_B`` — because the driver runs
    them either side of the wall pass. Reading the slot keys alone would declare
    the far pass array-path, sync around a pass that never ran on the host, and
    report a composition weaker than the one it actually executed. The plan
    DECLARES what it replaces; this reads the declaration.
    """
    covered = set()
    for slot, product in plan.plans.items():
        covered.add(slot)
        covered.update(getattr(product, "replaces_sub_steps", ()) or ())
    return covered


def run_metal_step(fields: Any, pml: Any, plan: Any, residency: Any,
                   live: Sequence[str]) -> None:
    """One complete driver step, Metal where the composer filled the slot.

    A slot the composer did NOT fill runs on the array path and is bracketed with
    an explicit ``sync_out`` / ``sync_in``. That bracket is the residency clause
    made operational, and leg ``armed`` measures that removing it diverges.

    THE TWO FILL PASSES ARE DISPATCHED SEPARATELY and never through the fill plan's
    own ``run()``. ``run()`` walks near then far back to back, which is the driver's
    order ONLY where the wall pass between them does nothing; this walk puts
    ``zero_metal_*`` where the driver puts it, so a walled folded run is stepped in
    the order the engine would step it rather than in the order that happens to be
    convenient. Whether the plan owns the far pass is read from its own
    ``replaces_sub_steps`` — a folded METALLIC axis has no far pass and says so.
    """
    covered = covered_passes(plan)
    for slot in live:
        if slot in FILL_SLOT_FAR.values():
            owner = next(name for name, far in FILL_SLOT_FAR.items()
                         if far == slot)
            product = plan.plans.get(owner)
            if product is not None and slot in covered:
                product.run_far()
                continue
        elif slot in FILL_SLOT_FAR:
            product = plan.plans.get(slot)
            if product is not None:
                product.run_near()
                continue
        elif slot in plan.plans:
            plan.plans[slot].run()
            continue
        residency.sync_out()
        ARRAY_PATH[slot](fields, pml)
        residency.sync_in()


def run_array_step(fields: Any, pml: Any, live: Sequence[str]) -> None:
    for slot in live:
        ARRAY_PATH[slot](fields, pml)


def probe_records() -> Dict[str, Any]:
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        complex_fields,
        cylindrical_complex,
        folded_complex,
        special_kz,
    )

    records = {"complex_probe": complex_fields.load_expansion_probe(),
               "beta_probe": special_kz.load_expansion_probe(),
               "folded_complex_probe": folded_complex.load_expansion_probe(),
               # PASSED EXPLICITLY, not left to the arm's env-var fallback: a gate
               # that certifies a family whose arm-binding artifact it never handed
               # over is certifying a composition it did not fully specify.
               "cylindrical_complex_probe":
                   cylindrical_complex.load_expansion_probe()}
    missing = sorted(name for name, record in records.items() if record is None)
    if missing:
        raise SystemExit(
            f"no expansion probe artifact for {missing}; those arms would refuse "
            f"by name and this gate would certify the families it could still "
            f"reach while silently dropping the rest")
    return records


# ---------------------------------------------------------------------------
# The case matrix
# ---------------------------------------------------------------------------

#: (label, builder, the families the row is here to exercise). Every row is a
#: configuration the composition sweep already pinned a SELECTION for, so a change
#: in what this gate steps shows up as a sweep mismatch first.
CASES: Tuple[Tuple[str, Any, str], ...] = (
    ("cart_pml_real", lambda: matrix.cart(), "pml_curl + constitutive"),
    ("cart_pml_real_metallic", lambda: matrix.cart(boundaries="metallic"),
     "pml_curl + constitutive, WALLED (the zero_metal seam is live)"),
    ("cart_pml_real_odd_courant", lambda: matrix.cart(courant=0.2718281828),
     "pml_curl + constitutive at a non-power-of-two Courant"),
    ("flat_pml_real_2d", lambda: matrix.flat(), "pml_curl + constitutive in 2-D"),
    ("cart_pml_conductive_electric", lambda: matrix.conductive(matrix.cart()),
     "conductive D PML curl with ordinary B/H/E under one persistent residency"),
    ("cart_pml_conductive_magnetic",
     lambda: matrix.conductive(matrix.cart(), magnetic=True),
     "conductive B PML curl with ordinary D/H/E under one persistent residency"),
    ("cart_pml_dispersive", lambda: matrix.dispersive(matrix.cart()),
     "real PML B/H/D, pole-aware stored E, and live ADE recurrence"),
    ("cart_pml_complex_dispersive",
     lambda: matrix.dispersive(matrix.cart(complex_storage=True)),
     "complex/Bloch PML B/D/H, pole-aware stored E, and live ADE recurrence"),
    ("cart_pml_complex_conductive_electric",
     lambda: matrix.conductive(matrix.cart(complex_storage=True)),
     "complex conductive D curl plus ordinary complex B/H/E under one residency"),
    ("cart_pml_bloch_conductive_magnetic",
     lambda: matrix.conductive(
         matrix.cart(complex_storage=True, k_point=(0.3, -0.2, 0.0)),
         magnetic=True),
     "complex Bloch conductive B curl plus ordinary complex D/H/E under one residency"),
    ("nopml_conductive",
     lambda: matrix.conductive(matrix.cart(pml=0, storage=False, eps=False)),
     "electric conductive no-PML curls plus the true null H/E pair"),
    ("nopml_real_conductive_magnetic",
     lambda: matrix.conductive(
         matrix.cart(pml=0, storage=False, eps=False), magnetic=True),
     "magnetic conductive no-PML curls plus the true null H/E pair"),
    ("nopml_complex_conductive_stored_e",
     matrix.complex_conductive_stored_e_no_pml,
     "complex conductive no-PML curls plus null H and zero-pole stored E"),
    ("nopml_complex_bloch_lossless", matrix.complex_lossless_stored_e_no_pml,
     "lossless complex/Bloch no-PML B/D curls beside null H and zero-pole stored E"),
    ("nopml_complex_bloch_offdiag", matrix.complex_lossless_offdiag_no_pml,
     "lossless complex/Bloch no-PML B/D curls beside null H and a live tensor-row E"),
    ("nopml_real_stored_e", matrix.real_stored_e_no_pml,
     "real stored-E no-PML E beside ordinary B/D curls and null H"),
    ("nopml_real_stored_e_ade",
     lambda: matrix.real_stored_e_no_pml(with_polarization=True),
     "real stored-E and live ADE recurrence after ordinary no-PML B/D and null H"),
    ("cart_pml_complex_forced", lambda: matrix.cart(complex_storage=True),
     "complex/Bloch, all four slots"),
    ("cart_pml_bloch_complex",
     lambda: matrix.cart(complex_storage=True, k_point=(0.3, 0.0, 0.0)),
     "complex/Bloch with a live phase on one axis"),
    ("beta_real_2d", lambda: matrix.flat(beta=0.33),
     "special_kz real beta, all four slots"),
    ("beta_real_2d_metallic",
     lambda: matrix.flat(beta=0.33, boundaries={"x": "metallic"}),
     "special_kz real beta, WALLED"),
    # MOVED HERE FROM `MIXED_CASES` ON 2026-08-19. This row was the gate's generic
    # partial composition — the complex beta curls on the device, both constitutive
    # slots on the array path, because no complex beta constitutive product existed.
    # `special_kz` gained one, so the row composes a WHOLE step and belongs on this
    # list; left where it was it would have been a "mixed" case with nothing mixed.
    ("beta_complex_2d", lambda: matrix.flat(beta=0.33, complex_storage=True),
     "special_kz COMPLEX beta, all four slots — the last two of them new"),
    ("beta_complex_2d_metallic",
     lambda: matrix.flat(beta=0.33, complex_storage=True,
                         boundaries={"x": "metallic"}),
     "special_kz complex beta, WALLED: the wall clear sits between the complex "
     "curl and the complex constitutive sub-step that reads it"),
    ("bfast_real_pml", lambda: matrix.cart(bfast_scaled_k=(0.2, 0.0, 0.0)),
     "BFAST curl + the certified constitutive pair under a restated predicate"),
    ("offdiag_real_one_row", lambda: matrix.cart(rows={"Ex": ("Ey",)}),
     "off-diagonal update_E beside the shipped curls"),

    # -- THE FOLD, which adds a FOURTH failure class to the three above -------
    # A folded step is TEN passes, not six: the near symmetry fill writes a ghost
    # plane that the NEXT sub-step reads, and on a folded periodic axis a far pass
    # writes another one after the wall clear. So a fold defect can be byte-perfect
    # in every per-sub-step gate and wrong per step — the plane is produced by one
    # kernel and consumed by a different one, and only a complete step puts the two
    # in the same measurement. Every row here composes all six slots.
    ("fold_real_2d", lambda: matrix.folded(),
     "the real fold: four arithmetic slots plus BOTH ghost passes, MIRROR_PERIODIC "
     "(the far pass is live)"),
    ("fold_real_2d_metallic",
     lambda: matrix.folded(boundaries={"y": "metallic"}),
     "the real fold, MIRROR_METALLIC: no far pass at all, so the plan declares one "
     "replaced pass rather than two"),
    ("fold_real_2d_odd",
     lambda: matrix.folded(phase=-1),
     "an ODD mirror plane — a different compiled fill on this backend, since the "
     "parity is a source specialisation"),
    ("fold_real_3d", lambda: matrix.folded(depth=1.2),
     "the real fold in 3-D, where a ghost pass writes 192 words rather than 16"),
    ("fold_real_2d_odd_count", lambda: matrix.folded(extent=2.1),
     "an ODD full count: the far reflect row is stored-3 here and stored-2 at the "
     "default extent, so the wrong formula passes on every other folded row"),
    ("fold_real_2d_dispersive", lambda: matrix.dispersive(matrix.folded()),
     "folded real PML B/H/D, mirror fills, pole-aware E, and live ADE recurrence"),
    ("fold_complex_2d", lambda: matrix.folded(complex_storage=True),
     "the folded COMPLEX family, all six slots"),
    ("fold_complex_2d_xy_mixed",
     lambda: matrix.folded(axis="XY", phase=(1, -1), complex_storage=True),
     "two folded axes at MIXED phase under complex storage — the doubly-unowned "
     "corner carries the product of both parities and complex multiplication does "
     "not associate"),
    ("fold_complex_3d", lambda: matrix.folded(complex_storage=True, depth=1.2),
     "the folded complex family in 3-D"),
    ("fold_complex_2d_offdiag",
     lambda: matrix.folded(complex_storage=True, rows={"Ey": ["Ez"]}),
     "folded complex B/D/H and fills with a live complex tensor-row E"),
    ("fold_offdiag_2d", lambda: matrix.folded(rows={"Ex": ("Ey",)}),
     "the folded OFF-DIAGONAL update_E beside the folded curls and fills — the "
     "intersection of two certified families in one step"),
    ("fold_offdiag_2d_metallic",
     lambda: matrix.folded(boundaries={"y": "metallic"},
                           rows={"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}),
     "the folded off-diagonal on the corpus's own shape (folded METALLIC), with "
     "two live rows"),
    ("fold_offdiag_2d_dispersive",
     lambda: matrix.dispersive(matrix.folded(rows={"Ex": ("Ey",)})),
     "folded tensor-row PML E with live packed poles and the ADE successor"),

    # -- TRANCHE 4: THE FOLDED BETA FAMILY, ON BOTH STORAGES ------------------
    # A folded beta step is the same TEN passes as any other fold, and the beta term
    # lands INSIDE the widened mask — between the curl and the top-plane mask — so
    # every one of the fold's four failure classes applies to it unchanged while the
    # arithmetic is a different kernel. Both terminations are carried for the reason
    # every folded row above carries both: MIRROR_PERIODIC runs the far ghost pass
    # and MIRROR_METALLIC does not, and a mutation caught on one is not caught on
    # the other.
    ("fold_real_2d_beta", lambda: matrix.folded(beta=0.3),
     "folded beta REAL, MIRROR_PERIODIC: four arithmetic slots plus both ghost "
     "passes, with the beta increment inside the widened mask"),
    ("fold_real_2d_beta_metallic",
     lambda: matrix.folded(beta=0.3, boundaries={"y": "metallic"}),
     "folded beta REAL, MIRROR_METALLIC: no far pass at all, and the top-plane "
     "mask that sits directly downstream of the beta term emits nothing"),
    ("fold_complex_2d_beta",
     lambda: matrix.folded(complex_storage=True, beta=0.3),
     "folded beta COMPLEX: the same seam under complex64 storage, where the beta "
     "partner is reached through a multiply by +/-1j rather than a sign flip"),
    ("fold_complex_2d_beta_bloch",
     lambda: matrix.folded(complex_storage=True, beta=0.3, k_point=(0.3, 0.0, 0.0)),
     "folded beta COMPLEX with a live Bloch phase on the UNFOLDED axis — the shape "
     "two of the four folded-beta corpus rows actually are, and the only arm that "
     "is not the zero-phase specialisation"),

    # -- TRANCHE 4: THE TWO CYLINDRICAL FAMILIES ------------------------------
    # A cylindrical step is SIX passes, not ten: no fold, so neither ghost pass is
    # live — but the r = 0 seam is live on every one of them, and on a z-METALLIC
    # run `zero_metal_B`/`_D` sit between the curl and the constitutive sub-step
    # exactly where they do on a Cartesian walled run. So these rows carry the WALL
    # class and the accumulation class, and the complex family adds one this gate
    # has never walked before: a plan whose `run` performs a HOST round trip (the
    # radial prefix scan) in the middle of the composition, which is the residency
    # invariant's hardest case rather than its easiest.
    ("dcyl_m1_complex", lambda: matrix.cylindrical(m=1),
     "cylindrical complex |m| = 1, z METALLIC — WALLED, and the prefix scan syncs "
     "one volume out and one in inside every curl launch"),
    ("dcyl_m1_complex_negative", lambda: matrix.cylindrical(m=-1),
     "m = -1: the i*m/r row's sign flips AND the axis-increment scalar carries a "
     "-0.0 in its real word, which is the operand class a plane-wise product gets "
     "wrong"),
    ("dcyl_m1_complex_zperiodic",
     lambda: matrix.cylindrical(m=1, z_kind="periodic"),
     "the other BCZ arm, and the one Triton's own predicate refuses by name — the "
     "8 Metal-only slots this round reports live on this shape"),
    ("dcyl_m3_complex", lambda: matrix.cylindrical(m=3),
     "|m| = 3: the M_MANY arm, whose near-axis zero-row count is 3 rather than 1"),
    ("dcyl_m0_real", lambda: matrix.cylindrical(m=0, complex_storage=False),
     "cylindrical REAL m = 0, z METALLIC — WALLED, and the only plan in the tree "
     "that dispatches TWICE per sub-step (the radial scan, then the curl)"),
    ("dcyl_m0_real_zperiodic",
     lambda: matrix.cylindrical(m=0, complex_storage=False, z_kind="periodic"),
     "the real family's other BCZ arm, unwalled"),
)

#: Rows whose composition is deliberately PARTIAL — some slots stay on the array
#: path — because that is the shape the residency clause exists for.
MIXED_CASES: Tuple[Tuple[str, Any, str], ...] = (
    # `beta_complex_2d` LEFT THIS LIST ON 2026-08-19 for the full-step list above:
    # its two constitutive slots stopped being on the array path. The row below is
    # the beta family's remaining partial composition and the closest replacement —
    # a real beta run with a susceptibility, where the ADE E leg is another family's
    # and `update_E` is the one host write a mirror is held across.
    ("beta_real_2d_dispersive", lambda: matrix.dispersive(matrix.flat(beta=0.33)),
     "the real beta curls, update_H and update_P on the device, update_E on the "
     "array path (real-storage ADE E is a separate family)"),
    ("cart_pml_complex_offdiag",
     lambda: matrix.cart(complex_storage=True, rows={"Ex": ("Ey",)}),
     "complex curls and update_H on the device, update_E on the array path"),
    # THE NULL PAIR IS A MIXED CASE, not a full-step one, and that is a MEASURED
    # coverage gap rather than a quirk of this list. Under an inactive absorber the
    # constitutive pair is null — correct, and it launches nothing — but BOTH CURLS
    # fall to the array path, because no Metal product carries the plain (no-PML)
    # curl at all. The Triton track has one (`triton_kernels/no_pml.py`); this
    # backend does not, so a no-PML run is a two-sync-per-step composition today.
    ("nopml_real_nostore", lambda: matrix.cart(pml=0, storage=False, eps=False),
     "the no-PML NULL pair, with both curls on the array path (no Metal plain-curl "
     "product exists)"),
    # THE WALLED FOLD, and it is a MIXED case for a reason worth stating: the two
    # fill slots are refused BY NAME on a walled run, because `zero_metal_*` runs
    # BETWEEN the near and far passes (driver.py:3285/:3286) and a plan may not fuse
    # across it. So the four arithmetic slots compose, both fills fall to the array
    # path with syncs, and the ten-pass order is walked with the wall clear in its
    # real position. This is the row where the fold's seam and the wall's seam are
    # live in the same step.
    ("fold_real_2d_walled", lambda: matrix.folded(boundaries={"x": "metallic"}),
     "a WALLED fold: four arithmetic slots on the device, both ghost passes on the "
     "array path (the wall clear sits between them)"),
    ("fold_complex_2d_walled",
     lambda: matrix.folded(complex_storage=True, boundaries={"x": "metallic"}),
     "the same seam under complex storage, where the fill helper is imported from "
     "the real family rather than restated"),
    # A folded run with NO absorber: the fills compose, the null pair takes the
    # constitutive slots and BOTH CURLS fall to the array path — this backend
    # carries no plain (no-PML) curl. The inverse partial composition of the row
    # above, and the one where a mirror is held across a host curl write.
    ("fold_real_2d_no_pml", lambda: matrix.folded_no_pml(),
     "a folded no-PML run: the two ghost passes on the device, the null "
     "constitutive pair, both curls on the array path"),
    # -- TRANCHE 4 --------------------------------------------------------------
    # THE WALLED FOLDED BETA RUN, both storages. Ten passes with the wall clear in
    # its real position BETWEEN the two ghost passes, which is why both fill slots
    # are refused by name and run on the array path with syncs. This is the row
    # where the fold's seam, the wall's seam and the beta term are live in one step
    # — three of the four classes at once — and it is the folded-beta family's only
    # partial composition.
    ("fold_real_2d_beta_walled",
     lambda: matrix.folded(beta=0.3, boundaries={"x": "metallic"}),
     "a WALLED folded beta run: four arithmetic slots on the device, both ghost "
     "passes on the array path with the wall clear between them"),
    ("fold_complex_2d_beta_walled",
     lambda: matrix.folded(complex_storage=True, beta=0.3,
                           boundaries={"x": "metallic"}),
     "the same seam under complex storage"),
    # THE PADE FAMILY IS A ONE-SLOT COMPOSITION and therefore only ever a mixed
    # case: the SHARED clause 10 refuses chi2/chi3 on every sub-step, so the two
    # curls and `update_H` stay on the array path while `update_E` runs on the
    # device. That makes it the strictest residency row in the gate — a mirror is
    # held across THREE host writes per step, not one.
    ("nonlinear_real", lambda: matrix.nonlinear(matrix.cart()),
     "the chi2/chi3 Pade update_E alone on the device, with both curls and "
     "update_H on the array path"),
    ("nonlinear_real_walled",
     lambda: matrix.nonlinear(matrix.cart(boundaries="metallic")),
     "the same, WALLED: `zero_metal_D` writes D between the host curl and the "
     "device update_E that reads it"),
    # THE NULL PAIR SPLITTING ON A NONLINEAR RUN. `update_H` is the null's and
    # `update_E` is NOBODY'S, because the null refuses the Pade factor by name.
    # Nothing dispatches on this row at all — the null performs no device work — so
    # it is here for the residency verdict and the byte compare rather than for a
    # launch count, and `assert_case` reads the null's `runs` counter instead.
    ("nonlinear_no_pml",
     lambda: matrix.nonlinear(matrix.cart(pml=0, storage=False, eps=False)),
     "a nonlinear run with an INACTIVE absorber: the null takes update_H, update_E "
     "falls to the array path, and nothing is dispatched"),
    # A CONDUCTIVITY ON A CYLINDRICAL RUN. The two curls fall to the array path (a
    # conductivity is a curl-only refusal) and the constitutive pair composes, so a
    # cylindrical mirror is held across two host curl writes per step. The scope of
    # the refusal is the property; a row where the conductivity also sank the
    # constitutive slots would be a coverage loss inside a correct answer.
    ("dcyl_m0_real_conductive",
     lambda: matrix.conductive(matrix.cylindrical(m=0, complex_storage=False)),
     "cylindrical m = 0 with an electric conductivity: both curls on the array "
     "path, the constitutive pair on the device, WALLED"),
)


# ---------------------------------------------------------------------------
# LEG 0 — the subject
# ---------------------------------------------------------------------------

def leg_arms(payload: Dict[str, Any], out: str) -> None:
    table = [{"slot": slot, "family": spec.family, "label": spec.label,
              "wired": bool(spec.wired)}
             for slot in arms.registered_slots()
             for spec in arms.registered(slot)]
    wired = [row for row in table if row["wired"]]
    payload["legs"]["arms"] = {
        "table": table,
        "arms_total": len(table),
        "arms_wired": len(wired),
        "slots": list(arms.registered_slots()),
        "environment": dict(ENVIRONMENT),
        "state_names": list(STATE_NAMES),
        "cycles": CYCLES,
    }
    save(payload, out)
    log(f"[leg0 arms] {len(table)} registered arms ({len(wired)} wired) over "
        f"{len(arms.registered_slots())} slots; {len(STATE_NAMES)} state volumes")
    assert len(wired) >= 20, f"only {len(wired)} wired arms — the table is short"
    # The state list must not be empty and must not be a hand-list that drifted:
    # it is derived from the composer's own residency table.
    assert "Bx" in STATE_NAMES and "f_w_Ex" in STATE_NAMES, STATE_NAMES


# ---------------------------------------------------------------------------
# LEG 1 / 2 — whole steps
# ---------------------------------------------------------------------------

def run_case(label: str, build: Any, why: str, probes: Dict[str, Any],
             cycles: int = CYCLES) -> Dict[str, Any]:
    """One configuration: array path and Metal composition in lockstep.

    TWO INDEPENDENT ENGINE OBJECTS, built from the same seeds, rather than one
    object snapshotted and restored. Restoring writes the host arrays a mirror
    shadows, so a snapshot/restore harness has to remember to re-sync — and a
    harness that forgets produces a green run for the wrong reason. Two objects
    cannot have that bug.
    """
    oracle_fields, oracle_pml = build()
    metal_fields, metal_pml = build()

    initial = frozen(oracle_fields)
    identical_at_start = compare(oracle_fields, metal_fields)
    assert not identical_at_start, (
        f"{label}: the two builds are not identical before stepping — the case "
        f"matrix is not deterministic and every comparison below is meaningless: "
        f"{identical_at_start}")

    live = live_slots(metal_fields, metal_pml)
    plan, residency, synced = compose(metal_fields, metal_pml, probes)

    per_step: List[Dict[str, Any]] = []
    first_divergence: Optional[int] = None
    for step in range(1, cycles + 1):
        run_array_step(oracle_fields, oracle_pml, live)
        run_metal_step(metal_fields, metal_pml, plan, residency, live)
        residency.sync_out()
        bad = compare(oracle_fields, metal_fields)
        per_step.append({"step": step, "differing": bad})
        if bad and first_divergence is None:
            first_divergence = step

    state_moved = moved(initial, oracle_fields)
    # PER-SLOT INVOCATION COUNTERS, and the attribute differs by plan KIND rather
    # than by family name. A kernel plan counts `launches` (the dispatch actually
    # happened); the no-PML NULL plan launches nothing at all by construction and
    # counts `runs` instead, which is its own gate's stated evidence. Reading
    # `launches` alone reported 0 for a null slot and would have failed a correct
    # composition; reading either one and calling it "launches" would have claimed
    # a dispatch that never occurred. Both are recorded, with the kind.
    launches = {}
    covered = covered_passes(plan)
    for slot, product in plan.plans.items():
        device_work = bool(getattr(product, "performs_device_work", True))
        counter = "launches" if device_work else "runs"
        # HOW MANY DISPATCHES ONE CYCLE OWES THIS SLOT. One for an ordinary
        # sub-step; for a FILL it is one per folded axis per pass it owns, because
        # the fill walks axes in X, Y, Z order and each is its own launch. Derived
        # from the plan's own axis entries rather than assumed to be one, so a
        # two-axis fold is required to dispatch four times per step and a plan that
        # quietly skipped an axis cannot satisfy the counter.
        near = getattr(product, "near", None)
        # A NON-FILL PLAN DECLARES ITS OWN DISPATCH COUNT rather than being assumed
        # to launch once. `cylindrical_real`'s curl plan dispatches TWICE per run —
        # the radial scan, then the curl that reads it — and reading 1 here would
        # have failed a correct composition, while relaxing the assertion to ">= 1"
        # would have let a slot pass by half-executing. `launches_per_run` is the
        # plan's own declaration (plans.KernelPlan and CylindricalRealCurlPlan).
        expected = int(getattr(product, "launches_per_run", 1))
        if near is not None:
            far = getattr(product, "far", ()) or ()
            far_slot = FILL_SLOT_FAR.get(slot)
            expected = len(near) + (len(far) if far_slot in covered else 0)
        launches[slot] = {"count": int(getattr(product, counter, 0)),
                          "counter": counter,
                          "per_cycle_expected": int(expected),
                          "performs_device_work": device_work}
    drift = residency.verify()

    grid = metal_fields.grid
    return {
        "case": label,
        "why": why,
        "shape": [int(v) for v in grid.shape],
        "courant": float(getattr(grid, "courant", 0.0)),
        "cycles": cycles,
        "folded": bool(grid.has_symmetry()
                       and any(grid.is_mirrored(axis) for axis in range(3))),
        "folded_periodic": bool(any(metal_coverage.folded_periodic_axes(grid))),
        "walled": bool(any(metal_coverage.zero_metal_axes(grid))),
        "live": list(live),
        "selected": dict(plan.selected),
        "synced": list(synced),
        "on_array_path": [slot for slot in live
                          if slot not in covered_passes(plan)],
        "mirrors": len(residency.names),
        "syncs": [int(residency.syncs_out), int(residency.syncs_in)],
        "launches": launches,
        "moved_words": int(state_moved),
        "compared_words": int(sum(words(v).size for v in state_of(oracle_fields).values())),
        "first_divergence": first_divergence,
        "per_step": per_step,
        "mirror_drift": drift,
    }


def assert_case(row: Dict[str, Any], expect_full: bool) -> None:
    label = row["case"]
    assert row["moved_words"] > 0, (
        f"VACUOUS: {label} moved no state over {row['cycles']} steps — a no-op "
        f"agreeing with a no-op is trivially identical")
    assert row["first_divergence"] is None, (
        f"{label}: first divergence at complete step {row['first_divergence']}: "
        f"{row['per_step'][row['first_divergence'] - 1]['differing']}")
    assert not row["mirror_drift"], f"{label}: residency drift {row['mirror_drift']}"
    # EVERY FILLED SLOT MUST HAVE RUN, once per cycle. A plan that was composed and
    # never invoked makes the byte compare a comparison of the array path with
    # itself, which is the hollow pass this assertion exists to refuse. The counter
    # is the plan kind's own — `launches` for a kernel, `runs` for the null — so
    # this cannot be satisfied by reading whichever attribute happens to be nonzero.
    for slot, record in row["launches"].items():
        owed = record["per_cycle_expected"] * row["cycles"]
        assert record["count"] == owed, (
            f"{label}: {slot} recorded {record['count']} {record['counter']} over "
            f"{row['cycles']} cycles; the plan owes "
            f"{record['per_cycle_expected']} per cycle = {owed}")
    device_slots = [slot for slot, record in row["launches"].items()
                    if record["performs_device_work"]]
    assert device_slots or not row["selected"] or all(
        not record["performs_device_work"] for record in row["launches"].values()), (
        f"{label}: slots were filled but none dispatches")
    if expect_full:
        # The wall clear is the one pass a full composition may leave on the array
        # path: no Metal product carries `zero_metal_*` and none is claimed to.
        assert not row["on_array_path"] or set(row["on_array_path"]) <= {
            "zero_metal_B", "zero_metal_D"}, (
            f"{label}: expected a full-step composition, but "
            f"{row['on_array_path']} stayed on the array path")
    # A FOLDED CASE MUST HAVE STEPPED TEN PASSES, NOT SIX. Reported as its own
    # assertion rather than folded into `expect_full` because the failure it
    # catches is the opposite one: a fold whose ghost passes were never in the walk
    # at all would compose, launch, compare and PASS — the array-path oracle would
    # not run them either, so both sides would be equally wrong and identical.
    if row["folded"]:
        assert "fill_B" in row["live"] and "fill_D" in row["live"], (
            f"{label}: a folded configuration whose live set is {row['live']} — "
            f"the ghost passes are missing from BOTH sides of the comparison, so "
            f"the fold is not being measured at all")
        if row["folded_periodic"]:
            assert "fill_folded_far_ghosts_B" in row["live"], (
                f"{label}: a folded PERIODIC axis with no far ghost pass live")


def leg_whole_step(payload: Dict[str, Any], out: str, probes: Dict[str, Any],
                   cases: Sequence[Any], key: str, expect_full: bool) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, build, why in cases:
        row = run_case(label, build, why, probes)
        rows.append(row)
        payload["legs"][key] = rows
        save(payload, out)
        log(f"[{key}] {label:<28} "
            f"selected={len(row['selected'])}/{len(row['live'])} "
            f"array_path={row['on_array_path']} "
            f"moved={row['moved_words']:<8} mirrors={row['mirrors']} "
            f"runs={sum(r['count'] for r in row['launches'].values())} "
            f"{'IDENTICAL' if row['first_divergence'] is None else '*** DIVERGES step ' + str(row['first_divergence'])}"
            f" ({time.time() - started:.1f}s)")
        assert_case(row, expect_full)


# ---------------------------------------------------------------------------
# LEG 3 — the armed leg: the comparison must be able to FAIL
# ---------------------------------------------------------------------------

def fold_order_mutations(probes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """THE FOLD'S OWN FAILURE CLASS: a ghost plane produced and consumed a pass apart.

    The three classes the rest of this leg arms are storage, seam and accumulation.
    A fold adds a fourth that none of them reaches and no per-sub-step gate can:
    the near fill writes cell 0 of every unowned plane and the NEXT sub-step reads
    it, and on a folded periodic axis the far pass writes another plane after the
    wall clear. Every kernel involved can be byte-perfect on its own launch while
    the STEP is wrong, because what changed is which value was in the plane when the
    reader ran.

    Both mutations here keep every shipped kernel exactly as it is and move only the
    ORDER, which is what makes them measurements of the composition rather than of
    the arithmetic:

    ``ghost_read_before_written``  ``update_H`` runs BEFORE the near fill. Asserted
                                   to be a PERMUTATION of the driver order, so it
                                   cannot silently be measuring a dropped pass.
    ``dropped_far_ghost_pass``     the far ghost pass is skipped on a folded
                                   PERIODIC run — the pass whose absence is
                                   invisible until the D curl's backward difference
                                   reaches the top row.
    """
    rows: List[Dict[str, Any]] = []
    for mutation, label, build in (
            ("ghost_read_before_written", "fold_real_2d", matrix.folded),
            ("dropped_far_ghost_pass", "fold_real_2d", matrix.folded)):
        oracle_fields, oracle_pml = build()
        metal_fields, metal_pml = build()
        live = live_slots(metal_fields, metal_pml)
        assert "fill_B" in live, f"{mutation} is DISARMED: no near fill is live"
        assert "fill_folded_far_ghosts_B" in live, (
            f"{mutation} is DISARMED: no far ghost pass is live, so this row is "
            f"not the MIRROR_PERIODIC shape it claims to be")

        if mutation == "ghost_read_before_written":
            order = [name for name in live if name != "fill_B"]
            order.insert(order.index("update_H") + 1, "fill_B")
            assert sorted(order) == sorted(live), (
                "the needle is not a permutation of the driver order — it drops or "
                "duplicates a pass, and would measure the wrong defect")
            assert tuple(order) != tuple(live), "the needle changed nothing"
        else:
            order = [name for name in live if name != "fill_folded_far_ghosts_B"]
            assert len(order) == len(live) - 1, "the needle dropped nothing"

        plan, residency, _ = compose(metal_fields, metal_pml, probes)
        launched_before = sum(int(getattr(p, "launches", 0))
                              for p in plan.plans.values())
        for _ in range(CYCLES):
            run_array_step(oracle_fields, oracle_pml, live)
            run_metal_step(metal_fields, metal_pml, plan, residency, order)
        residency.sync_out()
        bad = compare(oracle_fields, metal_fields)
        launched = sum(int(getattr(p, "launches", 0))
                       for p in plan.plans.values()) - launched_before
        rows.append({"mutation": mutation, "case": label,
                     "driver_order": list(live), "mutated_order": list(order),
                     "launches": launched,
                     "differing": {k: v for k, v in list(bad.items())[:6]},
                     "differing_arrays": len(bad),
                     "classification": "CAUGHT" if bad else "NEEDLE-MISSED"})
    return rows


def tranche_four_mutations(probes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The two composition defects the FOUR NEW FAMILIES made reachable.

    None of them is arithmetic — the four family gates own that — and none is
    visible to a per-sub-step gate, which is the bar for belonging in this leg.

    ``stale_cylindrical_prefix``   the complex cylindrical curl's ``run`` performs a
                                   HOST ROUND TRIP before its launch: it syncs the
                                   prefix SOURCE out, scans it on the host, and syncs
                                   the prefix mirror back in. Skipping that (taking
                                   the base's launch path directly) launches the same
                                   kernel against LAST STEP'S radial derivative — a
                                   smooth, plausible, entirely wrong field, and one a
                                   single-launch gate cannot see because the first
                                   launch's prefix is correct by construction.

    ``cylindrical_scan_after_curl`` the real m = 0 curl plan is the one plan in the
                                   tree that dispatches twice, and THE ORDER IS THE
                                   CONTRACT: the curl reads the prefix the scan
                                   writes. Inverting them keeps both launch counts
                                   exactly right, which is precisely why a counter
                                   cannot catch it and a whole-step compare must.

    The earlier ``dropped_sync_nonlinear`` mutation no longer belongs here: the
    nonlinear B/H/D/E spine is now fully device resident, so deleting a host-sync
    bracket would remove no operation.  Keeping it would be a deliberately
    disarmed mutation.  THE SAME THING HAPPENED TO THE COMPLEX BETA ROW on
    2026-08-19 — its constitutive pair left the array path — so ``dropped_sync``
    moved to the real beta DISPERSIVE composition, which still holds a mirror
    across one host write and asserts that it does before mutating anything.
    """
    from meep_gpu.metal_kernels.plans import KernelPlan  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []

    # --- stale_cylindrical_prefix -------------------------------------------
    label = "dcyl_m1_complex"
    oracle_fields, oracle_pml = matrix.cylindrical(m=1)
    metal_fields, metal_pml = matrix.cylindrical(m=1)
    live = live_slots(metal_fields, metal_pml)
    plan, residency, _ = compose(metal_fields, metal_pml, probes)
    curls = [slot for slot in ("step_B", "step_D") if slot in plan.plans]
    assert curls, "DISARMED: no cylindrical curl slot was composed"
    assert all(hasattr(plan.plans[slot], "refresh_prefix") for slot in curls), (
        "DISARMED: the composed curl plans carry no refresh_prefix, so this "
        "mutation would be removing a call that does not exist")
    syncs_before = int(sum(plan.plans[slot].prefix_syncs for slot in curls))
    launched = 0
    for _ in range(CYCLES):
        run_array_step(oracle_fields, oracle_pml, live)
        for slot in live:
            product = plan.plans.get(slot)
            if product is None:
                residency.sync_out()
                ARRAY_PATH[slot](metal_fields, metal_pml)
                residency.sync_in()
            elif slot in curls:
                KernelPlan.run(product, None)   # THE MUTATION: no prefix refresh
                launched += 1
            else:
                product.run()
                launched += 1
    residency.sync_out()
    bad = compare(oracle_fields, metal_fields)
    syncs_after = int(sum(plan.plans[slot].prefix_syncs for slot in curls))
    rows.append({"mutation": "stale_cylindrical_prefix", "case": label,
                 "launches": launched,
                 "prefix_syncs_performed": syncs_after - syncs_before,
                 "differing": {k: v for k, v in list(bad.items())[:6]},
                 "differing_arrays": len(bad),
                 "classification": "CAUGHT" if bad else "NEEDLE-MISSED"})

    # --- cylindrical_scan_after_curl ----------------------------------------
    label = "dcyl_m0_real"
    oracle_fields, oracle_pml = matrix.cylindrical(m=0, complex_storage=False)
    metal_fields, metal_pml = matrix.cylindrical(m=0, complex_storage=False)
    live = live_slots(metal_fields, metal_pml)
    plan, residency, _ = compose(metal_fields, metal_pml, probes)
    curls = [slot for slot in ("step_B", "step_D") if slot in plan.plans]
    assert curls and all(hasattr(plan.plans[slot], "run_prefix") for slot in curls), (
        "DISARMED: the composed real cylindrical curls expose no separate scan")
    launched = 0
    for _ in range(CYCLES):
        run_array_step(oracle_fields, oracle_pml, live)
        for slot in live:
            product = plan.plans.get(slot)
            if product is None:
                residency.sync_out()
                ARRAY_PATH[slot](metal_fields, metal_pml)
                residency.sync_in()
            elif slot in curls:
                product.run_curl()              # THE MUTATION: inverted order,
                product.run_prefix()            # same two launches
                launched += 2
            else:
                product.run()
                launched += 1
    residency.sync_out()
    bad = compare(oracle_fields, metal_fields)
    counted = sum(int(plan.plans[slot].launches) for slot in curls)
    rows.append({"mutation": "cylindrical_scan_after_curl", "case": label,
                 "launches": launched,
                 "curl_plan_launch_count": int(counted),
                 "launch_count_unchanged": bool(counted == 2 * CYCLES * len(curls)),
                 "differing": {k: v for k, v in list(bad.items())[:6]},
                 "differing_arrays": len(bad),
                 "classification": "CAUGHT" if bad else "NEEDLE-MISSED"})

    return rows


def leg_armed(payload: Dict[str, Any], out: str, probes: Dict[str, Any]) -> None:
    """Two mutations that MUST make the whole-step compare diverge.

    A green byte gate is evidence only if the same harness reports red when the
    composition is wrong. Both mutations here are composition-level rather than
    arithmetic — the arithmetic already has six family gates — and each is a defect
    a per-sub-step gate is structurally blind to:

    ``dropped_sync``  the array-path bracket is removed on a MIXED composition, so
                      a mirror is held across a host write. This is the residency
                      clause's whole reason for existing, and if it does not
                      diverge the clause is decorative.
    ``dropped_seam``  ``zero_metal_B`` is skipped on a walled run. The composition
                      then consumes the pre-seam value: byte-perfect per sub-step,
                      wrong per step.

    A mutation that does NOT diverge is reported as NEEDLE-MISSED and fails the leg,
    exactly as an armed mutation that never launched would be reported DISARMED.
    """
    rows: List[Dict[str, Any]] = []

    # --- dropped_sync, on a mixed composition -------------------------------
    # THE SUBSTRATE MOVED ON 2026-08-19. It used to be `beta_complex_2d`, whose two
    # constitutive slots were on the array path; `special_kz` gained its complex
    # constitutive companion, that row composes a whole step, and the mutation would
    # have had NO host write to drop — reported NEEDLE-MISSED, which is the honest
    # failure but not the measurement. The real beta DISPERSIVE row keeps the shape:
    # `update_E` stays on the array path, so a mirror is held across one host write.
    label = "beta_real_2d_dispersive"
    oracle_fields, oracle_pml = matrix.dispersive(matrix.flat(beta=0.33))
    metal_fields, metal_pml = matrix.dispersive(matrix.flat(beta=0.33))
    live = live_slots(metal_fields, metal_pml)
    plan, residency, _ = compose(metal_fields, metal_pml, probes)
    array_slots = [slot for slot in live if slot not in plan.plans]
    assert array_slots, (
        f"{label} is DISARMED as a dropped_sync substrate: every live slot "
        f"composed, so there is no host write for the mutation to strip the sync "
        f"from. Move this mutation to a row that is still partial")
    launched = 0
    for _ in range(CYCLES):
        run_array_step(oracle_fields, oracle_pml, live)
        for slot in live:
            if slot in plan.plans:
                plan.plans[slot].run()
                launched += 1
            else:
                # THE MUTATION: no sync_out / sync_in around the host write.
                ARRAY_PATH[slot](metal_fields, metal_pml)
    residency.sync_out()
    bad = compare(oracle_fields, metal_fields)
    rows.append({"mutation": "dropped_sync", "case": label,
                 "array_path_slots": array_slots, "launches": launched,
                 "differing": {k: v for k, v in list(bad.items())[:6]},
                 "differing_arrays": len(bad),
                 "classification": "CAUGHT" if bad else "NEEDLE-MISSED"})

    # --- dropped_seam, on a walled run --------------------------------------
    label = "cart_pml_real_metallic"
    oracle_fields, oracle_pml = matrix.cart(boundaries="metallic")
    metal_fields, metal_pml = matrix.cart(boundaries="metallic")
    live = live_slots(metal_fields, metal_pml)
    assert "zero_metal_B" in live, "the seam mutation is DISARMED: no wall pass"
    plan, residency, _ = compose(metal_fields, metal_pml, probes)
    launched = 0
    for _ in range(CYCLES):
        run_array_step(oracle_fields, oracle_pml, live)
        for slot in live:
            if slot == "zero_metal_B":
                continue                       # THE MUTATION: seam skipped
            if slot in plan.plans:
                plan.plans[slot].run()
                launched += 1
            else:
                residency.sync_out()
                ARRAY_PATH[slot](metal_fields, metal_pml)
                residency.sync_in()
    residency.sync_out()
    bad = compare(oracle_fields, metal_fields)
    rows.append({"mutation": "dropped_seam", "case": label,
                 "launches": launched,
                 "differing": {k: v for k, v in list(bad.items())[:6]},
                 "differing_arrays": len(bad),
                 "classification": "CAUGHT" if bad else "NEEDLE-MISSED"})

    rows.extend(fold_order_mutations(probes))
    rows.extend(tranche_four_mutations(probes))

    payload["legs"]["armed"] = rows
    save(payload, out)
    for row in rows:
        log(f"[leg3 armed] {row['mutation']:<14} {row['case']:<28} "
            f"launches={row['launches']:<4} arrays_differing={row['differing_arrays']:<3} "
            f"{row['classification']}")
        assert row["launches"] > 0, (
            f"DISARMED: {row['mutation']} launched no kernel, so it measured nothing")
        assert row["classification"] == "CAUGHT", (
            f"NEEDLE-MISSED: {row['mutation']} did not diverge — the whole-step "
            f"comparison cannot see this defect class and leg 1's pass is hollow "
            f"for it")


# ---------------------------------------------------------------------------
# LEG 5 — the folded tensor dispersion's live ADE-to-pack handoff
# ---------------------------------------------------------------------------

def leg_resident_pole_pack(payload: Dict[str, Any], out: str,
                           probes: Dict[str, Any]) -> None:
    """Prove the packed tensor poles do not require a per-step host round trip.

    The ordinary whole-step leg synchronizes after EACH step to compare the first
    divergent step.  That is appropriate for its broad arbitration claim, but it
    would mask this family's former error: copying the pack from NumPy works once
    the prior step has been synced out, yet reads stale pre-ADE bytes in a genuinely
    resident loop.  This narrow leg runs twelve whole steps without that sync.  Its
    armed twin restores the host-refresh route and must diverge, so an exact final
    state cannot be a pass caused by a never-exercised device mirror lookup.
    """
    from meep_gpu.metal_kernels import device as metal_device  # noqa: PLC0415

    def run(label: str, stale_host_refresh: bool) -> Dict[str, Any]:
        oracle_fields, oracle_pml = matrix.dispersive(
            matrix.folded(rows={"Ex": ("Ey",)}))
        metal_fields, metal_pml = matrix.dispersive(
            matrix.folded(rows={"Ex": ("Ey",)}))
        initial = frozen(oracle_fields)
        live = live_slots(metal_fields, metal_pml)
        plan, residency, _ = compose(metal_fields, metal_pml, probes)
        tensor_e = plan.plans.get("update_E")
        assert tensor_e is not None and hasattr(tensor_e, "_refresh_packs"), (
            "DISARMED: the folded tensor-dispersive E plan was not selected")
        assert not [slot for slot in live if slot not in covered_passes(plan)], (
            "DISARMED: an array-path slot would add a host synchronization and "
            "invalidate the resident-pack measurement")

        original_lookup = metal_device.Residency.tensor_for_host
        if stale_host_refresh:
            metal_device.Residency.tensor_for_host = lambda self, host: None
        try:
            for _ in range(CYCLES):
                run_array_step(oracle_fields, oracle_pml, live)
                run_metal_step(metal_fields, metal_pml, plan, residency, live)
        finally:
            metal_device.Residency.tensor_for_host = original_lookup
        residency.sync_out()
        bad = compare(oracle_fields, metal_fields)
        return {
            "label": label,
            "cycles": CYCLES,
            "syncs": [int(residency.syncs_out), int(residency.syncs_in)],
            "moved_words": moved(initial, oracle_fields),
            "differing": {name: count for name, count in bad.items()},
            "differing_arrays": len(bad),
            "e_launches": int(tensor_e.launches),
            "classification": "CAUGHT" if bad else "IDENTICAL",
        }

    resident = run("device_to_device_pack_refresh", stale_host_refresh=False)
    stale = run("stale_host_pack_refresh", stale_host_refresh=True)
    payload["legs"]["resident_pole_pack"] = [resident, stale]
    save(payload, out)
    for row in (resident, stale):
        log(f"[leg5 resident_pack] {row['label']:<30} "
            f"syncs={row['syncs']} e_launches={row['e_launches']:<3} "
            f"arrays_differing={row['differing_arrays']:<3} {row['classification']}")
        assert row["syncs"] == [1, 0], (
            f"{row['label']}: expected only the final sync_out, got {row['syncs']}")
        assert row["moved_words"] > 0, f"{row['label']}: the oracle did not move"
        assert row["e_launches"] == CYCLES, (
            f"{row['label']}: tensor E launched {row['e_launches']} times")
    assert resident["classification"] == "IDENTICAL", resident["differing"]
    assert stale["classification"] == "CAUGHT", (
        "NEEDLE-MISSED: host-staged pole packs survived twelve resident steps; "
        "the device refresh route is not demonstrated")


# ---------------------------------------------------------------------------
# LEG 4 — the checked precondition, as a WINDOW
# ---------------------------------------------------------------------------

def leg_subnormal(payload: Dict[str, Any], out: str, probes: Dict[str, Any],
                  cycles: int = 24) -> None:
    """Census every state array on every step; report the WINDOW, not a scalar.

    WHY A WINDOW. The chi3 round measured a real corpus row whose band entry was a
    TRANSIENT — first subnormal at step 55, last at step 3,726, then clean for the
    remaining 16,274 steps — driven by a narrow-band source envelope whose turn-on
    sweeps forty decades. A gate that censused once, or that reported a single
    boolean over a run, would have called that run clean or dirty depending on
    where it happened to look. First step, last step and live count per step are
    what make the precondition CHECKED rather than assumed.

    THE FLOORS ARE THE VACUITY DISCIPLINE. A census that observed nothing is not
    clean, it is empty. ``observed_words`` must exceed a floor per case, and the
    floor is asserted.
    """
    rows: List[Dict[str, Any]] = []
    for label, build, _why in CASES:
        fields, pml = build()
        live = live_slots(fields, pml)
        plan, residency, _ = compose(fields, pml, probes)
        window = preconditions.SubnormalWindow(first_step=0, last_step=cycles,
                                               per_array_words=1)
        observed = 0
        found_total = 0
        first_step: Optional[int] = None
        last_step: Optional[int] = None
        per_step: List[int] = []
        for step in range(1, cycles + 1):
            run_metal_step(fields, pml, plan, residency, live)
            residency.sync_out()
            step_found = 0
            for name, array in state_of(fields).items():
                found = window.observe(f"{label}:{name}", array, step=step)
                step_found += found
                observed += int(words(array).size)
            per_step.append(step_found)
            found_total += step_found
            if step_found:
                first_step = step if first_step is None else first_step
                last_step = step
        rows.append({
            "case": label, "steps": cycles,
            "observed_words": observed,
            "subnormal_words": found_total,
            "window": [first_step, last_step],
            "per_step_counts": per_step,
            "smallest_nonzero": float(min(
                (float(np.min(np.abs(a)[np.abs(a) > 0]))
                 for a in state_of(fields).values()
                 if np.any(np.abs(np.asarray(a)) > 0)), default=0.0)),
        })
        payload["legs"]["subnormal"] = rows
        save(payload, out)
        row = rows[-1]
        log(f"[leg4 subnormal] {label:<28} observed={row['observed_words']:<10} "
            f"subnormal={row['subnormal_words']:<6} window={row['window']} "
            f"min|f|={row['smallest_nonzero']:.3e}")
        assert row["observed_words"] > 10_000, (
            f"VACUOUS CENSUS: {label} observed only {row['observed_words']} words")
        assert row["subnormal_words"] == 0, (
            f"{label}: the subnormal-free precondition FAILED — {row['subnormal_words']} "
            f"words in the band over steps {row['window']}. The byte claim does not "
            f"hold for this run and must not be made for it")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        API_ROOT, "parity/meep_gpu/results/metal_whole_step_2026-08-16"))
    parser.add_argument("--legs", default="all")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "whole_step.json")

    import platform  # noqa: PLC0415

    import torch  # noqa: PLC0415

    payload: Dict[str, Any] = {
        "provenance": {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "mps_available": bool(torch.backends.mps.is_available()),
            "metal_frontend": launch.metal_frontend_version(),
            "environment": dict(ENVIRONMENT),
            "subnormal_policy": subnormal.mps_policy_report(),
        },
        "legs": {},
    }
    probes = probe_records()
    wanted = set(args.legs.split(",")) if args.legs != "all" else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    started = time.time()
    if want("arms"):
        leg_arms(payload, path)
    if want("whole_step"):
        leg_whole_step(payload, path, probes, CASES, "whole_step", expect_full=True)
    if want("mixed"):
        leg_whole_step(payload, path, probes, MIXED_CASES, "mixed",
                       expect_full=False)
    if want("armed"):
        leg_armed(payload, path, probes)
    if want("subnormal"):
        leg_subnormal(payload, path, probes)
    if want("resident_pole_pack"):
        leg_resident_pole_pack(payload, path, probes)

    # THE HEADLINE NUMBER, computed from the record rather than quoted: how many
    # uint32 words this gate actually compared. A gate whose comparison count is
    # not reported cannot be told apart from one that compared a handful.
    compared = 0
    cases = 0
    for key in ("whole_step", "mixed"):
        for row in payload["legs"].get(key, []):
            compared += int(row["compared_words"]) * int(row["cycles"])
            cases += 1
    censused = sum(int(row["observed_words"])
                   for row in payload["legs"].get("subnormal", []))
    payload["totals"] = {
        "cases": cases,
        "cycles_per_case": CYCLES,
        "uint32_comparisons": compared,
        "subnormal_censused_words": censused,
        "divergences": [row["case"] for key in ("whole_step", "mixed")
                        for row in payload["legs"].get(key, [])
                        if row["first_divergence"] is not None],
    }
    payload["elapsed_s"] = round(time.time() - started, 2)
    save(payload, path)
    log("")
    log("METAL WHOLE-STEP GATE")
    log(f"  cases (full + mixed)     : {cases}")
    log(f"  complete steps per case  : {CYCLES}")
    log(f"  UINT32 COMPARISONS       : {payload['totals']['uint32_comparisons']:,}")
    log(f"  subnormal words censused : {censused:,}")
    log(f"  divergences              : {payload['totals']['divergences'] or 'none'}")
    log(f"  elapsed                  : {payload['elapsed_s']}s -> {path}")
    return 0


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
