"""The Metal composer, as a test that cannot go stale.

WHAT THIS PINS THAT THE FAMILY SUITES CANNOT. Each family suite checks ONE
predicate against configurations that family cares about. Composition is the
question nobody's own suite asks: given a configuration, do TWO products claim the
same sub-step, and if so does the composer refuse rather than pick? A tree of eight
certified families in which any two co-admit is a tree that dispatches by import
order, and no family suite can see that.

THREE THINGS ARE HELD HERE:

1. **The matrix** — every configuration in ``metal_composition_matrix.MATRIX``,
   each with the EXACT ``selected`` dict the composer produces and the slots it
   leaves ambiguous. The count is deliberately not spelled here: a family that adds
   rows would otherwise have to edit this sentence, and a stale number reads as a
   claim about coverage. Shared with
   ``parity/meep_gpu/probe_metal_planner_composition.py`` through
   ``parity/meep_gpu/metal_composition_matrix.py``, so the artifact and the test
   cannot describe different things.
2. **The fail-closed contract** — planted double-admits, raising predicates,
   raising builders, builders returning None, and a composer that never raises on
   a degenerate object.
3. **The table's completeness** — that the arm table is populated by import and
   that reading it without ``ensure_registered`` is the defect it looks like.

WHY THE MATRIX LIVES UNDER ``parity/`` AND THE ASSERTIONS LIVE HERE. The
configurations are also the whole-step gate's cases, and a configuration list that
exists twice is a configuration list that diverges. The gate imports the same
module.

THE ENVIRONMENT IS A PRECONDITION, not a convenience. On MPS the float32 subnormal
flush is native and has no lever, so the resolved default ``keep`` makes EVERY
Metal predicate refuse by name and the whole matrix would collapse to "nothing
admits anything" — which would pass a naive disjointness check while measuring
nothing. ``prepare_environment`` sets ``flush`` and the two measured expansion
probe paths, and :func:`test_the_environment_this_suite_needs_is_present` fails
loudly if any of the three is missing rather than letting the suite go quietly
vacuous.
"""

from __future__ import annotations

import os
import sys

import pytest

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "parity", "meep_gpu")
_PARITY = os.path.abspath(_PARITY)
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu.metal_kernels import arms, device, launch  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()


def _probes():
    from meep_gpu.metal_kernels import (
        complex_fields,
        cylindrical_complex,
        folded_complex,
        special_kz,
    )

    return {"complex_probe": complex_fields.load_expansion_probe(),
            "beta_probe": special_kz.load_expansion_probe(),
            "folded_complex_probe": folded_complex.load_expansion_probe(),
            "cylindrical_complex_probe":
                cylindrical_complex.load_expansion_probe()}


def _plan(fields, pml, **kwargs):
    probes = _probes()
    probes.update(kwargs)
    return launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=(), **probes)


def _ambiguous(plan):
    return tuple(sorted(
        slot for slot, reasons in plan.reasons.items()
        if any("admit this configuration" in reason for reason in reasons)))


# ---------------------------------------------------------------------------
# 0. The precondition this suite runs under
# ---------------------------------------------------------------------------

def test_the_environment_this_suite_needs_is_present():
    """A vacuous suite must fail rather than pass quietly.

    Without ``flush`` every predicate refuses on the policy clause; without the
    FOUR expansion artifacts seven of the sixteen products refuse on the probe
    clause. In either case the matrix below still "passes" a naive read — nothing
    co-admits when nothing admits — so the precondition is asserted before anything
    else.

    Each artifact is asserted SEPARATELY rather than assumed to come with an
    earlier one, because each carries an operand orientation its predecessors never
    classified: the folded-complex artifact a complex coefficient on the LEFT, and
    the cylindrical-complex artifact a complex ROW on the left plus a complex scalar
    on the left whose real word is a SIGNED ZERO. An artifact cut before a tranche
    satisfies neither that family nor this check.
    """
    assert os.environ.get("MEEP_GPU_SUBNORMAL_POLICY") == "flush", (
        "the MPS executor cannot honour 'keep'; without an explicit flush every "
        "Metal predicate refuses and this suite measures nothing")
    assert ENVIRONMENT["MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"], (
        "no complex expansion probe artifact on this host")
    assert ENVIRONMENT["MEEP_GPU_METAL_EXPANSION_PROBE"], (
        "no beta expansion probe artifact on this host")
    assert ENVIRONMENT["MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE"], (
        "no folded-complex expansion probe artifact on this host")
    assert ENVIRONMENT["MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE"], (
        "no cylindrical-complex expansion probe artifact on this host")
    probes = _probes()
    assert (probes["complex_probe"] and probes["beta_probe"]
            and probes["folded_complex_probe"]
            and probes["cylindrical_complex_probe"])


# ---------------------------------------------------------------------------
# 1. The matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,build,expected,ambiguous", matrix.MATRIX,
                         ids=[row[0] for row in matrix.MATRIX])
def test_every_slot_resolves_to_at_most_one_product(label, build, expected,
                                                    ambiguous):
    """``selected`` names the slot's ONE admitter; anything else is unselected.

    A slot absent from ``selected`` and carrying an "admit this configuration"
    reason had TWO OR MORE admitters and failed closed. Any slot that becomes
    ambiguous without being declared here — or stops being ambiguous while declared
    — is a predicate defect and fails.
    """
    fields, pml = build()
    plan = _plan(fields, pml)
    assert plan.selected == expected, (label, plan.reasons)
    assert _ambiguous(plan) == tuple(ambiguous), (label, plan.reasons)
    for slot in ambiguous:
        reason, = plan.reasons[slot]
        assert reason.count(",") >= 1, reason


def test_every_wired_arm_wins_at_least_one_configuration():
    """NON-VACUITY. A matrix in which an arm never wins measures nothing about it.

    Read off the recorded expectations rather than re-run, so this stays cheap and
    still fails the moment an arm is registered without a configuration that
    exercises it.
    """
    winners = set()
    for _label, _build, expected, _ambiguous in matrix.MATRIX:
        winners.update(expected.values())
    for label in matrix.EXPECTED_WINNERS:
        assert label in winners, label

    registered = {spec.label for spec in arms.registered() if spec.wired}
    missing = registered - winners
    assert not missing, (
        f"{sorted(missing)} are wired but win no configuration in the matrix; a "
        f"wired arm nothing exercises is a product the composition claim does not "
        f"cover")


@pytest.mark.parametrize("label,family", sorted(matrix.UNCARRIED.items()),
                         ids=sorted(matrix.UNCARRIED))
def test_a_family_this_backend_does_not_carry_selects_nothing(label, family):
    """THE OTHER HALF OF THE NON-VACUITY FLOOR, and the half that catches a WRONG
    ANSWER rather than a missed opportunity.

    :func:`test_every_wired_arm_wins_at_least_one_configuration` fails when an arm
    fires nowhere. This fails when an arm fires WHERE NO PRODUCT EXISTS — a
    cylindrical, nonlinear, folded-beta or folded-BFAST configuration quietly
    admitted by some arm whose clause list forgot to ask. That arm would step a
    recurrence this backend does not implement, and every downstream number would
    report it as coverage.

    ONLY THE FOUR ARITHMETIC SLOTS ARE CHECKED. A seam fill copies
    ``phase * field[2]`` and reads no beta term, no BFAST term, no pole and no
    nonlinearity, so a fill still composing on such a row is the CORRECT scoped
    answer — the matrix pins exactly that for the folded beta and BFAST rows — and
    demanding an empty selection there would be demanding a refusal of a pass the
    missing product does not own.

    The expectation is read off the matrix rather than re-composed, for the reason
    the winner floor is: it must fail the moment a row is pinned with a selection,
    not only when a sweep is run.
    """
    for row_label, _build, expected, _ambiguous in matrix.MATRIX:
        if row_label != label:
            continue
        selected = {slot: arm for slot, arm in expected.items()
                    if slot in matrix.ARITHMETIC_SLOTS}
        assert not selected, (
            f"{label} is a {family} configuration — a family this backend does "
            f"not carry — yet the matrix pins {selected} on its arithmetic slots")
        return
    raise AssertionError(
        f"{label} is named in UNCARRIED but is not a row of MATRIX; the floor "
        f"would silently check nothing")


def test_every_driver_slot_has_a_registered_product():
    """The ADE family closes the last formerly unregistered driver slot."""
    assert matrix.UNREGISTERED_SLOTS == ()
    assert set(launch.STEP_ORDER) <= set(arms.registered_slots())


@pytest.mark.parametrize("base,claimants", (
    (matrix.cart, ("ordinary", "offdiag")),
    (matrix.folded, ("folded", "folded offdiag")),
), ids=("unfolded", "folded"))
def test_the_planted_double_admits_name_both_claimants(base, claimants):
    """THE TWO overlaps in the matrix, and both are unreachable on the engine.

    ``has_offdiagonal_epsilon`` is a read-only property over the same dict
    ``chi1inv_offdiagonal_for`` reads, so flag and slots move together and the
    engine cannot produce the disagreement. Each pair is planted because its two
    predicates are disjoint only through that coupling — not through an inverted
    clause — and the composer must refuse rather than pick.

    THE SECOND ROW IS THE SAME COUPLING ONE LEVEL UP. ``folded`` refuses an
    off-diagonal row on ``update_E`` because its constitutive body is element-wise,
    and ``folded offdiag`` requires a LIVE ROW SLOT because its plan builder raises
    without one. Requiring the FLAG there instead would look tidier and would hand
    the builder a configuration it raises on, so the residual is carried and
    measured rather than papered over.
    """
    fields, pml = matrix.plant_rows(base())
    plan = _plan(fields, pml)
    assert "update_E" not in plan.plans
    reason, = plan.reasons["update_E"]
    for claimant in claimants:
        assert claimant in reason, (claimant, reason)
    assert "refuses rather than picking by table order" in reason


@pytest.mark.parametrize("base", (matrix.cart, matrix.folded),
                         ids=("unfolded", "folded"))
def test_the_reachable_mirror_of_the_planted_case_has_no_ambiguity(base):
    """Flag True with every row slot dead: no arm admits, and no ambiguity.

    On the FOLDED half the two arms decline differently and that is the point:
    ``folded`` REFUSES BY NAME on the flag, while ``folded offdiag`` is GATED OUT
    (its gate asks for a live slot), so it contributes no reason at all. Either way
    ``update_E`` falls to the array path, which is always correct.
    """
    fields, pml = matrix.flag_only(base())
    plan = _plan(fields, pml)
    assert "update_E" not in plan.plans
    assert _ambiguous(plan) == ()
    assert len(plan.reasons["update_E"]) >= 2


# ---------------------------------------------------------------------------
# 2. The table is populated by import, and reading it early is a defect
# ---------------------------------------------------------------------------

def test_the_table_carries_every_certified_family_over_all_driver_slots():
    """Every certified family is in the table, and the table's SLOT SET is pinned.

    The slot set grew from four to six when the folded family landed: it is the
    first Metal family to fill a SEAM slot. The set is asserted exactly rather than
    as a subset, because a slot appearing without a family behind it is how a
    ``NO_ARM_REASONS`` entry silently stops being consulted.

    THE TWO SEAM SLOTS NOW CARRY TWO ARMS, and that pair is pinned by name because it
    is the one place in the package where two arms invert on STORAGE alone:
    ``folded`` needs real float32 (where the fold's parity is an exact sign flip) and
    ``folded_complex`` needs complex64 (where it is a full complex product with zero
    cross terms). Both are deliberately UNGATED so whichever one refuses says which
    storage it needed; a third arm here would need a third storage.
    """
    expected_families = {
        "pml_curl", "constitutive", "no_pml_curl", "no_pml_constitutive",
        "complex_fields",
        "special_kz_real", "special_kz_complex", "bfast_curl",
        "offdiag_constitutive", "folded", "folded_offdiag_constitutive",
        "folded_complex", "ade_update_p",
    }
    families = {spec.family for spec in arms.registered()}
    assert expected_families <= families, sorted(expected_families - families)
    assert set(arms.registered_slots()) == {"step_B", "step_D", "update_H",
                                            "update_E", "fill_B", "fill_D",
                                            "update_P"}
    for slot in ("fill_B", "fill_D"):
        assert {spec.family for spec in arms.registered(slot)} == {
            "folded", "folded_complex"}, slot
        assert all(spec.gate is None for spec in arms.registered(slot)), slot


def test_no_slot_carries_two_arms_from_one_family():
    """Self-ambiguity is the failure the registry refuses at registration time.

    Two rows from one family on one slot would leave that slot permanently
    UNSELECTED — a family silently disabling itself — so ``arms.register`` refuses
    a duplicate (family, label). This pins the resulting invariant rather than the
    mechanism, so a future family that registers two DIFFERENTLY LABELLED arms on
    one slot is caught here too.
    """
    for slot in arms.registered_slots():
        families = [spec.family for spec in arms.registered(slot)]
        assert len(families) == len(set(families)), (slot, families)


def test_the_registry_module_names_every_module_that_registers_an_arm():
    """A family in the tree but not in ``registry`` is invisible to ``plan_step``.

    That is a silent COVERAGE LOSS rather than an error: the composer would leave
    the slot on the array path, which is always correct and never complained about.
    """
    from meep_gpu.metal_kernels import registry

    modules = {spec.coverage.__module__.rsplit(".", 1)[-1]
               for spec in arms.registered()}
    # `launch` registers the two tranche-1 products through closures defined in
    # `launch`; the rest name their own module.
    assert modules <= set(registry.FAMILY_MODULES) | {"launch", "coverage"}, modules


def test_every_module_on_disk_that_can_register_an_arm_is_in_family_modules():
    """The direction the subset check above CANNOT see, and it is the dangerous one.

    That check reads the table that registration produced, so it only ever sees
    modules that were IMPORTED — and ``registry``'s import list is what imports
    them. A family sitting in the package but missing from ``FAMILY_MODULES`` is
    therefore never imported, never registers, never appears in ``modules``, and
    passes the subset assertion VACUOUSLY while being invisible to ``plan_step``.
    That is a silent coverage loss: the composer leaves the slot on the array path,
    which is always correct and never complained about.

    So this asks the PACKAGE DIRECTORY instead, statically — a module with a
    top-level ``register_arms`` is a family, whether or not anything imports it —
    and no import runs, so a module absent from the list cannot hide by not being
    loaded.
    """
    import ast
    import os

    from meep_gpu.metal_kernels import registry

    package = os.path.dirname(os.path.abspath(registry.__file__))
    with_arms = set()
    for entry in sorted(os.listdir(package)):
        if not entry.endswith(".py") or entry == "__init__.py":
            continue
        with open(os.path.join(package, entry), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        if any(isinstance(node, ast.FunctionDef) and node.name == "register_arms"
               for node in tree.body):
            with_arms.add(entry[:-3])

    # Non-vacuity: a scan that found nothing would pass this test for the wrong
    # reason, which is the same defect one level up.
    assert with_arms, "no module in the package defines register_arms"
    missing = sorted(with_arms - set(registry.FAMILY_MODULES))
    assert not missing, (
        f"{missing} define register_arms but are absent from FAMILY_MODULES, so "
        f"nothing imports them, so their arms never reach the table and every slot "
        f"they would have filled falls silently to the array path")


def test_ensure_registered_refuses_a_read_from_inside_registration():
    """A family that queried the table during its own import would get an answer
    that depended on registration order — the same defect one level down."""
    state = arms.registration_state()
    assert state == "done"
    arms._REGISTRATION_STATE = "running"
    try:
        with pytest.raises(RuntimeError, match="still being populated"):
            arms.ensure_registered()
    finally:
        arms._REGISTRATION_STATE = state


# ---------------------------------------------------------------------------
# 3. Fail-closed: raising predicates, raising builders, degenerate objects
# ---------------------------------------------------------------------------

class _RaisingProxy:
    """A ``fields`` whose every attribute access explodes."""

    def __init__(self, wrapped, attribute):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_attribute", attribute)

    def __getattr__(self, name):
        if name == object.__getattribute__(self, "_attribute"):
            raise RuntimeError(f"{name} is unreadable")
        return getattr(object.__getattribute__(self, "_wrapped"), name)


@pytest.mark.parametrize("attribute", ["force_complex_fields", "stores_E",
                                       "has_offdiagonal_epsilon",
                                       "polarizations", "has_nonlinearity"])
def test_an_attribute_that_raises_on_access_never_escapes_plan_step(attribute):
    """``plan_step`` NEVER RAISES. A raising predicate is a refusal, by name.

    THIS FOUND A REAL DEFECT on ``polarizations``. Every arm's predicate was
    already inside the fail-closed try/except, but ``live_sub_steps`` — which
    ``plan_step`` calls AFTER the arm loop, to compute the residency verdict — read
    four attributes bare and sat outside it. A ``fields.polarizations`` that raised
    on access therefore escaped ``plan_step`` into a caller that would otherwise
    have stepped correctly on the array path. The fix returns ``None`` (the live
    set could not be determined), which makes the residency verdict refuse, because
    an unreadable list is not an empty one.
    """
    fields, pml = matrix.cart()
    plan = _plan(_RaisingProxy(fields, attribute), pml)
    assert isinstance(plan.reasons, dict)
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert slot in plan.reasons or slot in plan.plans
    if attribute == "polarizations":
        # The live set is unknown, so the residency verdict must REFUSE rather than
        # licence a mirror held across an update_P that may well run.
        assert not plan.residency.covered
        assert any("live sub-step set was not declared" in reason
                   for reason in plan.residency.reasons)


@pytest.mark.parametrize("label,build", [
    ("gridless", lambda: (object(), None)),
    ("fields_is_none", lambda: (None, None)),
    ("pml_is_a_string", lambda: (matrix.cart()[0], "not a pml")),
])
def test_plan_step_never_raises_on_a_degenerate_object(label, build):
    fields, pml = build()
    plan = _plan(fields, pml)
    assert plan.plans == {} or set(plan.plans) <= {"update_H", "update_E"}
    assert plan.reasons


def test_a_raising_builder_leaves_its_slot_unselected_with_a_named_reason(
        monkeypatch):
    """A predicate that admits and a builder that then explodes must not crash the
    composition: the slot falls to the array path with the exception named."""
    from meep_gpu.metal_kernels import launch as launch_module

    def explode(*args, **kwargs):
        raise RuntimeError("builder exploded")

    monkeypatch.setattr(launch_module, "plan_pml_curl", explode)
    fields, pml = matrix.cart()
    plan = _plan(fields, pml)
    assert "step_B" not in plan.plans
    assert any("builder exploded" in reason for reason in plan.reasons["step_B"])
    # The OTHER slots are unaffected: one family's builder failing must not take
    # the constitutive pair down with it.
    assert plan.selected.get("update_H") == "ordinary"


def test_a_builder_returning_none_leaves_its_slot_unselected(monkeypatch):
    from meep_gpu.metal_kernels import launch as launch_module

    monkeypatch.setattr(launch_module, "plan_pml_curl",
                        lambda *args, **kwargs: None)
    fields, pml = matrix.cart()
    plan = _plan(fields, pml)
    assert "step_B" not in plan.plans and "step_D" not in plan.plans
    assert plan.selected.get("update_E") == "ordinary"


def test_two_admitting_predicates_leave_the_slot_on_the_array_path(monkeypatch):
    """The contract, exercised by PLANTING a second admitter on a clean slot.

    The matrix's own overlap is on ``update_E``; this one is on a CURL slot, so the
    ambiguity message's per-slot wording ("two curl products…") is exercised too.
    """
    from meep_gpu.metal_kernels.coverage import Coverage

    spec = arms.register(
        family="_test_double", slot="step_B", label="planted",
        coverage=lambda ctx, slot: Coverage(True, ()),
        plan=lambda ctx, slot: object(),
        prefix="planted: ", noun="planted curl", wired=True)
    try:
        fields, pml = matrix.cart()
        plan = _plan(fields, pml)
        assert "step_B" not in plan.plans
        reason, = plan.reasons["step_B"]
        assert "two curl products" in reason
        assert "PML" in reason and "planted" in reason
        # step_D is untouched: the ambiguity is per slot, not per composition.
        assert plan.selected.get("step_D") == "PML"
    finally:
        arms._REGISTRY["step_B"].remove(spec)


# ---------------------------------------------------------------------------
# 4. The composition facts the matrix records, restated as claims
# ---------------------------------------------------------------------------

def test_a_bfast_run_composes_on_all_four_slots():
    """The constitutive companion is what makes this true, and it is new.

    Without it a BFAST run took its curls on the device and its constitutive pair
    on the array path — two syncs per step and a residency verdict that refuses.
    """
    fields, pml = matrix.cart(bfast_scaled_k=(0.2, 0.0, 0.0))
    plan = _plan(fields, pml)
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert set(plan.selected.values()) == {"BFAST"}


def test_a_complex_beta_run_composes_a_WHOLE_step_on_this_family():
    """The asymmetry is closed, as a claim rather than a comment.

    THIS TEST USED TO PIN THE GAP. Until 2026-08-19 the complex beta curls composed
    and both constitutive slots were refused BY NAME — the sentence "no
    complex-storage constitutive product admits an UNFOLDED beta run" was the thing
    asserted, so a reader could tell the gap from ``reasons`` without reading two
    module docstrings. ``special_kz.beta_run_complex_constitutive_coverage`` is a
    real restatement now, delegating to ``complex_fields``' certified body, and the
    pin follows: all four sub-steps compose, and every one of them on THIS family.

    THE FAMILY CHECK IS THE POINT, not the count. "Four slots filled" would also
    pass if another family had started admitting a beta run, which is precisely the
    over-covering dispatch every beta clause exists to prevent.
    """
    fields, pml = matrix.flat(beta=0.33, complex_storage=True)
    plan = _plan(fields, pml)
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert plan.selected[slot] == "special_kz complex beta", (
            slot, plan.selected, plan.reasons.get(slot))


def test_a_no_pml_run_composes_both_curls_and_the_null_pair():
    """The direct curls and genuinely empty constitutive pair fill the step."""
    fields, pml = matrix.cart(pml=0, storage=False, eps=False)
    plan = _plan(fields, pml)
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.selected == {
        "step_B": "no-PML curl", "update_H": "no-PML null",
        "step_D": "no-PML curl", "update_E": "no-PML null"}


def test_an_empty_update_p_is_refused_by_the_registered_ade_product():
    """A non-dispersive ``update_P`` is a named predicate refusal, not silence.

    A slot recorded with a reason reads differently from one recorded silently: the
    first says "no product carries this", the second would say "nobody asked".

    ``update_P`` left this table when the ADE family landed. ``fill_B``/``fill_D``
    left it when the folded family landed — they now
    carry an arm — and the same distinction is preserved there by a different
    mechanism: that arm is deliberately UNGATED, so an unfolded run gets the fill's
    own named refusal rather than the composer's "no consulted product admitted
    fill_B, and none gave a reason". Both halves are asserted, because dropping the
    ``NO_ARM_REASONS`` entry without checking the replacement is exactly how a slot
    goes silent.
    """
    fields, pml = matrix.cart()
    plan = _plan(fields, pml)
    assert launch.NO_ARM_REASONS == {}
    assert "update_P" not in plan.plans
    assert any("no driven component" in reason
               for reason in plan.reasons["update_P"])
    for slot in ("fill_B", "fill_D"):
        assert slot not in plan.plans
        assert any("no mirror plane is active" in reason
                   for reason in plan.reasons[slot]), plan.reasons[slot]


def test_the_conductivity_split_is_per_sub_step_and_not_per_composition():
    """An electric conductivity is read by the D curl alone (stepping.py:508).

    Measured rather than assumed — the composition matrix's first pin claimed both
    curls fell away and the sweep corrected it.
    """
    fields, pml = matrix.conductive(matrix.cart())
    electric = _plan(fields, pml)
    assert electric.selected.get("step_B") == "PML"
    assert electric.selected.get("step_D") == "conductive PML curl"

    fields, pml = matrix.conductive(matrix.cart(), magnetic=True)
    magnetic = _plan(fields, pml)
    assert magnetic.selected.get("step_D") == "PML"
    assert magnetic.selected.get("step_B") == "conductive PML curl"
    for plan in (electric, magnetic):
        assert plan.selected.get("update_H") == "ordinary"
        assert plan.selected.get("update_E") == "ordinary"
        assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
        assert plan.residency.covered, plan.residency.reasons


# ---------------------------------------------------------------------------
# 5. Residency: the verdict is a property of the COMPOSITION
# ---------------------------------------------------------------------------

def test_an_undeclared_source_list_refuses_the_residency_verdict():
    """``Fields`` does not hold the source list; inferring "no seam" from not
    knowing is exactly the over-covering refusal the clause exists to prevent."""
    fields, pml = matrix.cart()
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=None, **_probes())
    assert plan.residency is not None and not plan.residency.covered
    assert any("live sub-step set was not declared" in reason
               for reason in plan.residency.reasons)


def test_an_array_path_sub_step_must_be_declared_synced():
    """A mirror held across a host write is stale, and the verdict says so.

    THE SUBSTRATE MOVED ON 2026-08-19 AND THAT IS THE WHOLE NOTE. This used to be
    the complex beta row, which left BOTH constitutive slots on the array path while
    its curls stayed device-resident. That row is now fully resident — ``special_kz``
    gained its complex constitutive companion — so it can no longer measure the
    mirror rule at all, and a test kept pointed at it would have gone on asserting a
    gap that had closed. The real beta DISPERSIVE row is the replacement: three
    sub-steps and ``update_P`` compose, ``update_E`` does not (complex-storage-free
    ADE E is another family's), so the mixed shape this test needs still exists.
    Undeclared, the verdict refuses and names the volumes; declared, it holds.
    """
    fields, pml = matrix.dispersive(matrix.flat(beta=0.33))
    undeclared = launch.plan_step(fields, pml, residency=device.Residency(),
                                  sources=(), **_probes())
    assert not undeclared.residency.covered
    assert any("update_E runs on the array path" in reason
               for reason in undeclared.residency.reasons), (
        undeclared.residency.reasons)

    declared = launch.plan_step(
        fields, pml, residency=device.Residency(), sources=(),
        synced=("update_E",), **_probes())
    assert declared.residency.covered, declared.residency.reasons


def test_a_null_slot_neither_requires_a_mirror_nor_stales_one():
    """The third residency case, and the one that was measurably missing.

    Declared ``planned``, a null plan made the verdict demand mirrors of arrays a
    no-PML run does not allocate at all.
    """
    fields, pml = matrix.cart(pml=0, storage=False, eps=False)
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=(), synced=("step_B", "step_D"), **_probes())
    assert plan.residency.covered, plan.residency.reasons
    for slot in ("update_H", "update_E"):
        assert plan.plans[slot].performs_device_work is False
