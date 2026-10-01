"""THE HAND-CUDA TABLE AS THE SEAM'S SECOND NVIDIA COMPOSER, and what holds it there.

The twin of ``test_triton_certified_fused_products`` for the table
``fastpath._decide`` composes after Triton. Five things are pinned here, and the
order is the order a reader should want them in:

1. THE TYPED TABLES ARE MEASURED AGAINST THE COMPOSER, never transcribed.
   ``fastpath_cuda.CUDA_FUSED_LABELS`` and ``CUDA_FUSED_ARM_CONSTITUENTS`` exist so
   a NumPy step can answer "is that a fused label" without importing
   ``cuda_kernels`` — and a typed copy of somebody else's data is a copy that
   drifts, so both are compared against ``fused_pairs.py``'s own AST and its absorb
   declarations here.
2. THE RELEASE PARTITIONS THE LABEL SPACE. Every label the composer can write is
   either RELEASED with its own cases and per-arm axes, or PENDING with a reason
   naming what is missing. No label is both, and none is neither.
3. EVERY CERTIFICATION ROW RESOLVES. A released arm whose artifact names no gate is
   the over-claim this whole layer exists to prevent, and
   :func:`test_every_cuda_certification_points_at_a_readable_record` is RED by
   design on the five rows the campaign has not seeded yet — see
   ``fastpath_cuda.CUDA_ARM_CERTIFICATION`` for which writer cuts each.
4. THE OFFER IS REAL, IN BOTH DIRECTIONS AND AT BOTH OF ITS TWO POINTS. An
   un-offered product is skipped before its predicate, does not supersede a shorter
   offered span, and does not withhold a slot as a neighbouring-seam claimant; and
   ``fuse_labels=None`` composes byte-for-byte what this composer composed before
   the offer existed.
5. THE SEAM'S OWN READS ANSWER ON THESE PLANS: the measured launch grid, the
   plan-time warm, and the transitive ``absorbed_by`` identity that makes a
   three-slot weld ONE unit rather than two.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest

from . import fastpath, fastpath_cuda
from .cuda_kernels import arms, fused_pairs
from .cuda_kernels.test_fused_pairs import build, magnetic_source, xp  # noqa: F401
from .triton_kernels.launch import NoopPlan

#: The architecture the shipped CUDA ledger has live run records for. Each weld keeps
#: one record per compute capability, so a lookup that names none quotes no host, no
#: ``recorded_utc`` and no policy — the point being that one card's run is not
#: evidence about another. Every run fact below is therefore asked on this one.
CERTIFIED_CAPABILITY = "8.6"


# ---------------------------------------------------------------------------
# 1. The typed tables against the composer
# ---------------------------------------------------------------------------

def _builder_labels() -> dict:
    """``FUSED_PRODUCTS`` family -> the ``label=`` literal ITS builder passes.

    Read off the AST rather than by calling the builders: a builder needs a frozen
    configuration to run and most of them need a device, and what is being pinned
    is the SOURCE agreeing with the table, which is a question about the file.
    """
    source = pathlib.Path(fused_pairs.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    by_function: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id in ("CudaFusedPairPlan", "CudaFusedTriplePlan")):
                for keyword in sub.keywords:
                    if keyword.arg == "label" and isinstance(keyword.value,
                                                             ast.Constant):
                        by_function.setdefault(node.name, []).append(
                            keyword.value.value)
    found: dict = {}
    for family, product in fused_pairs.FUSED_PRODUCTS.items():
        labels = by_function.get(product["plan"].__name__, [])
        assert len(labels) == 1, (family, labels)
        found[family] = labels[0]
    return found


def test_every_table_row_carries_the_label_its_builder_passes():
    """The row and the builder are one fact, and a rename must move both.

    The row is what the OFFER is asked about — it is read before the builder runs,
    which is the whole point of asking the offer before the predicate — so a row
    whose label had drifted from its builder's would offer one product and install
    another under a name nobody declared.
    """
    for family, label in _builder_labels().items():
        assert fused_pairs.FUSED_PRODUCTS[family]["label"] == label, family


def test_the_typed_label_set_is_exactly_what_the_composer_writes():
    """``CUDA_FUSED_LABELS`` is typed so a NumPy step need not import the composer."""
    assert set(fastpath_cuda.CUDA_FUSED_LABELS) == set(_builder_labels().values())
    assert len(fastpath_cuda.CUDA_FUSED_LABELS) == len(
        set(fastpath_cuda.CUDA_FUSED_LABELS)), "a duplicate label is a collision"


def test_the_constituent_map_is_the_absorb_declaration_namespaced():
    """What each product substitutes, from the composer's own two absorb tables.

    ``FUSED_PAIR_EXTRA_ARMS`` is UNIONED in rather than dropped: an extra row is an
    alternate arm pair the product may absorb on some admitted configuration, and
    the pending rung asks whether ANY of them is pending.
    """
    labels = _builder_labels()
    expected = {}
    for family, label in labels.items():
        absorbs = set(fused_pairs.FUSED_PAIR_ARMS.get(family, ()))
        for extra in fused_pairs.FUSED_PAIR_EXTRA_ARMS.get(family, ()):
            absorbs |= set(extra)
        expected[fastpath_cuda.namespaced(label)] = {
            fastpath_cuda.namespaced(arm) for arm in absorbs}
    got = {label: set(arms_) for label, arms_
           in fastpath_cuda.CUDA_FUSED_ARM_CONSTITUENTS.items()}
    assert got == expected


def test_the_constituent_map_is_total_over_the_labels_the_composer_can_write():
    """A fused label missing here is refused by NAME at rung 7a, never admitted."""
    assert set(fastpath_cuda.CUDA_FUSED_ARM_CONSTITUENTS) == {
        fastpath_cuda.namespaced(label)
        for label in fastpath_cuda.CUDA_FUSED_LABELS}


# ---------------------------------------------------------------------------
# 2. The release partitions the label space
# ---------------------------------------------------------------------------

def _installable_labels() -> set:
    return {fastpath_cuda.namespaced(product["label"])
            for family, product in fused_pairs.FUSED_PRODUCTS.items()
            if fused_pairs._declared_uninstallable(family, product) is None}


def _uninstallable_labels() -> set:
    return {fastpath_cuda.namespaced(product["label"])
            for family, product in fused_pairs.FUSED_PRODUCTS.items()
            if fused_pairs._declared_uninstallable(family, product) is not None}


def test_released_and_pending_partition_every_label_the_composer_writes():
    """Every label is accounted for exactly once, and the counts are pinned.

    THE COUNTS ARE THE DENOMINATOR of every dispatch number this table publishes:
    38 products, 31 of them installable, 12 released in Tier 1 (2026-09-11) plus 8
    in Tier 2 (2026-09-13: the cylindrical complex pairs, the two off-diagonal
    stencil welds, the complex beta pairs and the BFAST pairs) and 18 pending. A
    label that fell out of both maps would simply stop being judged, which is the
    one way this file can be green about a product nobody decided on.

    22 / 16 -> 24 / 14 ON 2026-09-17: the two COMPLEX off-diagonal stencil welds
    (`cuda:folded complex off-diagonal stencil weld` and `cuda:complex no-absorber
    off-diagonal stencil weld`). Neither was waiting on a measurement -- their gate
    block had stood released in certification.json since 2026-09-03 with 132 of 132
    S1 cases bit-identical under both canonical policies -- and both were waiting on
    a LEDGER: nothing in cuda_kernels/fingerprints.json bound that block, and six of
    its 53 imported pins had drifted, so it could only have been seeded DRIFTED. The
    gate was re-run on the live tree (2026-09-17, both policies released, the same
    132/66/11 counts) and the entry seeded at zero source drift.
    """
    every = {fastpath_cuda.namespaced(label)
             for label in fastpath_cuda.CUDA_FUSED_LABELS}
    released = set(fastpath_cuda.CUDA_RELEASED_FUSED_ARMS)
    pending = set(fastpath_cuda.CUDA_PENDING_DEVICE_GATE_ARMS)
    assert released | pending == every
    assert not released & pending
    assert len(every) == 38
    assert len(_installable_labels()) == 31
    assert len(_uninstallable_labels()) == 7
    # 24/14 until 2026-09-17: ``cuda:complex no-absorber off-diagonal stencil weld``
    # came out of the release the day it was driven -- it dispatches by preference,
    # and its arbitration-default-precedence control reads REFUSED because the
    # Triton table's PENDING ``complex no-PML off-diagonal`` sends the whole step to
    # the array path under the shipped order (CUDA_PENDING_DEVICE_GATE_ARMS names it).
    assert len(released) == 23
    assert len(pending) == 15


def test_the_magnetic_pair_off_diagonal_admission_is_a_corner_not_a_row():
    """The Tier 2 widening, both directions, and the rule that keeps it honest.

    ``cuda:fused magnetic pair`` carries no ``off_diagonal_epsilon`` row since
    2026-09-13 -- it drives both values -- and the True side is bounded by
    ``CUDA_OFF_DIAGONAL_CORNERS`` to the corner its off-diagonal cases sat at (2-D and
    3-D, lossless, no poles, no Bloch phase). A plain row could only refuse every
    off-diagonal case driven or admit every one that is not; the corner is what
    refuses an off-diagonal grid over a susceptibility or a conductivity BY THE AXIS.
    """
    arm = "cuda:fused magnetic pair"
    shape = {"pml_active": True, "dimensions": 2, "complex_storage": False,
             "cylindrical": False, "bfast": False, "beta": 0, "bloch": False,
             "nonlinearity": False, "off_diagonal_epsilon": True,
             "conductivity": False, "susceptibilities": 0}
    assert arm in fastpath_cuda.released_fused_arms(shape)
    assert arm in fastpath_cuda.released_fused_arms(dict(shape, dimensions=3))
    for axis, value in (("dimensions", 1), ("conductivity", True),
                        ("susceptibilities", 1), ("bloch", True)):
        why = fastpath_cuda.fused_release_arm_reasons(dict(shape, **{axis: value}), arm)
        assert why and axis in why[0] and "driven off-diagonal only" in why[0], (axis, why)
    # An off-diagonal grid whose corner axis did not read is refused, not admitted.
    unread = dict(shape); del unread["susceptibilities"]
    why = fastpath_cuda.fused_release_arm_reasons(unread, arm)
    assert why and "did not read" in why[0]
    # THE RULE: a corner lists exactly the axes the arm's row leaves open, minus the
    # presence axis ``folded`` (both sides driven; free text) and the axis itself.
    # THE CORNER IS A LITERAL, checked as one (Metal's METAL_OFF_DIAGONAL_CORNERS test
    # does the same): it lists, per axis, the values this arm's OFF-DIAGONAL cases
    # drove, for every axis its row leaves open OR pins LESS strictly than the
    # off-diagonal side. ``susceptibilities`` is the latter since 2026-09-13 -- the
    # row admits {0, 1} (the diagonal cases) and the corner narrows the off-diagonal
    # side to {0} (its off-diagonal cases drove 0 poles only), so the axis is in BOTH
    # and a computed open-axes equality would wrongly reject it.
    assert fastpath_cuda.CUDA_OFF_DIAGONAL_CORNERS[arm] == {
        "dimensions": frozenset({2, 3}),
        "conductivity": frozenset({False}),
        "susceptibilities": frozenset({0}),
        "bloch": frozenset({False}),
    }
    # Every other arm either pins the axis in its row or is refused off-diagonal.
    for other in fastpath_cuda.CUDA_RELEASED_FUSED_ARMS:
        if other == arm:
            continue
        row = dict((a, v) for a, v, _w in fastpath_cuda.CUDA_FUSED_RELEASE_ARM_AXES[other])
        if "off_diagonal_epsilon" in row:
            continue
        assert fastpath_cuda.fused_release_arm_reasons(shape, other), other


def test_no_released_label_is_one_the_composer_declares_uninstallable():
    """Releasing a product the composer refuses on every configuration would
    dispatch nothing and credit a gate for it."""
    assert not set(fastpath_cuda.CUDA_RELEASED_FUSED_ARMS) & _uninstallable_labels()


def test_every_uninstallable_product_is_pending_with_its_own_reason():
    for label in _uninstallable_labels():
        reason = fastpath_cuda.CUDA_PENDING_DEVICE_GATE_ARMS[label]
        assert "INSTALLABLE = False" in reason, label


def test_every_released_arm_has_cases_a_leg_and_its_own_axes():
    """Three tables, one release, and none of them substitutes for the others.

    The cases say WHICH driver-route case drove the arm; the leg says on which of
    the campaign's five legs it dispatched (a complex arm dispatches only under the
    expansion probe, and a complex arm appearing in the shipped leg's
    ``arms_driven`` would mean the licence had been consumed without it); the axes
    say on WHAT CONFIGURATION, which is what ``released_fused_arms`` decides with.
    """
    for arm, cases in fastpath_cuda.CUDA_RELEASED_FUSED_ARMS.items():
        assert cases, arm
        assert arm in fastpath_cuda.CUDA_RELEASED_ARM_LEGS, arm
        assert fastpath_cuda.CUDA_RELEASED_ARM_LEGS[arm], arm
        assert arm in fastpath_cuda.CUDA_FUSED_RELEASE_ARM_AXES, arm
        assert fastpath_cuda.CUDA_FUSED_RELEASE_ARM_AXES[arm], arm
        assert arm in fastpath_cuda.CUDA_ARM_CERTIFICATION, arm


def test_a_complex_storage_arm_rides_the_probe_leg_and_only_that_one():
    """The one leg split the recut walks ``(arm, case, leg)`` rather than
    ``(arm, case)`` for."""
    for arm, axes in fastpath_cuda.CUDA_FUSED_RELEASE_ARM_AXES.items():
        if arm not in fastpath_cuda.CUDA_RELEASED_FUSED_ARMS:
            continue
        complex_storage = dict((axis, value) for axis, value, _why in axes).get(
            "complex_storage")
        legs = fastpath_cuda.CUDA_RELEASED_ARM_LEGS[arm]
        if complex_storage is True:
            assert legs == ("cuda_shipped_expansion_probe",), arm
        else:
            assert "cuda_shipped_expansion_probe" not in legs, arm


def test_an_arm_with_no_per_arm_row_fails_closed():
    """Admitting on silence is the shape of every over-claim this layer prevents."""
    why = fastpath_cuda.fused_release_arm_reasons({"pml_active": True},
                                                  "cuda:invented product")
    assert why and "no per-arm axis row" in why[0]


def test_the_shared_envelope_refuses_a_shape_it_cannot_read():
    why = fastpath_cuda.fused_release_reasons({"unreadable": "RuntimeError()"})
    assert why and "the run shape did not read" in why[0]


def test_a_released_arm_is_admitted_on_its_own_case_shape_and_refused_off_it():
    """The release, both directions, on the two arms whose shapes this file can build."""
    shape = {"pml_active": True, "dimensions": 2, "complex_storage": False,
             "cylindrical": False, "bfast": False, "beta": 0,
             "nonlinearity": False, "off_diagonal_epsilon": False,
             "conductivity": False, "susceptibilities": 0}
    admitted = fastpath_cuda.released_fused_arms(shape)
    assert "cuda:fused magnetic pair" in admitted
    assert "cuda:fused electric pair" in admitted
    # The conductive product is NOT admitted on a lossless grid: it IS the
    # conductive one, and the ordinary electric pair is what the composer selects.
    assert "cuda:conductive fused electric pair" not in admitted
    # And no absorber takes every released arm out through the SHARED row.
    assert fastpath_cuda.released_fused_arms(dict(shape, pml_active=False)) == ()


# ---------------------------------------------------------------------------
# 3. Every certification row resolves
# ---------------------------------------------------------------------------

def test_every_cuda_certification_points_at_a_readable_record():
    """A released arm whose artifact names no gate is the over-claim to prevent.

    RED BY DESIGN until the campaign seeds the five rows
    ``fastpath_cuda.CUDA_ARM_CERTIFICATION`` names as promises: the gate is named in
    the shipping tree BEFORE the run that cuts it, so the run the tree asked for is
    the run that is measured, and this test is what keeps the promise from being
    mistaken for evidence in the meantime.

    RESOLVING IS NOW PER ARCHITECTURE: the run facts live in ``runs[<cc>]``, so a row
    "resolves" only if the gate it names has a live record for the card the arm would
    dispatch on. A weld carrying a run for some OTHER architecture is counted here as
    not resolving, which is the honest reading — it is evidence about a different
    device.
    """
    missing = []
    for arm in sorted(fastpath_cuda.CUDA_ARM_CERTIFICATION):
        entry = fastpath_cuda.certification_for(
            arm, capability=CERTIFIED_CAPABILITY)
        assert entry["family"] != "unmapped", arm
        if "recorded_utc" not in entry:
            missing.append((arm, entry["gate"]))
    assert not missing, (
        "CUDA_ARM_CERTIFICATION rows whose ledger key does not resolve on compute "
        f"capability {CERTIFIED_CAPABILITY}: "
        + "; ".join(f"{arm} -> {gate}" for arm, gate in missing))


def test_a_certification_miss_names_itself_rather_than_reading_as_a_gate():
    entry = fastpath_cuda.certification_for("cuda:invented product")
    assert entry["family"] == "unmapped"
    assert "cuda_kernels/fingerprints.json" in entry["gate_record"]


def test_the_certification_policy_is_the_tables_own_and_the_two_maps_agree():
    """One value, two files, pinned equal rather than agreeing by coincidence."""
    assert fastpath.TABLE_SUBNORMAL_POLICY["cuda"] == fastpath_cuda.SUBNORMAL_POLICY
    assert (tuple(fastpath.TABLE_GOVERNED_EXECUTORS["cuda"])
            == tuple(fastpath_cuda.GOVERNED_EXECUTORS))
    entry = fastpath_cuda.certification_for("cuda:fused magnetic pair",
                                            capability=CERTIFIED_CAPABILITY)
    assert entry["certification_policy"] == "keep"
    # And the weld it cites records BOTH canonical legs, the keep one being the
    # dispatching leg. THE POLICY IS THE RUN'S, so it is asked on the architecture
    # the run was cut on: two cards certified from one tree can be driven under
    # different policies, and a table-wide answer could not say which.
    assert "ieee_keep_ftz_stripped" in entry["subnormal_policy"]


def test_the_validated_capabilities_are_derived_from_the_cited_blocks():
    """DERIVED, never typed: the list is what the welds' own campaigns recorded.

    Re-derived here rather than compared against a literal, so a round that certifies
    a second architecture widens both the function and this expectation by writing
    records. The intersection is taken over the gates the arms cite, off the ledger
    on disk, because a capability one cited family never ran on is one the table
    cannot claim.
    """
    import json  # noqa: PLC0415

    ledger = json.loads(
        (pathlib.Path(fastpath_cuda.__file__).parent / "cuda_kernels"
         / "fingerprints.json").read_text(encoding="utf-8"))
    derived = None
    for _family, gate in fastpath_cuda.CUDA_ARM_CERTIFICATION.values():
        live = set(fastpath.live_capabilities(ledger.get(gate)))
        derived = live if derived is None else (derived & live)
    assert derived, (
        "every cited CUDA weld must carry a run that still binds the shipped bytes")
    assert fastpath_cuda.validated_compute_capabilities() == tuple(sorted(derived))
    assert CERTIFIED_CAPABILITY in derived, (
        f"this file reads run facts on {CERTIFIED_CAPABILITY} and the ledger now "
        f"derives {sorted(derived)}")


# ---------------------------------------------------------------------------
# 4. The namespace, and what the shared seam makes of it
# ---------------------------------------------------------------------------

def test_every_namespaced_label_is_recognised_as_fused_and_no_bare_one_is():
    """The namespace IS the answer, which is why it is a prefix on the literal.

    Not one of the 38 CUDA literals matches the Triton prefix or whole labels, so a
    bare CUDA label reaching ``arm_is_fused`` would answer False and a fused product
    would pass clause 8 unjudged. The prefix is what makes the question answerable
    at all, and both directions are asserted so a future rename cannot make one of
    them vacuous.
    """
    for label in fastpath_cuda.CUDA_FUSED_LABELS:
        assert fastpath.arm_is_fused(fastpath_cuda.namespaced(label)), label
        assert not fastpath.arm_is_fused(label), (
            f"{label!r} is recognised BARE, so the namespace is not what is "
            "deciding and a Triton label could collide with it")
    for label in ("fused pair B", "fused pair D", "dispersive fused pair",
                  "fused ADE state"):
        assert fastpath.arm_is_fused(label)
        assert not fastpath.arm_is_fused(fastpath_cuda.namespaced(label)), label


def test_the_two_composers_vocabularies_are_disjoint():
    triton_like = [label for label in fastpath_cuda.CUDA_FUSED_LABELS
                   if label in fastpath.FUSED_ARM_LABELS
                   or any(label.startswith(prefix)
                          for prefix in fastpath.FUSED_ARM_PREFIXES)]
    assert not triton_like, triton_like


def test_the_null_arm_label_is_shared_and_the_drop_reads_it_bare():
    """Both NVIDIA composers spell the null constitutive arm the same way."""
    assert fastpath_cuda.NULL_ARM_LABEL == fastpath.NULL_ARM_LABEL
    assert fastpath._bare("cuda:no-PML null") == fastpath.NULL_ARM_LABEL


# ---------------------------------------------------------------------------
# 5. The offer, at both of its points
# ---------------------------------------------------------------------------

def test_fuse_labels_none_composes_exactly_what_it_composed_before_the_offer(xp):
    """``None`` means no restriction, which is what the coverage census asks for."""
    fields, layer, grid = build(xp)
    without = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    fields, layer, grid = build(xp)
    with_none = arms.plan_step(fields, layer, grid, sources=(), fuse=True,
                               fuse_labels=None)
    assert with_none.selected == without.selected
    assert with_none.reasons == without.reasons
    assert sorted(with_none.plans) == sorted(without.plans)


def test_an_un_offered_product_is_skipped_before_its_predicate_and_says_so(xp):
    """The wording carries the gate's needle: "offered to run"."""
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True,
                          fuse_labels=["some other product"])
    assert plan.selected["step_B"] == "PML"
    assert plan.selected["update_H"] == "ordinary"
    reason = plan.reasons["fused_pair_cuda_fused_magnetic_pair"][0]
    assert "offered to run" in reason
    assert "fused magnetic pair" in reason


def test_the_offered_product_still_installs_beside_an_un_offered_sibling(xp):
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True,
                          fuse_labels=["fused magnetic pair"])
    assert plan.selected["step_B"] == "fused magnetic pair"
    assert isinstance(plan.plans["update_H"], NoopPlan)


def test_an_empty_offer_is_not_read_off_a_typo_as_no_offer_at_all():
    """A caller that passes something unreadable made NO offer; an EMPTY sequence
    made one, and it disables every product."""
    assert fused_pairs.offered_labels(None) is None
    assert fused_pairs.offered_labels(object()) is None
    assert fused_pairs.offered_labels([]) == frozenset()
    assert fused_pairs.offered_labels("one") == frozenset({"one"})


def test_an_un_offered_longer_span_does_not_supersede_an_offered_shorter_one():
    """The offer's FIRST point, and what its POSITION in the loop buys.

    ``_superseded_by_a_longer_span`` hands a seam to the longest strictly-containing
    candidate, which is right when the caller can run it: where the triple admits,
    the D->E pair does too, so the longer span costs no row. It is exactly WRONG when
    the caller cannot run the triple — the seam would then go to a product that never
    installs, and the D->E pair that the caller CAN run would have been refused for
    it. The same is true of the E->P claimant on the slot next door.

    Asking the offer BEFORE the predicate is what forecloses both at once: an
    un-offered product is not a candidate, so it is in neither clause's input. This
    is the 2026-09-02 conductive lesson (a released seam lost entirely to a sibling
    product with no ledger entry) in the two shapes this composer has and the Triton
    one does not.
    """
    from .cuda_kernels.test_fused_pairs import build_dispersive  # noqa: PLC0415

    fields, layer, grid = build_dispersive(poles=2, pml_on=True)
    # As shipped, with everything offered, the triple takes all three slots.
    shipped = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    assert shipped.selected["step_D"] == "three-slot dispersive weld"

    pair = "dispersive fused electric pair"
    offered = arms.plan_step(fields, layer, grid, sources=(), fuse=True,
                             fuse_labels=[pair])
    assert offered.selected["step_D"] == offered.selected["update_E"] == pair, \
        offered.selected
    # ...and BOTH of the clauses that would otherwise have taken the seam say why
    # they did not, in the wording the route gate's envelope leg greps for.
    for family in ("cuda_three_slot_dispersive_weld", "cuda_fused_polarization_pair"):
        assert "offered to run" in offered.reasons[f"fused_pair_{family}"][0]


def test_an_un_offered_claimant_does_not_withhold_a_slot_from_an_offered_product():
    """The offer's SECOND point, and it is not a duplicate of the first.

    ``_later_seam_claimant`` withholds ``update_E`` from a D->E product when an E->P
    product admits the same run — a trade measured net ZERO. A claimant this caller
    cannot run makes the trade net -1: the seam it would serve is one the caller
    cannot have. So an un-offered claimant is not a claimant.
    """
    dispersive = "dispersive fused electric pair"
    polarization = "fused polarization pair"
    called = []

    class _Context:
        fields = pml = grid = sources = None

    def admitting(context):
        called.append(True)
        return types.SimpleNamespace(covered=True, reasons=())

    original = {}
    for family, product in fused_pairs.FUSED_PRODUCTS.items():
        if product["label"] in (dispersive, polarization):
            original[family] = product["coverage"]
    assert original, "the two exemplar products must exist"
    try:
        for family in original:
            fused_pairs.FUSED_PRODUCTS[family]["coverage"] = admitting
        # With the E->P product OFFERED it is a claimant...
        claimant = fused_pairs._later_seam_claimant(
            {}, {}, {}, _Context(), "update_E", (),
            fused_pairs.offered_labels([dispersive, polarization]))
        assert claimant is not None
        # ...and with it withheld from the offer it is not.
        claimant = fused_pairs._later_seam_claimant(
            {}, {}, {}, _Context(), "update_E", (),
            fused_pairs.offered_labels([dispersive]))
        assert claimant is None
    finally:
        for family, coverage in original.items():
            fused_pairs.FUSED_PRODUCTS[family]["coverage"] = coverage


# ---------------------------------------------------------------------------
# 6. What the shared seam reads off these plans
# ---------------------------------------------------------------------------

def test_the_launch_grid_is_unknown_before_the_first_launch_and_measured_after():
    """``programs_per_dispatch`` tells "a kernel launched" from "a kernel computed
    over data", and a PREDICTED grid cannot draw that distinction."""
    launched = []

    def launcher(fields, arguments):
        launched.append(arguments)
        return {"launched": True, "blocks": 384, "threads": 256}

    plan = fused_pairs.CudaFusedPairPlan(
        family="cuda_fused_magnetic_pair", label="fused magnetic pair",
        kernel_label="k", replaces=("step_B",), slots=("step_B", "update_H"),
        context=types.SimpleNamespace(fields=object()),
        resolve_launch_args=lambda context: {"n": 1}, launch=launcher)
    assert plan.launch_grid is None
    assert fastpath._launch_grid_of(plan) is None
    plan.run()
    assert plan.launch_grid == (384,)
    assert fastpath._launch_grid_of(plan) == (384,)
    assert fastpath._plan_launches_of(plan) == 1


def test_a_component_major_group_spells_the_grid_its_leading_slot_is_read_for():
    """A MULTI-LAUNCH GROUP ON A LEADING SLOT STILL OWES A GRID, and until
    2026-09-14 one did not pay it.

    The route gate probes ``programs_per_dispatch`` on ``step_B``/``step_D`` only,
    so the real weld's component-major aggregate
    (``three_slot_dispersive_weld.run_three_slot_polarization``) could omit
    ``blocks`` for a year without consequence: it sits on ``update_P``, which is
    never probed. ``complex_no_pml_three_slot_dispersive_weld`` is the first product
    to put such a group on a LEADING slot, and its aggregate omitted ``blocks`` the
    same way. MEASURED on the GPU host 2026-09-14: ``step_D dispatches 600``,
    ``programs_per_dispatch null``, and the case refused as VACUOUS-PASS -- a
    correct dispatch failing a clause whose whole job is to tell "a kernel launched"
    from "a kernel computed over data".
    """
    def leading(fields, arguments):
        # The shipped shape: one launch per driven component, and a TOTAL, because
        # the counter is programs per DISPATCH and one dispatch launches all three.
        return {"launched": True, "launches": 3, "blocks": 3 * 128,
                "components": [{"blocks": 128}, {"blocks": 128}, {"blocks": 128}]}

    def trailing(fields, arguments):
        # THE CONSULT IS ANSWERED, NOT LAUNCHED: update_P was advanced inside the
        # leading group, so this group reaches no device at all.
        return {"launched": False, "launches": 0, "advanced_in": "leading"}

    triple = fused_pairs.CudaFusedTriplePlan(
        family="cuda_complex_no_pml_three_slot_dispersive_weld",
        label="complex no-absorber three-slot weld", kernel_label="k",
        replaces=("step_D", "update_E", "update_P"),
        slots=("step_D", "update_E", "update_P"),
        context=types.SimpleNamespace(fields=object()),
        resolve_launch_args=lambda context: {"n": 1},
        launch_leading=leading, launch_trailing=trailing)

    assert fastpath._launch_grid_of(triple.leading) is None, (
        "a grid that is known before the first launch is a PREDICTION, and the "
        "counter exists to refuse predictions")
    triple.leading.run()
    assert triple.leading.launch_grid == (384,)
    assert fastpath._launch_grid_of(triple.leading) == (384,)


def test_the_trailing_half_of_an_absorbed_tail_is_unknown_not_witnessed():
    """WHY ``plan_launches`` MAY NOT BE READ AS "the tail launched nothing".

    ``_TripleHalfPlan.run`` increments ``launches`` whether or not its launcher
    reached the device, so the component-major weld's trailing half books a launch
    per step while launching nothing: MEASURED on the GPU host 2026-09-14,
    ``update_P plan_launches 600`` on the very leg whose trailing group returns
    ``{"launched": False, "launches": 0}``. A gate clause that read that counter as
    a witness would confirm a launch that never happened, which is why
    ``expected_substitution`` takes ``tail_in_leading`` as a DECLARATION and leans
    on the measured drop to falsify a wrong one.
    """
    def trailing(fields, arguments):
        return {"launched": False, "launches": 0, "advanced_in": "leading"}

    triple = fused_pairs.CudaFusedTriplePlan(
        family="cuda_complex_no_pml_three_slot_dispersive_weld",
        label="complex no-absorber three-slot weld", kernel_label="k",
        replaces=("step_D", "update_E", "update_P"),
        slots=("step_D", "update_E", "update_P"),
        context=types.SimpleNamespace(fields=object()),
        resolve_launch_args=lambda context: {"n": 1},
        launch_leading=lambda fields, arguments: {"launched": True, "blocks": 1},
        launch_trailing=trailing)

    triple.trailing.run()
    assert fastpath._launch_grid_of(triple.trailing) is None, (
        "a group that launched nothing spells no grid, and None here means "
        "unknown rather than zero")
    assert fastpath._plan_launches_of(triple.trailing) == 1, (
        "this is the HAZARD being pinned, not a desirable property: the counter "
        "reads a launch the device never saw")


def test_warm_compiles_once_and_launches_nothing():
    compiles = []

    class _Kernel:
        def compile(self):
            compiles.append(True)

    module = types.ModuleType("stand_in_family")
    module._get_kernel = lambda: _Kernel()
    launched = []
    plan = fused_pairs.CudaFusedPairPlan(
        family="f", label="l", kernel_label="k", replaces=(),
        slots=("step_B", "update_H"), context=types.SimpleNamespace(fields=None),
        resolve_launch_args=lambda context: {},
        launch=lambda fields, arguments: launched.append(True))
    plan.warm_module = "stand_in_family"
    import sys
    sys.modules["meep_gpu.cuda_kernels.stand_in_family"] = module
    try:
        assert fastpath.warm_plan(plan) is None
    finally:
        del sys.modules["meep_gpu.cuda_kernels.stand_in_family"]
    assert compiles == [True]
    assert launched == [], "the warm pass must not launch"
    assert plan.launches == 0


def test_a_family_with_no_compile_entry_point_is_recorded_unwarmed_not_unfilled():
    """A raise here would take a working product to the array path over the absence
    of a warm mechanism."""
    module = types.ModuleType("bare_family")
    plan = fused_pairs.CudaFusedPairPlan(
        family="f", label="l", kernel_label="k", replaces=(),
        slots=("step_B", "update_H"), context=None,
        resolve_launch_args=lambda context: {}, launch=lambda f, a: None)
    plan.warm_module = "bare_family"
    import sys
    sys.modules["meep_gpu.cuda_kernels.bare_family"] = module
    try:
        reason = fastpath.warm_plan(plan)
    finally:
        del sys.modules["meep_gpu.cuda_kernels.bare_family"]
    assert reason is not None and "_get_kernel" in reason


def test_a_three_slot_weld_is_ONE_unit_under_the_transitive_identity():
    """A single ``absorbed_by`` hop puts a triple's three slots in TWO identities.

    ``_split_pairs``, ``_pair_of`` and the mid-step commitment clause would then
    treat ``update_P`` as unpaired — the same defect the deposit-repair shape had,
    one level deeper. The walk to the terminal object collapses all three.
    """
    triple = fused_pairs.CudaFusedTriplePlan(
        family="cuda_three_slot_dispersive_weld", label="three-slot dispersive weld",
        kernel_label="k", replaces=("step_D", "update_E", "update_P"),
        slots=("step_D", "update_E", "update_P"), context=None,
        resolve_launch_args=lambda context: {},
        launch_leading=lambda f, a: {"launched": True, "blocks": 8},
        launch_trailing=lambda f, a: {"launched": False, "launches": 0})
    plans = {"step_D": triple.leading,
             "update_E": NoopPlan("update_E", triple.leading),
             "update_P": triple.trailing}
    identities = {fastpath._pair_identity(plan) for plan in plans.values()}
    assert len(identities) == 1, "the three slots of one weld must be ONE unit"


def test_the_merge_adopts_a_whole_cuda_unit_and_never_half_of_one():
    """The merge's own contract, driven directly on two composer-shaped stubs."""

    class _Pair:
        def __init__(self):
            self.launchable = True
            self.launches = 0

        def run(self, *args, **kwargs):
            self.launches += 1

    pair = _Pair()
    cuda_plan = types.SimpleNamespace(
        plans={"step_D": pair, "update_E": NoopPlan("update_E", pair)},
        selected={"step_D": "fused electric pair",
                  "update_E": "fused electric pair"},
        reasons={})
    empty = types.SimpleNamespace(plans={}, selected={}, reasons={})
    record: dict = {}
    merged = fastpath_cuda.merge_tables(
        [("triton", empty), ("cuda", cuda_plan)],
        pending_of={"triton": {}, "cuda": {}}, record=record)
    assert sorted(merged.plans) == ["step_D", "update_E"]
    assert merged.selected["step_D"] == "cuda:fused electric pair"
    assert merged.backends == {"step_D": "cuda", "update_E": "cuda"}
    assert record["arbitration"]["adopted"] == {"step_D": "cuda",
                                                "update_E": "cuda"}


def test_a_cuda_unit_whose_slot_the_primary_holds_is_refused_WHOLE_and_by_name():
    class _Pair:
        launchable = True

        def run(self, *args, **kwargs):
            return None

    pair = _Pair()
    cuda_plan = types.SimpleNamespace(
        plans={"step_D": pair, "update_E": NoopPlan("update_E", pair)},
        selected={"step_D": "fused electric pair",
                  "update_E": "fused electric pair"},
        reasons={})
    # THE INCUMBENT'S PLAN HAS A ``run``. A bare object would be dropped by the
    # launchability filter every table now goes through, and the test would then be
    # measuring an empty primary rather than a contested slot.
    incumbent = types.SimpleNamespace(
        plans={"step_D": types.SimpleNamespace(run=lambda *a, **k: None)},
        selected={"step_D": "fused pair D"}, reasons={})
    record: dict = {}
    merged = fastpath_cuda.merge_tables(
        [("triton", incumbent), ("cuda", cuda_plan)],
        pending_of={"triton": {}, "cuda": {}}, record=record)
    assert merged.selected["step_D"] == "fused pair D"
    assert "update_E" not in merged.plans, "half a pair must never be adopted"
    refusal = record["arbitration"]["refused"]["cuda"]["cuda:fused electric pair"]
    assert "held by triton:fused pair D" in refusal


def test_an_unlaunchable_single_arm_plan_is_never_adopted():
    """A CUDA single with no established launch is refused BY SLOT, naming the rule."""
    single = types.SimpleNamespace(launchable=False)
    cuda_plan = types.SimpleNamespace(
        plans={"step_B": single}, selected={"step_B": "PML"}, reasons={})
    record: dict = {}
    merged = fastpath_cuda.merge_tables(
        [("triton", types.SimpleNamespace(plans={}, selected={}, reasons={})),
         ("cuda", cuda_plan)],
        pending_of={"triton": {}, "cuda": {}}, record=record)
    assert merged.plans == {}
    assert "not an adoptable seam" in record["arbitration"]["refused"]["cuda"]["step_B"]


def _certified_single(run: bool = True) -> types.SimpleNamespace:
    """A single as the composer would install it: launchable, with or without a run."""
    plan = types.SimpleNamespace(launchable=True)
    if run:
        plan.run = lambda *_a, **_k: {"launched": True, "blocks": 1}
    return plan


def _empty_triton() -> types.SimpleNamespace:
    return types.SimpleNamespace(plans={}, selected={}, reasons={})


def _merge(cuda_plan, primary=None):
    record: dict = {}
    merged = fastpath_cuda.merge_tables(
        [("triton", primary or _empty_triton()), ("cuda", cuda_plan)],
        pending_of={"triton": {}, "cuda": {}}, record=record)
    return merged, record["arbitration"]["refused"]["cuda"]


def test_a_certified_launchable_single_seam_is_adopted_whole():
    """Both partners present, launchable, with a run and a CUDA_ARM_CERTIFICATION row:
    the seam is adopted as one unit, namespaced, and nothing is refused."""
    for seam in (("step_B", "update_H"), ("step_D", "update_E")):
        curl, constitutive = seam
        merged, refused = _merge(types.SimpleNamespace(
            plans={curl: _certified_single(), constitutive: _certified_single()},
            selected={curl: "PML", constitutive: "ordinary"}, reasons={}))
        assert merged.selected == {curl: "cuda:PML", constitutive: "cuda:ordinary"}
        assert merged.backends == {curl: "cuda", constitutive: "cuda"}
        assert refused == {}


def test_a_lone_certified_single_is_refused_without_its_seam_partner():
    """A single is NEVER adopted alone: under cuda preference this table composes
    first, and a lone launchable step_D would block the incumbent's D/E unit."""
    merged, refused = _merge(types.SimpleNamespace(
        plans={"step_D": _certified_single()}, selected={"step_D": "PML"},
        reasons={}))
    assert merged.plans == {}
    assert "seam partner" in refused["step_D"]


def test_a_launchable_single_without_a_run_is_not_an_adoptable_seam():
    """``launchable`` alone is not enough: the consult calls ``run()``."""
    merged, refused = _merge(types.SimpleNamespace(
        plans={"step_B": _certified_single(run=False),
               "update_H": _certified_single()},
        selected={"step_B": "PML", "update_H": "ordinary"}, reasons={}))
    assert merged.plans == {}
    assert set(refused) == {"step_B", "update_H"}


def test_a_single_whose_label_has_no_certification_row_is_not_adoptable():
    """The record names the byte gate a dispatched arm rides on, so a single
    without a CUDA_ARM_CERTIFICATION row cannot be adopted however launchable.

    The label is a REAL registry update_E label with no row, so the test cannot pass
    on a spelling the composer never writes."""
    assert "nonlinear" in {spec.label for spec in arms.registered("update_E")}
    assert "cuda:nonlinear" not in fastpath_cuda.CUDA_ARM_CERTIFICATION
    merged, refused = _merge(types.SimpleNamespace(
        plans={"step_D": _certified_single(), "update_E": _certified_single()},
        selected={"step_D": "PML", "update_E": "nonlinear"}, reasons={}))
    assert merged.plans == {}
    assert set(refused) == {"step_D", "update_E"}


def test_an_unlaunchable_partner_drops_the_launchable_one_with_it():
    """step_D 'PML' launchable beside an update_E single that declares
    ``launchable=False``: neither is adopted, so the incumbent's D/E unit is left
    whole. This was pml_3d's shape until 2026-09-27, when update_E 'off-diagonal'
    gained a launch; the label is kept because it is CERTIFIED now, so the declared
    launchability is the only fact that fails here."""
    merged, refused = _merge(types.SimpleNamespace(
        plans={"step_D": _certified_single(),
               "update_E": types.SimpleNamespace(launchable=False)},
        selected={"step_D": "PML", "update_E": "off-diagonal"}, reasons={}))
    assert merged.plans == {}
    assert set(refused) == {"step_D", "update_E"}


def test_a_certified_single_seam_never_displaces_an_incumbent():
    """Second in order behind a Triton table holding the same seam, the CUDA seam is
    refused by name naming who holds its slots -- the shipped default precedence."""
    primary = types.SimpleNamespace(
        plans={"step_B": types.SimpleNamespace(run=lambda: None),
               "update_H": types.SimpleNamespace(run=lambda: None)},
        selected={"step_B": "PML", "update_H": "ordinary"}, reasons={})
    merged, refused = _merge(types.SimpleNamespace(
        plans={"step_B": _certified_single(), "update_H": _certified_single()},
        selected={"step_B": "PML", "update_H": "ordinary"}, reasons={}), primary)
    assert merged.selected == {"step_B": "PML", "update_H": "ordinary"}
    assert merged.backends == {"step_B": "triton", "update_H": "triton"}
    assert "held by triton:PML" in refused["cuda:PML"]
    assert "held by triton:ordinary" in refused["cuda:ordinary"]


def _triton_holding_one(slot: str, label: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        plans={slot: types.SimpleNamespace(run=lambda *_a, **_k: None)},
        selected={slot: label}, reasons={})


def _cuda_off_diagonal_seam(update_e_label: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        plans={"step_D": _certified_single(), "update_E": _certified_single()},
        selected={"step_D": "PML", "update_E": update_e_label}, reasons={})


OFF_DIAGONAL_SINGLES = ("off-diagonal", "folded off-diagonal",
                        "dispersive off-diagonal")


def test_the_cross_table_seam_rule_names_exactly_the_certified_off_diagonal_singles():
    """The closed set is the three off-diagonal update_E singles, every one of them
    a certified registry update_E label, and nothing on the PML + ordinary seams."""
    assert fastpath_cuda.CROSS_TABLE_SEAM_REFUSED_SINGLES == {
        f"cuda:{label}" for label in OFF_DIAGONAL_SINGLES}
    assert (fastpath_cuda.CROSS_TABLE_SEAM_REFUSED_SINGLES
            <= set(fastpath_cuda.CUDA_ARM_CERTIFICATION))
    registered = {spec.label for spec in arms.registered("update_E")}
    assert set(OFF_DIAGONAL_SINGLES) <= registered
    assert not {"cuda:PML", "cuda:ordinary"} & (
        fastpath_cuda.CROSS_TABLE_SEAM_REFUSED_SINGLES)


@pytest.mark.parametrize("label", OFF_DIAGONAL_SINGLES)
def test_an_off_diagonal_update_E_single_is_refused_beside_another_tables_step_D(label):
    """The two-table D/E seam: Triton holds step_D and leaves update_E unfilled. The
    CUDA step_D half is blocked on its slot, and the update_E half -- whose own slot
    is free -- is refused BY NAME rather than adopted alone."""
    merged, refused = _merge(_cuda_off_diagonal_seam(label),
                             _triton_holding_one("step_D", "PML"))
    assert merged.selected == {"step_D": "PML"}
    assert merged.backends == {"step_D": "triton"}
    assert "update_E" not in merged.plans
    assert "held by triton:PML" in refused["cuda:PML"]
    reason = refused[f"cuda:{label} (update_E)"]
    assert "its seam partner step_D is held by triton:PML" in reason


@pytest.mark.parametrize("label", OFF_DIAGONAL_SINGLES)
def test_the_step_D_single_is_refused_beside_another_tables_off_diagonal_update_E(
        label):
    """The mirror direction: Triton holds update_E and leaves step_D unfilled. The
    CUDA step_D 'PML' half is refused by name because its partner on this grid is an
    off-diagonal single another table holds."""
    merged, refused = _merge(_cuda_off_diagonal_seam(label),
                             _triton_holding_one("update_E", "off-diagonal"))
    assert merged.selected == {"update_E": "off-diagonal"}
    assert merged.backends == {"update_E": "triton"}
    assert "step_D" not in merged.plans
    reason = refused["cuda:PML (step_D)"]
    assert "its seam partner update_E is held by triton:off-diagonal" in reason


@pytest.mark.parametrize("label", OFF_DIAGONAL_SINGLES)
def test_the_off_diagonal_seam_is_adopted_whole_when_no_other_table_holds_it(label):
    """The control: with neither half held elsewhere the rule refuses nothing, so
    the seam this table composes alone is the one the route campaign drives."""
    merged, refused = _merge(_cuda_off_diagonal_seam(label))
    assert merged.selected == {"step_D": "cuda:PML", "update_E": f"cuda:{label}"}
    assert merged.backends == {"step_D": "cuda", "update_E": "cuda"}
    assert refused == {}


def test_the_pml_and_ordinary_seam_keeps_its_standing_behaviour_as_secondary():
    """Not widened: an ordinary update_E single beside a Triton step_D is still
    adopted alone, which is the open decision SINGLE_ARM_SEAMS' note records."""
    merged, refused = _merge(_cuda_off_diagonal_seam("ordinary"),
                             _triton_holding_one("step_D", "PML"))
    assert merged.backends == {"step_D": "triton", "update_E": "cuda"}
    assert merged.selected["update_E"] == "cuda:ordinary"
    assert "held by triton:PML" in refused["cuda:PML"]


def test_the_two_single_certification_rows_name_the_registry_families():
    """The seam rows point at the byte gates that certified the four entry points,
    and at the REGISTRY families, since a single has no product module."""
    assert fastpath_cuda.CUDA_ARM_CERTIFICATION["cuda:PML"] == (
        "cuda_curl", "bit_identity_gate")
    assert fastpath_cuda.CUDA_ARM_CERTIFICATION["cuda:ordinary"] == (
        "cuda_constitutive", "constitutive_2026-08-15")
    assert set(fastpath_cuda.SINGLE_ARM_SEAMS) == {
        "step_B", "update_H", "step_D", "update_E"}
    for slot, partner in fastpath_cuda.SINGLE_ARM_SEAMS.items():
        assert fastpath_cuda.SINGLE_ARM_SEAMS[partner] == slot


def test_a_pending_cuda_label_is_refused_at_the_merge_rather_than_dispatched():
    class _Pair:
        launchable = True

        def run(self, *args, **kwargs):
            return None

    pair = _Pair()
    label = sorted(fastpath_cuda.CUDA_PENDING_DEVICE_GATE_ARMS)[0]
    bare = fastpath_cuda.bare(label)
    cuda_plan = types.SimpleNamespace(
        plans={"step_D": pair, "update_E": NoopPlan("update_E", pair)},
        selected={"step_D": bare, "update_E": bare}, reasons={})
    record: dict = {}
    merged = fastpath_cuda.merge_tables(
        [("triton", types.SimpleNamespace(plans={}, selected={}, reasons={})),
         ("cuda", cuda_plan)],
        pending_of={"triton": {},
                    "cuda": fastpath_cuda.CUDA_PENDING_DEVICE_GATE_ARMS},
        record=record)
    assert merged.plans == {}
    assert record["arbitration"]["refused"]["cuda"][label]


def test_the_yield_clause_ships_false_with_its_reason_recorded():
    """A pending primary arm refuses the WHOLE plan at rung 7a, and yielding those
    slots to a secondary unit is a composition no leg has driven."""
    assert fastpath_cuda.YIELD_PENDING_PRIMARY_SLOTS is False


def _contested_pending_tables():
    """A primary holding a slot with a PENDING arm, and a launchable CUDA unit on it.

    The shape the campaign's ``cuda_default_precedence`` leg is looking for: Triton
    selects an arm its own table lists as pending, and the hand-CUDA table has a
    whole fused unit over the same slots.
    """

    class _Pair:
        launchable = True

        def run(self, *args, **kwargs):
            return None

    pair = _Pair()
    cuda_plan = types.SimpleNamespace(
        plans={"step_D": pair, "update_E": NoopPlan("update_E", pair)},
        selected={"step_D": "fused electric pair",
                  "update_E": "fused electric pair"},
        reasons={})
    pending_label = sorted(fastpath.PENDING_DEVICE_GATE_ARMS)[0]
    incumbent = types.SimpleNamespace(
        plans={"step_D": types.SimpleNamespace(run=lambda *a, **k: None),
               "update_E": types.SimpleNamespace(run=lambda *a, **k: None)},
        selected={"step_D": pending_label, "update_E": pending_label}, reasons={})
    return incumbent, cuda_plan, pending_label


@pytest.mark.parametrize("yielding", [False, True])
def test_a_pending_primary_slot_yields_ONLY_when_the_clause_is_true(monkeypatch,
                                                                   yielding):
    """BOTH SIDES OF THE CONSTANT, because only one of them ships and the other one
    is what the campaign may flip it to.

    A constant nothing exercises in the direction it is not set to is a constant that
    is measured for the first time ON DEVICE, where flipping it costs two route
    re-gates. The FALSE branch is the shipped rule — the slots stay with the
    incumbent and the CUDA unit is refused by name, and rung 7a then refuses the
    whole plan over the pending primary arm, which is the standing behaviour. The
    TRUE branch is the one the ``cuda_default_precedence`` leg would license: the
    whole unit takes both slots and the record says which table adopted them.
    """
    monkeypatch.setattr(fastpath_cuda, "YIELD_PENDING_PRIMARY_SLOTS", yielding)
    incumbent, cuda_plan, pending_label = _contested_pending_tables()
    record: dict = {}
    merged = fastpath_cuda.merge_tables(
        [("triton", incumbent), ("cuda", cuda_plan)],
        pending_of={"triton": fastpath.PENDING_DEVICE_GATE_ARMS, "cuda": {}},
        record=record)
    if not yielding:
        assert merged.selected["step_D"] == pending_label
        assert merged.backends == {"step_D": "triton", "update_E": "triton"}
        refusal = record["arbitration"]["refused"]["cuda"][
            "cuda:fused electric pair"]
        assert f"held by triton:{pending_label}" in refusal
        assert record["arbitration"]["adopted"] == {}
        return
    assert merged.selected["step_D"] == "cuda:fused electric pair"
    assert merged.selected["update_E"] == "cuda:fused electric pair"
    assert merged.backends == {"step_D": "cuda", "update_E": "cuda"}
    assert record["arbitration"]["adopted"] == {"step_D": "cuda",
                                                "update_E": "cuda"}
    # THE YIELD IS NOT A LICENCE TO SPLIT. The unit took BOTH of the incumbent's
    # slots; a yield that took one would leave half a primary pair standing, which
    # is the shape ``_split_pairs`` refuses at plan time.
    assert "cuda:fused electric pair" not in (
        record["arbitration"]["refused"].get("cuda") or {})
