"""THE INSTALLER WAVE, 2026-09-02: twenty-three certified products, routed and refused.

WHAT CHANGED. Thirty products on the Triton fusion board held a device gate and a
shipped predicate and were reached by no composer at all: ``launch.py`` imported
none of their modules. The board could not even name their cells — its
``served_by_a_product_this_file_cannot_name`` bucket held 136 of the Triton board's
341 served seam-instances (denominator 387), the largest single bucket on any
backend's board — because the reachability join maps a board PRODUCT to the arm
label ``TritonStepPlan.selected`` writes, and a product no composer installs writes
none. :func:`launch._install_certified_fused_products` routes TWENTY-THREE of them,
worth 105 of those 136. The other six were priced and left out, each for a reason
this file pins: the two SCRATCH-OUTPUT welds rotate the engine's own D volumes and
the plan-time warm pass would have driven that rotation (17 instances then), and
the four E->P CHAIN products occupy a seam whose second slot holds a list rather
than a plan (15 instances,
:func:`test_the_four_chain_products_are_deliberately_not_in_the_table`). The two
welds were routed on 2026-09-15, once their shared plan base gained a ``warm`` that
never rotates, and :func:`test_the_two_scratch_output_welds_warm_without_rotating`
pins that mechanism in its new direction.

WHAT THIS FILE ASSERTS, and the order is the order a reader should want it in:

1. THE TABLE IS MEASURED, NOT TRANSCRIBED. Every ``CERTIFIED_FUSED_PAIR_ARMS`` row
   is compared against the fusion board's independently written ``PRODUCTS[...]
   ["cell"]``, which is derived from the coverage conjunction each module opens
   with, citation included. Two files, one fact, and a disagreement is a failure
   here rather than an over-covering dispatch on a device.
2. EVERY PRODUCT REALLY INSTALLS, through ``plan_step`` rather than by calling the
   installer — one parametrised case per product, on the configuration that makes
   its two arms win, asserting BOTH slots and the label in both.
3. RELEASING IS A CAMPAIGN, NOT A WIRING ROW. As of the 2026-09-13 Phase B batch
   twenty of the twenty-six product labels are in ``fastpath.RELEASED_FUSED_ARMS``,
   each let through clause (8) by a driver-route gate rather than by a row added to
   the composer's table, and the remaining six are still refused there by name. A
   released label is admitted at clause (8) only on a shape its release axes cover —
   on the driver-route configuration that is exactly the two real-beta pairs — and
   the set is pinned in both directions, because a product reaching that list without
   a campaign would be the worst outcome available to this wave.
4. THE FAIL-CLOSED CLAUSES STILL STAND IN FRONT: the arm-absorb clause, the
   ambiguity rule, a raising predicate, a raising builder, a builder returning None.
5. THE TWO ORDERING FACTS THIS WAVE DEPENDS ON — the off-diagonal veto runs BEFORE
   the loop (so a slot it pops is refused by name instead of being absorbed, and so
   a future off-diagonal product is not torn out for "having no row-product term"),
   and the incumbent pairs install BEFORE it (so a slot they took is refused by
   name rather than overwritten).
6. THE HOLE THE WAVE OPENED, CLOSED. A fused product writes ONE label into BOTH
   slots, so the arms it substitutes vanish from ``selected`` and the pending-gate
   rung stopped being able to see them — and ``complex_conductive_fused_pair``
   absorbs two arms that are BOTH pending. ``fastpath.FUSED_ARM_CONSTITUENTS`` is
   the see-through and this file measures that the refusal survives the absorb.

NOTHING HERE LAUNCHES A KERNEL. The merge-bar host has no Triton, so the product
builders are stubbed at the one seam the composer reaches them through
(``launch._certified_fused_product_modules``) — the same technique
``test_folded_fused_pair_routing`` uses for the two folded pairs, and for the same
reason. What each product's kernel COMPUTES is its own device gate's business; what
is measured here is the composition.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys
import types

import pytest

_PARITY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       os.pardir, "parity", "meep_gpu"))
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

from meep_gpu import fastpath  # noqa: E402
from meep_gpu.triton_kernels import launch as launch_module  # noqa: E402
from meep_gpu.triton_kernels.coverage import Coverage  # noqa: E402
from meep_gpu.test_dispatch_contract import (  # noqa: E402
    CERTIFIED_CAPABILITY, certified_envelope_fields, certified_envelope_grid,
    certified_envelope_pml, complex_conductive_envelope_fields,
    complex_no_pml_3d_envelope_grid, inactive_envelope_pml, plan_on_a_cupy_host,
    step_plan)
from meep_gpu.test_triton_planner_composition import (  # noqa: E402
    folded_fields, install_stubs, make_fields)

PACKAGE_DIR = pathlib.Path(launch_module.__file__).parent
ADMITTED = Coverage(True, ())
REFUSED = Coverage(False, ("outside this routing test",))


@pytest.fixture(autouse=True)
def _opted_in(monkeypatch):
    """Default the ladder's first two rungs to "not vetoed, opted in".

    The same fixture ``test_dispatch_contract`` runs for the same reason: every
    test here wants to reach the rungs BELOW the enable, and leaving it to the
    environment would make a verdict depend on the shell the suite ran in. The
    composer-only tests are unaffected by it.
    """
    fastpath.reset_dispatch_announcements()
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.delenv(fastpath.DISPATCH_LOG, raising=False)
    monkeypatch.delenv(fastpath.WARM_SWITCH, raising=False)
    yield
    fastpath.reset_dispatch_announcements()

PRODUCTS = launch_module.CERTIFIED_FUSED_PRODUCTS
ARMS = launch_module.CERTIFIED_FUSED_PAIR_ARMS
EXTRA_ARMS = launch_module.CERTIFIED_FUSED_PAIR_EXTRA_ARMS
SEAMS = launch_module.CERTIFIED_FUSED_PAIR_SEAMS

#: The configuration that turns ON both of a product's arm gates, per product.
#:
#: WRITTEN OUT RATHER THAN DERIVED FROM THE LABEL, because deriving it would make
#: this table a second copy of ``plan_step``'s gate expressions and a test that
#: re-implements what it measures measures nothing. Each entry is the smallest
#: fields double whose gates admit the pair, plus the layer: ``None`` for the two
#: no-absorber products (their arms are gated on ``no_active_absorber``) and a live
#: layer for everything else. ``_fold``/``_offdiag``/``_poles`` are this file's
#: shorthands, expanded by :func:`_fields_for`.
CONFIGURATIONS = {
    "complex_fused_magnetic_pair": (dict(has_bloch=True), True),
    "complex_fused_electric_pair": (dict(has_bloch=True), True),
    "cylindrical_fused_magnetic_pair": (
        dict(cylindrical=True, force_complex_fields=True), True),
    "cylindrical_fused_electric_pair": (
        dict(cylindrical=True, force_complex_fields=True), True),
    "cylindrical_real_fused_magnetic_pair": (dict(cylindrical=True), True),
    "cylindrical_real_fused_electric_pair": (dict(cylindrical=True), True),
    "cylindrical_real_fused_hd_pair": (dict(cylindrical=True), True),
    "cylindrical_fused_hd_pair": (
        dict(cylindrical=True, force_complex_fields=True), True),
    # THE CARTESIAN COMPLEX H->D PRODUCT, routed by the 2026-09-07 wiring round. Its
    # shape is the two cylindrical H->D rows' exactly -- a certified product the
    # composer reaches and then refuses on its own INSTALLABLE = False -- so it takes
    # the same live_layer True and belongs in the uninstallable set below.
    "complex_fused_hd_pair": (dict(force_complex_fields=True), True),
    "folded_complex_fused_magnetic_pair": (
        dict(_fold=True, force_complex_fields=True), True),
    "folded_complex_fused_pair": (dict(_fold=True, force_complex_fields=True), True),
    "folded_beta_complex_fused_magnetic_pair": (
        dict(_fold=True, force_complex_fields=True, beta=0.7), True),
    "folded_beta_complex_fused_pair": (
        dict(_fold=True, force_complex_fields=True, beta=0.7), True),
    "folded_beta_fused_magnetic_pair": (dict(_fold=True, beta=0.7), True),
    "folded_beta_fused_electric_pair": (dict(_fold=True, beta=0.7), True),
    "complex_beta_fused_magnetic_pair": (
        dict(beta=0.7, force_complex_fields=True), True),
    "complex_beta_fused_electric_pair": (
        dict(beta=0.7, force_complex_fields=True), True),
    "beta_fused_magnetic_pair": (dict(beta=0.7), True),
    "beta_fused_electric_pair": (dict(beta=0.7), True),
    "bfast_fused_magnetic_pair": (dict(bfast_active=True), True),
    "bfast_fused_electric_pair": (dict(bfast_active=True), True),
    "nonlinear_fused_magnetic_pair": (dict(has_nonlinearity=True), True),
    "folded_dispersive_fused_pair": (dict(_fold=True, _poles=True), True),
    "conductive_fused_electric_pair": (dict(), True),
    # The two whose arms are gated on there being NO active absorber.
    "complex_conductive_fused_pair": (
        dict(force_complex_fields=True, _poles=True), False),
    "no_pml_fused_electric_pair": (dict(), False),
    # The two scratch-output off-diagonal welds, routed 2026-09-15: the off-diagonal
    # flag with a readable row, on the unfolded double and on the folded one.
    "offdiag_fused_electric_pair": (dict(_offdiag=True), True),
    "folded_offdiag_fused_electric_pair": (dict(_fold=True, _offdiag=True), True),
}


class _State:
    """A susceptibility double: the one method the ``update_P`` gate reads."""

    def driven(self):
        return ("Ex",)


def _fields_for(configuration):
    configuration = dict(configuration)
    fold = configuration.pop("_fold", False)
    offdiag = configuration.pop("_offdiag", False)
    poles = configuration.pop("_poles", False)
    if offdiag:
        configuration["has_offdiagonal_epsilon"] = True
        configuration["chi1inv_offdiagonal_for"] = lambda *a, **k: object()
    fields = (folded_fields(**configuration) if fold
              else make_fields(**configuration))
    if poles:
        fields.polarizations = (_State(),)
    return fields


def _stub_product(monkeypatch, name, *, verdict=ADMITTED, plan="<pair>"):
    """Stand in for one product module at the seam the composer imports it through.

    Every one of these modules ends its predicate in an "array module is 'numpy',
    not cupy" clause, so on this host the composer can be driven no further than
    that refusal with the real thing — which is exactly what
    :func:`test_a_real_grid_reaches_every_product_predicate` measures. Everything
    past it is measured here, through the same seam the composer uses rather than
    by calling the installer directly.
    """
    product = PRODUCTS[name]
    module = types.SimpleNamespace(**{
        product["coverage"]: lambda *a, **k: verdict,
        product["builder"]: lambda *a, **k: plan,
    })
    monkeypatch.setattr(launch_module, "_certified_fused_product_modules",
                        lambda: {product["module"]: module})


def _plan_with(monkeypatch, name, **stub_kwargs):
    """``plan_step`` on the configuration that makes ``name``'s two arms win."""
    configuration, live_layer = CONFIGURATIONS[name]
    install_stubs(monkeypatch, admit=ARMS[name])
    if configuration.get("_fold"):
        # ``install_stubs`` pins the fold gate False for every other suite; these
        # products exist for folded grids and need it on.
        monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: True)
    _stub_product(monkeypatch, name, **stub_kwargs)
    return launch_module.plan_step(_fields_for(configuration),
                                   object() if live_layer else None,
                                   fuse=True, sources=())


# ---------------------------------------------------------------------------
# 1. The table is measured against the board, not transcribed
# ---------------------------------------------------------------------------

def _whole_string_literals(source):
    """Every WHOLE string literal in ``source``, f-string fragments excluded.

    The fragments matter: ``launch.py`` writes the ordinary pairs' labels as
    ``f"fused pair {pair_name}"``, whose literal half is ``"fused pair "`` — a
    string ``arm_is_fused`` recognises (it matches on the prefix) and which no
    composer ever writes into a slot. Counting it would put a label that does not
    exist into a totality check and make the check fail on a correct tree.
    """
    tree = ast.parse(source)
    fragments = {id(node) for parent in ast.walk(tree)
                 if isinstance(parent, ast.JoinedStr)
                 for node in ast.walk(parent) if isinstance(node, ast.Constant)}
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in fragments}


def _board_products():
    """``PRODUCTS`` from the fusion board, read from its AST.

    Imported by parse rather than by ``import``: the board module runs corpus and
    census machinery at import time on some paths, and what is wanted here is one
    literal table.
    """
    source = (pathlib.Path(_PARITY) / "build_triton_fusion_matrix.py").read_text(
        encoding="utf-8")
    for node in ast.parse(source).body:
        target = getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id == "PRODUCTS":
            return {key.value: {k.value: ast.literal_eval(v)
                                for k, v in zip(value.keys, value.values)
                                if k.value in ("seam", "cell", "cells", "module")}
                    for key, value in zip(node.value.keys, node.value.values)}
    raise AssertionError("build_triton_fusion_matrix.py no longer defines PRODUCTS")


def test_every_routed_product_is_a_product_the_board_scores():
    """A product this composer installs that no board scores would be a
    composition nothing prices — and its instances would be credited to nobody."""
    board = _board_products()
    assert set(PRODUCTS) <= set(board), sorted(set(PRODUCTS) - set(board))


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_the_absorb_row_is_the_board_cell_for_the_same_product(name):
    """THE ROUTING FACT AND THE REPORTING FACT ARE THE SAME FACT.

    ``_pair_may_absorb`` lets a pair take two slots only when the arm table gave
    them to the arms its kernel implements, and the board derives the same pair
    from the coverage conjunction each module opens with. They are written in two
    files by two different arguments; if they disagree, one of them is describing
    a product that does not exist, and the dangerous direction is silent — a wrong
    absorb row substitutes a numerical product no arm admitted.
    """
    cell = tuple(_board_products()[name]["cell"])
    declared = (ARMS[name],) + tuple(EXTRA_ARMS.get(name, ()))
    assert cell in declared, (name, cell, declared)


def test_the_two_extra_arm_rows_are_the_boards_second_cells():
    """A product the board scores TWICE is one launch over two arm pairs.

    Read from the board rather than asserted: each second cell is a separate board
    product name against the SAME module and the same gate artifact, so the extra
    row here has to be that other product's cell and nothing else.
    """
    board = _board_products()
    seconds = {}
    for name, spec in board.items():
        module = spec["module"].split(".py")[0].split("/")[-1].split(":")[0].strip()
        if name not in PRODUCTS and module in {p["module"] for p in PRODUCTS.values()}:
            seconds[module] = tuple(spec["cell"])
    declared = {PRODUCTS[name]["module"]: tuple(rows)
                for name, rows in EXTRA_ARMS.items()}
    assert {module: (cell,) for module, cell in seconds.items()} == declared, (
        f"the board's second cells are {seconds} but "
        f"CERTIFIED_FUSED_PAIR_EXTRA_ARMS declares {declared}")


def test_the_four_chain_products_are_deliberately_not_in_the_table():
    """THE E->P SEAM IS NOT THIS LOOP'S, and the absence is a decision.

    ``fused_ade_chain``, ``complex_fused_ade_chain``, ``fused_dispersive_chain``
    and ``folded_offdiag_fused_ade_chain`` replace ``update_E`` -> ``update_P``:
    the second slot holds a LIST of per-susceptibility plans rather than a plan,
    the first is a constitutive sub-step rather than a curl, and the first three
    builders take the winning ARM rather than a source list. Any one of those
    makes this loop's contract false about them, so they are left out and the
    reachability join reports them as certified and not installed rather than as
    installed and unreleased.

    THE FOURTH is the folded off-diagonal dispersive body's chain, added with the
    round that built it. Its builder takes no arm at all — one body, one admission
    — so only the SLOT PROTOCOL objection applies to it, which is the one that
    was never about the arm argument in the first place.
    """
    board = _board_products()
    chains = {name for name, spec in board.items() if spec["seam"] == "E->P"}
    assert chains == {"fused_ade_chain", "complex_fused_ade_chain",
                      "fused_dispersive_chain",
                      "folded_offdiag_fused_ade_chain"}
    assert not (chains & set(PRODUCTS))
    assert "update_E" not in SEAMS


def test_the_table_the_seams_and_the_declared_imports_agree():
    """Three declarations, one set of products; any pair drifting is a real defect.

    A module in the table that ``SUPPORT_MODULES`` does not declare breaks the
    lazy-import seam the whole package rests on; a curl slot outside
    ``CERTIFIED_FUSED_PAIR_SEAMS`` would install into a seam with no deposit-list
    name and hand ``deposit_repair`` a list injected outside it.
    """
    assert set(PRODUCTS) == set(ARMS)
    assert set(EXTRA_ARMS) <= set(PRODUCTS)
    modules = {spec["module"] for spec in PRODUCTS.values()}
    assert modules <= set(launch_module.SUPPORT_MODULES), sorted(
        modules - set(launch_module.SUPPORT_MODULES))
    assert modules == set(launch_module._certified_fused_product_modules())
    for name, spec in PRODUCTS.items():
        assert spec["curl_slot"] in SEAMS, name
        assert spec["curl_slot"] in launch_module.STEP_ORDER, name
        assert SEAMS[spec["curl_slot"]][0] in launch_module.STEP_ORDER, name


def test_every_label_is_unique_recognised_as_fused_and_accounted_for():
    """The properties a label needs before it may reach a driver slot.

    "ACCOUNTED FOR" IS A PARTITION, NOT A SINGLE MAP, and the split is this wave's
    honest half. A row in ``ARM_CERTIFICATION`` is a promise that the lookup
    RESOLVES — ``test_every_recorded_certification_points_at_a_readable_record``
    holds every row to a readable ``fingerprints.json`` record with no exemption —
    and only FIVE of the twenty-six products have had a ledger entry cut for them.
    Twenty of the other twenty-one hold a RELEASED gate whose artifact lives in the
    gitignored results tree; they are named in ``PENDING_DEVICE_GATE_ARMS``
    instead, where the rung refuses them before anything warms. The twenty-first,
    ``fused pair D (folded dispersive)``, is on that same side for a different
    reason and one no ledger entry answers — its route gate's second consult site
    diverged, which the reason loop below asserts in its own words. Exactly one of
    the two, for every label: a label in neither could dispatch while naming no
    gate, and a label in both would be claiming a certification the rung says it
    lacks.
    """
    labels = [spec["label"] for spec in PRODUCTS.values()]
    assert len(set(labels)) == len(labels), "two products write the same label"
    certified, pending = set(), set()
    for label in labels:
        assert fastpath.arm_is_fused(label), label
        assert label in fastpath.FUSED_ARM_CONSTITUENTS, label
        in_certification = label in fastpath.ARM_CERTIFICATION
        in_pending = label in fastpath.PENDING_DEVICE_GATE_ARMS
        assert in_certification != in_pending, (
            f"{label} is in {'both' if in_certification else 'neither'} of "
            f"ARM_CERTIFICATION and PENDING_DEVICE_GATE_ARMS")
        (certified if in_certification else pending).add(label)
    # 3 -> 5 ON 2026-09-11: two products moved across the partition when
    # `dispatch_fused_route_2026-09-11_realarms` released them and
    # seed_triton_welds.py cut their ledger entries from that campaign's fleet
    # (`conductive_fused_electric_pair`, `cylindrical_real_fused_electric_pair`).
    # The third arm released that round, `fused pair B (cylindrical)`, was ALREADY
    # on this side of the partition — its ledger entry was cut on 2026-08-20 —
    # which is why five and not six.
    #
    # `folded_dispersive_fused_pair` DID NOT CROSS, though the same campaign drove
    # it: its preflight held it at the route gate's second consult site, where
    # `synchronize_magnetic_fields` read fused_vs_array False and unfused_vs_array
    # False on a leg installing no fused product. A ledger entry certifies BYTES
    # and the divergence is the seam's, so cutting one would not have moved it.
    #
    # 5 -> 20 ON 2026-09-13: the Phase B batch (`dispatch_fused_route_2026-09-13_phaseB`)
    # released fifteen more Triton arms — the real/complex/folded beta pairs, the BFAST
    # pairs, the nonlinear B pair, the complex pairs, the folded complex pairs and the
    # cylindrical complex pairs — and seed_triton_welds.py cut a ledger entry for each
    # from the 2026-09-12 identity fleet, so each crossed to the certified side.
    # 20 -> 22 ON 2026-09-14: the target round released `fused pair D (no-PML stored E)`
    # (its weld had existed since the 2026-09-13 identity fleet; the pending text was
    # stale) and the re-attribution round released `fused pair D (folded dispersive)`
    # after the 2026-09-11 synchronize_magnetic_fields hold was traced to the lazy
    # subnormal-policy install rather than to the seam. Both carry a PASS weld.
    # 22 -> 23 ON 2026-09-15: `fused pair D (complex conductive no-PML)` moves when its
    # ARM_CERTIFICATION row lands. The key that row names is cut by
    # seed_triton_welds.py from the pair gate re-run AFTER that edit, so the resolve
    # loop below is red for this one label until the seed lands, by design.
    # 23 -> 25 ON 2026-09-17: the two scratch-output off-diagonal welds were routed
    # and their labels go straight to the certified side, keyed to the ledger entries
    # seed_triton_welds.py cuts from the stencil gate re-run on the routing bytes.
    # Until that seed runs the `recorded_utc` lookup below is red on those two labels
    # BY NAME -- the window the 2026-09-11 wave ran through, and no pending hold.
    assert len(certified) == 25, sorted(certified)
    # 20 -> 22 ON 2026-09-07: the two cylindrical H->D products' labels, routed and
    # pending a ledger entry; each pending reason also names the INSTALLABLE = False
    # the product declares, which is what keeps the label out of every slot today.
    # 23 since the 2026-09-07 wiring round routed `complex_fused_hd_pair`, whose
    # label `fused pair H->D (complex)` is PENDING for the reason its own module
    # gives: INSTALLABLE = False, so the composer refuses it on every row.
    # 23 -> 21 ON 2026-09-11, the same two crossing the other way; the product
    # table did not move, so the denominator is still twenty-six. It is 21 and not
    # 20 because the campaign's fourth candidate, `fused pair D (folded
    # dispersive)`, stayed on this side — see the note on the certified count.
    # 21 -> 6 ON 2026-09-13: the fifteen arms the Phase B batch released left the
    # pending side, so the six that remain are `folded_dispersive_fused_pair` (held
    # for its own reason, below), the two no-absorber D products
    # (`complex_conductive_fused_pair`, `no_pml_fused_electric_pair`) and the three
    # H->D products, whose INSTALLABLE = False keeps them out of every slot. The
    # denominator is still twenty-six.
    # 6 -> 4 ON 2026-09-14: the no-PML stored E and folded dispersive arms left pending
    # (see the certified-count note above); what remains is the complex conductive
    # no-PML pair, whose PAIR weld the target fleet seeds, and the three H->D spans.
    # 4 -> 3 ON 2026-09-15: the complex conductive no-PML pair left for
    # ARM_CERTIFICATION (see the certified-count note above); the three H->D spans
    # remain, each held by its missing ledger entry and its product's
    # INSTALLABLE = False.
    assert len(pending) == 3, len(pending)
    # ...and each certified row really resolves, which is what the other map's
    # rows are held out of for. RESOLVING IS PER ARCHITECTURE since the welds keep
    # one run record per compute capability: the question is whether the gate has a
    # live run on the card the label would dispatch on, so the lookup names it. Asked
    # without a capability, every row would come back with its run fields withheld
    # by design and this loop would read as a ledger-wide failure.
    for label in certified:
        assert "recorded_utc" in fastpath._certification_for(
            label, capability=CERTIFIED_CAPABILITY), label
    # ...while each pending reason names the artifact the release was cut from, so
    # the run that must be repeated to lift it is identified, not searched for.
    #
    # THE LABEL THIS BLOCK USED TO HOLD BACK IS RELEASED, AND THE HOLD WAS WRONG.
    # Until 2026-09-14 `fused pair D (folded dispersive)` sat in the pending table
    # on the claim that the route gate's second consult site diverged from the
    # array path with fusion off as well as on, and this test asserted that reason
    # by name. The re-drive on the GPU host found the seam clean at 7/7 slots over the
    # full ladder and traced the 2026-09-11 signature to the LAZY subnormal-policy
    # install (a divergence carried INTO the site, not produced by it) -- see
    # the design notes (meep-gpu-dispatch-expansion-plan) section 11. The guard is
    # therefore inverted rather than deleted: the label is CERTIFIED, its record
    # is readable, and NO pending reason may still cite that consult site, because
    # a hold written against a retired diagnosis would blend in as a ledger wait.
    released_after_reattribution = "fused pair D (folded dispersive)"
    assert released_after_reattribution in certified, released_after_reattribution
    assert released_after_reattribution not in pending
    for label in pending:
        reason = fastpath.PENDING_DEVICE_GATE_ARMS[label]
        assert "synchronize_magnetic_fields" not in reason, (
            f"{label}: a pending reason still cites the consult site whose 2026-09-11 "
            f"divergence was re-attributed to the policy-install ordering")
        assert "gate.json" in reason and "fingerprints.json" in reason, label
    # ...and none collides with an ARM label, which would make `selected` ambiguous
    # between "an arm won this slot" and "a pair absorbed it".
    incumbents = {"fused pair B", "fused pair D", "dispersive fused pair",
                  "fused pair B (folded)", "fused pair D (folded)",
                  "fused ADE state"}
    arms = set(fastpath.ARM_CERTIFICATION) - set(labels) - incumbents
    assert not (set(labels) & arms)


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_the_constituent_map_names_exactly_the_arms_the_product_absorbs(name):
    """``FUSED_ARM_CONSTITUENTS`` is what the pending-gate rung sees through.

    Read against the COMPOSER'S tables rather than shared with them: a row that
    named fewer arms than the product absorbs would hide a pending arm, and one
    that named more would refuse a configuration nothing forbids.
    """
    label = PRODUCTS[name]["label"]
    declared = set(fastpath.FUSED_ARM_CONSTITUENTS[label])
    absorbed = set(ARMS[name])
    for extra in EXTRA_ARMS.get(name, ()):
        absorbed |= set(extra)
    assert declared == absorbed, (name, label, sorted(declared), sorted(absorbed))


def test_the_incumbent_pairs_are_in_the_constituent_map_too():
    """The map has to be total over fused labels, not only over the new ones: a
    fused label MISSING from it is refused by name at the same rung, which would
    turn the released arms off if they were left out."""
    assert launch_module.FUSED_PAIR_ARMS["B"] == \
        fastpath.FUSED_ARM_CONSTITUENTS["fused pair B"]
    assert launch_module.FUSED_PAIR_ARMS["D"] == \
        fastpath.FUSED_ARM_CONSTITUENTS["fused pair D"]
    assert launch_module.FUSED_PAIR_ARMS["dispersive_D"] == \
        fastpath.FUSED_ARM_CONSTITUENTS["dispersive fused pair"]
    for label in ("fused pair B (folded)", "fused pair D (folded)"):
        assert fastpath.FUSED_ARM_CONSTITUENTS[label] == \
            launch_module.FOLDED_FUSED_PAIR_ARMS


def test_the_constituent_map_is_total_over_the_labels_the_composer_can_write():
    """A LABEL MISSING FROM IT IS REFUSED BY NAME, so the map has to be total.

    The composer's fused vocabulary is four tables: the ordinary pairs, the two
    folded pairs, this wave's twenty-five, and the ``fuse_ade`` block's one. Read
    off ``launch.py``'s source for the last, because it is written inline at the
    slot rather than declared in a table, and a scan that only read the tables
    would call the map total while one label went unnamed.
    """
    writable = {"fused pair B", "fused pair D", "dispersive fused pair"}
    writable |= {spec["label"] for spec in launch_module.FOLDED_FUSED_PAIRS.values()}
    writable |= {spec["label"] for spec in PRODUCTS.values()}
    source = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    writable |= {text for text in _whole_string_literals(source)
                 if fastpath.arm_is_fused(text)}
    assert "fused ADE state" in writable, "the fuse_ade label is no longer inline"
    assert writable <= set(fastpath.FUSED_ARM_CONSTITUENTS), sorted(
        writable - set(fastpath.FUSED_ARM_CONSTITUENTS))
    assert set(fastpath.FUSED_ARM_CONSTITUENTS) == writable, sorted(
        set(fastpath.FUSED_ARM_CONSTITUENTS) - writable)


# ---------------------------------------------------------------------------
# 2. Every product installs, through the composer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_each_certified_product_takes_both_slots_and_names_itself(monkeypatch, name):
    """THE FAIL-CLOSED HALF OF THE TWO-SLOT CONTRACT, per product.

    ``FastPath.dispatch`` contracts that True means the whole sub-step ran, so a
    pair owning ``step_D`` and ``update_E`` must appear in BOTH — otherwise a
    reader, and every regression test that reads ``plan.replaces``, is told the
    constitutive update ran on the array path when the launch performed it.
    """
    from . import withdraw_hoist  # noqa: PLC0415

    plan = _plan_with(monkeypatch, name, plan=f"<{name}>")
    curl = PRODUCTS[name]["curl_slot"]
    update, seam_name = SEAMS[curl]
    label = PRODUCTS[name]["label"]

    leading = plan.plans[curl]
    if seam_name == withdraw_hoist.SEAM:
        # THE H->D SEAM (2026-09-07, the two cylindrical products): the installer's
        # first branch puts the pair INSIDE a ``LeadingWithdrawPlan`` that withdraws
        # the standing electric integrated sources before the launch, and the
        # sentinel in the second slot names the PAIR, not the wrapper -- the same
        # two-slot contract, read one level down.
        assert isinstance(leading, withdraw_hoist.LeadingWithdrawPlan), (
            leading, plan.reasons.get(f"fused_pair_{name}"))
        assert leading.inner == f"<{name}>"
        assert leading.span == (curl, update)
        pair = leading.inner
    else:
        assert leading == f"<{name}>", plan.reasons.get(f"fused_pair_{name}")
        pair = leading
    sentinel = plan.plans[update]
    assert isinstance(sentinel, launch_module.NoopPlan)
    assert sentinel.absorbed_by is pair
    assert plan.selected[curl] == plan.selected[update] == label
    assert curl in plan.replaces and update in plan.replaces


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_a_refused_predicate_leaves_both_slots_on_their_separate_arms(
        monkeypatch, name):
    """A refused fusion is not a refused step: both sub-steps keep the plans the
    arm table built, and the reason is reported beside them."""
    plan = _plan_with(monkeypatch, name, verdict=REFUSED)
    curl = PRODUCTS[name]["curl_slot"]
    update = SEAMS[curl][0]

    assert plan.selected[curl] == ARMS[name][0]
    assert plan.selected[update] == ARMS[name][1]
    assert plan.reasons[f"fused_pair_{name}"] == REFUSED.reasons


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_a_builder_returning_none_leaves_the_seam_unfused_with_a_named_reason(
        monkeypatch, name):
    plan = _plan_with(monkeypatch, name, plan=None)
    curl = PRODUCTS[name]["curl_slot"]

    assert plan.selected[curl] == ARMS[name][0]
    assert plan.reasons[f"fused_pair_{name}"] == (f"the {name} pair was refused",)


def test_a_raising_predicate_is_a_refusal_and_never_an_exception(monkeypatch):
    """``plan_step`` never raises; an opt-in optimisation that crashes it is
    strictly worse than one that refuses it."""
    name = "complex_fused_electric_pair"
    product = PRODUCTS[name]

    def boom(*args, **kwargs):
        raise RuntimeError("predicate exploded")

    install_stubs(monkeypatch, admit=ARMS[name])
    monkeypatch.setattr(
        launch_module, "_certified_fused_product_modules",
        lambda: {product["module"]: types.SimpleNamespace(**{
            product["coverage"]: boom,
            product["builder"]: lambda *a, **k: "<pair>"})})

    plan = launch_module.plan_step(_fields_for(CONFIGURATIONS[name][0]), object(),
                                   fuse=True, sources=())

    assert plan.selected["step_D"] == "complex PML"
    assert "predicate exploded" in plan.reasons[f"fused_pair_{name}"][0]


def test_a_raising_builder_leaves_the_seam_on_its_separate_plans(monkeypatch):
    name = "complex_fused_electric_pair"
    product = PRODUCTS[name]

    def boom(*args, **kwargs):
        raise RuntimeError("builder exploded")

    install_stubs(monkeypatch, admit=ARMS[name])
    monkeypatch.setattr(
        launch_module, "_certified_fused_product_modules",
        lambda: {product["module"]: types.SimpleNamespace(**{
            product["coverage"]: lambda *a, **k: ADMITTED,
            product["builder"]: boom})})

    plan = launch_module.plan_step(_fields_for(CONFIGURATIONS[name][0]), object(),
                                   fuse=True, sources=())

    assert plan.selected["step_D"] == "complex PML"
    assert plan.selected["update_E"] == "complex"
    assert "builder exploded" in plan.reasons[f"fused_pair_{name}"][0]


def test_an_unimportable_module_set_refuses_every_product_by_name(monkeypatch):
    """The merge-bar host's own case, generalised: if the import block raises,
    every product on the table reports it rather than one of them crashing the
    plan and the other twenty-four vanishing."""
    def boom():
        raise ImportError("no module named 'triton'")

    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "_certified_fused_product_modules", boom)

    plan = launch_module.plan_step(make_fields(), object(), fuse=True, sources=())

    for name in PRODUCTS:
        reasons = plan.reasons[f"fused_pair_{name}"]
        assert reasons and "ImportError" in reasons[0], name


def test_fusion_off_consults_no_certified_product(monkeypatch):
    """``fuse=False`` is the shipped default and must not pay for this block."""
    calls = []
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "_certified_fused_product_modules",
                        lambda: calls.append(1) or {})

    plan = launch_module.plan_step(make_fields(), object(), sources=())

    assert calls == []
    assert not [key for key in plan.reasons if key.startswith("fused_pair_")]


# ---------------------------------------------------------------------------
# 3. The arm-absorb clause and the ambiguity rule still stand in front
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_a_product_may_not_absorb_slots_a_different_arm_won(monkeypatch, name):
    """THE CLAUSE THAT MAKES THE WHOLE TABLE SAFE.

    The predicate admits, the builder is willing, and the arm table gave the slots
    to somebody else — so the pair does NOT take them, and the reason names the arm
    that did. Driven for every product rather than once, because a table with
    twenty-five rows has twenty-five chances to name the wrong pair and only this
    clause stands between a wrong row and an unmeasured substitution.
    """
    configuration, live_layer = CONFIGURATIONS[name]
    install_stubs(monkeypatch, admit=())          # nothing wins any slot
    if configuration.get("_fold"):
        monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: True)
    _stub_product(monkeypatch, name)

    plan = launch_module.plan_step(_fields_for(configuration),
                                   object() if live_layer else None,
                                   fuse=True, sources=())

    curl = PRODUCTS[name]["curl_slot"]
    assert curl not in plan.plans
    assert plan.reasons[f"fused_pair_{name}"] == (
        f"{curl} is not selected by any arm — the composition left it on the "
        "array path, and a fused pair may not claim a slot the arm table refused",)


def test_a_second_admitting_product_on_one_seam_leaves_it_unfused(monkeypatch):
    """TWO ADMITTERS ON ONE SEAM IS AN AMBIGUITY, NOT A PICK — clause (b)'s rule,
    one level up. Both products are refused with a reason naming both; neither is
    chosen by table order, and the arm table's own plans are what run."""
    first, second = "complex_fused_electric_pair", "cylindrical_fused_electric_pair"
    install_stubs(monkeypatch, admit=ARMS[first])
    monkeypatch.setattr(
        launch_module, "_certified_fused_product_modules",
        lambda: {PRODUCTS[name]["module"]: types.SimpleNamespace(**{
            PRODUCTS[name]["coverage"]: lambda *a, **k: ADMITTED,
            PRODUCTS[name]["builder"]: lambda *a, **k: f"<{name}>"})
            for name in (first, second)})

    plan = launch_module.plan_step(_fields_for(CONFIGURATIONS[first][0]), object(),
                                   fuse=True, sources=())

    assert plan.selected["step_D"] == "complex PML"
    assert plan.selected["update_E"] == "complex"
    for name in (first, second):
        reason, = plan.reasons[f"fused_pair_{name}"]
        assert first in reason and second in reason
        assert "step_D/update_E seam" in reason
        assert "left unfused rather than assigned by table order" in reason


def test_an_ambiguity_on_one_seam_does_not_disturb_the_other(monkeypatch):
    """The loop decides ONE SEAM AT A TIME, which is what lets a magnetic and an
    electric product share an arm pair without ever being ambiguous."""
    b_name, d_first = "complex_fused_magnetic_pair", "complex_fused_electric_pair"
    d_second = "cylindrical_fused_electric_pair"
    install_stubs(monkeypatch, admit=ARMS[b_name])
    monkeypatch.setattr(
        launch_module, "_certified_fused_product_modules",
        lambda: {PRODUCTS[name]["module"]: types.SimpleNamespace(**{
            PRODUCTS[name]["coverage"]: lambda *a, **k: ADMITTED,
            PRODUCTS[name]["builder"]: lambda *a, **k: f"<{name}>"})
            for name in (b_name, d_first, d_second)})

    plan = launch_module.plan_step(_fields_for(CONFIGURATIONS[b_name][0]), object(),
                                   fuse=True, sources=())

    assert plan.selected["step_B"] == PRODUCTS[b_name]["label"]
    assert plan.selected["update_H"] == PRODUCTS[b_name]["label"]
    assert plan.selected["step_D"] == "complex PML"


def test_an_extra_arm_row_absorbs_its_second_cell(monkeypatch):
    """The dual-cell products take the seam on EITHER declared arm pair.

    ``no_pml_fused_electric_pair``'s primary row is the conductive no-absorber
    curl; its extra row is the lossless one, and the same launch serves both. The
    primary-row test above drives the first; this drives the second, so neither is
    left to a claim.
    """
    name = "no_pml_fused_electric_pair"
    extra, = EXTRA_ARMS[name]
    install_stubs(monkeypatch, admit=extra)
    _stub_product(monkeypatch, name, plan=f"<{name}>")

    plan = launch_module.plan_step(_fields_for(CONFIGURATIONS[name][0]), None,
                                   fuse=True, sources=())

    assert plan.selected["step_D"] == PRODUCTS[name]["label"]
    assert plan.selected["update_E"] == PRODUCTS[name]["label"]


# ---------------------------------------------------------------------------
# 4. The two ordering facts the wave depends on
# ---------------------------------------------------------------------------

def test_the_two_scratch_output_welds_warm_without_rotating():
    """THE EXCLUSION THIS TEST USED TO PIN, INVERTED, WITH THE SAME MECHANISM.

    Until 2026-09-15 this was ``test_the_two_scratch_output_welds_are_deliberately_
    not_wired``. It read off the tree every step of why
    ``offdiag_fused_electric_pair`` and ``folded_offdiag_fused_electric_pair`` —
    certified, released and predicated like every product in the table — stayed
    out: their plan rotates the ENGINE'S OWN ``D``/``fu_D`` after each launch; the
    plan class defined no ``warm``, so ``fastpath.warm_plan`` fell through to
    ``_warm_with_empty_grid``; that function empties the settable ``_grid`` slot
    and CALLS ``run``; and ``run`` rotates unconditionally, leaving the engine's D
    volumes on the plan's zero twins at plan time.

    The base now defines ``warm``, and each step is read in its new direction:
    ``warm`` exists and ``warm_plan`` asks for it before it reaches the empty-grid
    ``run``; its body launches through ``_launch`` and neither rotates nor counts;
    ``run`` STILL rotates, so the design did not change and only the plan-time
    path did; and both products are routed on the arm pair the board records. What
    the warm DOES at runtime is measured by the two family suites (a recording stub
    kernel through the shipped builders, with the old empty-grid path kept as an
    armed null) and by the stencil gate's ``warm`` legs; this is the table's side.
    """
    import inspect

    from meep_gpu.triton_kernels import offdiag_scratch_weld

    routed = {
        "offdiag_fused_electric_pair": ("fused pair D (off-diagonal)",
                                        ("PML", "off-diagonal")),
        "folded_offdiag_fused_electric_pair": ("fused pair D (folded off-diagonal)",
                                               ("folded PML", "folded off-diagonal")),
    }
    board = _board_products()
    for name, (label, arms) in routed.items():
        assert name in board, name
        assert PRODUCTS[name]["label"] == label, name
        assert PRODUCTS[name]["curl_slot"] == "step_D", name
        assert ARMS[name] == arms == tuple(board[name]["cell"]), name
        assert name in launch_module.SUPPORT_MODULES, name
    plan_class = offdiag_scratch_weld.ScratchWeldPairPlan
    assert callable(getattr(plan_class, "warm", None)), \
        "without warm() the plan-time warm pass reaches run() and rotates"
    assert "_grid" in plan_class.__slots__, \
        "_warm_with_empty_grid would still reach run() through this slot"
    warm_body = inspect.getsource(plan_class.warm).split('"""')[-1]
    assert "self._launch(" in warm_body
    assert "_rotate" not in warm_body, "the warm rotates"
    assert "launches" not in warm_body, "the warm counts a launch"
    assert "finally:" in warm_body, "a raising compile would leave the grid empty"
    run_body = inspect.getsource(plan_class.run).split('"""')[-1]
    assert "self._rotate()" in run_body, \
        "run() no longer rotates; the reason warm() exists is gone"
    warm_plan_source = inspect.getsource(fastpath.warm_plan)
    assert (warm_plan_source.index('getattr(plan, "warm"')
            < warm_plan_source.index("_warm_with_empty_grid(plan, plan.run)")), \
        "warm_plan no longer asks for warm() before the empty-grid run()"


def test_a_vetoed_update_E_leaves_every_candidate_refused(monkeypatch):
    """AND THE FAIL-CLOSED HALF OF THAT ORDERING. When the veto DOES fire it pops
    ``selected['update_E']``, so the loop that runs next finds the slot unselected
    and refuses every candidate by name. There is no configuration on which this
    loop installs a product the veto would have refused."""
    name = "complex_fused_electric_pair"
    install_stubs(monkeypatch, admit=ARMS[name])
    _stub_product(monkeypatch, name, plan=f"<{name}>")
    fields = _fields_for(dict(CONFIGURATIONS[name][0], _offdiag=True))

    plan = launch_module.plan_step(fields, object(), fuse=True, sources=())

    assert "update_E" not in plan.plans
    assert "row-product term" in plan.reasons["update_E"][0]
    assert plan.reasons[f"fused_pair_{name}"] == (
        "update_E is not selected by any arm — the composition left it on the "
        "array path, and a fused pair may not claim a slot the arm table refused",)


def test_an_incumbent_pair_that_took_a_slot_refuses_the_certified_one_by_name(
        monkeypatch):
    """THE CROSS-BLOCK PRECEDENCE RULE, and it is not a table order.

    The incumbent hand-written blocks install first; this loop reads the LIVE
    ``selected``, so a slot an incumbent absorbed carries a fused label and
    ``_pair_may_absorb`` reports exactly which product holds it. No new rule was
    needed for that, and none was added.
    """
    name = "conductive_fused_electric_pair"
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "fused_pair_coverage",
                        lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_fused_pair",
                        lambda *a, **k: "<incumbent D>")
    _stub_product(monkeypatch, name, plan=f"<{name}>")

    plan = launch_module.plan_step(make_fields(), object(), fuse=True, sources=())

    assert plan.selected["step_D"] == "fused pair D"
    reason, = plan.reasons[f"fused_pair_{name}"]
    assert "'fused pair D' arm" in reason
    assert "'conductive PML'" in reason


# ---------------------------------------------------------------------------
# 5. WIRING IS NOT RELEASING
# ---------------------------------------------------------------------------

def test_only_the_released_product_labels_are_released(monkeypatch):
    """The released set, asserted against the shipped release rather than against a
    memory of it — an ENUMERATION, so a label reaching ``RELEASED_FUSED_ARMS``
    without a campaign fails on this line rather than dispatching quietly.

    THIS TEST WAS ``test_no_certified_product_label_is_released`` UNTIL 2026-09-11,
    then ``..._only_the_three_...`` until the 2026-09-13 Phase B batch. A product's
    label reaches ``RELEASED_FUSED_ARMS`` by a CAMPAIGN, never by a row added to the
    composer's table, which is why the assertion is a set: after Phase B twenty of
    the twenty-six product labels are released, and the six that are not are
    ``folded_dispersive_fused_pair`` (held at the route gate's SECOND consult site,
    where ``synchronize_magnetic_fields`` diverges from the array path whether or not
    anything is fused), the two no-absorber D products, and the three H->D products
    (``INSTALLABLE = False``).

    THE SECOND HALF NOW ADMITS TWO. On the Cartesian, lossless, unfolded, non-
    cylindrical shape the 2026-08-30 gate drove, the two real-beta pairs ARE
    admitted: their rows in ``FUSED_RELEASE_ARM_AXES`` do not constrain the kz/beta
    axis — a real-beta run is a 2-D Cartesian lossless grid like this envelope — so
    unlike the conductive and cylindrical pairs, which require a conductivity or a
    Dcyl grid the envelope lacks, nothing about this shape keeps them out.
    """
    labels = {spec["label"] for spec in PRODUCTS.values()}
    assert labels & set(fastpath.RELEASED_FUSED_ARMS) == {
        "fused pair B (BFAST)", "fused pair B (complex beta)", "fused pair B (complex)",
        "fused pair B (cylindrical complex)", "fused pair B (cylindrical)",
        "fused pair B (folded complex beta)", "fused pair B (folded complex)",
        "fused pair B (folded real beta)", "fused pair B (nonlinear)",
        "fused pair B (real beta)",
        "fused pair D (BFAST)", "fused pair D (complex beta)", "fused pair D (complex)",
        "fused pair D (conductive)", "fused pair D (cylindrical complex)",
        "fused pair D (cylindrical)", "fused pair D (folded complex beta)",
        "fused pair D (folded complex)", "fused pair D (folded real beta)",
        "fused pair D (real beta)",
        # RELEASED 2026-09-14 by the re-attribution round (see the certified-count note above).
        "fused pair D (folded dispersive)",
        "fused pair D (no-PML stored E)",
        # RELEASED 2026-09-15 on complex_no_pml_3d (see the certified-count note above).
        "fused pair D (complex conductive no-PML)",
        # RELEASED 2026-09-17 with the routing of the two scratch-output welds.
        "fused pair D (off-diagonal)", "fused pair D (folded off-diagonal)"}
    grid = certified_envelope_grid()
    shape = fastpath._run_shape(certified_envelope_fields(), certified_envelope_pml(),
                                grid)
    assert not fastpath.fused_release_reasons(shape), \
        "this fixture must be INSIDE the envelope or the next assertion is vacuous"
    assert labels & set(fastpath.released_fused_arms(shape)) == {
        "fused pair B (real beta)", "fused pair D (real beta)"}


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_a_plan_carrying_a_certified_label_is_refused_at_the_fusion_clause(
        monkeypatch, name):
    """EVEN INSIDE THE RELEASED ENVELOPE, the configuration below is the one the
    driver-route gate drove, so nothing about the SHAPE refuses a product here —
    only its release status does. Since the 2026-09-13 Phase B batch the two
    real-beta pairs are released FOR THIS SHAPE and are ADMITTED past clause (8) and
    every rung below it (the plan then fails to build on this Triton-less host, a
    different refusal); the other twenty-four are still refused at clause (8) by
    name, because their release axes are not this shape's or because they are not
    released at all. A product crossing from the second branch to the first without
    a campaign fails here rather than dispatching quietly.
    """
    label = PRODUCTS[name]["label"]
    curl = PRODUCTS[name]["curl_slot"]
    update = SEAMS[curl][0]
    shape = fastpath._run_shape(certified_envelope_fields(), certified_envelope_pml(),
                                certified_envelope_grid())
    admitted_for_shape = label in set(fastpath.released_fused_arms(shape))
    plan = plan_on_a_cupy_host(
        monkeypatch,
        step_plan({curl: object(), update: object()},
                  {curl: label, update: label}),
        fields=certified_envelope_fields(), grid=certified_envelope_grid(),
        pml=certified_envelope_pml())

    assert plan is None
    refused = fastpath.last_dispatch_report()["refused_because"]
    if admitted_for_shape:
        # RELEASED FOR THIS SHAPE (the two real-beta pairs): dispatch ADMITS the
        # label past clause (8) and every rung below it, so it is NOT refused there
        # by name. On this Triton-less host the plan then fails to build — the
        # refusal that proves the release clause was cleared and the build reached.
        assert "no slot is left carrying a kernel" in refused, refused
        assert "has not been driven through the driver seam" not in refused, refused
        assert label not in refused, refused
    else:
        assert label in refused, refused
        assert "has not been driven through the driver seam" in refused


def test_the_complex_conductive_pair_is_certified_and_released_on_its_own_shape(
        monkeypatch):
    """REWRITTEN AGAIN 2026-09-15, when the pair left ``PENDING_DEVICE_GATE_ARMS``.

    WHAT IT TESTED UNTIL THEN. After the 2026-09-14 release decision certified both
    constituents, the pair was still pending on its OWN account -- its gate had
    released and no ``fingerprints.json`` entry had been cut from it -- so an
    opted-in run had to be refused by that reason. The 2026-09-15 batch moves the
    label into ``ARM_CERTIFICATION`` and releases it on ``complex_no_pml_3d``, which
    removes that refusal.

    WHAT IT TESTS NOW is the release that replaced it, in both directions and with
    the switch UNSET: the partition moved whole (both constituents and the pair
    certified, nothing pending, the lookup naming the key
    ``seed_triton_welds.ledger_key`` derives from the driver's gate directory); on
    its own case's shape the shipped release admits the label and the ladder clears
    clause (8) and the pending rung, so the plan reaches the build; and on the
    certified PML envelope the release still refuses it by name.

    THE LOOK-STRAIGHT-PAST HAZARD STILL HAS NO INSTANCE, as the 2026-09-14 rewrite
    recorded: no label in ``FUSED_ARM_CONSTITUENTS`` has a pending constituent, and
    ``test_a_fused_label_the_constituent_map_cannot_name_refuses_the_whole_plan`` is
    the rung's coverage until one arrives.
    """
    name = "complex_conductive_fused_pair"
    label = PRODUCTS[name]["label"]
    for arm in fastpath.FUSED_ARM_CONSTITUENTS[label]:
        assert arm not in fastpath.PENDING_DEVICE_GATE_ARMS, arm
        assert arm in fastpath.ARM_CERTIFICATION, arm
    assert label not in fastpath.PENDING_DEVICE_GATE_ARMS, label
    assert fastpath.ARM_CERTIFICATION[label] == (name, f"triton_{name}_device_gate")
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)

    own = dict(fields=complex_conductive_envelope_fields(),
               grid=complex_no_pml_3d_envelope_grid(), pml=inactive_envelope_pml())
    assert label in fastpath.released_fused_arms(
        fastpath._run_shape(own["fields"], own["pml"], own["grid"]))
    plan = plan_on_a_cupy_host(
        monkeypatch,
        step_plan({"step_D": object(), "update_E": object()},
                  {"step_D": label, "update_E": label}),
        **own)
    assert plan is None
    refused = fastpath.last_dispatch_report()["refused_because"]
    # ADMITTED, so the refusal is the build's on this Triton-less host and not the
    # release clause's or the pending rung's.
    assert "no slot is left carrying a kernel" in refused, refused
    assert "has not been driven through the driver seam" not in refused, refused
    assert label not in refused, refused

    plan = plan_on_a_cupy_host(
        monkeypatch,
        step_plan({"step_D": object(), "update_E": object()},
                  {"step_D": label, "update_E": label}),
        fields=certified_envelope_fields(), grid=certified_envelope_grid(),
        pml=certified_envelope_pml())
    assert plan is None
    refused = fastpath.last_dispatch_report()["refused_because"]
    assert label in refused, refused
    assert "has not been driven through the driver seam" in refused, refused


def test_a_fused_label_the_constituent_map_cannot_name_refuses_the_whole_plan(
        monkeypatch):
    """FAIL-CLOSED ON AN UNNAMED ABSORB. A future product wired without a
    constituent row would otherwise reach every rung below with two arms nothing
    can see; it is refused by name instead, which is the direction that costs a
    coverage gap rather than a wrong number."""
    label = "fused pair D (a product this file cannot name)"
    assert fastpath.arm_is_fused(label)
    assert label not in fastpath.FUSED_ARM_CONSTITUENTS
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, label)

    plan = plan_on_a_cupy_host(
        monkeypatch,
        step_plan({"step_D": object(), "update_E": object()},
                  {"step_D": label, "update_E": label}),
        fields=certified_envelope_fields(), grid=certified_envelope_grid(),
        pml=certified_envelope_pml())

    assert plan is None
    assert "FUSED_ARM_CONSTITUENTS" in fastpath.last_dispatch_report()[
        "refused_because"]


# ---------------------------------------------------------------------------
# 6. The route is reachable with the REAL modules, not only with stubs
# ---------------------------------------------------------------------------

def test_the_composer_holds_every_products_own_predicate_and_builder():
    """The seam the stubs above replace, exercised unstubbed: the names in the
    table really resolve, on this host, to the shipped predicate and builder."""
    modules = launch_module._certified_fused_product_modules()
    for name, spec in PRODUCTS.items():
        coverage, builder, repair_paths = \
            launch_module._certified_fused_product_entry(modules, spec)
        module = modules[spec["module"]]
        assert coverage is getattr(module, spec["coverage"]), name
        assert builder is getattr(module, spec["builder"]), name
        assert repair_paths == tuple(getattr(module, "REPAIR_PATHS", ("split_field",)))


def test_a_real_grid_reaches_every_product_predicate():
    """WHAT STANDS BETWEEN THIS LAPTOP AND EACH OF THESE FUSED STEPS, itemised.

    Before this change the answer for all twenty-five was "the planner". After it,
    every one of them is CONSULTED on a real grid and reports its own reason — and
    for the products whose family this grid is, that reason is the backend clause
    and nothing structural, nothing about the seam, nothing about a slot.
    """
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(1.6, 3.0, 1.0), boundaries="metallic",
                symmetry=(Mirror("Y", 1),))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    plan = launch_module.plan_step(fields, PML(grid=grid, thickness=0.2),
                                   fuse=True, sources=())

    for name in PRODUCTS:
        reasons = plan.reasons[f"fused_pair_{name}"]
        assert reasons, name
    # 22 OF THE 25 GET AS FAR AS THEIR OWN BACKEND CLAUSE on this one grid (22 of
    # 23 until 2026-09-07, when the two cylindrical H->D products were routed and
    # are refused BEFORE their predicate by their own INSTALLABLE = False) — the
    # half that says "array module is 'numpy', not cupy", which is the LAST clause
    # each predicate reaches and therefore the statement that nothing structural,
    # nothing about the seam and nothing about a slot is refusing them here. The
    # denominators are the point: before this change the number was 0 of 23,
    # because none was consulted at all.
    reached = {name for name in PRODUCTS
               if any("not cupy" in reason
                      for reason in plan.reasons[f"fused_pair_{name}"])}
    # 22 -> 24 ON 2026-09-15: the two scratch-output off-diagonal welds were routed,
    # and both predicates report their backend clause on this grid beside their
    # structural refusals (a diagonal epsilon; the plain weld's curl half also
    # refuses the fold), so both land here and the unreached set does not move.
    assert len(reached) == 24, sorted(set(PRODUCTS) - reached)
    # The twenty-third is refused STRUCTURALLY and by name, which is the other
    # half of "consulted": a product whose family this grid is not says so. The two
    # cylindrical H->D products are refused BEFORE their predicate, by their own
    # INSTALLABLE = False, which the composer reports on every configuration.
    unreached = set(PRODUCTS) - reached
    uninstallable = {name for name in unreached
                     if "INSTALLABLE = False" in plan.reasons[f"fused_pair_{name}"][0]}
    assert uninstallable == {"cylindrical_real_fused_hd_pair",
                            "cylindrical_fused_hd_pair",
                            "complex_fused_hd_pair"}
    structural, = unreached - uninstallable
    assert structural == "nonlinear_fused_magnetic_pair"
    assert "no chi2/chi3 is installed" in \
        plan.reasons[f"fused_pair_{structural}"][0]


# ---------------------------------------------------------------------------
# 7. The reachability join: the board can now NAME every served product
# ---------------------------------------------------------------------------

def test_the_join_is_total_over_the_boards_products():
    """THE RECORDED JOIN GAP, closed, and measured in both directions.

    ``parity/meep_gpu/dispatch_reachability.PRODUCT_ARM_LABELS`` maps a board
    PRODUCT to the arm label ``TritonStepPlan.selected`` writes for it. Until
    2026-09-02 it held six rows: five installed products and ``fused_ade_state``,
    which no fusion board scores — while the board's own E->P product
    ``fused_ade_chain`` had no row at all and its ten seam-instances landed in
    "served by a product this file cannot name". Nothing measured either half.

    Now every product the board scores is in exactly one of two maps: joined to a
    label the composer really writes, or named in ``CERTIFIED_BUT_NOT_INSTALLED``
    with the reason no composer installs it. The "cannot name" bucket is what is
    left for a product neither map has heard of, and on this tree it is empty.
    """
    import dispatch_reachability as reach  # noqa: PLC0415

    board = set(_board_products())
    joined = set(reach.PRODUCT_ARM_LABELS) & board
    deferred = set(reach.CERTIFIED_BUT_NOT_INSTALLED)
    assert deferred <= board, sorted(deferred - board)
    assert joined | deferred == board, sorted(board - (joined | deferred))
    assert not (joined & deferred)
    # ...and the reasons are real sentences rather than placeholders.
    for product, reason in reach.CERTIFIED_BUT_NOT_INSTALLED.items():
        assert len(reason) > 40, product


def test_every_joined_label_is_one_the_shipped_composer_writes():
    """The join's own assertion, run rather than trusted.

    ``assert_the_label_map_covers_the_release`` raises ``SystemExit`` on any of
    four disagreements — a released arm with no product, a label ``arm_is_fused``
    does not recognise, a joined label no composer writes, and a written label no
    product claims. Calling it here is what makes those four checks part of the
    merge bar rather than of a board run nobody does on this host.
    """
    import dispatch_reachability as reach  # noqa: PLC0415

    summary = reach.assert_the_label_map_covers_the_release()
    written = set(summary["composer_installed_labels"])
    assert written == set(reach.PRODUCT_ARM_LABELS.values())
    assert {spec["label"] for spec in PRODUCTS.values()} <= written
    assert "fused ADE state" in written, "the inline fuse_ade label went unread"


def test_the_three_buckets_split_a_synthetic_board_exactly():
    """The counting itself, on instances this test writes.

    One instance per bucket, so a future edit that folded "certified, not
    installed" back into "cannot name" — the collapse this split exists to undo —
    fails here with the bucket it broke.
    """
    import dispatch_reachability as reach  # noqa: PLC0415

    # NOT `next(iter(PRODUCTS))` since the 2026-09-13 Phase B batch: the first
    # product's label (`fused pair B (complex)`) is released now, so on an empty
    # config it lands in `refused_by_envelope` rather than in the unreleased bucket
    # this row exists to exercise. Pick an installed product whose label is still
    # unreleased — the condition that bucket is defined on — instead.
    wired = next(name for name in PRODUCTS
                 if PRODUCTS[name]["label"] not in set(fastpath.RELEASED_FUSED_ARMS))
    deferred = next(iter(reach.CERTIFIED_BUT_NOT_INSTALLED))
    result = reach.served_in_dispatch("triton", [
        ("row-a", wired, {}),
        ("row-b", deferred, {}),
        ("row-c", "a_product_no_board_has_ever_scored", {}),
    ])
    assert result["served_counted"] == 3
    assert result["served_in_dispatch"] == 0
    assert result["served_by_an_arm_no_gate_released"] == 1
    assert result["not_installed_total"] == 1
    assert result["served_by_a_certified_product_the_composer_does_not_install"] == {
        deferred: 1}
    assert result["why_not_installed"][deferred] == \
        reach.CERTIFIED_BUT_NOT_INSTALLED[deferred]
    assert result["served_by_a_product_this_file_cannot_name"] == 1


def test_the_h_to_d_seam_row_is_last_because_placed_first_it_would_displace_a_product():
    """KEY ORDER IS LOAD-BEARING, pinned here as it is on the other two backends.

    ``_install_fused_pairs`` walks ``CERTIFIED_FUSED_PAIR_SEAMS.items()`` in dict
    order against the live ``selected``. Placed before ``step_B``, the H->D seam
    would be offered ``update_H`` ahead of the released B->H pair and would displace
    it on every row that admits both. The row is appended LAST for that reason, and
    this test states it so a later edit that re-orders the table fails here rather
    than in a board cut.
    """
    keys = list(launch_module.CERTIFIED_FUSED_PAIR_SEAMS)
    assert keys[-1] == "update_H", (
        f"the H->D seam row must be last in CERTIFIED_FUSED_PAIR_SEAMS; order is {keys}")
    assert keys.index("step_B") < keys.index("update_H")
    assert keys.index("step_D") < keys.index("update_H")


# ---------------------------------------------------------------------------
# THE END-EDGE GUARD, 2026-09-05 -- both directions
# ---------------------------------------------------------------------------
#
# Measured on the Metal composer on 2026-09-04: with the ``update_H`` seam row in the
# table, the later half of ``_neighbouring_seam_claimant`` ALONE refused the released
# B->H pair on every row an admitting H->D product reached -- a stand-in with no
# absorb declaration, which can never install, still cost the incumbent its seam.
# The guard (R1) asks the question only of a span whose two END slots both have
# degree >= 2 in the seam table. This composer has the same three rows; no H->D
# product is in its table yet, so every test below drives a stand-in through the
# real seam loop and asserts who INSTALLS beside who is withheld.

STAND_IN = "h_to_d_stand_in"
STAND_IN_LABEL = "fused pair H->D (stand-in)"
STAND_IN_PRODUCT = {"curl_slot": "update_H", "module": STAND_IN,
                    "coverage": f"{STAND_IN}_coverage", "builder": f"plan_{STAND_IN}",
                    "label": STAND_IN_LABEL}


def _stub_products(monkeypatch, verdicts, extra=None):
    """Like :func:`_stub_product`, for several products at once, plus optional extra
    (module name -> module) entries such as a stand-in the table does not carry."""
    modules = {}
    for name, verdict in verdicts.items():
        product = PRODUCTS[name]
        modules[product["module"]] = types.SimpleNamespace(**{
            product["coverage"]: (lambda *a, verdict=verdict, **k: verdict),
            product["builder"]: lambda *a, **k: "<pair>"})
    modules.update(extra or {})
    monkeypatch.setattr(launch_module, "_certified_fused_product_modules",
                        lambda: modules)
    return modules


def _stand_in_module(verdict=ADMITTED, installable=None):
    module = types.SimpleNamespace(**{
        f"{STAND_IN}_coverage": lambda *a, **k: verdict,
        f"plan_{STAND_IN}": lambda *a, **k: "<stand-in>"})
    if installable is not None:
        module.INSTALLABLE = installable
        module.INSTALLABLE_REASON = "a stand-in declaring itself uninstallable"
    return module


def test_slot_degrees_are_read_off_the_table_and_name_the_end_edges():
    """THE GUARD IS DERIVED, NOT SPELLED. This is what fails the day a fifth seam row
    is priced without thinking about position."""
    degrees = launch_module._slot_degrees(SEAMS)
    assert degrees == {"step_B": 1, "update_H": 2, "step_D": 2, "update_E": 1}, degrees
    spans = {curl: (curl, update) for curl, (update, _p) in SEAMS.items()}
    end_edges = {curl for curl, span in spans.items()
                 if launch_module._is_end_edge_span(span, SEAMS)}
    assert end_edges == {"step_B", "step_D"}, end_edges
    assert not launch_module._is_end_edge_span(spans["update_H"], SEAMS)
    assert launch_module._is_end_edge_span(
        ("step_B", "update_H", "step_D", "update_E"), SEAMS)


def _both_complex_pairs(monkeypatch, stand_in=None, absorb=None):
    """The has_bloch configuration with both complex pairs admitting, plus a stand-in."""
    configuration, _live = CONFIGURATIONS["complex_fused_magnetic_pair"]
    install_stubs(monkeypatch, admit=set(ARMS["complex_fused_magnetic_pair"])
                  | set(ARMS["complex_fused_electric_pair"]))
    extra = {STAND_IN: stand_in} if stand_in is not None else {}
    modules = _stub_products(monkeypatch, {"complex_fused_magnetic_pair": ADMITTED,
                                           "complex_fused_electric_pair": ADMITTED},
                             extra)
    if stand_in is not None:
        monkeypatch.setitem(PRODUCTS, STAND_IN, STAND_IN_PRODUCT)
    if absorb is not None:
        monkeypatch.setitem(ARMS, STAND_IN, absorb)
    fields = _fields_for(configuration)
    return modules, fields, launch_module.plan_step(fields, object(), fuse=True,
                                                    sources=())


def _owners(plan):
    return {slot: plan.selected.get(slot)
            for slot in ("step_B", "update_H", "step_D", "update_E")}


def test_an_end_edge_span_is_never_asked_even_when_an_interior_claimant_admits(
        monkeypatch):
    """Both certified pairs on this configuration sit on end edges; neither yields to
    an admitting, installable H->D stand-in. The armed control is the stand-in's own
    span, which is interior and DOES yield -- to the D->E product, via the later half."""
    modules, fields, _plan = _both_complex_pairs(
        monkeypatch, stand_in=_stand_in_module(), absorb=("complex", "complex PML"))
    for name, curl, update in (("complex_fused_magnetic_pair", "step_B", "update_H"),
                               ("complex_fused_electric_pair", "step_D", "update_E")):
        assert launch_module._neighbouring_seam_claimant(
            modules, fields, object(), (), name, curl, update, (curl, update)) is None, name
    asked = launch_module._neighbouring_seam_claimant(
        modules, fields, object(), (), STAND_IN, "update_H", "step_D",
        ("update_H", "step_D"))
    assert asked is not None and asked[0] == "complex_fused_electric_pair", asked
    assert "slot arbitration" in asked[1]


def test_a_registered_admitting_uninstallable_h_to_d_stand_in_does_not_cost_the_b_to_h_product_its_seam(
        monkeypatch):
    """ARRANGEMENT F OF THE TIE MEASUREMENT, ON THIS COMPOSER, FROM BOTH SIDES.

    Shipped (no stand-in): the B->H product holds ``step_B``/``update_H`` and the D->E
    product ``step_D``/``update_E``. With an admitting stand-in that has no absorb row:
    every owner unchanged, the stand-in refused on its missing row. With a row: every
    owner unchanged, the stand-in refused by ``_pair_may_absorb`` naming the B->H
    product that installed first. With a REFUSING stand-in (the negative control):
    unchanged again, so the test cannot pass by the stand-in being invisible.
    """
    _m, _f, shipped = _both_complex_pairs(monkeypatch)
    owners = _owners(shipped)
    b_label = PRODUCTS["complex_fused_magnetic_pair"]["label"]
    d_label = PRODUCTS["complex_fused_electric_pair"]["label"]
    assert owners == {"step_B": b_label, "update_H": b_label,
                      "step_D": d_label, "update_E": d_label}, owners

    _m, _f, plan = _both_complex_pairs(monkeypatch, stand_in=_stand_in_module())
    assert _owners(plan) == owners, _owners(plan)
    assert "has no absorb declaration" in plan.reasons[f"fused_pair_{STAND_IN}"][0]

    _m, _f, plan = _both_complex_pairs(monkeypatch, stand_in=_stand_in_module(),
                                       absorb=("complex", "complex PML"))
    assert _owners(plan) == owners, _owners(plan)
    assert plan.reasons[f"fused_pair_{STAND_IN}"][0].startswith(
        f"update_H was selected by the {b_label!r} arm"), plan.reasons[f"fused_pair_{STAND_IN}"]

    _m, _f, plan = _both_complex_pairs(monkeypatch, stand_in=_stand_in_module(REFUSED),
                                       absorb=("complex", "complex PML"))
    assert _owners(plan) == owners, _owners(plan)
    assert plan.reasons[f"fused_pair_{STAND_IN}"] == REFUSED.reasons


def test_the_interior_h_to_d_span_still_yields_to_a_d_to_e_claimant(monkeypatch):
    """The direction the rule exists for is unchanged: with ONLY the D->E product
    admitting (no B->H product to install first), an installable stand-in on the
    interior seam is refused by the later half naming the D->E claimant, and the D->E
    product holds its two slots."""
    configuration, _live = CONFIGURATIONS["complex_fused_electric_pair"]
    install_stubs(monkeypatch, admit=set(ARMS["complex_fused_electric_pair"])
                  | set(ARMS["complex_fused_magnetic_pair"]))
    _stub_products(monkeypatch, {"complex_fused_electric_pair": ADMITTED},
                   {STAND_IN: _stand_in_module()})
    monkeypatch.setitem(PRODUCTS, STAND_IN, STAND_IN_PRODUCT)
    monkeypatch.setitem(ARMS, STAND_IN, ("complex", "complex PML"))
    plan = launch_module.plan_step(_fields_for(configuration), object(), fuse=True,
                                   sources=())
    d_label = PRODUCTS["complex_fused_electric_pair"]["label"]
    assert plan.selected["step_D"] == plan.selected["update_E"] == d_label
    assert plan.selected["update_H"] == "complex", plan.selected
    # THE OUTCOME IS THE CLAIM; THE MECHANISM IS WHICHEVER FIRES FIRST. On this
    # fixture the D->E product installs before the stand-in (the composer walks the
    # product table in order), so by the time the stand-in is asked, ``step_D`` is
    # already held and ``_pair_may_absorb`` refuses it on the arm label — the
    # neighbour rule is never consulted. Asserting the neighbour rule's wording here
    # pinned an accident of install order rather than the behaviour, and went red the
    # moment R1 landed. What must hold is that the interior span does not take a slot
    # from an installed product, by whichever refusal reaches it first.
    reason = plan.reasons[f"fused_pair_{STAND_IN}"][0]
    assert STAND_IN not in plan.selected.values(), plan.selected
    assert ("may not substitute it" in reason
            or "admits this run's step_D/update_E seam" in reason), reason


def test_the_interior_span_is_the_only_one_the_neighbour_rule_yields(monkeypatch):
    """R1 AT FUNCTION LEVEL, which is where the rule can be read without an install
    order in the way: a span yields only when BOTH its slots are interior to the
    driver's slot path. Over ``step_B -- update_H -- step_D -- update_E`` the maximum
    matching is the two END edges, so an end-edge span must never yield and the
    interior one must. Degrees are computed from the seams table, never spelled."""
    seams = launch_module.CERTIFIED_FUSED_PAIR_SEAMS
    assert launch_module._slot_degrees(seams) == {
        "step_B": 1, "update_H": 2, "step_D": 2, "update_E": 1}
    assert launch_module._is_end_edge_span(("step_B", "update_H"), seams) is True
    assert launch_module._is_end_edge_span(("step_D", "update_E"), seams) is True
    assert launch_module._is_end_edge_span(("update_H", "step_D"), seams) is False
