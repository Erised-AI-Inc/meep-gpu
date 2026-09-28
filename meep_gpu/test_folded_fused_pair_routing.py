"""The folded grid's two fused pairs, ROUTED — one file, both tracks, 2026-08-27.

WHAT CHANGED. Until this round ``fuse=True`` on a folded grid was answered by a
BLANKET refusal on both composers. Triton returned "folded composition keeps the
source-adjacent mirror-fill seam explicit; cross-sub-step fusion for this product is
not implemented" for both pairs (``triton_kernels/launch.py``), and Metal returned
"<family> has no absorb declaration" for every folded weld
(``metal_kernels/launch.py``). Both sentences were true of the ORDINARY pair — its
kernel carries no mirror fill — and false of ``folded_fused_magnetic_pair`` and
``folded_fused_pair``, which carry the near fill, the wall clear and the far ghost
image inline (their ``REPLACES``). The refusal was a fact about the PLANNER, not
about those products, and it is what kept the largest measured fusion family off
both boards: 54 B->H seam-instances and 4 D->E, the top row of
``results/fusion_matrix_metal_2026-08-27_recut``.

WHAT IS ASSERTED HERE, in the order the composers reach it:

1. the refusals that were NOT implemented are still there, byte-for-byte
   (cylindrical, specialized-family, folded dispersive D/E) — a blanket refusal
   replaced by a blanket admission is the failure mode this change has;
2. the route is REACHABLE — Triton's ``_folded_fused_pair_entries`` really hands
   ``plan_step`` the two families' own predicates and builders, and on a real folded
   NumPy grid the only thing left refusing is the host backend clause;
3. the product lands in BOTH slots and both report the pair's passes, which is the
   fail-closed half: a plan owning ``step_B`` alone would tell a reader — and the
   residency verdict — that ``update_H`` ran on the array path;
4. the arm-absorb clause still stands in front of it;
5. BOTH TRITON families carry the deposit repair as of 2026-08-30, as both METAL
   families have since 2026-08-28, and both directions of each are measured here.

WHY (5) WAS A BOUNDARY, AND WHAT RETIRED IT. On a folded row the seam also carries
``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*``, and they run AFTER the
injection (driver.py:3293-3295, :3308-3311). The near fill writes cell 0 from cell 2
(stepping.py:1450-1451), so a deposit landing on an imaged cell changes the ghost the
driver then recomputes -- while ``deposit_repair.apply`` only ever wrote the deposit
INDEX and never its mirror image. Carrying the fill inside the kernel does not help:
the kernel fills BEFORE the injection, because there is no injection inside a launch.

``deposit_repair.repair_cells`` closed that on 2026-08-28 by saving and restoring the
CLOSURE of the cells those two fills image each deposit point into, and the two Metal
folded families flipped in the same edit. THE TRITON TWINS DID NOT, and the reason
given was: the closure is shared, but every Triton fused product's arithmetic is
certified on a device that round could not run, and a flag flipped ahead of its gate
is a claim no artifact stands behind.

THAT REASON WAS DISCHARGED ON 2026-08-30 THE ONLY WAY IT COULD BE -- by running the
gates, not by deciding the flag was fine. The three Triton families flipped and their
device gates were re-run on the GPU host against the flipped bytes; the weld tests that
bind each module's identity to the run that measured it are what would have caught a
flip that outran its gate, and they are green against the new records rather than
against the old ones.

WHAT MAKES THE CLEAN SEAM SAFE, AND THE CARRIED ONE TOO. The driver runs the three
fill/clear passes unconditionally -- driver.py:3293-3295 and :3308-3311 are plain calls
with no ``fast.dispatch`` consult -- so a routed pair has them run a second time on the
host after its launch. All three are idempotent images of cells the pair did not
change, and on a seam that carries a deposit the repair runs AFTER them, which is the
one slot where the injected field is final.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

_PARITY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       os.pardir, "parity", "meep_gpu"))
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

from meep_gpu import deposit_repair  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import coverage as coverage_module  # noqa: E402
from meep_gpu.triton_kernels import launch as launch_module  # noqa: E402
from meep_gpu.triton_kernels import folded_fused_magnetic_pair as triton_b  # noqa: E402
from meep_gpu.triton_kernels import folded_fused_pair as triton_d  # noqa: E402

TRITON_DIR = pathlib.Path(launch_module.__file__).parent
METAL_DIR = TRITON_DIR.parent / "metal_kernels"

ADMITTED = coverage_module.Coverage(True, ())
REFUSED = coverage_module.Coverage(False, ("outside this routing test",))


class Source:
    """The one attribute ``deposit_repair`` reads to place a source in a seam."""

    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def folded_numpy(cell=(1.6, 3.0, 1.0), boundaries="metallic",
                 mirrors=(("Y", 1),), thickness=0.2):
    """A real Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=thickness)


def install_folded_arm_stubs(monkeypatch):
    """Give the folded ARMS both curl slots and both constitutive slots.

    The absorb clause reads ``selected``, so a test of the fusion block that could
    not make the folded arms win would exercise nothing but the refusal. These are
    the same stubs ``test_triton_symmetry_composition`` uses for the unfused route.
    """
    monkeypatch.setattr(launch_module, "pml_curl_coverage", lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "conductive_pml_curl_coverage",
                        lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "plain_curl_coverage", lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "constitutive_coverage", lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage",
                        lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "folded_composition_curl_coverage",
                        lambda *a: ADMITTED, raising=False)
    monkeypatch.setattr(launch_module, "plan_folded_pml_curl",
                        lambda fields, pml, name, block=None: f"folded:{name}",
                        raising=False)
    monkeypatch.setattr(launch_module, "folded_constitutive_coverage",
                        lambda *a: ADMITTED, raising=False)
    monkeypatch.setattr(launch_module, "plan_folded_constitutive",
                        lambda fields, pml, side, block=None: f"folded:{side}",
                        raising=False)
    monkeypatch.setattr(launch_module, "mirror_ghost_fill_coverage",
                        lambda fields, family: ADMITTED, raising=False)
    monkeypatch.setattr(launch_module, "plan_mirror_ghost_fill",
                        lambda fields, family, block=None: f"fill:{family}",
                        raising=False)


def install_pair_stubs(monkeypatch, *, verdict=ADMITTED, plan_b="<B pair>",
                       plan_d="<D pair>"):
    """Stand in for the two family modules, which refuse on a NumPy host.

    The families' own predicates end in an "array module is 'numpy', not cupy"
    clause, so on this machine the composer can be driven no further than that
    refusal with real objects — which is what
    :func:`test_a_real_folded_grid_reaches_the_pair_predicate_and_only_the_host_refuses`
    measures. Everything past it is measured here, through the same seam the
    composer uses rather than by calling the installer directly.
    """
    monkeypatch.setattr(
        launch_module, "_folded_fused_pair_entries",
        lambda: {"B": (lambda *a, **k: verdict, lambda *a, **k: plan_b),
                 "D": (lambda *a, **k: verdict, lambda *a, **k: plan_d)})


# ---------------------------------------------------------------------------
# 1. The refusals that were NOT implemented are kept, verbatim
# ---------------------------------------------------------------------------

CYLINDRICAL_REFUSAL = (
    "cylindrical m=0 composition keeps the radial prefix and axis rules "
    "in separate curl plans; cross-sub-step fusion is not implemented")
SPECIALIZED_REFUSAL = (
    "a specialized family owns this grid's sub-steps; cross-sub-step "
    "fusion for it is not implemented")
FOLDED_BLANKET_REFUSAL = (
    "folded composition keeps the source-adjacent mirror-fill seam "
    "explicit; cross-sub-step fusion for this product is not implemented")


def _flat(text: str) -> str:
    """Whitespace-normalised, with Python's implicit string concatenation closed up.

    Each of these refusals is written across two source lines as two adjacent
    literals, so the shipped file holds ``...axis rules " "in separate...`` where the
    emitted reason holds one space. Comparing on the raw text would make this test
    pass on a refusal that no longer says what it used to.
    """
    return " ".join(text.split()).replace('" "', "")


def test_the_two_kinds_that_were_not_implemented_keep_their_refusal_text():
    """A routing change may retire ONE refusal, and it has to leave the rest alone.

    Read off the shipped file rather than off a plan, because these are the exact
    strings the frozen gate artifacts under ``parity/meep_gpu/results`` recorded.
    """
    flat = _flat((TRITON_DIR / "launch.py").read_text(encoding="utf-8"))
    for text in (CYLINDRICAL_REFUSAL, SPECIALIZED_REFUSAL):
        assert _flat(text) in flat, text
    for text in ("cylindrical dispersive D/E fusion is not implemented",
                 "specialized-family dispersive D/E fusion is not implemented",
                 "folded dispersive D/E fusion is not implemented"):
        assert _flat(text) in flat, text
    # ...and the one that WAS implemented is gone rather than left dangling.
    assert _flat(FOLDED_BLANKET_REFUSAL) not in flat


def test_a_cylindrical_grid_still_gets_the_blanket_refusal(monkeypatch):
    """The kind next door, driven through the composer rather than read."""
    from types import SimpleNamespace

    cylindrical = SimpleNamespace(
        grid=SimpleNamespace(cylindrical=True, has_symmetry=lambda: False),
        polarizations=())
    plan = launch_module.plan_step(cylindrical, object(), fuse=True, sources=())
    for pair in ("B", "D"):
        assert any("cross-sub-step fusion is not implemented" in reason
                   for reason in plan.reasons[f"fused_pair_{pair}"]), pair


# ---------------------------------------------------------------------------
# 2. The route is reachable
# ---------------------------------------------------------------------------

def test_the_composer_holds_the_two_families_own_predicates_and_builders():
    entries = launch_module._folded_fused_pair_entries()
    assert entries["B"] == (triton_b.folded_fused_magnetic_pair_coverage,
                            triton_b.plan_folded_fused_magnetic_pair)
    assert entries["D"] == (triton_d.folded_fused_pair_coverage,
                            triton_d.plan_folded_fused_pair)
    assert launch_module.FOLDED_FUSED_PAIRS["B"]["curl"] == "step_B"
    assert launch_module.FOLDED_FUSED_PAIRS["B"]["update"] == "update_H"
    assert launch_module.FOLDED_FUSED_PAIRS["D"]["curl"] == "step_D"
    assert launch_module.FOLDED_FUSED_PAIRS["D"]["update"] == "update_E"


def test_a_real_folded_grid_reaches_the_pair_predicate_and_only_the_host_refuses():
    """WHAT STANDS BETWEEN THIS LAPTOP AND A FUSED FOLDED STEP, itemised.

    Before this change the answer was "the planner". After it, the only reasons the
    composer reports on a real folded grid are the families' own backend clause —
    nothing structural, nothing about the seam, nothing about a slot.
    """
    fields, pml = folded_numpy()
    plan = launch_module.plan_step(fields, pml, fuse=True, sources=())
    for pair in ("B", "D"):
        reasons = plan.reasons[f"fused_pair_{pair}"]
        assert reasons, pair
        assert all("not cupy" in reason for reason in reasons), (pair, reasons)


# ---------------------------------------------------------------------------
# 3. Both slots, and both reported
# ---------------------------------------------------------------------------

def test_a_clean_folded_seam_fills_both_slots_with_the_cheap_sentinel(monkeypatch):
    """THE FAIL-CLOSED HALF OF THE TWO-SLOT CONTRACT.

    ``FastPath.dispatch`` contracts that True means the whole sub-step ran, so a
    pair that owns ``step_B`` and ``update_H`` must appear in both — otherwise a
    reader, and the regression tests that read ``plan.replaces``, are told the
    constitutive update ran on the array path when the launch performed it.
    """
    install_folded_arm_stubs(monkeypatch)
    install_pair_stubs(monkeypatch)
    fields, pml = folded_numpy()

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=())

    assert plan.plans["step_B"] == "<B pair>"
    assert plan.plans["step_D"] == "<D pair>"
    for curl, update, label in (("step_B", "update_H", "fused pair B (folded)"),
                                ("step_D", "update_E", "fused pair D (folded)")):
        sentinel = plan.plans[update]
        assert isinstance(sentinel, launch_module.NoopPlan)
        assert sentinel.absorbed_by is plan.plans[curl]
        assert plan.selected[curl] == plan.selected[update] == label
        assert curl in plan.replaces and update in plan.replaces
    # The mirror-fill slots are untouched: the driver runs those passes on the host
    # unconditionally, and the pair carrying them inline does not change that.
    assert plan.plans["fill_B"] == "fill:B" and plan.plans["fill_D"] == "fill:D"


def test_a_refusing_builder_leaves_both_slots_on_their_separate_folded_plans(
        monkeypatch):
    """A refused fusion is not a refused step."""
    install_folded_arm_stubs(monkeypatch)
    install_pair_stubs(monkeypatch, plan_b=None, plan_d=None)
    fields, pml = folded_numpy()

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=())

    assert plan.plans["step_B"] == "folded:step_B"
    assert plan.plans["update_H"] == "folded:H"
    assert plan.plans["step_D"] == "folded:step_D"
    assert plan.plans["update_E"] == "folded:E"
    for pair, noun in (("B", "folded B"), ("D", "folded D")):
        assert plan.reasons[f"fused_pair_{pair}"] == (
            f"the {noun} pair was refused",)


def test_a_raising_predicate_is_a_refusal_and_never_an_exception(monkeypatch):
    """``plan_step`` never raises; an opt-in optimisation that crashes it is worse
    than one that refuses it."""
    install_folded_arm_stubs(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("predicate exploded")

    monkeypatch.setattr(
        launch_module, "_folded_fused_pair_entries",
        lambda: {"B": (boom, boom), "D": (boom, boom)})
    fields, pml = folded_numpy()

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=())

    for pair in ("B", "D"):
        assert any("raised" in reason
                   for reason in plan.reasons[f"fused_pair_{pair}"]), pair
    assert plan.plans["step_B"] == "folded:step_B"


# ---------------------------------------------------------------------------
# 4. The absorb clause still stands in front of the block
# ---------------------------------------------------------------------------

def test_the_folded_pair_may_not_claim_a_slot_the_arm_table_left_unselected(
        monkeypatch):
    """Clause (b)/(c) survives fusion here exactly as it does for the ordinary pair.

    The constitutive predicate is made to refuse, so ``update_H``/``update_E`` are
    UNSELECTED — the array path, which is always correct — and the pair must not
    claim them anyway.
    """
    install_folded_arm_stubs(monkeypatch)
    monkeypatch.setattr(launch_module, "folded_constitutive_coverage",
                        lambda *a: REFUSED, raising=False)
    install_pair_stubs(monkeypatch)
    fields, pml = folded_numpy()

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=())

    for pair, update in (("B", "update_H"), ("D", "update_E")):
        assert update not in plan.plans
        assert any("not selected by any arm" in reason
                   for reason in plan.reasons[f"fused_pair_{pair}"]), pair
    assert plan.plans["step_B"] == "folded:step_B"


def test_the_folded_pair_may_not_substitute_a_product_another_arm_won(monkeypatch):
    """A folded COMPLEX curl is a different numerical product, and the arm labels
    are what hold the two apart."""
    install_folded_arm_stubs(monkeypatch)
    monkeypatch.setattr(launch_module, "folded_composition_curl_coverage",
                        lambda *a: REFUSED, raising=False)
    monkeypatch.setattr(launch_module, "plain_curl_coverage",
                        lambda *a: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_plain_curl",
                        lambda fields, pml, name, block=None: f"plain:{name}",
                        raising=False)
    install_pair_stubs(monkeypatch)
    fields, pml = folded_numpy()

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=())

    assert plan.selected["step_B"] == "no-PML"
    assert plan.plans["step_B"] == "plain:step_B"
    assert any("implements" in reason and "folded PML" in reason
               for reason in plan.reasons["fused_pair_B"]), plan.reasons


def test_the_arms_the_two_folded_pairs_absorb_are_the_arms_their_predicates_are():
    """THE ROW IS READ, NOT GUESSED — asserted against the family predicates' own
    bodies, which is the only thing that makes the mapping a measurement."""
    import inspect

    assert launch_module.FOLDED_FUSED_PAIR_ARMS == ("folded PML", "folded")
    b_source = inspect.getsource(triton_b.folded_fused_magnetic_pair_coverage)
    assert "folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP)" in b_source
    assert "folded_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)" in b_source
    d_source = inspect.getsource(triton_d.folded_fused_pair_coverage)
    assert "folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP)" in d_source
    assert "folded_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)" in d_source


# ---------------------------------------------------------------------------
# 5. The fold boundary: routed for fusion, NOT for the deposit repair
# ---------------------------------------------------------------------------

def _folded_electric_deposit(fields):
    """A REAL electric source that publishes the index it writes.

    The ``Source`` stub above carries a ``field_type`` and nothing else, which is
    enough to place a source in a seam and deliberately NOT enough to carry one: the
    repair refuses it by name for publishing no deposit index. A test about the
    bracket needs a source the repair can actually save, so this builds the engine's
    own ``VolumeSource`` — the same shape the Metal half of this file uses — and
    refuses to hand back one that deposits nothing.
    """
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    source = VolumeSource(grid=fields.grid, component="Ez",
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert source._n_source_points, "the case deposits nothing and measures nothing"
    assert deposit_repair._deposit_index(source) is not None, (
        "the case cannot exercise the repair: this source publishes no deposit index")
    return source


@pytest.mark.parametrize("module", [triton_b, triton_d])
def test_both_triton_folded_families_declare_the_deposit_repair_and_gate_it_on_the_flag(
        module):
    """THE FOLD BOUNDARY RETIRED ON THE TRITON SIDE TOO, 2026-08-30.

    This test used to assert the opposite (``is False``), and it is INVERTED rather
    than relaxed: the same two facts are still pinned, with the value flipped and
    the wiring clause kept. What changed is not the assertion's strength but the
    tree — ``deposit_repair.repair_cells`` closes the saved and restored set over
    the cells the post-injection fills image a deposit into, which is the premise
    the ``False`` rested on.

    THE SECOND ASSERTION IS THE ONE THAT MATTERS. A ``True`` flag that is never
    passed to the seam clause is a product claiming a carry it does not perform,
    which is strictly worse than the refusal it replaced.
    """
    assert module.CARRIES_DEPOSIT_REPAIR is True
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    assert "CARRIES_DEPOSIT_REPAIR = True" in source
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source, (
        "the flag gates nothing unless the seam clause is passed it")


def test_the_flag_is_what_decides_an_in_seam_deposit_on_each_folded_seam():
    """BOTH DIRECTIONS, on the real predicates, with the SHIPPED flag.

    The pair that owns the seam the source is injected into no longer refuses it
    for BEING in the seam — but it does not admit it blindly either. What refuses
    the stub below is that it publishes no deposit index, which is
    ``deposit_repair``'s own fail-closed clause and not the retired blanket one.
    The pair that does NOT own the seam is untouched, exactly as before, which is
    what makes this a seam fact rather than a plan-wide one.
    """
    fields, pml = folded_numpy()

    magnetic = triton_b.folded_fused_magnetic_pair_coverage(
        fields, pml, (Source("D"), Source("B")))
    assert not any("is magnetic" in reason for reason in magnetic.reasons), (
        magnetic.reasons)
    assert any("does not publish the index it writes" in reason
               and "source 1" in reason
               for reason in magnetic.reasons), magnetic.reasons

    clean = triton_d.folded_fused_pair_coverage(fields, pml, (Source("B"),))
    assert not any("source" in reason for reason in clean.reasons), clean.reasons

    electric = triton_d.folded_fused_pair_coverage(fields, pml, (Source("D"),))
    assert not any("is electric" in reason for reason in electric.reasons), (
        electric.reasons)
    assert any("does not publish the index it writes" in reason
               for reason in electric.reasons), electric.reasons


@pytest.mark.parametrize("module,other,pair,kind", [
    (triton_b, triton_d, "B", "B"),
    (triton_d, triton_b, "D", "D"),
])
def test_holding_a_triton_folded_flag_False_puts_the_seam_refusal_straight_back(
        monkeypatch, module, other, pair, kind):
    """THE OTHER DIRECTION, on the same deposit and through the SHARED helper.

    The refusal prose the boundary shipped with is still what the False branch
    returns, so the admission above is the flag's doing and not a clause that
    quietly went away. Asserted at ``deposit_repair.seam_source_reasons`` — the one
    place the decision lives — with the site's own refusal text supplied, because
    that is the seam every one of the four shipped sites routes through.
    """
    fields, _pml = folded_numpy()
    source = Source(kind)

    monkeypatch.setattr(module, "CARRIES_DEPOSIT_REPAIR", False)
    held = deposit_repair.seam_source_reasons(
        fields, (source,), pair, undeclared="no source list was declared",
        refusal=lambda index, s: f"source {index} is injected INSIDE this seam",
        carries_repair=module.CARRIES_DEPOSIT_REPAIR)
    assert list(held) == ["source 0 is injected INSIDE this seam"], held

    monkeypatch.setattr(module, "CARRIES_DEPOSIT_REPAIR", True)
    carried = deposit_repair.seam_source_reasons(
        fields, (source,), pair, undeclared="no source list was declared",
        refusal=lambda index, s: f"source {index} is injected INSIDE this seam",
        carries_repair=module.CARRIES_DEPOSIT_REPAIR)
    assert not any("injected INSIDE this seam" in reason for reason in carried), (
        carried)


def test_a_folded_deposit_IS_bracketed_now_and_the_clean_seam_still_is_not(
        monkeypatch):
    """THE CLAIM THE RETIREMENT RESTS ON, asserted at the composer's output.

    ``_install_fused_pair`` brackets a pair with the repair whenever the seam holds
    a deposit. Until 2026-08-30 a folded pair could never get there with one,
    because the repair wrote only the deposit index while the driver's
    post-injection fill images that cell somewhere else. That is closed, so the
    seam carrying the deposit gets the two repair plans and the OTHER seam keeps
    the bare fused plan — a case whose source touched both seams could not tell
    "this seam is bracketed" from "everything is".
    """
    install_folded_arm_stubs(monkeypatch)
    install_pair_stubs(monkeypatch)
    fields, pml = folded_numpy()
    deposit = _folded_electric_deposit(fields)

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=(deposit,))

    assert isinstance(plan.plans["step_D"], deposit_repair.LeadingRepairPlan)
    assert isinstance(plan.plans["update_E"], deposit_repair.TrailingRepairPlan)
    # The B seam carries no deposit, so it keeps the bare pair and its sentinel.
    assert plan.plans["step_B"] == "<B pair>"
    assert not isinstance(plan.plans["update_H"],
                          (deposit_repair.LeadingRepairPlan,
                           deposit_repair.TrailingRepairPlan))


def test_the_repair_helper_is_not_reached_while_a_folded_flag_is_held_False(
        monkeypatch):
    """The same composer, the same deposit, the flag held down: no bracket at all.

    This is the test the retirement replaced, kept as the OTHER direction of it
    rather than deleted. With the REAL family predicates running, a False flag
    still refuses the seam before ``_install_fused_pair`` can bracket it.
    """
    install_folded_arm_stubs(monkeypatch)
    monkeypatch.setattr(triton_d, "CARRIES_DEPOSIT_REPAIR", False)
    monkeypatch.setattr(triton_b, "CARRIES_DEPOSIT_REPAIR", False)
    fields, pml = folded_numpy()
    deposit = _folded_electric_deposit(fields)

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=(deposit,))

    for slot in ("step_B", "update_H", "step_D", "update_E"):
        assert not isinstance(plan.plans.get(slot),
                              (deposit_repair.LeadingRepairPlan,
                               deposit_repair.TrailingRepairPlan)), slot
    assert plan.plans["step_B"] == "folded:step_B"
    assert plan.plans["update_E"] == "folded:E"


def test_the_module_docstring_states_the_carry_rather_than_assuming_it():
    """A boundary nobody wrote down is a boundary the next round routes around.

    The prose moved with the flag on 2026-08-30 — the blocks at ``FOLDED_FUSED_PAIRS``
    and ``_install_folded_fused_pairs`` said the fold boundary rested on the flag
    being ``False``, which stopped being true. What is pinned here is that the file
    still NAMES the hazard and the mechanism, and that it no longer asserts the
    retired premise as a live guarantee.
    """
    source = (TRITON_DIR / "launch.py").read_text(encoding="utf-8")
    joined = " ".join(part for part in source.replace("#:", " ").split())
    assert "MIRROR IMAGE" in joined
    assert "CARRIES_DEPOSIT_REPAIR" in joined
    assert "repair_cells" in joined, (
        "the closure that retired the boundary must be named where the boundary was")
    assert "CARRIES_DEPOSIT_REPAIR = False on both families guarantees" not in joined, (
        "launch.py still states the retired premise as a live guarantee")


# ---------------------------------------------------------------------------
# 6. The Metal mirror — the same two families, the same boundary
# ---------------------------------------------------------------------------

metal = pytest.importorskip("meep_gpu.metal_kernels.launch")


def _metal():
    import metal_composition_matrix as matrix  # noqa: PLC0415

    matrix.prepare_environment()
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        complex_fields, cylindrical_complex, folded_complex, special_kz)

    probes = {"complex_probe": complex_fields.load_expansion_probe(),
              "beta_probe": special_kz.load_expansion_probe(),
              "folded_complex_probe": folded_complex.load_expansion_probe(),
              "cylindrical_complex_probe":
                  cylindrical_complex.load_expansion_probe()}
    return matrix, probes


def _metal_plan(sources=(), fuse=True):
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    matrix, probes = _metal()
    fields, pml = matrix.folded()
    return fields, metal.plan_step(fields, pml, residency=device.Residency(),
                                   sources=sources, fuse=fuse, **probes)


def test_the_metal_folded_arms_win_all_four_slots_before_any_fusion():
    """The precondition, so nothing below can pass vacuously: the absorb row this
    change adds names the arms that actually win."""
    _fields, plan = _metal_plan(fuse=False)
    for slot in ("step_B", "update_H", "step_D", "update_E"):
        assert plan.selected[slot] == "folded", slot
    assert metal.FUSED_PAIR_ARMS["folded_fused_magnetic_pair"] == ("folded", "folded")
    assert metal.FUSED_PAIR_ARMS["folded_fused_pair"] == ("folded", "folded")


def test_the_metal_folded_pairs_take_both_slots_of_both_seams():
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        folded_fused_magnetic_pair as metal_b, folded_fused_pair as metal_d)
    from meep_gpu.triton_kernels.launch import NoopPlan  # noqa: PLC0415

    _fields, plan = _metal_plan()

    assert isinstance(plan.plans["step_B"], metal_b.MetalFoldedFusedMagneticPairPlan)
    assert isinstance(plan.plans["step_D"], metal_d.MetalFoldedFusedPairPlan)
    for curl, update, family in (("step_B", "update_H", metal_b),
                                 ("step_D", "update_E", metal_d)):
        assert isinstance(plan.plans[update], NoopPlan)
        assert plan.plans[update].absorbed_by is plan.plans[curl]
        assert plan.selected[curl] == plan.selected[update]
        assert curl in plan.replaces and update in plan.replaces
        for slot in (curl, update):
            # Read through ``declaring_plan``: the sentinel declares nothing of its
            # own, and a bare read would report the pair's wall clear and ghost
            # fills as array-path work the kernel actually performed.
            assert (metal.declaring_plan(plan.plans[slot]).replaces_sub_steps
                    == family.REPLACES)


def _folded_magnetic_source():
    """A real magnetic deposit on the folded fixture.

    ``Hy``, not ``Hz``: an odd component centred on an even mirror plane is refused
    by ``sources._validate_symmetry_parity`` before it can deposit anything.
    """
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    matrix, _probes = _metal()
    fields, _pml = matrix.folded()
    source = VolumeSource(grid=fields.grid, component="Hy",
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert source._n_source_points, "the case deposits nothing and measures nothing"
    return source


def test_a_metal_magnetic_deposit_is_BRACKETED_on_its_seam_and_the_other_stays_fused():
    """THE FOLD BOUNDARY RETIRED, on the case that discriminates.

    ``CARRIES_DEPOSIT_REPAIR`` became True on both Metal folded families on
    2026-08-28, once ``deposit_repair.repair_cells`` extended the saved and restored
    set to the cells this seam's two fills image a deposit into. So the seam the
    source is injected into is no longer refused: it gets the two repair plans, and
    the OTHER seam keeps the bare fused plan. A test that used a source touching both
    seams could not tell "this seam is bracketed" from "everything is".
    """
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        folded_fused_pair as metal_d)

    source = _folded_magnetic_source()
    _fields, plan = _metal_plan(sources=[source])

    assert not plan.reasons.get("fused_pair_folded_fused_magnetic_pair", ())
    # BOTH slots carry the PAIR's label, not the arm's: `_install_fused_pair` writes
    # the same label into the curl slot and the constitutive one, which is how the
    # residency verdict stops reporting the absorbed slot as separate array work.
    assert plan.selected["step_B"] == plan.selected["update_H"] != "folded"
    assert isinstance(plan.plans["step_B"], deposit_repair.LeadingRepairPlan)
    assert isinstance(plan.plans["update_H"], deposit_repair.TrailingRepairPlan)
    # The other seam carries no deposit, so it keeps the bare pair and its sentinel —
    # which is what makes the bracket above a SEAM fact rather than a plan-wide one.
    assert isinstance(plan.plans["step_D"], metal_d.MetalFoldedFusedPairPlan)
    assert not plan.reasons.get("fused_pair_folded_fused_pair")


def test_holding_the_metal_flag_False_puts_the_seam_refusal_straight_back(monkeypatch):
    """THE OTHER DIRECTION, on the same deposit. The refusal prose the boundary shipped
    with is still what the False branch returns, so the bracket above is the flag's
    doing and not a clause that quietly went away."""
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        folded_fused_magnetic_pair as metal_b)

    source = _folded_magnetic_source()
    monkeypatch.setattr(metal_b, "CARRIES_DEPOSIT_REPAIR", False)
    _fields, plan = _metal_plan(sources=[source])

    reasons = plan.reasons.get("fused_pair_folded_fused_magnetic_pair", ())
    assert any("is magnetic" in reason and "BETWEEN step_B and update_H" in reason
               for reason in reasons), reasons
    assert not isinstance(plan.plans["step_B"], deposit_repair.LeadingRepairPlan)


@pytest.mark.parametrize("family_name",
                         ["folded_fused_magnetic_pair", "folded_fused_pair"])
def test_the_routed_metal_families_declare_the_repair_and_gate_it_on_the_flag(
        family_name):
    """A row in ``FUSED_PAIR_ARMS`` says which arm the kernel implements. It still
    does NOT say the seam's deposit can be repaired — that is the family's own
    ``CARRIES_DEPOSIT_REPAIR``, True since 2026-08-28, and it is only honest while the
    seam clause is actually passed it."""
    source = (METAL_DIR / f"{family_name}.py").read_text(encoding="utf-8")
    assert "CARRIES_DEPOSIT_REPAIR = True" in source
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source


#: The ten Metal welds the 2026-08-27 folded-routing round left unrouted. THE
#: POPULATION IS THE SNAPSHOT AND DOES NOT MOVE WHEN A FAMILY IS ROUTED — what moves
#: is how it splits, which is why the floor below is on the sum rather than on the
#: refused half. A family LEAVING this tuple would mean a weld disappeared, which is a
#: different event from a weld being routed and must not read the same.
METAL_UNROUTED_AFTER_THE_2026_08_27_ROUND = (
    "beta_fused_magnetic_pair", "bfast_fused_magnetic_pair",
    "complex_fused_magnetic_pair", "cylindrical_complex_fused_magnetic_pair",
    "cylindrical_real_fused_magnetic_pair",
    "folded_complex_fused_magnetic_pair",
    "folded_beta_complex_fused_magnetic_pair",
    "nonlinear_fused_magnetic_pair", "complex_conductive_fused_pair",
    "fused_dispersive_pair",
)

#: Which of them a LATER round routed, with the round that did it. Enumerated by name
#: so the shrink above is a decision somebody wrote down rather than a count that
#: quietly moved. 2026-08-28, the deposit-carry round: both gained a
#: ``FUSED_PAIR_ARMS`` row read off the arm table and flipped
#: ``CARRIES_DEPOSIT_REPAIR`` in the same edit. Both are UNFOLDED — each refuses a
#: mirrored axis by name — so neither is inside the fold boundary this file is about.
#:
#: 2026-08-30, the fusion-board round: the TWO FOLDED COMPLEX magnetic welds joined
#: them, by the same two-part edit — a ``FUSED_PAIR_ARMS`` row read off the arm table
#: (``fold_complex_2d`` returns ``'folded complex'`` on all four arithmetic slots,
#: ``fold_complex_2d_beta`` returns ``'folded beta complex'``) and
#: ``CARRIES_DEPOSIT_REPAIR`` flipped in the same edit. UNLIKE the 2026-08-28 pair
#: these two ARE folded, so their deposits are carried across the fold by
#: ``deposit_repair.repair_cells``' image closure rather than by a point repair —
#: which is the same machinery that let ``folded_fused_magnetic_pair`` and
#: ``folded_fused_pair`` route on 2026-08-28, and is why routing them is not a claim
#: about the fold boundary this file is about.
#: 2026-09-17, the all-paths round: THREE MORE, and the reason they were unrouted for
#: three weeks is the one this population exists to make visible — each was certified
#: and welded while holding no absorb row, so the seam loop never asked it. The two
#: Dcyl magnetic pairs and the complex conductive D/E weld each gained a
#: ``FUSED_PAIR_ARMS`` row read off its own predicate, and NONE of the three flipped
#: ``CARRIES_DEPOSIT_REPAIR``: all three sit on a seam whose injection their corpus
#: rows do not carry (the Dcyl rows are electric-sourced and the B->H injection is
#: magnetic; the complex conductive cell's four rows are magnetic-sourced and the
#: D->E injection is electric), so each refuses an in-seam deposit BY NAME and is
#: listed in ``test_metal_fused_pair_deposit_wiring.ROUTED_WITHOUT_THE_REPAIR``. That
#: is the difference from the 2026-08-28 and 2026-08-30 entries above, where the row
#: and the flag moved in one edit.
#:
#: THE OTHER THREE OF THE TEN STAY UNROUTED DELIBERATELY. ``beta_fused_magnetic_pair``,
#: ``bfast_fused_magnetic_pair`` and ``nonlinear_fused_magnetic_pair`` were measured
#: winning their slots with rows present (the 2026-09-17 dry run: ``bfast_1d``
#: selected both BFAST pairs, ``special_kz_2d`` the beta magnetic pair) and the rows
#: were taken back out anyway, because routing them empties
#: ``test_metal_fused_pair_deposit_wiring.UNDECLARED_WELDS`` and leaves the
#: refused-by-name property witnessed only by a synthetic stand-in. Four census
#: instances against a control exercised by three real families; the note is in
#: ``metal_kernels/launch.py`` where the rows would go.
METAL_ROUTED_BY_A_LATER_ROUND = ("complex_fused_magnetic_pair",
                                 "fused_dispersive_pair",
                                 "folded_complex_fused_magnetic_pair",
                                 "folded_beta_complex_fused_magnetic_pair",
                                 "cylindrical_complex_fused_magnetic_pair",
                                 "cylindrical_real_fused_magnetic_pair",
                                 "complex_conductive_fused_pair")


def test_the_metal_welds_this_round_did_not_route_are_still_refused_by_name():
    """The blanket refusal is retired for the routed families and kept for the rest.

    TWO families as of this round's own edit, four as of 2026-08-28, six as of
    2026-08-30. The split is
    asserted against the shipped absorb table rather than typed, so a family routed
    without being named in :data:`METAL_ROUTED_BY_A_LATER_ROUND` fails here.
    """
    _fields, plan = _metal_plan()
    population = METAL_UNROUTED_AFTER_THE_2026_08_27_ROUND
    routed = [f for f in population if f in metal.FUSED_PAIR_ARMS]
    undeclared = [f for f in population if f not in metal.FUSED_PAIR_ARMS]
    # The population does not move; only the split does.
    assert len(routed) + len(undeclared) == 10, population
    assert sorted(routed) == sorted(METAL_ROUTED_BY_A_LATER_ROUND), routed
    assert undeclared, "every weld was routed; this test now measures nothing"
    for family in undeclared:
        reasons = plan.reasons.get(f"fused_pair_{family}", ())
        assert any("no absorb declaration" in reason for reason in reasons), family
    # ...and a routed family must NOT still be carrying the absent-declaration
    # refusal, which is what says the row actually reached the seam loop.
    for family in routed:
        reasons = plan.reasons.get(f"fused_pair_{family}", ())
        assert not any("no absorb declaration" in reason for reason in reasons), (
            family, reasons)


# ---------------------------------------------------------------------------
# 7. The dispatch guard the new labels have to keep tripping
# ---------------------------------------------------------------------------

def test_the_routed_labels_still_trip_the_fastpath_fused_refusal():
    """THE HOLE THIS ROUND COULD HAVE PUNCHED, closed by the label and pinned here.

    ``fastpath.plan_fast_path`` clause (8) (fastpath.py:2084-2091) refuses the
    WHOLE dispatch when any slot's label satisfies ``arm_is_fused``, so that a
    future flip of ``fuse`` to default-on cannot silently reach a driver seam no
    gate drove. That recogniser is a PREFIX test against
    ``FUSED_ARM_PREFIXES = ("fused pair",)`` (fastpath.py:205-213) — it knows
    nothing about families, only about how a label starts.

    Before this round the folded seam was blanket-refused, so no label of this
    shape could exist and the prefix could not be evaded. Routing the two families
    creates two NEW labels, and a label reading ``folded fused pair B`` would sail
    past clause (8) while filling ``step_B`` and ``update_H`` — the guard silent at
    exactly the moment it acquired something to guard. The labels therefore lead
    with the prefix, and this reads them off the shipped table rather than
    restating them.
    """
    from meep_gpu import fastpath  # noqa: PLC0415

    labels = [spec["label"] for spec in launch_module.FOLDED_FUSED_PAIRS.values()]
    assert len(labels) == 2, labels
    for label in labels:
        assert fastpath.arm_is_fused(label), label
    # ...and they are still DISTINGUISHABLE from the ordinary pair's, which is the
    # other half: a slot report that could not tell the two products apart would
    # make `selected` useless for saying which kernel ran.
    ordinary = {f"fused pair {name}" for name in launch_module.FUSED_PAIRS}
    assert not (set(labels) & ordinary), sorted(set(labels) & ordinary)
