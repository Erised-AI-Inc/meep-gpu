"""The two-consult protocol, and the deposit repair riding it.

``deposit_repair`` proves the REPAIR against the driver's order. This proves the WIRING:
that ``_install_fused_pair`` puts the right object in each of the two slots, that the pair
of them reproduces the driver's order when consulted the way ``FdtdDriver.step`` consults,
and that a product which has not declared ``CARRIES_DEPOSIT_REPAIR`` still refuses every
in-seam deposit, so none of this can reach a run before it is wired.

WHO HAS DECLARED IT: :data:`WIRED_FOR_THE_REPAIR` below, and the allowlist test beside
it is what keeps that set equal to the set of products ``_install_fused_pair`` actually
brackets.
"""

from __future__ import annotations

import pathlib

import numpy
import pytest

from . import deposit_repair, stepping
from .fields import Fields
from .grid import Grid
from .pml import PML
from .sources import GaussianEnvelope, GaussianPulsedSource, VolumeSource
from .triton_kernels.launch import NoopPlan, _install_fused_pair
from .test_deposit_repair import SEEDED, _build, _differing, _reference, _words


class _ArrayPair:
    """Stands in for the fused kernel: one object doing curl + wall clear + constitutive.

    The real product is one launch and this is three array calls, but the two are the
    same THING to the protocol under test -- an object whose ``run()`` closes the seam
    against whatever the field holds when it is called. Triton is not installed on the
    merge-bar host, and a test that needed it would not run here at all.
    """

    def __init__(self, fields, pml, pair):
        self._fields, self._pml, self._pair = fields, pml, pair
        self.runs = 0

    def run(self, *_args, **_kwargs):
        self.runs += 1
        if self._pair == "B":
            stepping.step_B(self._fields, self._pml)
            stepping.zero_metal_B(self._fields)
            stepping.update_H(self._fields, self._pml)
        else:
            stepping.step_D(self._fields, self._pml)
            stepping.zero_metal_D(self._fields)
            stepping.update_E(self._fields, self._pml)


def _source(grid, pair, component):
    if pair == "B":
        return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                            amplitude=1.0)
    return GaussianPulsedSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                                size=(0.0, 0.0, 0.0), frequency=1.0, fwidth=0.2,
                                amplitude=1.0)


SEAM_CASES = [("B", "Hz", "step_B", "update_H"), ("D", "Ez", "step_D", "update_E")]


@pytest.mark.parametrize("pair,component,curl_name,update_name", SEAM_CASES,
                         ids=[c[0] for c in SEAM_CASES])
def test_a_sourceless_seam_still_gets_the_noop_second_slot(pair, component, curl_name,
                                                           update_name):
    """No deposit, no repair: the second slot must stay the cheap sentinel it was."""
    _grid, fields, pml = _build()
    plans, selected = {}, {}
    _install_fused_pair(plans, selected, fields, pml, (), pair, curl_name, update_name,
                        f"fused pair {pair}", _ArrayPair(fields, pml, pair))
    assert isinstance(plans[update_name], NoopPlan)
    assert selected[curl_name] == selected[update_name] == f"fused pair {pair}"


@pytest.mark.parametrize("pair,component,curl_name,update_name", SEAM_CASES,
                         ids=[c[0] for c in SEAM_CASES])
def test_a_seam_with_a_deposit_gets_the_repair_in_the_second_slot(pair, component,
                                                                  curl_name, update_name):
    grid, fields, pml = _build()
    sources = [_source(grid, pair, component)]
    plans, selected = {}, {}
    _install_fused_pair(plans, selected, fields, pml, sources, pair, curl_name,
                        update_name, f"fused pair {pair}", _ArrayPair(fields, pml, pair))
    assert isinstance(plans[curl_name], deposit_repair.LeadingRepairPlan)
    assert isinstance(plans[update_name], deposit_repair.TrailingRepairPlan)


@pytest.mark.parametrize("pair,component,curl_name,update_name", SEAM_CASES,
                         ids=[c[0] for c in SEAM_CASES])
def test_a_source_on_the_other_side_of_the_step_is_not_this_seam(pair, component,
                                                                 curl_name, update_name):
    """A magnetic source does not make the D/E pair carry a repair, or the reverse."""
    other = "D" if pair == "B" else "B"
    grid, fields, pml = _build()
    sources = [_source(grid, other, "Hz" if other == "B" else "Ez")]
    plans, selected = {}, {}
    _install_fused_pair(plans, selected, fields, pml, sources, pair, curl_name,
                        update_name, f"fused pair {pair}", _ArrayPair(fields, pml, pair))
    assert isinstance(plans[update_name], NoopPlan)


@pytest.mark.parametrize("pair,component,curl_name,update_name", SEAM_CASES,
                         ids=[c[0] for c in SEAM_CASES])
def test_the_two_slots_consulted_as_the_driver_consults_them_match_the_array_path(
        pair, component, curl_name, update_name):
    """THE ONE THAT MATTERS: the protocol, driven the way ``FdtdDriver.step`` drives it."""
    grid_a, fields_a, pml_a = _build()
    grid_b, fields_b, pml_b = _build()
    src_a = [_source(grid_a, pair, component)]
    src_b = [_source(grid_b, pair, component)]
    assert src_b[0]._n_source_points, "the case deposits nothing and cannot discriminate"
    plans, selected = {}, {}
    _install_fused_pair(plans, selected, fields_b, pml_b, src_b, pair, curl_name,
                        update_name, f"fused pair {pair}", _ArrayPair(fields_b, pml_b, pair))
    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, pair)
        # The driver's own order: first consult, its own inject/fill/clear, second consult.
        plans[curl_name].run()
        for source in src_b:
            source.inject(fields_b, step * dt + (0.5 * dt if pair == "D" else 0.0))
        if pair == "B":
            stepping.fill_symmetry_bc_B(fields_b)
            stepping.zero_metal_B(fields_b)
            stepping.fill_folded_far_ghosts_B(fields_b)
        else:
            stepping.fill_symmetry_bc_D(fields_b)
            stepping.zero_metal_D(fields_b)
            stepping.fill_folded_far_ghosts_D(fields_b)
        plans[update_name].run()
    assert not _differing(_words(fields_a), _words(fields_b))
    assert plans[curl_name].repairs > 0, "the repair reported touching no point"


@pytest.mark.parametrize("pair,component,curl_name,update_name", SEAM_CASES,
                         ids=[c[0] for c in SEAM_CASES])
def test_the_same_protocol_without_the_repair_diverges(pair, component, curl_name,
                                                       update_name):
    """The null control for the test above, with the second slot's repair withheld."""
    grid_a, fields_a, pml_a = _build()
    grid_b, fields_b, pml_b = _build()
    src_a = [_source(grid_a, pair, component)]
    src_b = [_source(grid_b, pair, component)]
    inner = _ArrayPair(fields_b, pml_b, pair)
    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, pair)
        inner.run()
        for source in src_b:
            source.inject(fields_b, step * dt + (0.5 * dt if pair == "D" else 0.0))
        if pair == "B":
            stepping.fill_symmetry_bc_B(fields_b)
            stepping.zero_metal_B(fields_b)
            stepping.fill_folded_far_ghosts_B(fields_b)
        else:
            stepping.fill_symmetry_bc_D(fields_b)
            stepping.zero_metal_D(fields_b)
            stepping.fill_folded_far_ghosts_D(fields_b)
    assert _differing(_words(fields_a), _words(fields_b)), (
        "the unrepaired protocol matched the reference, so the case above proves nothing")


@pytest.mark.parametrize("pair,component,curl_name,update_name", SEAM_CASES,
                         ids=[c[0] for c in SEAM_CASES])
def test_the_second_slot_refuses_when_the_first_never_ran(pair, component, curl_name,
                                                          update_name):
    """``dispatch`` contracts that True means the whole sub-step ran; a silent no-op here
    would leave the constitutive half undone and report success."""
    grid, fields, pml = _build()
    plans, selected = {}, {}
    _install_fused_pair(plans, selected, fields, pml, [_source(grid, pair, component)],
                        pair, curl_name, update_name, f"fused pair {pair}",
                        _ArrayPair(fields, pml, pair))
    with pytest.raises(deposit_repair.DepositNotRepairable):
        plans[update_name].run()


@pytest.mark.parametrize("pair,component,curl_name,update_name", SEAM_CASES,
                         ids=[c[0] for c in SEAM_CASES])
def test_the_protocol_taken_through_dispatch_still_matches_the_array_path(
        pair, component, curl_name, update_name):
    """The same word-for-word comparison, taken through the CONSULT rather than ``run()``.

    The tests above call each slot's ``run()`` directly, so they measure the pair and not
    the seam that decides whether to call it. This one goes through
    ``FastPathPlan.dispatch`` — the licence clause, the pair bookkeeping and the step
    roll-over included — for eight steps. That is where a change to the bookkeeping shows
    up as fields that differ, rather than as an argument about flags: a consult that
    wrongly answered False would put the array call on top of the launch, and one that
    wrongly answered True would skip a sub-step.
    """
    grid_a, fields_a, pml_a = _build()
    grid_b, fields_b, pml_b = _build()
    src_a = [_source(grid_a, pair, component)]
    src_b = [_source(grid_b, pair, component)]
    assert src_b[0]._n_source_points, "the case deposits nothing and cannot discriminate"
    plans, selected = {}, {}
    _install_fused_pair(plans, selected, fields_b, pml_b, src_b, pair, curl_name,
                        update_name, f"fused pair {pair}", _ArrayPair(fields_b, pml_b, pair))
    plan = _plan_with(plans, _LapsingPolicy(), fields_b)
    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, pair)
        assert plan.dispatch(curl_name, fields_b) is True
        for source in src_b:
            source.inject(fields_b, step * dt + (0.5 * dt if pair == "D" else 0.0))
        if pair == "B":
            stepping.fill_symmetry_bc_B(fields_b)
            stepping.zero_metal_B(fields_b)
            stepping.fill_folded_far_ghosts_B(fields_b)
        else:
            stepping.fill_symmetry_bc_D(fields_b)
            stepping.zero_metal_D(fields_b)
            stepping.fill_folded_far_ghosts_D(fields_b)
        assert plan.dispatch(update_name, fields_b) is True
    assert not _differing(_words(fields_a), _words(fields_b))
    assert plan.launches == {curl_name: 8, update_name: 8}, (
        "both consults must dispatch on every one of the eight steps")
    assert plan.licence_lapses == {}


#: The products whose leading/trailing slots are WIRED, so that the declaration is a
#: statement about a plan that exists rather than an aspiration. Membership is decided by
#: one question only -- is this product installed through a call to
#: ``_install_fused_pair``, the sole constructor of ``LeadingRepairPlan`` /
#: ``TrailingRepairPlan``, of which there is one copy per composer:
#:
#: * ``triton_kernels/coverage.py`` owns the ordinary B/H and D/E pairs and is installed
#:   at ``triton_kernels/launch.py:2967``;
#: * ``triton_kernels/dispersive_fused_pair.py`` is installed at
#:   ``triton_kernels/launch.py:2988``;
#: * ``metal_kernels/fused_magnetic_pair.py`` is installed by the seam loop in
#:   ``metal_kernels/launch._install_fused_pairs``, whose reach is bounded by
#:   ``metal_kernels.launch.FUSED_PAIR_ARMS``: a Metal weld with no row there is
#:   refused at the absorb table before its predicate is asked. That table, not this
#:   allowlist, is what makes the Metal membership a fact.
#:
#: INSTALLABLE IS NOT THE SAME SET AS REPAIRED, since 2026-08-27. Routing a family
#: and letting it carry the repair are two decisions, and the folded pairs are where
#: they came apart: both folded families gained an absorb row (Metal) and a routing
#: branch (Triton) so a CLEAN folded seam can fuse, and neither may declare the
#: repair. :data:`ROUTED_WITHOUT_THE_REPAIR` is that complement, with the reason.
#:
#: * ``cuda_kernels/fused_magnetic_pair.py`` is installed by the seam loop in
#:   ``cuda_kernels/fused_pairs.install_fused_pairs``, bounded by that module's own
#:   ``FUSED_PAIR_ARMS``, and reached from ``arms.plan_step(fuse=True)``. THE THIRD
#:   COMPOSER, added 2026-08-28. Its block finds candidates in its own
#:   ``FUSED_PRODUCTS`` table rather than by walking the arm table, because every
#:   hand-CUDA family module imports CuPy at module scope and the census that
#:   establishes what this composition covers is backend-free
#:   (``cuda_kernels/registry.py``'s own reason for the same shape).
#:
#: Every other fused product in the tree -- the eight other Triton families and the
#: eleven other Metal ones -- has a coverage predicate and a plan builder but NO
#: installer that brackets its launch. For those, True would admit an in-seam source into
#: a launch nothing repairs, which is the exact failure ``deposit_repair`` exists to
#: prevent, so they stay False until their own installer lands.
#:
#: TWO MORE METAL FAMILIES, 2026-08-28. ``metal_kernels/complex_fused_magnetic_pair.py``
#: and ``metal_kernels/fused_dispersive_pair.py`` each gained a ``FUSED_PAIR_ARMS`` row
#: — read off the arm table the same way the other Metal rows were, and recorded in the
#: ``cart_pml_complex_forced`` and ``cart_pml_dispersive`` rows of
#: ``parity/meep_gpu/metal_composition_matrix.py`` — so the seam loop can bracket them.
#: Each also brings a constitutive half the repair had never reconstructed (complex64
#: storage; a live pole), and each is byte-compared against the driver's own order with
#: a diverging null control in ``test_deposit_repair.py``'s ``COMPLEX_CASES`` and
#: ``DISPERSIVE_CASES``. Both are UNFOLDED families — each refuses a mirrored axis by
#: name — so the fold boundary below does not reach them.
#: THE TWO METAL FOLDED FAMILIES, 2026-08-28. They are the first products to carry a
#: deposit across a FOLDED seam, and what let them is not a table row but
#: ``deposit_repair.repair_cells``: the saved and restored set became the CLOSURE of the
#: cells ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*`` image each deposit point
#: into, so the mirror image the point repair left describing the pre-injection field is
#: recomputed with it. Both directions are measured in ``test_deposit_repair.py`` -- the
#: shipped closure carries a deposit on the near fill's source row and on the far ghost's
#: reflect row byte-exactly, and the same cases with the closure cut back to the deposit
#: index diverge, at those cells. Their Triton twins keep the flag at False: the closure
#: is shared, but nothing has re-cut those products' gates on a device.
#: THE PLAIN METAL D/E PAIR, 2026-08-28. `metal_kernels/fused_electric_pair.py` is the
#: D-side twin of `fused_magnetic_pair.py` -- `step_D` welded into the ORDINARY
#: `update_E`, all three components in one launch -- and on THAT seam the repair is not
#: a widening but the product itself: every one of the 26 corpus rows the cell reaches
#: declares an electric source inside the seam, so the same family with the flag at
#: False would compile, gate green on every other leg, and serve nothing. Its gate's
#: `deposit` leg byte-compares complete driver steps with a real electric VolumeSource
#: in the seam on all six cases, and its `deposit_null_control` leg builds the SAME
#: shipped plan for an empty source list -- the NoopPlan the False branch would leave in
#: the trailing slot -- injects anyway, and requires DIVERGENCE (measured: 4 words of Ez
#: and f_w_Ez at step 1).
WIRED_FOR_THE_REPAIR = frozenset({
    "triton_kernels/coverage.py",
    "triton_kernels/dispersive_fused_pair.py",
    # 2026-08-30: the three Triton families whose flags flipped once their device
    # gates were re-run against the flipped bytes. All three are installed through
    # `triton_kernels.launch._install_fused_pair` -- the two folded ones via
    # `_install_folded_fused_pairs`, the complex one is UNROUTED and so is asserted
    # separately below rather than by the call count.
    "triton_kernels/folded_fused_magnetic_pair.py",
    "triton_kernels/folded_fused_pair.py",
    "triton_kernels/complex_fused_magnetic_pair.py",
    # 2026-08-30, THE TRITON D->E COMPLEX PAIR, landed by the Triton track while this
    # round was closing the Metal one. UNROUTED like its magnetic twin above --
    # `triton_kernels/launch.py` imports no such module and names no
    # `plan_complex_fused_electric_pair` -- so it is asserted through
    # `DECLARED_BUT_NOT_ROUTED` rather than by the installer call count, which stays
    # at 3. Named here because the scan below reads the TREE: a declared flag that
    # this set did not name would fail as a stray flip, and the two facts about it
    # (declared, unrouted) are both measured rather than assumed.
    "triton_kernels/complex_fused_electric_pair.py",
    # 2026-08-30, THE TRITON Dcyl D->E PAIR. Same position as the Cartesian D->E
    # twin above -- declared, built, UNROUTED -- and on this family the flag is not
    # a widening but the product: ALL SIXTEEN Dcyl corpus rows declare an electric
    # source in the seam, so the same module with the flag at False would compile,
    # pass every other leg and serve nothing at all. Asserted through
    # `DECLARED_BUT_NOT_ROUTED`, which keeps the installer call count at 3.
    "triton_kernels/cylindrical_fused_electric_pair.py",
    # 2026-08-31, THE TRITON FOLDED DISPERSIVE D->E PAIR. Same position as the two
    # D->E entries above -- declared, built, UNROUTED -- and, like the Dcyl one, the
    # flag IS the product here rather than one clause of it: all four corpus rows of
    # its board cell declare an electric source in the seam, so the same module with
    # the flag at False would compile, pass every other leg and serve nothing at all.
    # Its ROUTED sibling `triton_kernels/folded_fused_pair.py` is the plain-source
    # twin on the same five driver call sites; this one is not routed because
    # `_install_folded_fused_pairs` installs ONE electric pair per folded grid and
    # choosing between the two is a composition rule nothing has measured. Asserted
    # through `DECLARED_BUT_NOT_ROUTED`, which keeps the installer call count at 3.
    "triton_kernels/folded_dispersive_fused_pair.py",
    # 2026-08-31, THE TRITON Dcyl m = 0 D->E PAIR. Same position as the three
    # D->E entries above -- declared, built, UNROUTED -- and the flag is the
    # product here too: all three corpus rows of its board cell declare an
    # ELECTRIC source, so the same module with the flag at False would compile,
    # pass every other leg and serve nothing. Its MAGNETIC twin
    # (`cylindrical_real_fused_magnetic_pair`) is not in this set at all: it
    # declares the flag False, because the same three rows are electric-only and
    # so its magnetic seam is empty without any repair.
    "triton_kernels/cylindrical_real_fused_electric_pair.py",
    # 2026-08-31, THE TRITON FOLDED COMPLEX D->E PAIR. Same position as the four
    # D->E entries above -- declared, built, UNROUTED -- and the flag is the product
    # here too: BOTH corpus rows of its board cell
    # (`TestEigCoeffs.test_binary_grating_special_kz_2_21_2` and
    # `TestModeDecomposition.test_triangular_lattice_oblique`) declare an ELECTRIC
    # source in the seam, so the same module with the flag at False would compile,
    # pass every other leg and serve nothing at all. Its MAGNETIC twin
    # (`folded_complex_fused_magnetic_pair`) is not in this set: it declares the flag
    # False, because on the B->H seam those same two rows are electric-only and so
    # its seam is empty without any repair. Asserted through
    # `DECLARED_BUT_NOT_ROUTED`, which keeps the installer call count at 3.
    "triton_kernels/folded_complex_fused_pair.py",
    # 2026-08-31, THE TRITON REAL-BETA D->E PAIR. Same position as the five D->E
    # entries above -- declared, built, UNROUTED -- and the flag is the product here
    # too: the cell's single corpus row (``examples:refl-angular-kz2d.py``) declares
    # an ELECTRIC source, so the same module with the flag at False would compile,
    # pass every other leg and serve nothing. Its MAGNETIC twin
    # (``beta_fused_magnetic_pair``) is not in this set: it declares the flag False,
    # because on the B->H seam that same row is electric-only.
    "triton_kernels/beta_fused_electric_pair.py",
    # 2026-08-31, THE TRITON BFAST D->E PAIR. Same position as the six D->E entries
    # above -- declared, built, UNROUTED -- and the flag is the product here too:
    # the cell's single corpus row
    # (``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7``) declares an
    # ELECTRIC source. Its MAGNETIC twin declares the flag False, because on the
    # B->H seam that same row is electric-only.
    "triton_kernels/bfast_fused_electric_pair.py",
    # 2026-08-31, THE TRITON CONDUCTIVE PML D->E PAIR. Same position as the seven
    # D->E entries above -- declared, built, UNROUTED -- and the flag is the product
    # here too: the cell's single corpus row
    # (``tests:TestAdjointSolver.test_damping``) declares TWO sources, one of them
    # ELECTRIC. It has no magnetic twin; the conductive B->H cell is served by the
    # ordinary pair.
    "triton_kernels/conductive_fused_electric_pair.py",
    # 2026-08-31, THE TRITON FOLDED COMPLEX B->H CLAUSE DISCHARGE. Not a new module:
    # the family already shipped a product and already consulted the seam clause
    # through this flag, and the plainrepair7 board recorded its one refused
    # instance (tests:TestHoleyWvgBands.test_fields_at_kx) as "a product exists and
    # REFUSES on another clause" — the clause being the flag itself, since the
    # board's own `in_seam_source_blocks` says the repair CAN carry that row's
    # magnetic deposit. Same position as Metal's own
    # folded_complex_fused_magnetic_pair discharge of 2026-08-30, one backend over.
    # UNROUTED, so it is asserted through `DECLARED_BUT_NOT_ROUTED`.
    "triton_kernels/folded_complex_fused_magnetic_pair.py",
    # 2026-08-31, THE TWO TRITON FOLDED-BETA PAIRS — the last two "not built" cells
    # on the plainrepair7 board. On BOTH the flag is the product: their single
    # corpus row (tests:TestSpecialKz.test_eigsrc_kz_1_real_imag) declares two
    # electric AND two magnetic sources, one pair in each product's own seam, so
    # either module with the flag at False would compile, pass every other leg and
    # serve nothing. Both UNROUTED, asserted through `DECLARED_BUT_NOT_ROUTED`.
    "triton_kernels/folded_beta_fused_electric_pair.py",
    "triton_kernels/folded_beta_fused_magnetic_pair.py",
    # 2026-09-01, THE TWO TRITON COMPLEX-BETA PAIRS — the unfolded half of the
    # no-admitting-arm residue. On the ELECTRIC one the flag is the product: the
    # cell's single corpus row (tests:TestSpecialKz.test_special_kz) declares one
    # electric source in that product's own seam, so the module with the flag at
    # False would compile, pass every other leg and serve nothing. The MAGNETIC
    # one declares True in its siblings' shape (the row's B seam is empty, so the
    # clause is inert on it and the gate's carry legs are what measure the
    # bracket). Both UNROUTED, asserted through `DECLARED_BUT_NOT_ROUTED`.
    "triton_kernels/complex_beta_fused_electric_pair.py",
    "triton_kernels/complex_beta_fused_magnetic_pair.py",
    # 2026-09-02, THE TWO TRITON FOLDED COMPLEX-BETA PAIRS — the folded half of
    # the same residue (K3b's curl riding the certified folded complex weld). On
    # the ELECTRIC one the flag is the whole cell: all THREE corpus rows (the two
    # binary gratings and test_eigsrc_kz_0_complex) declare an electric source in
    # that seam, so the flag at False serves nothing. On the MAGNETIC one it buys
    # one of three: test_eigsrc_kz_0_complex declares TWO in-seam magnetic
    # deposits and the binary gratings none. Both ride the FOLD closure
    # (`repair_cells`' image rules) exactly as their beta-less twins do. Both
    # UNROUTED, asserted through `DECLARED_BUT_NOT_ROUTED`.
    "triton_kernels/folded_beta_complex_fused_pair.py",
    "triton_kernels/folded_beta_complex_fused_magnetic_pair.py",
    "metal_kernels/fused_magnetic_pair.py",
    "metal_kernels/complex_fused_magnetic_pair.py",
    "metal_kernels/fused_dispersive_pair.py",
    "metal_kernels/folded_fused_magnetic_pair.py",
    "metal_kernels/folded_fused_pair.py",
    "metal_kernels/fused_electric_pair.py",
    # 2026-08-30, THE SEVEN NEW METAL PRODUCTS. Each one is a family whose module did
    # not exist before this round; each landed its `launch.FUSED_PAIR_ARMS` row in the
    # same edit as its flag, which is the only thing that lets the seam loop reach
    # `_install_fused_pair` for it (a family with no row is refused by name, "has no
    # absorb declaration", on every configuration). The arm-table equality below is
    # what holds that pairing exact in both directions.
    "metal_kernels/beta_fused_electric_pair.py",
    "metal_kernels/complex_fused_electric_pair.py",
    "metal_kernels/folded_complex_fused_pair.py",
    "metal_kernels/folded_beta_complex_fused_pair.py",
    "metal_kernels/folded_beta_real_fused_magnetic_pair.py",
    "metal_kernels/folded_beta_real_fused_pair.py",
    "metal_kernels/folded_fused_dispersive_pair.py",
    # 2026-08-30, THE TWO CLAUSE DISCHARGES. Not new modules: both families already
    # shipped a product and already consulted the seam clause, and both were held at
    # False by the ABSENCE of an absorb row rather than by anything about their
    # arithmetic -- the board recorded each as `product_exists_but_refuses` on its
    # `in_seam_magnetic_deposit_clears` clause. The row landing is what discharged it,
    # so they add to this set without adding to the consumer count below.
    "metal_kernels/folded_complex_fused_magnetic_pair.py",
    "metal_kernels/folded_beta_complex_fused_magnetic_pair.py",
    "cuda_kernels/fused_magnetic_pair.py",
    # 2026-08-30, THE TWO NEW CUDA PRODUCTS, each with its row in that track's own
    # `fused_pairs.FUSED_PAIR_ARMS` and an entry in `FUSED_PRODUCTS` pointing at the
    # module named here. Both are gated on the GPU host with all four legs released,
    # bit-identity per complete driver step, so the flag stands on an artifact.
    "cuda_kernels/complex_fused_magnetic_pair.py",
    "cuda_kernels/cylindrical_fused_magnetic_pair.py",
    # 2026-09-02, THE FOLDED-COMPLEX CUDA B/H WELD -- the only one of the five
    # residual magnetic products whose flag is LOAD-BEARING on its own cell:
    # four of its five rows are electric-only sourced, and the fifth
    # (TestHoleyWvgBands.test_fields_at_kx) deposits a plain point B Source
    # between the halves, which the bracket reconstructs -- through the FOLD
    # CLOSURE, since every row of the cell is folded. Installed through
    # cuda_kernels/fused_pairs.install_fused_pairs like its siblings. Its four
    # sibling residual welds declare False, each measured off its own cell (all
    # of their clearing rows are electric-only sourced, and the only B-sourced
    # rows deposit through EigenModeSource, which publishes no point index and
    # is beyond any value of the flag); they are in
    # ROUTED_WITHOUT_THE_REPAIR below.
    "cuda_kernels/complex_folded_fused_magnetic_pair.py",
    # 2026-08-30, THE FIRST CUDA PRODUCT ON THE ELECTRIC SEAM, and the one entry here
    # for which the flag is the whole product rather than one clause of it: 75 of the
    # 79 corpus rows on its board cell carry an electric deposit inside the seam, so
    # without the bracket it would serve four. It has its row in that track's
    # `fused_pairs.FUSED_PAIR_ARMS`, its entry in `FUSED_PRODUCTS`, and the `step_D`
    # row in `FUSED_PAIR_SEAMS` that landed with it. Gated on the GPU host under BOTH
    # float32 subnormal policies with the deposit legs byte-identical (228 points
    # repaired) and both unbracketed null controls diverging, so the flag stands on an
    # artifact rather than on a declaration.
    "cuda_kernels/fused_electric_pair.py",
    # 2026-08-31, THE Dcyl m = 0 D->E PAIR ON METAL, and the first product on
    # this board recovered from the BINDING ceiling rather than from a clause.
    # Its cell was refused as UNFUSABLE ON METAL -- 31 pointers against a 30
    # pointer ceiling, over by exactly one -- and what cleared it is
    # `metal_kernels/coefficient_pack.py`: the curl half's six read-only PML
    # coefficient vectors became ONE buffer with six element offsets in the
    # Params struct the six scalars already ride in. On THIS family the flag is
    # the product rather than one clause of it: all three corpus rows the cell
    # reaches declare an electric source in the seam, so the same module with the
    # flag at False would compile, pass every other leg and serve nothing. ROUTED,
    # unlike the Triton twin: it has its row in `metal_kernels/launch.py`'s
    # FUSED_PAIR_ARMS -- ("cylindrical m=0", "cylindrical m=0"), read off
    # `plan_step(..., fuse=False)` on the Dcyl m = 0 matrix row -- so
    # `_install_fused_pairs` reaches it and brackets it, measured in its gate's
    # `deposit` leg with a diverging unbracketed null control beside it.
    "metal_kernels/cylindrical_real_fused_electric_pair.py",
    # 2026-08-31, THE Dcyl |m| >= 1 COMPLEX D->E PAIR ON METAL, and the LARGEST
    # cell on that board — 16 seam-instances. Recovered from the binding ceiling
    # by the same pack as the m = 0 row above: 33 pointers unpacked against a
    # 30-pointer ceiling, over by THREE, and 28 with the curl half's six PML
    # coefficient vectors in one buffer. On this family too the flag is the
    # product rather than one clause of it: all sixteen corpus rows declare an
    # electric source in the seam. ROUTED, through its own FUSED_PAIR_ARMS row
    # ("cylindrical complex", "cylindrical complex"), read off `plan_step(...,
    # fuse=False)` on the Dcyl |m| = 1 and |m| = 2 matrix rows.
    "metal_kernels/cylindrical_fused_electric_pair.py",
    # 2026-09-01, THE TWO METAL PRODUCTS ON THE SECOND REPAIR PATH — the last
    # buildable cells of the 2026-08-31_plainrepair Metal board. Both declare the
    # flag AND `REPAIR_PATHS = (deposit_repair.PLAIN_PATH,)`, because both halves
    # refuse an active absorber and the recurrence their seam inverts is therefore
    # `update_E`'s plain overwrite (stepping.py:1019-1022) — the Metal siblings of
    # `triton_kernels/no_pml_fused_electric_pair.py` above, split into two modules
    # the way the two certified no-absorber curl arms are split. On BOTH the flag
    # is the product: every corpus row of either cell declares an ELECTRIC source
    # in the seam (examples:material-dispersion.py for the plain family;
    # examples:absorber-1d.py and tests:TestAbsorber.test_absorber for the
    # conductive one), so either module with the flag at False would compile, pass
    # every other leg and serve nothing. ROUTED, unlike the Triton twin: each has
    # its own `metal_kernels/launch.py` FUSED_PAIR_ARMS row — ("no-PML curl",
    # "no-PML stored E") and ("conductive no-PML curl", "no-PML stored E"), read
    # off `plan_step(..., fuse=False)` on the no-absorber stored-E matrix rows —
    # so `_install_fused_pairs` reaches and brackets both, and the installer
    # threading hands the bracket each plan's own `repair_paths` rather than the
    # split-field default. The conductive family shipped serving 0 of its 2 rows —
    # its predicate refused a NON-integrated electric source on a conductive run
    # by name, on the driver's then whole-volume condinv rescale — and now serves
    # both: the driver's rescale went SPARSE (per published deposit cell, identity
    # everywhere else) and the clause was LIFTED on the re-measured signed-zero
    # walk (the gate's lifted_refusal leg, byte-identical with the retired
    # whole-volume passes still diverging as the armed control). What the clause
    # still refuses is a scaled source publishing NO deposit table — the driver's
    # dense fallback, which no in-tree source class is.
    "metal_kernels/no_pml_fused_electric_pair.py",
    "metal_kernels/no_pml_conductive_fused_electric_pair.py",
    # 2026-09-01, THE THREE ELECTRIC WELDS OF THE METAL RESIDUE ROUND. Each landed
    # its `launch.FUSED_PAIR_ARMS` row in the same edit as its flag, so the seam
    # loop reaches `_install_fused_pair` for all three. On each the flag is the
    # product: the BFAST cell's one row (TestReflectanceAngular) and the
    # special_kz complex-beta cell's one row (TestSpecialKz.test_special_kz) are
    # electric-sourced, and the conductive-PML cell's one row
    # (TestAdjointSolver.test_damping) carries an electric deposit beside its
    # magnetic one — so any of the three with the flag at False would compile,
    # pass every arithmetic leg and serve nothing. The two packed families
    # (`bfast_fused_electric_pair`, `conductive_fused_electric_pair`) are the
    # third and fourth cells `coefficient_pack.py` recovered from the binding
    # ceiling; the conductive one consults the SAME lifted rescale clause as the
    # no-PML sibling above, one home. The magnetic beta-complex twin declares
    # False on its electric-only cell (the BFAST magnetic pair's measured
    # position) and so is not in this set.
    "metal_kernels/bfast_fused_electric_pair.py",
    "metal_kernels/conductive_fused_electric_pair.py",
    "metal_kernels/beta_complex_fused_electric_pair.py",
    # 2026-08-31, THE TWO CUDA COMPLEX D->E PAIRS, which make that track's `step_D`
    # a three-candidate seam the way its `step_B` already is. Each is ROUTED --
    # a row in `cuda_kernels/fused_pairs.FUSED_PAIR_ARMS` keyed by the module's own
    # `FAMILY` (`cuda_complex_fused_electric_pair`, `cuda_cylindrical_fused_electric_pair`)
    # and a `FUSED_PRODUCTS` row carrying `curl_slot: "step_D"` and pointing at the
    # module named here, which is what lets `install_fused_pairs` reach
    # `_install_fused_pair` for it. The `curl_slot` is load-bearing rather than
    # decorative: it is what keeps each out of the `step_B` loop, so the arm labels
    # each shares with its MAGNETIC twin can never make one seam ambiguous.
    #
    # Both are gated on the GPU host under BOTH float32 subnormal policies with
    # `deposit_legs_passed` true and `released` true on all four artifacts
    # (`results/cuda_fused_complex_pairs_2026-08-31/{complex,cylindrical}_electric_{keep,flush}/gate.json`
    # -- 18 and 16 cases respectively, every case bit-identical to the array path,
    # 27/27 and 33/33 live mutations caught), so each flag stands on an artifact
    # rather than on a declaration.
    "cuda_kernels/complex_fused_electric_pair.py",
    "cuda_kernels/cylindrical_fused_electric_pair.py",
    # 2026-08-31, THE FIRST PRODUCT ON THE SECOND REPAIR PATH.
    # `triton_kernels/no_pml_fused_electric_pair.py` declares the flag AND
    # `REPAIR_PATHS = (deposit_repair.PLAIN_PATH,)`, because both halves refuse an
    # active absorber and the recurrence its seam inverts is therefore `update_E`'s
    # plain overwrite (stepping.py:1019-1022) rather than the split-field one every
    # other entry above inverts. That declaration is not decoration:
    # `deposit_repair.repairable` refuses BY NAME any configuration whose recurrence
    # is not among the paths its caller declares, so this module with the default
    # `(SPLIT_FIELD_PATH,)` would be refused on every row it serves rather than
    # mis-repairing one. UNROUTED, like the nine below.
    "triton_kernels/no_pml_fused_electric_pair.py",
    # 2026-09-02, THE TWO CUDA FLAG FLIPS OF THE RESIDUE ROUND. Both stood at
    # False on the premise that their cells' B-sourced rows "deposit through an
    # EigenModeSource, which publishes no point index" -- RE-MEASURED FALSE of
    # the live tree (the lift synthesises VolumeSource equivalent-current
    # sheets, and every source class publishes _point_ix/_point_iy/_point_iz at
    # setup; a sheet with the lift's amp_func shape returns index ARRAYS from
    # deposit_repair._deposit_index, pinned in
    # test_residual_fused_magnetic_pairs.py). Both are installed through
    # cuda_kernels/fused_pairs.install_fused_pairs like their siblings; the
    # special_kz weld additionally ported the real pair's fill carry in the
    # same edit, because its blocked row is folded and the fold refusal's
    # premise fell with the flag.
    "cuda_kernels/complex_beta_fused_magnetic_pair.py",
    "cuda_kernels/special_kz_fused_magnetic_pair.py",
    # 2026-09-02, THE FIVE CUDA ELECTRIC TWINS of the residue round (the
    # fusion-residue audit's 12 electric-twin instances). On EVERY one the flag
    # is the product rather than one clause of it: every row of every twin's
    # board cell declares an ELECTRIC source inside the D seam (measured off
    # the residualwelds census -- 4 complex-beta rows, 2 folded-complex, 3
    # Dcyl-real, 2 special_kz, 1 BFAST), so any of the five with the flag at
    # False would compile, gate green on every arithmetic leg and serve
    # nothing. Each is ROUTED through its own FUSED_PAIR_ARMS row and
    # FUSED_PRODUCTS entry, landed in the same edit as its flag; each magnetic
    # twin's flag is decided on ITS own seam (the two carrying ones above, the
    # two measured-False ones below).
    "cuda_kernels/complex_beta_fused_electric_pair.py",
    "cuda_kernels/complex_folded_fused_electric_pair.py",
    "cuda_kernels/cylindrical_real_fused_electric_pair.py",
    "cuda_kernels/special_kz_fused_electric_pair.py",
    "cuda_kernels/bfast_fused_electric_pair.py",
    # 2026-09-02, THE CUDA DISPERSIVE D->E PAIR, and the entry on this list whose
    # cell is the largest: `D_to_E (cuda_curl/PML, cuda_dispersive/dispersive)`
    # carries 7 corpus rows and ALL SEVEN declare an electric deposit inside the
    # seam (`source_field_types == ['D']` on six, ten D sources on
    # stochastic_emitter.py, measured off the stamped census). The flag is
    # therefore the entire product: at False this module would compile, gate green
    # on every arithmetic leg and serve ZERO rows -- not the four its non-dispersive
    # twin would still have served. It is ROUTED through its own FUSED_PAIR_ARMS row
    # and FUSED_PRODUCTS entry (`curl_slot: "step_D"`), landed in the same edit as
    # the flag.
    #
    # WHAT ITS BRACKET HAS TO RECONSTRUCT IS NEW ON THIS LIST: a DISPERSIVE
    # update_E. `deposit_repair.apply` recomputes `fw_fresh` through
    # `Fields.displacement_minus_polarization`, which IS this half's source, and it
    # is entitled to the pole arrays it reads because `update_P` runs after
    # `update_E` (driver.py:3315). The sibling Metal family measured that same
    # entitlement in `test_deposit_repair.py`'s DISPERSIVE_CASES; this product
    # inherits the clause rather than a second copy of it.
    "cuda_kernels/dispersive_fused_electric_pair.py",
    # 2026-09-02, THE TWO CONDUCTIVE-CURL D->E PAIRS, which close the last
    # POINTWISE-BUILDABLE cells this board had on that seam. All four of their
    # corpus rows declare an electric deposit inside the seam, so on both the flag
    # is the entire product: at False either module would compile, gate green on
    # every arithmetic leg and serve ZERO rows. Both are ROUTED through their own
    # FUSED_PAIR_ARMS row and FUSED_PRODUCTS entry (`curl_slot: "step_D"`), landed
    # in the same edit as the flag; the no-absorber one additionally declares the
    # SECOND cell it occupies in FUSED_PAIR_EXTRA_ARMS.
    #
    # WHAT THE FIRST ONE'S BRACKET HAS TO RECONSTRUCT IS NEW ON THIS LIST, AND IT IS
    # NOT THE SPLIT-FIELD RECURRENCE. Its layer is INERT, so `update_E` takes the
    # pure overwrite at stepping.py:1019-1022 and the repair that inverts it is
    # `deposit_repair.PLAIN_PATH`. It is the first module on this list to declare a
    # non-default `REPAIR_PATHS`, and the declaration is load-bearing rather than
    # descriptive: `deposit_repair.repairable` refuses BY NAME any configuration
    # whose recurrence is not among the caller's declared paths, so the wrong
    # declaration is a refusal and not a mis-repair. Measured on all seven of the
    # probe's configurations
    # (`parity/meep_gpu/probe_cuda_no_pml_dispersive_electric_pair.py`'s
    # `split_field_declared` leg).
    "cuda_kernels/no_pml_dispersive_fused_electric_pair.py",
    # THE SECOND IS THE SIGNED-ZERO ROW, and its flag was unreachable rather than
    # merely unset until 2026-09-01: the driver rescaled its condinv injection over
    # the WHOLE volume, rewriting -0.0 to +0.0 at cells no deposit closure can name,
    # so no bracket could have made it byte-identical. The driver now replays that
    # rescale sparsely at the published deposit indices, and
    # `parity/meep_gpu/probe_cuda_conductive_electric_pair.py`'s `rescale` leg
    # re-measures both halves of that: the shipped injection leaves every
    # non-deposit word untouched on 4 of 4 configurations, and the retired
    # whole-volume control diverges on all four in the signed-zero class alone.
    "cuda_kernels/conductive_fused_electric_pair.py",
    # 2026-09-02, THE TWO THREE-SLOT WELDS, and the first entries on this list
    # whose product owns THREE driver slots. Every one of their ten corpus rows
    # carries an electric deposit inside the `step_D` -> `update_E` span, so on
    # both the flag is the entire product; both are ROUTED through their own
    # `FUSED_PAIR_ARMS` row and `FUSED_PRODUCTS` entry (`curl_slot: "step_D"`,
    # plus a `slots` key naming all three), landed in the same edit as the flag.
    #
    # WHAT IS NEW HERE IS NOT THE BRACKET BUT WHAT SITS AFTER IT. On every other
    # entry the product's device work is finished by the time
    # `TrailingRepairPlan.run` calls `deposit_repair.apply`. These two launch
    # AGAIN afterwards, in `update_P`, and the ORDER IS LOAD-BEARING: `apply`
    # recomputes the constitutive product through
    # `Fields.displacement_minus_polarization` on both of its paths
    # (deposit_repair.py:781-782, :922-923), which subtracts `state.P[c]` AS IT
    # STANDS (fields.py:1096-1105) -- so the repair is correct only while P is
    # still P^n. `fused_pairs._install_fused_triple` is the one place that order
    # is written, which is why the third slot is filled there beside the bracket
    # rather than by a caller free to order the two differently. Measured:
    # `parity/meep_gpu/probe_cuda_three_slot_weld.py`'s `repair_after_p` and
    # `p_inside_the_launch` orderings each diverge on 6 of 7 configurations.
    #
    # THE PML ONE DECLARES THE SPLIT-FIELD REPAIR AND THE NO-ABSORBER ONE THE
    # PLAIN ONE, each inherited from the D->E half it welds rather than
    # re-decided, which is why they are two modules: `_repair_paths_of` reads
    # that declaration off the MODULE and one module cannot carry two answers.
    "cuda_kernels/three_slot_dispersive_weld.py",
    "cuda_kernels/no_pml_three_slot_dispersive_weld.py",
})

#: Wired modules whose composer does NOT yet route them, so the call-count and
#: arm-table bounds below cannot speak for them. Enumerated rather than skipped: a
#: module here declares the repair and is bracketed the moment its composer reaches
#: it, and until then the ONLY thing standing between it and an unbracketed launch
#: is that nothing builds it at all.
#:
#: EMPTY SINCE 2026-09-02, and the emptying is the installer wave.
#:
#: It held SEVENTEEN Triton modules, every one of them declared, built and reached
#: by no composer, and every one of them carrying a variant of the same reason:
#: "wiring it would ALSO put two admitters on ``step_D``, which is a coverage loss
#: rather than a gain". That reason was about registering the product as an ARM in
#: ``_select_slot``'s table — a route none of them was ever going to take. A fused
#: pair is not an arm: it never registers on a slot, and ``_pair_may_absorb`` lets
#: it claim two slots exactly when the arm table has already given both to the arms
#: its kernel implements. ``launch._install_certified_fused_products`` routes all
#: seventeen through that clause, so each is now asserted from the OTHER direction
#: — ``plan_<stem>`` PRESENT in ``launch.py``, by
#: :func:`test_every_triton_routed_module_has_decided_about_the_bracket` below.
#:
#: THE SET IS KEPT RATHER THAN DELETED because the direction it guards is real and
#: is not the same as the routed set's: a module that claims the repair and is
#: reached by no installer runs unbracketed the day something starts building it.
#: A future declared-but-unrouted module belongs here, and the loop that reads it
#: is still in place.
DECLARED_BUT_NOT_ROUTED = frozenset()

#: Per wired module, the ARM TABLE ROW that proves its composer can actually reach it.
#: Triton's fusion block is hand-written per pair and is pinned by the call count below;
#: Metal's is a LOOP over ``arms.registered``, so what bounds it is
#: ``FUSED_PAIR_ARMS``. A Metal family flipping the flag without gaining a row there
#: would be declaring a bracket the loop refuses to build.
#:
#: 6 -> 15 ON 2026-08-30, one row per family named in :data:`WIRED_FOR_THE_REPAIR`
#: under ``metal_kernels/``. The two sets are the same fifteen families, which is why
#: ``ROUTED_WITHOUT_THE_REPAIR`` below is still empty: no family the seam loop can
#: install declines the repair.
#:
#: 17 -> 19 ON 2026-09-01 with the two no-absorber stored-E D->E welds
#: (``no_pml_fused_electric_pair``, ``no_pml_conductive_fused_electric_pair``) —
#: the first Metal rows whose bracket installs ``deposit_repair.PLAIN_PATH``
#: rather than the split-field repair. Each landed its row, its flag and its
#: ``REPAIR_PATHS`` declaration in the same edit, and the installer reads the
#: declaration off the built plan (``getattr(pair, "repair_paths", ...)``), so
#: the row is what lets the seam loop bracket them with the repair each actually
#: inverts. THREE FAMILIES THE LOOP REACHES ARE DELIBERATELY ABSENT
#: -- ``beta_fused_magnetic_pair``, ``bfast_fused_magnetic_pair`` and
#: ``nonlinear_fused_magnetic_pair`` (there were six until 2026-09-17; the
#: 37 -> 40 entry below names the three that left and why these three stay).
#: Each is registered as a weld that replaces its update slot, so the loop does visit
#: it, and each is refused by name for want of a row here while its module ships the
#: flag at False. That is the pairing this tuple exists to keep exact: a family may
#: gain the row and the flag together or neither, never one alone.
#: 23 -> 27 ON 2026-09-02 with the four OFF-DIAGONAL SCRATCH-OUTPUT welds. These are
#: the first Metal rows to hold a row here while shipping the flag at False, and they
#: REFINE the pairing rule above rather than breaking it: the row and the flag answer
#: different questions, and the composer is what settles it. `FUSED_PAIR_ARMS.get(
#: spec.family)` returning None REFUSES the family outright before its predicate is
#: asked (``metal_kernels/launch.py:1025-1031``), so the row is what makes a family
#: INSTALLABLE AT ALL; the flag only says whether the seam loop brackets a deposit
#: around it. For these four the flag is FORCED False -- ``deposit_repair.repairable``
#: refuses an off-diagonal chi1inv BY NAME, since a point repair recomputes E at the
#: deposit cell from that cell's own displacement while this constitutive reads its
#: NEIGHBOURS' -- so each serves exactly the rows that clear the source seam without a
#: bracket (8 + 9 + 1 + 1 = 19 of the 387 priced on the 2026-09-02 board) and refuses
#: the rest by name. Dropping their rows to preserve the older reading would unserve
#: all 19. They are declared here AND in :data:`ROUTED_WITHOUT_THE_REPAIR`, which is
#: the pair of declarations that spells "installable, deliberately unbracketed".
#: 27 -> 28 ON 2026-09-05 with ``fused_hd_pair``, the H->D weld. Its flag is False
#: for a reason that is a fact about the DRIVER rather than about the repair: nothing
#: is injected between the ``update_H`` and ``step_D`` consults (the electric
#: withdraw sits there, and ``withdraw_hoist`` owns that), so there is no deposit to
#: bracket and ``deposit_repair`` is not consulted at all. The row is what lets the
#: seam loop ASK the product; the product declares ``INSTALLABLE = False`` on the
#: measured composition verdict, and ``_pair_may_absorb`` refuses it on every corpus
#: row its primary arm reaches even without that flag.
#: 32 -> 37 ON 2026-09-08 with the five H->D rows the wiring round landed. Each is the
#: same shape as the three 2026-09-07 ones already here: a two-slot span over
#: ``update_H``/``step_D``, ``CARRIES_DEPOSIT_REPAIR = False`` because nothing is
#: injected between those two consults, and ``INSTALLABLE = False`` on a composition
#: verdict measured through the shipped composer -- so all five are
#: ``ROUTED_WITHOUT_THE_REPAIR`` entries too, and the row is only what lets the seam
#: loop ASK them.
#: 37 -> 40 ON 2026-09-17 with the three B->H and D->E rows the all-paths round
#: routed (``metal_kernels/launch.py:1234-1246``): ``complex_conductive_fused_pair``,
#: ``cylindrical_complex_fused_magnetic_pair`` and
#: ``cylindrical_real_fused_magnetic_pair``. Each was CERTIFIED and carried a released
#: weld while holding no row, so
#: ``dispatch_reachability.metal_certified_but_not_installed`` reported it and the
#: seam loop never asked it. Each ships
#: ``CARRIES_DEPOSIT_REPAIR = False`` (complex_conductive_fused_pair.py:229,
#: cylindrical_real_fused_magnetic_pair.py:227, cylindrical_fused_magnetic_pair.py:277)
#: and passes that flag to ``deposit_repair.seam_source_reasons`` (:589, :539, :721),
#: so an in-seam deposit is refused BY NAME rather than run unbracketed -- the
#: 2026-09-02 "installable, deliberately unbracketed" pairing, and so all three are
#: :data:`ROUTED_WITHOUT_THE_REPAIR` entries too. THE STEM TRAP A SECOND TIME:
#: ``cylindrical_complex_fused_magnetic_pair`` lives in
#: ``cylindrical_fused_magnetic_pair.py``, exactly as its electric twin does, and the
#: routed set is keyed by that MODULE path. The other three magnetic welds stay out
#: on purpose (launch.py's "BELOW-THE-CUT THREE", :1247-1264): measured 2026-09-19,
#: they are the whole real population ``UNDECLARED_WELDS`` in
#: ``test_metal_fused_pair_deposit_wiring.py`` sweeps for a by-name refusal.
METAL_ABSORB_DECLARATIONS = ("beta_complex_fused_electric_pair",
                             # 2026-09-08, THE TWO BETA H->D ROWS: one product per
                             # storage, each covering a SECOND cell through a
                             # FUSED_PAIR_EXTRA_ARMS row (the folded variant), so one
                             # row here bounds two cells. INSTALLABLE = False on a
                             # measured arbitration -- both neighbouring seams carry
                             # Metal products on both variants, so the span is a LOSS
                             # where both install and a TIE the released B->H
                             # incumbent wins where one does.
                             "beta_complex_fused_hd_pair",
                             "beta_complex_fused_magnetic_pair",
                             "beta_fused_electric_pair",
                             "beta_real_fused_hd_pair",
                             "bfast_fused_electric_pair",
                             # 2026-09-08, THE BFAST H->D ROW. Its arbitration is the
                             # one the measurement CORRECTED: only ONE neighbour
                             # installs on a BFAST run (bfast_fused_magnetic_pair
                             # carries no absorb row at all), so the span is a TIE
                             # rather than a loss -- resolved for the incumbent.
                             "bfast_fused_hd_pair",
                             # 2026-09-17, THE COMPLEX CONDUCTIVE D->E ROW, routed by
                             # the all-paths round: CARRIES_DEPOSIT_REPAIR = False and
                             # its source clause refuses an in-seam electric deposit
                             # by name, so a ROUTED_WITHOUT_THE_REPAIR entry.
                             "complex_conductive_fused_pair",
                             "complex_fused_electric_pair",
                             # 2026-09-07, THE CARTESIAN COMPLEX/BLOCH H->D ROW: one
                             # dispatch owning update_H and step_D on the complex
                             # cell; INSTALLABLE = False on a measured arbitration
                             # (both neighbouring released pairs install on 17 of 17
                             # rows), so it is a ROUTED_WITHOUT_THE_REPAIR entry like
                             # fused_hd_pair.
                             "complex_fused_hd_pair",
                             "complex_fused_magnetic_pair",
                             "complex_no_pml_offdiag_fused_electric_pair",
                             "conductive_fused_electric_pair",
                             # 2026-09-08, THE CONDUCTIVE H->D ROW: on an electric-
                             # conductivity run the magnetic curl is the ORDINARY one,
                             # so this span's two neighbours are the released
                             # conductive_fused_electric_pair and fused_magnetic_pair
                             # and installing it is a LOSS wherever both serve.
                             "conductive_fused_hd_pair",
                             "cylindrical_complex_fused_electric_pair",
                             # 2026-09-07, THE TWO CYLINDRICAL H->D ROWS: two-launch
                             # products owning update_H and step_D on the Dcyl complex
                             # and m=0 real cells; INSTALLABLE = False for the measured
                             # composition TIE with the released D->E pair, so both are
                             # ROUTED_WITHOUT_THE_REPAIR entries like fused_hd_pair.
                             "cylindrical_complex_fused_hd_pair",
                             # 2026-09-17, THE TWO DCYL B->H ROWS, routed by the
                             # all-paths round and each the magnetic twin of a Dcyl
                             # D->E row here: CARRIES_DEPOSIT_REPAIR = False with an
                             # in-seam magnetic source refused by name, so both are
                             # ROUTED_WITHOUT_THE_REPAIR entries (the complex one
                             # under its module stem, cylindrical_fused_magnetic_pair).
                             "cylindrical_complex_fused_magnetic_pair",
                             "cylindrical_real_fused_electric_pair",
                             "cylindrical_real_fused_hd_pair",
                             "cylindrical_real_fused_magnetic_pair",
                             "folded_beta_complex_fused_magnetic_pair",
                             "folded_beta_complex_fused_pair",
                             "folded_beta_real_fused_magnetic_pair",
                             "folded_beta_real_fused_pair",
                             # 2026-09-08, THE FOLDED COMPLEX H->D ROW: both of this
                             # cell's neighbouring seams carry released Metal products
                             # (folded_complex_fused_magnetic_pair and
                             # folded_complex_fused_pair), so the span is a strict LOSS
                             # where both install and a TIE the incumbent wins where
                             # one does; INSTALLABLE = False on that measurement.
                             "folded_complex_fused_hd_pair",
                             "folded_complex_fused_magnetic_pair",
                             "folded_complex_fused_pair",
                             "folded_complex_offdiag_fused_electric_pair",
                             "folded_fused_dispersive_pair",
                             # 2026-09-07, THE FOLDED H->D ROW: one dispatch owning
                             # update_H and step_D on the fold, the largest cell on
                             # this backend (78 rows). INSTALLABLE = False on a
                             # measured arbitration -- the released
                             # folded_fused_magnetic_pair holds update_H on 78 of 78 --
                             # so it too is a ROUTED_WITHOUT_THE_REPAIR entry.
                             "folded_fused_hd_pair",
                             "folded_fused_magnetic_pair",
                             "folded_fused_pair",
                             "folded_offdiag_fused_electric_pair",
                             "fused_dispersive_pair",
                             "fused_electric_pair",
                             # 2026-09-05, THE H->D WELD -- the first row on this
                             # table for the fourth seam, written in that seam's own
                             # slot order (constitutive, then curl). It ships
                             # CARRIES_DEPOSIT_REPAIR = False because NOTHING is
                             # injected between update_H and step_D, and
                             # INSTALLABLE = False for the measured composition
                             # verdict; a ROUTED_WITHOUT_THE_REPAIR entry.
                             "fused_hd_pair",
                             "fused_magnetic_pair",
                             "no_pml_conductive_fused_electric_pair",
                             "no_pml_fused_electric_pair",
                             "offdiag_fused_electric_pair")

#: The same bound on the CUDA composer, whose block is a loop over its own product
#: table rather than over the arm table. The first row is the certified B/H product,
#: absorbing the ``PML`` curl arm and the ``ordinary`` constitutive arm — which is
#: what its predicate literally is, a conjunction of those two arms' own predicates.
#:
#: 1 -> 3 ON 2026-08-30 with the two complex B/H products, whose arms are read from
#: their predicates the same way: ``cuda_complex_fused_magnetic_pair`` opens on the
#: ``cuda_complex`` family's ``complex`` arm in both slots, and
#: ``cuda_cylindrical_fused_magnetic_pair`` on the ``cuda_cyl_complex`` family's
#: ``cylindrical complex`` curl arm with that same complex constitutive. Both are
#: also rows in ``FUSED_PRODUCTS``, which the equality below holds equal to this one.
#: 3 -> 4 LATER THE SAME DAY with ``cuda_fused_electric_pair``, the first product on
#: that track's ``step_D``/``update_E`` seam. Its arms are read from its predicate the
#: same way — ``covers_fused_electric_pair`` opens on ``covers_real_pml_curl(...,
#: "step_D")`` and ``covers_real_pml_constitutive(..., "E")``, which are the ``PML``
#: curl arm's and the ``ordinary`` constitutive arm's own predicates. IT NAMES THE SAME
#: TWO ARM LABELS AS THE FIRST ROW AND THAT IS NOT AN AMBIGUITY: the two are keyed on
#: different ``curl_slot`` values and ``install_fused_pairs`` decides one seam at a
#: time, which the CUDA suite asserts directly.
#:
#: 4 -> 6 ON 2026-08-31 with the two COMPLEX electric products, whose arms are read
#: off their predicates exactly as every row above was.
#: ``covers_complex_fused_electric_pair`` opens on ``covers_real_pml_complex_curl(...,
#: "step_D")`` and ``covers_real_pml_complex_constitutive(..., "E")`` — both the
#: ``cuda_complex`` family's ``complex`` arm — and
#: ``covers_cylindrical_fused_electric_pair`` on
#: ``covers_pml_cylindrical_complex_curl(..., "step_D")``, the ``cuda_cyl_complex``
#: family's ``cylindrical complex`` arm, with that same complex constitutive. EACH
#: SHARES ITS LABEL PAIR WITH ITS MAGNETIC TWIN and that is not an ambiguity for the
#: reason the real electric row already carries: the twins are keyed on different
#: ``curl_slot`` values and ``install_fused_pairs`` decides one seam at a time.
#: The names here are each module's own ``FAMILY``, not its stem — the Dcyl module
#: is ``cylindrical_fused_electric_pair.py`` and its family is
#: ``cuda_cylindrical_fused_electric_pair`` — and the per-row module assertion below
#: is what holds the two together.
#: 6 -> 7 ON 2026-08-31 with the complex NO-ABSORBER electric product, whose arms are
#: read off its predicate the same way: ``covers_no_pml_complex_fused_electric_pair``
#: opens on ``complex_no_pml_kernels.covers_complex_no_pml_curl(..., "step_D")`` and
#: ``covers_complex_no_pml_stored_e``, which ARE the ``cuda_complex_no_pml`` family's
#: ``complex no-PML curl`` and ``complex no-PML stored E`` arms. IT IS THE FIRST CUDA
#: ROW WHOSE MODULE DECLARES THE FLAG FALSE, so it is the first member of
#: :data:`ROUTED_WITHOUT_THE_REPAIR` on this track rather than of
#: :data:`WIRED_FOR_THE_REPAIR` — routable and deliberately unbracketed, which is the
#: distinction that set exists to carry.
CUDA_ABSORB_DECLARATIONS = (
                            # SORTED, because the assertion compares this
                            # against tuple(sorted(FUSED_PAIR_ARMS)). Three
                            # products landed on 2026-09-02 (the two stencil
                            # welds and the off-diagonal E->P weld) and were
                            # appended rather than inserted, which failed the
                            # comparison on ORDER while the sets matched --
                            # a difference no reader of the message could see.
                            "cuda_bfast_fused_electric_pair",
                            "cuda_bfast_fused_magnetic_pair",
                            "cuda_complex_beta_fused_electric_pair",
                            # 2026-09-07, THE COMPLEX BETA H->D WELD. Its row here
                            # is what lets the seam loop REACH and NAME it; the flag
                            # is False because nothing is injected between the
                            # update_H and step_D consults, so it joins
                            # ROUTED_WITHOUT_THE_REPAIR below beside the four other
                            # H->D spans.
                            "cuda_complex_beta_fused_hd_pair",
                            "cuda_complex_beta_fused_magnetic_pair",
                            "cuda_complex_folded_fused_electric_pair",
                            "cuda_complex_folded_fused_magnetic_pair",
                            "cuda_complex_fused_electric_pair",
                            # 2026-09-07, THE CARTESIAN COMPLEX H->D WELD -- the
                            # fourth row on this table for the fourth seam, and one
                            # product over TWO board cells (the plain complex curl
                            # and the folded one, selected from the grid). Its row
                            # here is what lets the seam loop REACH and NAME it; the
                            # flag is False because nothing is injected between the
                            # update_H and step_D consults, and the module declares
                            # INSTALLABLE = False, so it joins
                            # ROUTED_WITHOUT_THE_REPAIR below beside fused_hd_pair
                            # and the two cylindrical siblings.
                            "cuda_complex_fused_hd_pair",
                            "cuda_complex_fused_magnetic_pair",
                            "cuda_complex_no_pml_fused_polarization_pair",
                            # 2026-09-02, THE TWO COMPLEX STENCIL WELDS -- the last
                            # two cells the hand-CUDA board recorded UNBUILDABLE.
                            # Both declare CARRIES_DEPOSIT_REPAIR=False (an
                            # off-diagonal chi1inv row is refused by name), so
                            # their rows here are what lets the seam loop reach
                            # them at all rather than what lets it bracket them.
                            "cuda_complex_no_pml_offdiag_fused_electric_pair",
                            # 2026-09-08, THE CONDUCTIVE/BFAST H->D WELD -- ONE
                            # product on TWO cells, its second reached through a
                            # FUSED_PAIR_EXTRA_ARMS row (('BFAST', 'BFAST')) beside
                            # the primary ('ordinary', 'conductive') one. Its row
                            # here is what lets the seam loop REACH and NAME it;
                            # the flag is False on both variants because nothing is
                            # injected between the update_H and step_D consults, so
                            # it is a ROUTED_WITHOUT_THE_REPAIR entry (already
                            # listed there) like the four other H->D spans.
                            "cuda_conductive_bfast_fused_hd_pair",
                            # 2026-09-02, the conductive x ordinary D/E weld --
                            # the signed-zero row, servable only since the
                            # driver's condinv injection began rescaling
                            # sparsely at the published deposit indices.
                            "cuda_conductive_fused_electric_pair",
                            "cuda_cylindrical_fused_electric_pair",
                            # 2026-09-07, THE TWO CYLINDRICAL H->D WELDS (complex
                            # any-m and real m = 0), the second and third rows on
                            # this table the composer refuses on EVERY
                            # configuration (INSTALLABLE False, the cylindrical
                            # launch algebra measured a composition TIE with the
                            # released pairs). Their rows let the seam loop REACH
                            # and NAME them; the flag is False and the seam carries
                            # the electric withdraw, not a deposit, so both join
                            # ROUTED_WITHOUT_THE_REPAIR below beside fused_hd_pair.
                            "cuda_cylindrical_fused_hd_pair",
                            "cuda_cylindrical_fused_magnetic_pair",
                            "cuda_cylindrical_real_fused_electric_pair",
                            "cuda_cylindrical_real_fused_hd_pair",
                            "cuda_cylindrical_real_fused_magnetic_pair",
                            # 2026-09-02, the dispersive D/E weld: the largest
                            # single cell any backend had left unoccupied (7
                            # corpus rows), and the row here is what lets the
                            # seam loop bracket it. Without the row its flag
                            # would stand at True and never be reached, which is
                            # the exact shape this equality exists to catch.
                            "cuda_dispersive_fused_electric_pair",
                            "cuda_dispersive_offdiag_fused_polarization_pair",
                            "cuda_folded_complex_offdiag_fused_electric_pair",
                            "cuda_folded_offdiag_fused_electric_pair",
                            "cuda_fused_electric_pair",
                            # 2026-09-06, THE H->D WELD -- the FOURTH seam, and the
                            # first row on this list whose product the composer
                            # refuses on EVERY configuration
                            # (fused_hd_pair.INSTALLABLE False). Its row here is what
                            # lets the seam loop REACH and NAME it rather than what
                            # lets it bracket it: the flag is False and the seam
                            # carries a withdrawal, not a deposit, so it joins
                            # ROUTED_WITHOUT_THE_REPAIR below.
                            "cuda_fused_hd_pair",
                            "cuda_fused_magnetic_pair",
                            "cuda_fused_polarization_pair",
                            "cuda_no_pml_complex_fused_electric_pair",
                            # 2026-09-02, the no-absorber dispersive-store D/E
                            # weld: ONE product on TWO board cells, and the only
                            # row in this tuple whose module declares a
                            # non-default REPAIR_PATHS.
                            "cuda_no_pml_dispersive_fused_electric_pair",
                            "cuda_no_pml_fused_polarization_pair",
                            "cuda_offdiag_fused_electric_pair",
                            "cuda_special_kz_fused_electric_pair",
                            # 2026-09-07, THE REAL BETA H->D WELD -- same shape,
                            # same flag, same ROUTED_WITHOUT_THE_REPAIR entry.
                            "cuda_special_kz_fused_hd_pair",
                            "cuda_special_kz_fused_magnetic_pair",
                            # 2026-09-02, THE TWO THREE-SLOT WELDS. Their rows
                            # here are the first on this table of length THREE
                            # rather than two -- the value is still "the arm
                            # each slot the product absorbs implements", in the
                            # product's own slot order, and `install_fused_pairs`
                            # asks `_pair_may_absorb` once per seam the span
                            # crosses rather than growing a second copy of that
                            # clause. Without these rows the seam loop would
                            # refuse both by name ("has no absorb declaration")
                            # while their flags stood at True, which is the exact
                            # shape this equality exists to catch.
                            # 2026-09-04: the COMPLEX no-absorber three-slot weld
                            # (the four TestLoadDump 3-D rows), sorted before its
                            # two real siblings; a ROUTED_WITHOUT_THE_REPAIR entry.
                            "cuda_three_slot_complex_no_pml_dispersive_weld",
                            "cuda_three_slot_dispersive_weld",
                            "cuda_three_slot_no_pml_dispersive_weld")

#: Families whose composer CAN install them but which must NOT declare the repair,
#: with the reason. Kept as a named list rather than a hole in an equality, so a
#: third family arriving in this position is a decision somebody has to write down.
#:
#: THE FOLD BOUNDARY, 2026-08-27, RETIRED FOR METAL ON 2026-08-28. A folded seam also
#: runs ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*``, and the driver runs them
#: AFTER the injection (driver.py:3293-3295, :3308-3311). The near fill writes cell 0
#: from cell 2 (stepping.py:1450-1450), while ``deposit_repair.apply`` used to write only
#: the deposit INDEX and never its mirror image — so a repaired folded deposit left the
#: ghost cell describing the pre-injection field. ``deposit_repair.repair_cells`` closed
#: that, and the two METAL folded families flipped in the same edit.
#:
#: THE TWO TRITON TWINS USED TO STAY HERE, and not because the closure is
#: backend-specific — it is the one shared module both tracks reach. What they lacked
#: was a re-cut gate: every Triton fused product's arithmetic is certified on a device
#: that round could not run, so flipping them would have been a claim no artifact
#: stood behind. THAT WAS DISCHARGED ON 2026-08-30 by re-running the gates, which is
#: the only thing that could discharge it — not by deciding the flip was safe. Both
#: moved to :data:`WIRED_FOR_THE_REPAIR`.
#:
#: THE SET STOPPED BEING EMPTY ON 2026-08-31, on the CUDA track, and the entry is a
#: MEASUREMENT rather than a caution. ``cuda_kernels/no_pml_complex_fused_electric_pair.py``
#: is installable — it has a ``FUSED_PAIR_ARMS`` row and a ``FUSED_PRODUCTS`` row, so
#: ``fused_pairs.install_fused_pairs`` can reach ``_install_fused_pair`` with it — and
#: it declares ``CARRIES_DEPOSIT_REPAIR = False`` because ALL FOUR rows of its board
#: cell declare ``source_field_types == ['B']``: a magnetic source, injected one seam
#: earlier (driver.py:3293), and NOTHING in the D seam. The flag at True would buy
#: exactly zero rows there while claiming a bracket this module installs nowhere, which
#: is the direction that would be a defect. So the installer always takes its
#: ``NoopPlan`` branch on this family, and that is what the entry records.
#:
#: THE CONTRAST WITH ITS PML SIBLINGS IS THE POINT: on
#: ``cuda_kernels/complex_fused_electric_pair.py``'s cell 15 of 16 rows declare an
#: ELECTRIC source, so there the flag IS the product. Two products, one seam, opposite
#: answers, each read off its own cell rather than off the other's.
ROUTED_WITHOUT_THE_REPAIR: frozenset = frozenset({
    "cuda_kernels/no_pml_complex_fused_electric_pair.py",
    # THE TWO RESIDUAL MAGNETIC WELDS THAT MEASURED FALSE AND STAYED THERE,
    # each off its own cell: every row on both cells declares
    # source_field_types == ['D'] -- the magnetic seam is empty -- so True
    # would buy zero rows while claiming a bracket nothing needs. (The beta and
    # special_kz siblings LEFT this set on 2026-09-02: their False rested on the
    # "EigenModeSource publishes no point index" premise the residue audit
    # re-measured false, and both now sit in WIRED_FOR_THE_REPAIR above.)
    "cuda_kernels/cylindrical_real_fused_magnetic_pair.py",
    "cuda_kernels/bfast_fused_magnetic_pair.py",
    # THE COMPLEX NO-ABSORBER E->P PAIR, 2026-09-02: like its two REAL siblings
    # its seam (update_E -> update_P) carries NO driver injection at all, so
    # there is no deposit to bracket and the module declares the flag False.
    "cuda_kernels/complex_no_pml_fused_polarization_pair.py",
    # THE COMPLEX NO-ABSORBER THREE-SLOT WELD, 2026-09-04: the conjunction of the
    # two entries above, on the same four TestLoadDump 3-D rows (magnetic source
    # only, so the D seam is empty on 4 of 4). Routable through its own
    # FUSED_PAIR_ARMS and FUSED_PRODUCTS rows, and False is the product's whole
    # shape: with no deposit in the span its three sub-steps run in ONE launch
    # per component, which a bracket would forbid (the real three-slot weld's
    # probe measured the single-launch ordering diverging AT the deposit cells).
    "cuda_kernels/complex_no_pml_three_slot_dispersive_weld.py",
    # The two E->P polarization pairs, 2026-09-01: their seam (update_E -> update_P)
    # carries NO driver injection at all — the driver reads and writes nothing between
    # :3313 and :3315 — so there is no deposit to bracket and both modules declare
    # CARRIES_DEPOSIT_REPAIR = False (fused_polarization_pair.py:213). Routable and
    # deliberately unbracketed, which is exactly what this set exists to carry.
    "cuda_kernels/fused_polarization_pair.py",
    # THE TWO STENCIL WELDS, 2026-09-02, and the first entries in this set whose
    # False is a MEASUREMENT of the repair rather than of the seam. Both cells'
    # rows carry an off-diagonal chi1inv row -- the constitutive arms REQUIRE one
    # -- and `deposit_repair.repairable` refuses exactly that by name
    # (deposit_repair.py:686-691), because with an off-diagonal row update_E's
    # output at a repaired cell depends on cells a point repair does not restore.
    # So True was never available, and the seam clause refuses every electric
    # deposit instead. Measured off the census per cell in
    # cuda_kernels/test_offdiag_stencil_welds.py.
    # 2026-09-06, THE H->D WELD. Its seam holds NO INJECTION at all -- the driver
    # injects the magnetic sources one seam earlier and the electric ones one seam
    # later -- so there is no deposit for a bracket to carry and
    # CARRIES_DEPOSIT_REPAIR False is a fact about driver.py rather than a decision.
    # What DOES sit between its two consults is the electric integrated-source
    # withdraw, which meep_gpu/withdraw_hoist.py owns; the product declines to hoist
    # it in this round (HOISTS_THE_WITHDRAW False, because INSTALLABLE False makes
    # _install_fused_pair's hoist branch unreachable for it) and its predicate
    # therefore refuses every row carrying one BY NAME.
    "cuda_kernels/fused_hd_pair.py",
    # ITS TWO CYLINDRICAL SIBLINGS, 2026-09-07: the same seam, the same absence of
    # any injection between the two consults, the same declined hoist
    # (HOISTS_THE_WITHDRAW False) and the same INSTALLABLE False -- so the same
    # reading, one Dcyl grid over. Each predicate refuses a row carrying a standing
    # integrated electric withdraw BY NAME (measured in their gates' withdraw legs,
    # results/cuda_cylindrical_{real_,}fused_hd_pair_2026-09-07).
    "cuda_kernels/cylindrical_real_fused_hd_pair.py",
    "cuda_kernels/cylindrical_fused_hd_pair.py",
    # ITS CARTESIAN COMPLEX SIBLING, 2026-09-07: the same seam and the same reading
    # over complex64 storage, in ONE launch and TWO variants (the plain complex curl
    # and the folded one, selected from the grid). Nothing is injected between the two
    # consults, so CARRIES_DEPOSIT_REPAIR False is again a fact about driver.py; the
    # electric integrated-source withdraw that does sit there is `withdraw_hoist`'s,
    # the product declines to hoist it (HOISTS_THE_WITHDRAW False) and its predicate
    # refuses every row carrying a standing one BY NAME (measured in the gate's
    # withdraw leg, results/cuda_complex_fused_hd_pair_2026-09-07). INSTALLABLE False
    # on the measured composition verdict, so the installer is not reached today.
    "cuda_kernels/complex_fused_hd_pair.py",
    # THE TWO BETA H->D WELDS, 2026-09-07. Same seam, same reason: nothing is
    # INJECTED between the update_H and step_D consults -- the electric injection is
    # one seam later and the magnetic one is one seam earlier -- so there is no
    # deposit to bracket and CARRIES_DEPOSIT_REPAIR is a FACT about the driver rather
    # than a decision. What IS in the seam is the electric withdraw, and
    # withdraw_hoist owns it; both products decline the hoist together with
    # INSTALLABLE and refuse the withdraw rows BY NAME.
    "cuda_kernels/special_kz_fused_hd_pair.py",
    "cuda_kernels/complex_beta_fused_hd_pair.py",
    # ITS CONDUCTIVE/BFAST SIBLING, 2026-09-07: ONE product on the last TWO H->D
    # cells, and the same reading a third time. Nothing is injected between the two
    # consults on either variant, so CARRIES_DEPOSIT_REPAIR False is a fact about
    # driver.py; the electric integrated-source withdraw that DOES sit there is
    # withdraw_hoist's, the product declines to hoist it (HOISTS_THE_WITHDRAW False,
    # because INSTALLABLE False makes _install_fused_pair's hoist branch unreachable)
    # and its predicate refuses every row carrying a standing one BY NAME (measured in
    # the gate's withdraw leg on BOTH variants,
    # results/cuda_conductive_bfast_fused_hd_pair_2026-09-07). WHAT THAT REFUSAL COSTS
    # ON THESE TWO CELLS IS ZERO, and it is measured rather than assumed: both corpus
    # rows record withdraw_in_seam false and
    # n_electric_withdraws_that_do_work 0 (h_to_d_seam_2026-09-04).
    "cuda_kernels/conductive_bfast_fused_hd_pair.py",
    "cuda_kernels/offdiag_fused_electric_pair.py",
    "cuda_kernels/folded_offdiag_fused_electric_pair.py",
    # THEIR TWO COMPLEX64 TWINS, 2026-09-02, closing the LAST two cells this board
    # recorded UNBUILDABLE. The measurement is identical and is a property of the
    # REPAIR rather than of the storage width: `deposit_repair.repairable` refuses
    # an off-diagonal chi1inv row by name, and both cells' constitutive arms
    # REQUIRE one, so True was never available on either. Measured off the census
    # per cell in cuda_kernels/test_complex_offdiag_stencil_welds.py.
    "cuda_kernels/folded_complex_offdiag_fused_electric_pair.py",
    "cuda_kernels/complex_no_pml_offdiag_fused_electric_pair.py",
    # THE OFF-DIAGONAL E->P WELD, 2026-09-02: like the three polarization pairs
    # above it, its seam carries NO driver injection at all, so FUSED_PAIR_SEAMS
    # gives it None and the installer reads the seam as unconditionally empty.
    "cuda_kernels/dispersive_offdiag_fused_polarization_pair.py",
    # THE METAL COMPLEX-BETA MAGNETIC WELD, 2026-09-01, the first METAL entry in
    # this set and the same measurement one backend over: it is installable (its
    # FUSED_PAIR_ARMS row landed with it, so the seam loop reaches
    # _install_fused_pair) and its one corpus row
    # (tests:TestSpecialKz.test_special_kz) declares source_field_types == ['D']
    # — the magnetic seam is empty — so the flag at True would buy exactly zero
    # rows while claiming a bracket this module installs nowhere. The installer
    # always takes its NoopPlan branch here; its ELECTRIC twin carries the flag
    # True on the same row's D seam, two products, one row, opposite answers,
    # each read off its own seam.
    "metal_kernels/beta_complex_fused_magnetic_pair.py",
    # THE FOUR METAL OFF-DIAGONAL SCRATCH WELDS, 2026-09-02. Routable (each holds a
    # METAL_ABSORB_DECLARATIONS row above) and deliberately unbracketed: the flag is
    # forced False by deposit_repair.repairable's by-name refusal of an off-diagonal
    # chi1inv, so 24 rows carrying an in-seam electric source on these cells are
    # served by no fused product on ANY backend. Measured per cell, not assumed.
    "metal_kernels/offdiag_fused_electric_pair.py",
    "metal_kernels/folded_offdiag_fused_electric_pair.py",
    "metal_kernels/complex_no_pml_offdiag_fused_electric_pair.py",
    "metal_kernels/folded_complex_offdiag_fused_electric_pair.py",
    # THE METAL H->D WELD, 2026-09-05, and the first entry here whose seam is the
    # fourth one: update_H -> step_D. Routable (its FUSED_PAIR_ARMS row landed with
    # this entry) and unbracketed because the seam has NO injection in it -- the
    # driver runs the electric integrated-source withdraw between the two consults
    # and nothing else, which is `withdraw_hoist`'s contract rather than
    # `deposit_repair`'s. So the module never consults `seam_source_reasons`, and
    # the installer routes the seam NAME (`withdraw_hoist.SEAM`) to the hoist branch
    # rather than to either repair plan. It also declares INSTALLABLE = False on the
    # measured composition verdict, so the installer is not reached at all today.
    "metal_kernels/fused_hd_pair.py",
    # THE TWO CYLINDRICAL H->D PRODUCTS, 2026-09-07, on the same seam and under the
    # same contract as the entry above: two launches owning update_H and step_D on
    # the Dcyl m=0 real and complex cells, nothing injected between the consults
    # (the electric withdraw is `withdraw_hoist`'s and both refuse a standing one by
    # name), rows in `FUSED_PAIR_ARMS` landed with these entries, and
    # INSTALLABLE = False on the measured composition TIE with the released D->E
    # pair, so the installer is not reached today.
    "metal_kernels/cylindrical_real_fused_hd_pair.py",
    "metal_kernels/cylindrical_complex_fused_hd_pair.py",
    # THE FOLDED AND THE CARTESIAN COMPLEX/BLOCH H->D PRODUCTS, 2026-09-07, on the
    # same seam and under the same contract as the four entries above. Each is ONE
    # dispatch owning update_H and step_D -- on the fold (78 rows, the largest cell
    # on this backend) and on the complex cell (17 rows) -- with nothing injected
    # between the consults, so CARRIES_DEPOSIT_REPAIR False is again a fact about
    # driver.py rather than about either family. On a FOLD that fact is stronger:
    # all three B-side fill/clear/far passes close before the update_H consult and
    # all three D-side ones open after the step_D consult. Rows in FUSED_PAIR_ARMS
    # landed with these entries, and both declare INSTALLABLE = False on a measured
    # arbitration (the released folded_fused_magnetic_pair holds update_H on 78 of
    # 78 folded rows; both neighbouring released complex pairs install on 17 of 17),
    # so the installer is not reached on either today.
    "metal_kernels/folded_fused_hd_pair.py",
    "metal_kernels/complex_fused_hd_pair.py",
    # THE SEAM'S LAST TWO SMALL CELLS, 2026-09-07, and the SAME reading a fourth and
    # fifth time: nothing is injected between the update_H and step_D consults, so
    # there is no deposit for a bracket to carry and CARRIES_DEPOSIT_REPAIR False is
    # a fact about driver.py rather than a decision. Each is routed by its own
    # launch.FUSED_PAIR_ARMS row, declines the electric integrated-source withdraw
    # that DOES sit in the seam (HOISTS_THE_WITHDRAW False, because INSTALLABLE False
    # makes _install_fused_pair's hoist branch unreachable), and refuses every row
    # carrying a standing one BY NAME -- measured in each gate's withdraw leg,
    # results/metal_{conductive,bfast}_fused_hd_pair_2026-09-07/flush.
    "metal_kernels/conductive_fused_hd_pair.py",
    "metal_kernels/bfast_fused_hd_pair.py",
    # THE SEAM'S THREE REMAINING METAL CELLS, 2026-09-08, and the SAME reading a
    # sixth, seventh and eighth time: nothing is injected between the update_H and
    # step_D consults, so CARRIES_DEPOSIT_REPAIR False is a fact about driver.py
    # rather than a decision about these families (beta_complex_fused_hd_pair.py:297,
    # beta_real_fused_hd_pair.py:239, folded_complex_fused_hd_pair.py:246 — each says
    # so on BOTH of its variants, and on a folded variant the statement is stronger
    # still, every fill sitting outside the span). Each is routed by its own
    # launch.FUSED_PAIR_ARMS row, declines the electric integrated-source withdraw
    # that does sit in the seam (HOISTS_THE_WITHDRAW False), and declares
    # INSTALLABLE = False on the arbitration its gate measured through the shipped
    # composer: both neighbouring seams carry Metal products on every variant, so the
    # span is a LOSS where both install and a TIE the released B->H incumbent wins
    # where one does. THE TWO BETA ENTRIES EACH COVER TWO CELLS through one
    # FUSED_PAIR_ARMS row plus one FUSED_PAIR_EXTRA_ARMS row.
    "metal_kernels/beta_complex_fused_hd_pair.py",
    "metal_kernels/beta_real_fused_hd_pair.py",
    "metal_kernels/folded_complex_fused_hd_pair.py",
    # THE THREE METAL FAMILIES THE ALL-PATHS ROUND ROUTED, 2026-09-17. Each was
    # CERTIFIED with a released weld and no `launch.FUSED_PAIR_ARMS` row, so the seam
    # loop could never ask it; the rows landed at metal_kernels/launch.py:1234-1246
    # and these entries are their other half. The flag was already False in each
    # module and is the family's own measurement, not a wiring choice:
    # - the two Dcyl B->H pairs (cylindrical_fused_magnetic_pair.py:277,
    #   cylindrical_real_fused_magnetic_pair.py:227) sit where the driver injects a
    #   MAGNETIC source between the consults (driver.py:3293-3294); their corpus cost
    #   is ZERO because every one of their rows (sixteen and three) declares
    #   electric sources only (cylindrical_fused_magnetic_pair.py:705,
    #   cylindrical_real_fused_magnetic_pair.py:522-525);
    # - complex_conductive_fused_pair.py:229 sits on D->E, where the injection is
    #   ELECTRIC (driver.py:3318-3322), and its four corpus rows are magnetic-sourced
    #   (complex_conductive_fused_pair.py:570-577).
    # (driver.py lines read 2026-09-19; the three modules' own comments still cite
    # the pre-move :3283-3284 and :3294-3299.)
    # All three pass the flag to `deposit_repair.seam_source_reasons` (:721, :539,
    # :589), which refuses an in-seam deposit BY NAME and refuses an undeclared source
    # list rather than inferring an empty one. KEYED BY MODULE PATH: family
    # `cylindrical_complex_fused_magnetic_pair` lives in
    # `cylindrical_fused_magnetic_pair.py`, the stem trap its electric twin set.
    "metal_kernels/complex_conductive_fused_pair.py",
    "metal_kernels/cylindrical_fused_magnetic_pair.py",
    "metal_kernels/cylindrical_real_fused_magnetic_pair.py",
    # ---------------------------------------------------------------------
    # THE SIX TRITON MODULES THE INSTALLER WAVE MADE ROUTABLE, 2026-09-02.
    # ---------------------------------------------------------------------
    # Each holds a row in ``triton_kernels/launch.CERTIFIED_FUSED_PAIR_ARMS``, so
    # the seam loop can install it; each ships ``CARRIES_DEPOSIT_REPAIR = False``,
    # so the installer always takes its ``NoopPlan`` branch. That is exactly the
    # pair of declarations this set exists to spell — "installable, deliberately
    # unbracketed" — and it is the same pair the Metal off-diagonal welds carry.
    # The flag is the FAMILY's own measurement in every case (each predicate
    # refuses an in-seam source by name rather than carrying one), so a flip here
    # is a family-gate event and not a wiring one.
    "triton_kernels/beta_fused_magnetic_pair.py",
    "triton_kernels/bfast_fused_magnetic_pair.py",
    "triton_kernels/complex_conductive_fused_pair.py",
    "triton_kernels/cylindrical_fused_magnetic_pair.py",
    "triton_kernels/cylindrical_real_fused_magnetic_pair.py",
    "triton_kernels/nonlinear_fused_magnetic_pair.py",
    # THE TWO TRITON CYLINDRICAL H->D PRODUCTS, 2026-09-07, on the same seam and
    # under the same contract as the two Metal entries above: two launches and one
    # array-module scan owning update_H and step_D on the Dcyl m=0 real and complex
    # cells, nothing injected between the consults (the electric withdraw is
    # `withdraw_hoist`'s, and both refuse a standing one by name), rows in
    # `launch.CERTIFIED_FUSED_PRODUCTS` and `CERTIFIED_FUSED_PAIR_ARMS` landed with
    # these entries, and INSTALLABLE = False on the measured arbitration (the
    # composer's `4 - (installed pairs)` rule prices the 7 -> 3 launch saving as a
    # loss), so `_declared_uninstallable` refuses each by name before the seam loop
    # reaches the installer.
    "triton_kernels/cylindrical_real_fused_hd_pair.py",
    "triton_kernels/cylindrical_fused_hd_pair.py",
    # THE TRITON CARTESIAN COMPLEX H->D PRODUCT, 2026-09-08, on the same seam and
    # under the same contract as the two entries above, one grid over: its rows in
    # `launch.CERTIFIED_FUSED_PRODUCTS` and `CERTIFIED_FUSED_PAIR_ARMS`
    # (`('complex', 'complex PML')`) landed with this entry, so the seam loop reaches
    # it; `CARRIES_DEPOSIT_REPAIR = False` (complex_fused_hd_pair.py:292) is again a
    # fact about driver.py -- nothing is injected between the update_H consult
    # (:3311) and the step_D consult (:3315) -- and the electric integrated-source
    # withdraw that DOES sit there is `withdraw_hoist`'s, which the product declines
    # to hoist (HOISTS_THE_WITHDRAW False) while refusing every row carrying a
    # standing one by name. INSTALLABLE = False on a per-row measurement: all TWELVE
    # driven corpus rows price as a LOSS (results/triton_complex_fused_hd_pair_
    # 2026-09-07/keep/gate.json, `arbitration_over_the_driven_rows`), so
    # `_declared_uninstallable` refuses it before the installer is reached.
    "triton_kernels/complex_fused_hd_pair.py",
    # THE TWO TRITON OFF-DIAGONAL SCRATCH WELDS, routed by their
    # `launch.CERTIFIED_FUSED_PRODUCTS` rows once `ScratchWeldPairPlan.warm()` stopped
    # driving the rotation at plan time, and unbracketed for the same measured reason as
    # their CUDA and Metal twins above: `deposit_repair.repairable` refuses an
    # off-diagonal chi1inv row by name, so each module declares
    # `CARRIES_DEPOSIT_REPAIR = False` and its predicate refuses an in-seam electric
    # source by name rather than carrying one.
    "triton_kernels/offdiag_fused_electric_pair.py",
    "triton_kernels/folded_offdiag_fused_electric_pair.py",
})


def test_only_the_wired_products_have_declared_that_they_carry_the_repair():
    """THE GATE ON ALL OF THE ABOVE. A product may claim the repair only once something
    actually brackets its launch. This test is what makes a stray flip loud.

    IT IS AN ALLOWLIST, NOT A COUNT. Naming the two wired modules means a flip anywhere
    else fails here with the module that did it, and it means DROPPING the flip on a
    wired product fails here too -- so neither direction can drift silently.

    READ FROM SOURCE, NOT BY IMPORT: several of these modules import Triton or Metal at
    module scope, so an importing version of this test would skip exactly the hosts where
    a stray flip is most likely to go unnoticed.
    """
    import ast

    here = pathlib.Path(__file__).parent
    declared = set()
    for path in sorted(here.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "CARRIES_DEPOSIT_REPAIR" not in names:
                continue
            if not (isinstance(node.value, ast.Constant) and node.value.value is False):
                declared.add(str(path.relative_to(here)))
    assert declared == set(WIRED_FOR_THE_REPAIR), (
        f"declared={sorted(declared)} but the wired set is {sorted(WIRED_FOR_THE_REPAIR)}. "
        f"A product outside the wired set declares CARRIES_DEPOSIT_REPAIR without a "
        f"LeadingRepairPlan/TrailingRepairPlan bracketing its launch, or a wired one has "
        f"dropped the declaration; see deposit_repair for why the two change together")


def test_every_wired_product_is_installed_through_the_one_installer():
    """The allowlist is not a list of names someone may extend by typing.

    ``_install_fused_pair`` is the only constructor of the two repair plans -- one copy
    per composer, Triton's and Metal's -- so a module that claims the repair must be
    reachable from a call to it. This reads the planners' source and pins that both
    wired modules are installed there, which is what would fail if a future refactor
    moved one of them onto a path that builds a bare ``FusedPairPlan`` and a
    ``NoopPlan`` instead.
    """
    import ast

    here = pathlib.Path(__file__).parent
    launch = (here / "triton_kernels" / "launch.py").read_text(encoding="utf-8")
    installs = [node for node in ast.walk(ast.parse(launch))
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "_install_fused_pair"]
    assert len(installs) == 4, (
        f"{len(installs)} calls to _install_fused_pair; the Triton composer installs "
        f"the ordinary pairs, the dispersive product, — since 2026-08-27 — the two "
        f"folded pairs from `_install_folded_fused_pairs`, and — since 2026-09-02 — "
        f"the certified products from `_install_certified_fused_products`'s seam loop")
    # The ordinary pairs go in through the FUSED_PAIRS loop, the dispersive product
    # through its own branch, the folded pairs through the fold branch's own loop.
    # Every builder is named in this file and nowhere else.
    assert "plan_fused_pair(" in launch and "plan_dispersive_fused_pair(" in launch
    assert "plan_folded_fused_magnetic_pair" in launch
    assert "plan_folded_fused_pair" in launch
    for module_path in WIRED_FOR_THE_REPAIR:
        text = (here / module_path).read_text(encoding="utf-8")
        assert "CARRIES_DEPOSIT_REPAIR = True" in text, module_path
    # A DECLARED-BUT-UNROUTED MODULE IS PINNED AS UNROUTED, not waved through. The
    # danger it carries is the opposite of the usual one: it claims a bracket, and
    # if a composer started building it without going through `_install_fused_pair`
    # the claim would become false silently. So the absence of a builder is asserted
    # here, in the same place the presence of one is asserted for everything else.
    for module_path in DECLARED_BUT_NOT_ROUTED:
        stem = module_path.split("/", 1)[1][: -len(".py")]
        assert f"plan_{stem}" not in launch, (
            f"{module_path} is in DECLARED_BUT_NOT_ROUTED but triton_kernels/launch.py "
            f"names plan_{stem}; either route it through _install_fused_pair and move "
            f"it out of that set, or it is claiming a bracket nothing builds")
    # And the only places either plan class is constructed are the installers. There
    # are TWO composers in the tree and each owns one: the Metal one landed its own
    # `_install_fused_pair` with the same shape and the same two slots, so a Metal
    # family that flips the flag has a bracket to flip it against. What this
    # assertion keeps is the invariant it always kept -- that the plan classes are
    # built by an installer and nowhere else -- so a family module that constructed
    # one directly, bypassing the two-slot protocol, still fails here by name.
    constructors = [path for path in sorted(here.rglob("*.py"))
                    if not path.name.startswith("test_")
                    and "RepairPlan(" in path.read_text(encoding="utf-8")]
    assert [str(path.relative_to(here)) for path in constructors] == [
        "cuda_kernels/fused_pairs.py", "metal_kernels/launch.py",
        "triton_kernels/launch.py"], constructors
    metal = (here / "metal_kernels" / "launch.py").read_text(encoding="utf-8")
    metal_installs = [node for node in ast.walk(ast.parse(metal))
                      if isinstance(node, ast.Call)
                      and getattr(node.func, "id", None) == "_install_fused_pair"]
    assert len(metal_installs) == 1, (
        f"{len(metal_installs)} calls to the Metal _install_fused_pair; the seam loop "
        f"in `_install_fused_pairs` is the one caller")
    # AND THE LOOP'S REACH, which is what a call count cannot show. The Metal installer
    # is called once from inside a loop over every registered weld, so "one call" says
    # nothing about WHICH families can arrive there. `FUSED_PAIR_ARMS` is the bound:
    # a family with no row is refused before its predicate is asked, so the wired Metal
    # set can only ever be the families named in it.
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    assert tuple(sorted(metal_launch.FUSED_PAIR_ARMS)) == METAL_ABSORB_DECLARATIONS, (
        f"metal_kernels.launch.FUSED_PAIR_ARMS is "
        f"{sorted(metal_launch.FUSED_PAIR_ARMS)}; a family gaining a row there gains "
        f"the ability to be bracketed, and a family losing one loses it while its flag "
        f"stays True")
    # THE FAMILY NAME, READ FROM THE MODULE, NOT THE MODULE STEM. `FUSED_PAIR_ARMS`
    # is keyed by `FAMILY`, and the two DIVERGE on the Dcyl complex products: module
    # `cylindrical_fused_electric_pair.py`, family
    # `cylindrical_complex_fused_electric_pair`. Keyed by the stem, such a module
    # reads as "declares the repair and the seam loop cannot install it" — the exact
    # defect this assertion exists to catch, reported about a family that HAS a row.
    # `build_fusion_matrix._wiring()` carries the same correction for the same module
    # and records that keying by the wrong name once disabled a check silently.
    def _family_of_module(path):
        text = (here / path).read_text(encoding="utf-8")
        declared = [node.value.value for node in ast.walk(ast.parse(text))
                    if isinstance(node, ast.Assign)
                    and any(getattr(target, "id", None) == "FAMILY"
                            for target in node.targets)
                    and isinstance(node.value, ast.Constant)]
        assert len(declared) == 1, (path, declared)
        return declared[0]

    metal_wired = {_family_of_module(path) for path in WIRED_FOR_THE_REPAIR
                   if path.startswith("metal_kernels/")}
    # THE DIRECTION THAT IS A DEFECT stays exact: a module declaring the repair that
    # the seam loop cannot install would run unbracketed. The other direction — a
    # routable family that deliberately does not carry the repair — is enumerated
    # rather than tolerated, so it cannot grow a family nobody decided about.
    assert metal_wired <= set(METAL_ABSORB_DECLARATIONS), (
        f"the Metal modules declaring the repair are {sorted(metal_wired)} but the "
        f"families the seam loop can install are {sorted(METAL_ABSORB_DECLARATIONS)}")
    # THE REVERSE MAP goes family -> module by ASKING each module, for the same
    # reason: `metal_kernels/{family}.py` is the right path for 38 of the 40 absorb
    # families and the wrong one for the two Dcyl complex pairs -- the electric one,
    # and since 2026-09-17 its magnetic twin (measured 2026-09-19).
    _module_of_family = {
        _family_of_module(f"metal_kernels/{path.name}"): f"metal_kernels/{path.name}"
        for path in sorted((here / "metal_kernels").glob("*.py"))
        if "\nFAMILY = " in path.read_text(encoding="utf-8")}
    metal_routed_only = {_module_of_family[name]
                         for name in set(METAL_ABSORB_DECLARATIONS) - metal_wired}
    assert metal_routed_only == {path for path in ROUTED_WITHOUT_THE_REPAIR
                                 if path.startswith("metal_kernels/")}, (
        sorted(metal_routed_only))
    # ...and each of them really does ship the flag False, which is what makes the
    # complement a declaration rather than an omission.
    for module_path in ROUTED_WITHOUT_THE_REPAIR:
        text = (here / module_path).read_text(encoding="utf-8")
        assert "CARRIES_DEPOSIT_REPAIR = False" in text, module_path

    # THE THIRD COMPOSER, 2026-08-28. Read exactly as the Metal one is: a call count
    # bounds the copies of the protocol, and the product table bounds which families
    # can arrive at it. The CUDA block loops over its OWN table rather than over the
    # arm table (every hand-CUDA family module imports CuPy at module scope, so an
    # eager arm-table row would be unbuildable on the backend-free host the census is
    # cut on), so `FUSED_PRODUCTS` and `FUSED_PAIR_ARMS` together are the bound.
    cuda = (here / "cuda_kernels" / "fused_pairs.py").read_text(encoding="utf-8")
    cuda_installs = [node for node in ast.walk(ast.parse(cuda))
                     if isinstance(node, ast.Call)
                     and getattr(node.func, "id", None) == "_install_fused_pair"]
    assert len(cuda_installs) == 1, (
        f"{len(cuda_installs)} calls to the CUDA _install_fused_pair; the seam loop "
        f"in `install_fused_pairs` is the one caller")
    # THE THREE-SLOT INSTALLER, 2026-09-02, bounded the same way and for the same
    # reason. It is a SECOND constructor of the two repair plans, which is exactly
    # what the count above exists to keep rare: it wraps the LEADING half of a
    # three-slot product and then writes the trailing half into `update_P`, and
    # that ordering -- repair between the two device groups -- is the product. A
    # second caller would be a second place the two could be ordered differently,
    # and the wrong order computes and is wrong only at the deposit cells.
    cuda_triple_installs = [node for node in ast.walk(ast.parse(cuda))
                            if isinstance(node, ast.Call)
                            and getattr(node.func, "id", None)
                            == "_install_fused_triple"]
    assert len(cuda_triple_installs) == 1, (
        f"{len(cuda_triple_installs)} calls to the CUDA _install_fused_triple; the "
        f"seam loop in `install_fused_pairs` is the one caller")

    from meep_gpu.cuda_kernels import fused_pairs as cuda_fused  # noqa: PLC0415

    assert tuple(sorted(cuda_fused.FUSED_PAIR_ARMS)) == CUDA_ABSORB_DECLARATIONS, (
        f"cuda_kernels.fused_pairs.FUSED_PAIR_ARMS is "
        f"{sorted(cuda_fused.FUSED_PAIR_ARMS)}; a family gaining a row there gains "
        f"the ability to be bracketed, and a family losing one loses it while its "
        f"flag stays True")
    # A product in the block's own table with no absorb row would be REFUSED by name
    # rather than bracketed, so the two tables may not drift apart either.
    assert set(cuda_fused.FUSED_PRODUCTS) == set(cuda_fused.FUSED_PAIR_ARMS), (
        f"FUSED_PRODUCTS={sorted(cuda_fused.FUSED_PRODUCTS)} but "
        f"FUSED_PAIR_ARMS={sorted(cuda_fused.FUSED_PAIR_ARMS)}")
    # And the module each CUDA row reaches is a module that has DECIDED about the flag.
    # Pinned per row rather than for the first one: with seven products a single
    # spot-check would leave six rows free to point at a module that decided nothing,
    # which is the one direction that puts an unbracketed launch on a live seam.
    #
    # THE PARTITION IS EXACT IN BOTH DIRECTIONS since 2026-08-31, and the split is the
    # SAME one the Metal branch above already makes: a routable family either carries
    # the repair (`WIRED_FOR_THE_REPAIR`) or deliberately does not
    # (`ROUTED_WITHOUT_THE_REPAIR`), and a family in NEITHER set is a module nobody
    # decided about. Before this the check was one-directional membership in the wired
    # set, which had no way to express the second case at all.
    cuda_modules = {family: f"cuda_kernels/{cuda_fused.FUSED_PRODUCTS[family]['module']}.py"
                    for family in CUDA_ABSORB_DECLARATIONS}
    for family, module_path in cuda_modules.items():
        assert module_path in WIRED_FOR_THE_REPAIR or \
            module_path in ROUTED_WITHOUT_THE_REPAIR, (
                f"{family} routes to {module_path}, which is in neither the wired set "
                f"nor the routed-without-the-repair set; the CUDA seam loop would "
                f"reach a module that has decided nothing about the bracket")
    cuda_routed_only = {path for path in cuda_modules.values()
                        if path not in WIRED_FOR_THE_REPAIR}
    assert cuda_routed_only == {path for path in ROUTED_WITHOUT_THE_REPAIR
                                if path.startswith("cuda_kernels/")}, (
        f"the CUDA modules the seam loop can install that do NOT carry the repair are "
        f"{sorted(cuda_routed_only)}, but ROUTED_WITHOUT_THE_REPAIR names "
        f"{sorted(path for path in ROUTED_WITHOUT_THE_REPAIR if path.startswith('cuda_kernels/'))}")
    assert cuda_fused.FUSED_PRODUCTS["cuda_fused_magnetic_pair"]["module"] == \
        "fused_magnetic_pair"


def _executable_text(path) -> str:
    """A module's source with comments and docstrings removed.

    THE PACKAGE ALREADY WROTE THIS RULE DOWN -- `test_triton_kernels.code_of` and the
    two cylindrical pair tests each carry it, over the shared primitive
    `code_identity.strip_docstrings` -- because the prose in these modules NAMES the
    things they must not touch, so an ownership check that greps the RAW file fires on
    its own documentation.
    """
    import ast as _ast

    from .code_identity import strip_docstrings

    return _ast.unparse(strip_docstrings(_ast.parse(
        pathlib.Path(path).read_text(encoding="utf-8"))))


def test_every_fused_pair_predicate_carries_the_flag_the_gate_reads():
    """A product that dropped the flag would silently stop being covered by the test
    above, which is the failure mode that makes a gate-on-a-constant worthless."""
    import ast

    here = pathlib.Path(__file__).parent
    missing, consumers = [], []
    for path in sorted(here.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        # EXECUTABLE TEXT, NOT THE RAW FILE, for the reason `_executable_text` above
        # gives and this test measured on 2026-09-21: `dispatch_preference.py` NAMES
        # `deposit_repair.seam_source_reasons(..., 'B'/'D')` in its docstring, to say
        # what the eventual wiring site in `fastpath.py` will call -- it consults
        # nothing and holds no product. A raw grep read that prose as a consult and
        # demanded a fused-pair flag from an arbitration module, which is the false
        # positive that makes an ownership check stop meaning what it says.
        text = _executable_text(path)
        if "seam_source_reasons(" not in text or "def seam_source_reasons(" in text:
            continue   # the module that DEFINES the clause is not a consumer of it
        consumers.append(str(path.relative_to(here)))
        if "CARRIES_DEPOSIT_REPAIR" not in text:
            missing.append(str(path.relative_to(here)))
    assert not missing, f"{missing} consult the seam clause without declaring the flag"
    # A scan that found nothing would pass the assertion above and prove nothing. 23 was
    # the count the delegation landed with: 10 Triton predicates, 12 Metal ones and the
    # CUDA one. Each flip since has added exactly one, and always for the same reason --
    # a product that INHERITED its seam verdict from a predicate that refused every
    # in-seam source can no longer inherit it once that predicate starts admitting, so
    # it asks the clause itself with its own (False) declaration:
    #
    #   +2 = 25  triton_kernels/nonlinear_fused_magnetic_pair.py and
    #            triton_kernels/fused_dispersive_chain.py, when coverage.py and
    #            dispersive_fused_pair.py flipped;
    #   +1 = 26  metal_kernels/nonlinear_fused_magnetic_pair.py, when
    #            metal_kernels/fused_magnetic_pair.py flipped. It borrows that family's
    #            whole predicate and holds no row in launch.FUSED_PAIR_ARMS, so nothing
    #            would bracket its launch.
    #
    #   +1 = 27  metal_kernels/fused_electric_pair.py, the plain D/E pair built on
    #            2026-08-28. It is a NEW PRODUCT asking the clause, not a borrower that
    #            stopped inheriting it -- the first entry on this list of that kind
    #            since the delegation landed.
    #
    #   +9 = 36  2026-08-30, and all nine are of that second kind: NEW PRODUCTS, one
    #            module each, every one asking the clause in its own predicate.
    #            Seven Metal -- beta_fused_electric_pair, complex_fused_electric_pair,
    #            folded_complex_fused_pair, folded_beta_complex_fused_pair,
    #            folded_beta_real_fused_magnetic_pair, folded_beta_real_fused_pair,
    #            folded_fused_dispersive_pair -- and two CUDA,
    #            complex_fused_magnetic_pair and cylindrical_fused_magnetic_pair.
    #            THE SAME ROUND'S TWO CLAUSE DISCHARGES ARE NOT HERE and must not be:
    #            folded_complex_fused_magnetic_pair and
    #            folded_beta_complex_fused_magnetic_pair were already consulting the
    #            clause with False and only flipped the constant they pass, so they
    #            moved the wired set without moving this count. A round that grew both
    #            by the same number would have been the tell that a discharge was
    #            miscounted as a product.
    #
    #   +1 = 37  2026-08-30, and NOT this round's work: the TRITON track landed
    #            `triton_kernels/complex_fused_electric_pair.py`, the D->E complex
    #            pair, while the Metal round above was closing. It is a new product
    #            asking the clause in its own predicate, so it counts here; it is
    #            UNROUTED, so it is asserted through DECLARED_BUT_NOT_ROUTED rather
    #            than by an installer call. EXPECT THIS COUNT TO KEEP MOVING while
    #            that track lands the rest of its D->E families -- each landing is
    #            one more consumer, and this assertion going red is the designed
    #            signal that it did, not a defect in either round.
    #
    #   +4 = 41  2026-08-30, and they arrive from TWO tracks at once, which is why
    #            they are named per track rather than counted together:
    #
    #            TRITON, this round's fusion work --
    #              triton_kernels/cylindrical_fused_electric_pair.py, the Dcyl D->E
    #                weld (the board's largest single cell, 16 seam-instances), which
    #                asks the clause with CARRIES_DEPOSIT_REPAIR True; and
    #              triton_kernels/complex_conductive_fused_pair.py, the complex
    #                conductive no-PML D->E weld, which asks it with the flag FALSE
    #                and is served in full anyway -- all four of its corpus rows
    #                declare a MAGNETIC source.
    #
    #            CUDA, landed concurrently by that track --
    #              cuda_kernels/fused_electric_pair.py and cuda_kernels/fused_pairs.py.
    #
    #            WHAT THIS COUNT ASSERTS IS A TREE FACT AND NOTHING MORE: that a
    #            predicate consults the shared clause and declares the flag the gate
    #            reads. It is NOT a statement that a family is wired, bracketed or
    #            gated -- those are `WIRED_FOR_THE_REPAIR`, the installer call count
    #            and the arm tables above, and a family that is missing from one of
    #            THOSE still fails there. Raising this number does not settle any of
    #            them for either track.
    #
    # Raise it when a product is added; a DROP means a predicate stopped asking.
    #
    # +1 = 42 on 2026-08-31: triton_kernels/folded_dispersive_fused_pair.py, the
    # `(folded PML, folded dispersive)` D->E weld. It asks the clause with
    # CARRIES_DEPOSIT_REPAIR True, and on that cell the flag is the product: all four
    # of its corpus rows declare an ELECTRIC source, so the same module with the flag
    # at False would serve nothing at all.
    #
    # +1 = 43 on 2026-08-31: triton_kernels/cylindrical_real_fused_electric_pair.py,
    # the Dcyl m = 0 `(cylindrical PML, cylindrical)` D->E weld. It asks the clause
    # with CARRIES_DEPOSIT_REPAIR True, and on that cell the flag is the product:
    # all three of its corpus rows declare an ELECTRIC source.
    #
    # +1 = 45 on 2026-08-31: metal_kernels/cylindrical_fused_electric_pair.py, the
    # Dcyl |m| >= 1 COMPLEX D->E weld and the largest cell on the Metal board (16
    # seam-instances). Same seam, same flag and the same coefficient pack as the row
    # below; the two landed in one change because one technique cleared both.
    #
    # +1 = 44 on 2026-08-31: metal_kernels/cylindrical_real_fused_electric_pair.py,
    # the METAL sibling of the row directly above -- the same seam, the same three
    # corpus rows, the same flag, and the same reason the flag is the product there.
    # It is the first product on this board recovered from the BINDING ceiling: its
    # unpacked signature needs 31 pointers against a 30-pointer ceiling, and packing
    # the curl half's six read-only PML coefficient vectors into one buffer
    # (`metal_kernels/coefficient_pack.py`) is what let it exist at all.
    #
    # +1 = 46 on 2026-08-31: triton_kernels/folded_complex_fused_pair.py, the
    # `(folded complex PML, folded complex)` D->E weld. It asks the clause with
    # CARRIES_DEPOSIT_REPAIR True, and on that cell the flag is the product: BOTH of
    # its corpus rows declare an ELECTRIC source, so the same module with the flag at
    # False would serve nothing at all.
    #
    # +1 = 47 on 2026-08-31: triton_kernels/beta_fused_electric_pair.py, the
    # `(real beta PML, real beta run)` D->E weld. It asks the clause with
    # CARRIES_DEPOSIT_REPAIR True, and on that cell the flag is the product: its
    # single corpus row declares an ELECTRIC source.
    #
    # +1 = 48 on 2026-08-31: triton_kernels/bfast_fused_electric_pair.py, the
    # `(BFAST PML, BFAST run)` D->E weld. It asks the clause with
    # CARRIES_DEPOSIT_REPAIR True, and on that cell the flag is the product.
    #
    # +1 = 49 on 2026-08-31: triton_kernels/conductive_fused_electric_pair.py, the
    # `(conductive PML, ordinary)` D->E weld. Same clause, same flag, same reason:
    # its single corpus row declares an electric source among its two.
    #
    # +2 = 51 on 2026-08-31, and BOTH from the CUDA track rather than from Triton:
    # cuda_kernels/complex_fused_electric_pair.py and
    # cuda_kernels/cylindrical_fused_electric_pair.py, the two complex D->E welds
    # that make that track's `step_D` a three-candidate seam. Each is a NEW PRODUCT
    # asking the clause in its own predicate with `carries_repair=CARRIES_DEPOSIT_REPAIR`
    # -- read back from the call itself, not from the module's stem -- so each counts
    # here exactly once. THE COUNT GREW BY TWO WHILE `cuda_kernels/fused_pairs.py`
    # STAYED PUT, which is the arithmetic tell that this round added products and not
    # an installer: that file is the CUDA block's composer and has been a consumer
    # since its own electric row landed.
    # +1 = 52 on 2026-08-31: triton_kernels/no_pml_fused_electric_pair.py, the
    # `(conductive no-PML | no-PML, no-PML stored E)` D->E weld -- and the FIRST
    # consumer to ask the clause with a `repair_paths=` argument. It asks with
    # CARRIES_DEPOSIT_REPAIR True and REPAIR_PATHS = (deposit_repair.PLAIN_PATH,),
    # because both its halves refuse an active absorber and the recurrence its seam
    # inverts is `update_E`'s plain overwrite rather than the split-field one every
    # other consumer on this list inverts. On this cell the flag is the product: all
    # THREE corpus rows declare an ELECTRIC source, so the same module with the flag
    # at False would serve nothing at all.
    # +1 = 53 on 2026-08-31: cuda_kernels/no_pml_complex_fused_electric_pair.py, the
    # `(complex no-PML curl, complex no-PML stored E)` D->E weld and the first CUDA
    # product to refuse an absorber rather than require one. It asks the clause with
    # CARRIES_DEPOSIT_REPAIR FALSE, and on THIS cell the flag at False costs nothing
    # and buys the whole product: all four of its corpus rows declare
    # `source_field_types == ['B']`, a magnetic source injected one seam earlier, so
    # the D seam is empty on every row it serves. Its two CUDA electric siblings on the
    # count above answer the opposite way on their own cells, which is what makes each
    # a measurement rather than a convention.
    # +2 = 55, owed by the 2026-08-31 folded-beta round and recorded here on
    # 2026-09-01: triton_kernels/folded_beta_fused_electric_pair.py and
    # triton_kernels/folded_beta_fused_magnetic_pair.py. Both are NEW MODULES of
    # that round — their `WIRED_FOR_THE_REPAIR` and `DECLARED_BUT_NOT_ROUTED`
    # entries landed with them (see "THE TWO TRITON FOLDED-BETA PAIRS" above) but
    # neither count ledger gained its increment, so this count read 2 stale from
    # that evening until this entry. Each asks the clause once in its own
    # predicate; the electric one answers 'D' and the magnetic one 'B', the same
    # two-seams-one-cell pairing their single corpus row
    # (tests:TestSpecialKz.test_eigsrc_kz_1_real_imag) drives.
    # +2 = 57 on 2026-09-01: metal_kernels/no_pml_fused_electric_pair.py and
    # metal_kernels/no_pml_conductive_fused_electric_pair.py, the two Metal welds
    # over the no-absorber stored-E D->E cells and the second and third consumers
    # to ask the clause with a `repair_paths=` argument (the PLAIN path alone,
    # exactly as the Triton twin at 52 does, and for the same reason: both halves
    # refuse an active absorber, so the recurrence their seam inverts is
    # `update_E`'s plain overwrite). Each asks in its own predicate with
    # `carries_repair=CARRIES_DEPOSIT_REPAIR` read from the call itself, so each
    # counts here exactly once; the conductive one asks a SECOND question beside
    # the clause (`_scaled_conductive_injection_reasons`, the whole-volume condinv
    # rescale) which is its own function, not a second consumer of this seam
    # clause — the plain family imports and consults that clause from the
    # conductive module, one home, no extra call site.
    # +5 = 62 on 2026-09-02: the five RESIDUAL CUDA magnetic welds, all NEW
    # PRODUCTS each asking the clause in its own predicate --
    # cuda_kernels/complex_folded_fused_magnetic_pair.py (flag True, the one
    # cell with a repairable point B deposit,
    # TestHoleyWvgBands.test_fields_at_kx), and
    # cuda_kernels/complex_beta_fused_magnetic_pair.py,
    # cuda_kernels/cylindrical_real_fused_magnetic_pair.py,
    # cuda_kernels/special_kz_fused_magnetic_pair.py,
    # cuda_kernels/bfast_fused_magnetic_pair.py (all four flag FALSE, measured
    # off their cells: every clearing row is electric-only sourced and the only
    # B-sourced rows deposit through EigenModeSource, which publishes no point
    # index). cuda_kernels/complex_fill_carry.py holds no seam_source_reasons
    # call and correctly does not count.
    # +4 = 66 on 2026-09-01: the four METAL RESIDUE welds, all NEW PRODUCTS each
    # asking the clause once in its own predicate —
    # metal_kernels/bfast_fused_electric_pair.py,
    # metal_kernels/conductive_fused_electric_pair.py and
    # metal_kernels/beta_complex_fused_electric_pair.py answer 'D' (flag True,
    # every reachable row electric-sourced), and
    # metal_kernels/beta_complex_fused_magnetic_pair.py answers 'B' (flag False
    # on its electric-only cell). The conductive one's second question
    # (_scaled_conductive_injection_reasons, the now-lifted rescale clause) is
    # imported from the no-PML conductive module — one home, no extra call site
    # of THIS clause.
    # +2 = 68 on 2026-09-01: the two TRITON COMPLEX-BETA welds, both NEW
    # PRODUCTS each asking the clause once in its own predicate —
    # triton_kernels/complex_beta_fused_electric_pair.py answers 'D' (flag True;
    # the cell's one row, tests:TestSpecialKz.test_special_kz, is
    # electric-sourced in that seam) and
    # triton_kernels/complex_beta_fused_magnetic_pair.py answers 'B' (flag True
    # in its siblings' shape; the same row's B seam is empty).
    # +5 = 73 on 2026-09-02: the five CUDA ELECTRIC TWINS of the residue round,
    # all NEW PRODUCTS each asking the clause once in its own predicate and
    # every one answering 'D' with the flag True --
    # cuda_kernels/complex_beta_fused_electric_pair.py,
    # cuda_kernels/complex_folded_fused_electric_pair.py,
    # cuda_kernels/cylindrical_real_fused_electric_pair.py,
    # cuda_kernels/special_kz_fused_electric_pair.py and
    # cuda_kernels/bfast_fused_electric_pair.py. On every one the flag is the
    # product: every row of every twin's board cell declares an ELECTRIC source
    # inside the D seam (measured off the residualwelds census). The SAME
    # ROUND'S TWO CUDA FLAG FLIPS (complex_beta and special_kz magnetic pairs)
    # are deliberately NOT on this count: both were already consulting the
    # clause with False and only flipped the constant they pass, the same
    # discharge shape as Metal's 2026-08-30 pair. The round's other two new
    # modules hold no seam_source_reasons call and correctly do not count:
    # cuda_kernels/complex_electric_fill_carry.py is a carry, and
    # cuda_kernels/complex_no_pml_fused_polarization_pair.py sits on the E->P
    # seam, which carries no injection for the clause to ask about.
    # 73 -> 75 on 2026-09-02: the two Triton FOLDED complex-beta pairs
    # (triton_kernels/folded_beta_complex_fused_pair.py and its magnetic twin),
    # each consulting the clause with CARRIES_DEPOSIT_REPAIR=True from the first
    # cut — the folded half of the no-admitting-arm residue (audit §1.4).
    # 75 -> 83 on 2026-09-02, THE STENCIL ROUND, and the eight are three lanes'
    # welds on the same two off-diagonal D_to_E cells, landed into one tree:
    # cuda_kernels/{offdiag,folded_offdiag}_fused_electric_pair.py,
    # metal_kernels/{offdiag,folded_offdiag,complex_no_pml_offdiag,
    # folded_complex_offdiag}_fused_electric_pair.py and
    # triton_kernels/{offdiag,folded_offdiag}_fused_electric_pair.py. EVERY ONE
    # consults the clause with CARRIES_DEPOSIT_REPAIR=FALSE, and on these cells
    # that is a MEASUREMENT rather than a default: `deposit_repair.repairable`
    # refuses an off-diagonal chi1inv row BY NAME (deposit_repair.py:686-691),
    # and every row of both cells has one installed because the constitutive arm
    # REQUIRES it. So these products consult the clause exactly to refuse every
    # electric deposit, which is what puts them on this count.
    # 83 -> 85 on 2026-09-02, THE CONDUCTIVE-CURL ROUND: the two CUDA welds that
    # close the last POINTWISE-BUILDABLE D_to_E cells on that board --
    # cuda_kernels/no_pml_dispersive_fused_electric_pair.py and
    # cuda_kernels/conductive_fused_electric_pair.py. Both consult the clause with
    # CARRIES_DEPOSIT_REPAIR=TRUE, and on these cells that is the whole product: all
    # four of their corpus rows declare an electric deposit inside the seam, so at
    # False either module would gate green and serve ZERO.
    #
    # THE FIRST OF THEM IS THE ONLY CONSUMER IN THIS COUNT THAT PASSES A NON-DEFAULT
    # `repair_paths`. Its layer is inert, so `update_E` takes the pure overwrite at
    # stepping.py:1019-1022 and the recurrence its bracket inverts is
    # `deposit_repair.PLAIN_PATH`, not the split-field one every other consumer here
    # declares implicitly. That is why the clause has to be passed the paths rather
    # than assume them: `repairable` refuses BY NAME any configuration whose
    # recurrence is not among the caller's declared set, so the alternative to
    # passing it is a refusal of every row the product exists for.
    # 85 -> 87 on 2026-09-02, THE COMPLEX STENCIL ROUND: the two CUDA welds that
    # close the LAST two cells the hand-CUDA board recorded UNBUILDABLE --
    # cuda_kernels/folded_complex_offdiag_fused_electric_pair.py and
    # cuda_kernels/complex_no_pml_offdiag_fused_electric_pair.py. They are the
    # COMPLEX64 twins of the two CUDA entries the stencil round added above, and
    # like those they consult the clause with CARRIES_DEPOSIT_REPAIR=FALSE for the
    # same MEASURED reason rather than by default: `deposit_repair.repairable`
    # refuses an off-diagonal chi1inv row BY NAME, and every row of both cells has
    # one installed because the constitutive arm REQUIRES it. So they consult the
    # clause exactly to refuse every electric deposit, which is what puts them on
    # this count -- and the ceiling each buys is the rows that clear the injection
    # on the driver fact alone, 1 of 3 on the folded cell and 1 of 2 on the
    # no-absorber one.
    #
    #   -1 = 86  2026-09-21, and this one is a CORRECTION, not a product leaving. The
    #            scan now reads EXECUTABLE TEXT (see `_executable_text` above), and on
    #            the raw file it had been counting `cuda_kernels/fused_pairs.py`, which
    #            names `deposit_repair.seam_source_reasons` in EIGHT docstrings while
    #            consulting it nowhere: that module's own prose says the check belongs
    #            to "the family's own coverage predicate ... Checking it twice would let
    #            the two answers disagree" (fused_pairs.py:85-90). It declares the flag,
    #            so it never showed up as MISSING -- it inflated the count only, which is
    #            how a false consumer hides. The same round's `dispatch_preference.py`
    #            would have been an 88th of the same kind: it names the clause to say
    #            what the eventual wiring site will call, holds no product and consults
    #            nothing, and a raw grep demanded a fused-pair flag from an arbitration
    #            module. 86 is the count of modules that ACTUALLY ask the clause.
    assert len(consumers) == 86, f"expected 86 seam-clause consumers, found {len(consumers)}: {consumers}"


# ---------------------------------------------------------------------------
# The shared clause against the algorithm it replaced
# ---------------------------------------------------------------------------


def _original_clause(sources, pair, undeclared, refusal):
    """The clause as it was written out by hand in all twenty-one products.

    Kept here as the ORACLE. The delegation is only safe if it answers identically for
    every source configuration while ``carries_repair`` is False, and "identically"
    includes the index each refusal reports -- see ``_in_seam_indexed``.
    """
    reasons = []
    if sources is None:
        reasons.append(undeclared)
    else:
        for index, source in enumerate(tuple(sources)):
            magnetic = str(getattr(source, "field_type", "")) == "B"
            if (pair == "B" and magnetic) or (pair == "D" and not magnetic):
                reasons.append(refusal(index, source))
    return tuple(reasons)


def _matrix(grid):
    m = lambda: _source(grid, "B", "Hz")          # noqa: E731
    e = lambda: _source(grid, "D", "Ez")          # noqa: E731
    return [("none", None), ("empty", ()), ("one magnetic", (m(),)),
            ("one electric", (e(),)), ("magnetic then electric", (m(), e())),
            ("electric then magnetic", (e(), m())),
            ("two electric", (e(), e())), ("two magnetic", (m(), m()))]


@pytest.mark.parametrize("pair", ["B", "D"])
def test_the_shared_clause_answers_exactly_what_the_hand_written_one_did(pair):
    grid, fields, _pml = _build()
    undeclared = "the source set was not declared"
    refusal = (lambda index, source:
               f"source {index} ({type(source).__name__}) is in the {pair} seam")
    for name, sources in _matrix(grid):
        got = deposit_repair.seam_source_reasons(
            fields, sources, pair, undeclared=undeclared, refusal=refusal,
            carries_repair=False)
        want = _original_clause(sources, pair, undeclared, refusal)
        assert got == want, f"{pair} / {name}: {got!r} != {want!r}"


@pytest.mark.parametrize("pair", ["B", "D"])
def test_the_mixed_case_reports_the_index_in_the_callers_list(pair):
    """The regression the filtered-subset implementation would have introduced."""
    grid, fields, _pml = _build()
    sources = ((_source(grid, "D", "Ez"), _source(grid, "B", "Hz")) if pair == "B"
               else (_source(grid, "B", "Hz"), _source(grid, "D", "Ez")))
    reasons = deposit_repair.seam_source_reasons(
        fields, sources, pair, undeclared="x",
        refusal=lambda index, source: f"source {index}", carries_repair=False)
    assert reasons == ("source 1",), reasons


@pytest.mark.parametrize("pair", ["B", "D"])
def test_declaring_the_repair_admits_the_seam_the_clause_used_to_refuse(pair):
    """What the flag buys, once a product may set it."""
    grid, fields, _pml = _build()
    sources = (_source(grid, pair, "Hz" if pair == "B" else "Ez"),)
    refused = deposit_repair.seam_source_reasons(
        fields, sources, pair, undeclared="x", refusal=lambda i, s: "refused",
        carries_repair=False)
    admitted = deposit_repair.seam_source_reasons(
        fields, sources, pair, undeclared="x", refusal=lambda i, s: "refused",
        carries_repair=True)
    assert refused == ("refused",)
    assert admitted == ()


def test_no_site_hard_codes_the_flag_instead_of_reading_its_module_constant():
    """``carries_repair=True`` written literally at a site would bypass the source-level
    gate that keeps a premature flip loud."""
    import ast

    here = pathlib.Path(__file__).parent
    offenders = []
    for path in sorted(here.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        if "seam_source_reasons(" not in text or "def seam_source_reasons(" in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "seam_source_reasons"):
                continue
            for keyword in node.keywords:
                if keyword.arg != "carries_repair":
                    continue
                if not (isinstance(keyword.value, ast.Name)
                        and keyword.value.id == "CARRIES_DEPOSIT_REPAIR"):
                    offenders.append(f"{path.relative_to(here)}:{node.lineno}")
    assert not offenders, f"{offenders} pass carries_repair as something other than the "\
                          f"module constant the gate reads"


def test_each_predicate_asks_about_the_seam_its_own_refusal_names():
    """The seam argument and the refusal prose are two statements of the same fact.

    CAUGHT A REAL BUG. The twenty-one clauses were delegated mechanically, with the seam
    inferred from each site's `== / != MAGNETIC_FIELD_TYPE` test.
    `metal_kernels/complex_conductive_fused_pair.py` spells that condition differently and
    was assigned seam 'B' while its refusal says "is electric ... BETWEEN step_D and
    update_E". Nothing else would have failed: with `carries_repair=False` a wrong seam
    still returns a refusal, just for the wrong sources — it would have refused on a
    magnetic source and ADMITTED an electric one, which is the over-covering this whole
    clause exists to prevent, and it would have surfaced as wrong fields on a fused
    dispatch rather than as a red test.

    The prose is an INDEPENDENT witness because it was written by hand, per product,
    before any of this delegation existed.
    """
    import ast

    here = pathlib.Path(__file__).parent
    checked, mismatched = [], []
    for path in sorted(here.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        if "seam_source_reasons(" not in text or "def seam_source_reasons(" in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "seam_source_reasons"):
                continue
            source = ast.get_source_segment(text, node) or ""
            pair = (node.args[2].value
                    if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant)
                    else None)
            if pair is None:
                # coverage.py is parameterised by `pair` and builds its prose from the
                # same spec, so it cannot disagree with itself.
                continue
            expected = ("B" if "is magnetic" in source
                        else "D" if "is electric" in source else None)
            if expected is None:
                continue
            checked.append(str(path.relative_to(here)))
            if pair != expected:
                mismatched.append(
                    f"{path.relative_to(here)}:{node.lineno} asks seam {pair!r} but its "
                    f"refusal says {'magnetic' if expected == 'B' else 'electric'}")
    assert not mismatched, mismatched
    # A scan that matched nothing would pass and prove nothing. 25 = the 22 the
    # delegation landed with, plus the three sites that stopped INHERITING the clause as
    # each wired product declared the repair: triton nonlinear_fused_magnetic_pair
    # (inherits fused_pair_coverage through a view that HIDES its nonlinearity),
    # triton fused_dispersive_chain (inherits the dispersive pair but owns a third slot
    # the two-consult repair does not span), and metal nonlinear_fused_magnetic_pair
    # (inherits metal fused_magnetic_pair's predicate but holds no absorb row, so
    # nothing would bracket its launch). +1 = 26 on 2026-08-28: metal
    # fused_electric_pair, the plain D/E pair, whose refusal prose says "is electric"
    # and whose seam is 'D' -- the polarity this scan exists to keep honest, and the
    # first D-side unfolded product to state it.
    # +9 = 35 ON 2026-08-30: the same nine new products the consumer count above gained,
    # every one of them stating its polarity in prose and checked here. SIX OF THE NINE
    # ARE D-SIDE -- beta_fused_electric_pair, complex_fused_electric_pair,
    # folded_complex_fused_pair, folded_beta_complex_fused_pair,
    # folded_beta_real_fused_pair and folded_fused_dispersive_pair, against three B-side
    # (folded_beta_real_fused_magnetic_pair and the two CUDA products). That is the
    # largest D-side arrival this scan has seen, and D is the side its docstring's bug
    # came from: a D-seam product mis-assigned 'B' refuses the magnetic source it should
    # admit and ADMITS the electric one it must repair. All six were read back from the
    # `seam_source_reasons` call itself and answer 'D'.
    # THIS COUNT TRAILS THE CONSUMER COUNT BY ONE and always has:
    # triton_kernels/coverage.py is parameterised by `pair` and leaves via the
    # `pair is None` branch above, so 37 consumers give 36 prose-bearing sites.
    # +1 = 36 on 2026-08-30 with the Triton D->E complex pair named above; it states
    # its polarity in prose and answers 'D', read back from its own
    # `seam_source_reasons` call.
    # +3 = 39 on 2026-08-30, and the arithmetic is the tell that the pairing held:
    # the consumer count above gained FOUR and this one gains THREE, because
    # cuda_kernels/fused_pairs.py is the CUDA block's installer rather than a
    # predicate and states no refusal prose of its own. ALL THREE ARE D-SIDE --
    # triton cylindrical_fused_electric_pair, triton complex_conductive_fused_pair
    # and cuda fused_electric_pair -- and D is the side this scan's docstring bug
    # came from: a D-seam product mis-assigned 'B' refuses the magnetic source it
    # should admit and ADMITS the electric one it must repair. All three were read
    # back from the `seam_source_reasons` call itself and answer 'D'.
    # +1 = 40 on 2026-08-31: triton_kernels/folded_dispersive_fused_pair.py, the
    # folded dispersive D->E weld. D-SIDE, which is the side this scan's docstring
    # bug came from -- a D-seam product mis-assigned 'B' refuses the magnetic source
    # it should admit and ADMITS the electric one it must repair -- and it was read
    # back from its own `seam_source_reasons` call, which answers 'D'.
    # +1 = 41 on 2026-08-31: triton_kernels/cylindrical_real_fused_electric_pair.py.
    # D-SIDE, read back from its own `seam_source_reasons` call, which answers 'D' --
    # and its MAGNETIC twin, whose prose is the mirror image, answers 'B'. That pair
    # is the sharpest check this scan has: the same two halves on two seams, each
    # naming its own polarity.
    # +1 = 43 on 2026-08-31: metal_kernels/cylindrical_fused_electric_pair.py, the
    # Dcyl |m| >= 1 complex D->E weld. D-SIDE, read back from its own
    # `seam_source_reasons` call; its magnetic twin answers 'B'.
    #
    # +1 = 42 on 2026-08-31: metal_kernels/cylindrical_real_fused_electric_pair.py,
    # the METAL sibling of the entry above. D-SIDE, read back from its own
    # `seam_source_reasons` call, which answers 'D'; its Metal MAGNETIC twin
    # answers 'B', so the two-seams-one-cell pairing the entry above describes now
    # holds on BOTH backends rather than on one.
        #
        # +1 = 44 on 2026-08-31: triton_kernels/folded_complex_fused_pair.py, the
        # folded complex D->E weld. D-SIDE, read back from its own
        # `seam_source_reasons` call, which answers 'D'; its MAGNETIC twin
        # (`folded_complex_fused_magnetic_pair`) answers 'B', so the
        # two-seams-one-cell pairing now holds on the folded complex family as well.
        #
        # +1 = 45 on 2026-08-31: triton_kernels/beta_fused_electric_pair.py, the
        # real-beta D->E weld. D-SIDE, read back from its own `seam_source_reasons`
        # call, which answers 'D'; its MAGNETIC twin answers 'B'.
        #
        # +1 = 46 on 2026-08-31: triton_kernels/bfast_fused_electric_pair.py, the
        # BFAST D->E weld. D-SIDE, read back from its own `seam_source_reasons`
        # call, which answers 'D'; its MAGNETIC twin answers 'B'.
        #
        # +1 = 47 on 2026-08-31: triton_kernels/conductive_fused_electric_pair.py,
        # the conductive PML D->E weld. D-SIDE, read back from its own
        # `seam_source_reasons` call, which answers 'D'. It has no magnetic twin.
        #
        # +2 = 49 on 2026-08-31: cuda_kernels/complex_fused_electric_pair.py and
        # cuda_kernels/cylindrical_fused_electric_pair.py. BOTH D-SIDE, each read
        # back from its own `seam_source_reasons` call, which answers 'D' against
        # prose reading "is electric: the driver injects it BETWEEN step_D and
        # update_E". Each has a MAGNETIC twin on the same track whose mirror-image
        # prose answers 'B', so the two-seams-one-cell pairing now holds on the CUDA
        # track as well as on Metal and Triton -- and that pairing is what this scan
        # is sharpest on, because a D-seam product mis-assigned 'B' (the bug in the
        # docstring above) fails here by name.
        # THIS COUNT AND THE CONSUMER COUNT ABOVE BOTH GAINED TWO, which is the
        # arithmetic tell that neither new module is `pair`-parameterised: a consumer
        # that leaves through the `pair is None` branch adds to that count and not to
        # this one, as triton_kernels/coverage.py and cuda_kernels/fused_pairs.py
        # already do. The standing gap between the two counts stays at exactly those
        # two files.
    # +1 = 50 on 2026-08-31: triton_kernels/no_pml_fused_electric_pair.py. D-SIDE,
    # read back from its own `seam_source_reasons` call, which answers 'D' against
    # prose reading "is electric: the driver injects it BETWEEN step_D and update_E".
    # It has no magnetic twin: on the plain branch `update_H` returns without touching
    # H (stepping.py:944-945), so there is no magnetic constitutive to weld to and
    # `deposit_repair.PLAIN_PATH_SEAMS` names 'D' alone.
    # THIS COUNT AND THE CONSUMER COUNT ABOVE BOTH GAINED ONE, which is the arithmetic
    # tell that the new module is not `pair`-parameterised.
    # +1 = 51 on 2026-08-31: cuda_kernels/no_pml_complex_fused_electric_pair.py.
    # D-SIDE, read back from its own `seam_source_reasons` call, which answers 'D'
    # against prose reading "is electric: the driver injects it BETWEEN step_D and
    # update_E (driver.py:3305/:3308)". It has no magnetic twin on this track: with no
    # absorber `update_H` returns without touching H (stepping.py:944-945), so the B
    # seam's constitutive half is the `cuda_no_pml` NULL arm and there is no
    # constitutive launch to weld a curl to -- the board reports that cell "NOT A
    # FUSION CANDIDATE" for exactly that reason.
    # THIS COUNT AND THE CONSUMER COUNT ABOVE BOTH GAINED ONE, which is the arithmetic
    # tell that the new module is not `pair`-parameterised.
    # +2 = 53, owed by the 2026-08-31 folded-beta round and recorded here on
    # 2026-09-01: triton_kernels/folded_beta_fused_electric_pair.py (D-SIDE) and
    # triton_kernels/folded_beta_fused_magnetic_pair.py (B-SIDE), the same two
    # modules the consumer count's stale-entry note names — their prose was read
    # back from each `seam_source_reasons` call and the two answer opposite
    # polarities on the one corpus row they share, but neither ledger gained its
    # increment the evening they landed.
    # +2 = 55 on 2026-09-01: metal_kernels/no_pml_fused_electric_pair.py and
    # metal_kernels/no_pml_conductive_fused_electric_pair.py. BOTH D-SIDE, each
    # read back from its own `seam_source_reasons` call, which answers 'D' against
    # prose reading "is electric: the driver injects it BETWEEN step_D and
    # update_E" (the conductive one adds "and on a conductive row through
    # _inject_electric_through_conductivity"). Neither has a magnetic twin, for
    # the reason the Triton entry at 50 records: on the plain branch `update_H`
    # returns without touching H (stepping.py:944-945), so there is no magnetic
    # constitutive to weld to and `deposit_repair.PLAIN_PATH_SEAMS` names 'D'
    # alone. THIS COUNT AND THE CONSUMER COUNT ABOVE BOTH GAINED TWO, the
    # arithmetic tell that neither new module is `pair`-parameterised.
    # +5 = 60 on 2026-09-02: the five residual CUDA magnetic welds. ALL B-SIDE,
    # each read back from its own `seam_source_reasons` call answering 'B'
    # against prose reading "is magnetic: the driver injects it BETWEEN step_B
    # and update_H". This count and the consumer count above both gained five,
    # the arithmetic tell that none of the new modules is `pair`-parameterised.
    # +4 = 64 on 2026-09-01: the four METAL RESIDUE welds. Three D-SIDE
    # (metal_kernels/bfast_fused_electric_pair.py,
    # metal_kernels/conductive_fused_electric_pair.py — whose prose adds "and on
    # a conductive row through _inject_electric_through_conductivity" —
    # and metal_kernels/beta_complex_fused_electric_pair.py) and one B-SIDE
    # (metal_kernels/beta_complex_fused_magnetic_pair.py). This count and the
    # consumer count above both gained four, the arithmetic tell that none of
    # the new modules is `pair`-parameterised.
    # +2 = 66 on 2026-09-01: the two TRITON COMPLEX-BETA welds. One D-SIDE
    # (triton_kernels/complex_beta_fused_electric_pair.py, prose "is electric:
    # the driver injects it BETWEEN step_D and update_E") and one B-SIDE
    # (triton_kernels/complex_beta_fused_magnetic_pair.py, prose "is magnetic:
    # the driver injects it BETWEEN step_B and update_H"). This count and the
    # consumer count above both gained two, the arithmetic tell that neither new
    # module is `pair`-parameterised.
    # +5 = 71 on 2026-09-02: the five CUDA ELECTRIC TWINS of the residue round.
    # ALL D-SIDE, each read back from its own `seam_source_reasons` call
    # answering 'D' against prose reading "is electric: the driver injects it
    # BETWEEN step_D and update_E". This count and the consumer count above
    # both gained five, the arithmetic tell that none of the new modules is
    # `pair`-parameterised.
    # +2 = 73 on 2026-09-02: the two TRITON FOLDED COMPLEX-BETA welds, one
    # D-side and one B-side, each with its beta-less twin's prose. Both counts
    # gained two, the arithmetic tell that neither is `pair`-parameterised.
    # +6 = 79 on 2026-09-02, THE STENCIL ROUND. Six of the eight new consumers
    # above carry the prose (the two whose modules hold the clause but not a
    # per-module refusal string are counted by the consumer walk alone), all
    # D-SIDE, each reading "is electric: the driver injects it BETWEEN step_D and
    # update_E" with the off-diagonal repair refusal named beside it. Both counts
    # moving together is the arithmetic tell that none of the new modules is
    # `pair`-parameterised.
        # +2 = 81 on 2026-09-02, THE COMPLEX STENCIL ROUND: the two CUDA welds on
        # the last two UNBUILDABLE cells, both D-SIDE and each reading "is
        # electric: the driver injects it BETWEEN step_D and update_E" with the
        # off-diagonal repair refusal named beside it -- the complex64 twins of
        # the two CUDA entries the stencil round added. Both counts moving
        # together is the arithmetic tell that neither is `pair`-parameterised.
    assert len(checked) == 81, f"expected 81 prose-bearing sites, found {len(checked)}: {sorted(checked)}"


# ---------------------------------------------------------------------------
# The two slots answer together
# ---------------------------------------------------------------------------


class _CountingPlan:
    def __init__(self): self.runs = 0
    def run(self, *_a, **_k): self.runs += 1


class _Sentinel:
    """What a fused pair leaves in the slot it absorbed (NoopPlan / TrailingRepairPlan)."""
    def __init__(self, leading): self.absorbed_by = leading; self.runs = 0
    def run(self, *_a, **_k): self.runs += 1


class _LapsingPolicy:
    """Stands in for subnormal_policy, and lapses when told to."""
    def __init__(self): self.installed = True
    def policy_epoch(self): return 7
    def policy_is_installed(self): return self.installed
    def get_subnormal_policy(self): return "keep" if self.installed else None


class _StepPlan:
    def __init__(self, plans): self.plans = plans


def _plan_with(plans, policy, fields):
    from .fastpath import FastPathPlan
    return FastPathPlan(
        fields=fields, step_plan=_StepPlan(plans), slots=tuple(plans),
        arms={s: "fused pair D" for s in plans}, dropped_null={}, unwarmed={},
        record={}, policy_licence=(policy, "keep", 7))


def test_a_licence_lapse_between_a_pairs_two_consults_cannot_split_them():
    """THE DOUBLE-APPLY. The leading consult runs BOTH halves; if the licence lapses
    before the second, refusing it makes the driver run the array constitutive call on
    top of the one already performed. ``_apply_constitutive_pml`` accumulates rather
    than assigns, so the field gains an extra ``(kps - kms) * source`` — non-zero
    wherever the PML is. Refusing the FIRST consult is always safe; refusing the second
    after the first committed is not."""
    fields, policy = object(), _LapsingPolicy()
    leading = _CountingPlan()
    plans = {"step_D": leading, "update_E": _Sentinel(leading)}
    plan = _plan_with(plans, policy, fields)

    assert plan.dispatch("step_D", fields) is True
    assert leading.runs == 1
    policy.installed = False                      # the lapse, mid-step
    assert plan.dispatch("update_E", fields) is True, (
        "the absorbed slot refused after its pair had already run, so the driver would "
        "double-apply the constitutive half")
    assert plans["update_E"].runs == 1


def test_the_lapse_still_refuses_an_ordinary_slot():
    """THE CONTROL. Without it, the test above would also pass if the licence check had
    simply been deleted."""
    fields, policy = object(), _LapsingPolicy()
    plans = {"step_D": _CountingPlan()}           # no absorbed partner
    plan = _plan_with(plans, policy, fields)
    policy.installed = False
    assert plan.dispatch("step_D", fields) is False
    assert plans["step_D"].runs == 0


def test_a_lapse_before_the_leading_consult_refuses_both_slots():
    """The safe direction is unchanged: nothing has run, so both fall to the array path
    and the driver computes the whole step there."""
    fields, policy = object(), _LapsingPolicy()
    leading = _CountingPlan()
    plans = {"step_D": leading, "update_E": _Sentinel(leading)}
    plan = _plan_with(plans, policy, fields)
    policy.installed = False
    assert plan.dispatch("step_D", fields) is False
    assert plan.dispatch("update_E", fields) is False
    assert leading.runs == 0 and plans["update_E"].runs == 0


def test_the_commitment_does_not_leak_into_the_next_step():
    """A COMPLETE step's commitment must not reach the next one either.

    The WHOLE-STEP case of the lifetime rule, beside the half-consulted one below. It
    is released by the step ROLL-OVER, not by the consult that reads it — a consult
    whose position does not advance opens a new step and clears what the last one left
    (``fastpath.py``'s roll-over comment in ``dispatch``, and ``_ran``'s "nothing is
    discarded here"). Its docstring used to say the second consult consumed it, which
    described the bookkeeping this file's last test proves is not enough: consuming on
    read leaves a step that never REACHED the second consult holding a live commitment.
    Both consults happen here, so this pins that the roll-over covers the ordinary step
    as well — if it did not, the next step's absorbed slot would dispatch on a licence
    that had genuinely lapsed.
    """
    fields, policy = object(), _LapsingPolicy()
    leading = _CountingPlan()
    plans = {"step_D": leading, "update_E": _Sentinel(leading)}
    plan = _plan_with(plans, policy, fields)

    assert plan.dispatch("step_D", fields) is True
    assert plan.dispatch("update_E", fields) is True     # step 1, both consults
    policy.installed = False
    assert plan.dispatch("step_D", fields) is False      # step 2: refused, nothing ran
    assert plan.dispatch("update_E", fields) is False, (
        "the previous step's commitment survived into this one")


# ---------------------------------------------------------------------------
# ... on the DEPOSIT-REPAIR shape, which is 38 of the 55 pairs actually run
# ---------------------------------------------------------------------------

#: The two seams a fused pair can own, as ``(pair, leading slot, absorbed slot)``.
#: Both are exercised because the released composition runs both — 27 deposit-repair
#: instances on D and 11 on B — and because the driver-route gate drove only D
#: (``leading_repair_slots_left_running`` reads ``{"step_D": 12}`` in all three legs),
#: so the B-seam repair reaches a user through a shape no gate has stepped.
REPAIR_SEAMS = [("B", "step_B", "update_H"), ("D", "step_D", "update_E")]


@pytest.fixture
def stubbed_repair_io(monkeypatch):
    """Neutralise the repair's two device-touching helpers, and count them.

    What these tests measure is ``FastPathPlan.dispatch``'s bookkeeping ACROSS the two
    consults, not the arithmetic ``save``/``apply`` perform — ``test_deposit_repair``
    proves that against the driver's order on real arrays, and the tests above prove the
    wiring. Stubbing the pair lets the REAL ``LeadingRepairPlan`` / ``TrailingRepairPlan``
    objects — whose ``absorbed_by`` shape is the whole question here — sit in a plan
    built from bare stubs.
    """
    calls = {"save": 0, "apply": 0}

    def stub_save(*_args, **_kwargs):
        calls["save"] += 1
        return {"stub": True}

    def stub_apply(*_args, **_kwargs):
        calls["apply"] += 1
        return 0

    monkeypatch.setattr(deposit_repair, "save", stub_save)
    monkeypatch.setattr(deposit_repair, "apply", stub_apply)
    return calls


def _repair_pair(fields, policy, pair, curl, update):
    """A plan holding the real repair shape: ``LeadingRepairPlan`` + ``TrailingRepairPlan``."""
    inner = _CountingPlan()
    leading = deposit_repair.LeadingRepairPlan(inner, fields, None, (), pair)
    trailing = deposit_repair.TrailingRepairPlan(update, leading, fields, None)
    plan = _plan_with({curl: leading, update: trailing}, policy, fields)
    return plan, inner, leading, trailing


@pytest.mark.parametrize("pair,curl,update", REPAIR_SEAMS)
def test_a_lapse_between_a_deposit_repair_pairs_consults_cannot_split_it(
        pair, curl, update, stubbed_repair_io):
    """THE SAME DOUBLE-APPLY, on the shape most of the released pairs actually are.

    The sentinel shape has been covered since the clause was written. This one was not,
    while the clause's own comment said it was: in the repair shape BOTH slots carry
    ``absorbed_by`` (deposit_repair.py:683 and :711 name the same ``inner``), so reading
    that attribute as "I am the absorbed slot" made the LEADING consult look absorbed. It
    then committed nothing, and the trailing consult answered False on a lapse — with the
    driver running the array constitutive call ON TOP of the launch that had already
    performed it, accumulating an extra ``(kps - kms) * source`` everywhere in the PML.
    """
    fields, policy = object(), _LapsingPolicy()
    plan, inner, leading, _ = _repair_pair(fields, policy, pair, curl, update)

    assert plan.dispatch(curl, fields) is True
    assert inner.runs == 1 and stubbed_repair_io["save"] == 1
    policy.installed = False                        # the lapse, mid-step
    assert plan.dispatch(update, fields) is True, (
        "the trailing repair refused after its pair had already launched, so the driver "
        "would run the array constitutive call on top of it")
    assert stubbed_repair_io["apply"] == 1, "and the repair itself still ran"
    assert plan.licence_lapses == {}, (
        "a committed consult must not be counted as a lapse — it did not fall back")


@pytest.mark.parametrize("pair,curl,update", REPAIR_SEAMS)
def test_a_lapse_before_a_deposit_repair_pairs_leading_consult_refuses_both(
        pair, curl, update, stubbed_repair_io):
    """THE CONTROL. Without it the clause above would also pass with the licence check
    deleted outright: nothing has launched, so both slots must fall to the array path."""
    fields, policy = object(), _LapsingPolicy()
    plan, inner, leading, _ = _repair_pair(fields, policy, pair, curl, update)
    policy.installed = False

    assert plan.dispatch(curl, fields) is False
    assert plan.dispatch(update, fields) is False
    assert inner.runs == 0 and leading.saved is None
    assert stubbed_repair_io == {"save": 0, "apply": 0}
    assert plan.licence_lapses == {curl: 1, update: 1}


# ---------------------------------------------------------------------------
# The commitment's LIFETIME is one step
# ---------------------------------------------------------------------------


def _sentinel_pair(fields, policy, _pair, curl, update):
    """The other shape: the pair plan itself, with ``NoopPlan`` in the absorbed slot."""
    inner = _CountingPlan()
    sentinel = _Sentinel(inner)
    plan = _plan_with({curl: inner, update: sentinel}, policy, fields)
    return plan, inner, inner, sentinel


@pytest.mark.parametrize("build", [_sentinel_pair, _repair_pair],
                         ids=["sentinel", "deposit_repair"])
@pytest.mark.parametrize("pair,curl,update", REPAIR_SEAMS)
def test_a_step_that_never_reaches_the_absorbed_consult_cannot_poison_the_next(
        build, pair, curl, update, stubbed_repair_io):
    """A COMMITMENT MUST NOT OUTLIVE ITS STEP, and a leaked one SKIPS a sub-step.

    The commitment used to be released only by the absorbed consult that consumed it, so
    a step that reached the leading consult and not the absorbed one — an exception
    between the two driver sites, or a caller driving the seam itself — left it standing.
    The next step then refused the leading slot on a genuinely lapsed licence, ran the
    ARRAY curl, and answered the absorbed slot True off the stale entry: only the
    sentinel ran, and the constitutive sub-step happened on neither path. That is a
    worse failure than the double-apply the clause exists to stop, and it is silent.
    """
    fields, policy = object(), _LapsingPolicy()
    plan, inner, _leading, absorbed = build(fields, policy, pair, curl, update)

    assert plan.dispatch(curl, fields) is True      # step 1: leading only, then abort
    policy.installed = False
    assert plan.dispatch(curl, fields) is False, (  # step 2: the licence has lapsed
        "the leading slot must refuse — refusing it is always safe, nothing has run")
    assert plan.dispatch(update, fields) is False, (
        "the absorbed slot dispatched on step 1's commitment, so the driver skipped its "
        "array call while only the sentinel ran: the sub-step happened nowhere")
    assert inner.runs == 1, "step 1's launch, and nothing on step 2"
    assert getattr(absorbed, "runs", 0) == 0, "the sentinel never ran"
    assert stubbed_repair_io["apply"] == 0, "and neither did the repair"
    assert plan.licence_lapses == {curl: 1, update: 1}


def test_adding_a_source_after_the_freeze_replans():
    """THE SOURCE LIST IS A PLAN INPUT, so it must invalidate like the material ones.

    Harmless until 2026-08-23: `plan_fast_path` never saw the source list, so a source
    added after the configuration froze could not stale a plan. It is an input now — the
    fused-pair predicates cannot read it off `Fields` and refuse outright when it is not
    declared — and neither `add_source` path called `invalidate_fast_path`. A pair
    admitted for one source set and then run against another computes its constitutive
    half against a deposit nothing repaired, which is a silent wrong answer rather than a
    refusal. Unreachable while `fuse=False`; this is what stops the flip discovering it.
    """
    from . import driver as driver_module

    calls = {"n": 0}
    real = driver_module.FdtdDriver.invalidate_fast_path

    def counting(self):
        calls["n"] += 1
        return real(self)

    driver = driver_module.FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    driver_module.FdtdDriver.invalidate_fast_path = counting
    try:
        before = calls["n"]
        driver.add_source({"source_type": "gaussian", "component": "Ez",
                           "center": [0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0],
                           "frequency": 1.0, "fwidth": 0.2})
        electric = calls["n"] - before
        before = calls["n"]
        driver.add_source({"source_type": "gaussian", "component": "Hz",
                           "center": [0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0],
                           "frequency": 1.0, "fwidth": 0.2})
        magnetic = calls["n"] - before
    finally:
        driver_module.FdtdDriver.invalidate_fast_path = real

    # Both add_source routes: the ExtendedSource one and the VolumeSource one a magnetic
    # component is forced down. Missing either leaves half the seam stale.
    assert electric >= 1, "adding an electric source did not re-plan"
    assert magnetic >= 1, "adding a magnetic source did not re-plan (VolumeSource route)"


def test_every_triton_routed_module_has_decided_about_the_bracket():
    """THE TRITON PARTITION, exact in both directions, as Metal's and CUDA's are.

    Until the 2026-09-02 installer wave the Triton composer's reach was bounded by
    a CALL COUNT alone: three hand-written branches, three calls, and a reader
    could see how many installers there were but not which families could arrive
    at one. ``launch.CERTIFIED_FUSED_PRODUCTS`` is now a table, exactly as
    ``metal_kernels.launch.FUSED_PAIR_ARMS`` and ``cuda_kernels.fused_pairs.
    FUSED_PRODUCTS`` are, so the same partition the other two backends carry can be
    asserted here: a routable family either carries the repair
    (:data:`WIRED_FOR_THE_REPAIR`) or deliberately does not
    (:data:`ROUTED_WITHOUT_THE_REPAIR`), and a family in NEITHER set is a module
    nobody decided about — which is the one direction that puts an unbracketed
    launch on a live seam.

    BOTH DIRECTIONS. The forward one catches a product wired without a flag
    decision; the reverse catches a ``ROUTED_WITHOUT_THE_REPAIR`` entry whose
    module the composer cannot actually install, which would be a declaration
    about nothing.
    """
    from meep_gpu.triton_kernels import launch as triton_launch  # noqa: PLC0415

    routed = {f"triton_kernels/{spec['module']}.py"
              for spec in triton_launch.CERTIFIED_FUSED_PRODUCTS.values()}
    # The three hand-written branches route these five in the same protocol.
    routed |= {"triton_kernels/dispersive_fused_pair.py",
               "triton_kernels/folded_fused_magnetic_pair.py",
               "triton_kernels/folded_fused_pair.py"}
    here = pathlib.Path(__file__).parent
    launch_source = (here / "triton_kernels" / "launch.py").read_text(
        encoding="utf-8")
    for module_path in sorted(routed):
        assert module_path in WIRED_FOR_THE_REPAIR or \
            module_path in ROUTED_WITHOUT_THE_REPAIR, (
                f"{module_path} is routed by the Triton composer but is in neither "
                f"the wired set nor the routed-without-the-repair set; the seam loop "
                f"would reach a module that has decided nothing about the bracket")
    triton_routed_only = {path for path in routed if path not in WIRED_FOR_THE_REPAIR}
    assert triton_routed_only == {path for path in ROUTED_WITHOUT_THE_REPAIR
                                  if path.startswith("triton_kernels/")}, (
        f"the Triton modules the composer routes without the repair are "
        f"{sorted(triton_routed_only)}, but ROUTED_WITHOUT_THE_REPAIR names "
        f"{sorted(p for p in ROUTED_WITHOUT_THE_REPAIR if p.startswith('triton_kernels/'))}")
    # ...and every routed module really is named by a builder in the composer, which
    # is what DECLARED_BUT_NOT_ROUTED asserts the absence of for the other direction.
    for module_path in sorted(routed):
        stem = module_path.split("/", 1)[1][: -len(".py")]
        if stem in ("dispersive_fused_pair", "folded_fused_magnetic_pair",
                    "folded_fused_pair"):
            continue          # built through their own hand-written branches
        assert f"plan_{stem}" in launch_source, module_path
