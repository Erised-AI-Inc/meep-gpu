"""The two COMPLEX hand-CUDA fused pairs on the ELECTRIC seam: predicate, lift, protocol.

WHAT THIS FILE SETTLES. ``complex_fused_electric_pair`` and
``cylindrical_fused_electric_pair`` put the ``cuda_complex`` and ``cuda_cyl_complex``
families on ``step_D -> zero_metal_D -> update_E``. Between them they occupy the
board's two largest D cells that no product occupied
(``D_to_E (cuda_complex/complex, cuda_complex/complex)`` and ``D_to_E
(cuda_cyl_complex/cylindrical complex, cuda_complex/complex)``, 16 seam-instances
each), and 31 of those 32 instances are blocked by the ELECTRIC INJECTION rather than
by anything about the weld -- so what has to be right, and what a byte comparison on
an admitted row can never see, is:

1. **Does each predicate refuse everything the launch cannot serve, BY NAME?** A
   bit-comparison on an admitted row says nothing about a row that should never have
   been admitted. Two clauses here are NOT inherited from either half and so are the
   whole guarantee: the fold refusal (which is what makes ``REPLACES`` honest) and the
   CONDUCTIVITY refusal -- both complex halves admit a conductivity by name
   (``coverage.covers_real_pml_complex_constitutive``'s own paragraph), and the driver
   deposits an electric source through ``_inject_electric_through_conductivity`` when
   one is present, a route the repair has no verdict on.

2. **Are three products on one seam still DISJOINT?** ``install_fused_pairs`` leaves a
   seam UNFUSED when two products claim it, so an overlap costs slots rather than
   gaining them. Measured on three fixtures, not argued.

3. **Is the lift the certified text?** Every anchor is asserted against what the
   certified emitters produce. Three edits are silent when wrong: the sub-lattice
   rename (both vectors are real, both indexed the same way, half a cell apart in the
   absorber), the OFF-DIAGONAL wall table (the B family's diagonal compiles and clears
   one wrong component per wall), and the cylindrical AXIS TAIL, whose two branches
   both overwrite D after the recurrence.

4. **Does the two-consult protocol reproduce the driver's own order, bit for bit,
   UNDER COMPLEX STORAGE?** ``test_deposit_repair``'s ``COMPLEX_CASES`` drive the B
   seam only; the D seam under complex64 is what these two products need and had no
   case anywhere.

NOTHING HERE COMPILES OR LAUNCHES ANYTHING. Each fused kernel's arithmetic is a DEVICE
claim and no test on this host may speak to it (``parity/meep_gpu/
gate_cuda_fused_complex_pairs.py`` is where that is measured); what is settled here is
the predicate, the lift and the protocol.
"""

from __future__ import annotations

import ast
import pathlib

import numpy
import pytest

from .. import deposit_repair, stepping
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from ..sources import GaussianEnvelope, VolumeSource
from ..triton_kernels.launch import NoopPlan
from . import arms, fused_pairs, in_seam_coverage
from . import complex_fused_electric_pair as cartesian
from . import cylindrical_fused_electric_pair as cylindrical
from .registry import LICENSE_COMPLEX
from ..test_deposit_repair import _build, _differing, _reference, _words
from .test_fused_pairs import _NumpyWearingCupysName

#: The two products this file covers, with the pieces that differ between them. A test
#: that special-cased a family in its own body would be a place the two verdicts could
#: stop meaning the same thing.
FAMILIES = {
    "cartesian": cartesian,
    "cylindrical": cylindrical,
}

#: The expansion licence both predicates require. It is NOT optional and never
#: defaulted: the arm is compiled into the binary at this seam, and asking with
#: ``license=None`` returns the complex family's own named refusal on every
#: configuration -- an answer about the caller, recorded as if it were about the run.
LICENCE = {"arm": "FMA_V1", "expansion": 1, "basis": "measured",
           "refusals": [], "policy_resolved": "keep"}
POLICY = "keep"


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def build_cartesian(xp, *, boundaries=("metallic", "periodic", "periodic"),
                    symmetry=(), cell=(8.0, 9.0, 10.0), complex_storage=True):
    """A frozen ``(fields, pml, grid)`` triple the Cartesian complex D/E pair admits.

    THE EXTENTS ARE DELIBERATELY UNEQUAL: a cube lets an index decomposition swap i and
    k and stay identical, and this seam indexes three coefficient vectors of three
    different lengths.
    """
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=symmetry, xp=xp)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def build_cylindrical(xp, *, m=2, shape=(9, 1, 11), z_kind="metallic",
                      complex_storage=True):
    """A frozen ``(fields, pml, grid)`` triple the Dcyl complex D/E pair admits."""
    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=m, boundaries={"z": z_kind}, courant=0.5, xp=xp)
    assert tuple(int(n) for n in grid.shape) == shape, grid.shape
    layer = PML(grid=grid, thickness={"x": (0, 2), "z": 2})
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


BUILD = {"cartesian": build_cartesian, "cylindrical": build_cylindrical}


def covers(name, fields, layer, grid, sources=()):
    """The named family's predicate, always asked WITH the licence."""
    module = FAMILIES[name]
    predicate = getattr(module, f"covers_{module.FAMILY[len('cuda_'):]}")
    return predicate(fields, layer, grid, sources, LICENCE, POLICY)


def electric_source(grid, component="Ez"):
    return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


def magnetic_source(grid, component="Hz"):
    return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


class _ArrayPair:
    """Stands in for the fused kernel: curl + wall clear + constitutive in one object.

    THE THING UNDER TEST IS THE PROTOCOL, NOT THE KERNEL -- the same stand-in the two
    sibling files use, with ``REPLACES`` read from the module rather than transcribed,
    so a pass added to or dropped from that tuple changes this stand-in without an edit
    here.
    """

    TAKES_PML = frozenset({"step_D", "update_E"})

    def __init__(self, fields, pml, replaces):
        self._fields, self._pml, self._replaces = fields, pml, replaces
        self.runs = 0

    def run(self, *_args, **_kwargs):
        self.runs += 1
        for name in self._replaces:
            pass_ = getattr(stepping, name)
            if name in self.TAKES_PML:
                pass_(self._fields, self._pml)
            else:
                pass_(self._fields)


# --------------------------------------------------------------------------
# 1. THE DECLARATIONS, AND THE WIRING THEY MAY NOT DRIFT FROM
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_product_declares_the_repair_and_the_block_that_brackets_it_exists(name):
    """The flag and the wiring change together or not at all.

    A product that declared ``CARRIES_DEPOSIT_REPAIR`` without the two slots would
    compute ``update_E`` against a pre-injection D and report success -- the exact
    failure ``deposit_repair`` exists to make impossible. Both halves are READ rather
    than assumed.
    """
    module = FAMILIES[name]
    assert module.CARRIES_DEPOSIT_REPAIR is True
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source, (
        "the predicate must pass the module's own constant, not a literal")
    block = pathlib.Path(fused_pairs.__file__).read_text(encoding="utf-8")
    assert "LeadingRepairPlan(" in block and "TrailingRepairPlan(" in block
    assert fused_pairs.FUSED_PRODUCTS[module.FAMILY]["curl_slot"] == module.SLOT
    assert module.SLOT == "step_D"
    assert fused_pairs.FUSED_PAIR_SEAMS["step_D"] == ("update_E", "D")
    assert module.FAMILY in fused_pairs.FUSED_PAIR_ARMS


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_replaced_passes_are_a_run_of_the_drivers_own_order(name):
    """``REPLACES`` is a claim about what one launch performed; it is checked against
    ``FdtdDriver.step``'s SOURCE rather than against a list retyped here.

    The two mirror fills sit between ``zero_metal_D`` and ``update_E`` in the driver and
    are NOT in ``REPLACES``: they are refused, not carried, so what has to hold is that
    every pass named is a driver pass in this seam and in this order.
    """
    driver = pathlib.Path(stepping.__file__).with_name("driver.py").read_text(
        encoding="utf-8")
    step = driver.split("def step(", 1)[1].split("\n    def ", 1)[0]
    order = [pass_ for pass_ in ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                                 "fill_folded_far_ghosts_D", "update_E")
             if f"{pass_}(self.fields" in step]
    assert order == ["step_D", "fill_symmetry_bc_D", "zero_metal_D",
                     "fill_folded_far_ghosts_D", "update_E"]
    replaces = FAMILIES[name].REPLACES
    positions = [order.index(pass_) for pass_ in replaces]
    assert positions == sorted(positions), replaces
    assert replaces == ("step_D", "zero_metal_D", "update_E")


#: Which two arm predicates each product's own predicate conjoins, and the label pair
#: ``FUSED_PAIR_ARMS`` must therefore record. READ FROM THE PREDICATE by ``ast`` below,
#: not assigned to it: if a predicate ever conjoined different arms, absorbing those two
#: slots would substitute a numerical product no arm admitted.
CONJOINED = {
    "cartesian": ("covers_real_pml_complex_curl",
                  "covers_real_pml_complex_constitutive"),
    "cylindrical": ("covers_pml_cylindrical_complex_curl",
                    "covers_real_pml_complex_constitutive"),
}


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_absorb_declaration_matches_the_arms_the_predicate_conjoins(name):
    from . import registry  # noqa: PLC0415 - the table, read for its labels

    module = FAMILIES[name]
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    predicate = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name.startswith("covers_"))
    called = {node.func.id for node in ast.walk(predicate)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert set(CONJOINED[name]) <= called

    by_predicate = {row["name"]: row["label"] for row in registry._TABLE}
    assert fused_pairs.FUSED_PAIR_ARMS[module.FAMILY] == (
        by_predicate[CONJOINED[name][0]], by_predicate[CONJOINED[name][1]])


def test_each_new_product_shares_its_labels_only_with_its_own_magnetic_twin():
    """The one thing that would turn a product into a lost seam rather than a gain.

    ``install_fused_pairs`` leaves a seam UNFUSED when two products claim it, so a
    product sharing another's two arm LABELS is safe only because the two are keyed on
    different curl slots. Asserted rather than argued: the labels really are the same
    strings, and the slots really do differ.
    """
    slots = {family: product["curl_slot"]
             for family, product in fused_pairs.FUSED_PRODUCTS.items()}
    for electric, magnetic in (
            ("cuda_complex_fused_electric_pair",
             "cuda_complex_fused_magnetic_pair"),
            ("cuda_cylindrical_fused_electric_pair",
             "cuda_cylindrical_fused_magnetic_pair"),
            # THE FIVE ELECTRIC TWINS, 2026-09-02 (the residue round): each is
            # its magnetic twin's label pair on the other seam, the same
            # different-curl-slot keying that makes the sharing safe.
            ("cuda_complex_folded_fused_electric_pair",
             "cuda_complex_folded_fused_magnetic_pair"),
            ("cuda_complex_beta_fused_electric_pair",
             "cuda_complex_beta_fused_magnetic_pair"),
            ("cuda_cylindrical_real_fused_electric_pair",
             "cuda_cylindrical_real_fused_magnetic_pair"),
            ("cuda_special_kz_fused_electric_pair",
             "cuda_special_kz_fused_magnetic_pair"),
            ("cuda_bfast_fused_electric_pair",
             "cuda_bfast_fused_magnetic_pair")):
        assert (fused_pairs.FUSED_PAIR_ARMS[electric]
                == fused_pairs.FUSED_PAIR_ARMS[magnetic])
        assert slots[electric] == "step_D" and slots[magnetic] == "step_B"
    # And the products now on step_D carry that many DIFFERENT label pairs, so the
    # ambiguity cannot arrive through the absorb declaration either.
    #
    # 3 -> 4 ON 2026-08-31 with `cuda_no_pml_complex_fused_electric_pair`, the complex
    # NO-ABSORBER weld. It is the first `step_D` product with no magnetic twin on this
    # track and so is absent from the pairing loop above: with no absorber `update_H`
    # returns without touching H (stepping.py:944-945), so the B seam's constitutive
    # half is the `cuda_no_pml` NULL arm and there is no constitutive launch to weld a
    # curl to. Its own file measures the four-way disjointness on a fixture set that
    # includes an absorber-free run, which is the shape this file's three fixtures do
    # not have.
    #
    # 4 -> 9 ON 2026-09-02 with the five electric twins, whose label pairs are
    # their magnetic twins' -- so `step_D` mirrors `step_B`'s measured
    # eight-way partition plus the absorber boolean. Nine products, nine
    # DISTINCT label pairs.
    #
    # 9 -> 11 LATER THAT DAY with the two STENCIL WELDS, which are the first
    # `step_D` products taking an OFF-DIAGONAL constitutive arm. They have no
    # magnetic twin either -- `update_H` reads nothing off-diagonal
    # (stepping.py:907-923), so there is no off-diagonal constitutive arm on the B
    # seam to weld a curl to -- and they are absent from the pairing loop above for
    # that reason. They partition against the nine on a clause NEITHER of those
    # nine has to state: `covers_real_pml_constitutive(side='E')` refuses every
    # off-diagonal run by its own clause, and both of these REQUIRE one, so no row
    # can reach both sets. `test_offdiag_stencil_welds.py` measures that on
    # fixtures with the rows installed, which this file's fixtures do not have.
    #
    # 11 -> 12 ON 2026-09-02 with `cuda_dispersive_fused_electric_pair`, the first
    # `step_D` product taking a DISPERSIVE constitutive arm, and 12 -> 14 later the
    # same day with the two CONDUCTIVE-CURL welds -- the first taking a conductive
    # CURL arm, which the other twelve refuse by name
    # (`coverage.covers_real_pml_curl`: "Dx carries a conductivity: routes to the
    # three-history conductive-PML recurrence"), and the only two whose curl half is
    # not one of the real/complex/cylindrical families.
    #
    # 14 -> 16 ON 2026-09-02 with the two COMPLEX STENCIL WELDS, which close the
    # last two cells this board recorded UNBUILDABLE. They are the first `step_D`
    # products taking a COMPLEX OFF-DIAGONAL constitutive arm, and like the two real
    # stencil welds they have no magnetic twin -- `update_H` reads nothing
    # off-diagonal (stepping.py:907-923). They partition against the other fourteen
    # on TWO clauses at once rather than one: every one of those either refuses
    # complex64 storage BY NAME or refuses an off-diagonal chi1inv row BY NAME, and
    # these two REQUIRE both. Between themselves they split on the absorber, which
    # one constitutive arm requires active and the other requires inert.
    # `test_complex_offdiag_stencil_welds.py` measures their wiring on fixtures with
    # complex storage and the rows installed, which this file's do not have.
    #
    # THE SECOND ASSERTION IS THE ONE THAT MATTERS AND IT IS UNCHANGED IN FORM:
    # sixteen products, sixteen DISTINCT label pairs. Two products sharing a pair
    # could both be given the seam by the arm table, and `install_fused_pairs` would
    # then leave it UNFUSED naming both -- a coverage loss, not a duplicate count.
    #
    # 16 -> 18 ON 2026-09-02 with the two THREE-SLOT WELDS, and they are the first
    # entries here whose label tuple is of length THREE rather than two -- their span
    # crosses two seams, so it names three arms. That does not weaken the assertion
    # below and it sharpens what it says: their FIRST TWO labels are exactly the two
    # of the D->E products they supersede (`("PML", "dispersive")` and
    # `("conductive", "no-PML dispersive store")`), because their predicates conjoin
    # those very products' predicates whole. So the pairs are NOT distinct at the
    # step_D seam, ON PURPOSE, and `install_fused_pairs` does not read that as an
    # ambiguity: `_superseded_by_a_longer_span` settles a strictly-contained span
    # BEFORE the ambiguity check, by giving the slots to the longer one.
    #
    # THE ASSERTION IS THEREFORE SPLIT rather than relaxed. The sixteen two-slot
    # products must still carry sixteen DISTINCT pairs -- two of THOSE sharing one
    # would leave the seam UNFUSED naming both, which is the coverage loss this
    # measures. The two three-slot welds are asserted separately, and what is
    # required of them is the property the composer actually rests on: each one's
    # leading pair is the pair of the product it supersedes, and the two welds differ
    # from each other.
    on_d = [family for family, slot in slots.items() if slot == "step_D"]
    triples = [family for family in on_d
               if len(fused_pairs.FUSED_PAIR_ARMS[family]) == 3]
    pairs = [family for family in on_d if family not in triples]
    # 18 -> 19 ON 2026-09-04: the COMPLEX no-absorber three-slot weld, the third
    # triple, superseding the complex no-absorber D->E pair on its four rows.
    assert len(on_d) == 19
    assert len(triples) == 3 and len(pairs) == 16
    assert len({fused_pairs.FUSED_PAIR_ARMS[family] for family in pairs}) == 16
    superseded = {"cuda_three_slot_dispersive_weld":
                  "cuda_dispersive_fused_electric_pair",
                  "cuda_three_slot_no_pml_dispersive_weld":
                  "cuda_no_pml_dispersive_fused_electric_pair",
                  "cuda_three_slot_complex_no_pml_dispersive_weld":
                  "cuda_no_pml_complex_fused_electric_pair"}
    assert set(triples) == set(superseded)
    for weld, pair in superseded.items():
        assert fused_pairs.FUSED_PAIR_ARMS[weld][:2] == \
            fused_pairs.FUSED_PAIR_ARMS[pair], weld
        assert tuple(fused_pairs.FUSED_PRODUCTS[weld]["slots"]) == \
            ("step_D", "update_E", "update_P"), weld
    # EVERY triple's arm triple is its own -- written against len(triples) rather
    # than a literal, because the literal is what went stale when the third weld
    # landed (18 -> 19 above was updated in the same change and this was not).
    assert len({fused_pairs.FUSED_PAIR_ARMS[family]
                for family in triples}) == len(triples)


# --------------------------------------------------------------------------
# 2. THREE PRODUCTS ON ONE SEAM, AND THE DISJOINTNESS THAT LETS THEM COEXIST
# --------------------------------------------------------------------------

def _electric_seam_admitters(fields, layer, grid):
    """Which shipped products admit this configuration at ``step_D``, by name.

    ALL FOUR ARE ASKED, not only the two this file owns. The no-absorber product landed
    on 2026-08-31 and is asked here rather than only in its own file, because the three
    fixtures below are exactly the shapes it must NOT take -- each carries an active
    layer -- and a disjointness measured over a subset of the candidates would report
    the absence of a product rather than the absence of an overlap. It takes the
    no-absorber licence, which is a different verdict from :data:`LICENCE`.
    """
    from . import fused_electric_pair as real  # noqa: PLC0415
    from . import no_pml_complex_fused_electric_pair as no_pml  # noqa: PLC0415

    admitters = []
    if real.covers_fused_electric_pair(fields, layer, grid, ())[0]:
        admitters.append(real.FAMILY)
    for name, module in FAMILIES.items():
        if covers(name, fields, layer, grid)[0]:
            admitters.append(module.FAMILY)
    if no_pml.covers_no_pml_complex_fused_electric_pair(
            fields, layer, grid, (), LICENCE, POLICY)[0]:
        admitters.append(no_pml.FAMILY)
    return admitters


def test_exactly_one_product_admits_each_of_the_three_electric_fixtures(xp):
    """TWO ADMITTERS ON ONE SEAM IS A REFUSAL, NOT A GAIN.

    Measured on the three shapes the corpus actually drives at this seam rather than
    read off the three predicates' prose: a real run, a Cartesian complex run, and a
    Dcyl complex run at |m| >= 1.
    """
    real_fields, real_layer, real_grid = build_cartesian(xp, complex_storage=False)
    assert _electric_seam_admitters(real_fields, real_layer, real_grid) == [
        "cuda_fused_electric_pair"]

    fields, layer, grid = build_cartesian(xp)
    assert _electric_seam_admitters(fields, layer, grid) == [
        "cuda_complex_fused_electric_pair"]

    cyl_fields, cyl_layer, cyl_grid = build_cylindrical(xp)
    assert _electric_seam_admitters(cyl_fields, cyl_layer, cyl_grid) == [
        "cuda_cylindrical_fused_electric_pair"]


# --------------------------------------------------------------------------
# 3. THE PREDICATE — WHAT IT ADMITS, AND WHAT IT REFUSES BY NAME
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_an_electric_deposit_is_admitted_now_and_would_be_refused_without_the_flag(
        name, xp, monkeypatch):
    """The whole point of both products, measured against the SHIPPED flag.

    31 of the 32 corpus seam-instances on these two cells carry an electric source in
    this seam. The same configuration is asked twice -- once with
    ``CARRIES_DEPOSIT_REPAIR`` held at ``False`` -- so the admission is attributed to
    the bracket rather than assumed.
    """
    module = FAMILIES[name]
    fields, layer, grid = BUILD[name](xp)
    sources = [electric_source(grid)]
    covered, reason = covers(name, fields, layer, grid, sources)
    assert covered, reason

    monkeypatch.setattr(module, "CARRIES_DEPOSIT_REPAIR", False)
    refused, why = covers(name, fields, layer, grid, sources)
    assert not refused
    assert "is electric" in why and "driver.py:3308" in why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_magnetic_source_never_disqualifies_this_seam(name, xp):
    """The mirror image of the magnetic twins' clause, and the reason the shared
    ``deposit_repair`` helper is imported rather than re-spelled: a product that asked
    about the wrong seam would refuse the harmless source and admit the fatal one."""
    fields, layer, grid = BUILD[name](xp)
    covered, reason = covers(name, fields, layer, grid, [magnetic_source(grid)])
    assert covered, reason


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_source_that_hides_the_index_it_writes_is_refused_by_name(name, xp):
    """The repair saves and restores the cells a deposit writes; a source that does not
    publish them leaves the fused launch's pre-injection ``update_E`` standing."""
    fields, layer, grid = BUILD[name](xp)

    class _Opaque:
        field_type = "D"

    covered, why = covers(name, fields, layer, grid, [_Opaque()])
    assert not covered
    assert "does not publish the index it writes" in why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_an_undeclared_source_set_is_still_a_refusal(name, xp):
    """IGNORANCE IS NEVER AN EMPTY SET. ``Fields`` does not hold the source list, so a
    predicate that inferred "no sources" from not being told would over-cover every row
    in the corpus that has one -- which on these two cells is 31 of 32."""
    fields, layer, grid = BUILD[name](xp)
    covered, why = covers(name, fields, layer, grid, None)
    assert not covered
    assert "was not declared" in why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_conductivity_is_refused_because_the_driver_deposits_it_differently(name, xp):
    """THE CLAUSE NEITHER HALF MAKES, and therefore the one this file has to pin.

    ``covers_real_pml_complex_constitutive`` ADMITS a conductivity by name -- it is read
    in ``stepping._apply_curl`` and nowhere else -- so unlike the real electric pair
    this refusal is not inherited. With one present the driver deposits through
    ``_inject_electric_through_conductivity`` (driver.py:3305), which rescales the
    increment by ``condinv``; the repair has no verdict on that route.
    """
    fields, layer, grid = BUILD[name](xp)

    class _Conductive:
        def __getattr__(self, item):
            return getattr(fields, item)

        has_conductivity = True

    covered, why = covers(name, _Conductive(), layer, grid)
    assert not covered
    assert "condinv" in why and "driver.py:3305" in why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_conductivity_refusal_is_this_products_own_and_not_a_halfs(name, xp):
    """The clause above is only a guarantee if neither half already made it.

    Measured rather than argued: the constitutive half is asked directly on a
    conductive configuration and ADMITS it. If it ever started refusing, this test
    would go red and the module docstring's claim would need rewriting.
    """
    from .coverage import covers_real_pml_complex_constitutive  # noqa: PLC0415

    fields, layer, grid = BUILD[name](xp)

    class _Conductive:
        def __getattr__(self, item):
            return getattr(fields, item)

        has_conductivity = True

    covered, reason = covers_real_pml_complex_constitutive(
        _Conductive(), layer, grid, "E", LICENCE, POLICY)
    assert covered, reason


@pytest.mark.parametrize("axis,mirror", ((0, "X"), (1, "Y"), (2, "Z")))
def test_a_folded_axis_is_refused_at_all(xp, axis, mirror):
    """A folded grid must not reach either launch, whichever clause says so first."""
    fields, layer, grid = build_cartesian(
        xp, boundaries=("periodic", "periodic", "periodic"),
        symmetry=(Mirror(mirror),))
    assert grid.is_mirrored(axis), "the fixture did not fold the axis under test"
    covered, why = covers("cartesian", fields, layer, grid)
    assert not covered, why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_products_own_fold_clause_is_reached_and_names_both_fills(name, xp,
                                                                      monkeypatch):
    """THE CLAUSE THAT KEEPS ``REPLACES`` HONEST, EXERCISED rather than read.

    ``fill_symmetry_bc_D`` (driver.py:3309) and ``fill_folded_far_ghosts_D`` (:3311) run
    INSIDE this seam and neither is carried. The curl half already refuses a fold, so
    the product's own clause is SHADOWED by the conjunction -- which is exactly why the
    product states it anyway: a guarantee about what one launch performed may not rest
    on a clause in another module that a future device verdict could licence away.

    A clause that is never executed is a clause that can rot, so both halves are held
    at "covered" here and the product is asked on a really folded grid. Both passes must
    be named in the reason.
    """
    module = FAMILIES[name]
    fields, layer, grid = build_cartesian(
        xp, boundaries=("periodic", "periodic", "periodic"),
        symmetry=(Mirror("Y"),))
    assert grid.has_symmetry() and grid.is_mirrored(1)
    for half in ("covers_real_pml_complex_curl", "covers_real_pml_complex_constitutive",
                 "covers_pml_cylindrical_complex_curl"):
        if hasattr(module, half):
            monkeypatch.setattr(module, half, lambda *a, **k: (True, "covered"))
    covered, why = covers(name, fields, layer, grid)
    assert not covered
    assert "fill_symmetry_bc_D" in why and "fill_folded_far_ghosts_D" in why
    assert "driver.py:3309" in why and ":3311" in why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_grid_that_cannot_say_which_axes_are_walled_is_refused(name, xp):
    """``zero_metal_D`` is CARRIED, so an unanswerable grid would silently be treated as
    unwalled -- two planes of wrong values per walled axis, not a crash."""
    fields, layer, grid = BUILD[name](xp)

    class _Mute:
        def __getattr__(self, item):
            return getattr(grid, item)

        has_metallic = None

    covered, why = covers(name, fields, layer, _Mute())
    assert not covered
    assert "zero_metal_D cannot be carried" in why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_stored_extent_that_disagrees_with_the_walked_shape_is_refused(name, xp):
    """The wall plane is derived from ``grid.stored_cells`` and indexed into
    ``Dx.shape``; a disagreement writes the wipe into an array this launch is not
    walking. On the Dcyl product the radial prefix is derived from the same extent."""
    fields, layer, grid = BUILD[name](xp)

    class _Shrunk:
        def __getattr__(self, item):
            return getattr(grid, item)

        def stored_cells(self, axis):
            return 2 if axis == 2 else grid.stored_cells(axis)

    covered, why = covers(name, fields, layer, _Shrunk())
    assert not covered
    assert "stored_cells" in why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_an_inverse_permittivity_volume_of_the_wrong_extent_is_refused(name, xp):
    """The three inv_eps volumes are indexed with THIS launch's flat index, so a
    component whose volume is a different shape reads outside it."""
    fields, layer, grid = BUILD[name](xp)

    class _Ragged:
        def __getattr__(self, item):
            return getattr(fields, item)

        @staticmethod
        def inverse_epsilon_for(component):
            if component == "Ez":
                return numpy.zeros((2, 2, 2), dtype=numpy.float32)
            return fields.inverse_epsilon_for(component)

    covered, why = covers(name, _Ragged(), layer, grid)
    assert not covered
    assert "inverse_epsilon_for" in why


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_licence_is_required_and_never_defaulted(name, xp):
    """A wrong arm is a WRONG ANSWER, not a crash: both arms compile and run, and the
    arm is baked into the binary at this seam with no later rung to check it."""
    module = FAMILIES[name]
    predicate = getattr(module, f"covers_{module.FAMILY[len('cuda_'):]}")
    fields, layer, grid = BUILD[name](xp)
    covered, why = predicate(fields, layer, grid, (), None, None)
    assert not covered
    assert "licence" in why


def test_the_real_storage_refusal_is_the_partition_with_the_real_electric_pair(xp):
    """No configuration is admitted by both families, and none falls between."""
    fields, layer, grid = build_cartesian(xp, complex_storage=False)
    covered, why = covers("cartesian", fields, layer, grid)
    assert not covered
    assert "complex" in why


def test_the_cartesian_product_refuses_a_dcyl_grid_and_the_dcyl_one_requires_it(xp):
    """The two complex products' own partition, which is what lets both sit on
    ``step_D`` beside the real pair without leaving the seam unfused."""
    fields, layer, grid = build_cartesian(xp)
    covered, why = covers("cylindrical", fields, layer, grid)
    assert not covered and "cylindrical" in why.lower()

    cyl_fields, cyl_layer, cyl_grid = build_cylindrical(xp)
    covered, why = covers("cartesian", cyl_fields, cyl_layer, cyl_grid)
    assert not covered and "cylindrical" in why.lower()


# --------------------------------------------------------------------------
# 4. THE LIFT — THE CERTIFIED TEXT, AND THE EDITS THAT ARE NOT VERBATIM
# --------------------------------------------------------------------------

def _source(name, arm="NAIVE"):
    module = FAMILIES[name]
    emit = getattr(module, f"{module.FAMILY[len('cuda_'):]}_source")
    return emit(arm)


#: The name each product's shared flux-density group carries in its own signature.
FLUX = {"cartesian": ("f0", "f1", "f2"), "cylindrical": ("Dx", "Dy", "Dz")}
ENTRY = {"cartesian": "fused_electric_pair_pml_complex(",
         "cylindrical": "fused_electric_pair_pml_cyl_complex("}


@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("arm", ("NAIVE", "FMA_V1"))
def test_the_shared_flux_density_is_bound_exactly_once(name, arm):
    """THE ALIASING HAZARD. Two ``__restrict__`` pointers to one allocation is UB that
    NVRTC miscompiles without a diagnostic, so the constitutive half must have no D
    source pointers at all -- which is only true if the seam removed all three loads."""
    source = _source(name, arm)
    # SPLIT AT THE ENTRY POINT, not at the first ``\n) {``: the prelude's own device
    # helpers (``cshift_dn`` above all) close their parameter lists the same way, so a
    # split at the first terminator would inspect the helper library instead of the
    # fused body.
    after_entry = source.split(ENTRY[name], 1)[1]
    signature = after_entry.split("\n) {", 1)[0]
    for target in FLUX[name]:
        bound = signature.count(f" {target},") + signature.count(f" {target}\n")
        assert bound == 1, f"{target} is bound {bound} times in the signature"
    body = after_entry.split("\n) {", 1)[1]
    # THE CONSTITUTIVE HALF'S THREE RELOADS ARE THE WHOLE HAZARD, and they are the
    # certified lines the seam replaced. ``cf_load(g*)`` on its own is NOT the test: on
    # the Cartesian product ``g0/g1/g2`` is H, the CURL's own operand group, and the
    # curl half loads it legitimately.
    for index in ("0", "1", "2"):
        assert f"cf s{index} = cf_load(" not in body, (
            f"the constitutive reload of s{index} survived the seam rewrite; D would "
            f"need binding a second time, which is the aliasing hazard the signature "
            f"exists to avoid")
        assert f"cf s{index} = d{index};" in body


@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("arm", ("NAIVE", "FMA_V1"))
def test_the_seam_reads_the_register_and_keeps_the_certified_operand_order(name, arm):
    """``D * inv_eps`` with D ON THE LEFT (stepping.py:1011), and the register in place
    of the reload. A float32 multiply is commutative on the bits, but the certified text
    is what this family lifts and a reordered operand is an edit with no LIFT_EDITS row."""
    source = _source(name, arm)
    for index in ("0", "1", "2"):
        assert f"cf s{index} = d{index};" in source
        assert (f"s{index} = mul_field_left(s{index}, inv_eps_{index}[idx]);"
                in source)


@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("arm", ("NAIVE", "FMA_V1"))
def test_the_constitutive_half_reads_the_half_integer_sub_lattice(name, arm):
    """THE COLLISION THAT MATTERS. ``kms_a`` is the INTEGER vector for the D curl and
    the HALF-INTEGER one for ``update_E``; letting one shadow the other compiles
    perfectly and is half a cell wrong in the absorber profile."""
    source = _source(name, arm)
    target = "h" if name == "cartesian" else "f"
    for index, axis, coordinate in (("0", "x", "i"), ("1", "y", "j"),
                                    ("2", "z", "k")):
        # The Dcyl pair carries the own-cell hoist: the same statement on the derived
        # helper, taking its two preloaded own-cell words (the hoist's own claims are
        # test_cylindrical_fused_electric_pair_hoist.py's). The coefficient pair -- the
        # claim this test guards -- is spelt identically on both products.
        call = ("constitutive_apply(" if name == "cartesian" else "constitutive_apply_pre(")
        preloads = ("" if name == "cartesian" else f"pre_w_{index}, pre_e_{index}, ")
        assert (f"{call}{target}{index}, w{index}, idx, s{index}, {preloads}"
                f"kps_{axis}[{coordinate}], kms_half_{axis}[{coordinate}]);") in source
        # The curl half keeps the bare name, so both groups are present and distinct.
        assert f"kms_{axis}[" in source and f"kms_half_{axis}[" in source


@pytest.mark.parametrize("name,builder,literals", (
    ("cartesian", "complex_fused_electric_pair_tables", [False, True]),
    ("cylindrical", "cylindrical_fused_electric_pair_tables", [False, True]),
))
def test_the_sub_lattice_pairing_is_the_mirror_image_of_the_magnetic_seams(
        name, builder, literals):
    """Read off the table builders rather than asserted in prose: the D curl takes
    ``half_integer=False`` and ``update_E`` takes ``True``, which is the opposite of the
    magnetic twins' pairing. A swap is a half-cell error, not a crash."""
    from . import complex_fused_magnetic_pair, cylindrical_fused_magnetic_pair  # noqa: PLC0415

    def booleans(module, function):
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        node = next(item for item in tree.body
                    if isinstance(item, ast.FunctionDef) and item.name == function)
        return [child.value for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, bool)]

    assert booleans(FAMILIES[name], builder) == literals
    twin = (complex_fused_magnetic_pair if name == "cartesian"
            else cylindrical_fused_magnetic_pair)
    twin_function = (f"{twin.FAMILY[len('cuda_'):]}_tables")
    assert booleans(twin, twin_function) == [True, False], (
        "the magnetic twin's pairing is supposed to be the mirror image of this one")


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_carried_wall_clear_is_the_off_diagonal_table_and_a_plus_zero(name):
    """``zero_metal_D`` wipes TWO components per wall -- the complement of the B
    family's diagonal. A pair that reused the diagonal would clear one wrong component
    and leave two right ones standing on every walled run."""
    module = FAMILIES[name]
    carry = module.zero_metal_carry()
    volumes = FLUX[name]
    expected = {
        "wall_x": ("i", (volumes[1], volumes[2])),
        "wall_y": ("j", (volumes[0], volumes[2])),
        "wall_z": ("k", (volumes[0], volumes[1])),
    }
    for flag, (coordinate, written) in expected.items():
        line = next(text for text in carry.splitlines()
                    if text.startswith(f"    if ({flag} && {coordinate} == 0)"))
        for index, volume in enumerate(volumes):
            assert (f"cf_store({volume}, idx," in line) == (volume in written), (
                f"{flag}: this module writes {volume}={volume in line}, the "
                f"off-diagonal table says {volume in written}")
            del index
    # SIX register clears in all: two per walled axis, each beside its own store.
    assert carry.count("= cf_zero();") == 6
    assert "-0.0f" not in carry, "the array path assigns the Python int 0, never -0.0"


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_carried_wall_table_agrees_with_the_certified_pass(name):
    """The table is spelled in each module; the certified pass is the authority.

    Read from ``in_seam_passes``' own device text so a change there is a red test rather
    than a divergence: x wall -> Dy and Dz, y -> Dx and Dz, z -> Dx and Dy.
    """
    text = _string_constants("in_seam_passes")["_zero_metal_D_kernel_code"]
    module = FAMILIES[name]
    for flag, _coordinate, registers in module._ZERO_METAL_ROWS:
        block = text.split(f"if ({flag})", 1)[1].split("}", 1)[0]
        for index, axis in enumerate("xyz"):
            written = f"D{axis}[base] = 0.0f;" in block
            assert written == (str(index) in registers), (
                f"{flag}: the certified pass writes D{axis}={written} but this "
                f"module's table says {str(index) in registers}")


def _string_constants(module_name):
    """Every module-level ``name = <string expression>`` in one sibling, by ``ast``.

    THE CERTIFIED TEXT WITHOUT THE DEVICE LIBRARY, AND WITHOUT A SKIP. Several siblings
    import CuPy at module scope, so on the merge-bar host they cannot be imported -- and
    a test that skipped for that reason would leave the SPLICE, the part of these
    products that can be silently wrong, unchecked everywhere a laptop runs.
    """
    path = pathlib.Path(cartesian.__file__).with_name(f"{module_name}.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound = {}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return bound.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = value(node.left), value(node.right)
            return None if left is None or right is None else left + right
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        text = value(node.value)
        if text is not None:
            bound[target.id] = text
    return bound


@pytest.mark.parametrize("arm", ("NAIVE", "FMA_V1"))
def test_the_cylindrical_axis_tail_is_carried_on_both_branches(arm):
    """THE EDIT THIS FAMILY ADDS, and the one a Cartesian port would silently omit.

    ``stepping._cylindrical_axis_zero_D`` runs AFTER the recurrence with TWO branches:
    Dz alone at |m| = 1 (the field only, never fu_Dz) and all six volumes on rows
    ``[0:zero_rows]`` at |m| >= 2. A weld that captured at ``pml_apply`` and stopped
    would hand ``update_E`` the pre-zeroing displacement on exactly the rows a Dcyl run
    cares most about.
    """
    source = _source("cylindrical", arm)
    tail = source.split("stepping._cylindrical_axis_zero_D", 1)[1]
    assert "if (m_class == 1 && i == 0) {\n        d2 = cf_zero(); cf_store(Dz, idx, d2);\n    }" in tail
    for index, volume in enumerate(("Dx", "Dy", "Dz")):
        assert f"d{index} = cf_zero(); cf_store({volume}, idx, d{index});" in tail
    # THE THREE fu STORES KEEP THE CERTIFIED SPELLING AND GET NO REGISTER CLEAR: there
    # is no fu register and the constitutive half never reads one. Writing one would be
    # device text with nothing behind it.
    for volume in ("fu_Dx", "fu_Dy", "fu_Dz"):
        assert f"cf_store({volume}, idx, cf_zero());" in tail
        assert f"cf_zero(); cf_store({volume}," not in tail


@pytest.mark.parametrize("arm", ("NAIVE", "FMA_V1"))
def test_the_cylindrical_wall_clear_runs_after_the_axis_tail(arm):
    """The driver's order, and not a choice: the axis tail is INSIDE ``step_D``
    (stepping.py:485-486) and ``zero_metal_D`` is the separate driver pass at :3310."""
    source = _source("cylindrical", arm)
    assert (source.index("in_seam_passes.zero_metal_D")
            > source.index("stepping._cylindrical_axis_zero_D"))


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_lift_edits_are_data_and_every_one_is_complete(name):
    """``LIFT_EDITS`` is DATA so a gate can assert it. An edit that appeared in the
    emitter without a row here would be certified text changed with no record."""
    edits = FAMILIES[name].LIFT_EDITS
    assert edits
    assert all(set(edit) == {"line", "became", "why"} for edit in edits)
    assert all(all(isinstance(value, str) and value for value in edit.values())
               for edit in edits)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_emitted_source_is_pure_ascii_and_arm_dependent(name):
    """Pure ASCII is an NVRTC requirement; and the two arms must really differ, or the
    licence this seam bakes in would be measuring nothing."""
    naive, fma = _source(name, "NAIVE"), _source(name, "FMA_V1")
    for text in (naive, fma):
        text.encode("ascii")
    assert naive != fma


def test_the_cylindrical_emitter_refuses_by_name_where_the_text_is_unreachable():
    """THE SPLICE IS THE LIFT, so a missing half is not a degraded emit. The refusal has
    to name the module, one frame from the caller, rather than surface as an
    AttributeError inside a string operation -- and the predicate must still answer,
    which is why that import is defensive."""
    saved = cylindrical.cylindrical_complex_kernels
    try:
        cylindrical.cylindrical_complex_kernels = None
        with pytest.raises(RuntimeError) as raised:
            cylindrical.cylindrical_fused_electric_pair_source("NAIVE")
        assert "cylindrical_complex_kernels" in str(raised.value)
        assert cylindrical.covers_cylindrical_fused_electric_pair(
            None, None, None, (), LICENCE, POLICY)[0] is False
    finally:
        cylindrical.cylindrical_complex_kernels = saved


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_walls_the_predicate_admits_are_the_walls_the_launch_binds(name, xp):
    """One function decides which axes are walled, so the carry and the certified pass
    cannot disagree -- and a folded metallic axis is excluded by BOTH by construction,
    which is why each product's fold refusal makes the question moot rather than
    dangerous."""
    fields, _layer, grid = BUILD[name](xp)
    walls = in_seam_coverage.zero_metal_axes(grid)
    assert any(walls), "the fixture has no wall and cannot discriminate"
    assert fields.Dx.shape == tuple(grid.stored_cells(axis) for axis in range(3))


# --------------------------------------------------------------------------
# 4b. THE THREE LAUNCH ARGUMENTS THAT ARE SILENT WHEN WRONG
# --------------------------------------------------------------------------
#
# THESE CANNOT BE EXECUTED ON THIS HOST and are read from the SOURCE instead, which is
# a deliberate second-best with a reason: ``complex_pml_kernels`` imports CuPy at module
# scope, so every resolver that binds a phase table or a constitutive table refuses here
# by name. Each of the three is ALSO armed as a device mutation in
# ``parity/meep_gpu/gate_cuda_fused_complex_pairs.py``; what these hold is the half a
# laptop can hold, and they are here because all three compile, run, and are a different
# engine -- the class of defect that never turns anything red on its own.


def _call_arguments(module, function, callee):
    """Every call to ``callee`` inside ``function``, as its literal arguments."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    found = []

    def walk(node):
        for child in ast.walk(node):
            if (isinstance(child, ast.Call)
                    and getattr(child.func, "attr", getattr(child.func, "id", None))
                    == callee):
                found.append([
                    argument.value if isinstance(argument, ast.Constant) else None
                    for argument in child.args])

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            walk(node)
    assert found, f"{function} makes no call to {callee}"
    return found


def test_the_cartesian_resolver_binds_the_BACKWARD_bloch_table():
    """``step_D`` takes ``cshift_dn`` and the CONJUGATED phase factor
    (``stepping._shift_down``:1818-1822), which ``bloch_phase_arguments(grid, True)``
    supplies. The forward table compiles, runs, and is the wrong phase on every phased
    corpus row -- 12 of this cell's 16.

    Held on BOTH sites the argument appears: the module's own ``run_...`` and the
    composer's plan builder, because a resolver corrected in one place and not the other
    is two engines wearing one name.
    """
    for module, function in ((cartesian, "run_complex_fused_electric_pair"),
                             (fused_pairs, "_complex_fused_electric_pair_plan")):
        calls = _call_arguments(module, function, "bloch_phase_arguments")
        assert calls, function
        for arguments in calls:
            assert arguments[-1] is True, (
                f"{function} asks bloch_phase_arguments for the FORWARD table; "
                f"step_D takes the conjugate")
    # And the two tables really are different, so the assertion is not vacuous.
    from . import complex_emitter  # noqa: PLC0415

    assert complex_emitter.KERNELS["step_D"][1] is True
    assert complex_emitter.KERNELS["step_B"][1] is False


def test_the_cylindrical_resolver_binds_the_step_D_prefix_and_imr_rows():
    """The radial prefix sums Hp on this side and Ep on the other, and the two i*m/r
    rows carry the OPPOSITE SIGNS (``cylindrical_complex_kernels.IMR_TERMS``). Both
    compile, both run, and either taken from the B side is a different engine.
    """
    from . import cylindrical_complex_kernels as certified  # noqa: PLC0415

    for module, function in (
            (cylindrical, "run_cylindrical_fused_electric_pair"),
            (fused_pairs, "_cylindrical_fused_electric_pair_plan")):
        for callee in ("imr_rows_for", "cylindrical_prefix"):
            for arguments in _call_arguments(module, function, callee):
                # The sub-step is the FIRST argument of one and the SECOND of the
                # other, so the literal is looked for rather than positioned -- and
                # ``step_B`` is refused explicitly, because an absent ``step_D`` and a
                # present ``step_B`` are different failures.
                assert "step_D" in arguments, (
                    f"{function} does not ask {callee} for 'step_D' ({arguments})")
                assert "step_B" not in arguments, (
                    f"{function} asks {callee} for the B side's rows ({arguments})")
    # NOT VACUOUS: the two sub-steps really do carry different terms.
    assert (certified.IMR_TERMS["step_D"] != certified.IMR_TERMS["step_B"])
    from .cylindrical_prefix import PREFIX_COMPONENT  # noqa: PLC0415

    assert PREFIX_COMPONENT["step_D"] != PREFIX_COMPONENT["step_B"]


def test_both_launchers_bind_the_inverse_permittivity_through_the_per_component_reader():
    """``Fields.inverse_epsilon_for`` and NEVER ``fields.inv_eps``, which is the Ez view
    (fields.py:1259-1260) -- binding it for all three is the defect the twelve
    uncertified complex kernels in ``step_curl_kernels`` carry."""
    for module in FAMILIES.values():
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        # ATTRIBUTE ACCESS, not a text grep: both modules name ``fields.inv_eps`` in
        # prose to say what they must not bind, and a grep would read the warning as
        # the defect.
        attributes = {node.attr for node in ast.walk(tree)
                      if isinstance(node, ast.Attribute)}
        assert "inverse_epsilon_for" in attributes
        assert "inv_eps" not in attributes
        assert module._INVERSE_EPSILON_COMPONENTS == ("Ex", "Ey", "Ez")


# --------------------------------------------------------------------------
# 5. THE TWO-CONSULT PROTOCOL, UNDER COMPLEX STORAGE
# --------------------------------------------------------------------------
#
# ``test_deposit_repair.COMPLEX_CASES`` drives the B seam under complex64 only. The D
# seam under complex64 is what BOTH of these products need and had no case anywhere:
# the repair has to invert ``(f + kps*fw_fresh) - kms*fw_prev`` with ``fw_fresh`` being
# ``D * inv_eps`` rather than ``B``, over complex adds and float32 x complex64 products.

#: The two grids the numeric leg runs on: the 2-D shape the sibling deposit tests use
#: and a 3-D one, because the deposit is a POINT and a 2-D grid leaves one axis with no
#: interior for an index error to land in.
GRIDS = ((1.2, 1.2, 0.0), (1.0, 1.0, 1.0))


@pytest.mark.parametrize("cell", GRIDS, ids=("2d", "3d"))
def test_the_two_slots_consulted_as_the_driver_consults_them_match_the_array_path(cell):
    """THE ONE THAT MATTERS, and the only numeric claim this file makes.

    Two independently seeded copies of the same COMPLEX-STORAGE state. One is stepped in
    the DRIVER'S order -- curl, inject, fill, clear, far fill, constitutive. The other
    is stepped the way a fused pair owning both consults is: the whole seam closed
    against an UNINJECTED field at the first consult, then the driver's own
    inject/fill/clear, then the repair at the second. Every word of every array is
    compared as raw bytes, so "close" is not a passing answer.

    THE ELECTRIC SEAM'S INJECTION IS AT ``when + 0.5*dt`` (driver.py:3303), not at
    ``when``, and ``_reference`` is the shared helper that knows it -- a protocol test
    that injected at the wrong instant would compare two different runs and could pass
    only by cancelling its own error.
    """
    grid_a, fields_a, pml_a = _build(cell=cell, complex_storage=True)
    grid_b, fields_b, pml_b = _build(cell=cell, complex_storage=True)
    assert fields_b.Dz.dtype == numpy.complex64, "the fixture is not complex storage"
    src_a = [electric_source(grid_a)]
    src_b = [electric_source(grid_b)]
    assert src_b[0]._n_source_points, "the case deposits nothing and cannot discriminate"

    plans, selected = {}, {}
    inner = _ArrayPair(fields_b, pml_b, cartesian.REPLACES)
    fused_pairs._install_fused_pair(
        plans, selected, fields_b, pml_b, src_b, "D", "step_D", "update_E",
        "complex fused electric pair", inner)
    assert isinstance(plans["step_D"], deposit_repair.LeadingRepairPlan)
    assert isinstance(plans["update_E"], deposit_repair.TrailingRepairPlan)

    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, "D")
        plans["step_D"].run()
        for source in src_b:
            source.inject(fields_b, step * dt + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields_b)
        stepping.zero_metal_D(fields_b)
        stepping.fill_folded_far_ghosts_D(fields_b)
        plans["update_E"].run()

    differing = _differing(_words(fields_a), _words(fields_b))
    assert not differing, differing
    assert inner.runs == 8
    assert plans["step_D"].repairs > 0, "the repair reported touching no point"


@pytest.mark.parametrize("cell", GRIDS, ids=("2d", "3d"))
def test_the_same_protocol_without_the_repair_diverges(cell):
    """The null control. Without it the case above proves only that both routes ran."""
    grid_a, fields_a, pml_a = _build(cell=cell, complex_storage=True)
    grid_b, fields_b, pml_b = _build(cell=cell, complex_storage=True)
    src_a = [electric_source(grid_a)]
    src_b = [electric_source(grid_b)]
    inner = _ArrayPair(fields_b, pml_b, cartesian.REPLACES)
    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, "D")
        inner.run()
        for source in src_b:
            source.inject(fields_b, step * dt + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields_b)
        stepping.zero_metal_D(fields_b)
        stepping.fill_folded_far_ghosts_D(fields_b)
    differing = _differing(_words(fields_a), _words(fields_b))
    assert differing, ("the unrepaired protocol matched the reference, so the case "
                       "above proves nothing")
    # WHERE it lands is the discriminating half: the constitutive target the deposit
    # feeds and its PML auxiliary, which are exactly the two arrays `apply` restores.
    assert set(differing) == {"Ez", "f_w_Ez"}, differing


def test_a_clean_seam_keeps_the_cheap_sentinel_in_the_second_slot():
    """No deposit, no repair: the second consult must stay the no-op it always was."""
    _grid, fields, pml = _build(complex_storage=True)
    plans, selected = {}, {}
    fused_pairs._install_fused_pair(
        plans, selected, fields, pml, (), "D", "step_D", "update_E",
        "complex fused electric pair", _ArrayPair(fields, pml, cartesian.REPLACES))
    assert isinstance(plans["update_E"], NoopPlan)
    assert (selected["step_D"] == selected["update_E"]
            == "complex fused electric pair")


def test_the_second_slot_refuses_when_the_first_never_ran():
    """``dispatch`` contracts that True means the whole sub-step ran; a silent no-op
    here would leave the constitutive half undone and report success."""
    grid, fields, pml = _build(complex_storage=True)
    plans, selected = {}, {}
    fused_pairs._install_fused_pair(
        plans, selected, fields, pml, [electric_source(grid)], "D", "step_D",
        "update_E", "complex fused electric pair",
        _ArrayPair(fields, pml, cartesian.REPLACES))
    with pytest.raises(deposit_repair.DepositNotRepairable):
        plans["update_E"].run()


# --------------------------------------------------------------------------
# 6. THE BLOCK, DRIVEN THROUGH THE COMPOSER
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,label", (
    ("cartesian", "complex fused electric pair"),
    ("cylindrical", "cylindrical fused electric pair"),
))
def test_the_seam_fuses_and_the_pair_reports_the_pass_it_carries(name, label, xp):
    """Through ``plan_step(fuse=True)``, which is the only route a caller has.

    ``zero_metal_D`` is in neither slot's name, so a composition that reported only the
    slots would say it ran on the array path when the kernel performed it. The licence
    is supplied because the complex arms bind through it -- without it every complex
    product refuses, which is a fact about the caller and not about the run.
    """
    fields, layer, grid = BUILD[name](xp)
    plan = arms.plan_step(fields, layer, grid,
                          licenses={LICENSE_COMPLEX: LICENCE},
                          subnormal_policy=POLICY, sources=(), fuse=True)
    assert plan.selected.get("step_D") == label
    assert plan.selected.get("update_E") == label
    assert "zero_metal_D" in fused_pairs.replaced_sub_steps(plan.plans)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_fusion_is_opt_in_on_this_seam_too(name, xp):
    """``fuse`` defaults to off, so the composition the coverage census measured is
    unchanged and these products add nothing nobody asked for."""
    fields, layer, grid = BUILD[name](xp)
    plan = arms.plan_step(fields, layer, grid,
                          licenses={LICENSE_COMPLEX: LICENCE},
                          subnormal_policy=POLICY, sources=())
    assert plan.selected.get("step_D") not in (
        "complex fused electric pair", "cylindrical fused electric pair")
