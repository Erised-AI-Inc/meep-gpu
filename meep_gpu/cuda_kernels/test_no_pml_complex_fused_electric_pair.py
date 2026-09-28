"""The complex NO-ABSORBER hand-CUDA fused pair: predicate, lift, protocol.

WHAT THIS FILE SETTLES. ``no_pml_complex_fused_electric_pair`` puts the
``cuda_complex_no_pml`` family on ``step_D -> update_E``. It occupies the board's
``D_to_E (cuda_complex_no_pml/complex no-PML curl, cuda_complex_no_pml/complex no-PML
stored E)`` cell -- 4 seam-instances, all four clearing the source seam on the driver
fact alone and all four carrying NOTHING in the seam -- and what has to be right, and
what a byte comparison on an admitted row can never see, is:

1. **The two DECISIONS this product had to make, and that its constants are the ones
   the measurement demands.** ``CARRIES_DEPOSIT_REPAIR`` is ``False`` and
   ``zero_metal_D`` is absent from ``REPLACES``: both are the OPPOSITE of every
   shipped sibling's answer, and each is measured rather than copied. The tests below
   pin the CONSEQUENCE of each (what the flag refuses; what the wall clause refuses)
   rather than the census number, because a merge-bar test may not read a gitignored
   artifact -- the numbers themselves are in the module's own constants, cited.

2. **Is a FOURTH product on this seam still DISJOINT from the other three?**
   ``install_fused_pairs`` leaves a seam UNFUSED when two products claim it, so an
   overlap costs slots rather than gaining them. Measured on four fixtures, not argued
   from three predicates' prose.

3. **Is the lift the certified text, on BOTH curl arms?** This is the first fused
   product whose curl half ships two kernels. Every corpus row on its cell takes the
   CONDUCTIVE one, so a weld that built only the plain arm would compile, pass every
   other leg and serve nothing -- and one that built the plain arm behind a predicate
   admitting a conductive run would be silently wrong on every word.

4. **Is D bound exactly once?** The constitutive half's ``minus_poles(g*, ...)`` and
   the certified launcher's binding of every UNUSED pole slot to the SOURCE pointer are
   TWO routes by which D could be bound a second time. Both are closed and both are
   measured here.

NOTHING HERE COMPILES OR LAUNCHES ANYTHING. The fused kernel's arithmetic is a DEVICE
claim and no test on this host may speak to it
(``parity/meep_gpu/gate_cuda_fused_complex_pairs.py --family no_pml_complex_electric``
is where that is measured); what is settled here is the predicate, the lift and the
protocol.
"""

from __future__ import annotations

import ast
import pathlib

import numpy
import pytest

from .. import deposit_repair, stepping
from ..dispersion import PolarizationState, Susceptibility
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from ..sources import GaussianEnvelope, VolumeSource
from ..triton_kernels.launch import NoopPlan
from . import arms, complex_emitter, fused_pairs
from .no_pml_complex_fused_electric_pair import FAMILY
from . import complex_no_pml_kernels as certified
from . import no_pml_complex_fused_electric_pair as product
from .registry import LICENSE_COMPLEX, LICENSE_COMPLEX_NO_PML
from .test_fused_pairs import _NumpyWearingCupysName

MODULE_PATH = pathlib.Path(product.__file__)

#: The expansion licence the predicate requires. NOT optional and never defaulted: the
#: arm is compiled into the binary at this seam, and asking with ``license=None``
#: returns the family's own named refusal on every configuration -- an answer about the
#: caller, recorded as if it were about the run. ``patterns`` carries the fifth
#: orientation ``update_P`` needs, so the SAME licence can drive the composer legs.
LICENCE = {"arm": "FMA_V1", "expansion": 1, "basis": "measured", "refusals": [],
           "policy_resolved": "keep",
           "patterns": {certified.ADE_PROBE_PATTERN: "FMA_V1"}}
POLICY = "keep"

ARMS = ("NAIVE", "FMA_V1")
CURL_ARMS = ("plain", "conductive")


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def build(xp, *, boundaries=("periodic", "periodic", "periodic"),
          k_point=(0.2, -0.35, 0.1), cell=(8.0, 9.0, 10.0), symmetry=(),
          conductive=True, poles=1, complex_storage=True, layer=None, seed=20260831):
    """A frozen ``(fields, layer, grid)`` triple this product admits.

    THE EXTENTS ARE DELIBERATELY UNEQUAL: a cube lets an index decomposition swap i and
    k and stay identical.

    ``enable_pml_storage`` IS NEVER CALLED and that is load-bearing rather than an
    omission -- it allocates the ``fu``/``f_w`` auxiliaries, and both certified halves
    refuse a run where one exists while the layer is inert.

    THE EPSILON IS ANISOTROPIC AND THE CONDUCTIVITY IS PER COMPONENT. On a vacuum
    fixture all three ``inv_eps`` pointers are ONE allocation of ones, so binding them
    in the wrong order or dropping the multiply outright is BIT-IDENTICAL and every
    mutation on that binding scores as uncaught while looking green. Three distinct
    inhomogeneous volumes are what make the binding observable at all -- and this
    launcher binds them a SECOND time, in the unused pole slots, so the check that they
    are distinct is doubly load-bearing here.
    """
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                k_point=k_point, symmetry=symmetry, xp=xp, courant=0.5)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    shape = tuple(int(n) for n in grid.shape)
    rng = numpy.random.default_rng(seed)

    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(0.2, 0.9, size=shape).astype(numpy.float32)
        inverse[component] = xp.asarray(numpy.ascontiguousarray(values))
        epsilon[component] = xp.asarray(
            numpy.ascontiguousarray((1.0 / values).astype(numpy.float32)))
    fields.set_epsilon_volumes(epsilon, inverse)
    distinct = len({_address(fields.inverse_epsilon_for(component))
                    for component in ("Ex", "Ey", "Ez")})
    assert distinct == 3, (
        f"the fixture installed {distinct} distinct inverse-epsilon allocations, not "
        f"3; every binding claim on this seam would be inert")

    if conductive:
        # PER COMPONENT, which is the mapping form `mp.Absorber` produces
        # (fields.py:755-759). One shared volume is legal and makes a component-aliasing
        # defect unobservable, so the fixture draws three graded ones.
        for setter, names in ((fields.set_d_conductivity, ("Dx", "Dy", "Dz")),
                              (fields.set_b_conductivity, ("Bx", "By", "Bz"))):
            setter({name: xp.asarray(numpy.ascontiguousarray(
                rng.uniform(0.05, 0.4, size=shape).astype(numpy.float32)))
                for name in names})
    for index in range(poles):
        sigma = {name: xp.asarray(numpy.ascontiguousarray(
            rng.uniform(0.1, 0.9, size=shape).astype(numpy.float32)))
            for name in ("Ex", "Ey", "Ez")}
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.05 + 0.4 * index, gamma=0.05 + 0.02 * index),
            sigma, grid, fields._field_dtype()))
    return fields, layer, grid


#: The module's own address reader, used rather than a second one here: a fixture that
#: measured aliasing with a different rule than the shipped check uses would be
#: measuring a different question.
_address = product.base_address


def covers(fields, layer, grid, sources=(), licence=None, policy=POLICY):
    return product.covers_no_pml_complex_fused_electric_pair(
        fields, layer, grid, sources, LICENCE if licence is None else licence, policy)


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


def source_text(arm="NAIVE", conductive=True):
    return product.no_pml_complex_fused_electric_pair_source(arm, conductive)


def _fused_body(arm="NAIVE", conductive=True, code_only=False):
    """The fused kernel's own text, past its entry point.

    SPLIT AT THE ENTRY POINT, not at the first ``\\n) {``: the certified prelude's own
    device helpers (``cshift_dn`` above all) close their parameter lists the same way,
    so a split at the first terminator would inspect the helper library instead.

    ``code_only`` strips whole-line ``//`` comments, for the checks that ask whether an
    IDENTIFIER survived. The comments this module splices in NAME the parameters they
    explain -- a reader has to be told that there is no ``f_w`` and no ``n_elem`` -- so a
    substring test over the raw text would fire on the prose that says they are absent.
    """
    entry = f"{product.kernel_name('conductive' if conductive else 'plain')}("
    body = source_text(arm, conductive).split(entry, 1)[1]
    if not code_only:
        return body
    return "\n".join(line for line in body.splitlines()
                     if not line.lstrip().startswith("//"))


class _ArrayPair:
    """Stands in for the fused kernel: curl + constitutive in one object.

    THE THING UNDER TEST IS THE PROTOCOL, NOT THE KERNEL -- the same stand-in the
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
# 1. THE TWO DECISIONS, AND THE WIRING THEY MAY NOT DRIFT FROM
# --------------------------------------------------------------------------

def test_the_flag_is_declared_False_and_is_passed_to_the_clause_that_reads_it():
    """A flag the seam clause is not handed is a claim nothing acts on.

    FALSE IS THE UNUSUAL ANSWER on this seam -- every other shipped electric product
    declares True -- so the constant, its spelling in the source (which the tree-walking
    allowlist in ``test_fused_pair_deposit_wiring`` reads) and its arrival at the clause
    are each read rather than assumed.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert product.CARRIES_DEPOSIT_REPAIR is False
    assert "CARRIES_DEPOSIT_REPAIR = False" in source
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source


def test_the_flag_at_False_refuses_every_deposit_the_installer_could_have_bracketed(xp):
    """THE SOUNDNESS OF THE FALSE, measured over a source matrix rather than argued.

    ``CARRIES_DEPOSIT_REPAIR = False`` is sound exactly when no configuration this
    predicate ADMITS can reach ``_install_fused_pair`` with a non-empty D seam -- if one
    could, the installer would build a ``NoopPlan`` over a launch that consumed a
    pre-injection D and report success. So the two answers are required to agree in
    both directions: admitted implies the seam is empty, and a non-empty seam implies
    refused.
    """
    fields, layer, grid = build(xp)
    cases = (
        (), (magnetic_source(grid),), (magnetic_source(grid, "Hx"),),
        (electric_source(grid),), (electric_source(grid, "Ex"),),
        (magnetic_source(grid), electric_source(grid)),
        (electric_source(grid), magnetic_source(grid, "Hy")),
    )
    for sources in cases:
        covered, why = covers(fields, layer, grid, sources)
        in_seam = deposit_repair.in_seam_sources(sources, "D")
        assert covered == (not in_seam), (sources, covered, why, in_seam)
        if not covered:
            assert "is electric" in why, why


def test_the_flag_is_load_bearing_and_flipping_it_changes_the_answer(xp, monkeypatch):
    """The null control on the test above. Without it, "electric sources are refused"
    could be true of a predicate that refuses them for some unrelated reason.

    FLIPPING THE FLAG TO True IS NOT A PROPOSAL. This module installs no
    ``LeadingRepairPlan``/``TrailingRepairPlan``, so True would be a claimed bracket
    nothing builds; the flip is here only to show that the shipped False is what
    produces the refusal.
    """
    fields, layer, grid = build(xp)
    sources = (electric_source(grid),)
    covered, why = covers(fields, layer, grid, sources)
    assert not covered and "is electric" in why, why

    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", True)
    covered, flipped = covers(fields, layer, grid, sources)
    assert flipped != why, (
        "flipping the flag changed nothing, so the shipped False is not what refuses "
        f"an electric deposit: {why}")
    assert "is electric" not in flipped, flipped
    # AND WHAT THE FLIP WOULD HAVE MET INSTEAD, which is the other half of why True is
    # not a proposal here: `deposit_repair.repairable` asks for the split-field history
    # this seam does not have, because there is no absorber to have written one.
    assert "f_w_Ex is not allocated" in flipped, flipped


def test_an_undeclared_source_set_is_still_a_refusal(xp):
    """IGNORANCE IS NEVER AN EMPTY SET. ``Fields`` does not hold the source list, so a
    predicate that inferred "no sources" from not being told would over-cover exactly
    the configurations the clause above exists to refuse."""
    fields, layer, grid = build(xp)
    covered, why = covers(fields, layer, grid, None)
    assert not covered and "was not declared" in why, why


def test_the_replaced_passes_are_a_contiguous_run_of_the_drivers_own_order(xp):
    """``REPLACES`` is a claim about what ONE launch performed, checked against
    ``FdtdDriver.step``'s SOURCE rather than against a list retyped here.

    THIS PRODUCT'S CLAIM IS STRONGER THAN ITS SIBLINGS'. They declare a run with
    ``zero_metal_D`` carried in registers and the two mirror fills refused; this one
    declares a CONTIGUOUS run of two, which is only honest if EVERY driver pass between
    the two consults is refused by name. So each intervening pass is looked up in the
    driver and then shown to be one the predicate refuses.
    """
    driver = pathlib.Path(stepping.__file__).with_name("driver.py").read_text(
        encoding="utf-8")
    step = driver.split("def step(", 1)[1].split("\n    def ", 1)[0]
    order = [pass_ for pass_ in ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                                 "fill_folded_far_ghosts_D", "update_E")
             if f"{pass_}(self.fields" in step]
    assert order == ["step_D", "fill_symmetry_bc_D", "zero_metal_D",
                     "fill_folded_far_ghosts_D", "update_E"]
    assert product.REPLACES == ("step_D", "update_E")
    positions = [order.index(pass_) for pass_ in product.REPLACES]
    assert positions == sorted(positions) == [0, 4]
    # And the injection sites, which are driver lines rather than named passes.
    assert "self._inject_electric_through_conductivity(electric, source_time)" in step
    assert "source.inject(fields, source_time)" in step.replace("self.", "")

    # EVERY PASS BETWEEN THEM IS REFUSED BY NAME, which is what makes the run
    # contiguous. Read as refusals from the predicate, one fixture per pass.
    for pass_ in order[1:-1]:
        assert pass_ in _REFUSAL_FIXTURE, (
            f"{pass_} sits inside this seam and no fixture below shows it refused; "
            f"REPLACES claims a contiguous run that nothing establishes")


#: Which fixture keyword makes each in-seam driver pass live, so the test above can
#: require a refusal for every one of them. A pass with no row here would sit inside
#: the seam unaccounted for.
_REFUSAL_FIXTURE = {
    "fill_symmetry_bc_D": "symmetry",
    "zero_metal_D": "boundaries",
    "fill_folded_far_ghosts_D": "symmetry",
}


def test_the_absorb_declaration_matches_the_arms_the_predicate_conjoins():
    """READ FROM THE PREDICATE by ``ast``, not assigned to it: if it ever conjoined
    different arms, absorbing those two slots would substitute a numerical product no
    arm admitted."""
    from . import registry  # noqa: PLC0415 - the table, read for its labels

    conjoined = ("covers_complex_no_pml_curl", "covers_complex_no_pml_stored_e")
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    predicate = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name.startswith("covers_"))
    called = {node.func.attr for node in ast.walk(predicate)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert set(conjoined) <= called

    by_predicate = {row["name"]: row["label"] for row in registry._TABLE}
    assert fused_pairs.FUSED_PAIR_ARMS[product.FAMILY] == (
        by_predicate[conjoined[0]], by_predicate[conjoined[1]])
    assert fused_pairs.FUSED_PAIR_ARMS[product.FAMILY] == (
        "complex no-PML curl", "complex no-PML stored E")


def test_the_product_is_wired_into_the_block_and_declared_not_registered():
    """A fused product bids at no slot, so it must be REACHABLE through the block and
    EXPLAINED in the arm table's own not-registered map -- a predicate that is in
    neither is invisible to ``plan_step``, which is a silent coverage loss."""
    from . import registry  # noqa: PLC0415

    assert fused_pairs.FUSED_PRODUCTS[product.FAMILY]["curl_slot"] == product.SLOT
    assert product.SLOT == "step_D"
    assert fused_pairs.FUSED_PRODUCTS[product.FAMILY]["module"] == MODULE_PATH.stem
    assert fused_pairs.FUSED_PAIR_SEAMS["step_D"] == ("update_E", "D")
    key = (f"{MODULE_PATH.stem}."
           f"covers_no_pml_complex_fused_electric_pair")
    assert key in registry.NOT_REGISTERED
    assert "step_D -> update_E" in registry.NOT_REGISTERED[key]


def test_the_kernel_label_is_the_bracket_form_of_the_two_real_symbols():
    """``KERNEL_NAME`` is a LABEL and the two symbols are ``KERNEL_KEYS``' values.

    The bracket spelling is this family's own (``registry._TABLE``'s
    ``fused_{slot}_no_pml_complex[_conductive]``) and the resolver is what keeps a
    label out of ``cp.RawKernel``, where it would fail three frames from the cause.
    """
    plain, conductive = (product.KERNEL_KEYS["plain"],
                         product.KERNEL_KEYS["conductive"])
    assert conductive == f"{plain}_conductive"
    assert product.KERNEL_NAME == f"{plain}[_conductive]"
    assert product.kernel_name("plain") == plain
    assert product.kernel_name("conductive") == conductive
    with pytest.raises(ValueError, match="plain"):
        product.kernel_name("NAIVE")
    # THE PARTITION COVERS BOTH SYMBOLS, on whichever side of it they sit. Until
    # 2026-09-11 both were UNCERTIFIED and this line pinned that; the campaign
    # ``certification.json:cuda_no_pml_complex_fused_electric_pair_2026-09-11_regate``
    # then released them on device (keep and flush legs, 14/14 cases identical,
    # 17/17 mutations caught) and they moved. What must stay true either way is that
    # every emitted symbol is accounted for exactly once -- an unnamed kernel is the
    # gap this assertion exists to close, and which side it is on is the gate's
    # business, not this file's.
    assert (set(product.CERTIFIED_KERNELS) | set(product.UNCERTIFIED_KERNELS)
            == set(product.KERNEL_KEYS.values()))
    assert not (set(product.CERTIFIED_KERNELS) & set(product.UNCERTIFIED_KERNELS))
    assert set(product.CERTIFIED_KERNELS) == set(product.KERNEL_KEYS.values())


# --------------------------------------------------------------------------
# 2. FOUR PRODUCTS ON ONE SEAM, AND THE DISJOINTNESS THAT LETS THEM COEXIST
# --------------------------------------------------------------------------

def _electric_seam_admitters(fields, layer, grid):
    """Which shipped products admit this configuration at ``step_D``, by name."""
    from . import complex_fused_electric_pair as complex_pml  # noqa: PLC0415
    from . import cylindrical_fused_electric_pair as dcyl  # noqa: PLC0415
    from . import fused_electric_pair as real  # noqa: PLC0415

    admitters = []
    if real.covers_fused_electric_pair(fields, layer, grid, ())[0]:
        admitters.append(real.FAMILY)
    if complex_pml.covers_complex_fused_electric_pair(
            fields, layer, grid, (), LICENCE, POLICY)[0]:
        admitters.append(complex_pml.FAMILY)
    if dcyl.covers_cylindrical_fused_electric_pair(
            fields, layer, grid, (), LICENCE, POLICY)[0]:
        admitters.append(dcyl.FAMILY)
    if covers(fields, layer, grid, ())[0]:
        admitters.append(product.FAMILY)
    return admitters


def _pml_fixture(xp, *, complex_storage=True, cylindrical=False):
    """One of the three shapes the OTHER three products serve."""
    if cylindrical:
        grid = Grid(resolution=1.0, cell_size=(9.0, 0.0, 11.0), cylindrical=True,
                    m=2, boundaries={"z": "metallic"}, courant=0.5, xp=xp)
        layer = PML(grid=grid, thickness={"x": (0, 2), "z": 2})
    else:
        grid = Grid(resolution=1.0, cell_size=(8.0, 9.0, 10.0),
                    boundaries=("metallic", "periodic", "periodic"), xp=xp)
        layer = PML(grid=grid, thickness=tuple(
            (0, 0) if grid.shape[axis] < 6 else (2, 2) for axis in range(3)))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def test_exactly_one_product_admits_each_of_the_four_electric_fixtures(xp):
    """TWO ADMITTERS ON ONE SEAM IS A REFUSAL, NOT A GAIN.

    Measured on the four shapes the corpus drives at this seam rather than read off the
    four predicates' prose: a real PML run, a Cartesian complex PML run, a Dcyl complex
    PML run, and -- the shape this product adds -- a Cartesian complex run with NO
    ABSORBER. The fourth fixture is the one the other three files could not build.
    """
    real = _pml_fixture(xp, complex_storage=False)
    assert _electric_seam_admitters(*real) == ["cuda_fused_electric_pair"]

    complex_pml = _pml_fixture(xp)
    assert _electric_seam_admitters(*complex_pml) == [
        "cuda_complex_fused_electric_pair"]

    dcyl = _pml_fixture(xp, cylindrical=True)
    assert _electric_seam_admitters(*dcyl) == [
        "cuda_cylindrical_fused_electric_pair"]

    for conductive in (False, True):
        no_pml = build(xp, conductive=conductive)
        assert _electric_seam_admitters(*no_pml) == [product.FAMILY], conductive


def test_the_absorber_is_the_partition_and_both_directions_are_named(xp):
    """The clause that makes the fourth candidate free. Each of the other three refuses
    an INACTIVE layer by name and this one requires exactly that, so the partition is a
    single boolean rather than a conjunction of storage and grid clauses."""
    from . import complex_fused_electric_pair as complex_pml  # noqa: PLC0415

    fields, layer, grid = build(xp)
    covered, why = complex_pml.covers_complex_fused_electric_pair(
        fields, layer, grid, (), LICENCE, POLICY)
    assert not covered and "no active PML layer" in why, why

    pml_fields, pml_layer, pml_grid = _pml_fixture(xp)
    covered, why = covers(pml_fields, pml_layer, pml_grid)
    assert not covered and "active absorber" in why, why


def test_an_inert_layer_is_admitted_because_stepping_takes_the_same_tail(xp):
    """A ZERO-THICKNESS PML is not an absorber: ``pml.is_active`` is False, the array
    path takes the plain tail, and refusing it would be a refusal with no line behind
    it. Swept because "inert is the same as absent" is otherwise an inference."""
    fields, _none, grid = build(xp)
    inert = PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
    assert not inert.is_active
    assert covers(fields, inert, grid)[0]


# --------------------------------------------------------------------------
# 3. THE PREDICATE — WHAT IT ADMITS, AND WHAT IT REFUSES BY NAME
# --------------------------------------------------------------------------

def test_a_conductivity_is_admitted_and_routes_to_the_conductive_arm(xp):
    """THE CLAUSE THIS PRODUCT WOULD HAVE BEEN WORTHLESS WITHOUT.

    Every corpus row on this product's cell records ``has_conductivity == True`` and
    the composer's own recorded arm is ``conductive`` at ``step_D``. A predicate that
    refused a conductivity -- as the PML complex sibling does, for a reason that is real
    THERE (the driver's ``_inject_electric_through_conductivity`` route) and vacuous
    here (that route needs an electric source, which this predicate refuses outright) --
    would serve 0 of 4.
    """
    fields, layer, grid = build(xp, conductive=True)
    assert fields.has_conductivity
    assert covers(fields, layer, grid)[0]
    assert product.no_pml_complex_curl_arm(fields, "step_D") == "conductive"

    lossless, layer, grid = build(xp, conductive=False)
    assert not lossless.has_conductivity
    assert covers(lossless, layer, grid)[0]
    assert product.no_pml_complex_curl_arm(lossless, "step_D") == "plain"


def test_the_arm_classifier_is_the_certified_familys_own(xp):
    """One classifier, so the arm the predicate admits and the arm the launch compiles
    cannot disagree. Delegated rather than re-derived, and checked by identity of
    answer on a fixture whose two objects differ only in the conductivity."""
    for conductive in (False, True):
        fields, _layer, _grid = build(xp, conductive=conductive)
        assert (product.no_pml_complex_curl_arm(fields, "step_D")
                == certified.complex_no_pml_curl_arm(fields, "step_D"))


def test_a_sub_step_whose_three_targets_disagree_is_refused_by_name(xp):
    """``_apply_curl`` reads the conductivity PER TARGET (stepping.py:508), so a
    sub-step with one lossy target takes two different tails in one launch -- which
    neither certified kernel implements and this weld therefore cannot either. The
    classifier RAISES on it; the predicate must refuse rather than let the raise out."""
    fields, layer, grid = build(xp, conductive=True)
    # Drop ONE target's coefficient, leaving the other two: `_apply_curl` reads
    # `condfac_for(target)` and would take the conductive tail on two components and the
    # plain one on the third, inside a single sub-step.
    assert all(fields.condfac_for(name) is not None
               for name in ("Dx", "Dy", "Dz")), "the fixture is not lossy"
    fields._condfac = {name: value for name, value in fields._condfac.items()
                       if name != "Dx"}
    assert fields.condfac_for("Dx") is None
    covered, why = covers(fields, layer, grid)
    assert not covered and "disagree about conductivity" in why, why


def test_a_walled_run_is_refused_by_name_and_names_the_pass_it_does_not_carry(xp):
    """THE DECISION ``REPLACES`` RESTS ON. ``zero_metal_D`` (driver.py:3310) runs inside
    this seam on a walled run and this pair does not carry it, so the run is refused --
    which is what makes ``REPLACES`` a contiguous run of two rather than a claim that a
    pass did nothing when it did."""
    for boundaries in (("metallic", "periodic", "periodic"),
                       ("periodic", "metallic", "periodic"),
                       ("periodic", "periodic", "metallic"),
                       ("metallic", "metallic", "metallic")):
        fields, layer, grid = build(xp, boundaries=boundaries, k_point=(0.0, 0.0, 0.0))
        assert grid.has_metallic
        covered, why = covers(fields, layer, grid)
        assert not covered, boundaries
        assert "zero_metal_D" in why and "driver.py:3310" in why, why


def test_the_wall_refusal_is_this_products_own_and_not_a_halfs(xp):
    """A clause inherited from a half could be licensed away by a device verdict on
    that half without anyone revisiting ``REPLACES``. Both certified halves ADMIT a
    walled run -- measured -- so the refusal above is this file's alone."""
    fields, layer, grid = build(xp, boundaries=("metallic", "metallic", "metallic"),
                                k_point=(0.0, 0.0, 0.0))
    assert certified.covers_complex_no_pml_curl(
        fields, layer, grid, "step_D", LICENCE, POLICY)[0]
    assert certified.covers_complex_no_pml_stored_e(
        fields, layer, grid, LICENCE, POLICY)[0]
    assert not covers(fields, layer, grid)[0]


def test_a_grid_that_cannot_say_which_axes_are_walled_is_refused(xp):
    """An unanswerable grid is refused rather than read as unwalled: treating a missing
    accessor as "no wall" would put a launch where the driver clears a plane."""
    fields, layer, grid = build(xp)

    class _Mute:
        def __getattr__(self, item):
            if item == "has_metallic":
                raise AttributeError(item)
            return getattr(grid, item)

    covered, why = covers(fields, layer, _Mute())
    assert not covered and "has_metallic" in why, why


@pytest.mark.parametrize("axis,mirror", ((0, "x"), (1, "y"), (2, "z")))
def test_a_folded_axis_is_refused_and_the_products_own_clause_names_both_fills(
        xp, axis, mirror, monkeypatch):
    """``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` run inside this seam on a
    folded grid and this pair carries neither. The curl half already refuses a fold, so
    this file's own clause is unreachable today -- and that is why it is written and why
    it is REACHED here, with the half's refusal monkeypatched away."""
    fields, layer, grid = build(xp, symmetry=(Mirror(mirror),), k_point=(0.0, 0.0, 0.0))
    assert grid.is_mirrored(axis)
    covered, why = covers(fields, layer, grid)
    assert not covered, why

    monkeypatch.setattr(certified, "covers_complex_no_pml_curl",
                        lambda *a, **k: (True, "covered"))
    monkeypatch.setattr(certified, "covers_complex_no_pml_stored_e",
                        lambda *a, **k: (True, "covered"))
    covered, why = covers(fields, layer, grid)
    assert not covered
    assert "fill_symmetry_bc_D" in why and "fill_folded_far_ghosts_D" in why, why


def test_a_stored_extent_that_disagrees_with_the_walked_shape_is_refused(xp):
    """THE DROPPED BOUNDS GUARD. This kernel drops ``update_E``'s own ``n_elem`` guard
    and runs the constitutive half over the CURL's range, which is only sound where
    ``grid.stored_cells``, ``Dx.shape`` and ``Ex.shape`` are one tuple."""
    fields, layer, grid = build(xp)

    class _Shrunk:
        def stored_cells(self, axis):
            return int(grid.stored_cells(axis)) - (1 if axis == 1 else 0)

        def __getattr__(self, item):
            return getattr(grid, item)

    covered, why = covers(fields, layer, _Shrunk())
    assert not covered and "bounds guard" in why, why


def test_an_inverse_permittivity_volume_of_the_wrong_extent_is_refused(xp):
    """The kernel indexes ``inv_eps`` with THIS launch's flat index, and binds it a
    second time in the unused pole slots."""
    fields, layer, grid = build(xp)
    short = numpy.ones(tuple(int(n) for n in grid.shape)[:-1] + (1,),
                       dtype=numpy.float32)
    fields._inv_eps_components = dict(fields._inv_eps_components)
    fields._inv_eps_components["Ey"] = short
    covered, why = covers(fields, layer, grid)
    assert not covered and "inverse_epsilon_for" in why, why


def test_the_licence_is_required_and_never_defaulted(xp):
    """The arm is compiled into the binary at this seam and there is no later rung to
    check it at, so ``license=None`` is a refusal rather than a default."""
    fields, layer, grid = build(xp)
    # The same configuration, with and without the verdict. The licence is the ONLY
    # thing that differs, so the refusal cannot be about anything else.
    assert covers(fields, layer, grid)[0]
    covered, why = product.covers_no_pml_complex_fused_electric_pair(
        fields, layer, grid, (), None, POLICY)
    assert not covered, why
    assert why.startswith("curl half:"), why
    # AND THE POLICY: a licence resolved under one float32 subnormal policy does not
    # license a run under the other.
    covered, why = product.covers_no_pml_complex_fused_electric_pair(
        fields, layer, grid, (), LICENCE, "meep_x86_flush")
    assert not covered, why


def test_an_offdiagonal_row_is_refused_through_the_constitutive_half(xp):
    """``has_offdiagonal_epsilon`` turns ``update_E`` into MEEP's tensor row product -- a
    four-point transverse average over the PARTNER components' volumes, which is a
    STENCIL and not a different coefficient. The half refuses it and the conjunction
    reports which half said so."""
    fields, layer, grid = build(xp)
    shape = tuple(int(n) for n in grid.shape)
    fields._chi1inv_offdiagonal = {
        "Ex": {"Ey": numpy.full(shape, 0.05, dtype=numpy.float32)}}
    assert fields.has_offdiagonal_epsilon
    covered, why = covers(fields, layer, grid)
    assert not covered
    assert why.startswith("constitutive half:") and "off-diagonal" in why, why


def test_pml_storage_mode_with_an_inert_layer_is_refused_through_the_curl_half(xp):
    """``Fields.enable_pml_storage`` is a one-way switch that also changes what
    ``get_H`` returns; with an inert layer the curl would difference a frozen H."""
    fields, layer, grid = build(xp)
    fields.enable_pml_storage()
    covered, why = covers(fields, layer, grid)
    assert not covered and "PML storage mode" in why, why


def test_real_storage_is_refused_through_the_curl_half(xp):
    """The partition with the real no-absorber families: a run that is neither
    ``force_complex_fields`` nor Bloch-phased belongs to the real kernels."""
    fields, layer, grid = build(xp, complex_storage=False, k_point=(0.0, 0.0, 0.0))
    covered, why = covers(fields, layer, grid)
    assert not covered and "real float32 storage" in why, why


def test_a_run_with_no_pole_is_admitted_and_is_not_a_no_op(xp):
    """With ``stores_E`` True the array path still writes ``E = D * inv_eps``
    (stepping.py:1022), which is this kernel with every ``np`` zero. Admitting it is
    what keeps the pole-free rows of this family inside the product."""
    fields, layer, grid = build(xp, poles=0)
    assert not fields.polarizations
    assert covers(fields, layer, grid)[0]
    _bank, counts = product.pole_bank_bindings(fields)
    assert counts == (0, 0, 0)


def test_more_poles_than_the_compiled_slots_is_refused_by_the_shared_number(xp):
    """The ceiling is ``complex_no_pml_kernels.MAX_POLES`` and the two routes to it --
    the predicate's clause and the launcher's -- may not drift."""
    fields, layer, grid = build(xp, poles=certified.MAX_POLES + 1)
    covered, why = covers(fields, layer, grid)
    assert not covered and f"MAX_POLES={certified.MAX_POLES}" in why, why
    with pytest.raises(ValueError, match=f"MAX_POLES={certified.MAX_POLES}"):
        product.pole_bank_bindings(fields)


# --------------------------------------------------------------------------
# 4. THE LIFT — THE CERTIFIED TEXT, AND THE EDITS THAT ARE NOT VERBATIM
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("conductive", (False, True))
def test_the_shared_flux_density_is_bound_exactly_once(arm, conductive):
    """THE ALIASING HAZARD, ROUTE 1. Two pointers to one allocation with a
    ``__restrict__`` promise on one of them is UB NVRTC miscompiles without a
    diagnostic, so the constitutive half must have no D source pointers at all -- which
    is only true if the seam removed all three ``minus_poles(g*)`` sources."""
    after_entry = _fused_body(arm, conductive)
    signature = after_entry.split("\n) {", 1)[0]
    for target in ("f0", "f1", "f2"):
        bound = signature.count(f" {target},") + signature.count(f" {target}\n")
        assert bound == 1, f"{target} is bound {bound} times in the signature"
    body = after_entry.split("\n) {", 1)[1]
    for index in ("0", "1", "2"):
        assert f"minus_poles(g{index}," not in body, (
            f"the constitutive source of component {index} survived the seam rewrite; "
            f"D would need binding a second time")
        assert f"minus_poles_reg(d{index}," in body


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("conductive", (False, True))
def test_the_seam_reads_the_register_and_keeps_the_certified_operand_order(
        arm, conductive):
    """``(D - sum P) * inv_eps`` with the SOURCE ON THE LEFT (stepping.py:1013-1014), and
    the register in place of the reload. The enclosing store, the multiply and its
    operand order are lifted rather than retyped, so a reordered operand would be an
    edit with no ``LIFT_EDITS`` row."""
    source = source_text(arm, conductive)
    for index, stem in zip(("0", "1", "2"), ("a", "b", "c")):
        slots = ", ".join(f"{stem}{slot}" for slot in range(certified.MAX_POLES))
        assert (f"    cf_store(h{index}, idx, mul_field_left(\n"
                f"        minus_poles_reg(d{index}, {slots}, idx, np{index}),"
                in source)
        assert f"        inv_eps_{index}[idx]));" in source


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("conductive", (False, True))
def test_the_pole_subtraction_is_left_to_right_and_never_pre_summed(arm, conductive):
    """REGISTRATION ORDER IS BIT-LOAD-BEARING. ``((D - P0) - P1)`` against
    ``D - (P0 + P1)`` differs on 50279 of 400000 float32 words at two poles (measured,
    ``test_complex_no_pml.py``), so the eight guarded subtractions are lifted whole."""
    source = source_text(arm, conductive)
    for slot in range(certified.MAX_POLES):
        assert (f"    if (np > {slot}) s = cf_sub(s, cf_load(p{slot}, idx));"
                in source)
    assert "cf_add(cf_load(p" not in source, "the poles were pre-summed"


@pytest.mark.parametrize("arm", ARMS)
def test_the_conductive_tail_keeps_the_three_operations_and_their_order(arm):
    """THE ORDER IS THE ARITHMETIC. ``field *= condfac; field -= curl; field *= condinv``
    reassociated to ``((field - curl) * condfac) * condinv`` differs on 400000 of 400000
    float32 words (measured). Only the last statement's result is named."""
    source = source_text(arm, conductive=True)
    assert ("    cf t = mul_field_left(cf_load(f, idx), condfac);\n"
            "    t = cf_sub(t, curl);\n") in source
    assert ("    cf value = mul_field_left(t, condinv);\n"
            "    cf_store(f, idx, value);\n"
            "    return value;\n") in source
    for index in ("0", "1", "2"):
        assert (f"        d{index} = conductive_apply_reg(f{index}, idx, curl, "
                f"condfac_{index}[idx], condinv_{index}[idx]);") in source


@pytest.mark.parametrize("arm", ARMS)
def test_the_plain_tail_is_the_direct_subtraction_and_not_the_split_field_binding(arm):
    """``target -= curl`` written directly. The certified split-field body reaches the
    same value through ``fu`` and unit coefficient columns, and that binding is NOT
    exact: with target = -0.0 and curl = +0.0 it yields +0.0 where the array path
    yields -0.0."""
    source = source_text(arm, conductive=False)
    assert ("    cf value = cf_sub(cf_load(f, idx), curl);\n"
            "    cf_store(f, idx, value);\n"
            "    return value;\n") in source
    for index in ("0", "1", "2"):
        assert (f"        d{index} = no_pml_apply_reg(f{index}, idx, curl);"
                in source)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("conductive", (False, True))
def test_the_curl_stencil_and_its_masks_survive_unchanged(arm, conductive):
    """The arithmetic this family reuses. Three stencil lines and six metallic masks
    survive the tail substitution, which is what makes "the curl itself is unchanged" a
    property of the build rather than a claim in a docstring."""
    body = _fused_body(arm, conductive)
    stencil = ("        cf curl = mul_coefficient_left(dtdx, "
               "cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));")
    assert body.count(stencil) == 3
    assert body.count("curl = cf_zero();") == 6
    # SIX BACKWARD SHIFTS AND NO FORWARD ONE, checked on the BODY: the certified
    # prelude DEFINES both helpers, so the same count over the whole source would read
    # the helper library rather than the kernel.
    assert body.count("cshift_dn(") == 6 and "cshift_up(" not in body


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("conductive", (False, True))
def test_the_split_field_machinery_is_absent_rather_than_neutralised(arm, conductive):
    """NO ``fu``, NO ``f_w``, NO coefficient vector, NO ownership mask and NO second
    index decomposition. Each is a thing the PML sibling carries and this seam does
    not, and each is checked on the emitted BODY -- the certified prelude carries
    ``pml_apply`` and ``constitutive_apply`` as dead code by design."""
    body = _fused_body(arm, conductive, code_only=True)
    for absent in ("kms_", "kps_", "sinv_", "f_w", "fu_", "n_elem",
                   "pml_apply(", "constitutive_apply("):
        assert absent not in body, f"{absent!r} reached the fused body"
    assert body.count("int idx = blockIdx.x") == 1
    assert body.count("return;") == 1


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("conductive", (False, True))
def test_the_three_registers_are_declared_before_the_certified_braced_blocks(
        arm, conductive):
    """A value declared inside a braced component block does not outlive it, so the
    three carried registers are hoisted above them."""
    body = _fused_body(arm, conductive)
    for index in ("0", "1", "2"):
        assert f"    cf d{index} = cf_zero();\n" in body
        assert (body.index(f"    cf d{index} = cf_zero();")
                < body.index("    // Target 0:"))


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("conductive", (False, True))
def test_the_emitted_source_is_pure_ascii_and_depends_on_both_arms(arm, conductive):
    """PURE ASCII is a compile requirement, and the four sources are four strings: a
    memo or a digest that collapsed two of them would hand back the wrong bytes."""
    source = source_text(arm, conductive)
    source.encode("ascii")
    assert len(set(product.device_sources().values())) == 4
    assert source == product.device_sources()[f"{arm}/"
                                              f"{'conductive' if conductive else 'plain'}"]
    other = source_text("FMA_V1" if arm == "NAIVE" else "NAIVE", conductive)
    assert other != source, "the expansion arm did not change the emitted bytes"


def test_the_unused_pole_slots_bind_inv_eps_and_never_the_shared_flux_density(xp):
    """THE ALIASING HAZARD, ROUTE 2, and the one place this launcher deviates from the
    certified one. ``update_E_complex_no_pml_stored`` binds every unused slot to the
    SOURCE pointer, which is D; doing that here would bind D a second time."""
    fields, _layer, _grid = build(xp, poles=1)
    bank, counts = product.pole_bank_bindings(fields)
    assert counts == (1, 1, 1)
    assert len(bank) == 3 * certified.MAX_POLES
    displacement = {_address(fields.Dx), _address(fields.Dy), _address(fields.Dz)}
    inverse = [_address(volume)
               for volume in product.inverse_epsilon_bindings(fields)]
    for component in range(3):
        used = bank[component * certified.MAX_POLES]
        assert _address(used) == _address(
            fields.polarizations[0].P[("Ex", "Ey", "Ez")[component]])
        for slot in range(1, certified.MAX_POLES):
            spare = bank[component * certified.MAX_POLES + slot]
            assert _address(spare) == inverse[component]
            assert _address(spare) not in displacement


@pytest.mark.parametrize("conductive", (False, True))
def test_the_binding_check_finds_the_flux_density_bound_twice(xp, conductive, monkeypatch):
    """``assert_disjoint_bindings`` is the claim the whole module rests on, so it is
    armed: with the spare slots pointed at D -- the certified launcher's own choice --
    it must RAISE naming the second binding, and it must report a nonzero count of
    allocations inspected on the shipped path."""
    fields, _layer, _grid = build(xp, conductive=conductive, poles=1)
    arm = "conductive" if conductive else "plain"
    inspected = product.assert_disjoint_bindings(fields, arm)
    assert inspected >= 12, inspected

    original = product.pole_bank_bindings

    def sourcelike(fields_, poles=None):
        # THE CERTIFIED LAUNCHER'S OWN CHOICE, planted: it binds every unused slot to
        # the SOURCE pointer, which on this seam is D. The bank is un-viewed here, so
        # the plant is the array itself and the check reads the same address either
        # way -- a view shares its base's pointer.
        bank, counts = original(fields_, poles)
        spoiled = list(bank)
        spoiled[1] = fields_.Dx
        return tuple(spoiled), counts

    monkeypatch.setattr(product, "pole_bank_bindings", sourcelike)
    with pytest.raises(ValueError, match="bound a second time"):
        product.assert_disjoint_bindings(fields, arm)


def test_the_lift_edits_are_data_and_every_one_is_complete():
    """``LIFT_EDITS`` is what a gate asserts instead of a docstring, so every row must
    carry all three fields and a non-empty reason."""
    assert isinstance(product.LIFT_EDITS, tuple) and len(product.LIFT_EDITS) >= 8
    for edit in product.LIFT_EDITS:
        assert set(edit) == {"line", "became", "why"}, edit
        assert edit["line"] and edit["why"], edit
        assert edit["line"] != edit["became"], edit


@pytest.mark.parametrize("anchor", (
    "_PLAIN_APPLY_SIGNATURE", "_PLAIN_APPLY_STORE",
    "_CONDUCTIVE_APPLY_SIGNATURE", "_CONDUCTIVE_APPLY_STORE",
    "_MINUS_POLES_SIGNATURE", "_MINUS_POLES_LOAD",
    "_CONSTITUTIVE_HEAD", "_DECODE_END",
))
def test_the_splice_raises_rather_than_emitting_when_an_anchor_stops_matching(
        anchor, monkeypatch):
    """EVERY ANCHOR IS AN EXACT LINE OF CERTIFIED TEXT. If one stops matching, the
    certified string changed and splicing around it would produce a kernel that
    compiles and is quietly not the certified arithmetic. Each is broken in turn and
    the emitter is required to refuse."""
    monkeypatch.setattr(product, anchor,
                        getattr(product, anchor) + "// no such line\n")
    with pytest.raises(AssertionError):
        source_text("NAIVE", conductive=True)
    with pytest.raises(AssertionError):
        source_text("NAIVE", conductive=False)


def test_the_tail_capture_anchors_are_the_certified_calls(monkeypatch):
    """The two tail sites are the other half of the anchor set and are matched by
    whole line, so a certified body that moved a call onto two lines is a named failure
    rather than a partial substitution."""
    plain = certified.kernel_source("step_D", "NAIVE")
    conductive = certified.kernel_source("step_D_conductive", "NAIVE")
    for index in ("0", "1", "2"):
        assert f"        no_pml_apply(f{index}, idx, curl);" in plain
        assert (f"        conductive_apply(f{index}, idx, curl, "
                f"condfac_{index}[idx], condinv_{index}[idx]);") in conductive


def test_the_emitter_refuses_a_non_boolean_curl_arm():
    """The two tails are different arithmetic; a truthy value is not a choice."""
    with pytest.raises(TypeError, match="must be a bool"):
        product.no_pml_complex_fused_electric_pair_source("NAIVE", 1)


def test_the_launcher_binds_the_inverse_permittivity_through_the_per_component_reader():
    """``Fields.inverse_epsilon_for`` and NEVER ``fields.inv_eps``, which is the Ez view
    (fields.py:1259-1260) -- binding it for all three is the defect the twelve
    uncertified complex kernels in ``step_curl_kernels`` carry."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute)}
    assert "inverse_epsilon_for" in attributes
    assert "inv_eps" not in attributes
    assert product._ELECTRIC == ("Ex", "Ey", "Ez")


# --------------------------------------------------------------------------
# 5. THE PLAN — WHAT THE RESOLVER BINDS
# --------------------------------------------------------------------------

#
# THE RESOLVERS CANNOT BE EXECUTED ON THIS HOST and are read from the SOURCE instead,
# which is a deliberate second-best with a reason: ``complex_pml_kernels`` imports CuPy
# at module scope, so every resolver that binds a phase table refuses here by name. Both
# arguments below are ALSO armed as device mutations in
# ``parity/meep_gpu/gate_cuda_fused_complex_pairs.py``; what these hold is the half a
# laptop can hold, and they are here because each compiles, runs, and is a different
# engine -- the class of defect that never turns anything red on its own.

def _call_arguments(module, function, callee):
    """Every call to ``callee`` inside ``function``, as its literal arguments."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == function):
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.Call)
                    and getattr(child.func, "attr",
                                getattr(child.func, "id", None)) == callee):
                found.append([
                    argument.value if isinstance(argument, ast.Constant) else None
                    for argument in child.args])
    assert found, f"{function} makes no call to {callee}"
    return found


def test_the_resolver_binds_the_BACKWARD_bloch_table():
    """``step_D`` takes ``cshift_dn`` and the CONJUGATED Bloch factor
    (``complex_emitter.KERNELS['step_D'][1]``). The forward table compiles, runs, and is
    the wrong phase on every phased row rather than a crash.

    Held on BOTH sites the argument appears -- the module's own ``run_...`` and the
    composer's plan builder -- because a resolver corrected in one place and not the
    other is two engines wearing one name.
    """
    for module, function in (
            (product, "run_no_pml_complex_fused_electric_pair"),
            (fused_pairs, "_no_pml_complex_fused_electric_pair_plan")):
        for arguments in _call_arguments(module, function, "bloch_phase_arguments"):
            assert arguments[-1] is True, (
                f"{function} asks bloch_phase_arguments for the FORWARD table; "
                f"step_D takes the conjugate")
    # And the two tables really are different, so the assertion is not vacuous.
    assert complex_emitter.KERNELS["step_D"][1] is True
    assert complex_emitter.KERNELS["step_B"][1] is False


def test_the_plan_builder_resolves_the_curl_arm_and_binds_no_table_or_wall():
    """THE ONE RESOLVER ON THIS TRACK THAT MUST READ AN ARM. Every corpus row on this
    cell takes the conductive tail, so a builder that defaulted to the plain one would
    serve the whole cell with the wrong arithmetic rather than serving nothing.

    AND NO ``tables``/``walls`` KEY: this pair reads no coefficient vector and carries
    no wall pass, and a resolver carrying an empty one would suggest the launch binds
    something it does not.
    """
    source = pathlib.Path(fused_pairs.__file__).read_text(encoding="utf-8")
    builder = source.split("def _no_pml_complex_fused_electric_pair_plan(", 1)[1]
    builder = builder.split("\ndef ", 1)[0]
    assert '"curl_arm": no_pml_complex_curl_arm(ctx.fields, "step_D")' in builder
    assert '"expansion": ctx.license_for(LICENSE_COMPLEX_NO_PML)["arm"]' in builder
    assert '"tables"' not in builder and '"walls"' not in builder
    # THE LICENCE IS THIS FAMILY'S OWN and not the PML complex one; asking with the
    # wrong verdict would be asking about a different arm.
    assert "LICENSE_COMPLEX_NO_PML" in builder and "LICENSE_COMPLEX)" not in builder


@pytest.mark.parametrize("conductive", (False, True))
def test_the_arm_the_builder_would_resolve_is_the_arm_the_fields_declare(xp, conductive):
    """The runtime half of the test above: the function the builder calls, asked on the
    two fixtures, answers what their conductivity says."""
    fields, _layer, _grid = build(xp, conductive=conductive)
    assert (product.no_pml_complex_curl_arm(fields, "step_D")
            == ("conductive" if conductive else "plain"))


def test_the_plan_declares_the_two_passes_it_performs_and_both_slots():
    """Read from the builder's own tail rather than by building one: constructing the
    plan needs no device, but the ``CudaFusedPairPlan`` it returns is checked here for
    the three declarations a wrapper later reads off it."""
    source = pathlib.Path(fused_pairs.__file__).read_text(encoding="utf-8")
    builder = source.split("def _no_pml_complex_fused_electric_pair_plan(", 1)[1]
    builder = builder.split("\ndef ", 1)[0]
    assert "replaces=tuple(REPLACES)" in builder
    assert 'slots=("step_D", "update_E")' in builder
    assert "kernel_label=KERNEL_NAME" in builder
    assert "family=FAMILY" in builder
    assert product.REPLACES == ("step_D", "update_E")


# --------------------------------------------------------------------------
# 6. THE TWO-CONSULT PROTOCOL, ON A SEAM THAT CARRIES NOTHING
# --------------------------------------------------------------------------

def _protocol_fixture(cell=(1.2, 1.2, 0.0), seed=20260831, conductive=True):
    """A real-NumPy ``(grid, fields)`` pair for the numeric protocol leg.

    NOT the coverage fixture: this one runs the ARRAY PATH, so it must not wear CuPy's
    name. There is no layer at all -- ``None``, which is what ``stepping`` takes on a
    run with no absorber.
    """
    dimensions = sum(1 for extent in cell if extent > 0.0)
    grid = Grid(resolution=12.0, cell_size=cell, dimensions=dimensions)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    shape = tuple(int(n) for n in grid.shape)
    rng = numpy.random.default_rng(seed)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(0.2, 0.9, size=shape).astype(numpy.float32)
        inverse[component] = values
        epsilon[component] = (1.0 / values).astype(numpy.float32)
    fields.set_epsilon_volumes(epsilon, inverse)
    if conductive:
        for setter, names in ((fields.set_d_conductivity, ("Dx", "Dy", "Dz")),
                              (fields.set_b_conductivity, ("Bx", "By", "Bz"))):
            setter({name: rng.uniform(0.05, 0.4, size=shape).astype(numpy.float32)
                    for name in names})
    sigma = {name: rng.uniform(0.1, 0.9, size=shape).astype(numpy.float32)
             for name in ("Ex", "Ey", "Ez")}
    fields.polarizations.append(PolarizationState(
        Susceptibility(frequency=1.05, gamma=0.05), sigma, grid,
        fields._field_dtype()))
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        array = getattr(fields, name)
        host = numpy.empty(shape, dtype=numpy.complex64)
        host.real = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
        host.imag = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
        array[...] = host
    for state in fields.polarizations:
        for component in state.driven():
            for attribute in ("P", "P_prev"):
                array = getattr(state, attribute)[component]
                host = numpy.empty(shape, dtype=numpy.complex64)
                host.real = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
                host.imag = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
                array[...] = host
    return grid, fields


def _state_words(fields):
    """Every array a complete step touches, as raw float32 words, by slot name."""
    words = {}
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        words[name] = numpy.ascontiguousarray(
            numpy.asarray(getattr(fields, name))).view(numpy.float32).ravel().copy()
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            for attribute in ("P", "P_prev"):
                words[f"{attribute}[{index}][{component}]"] = numpy.ascontiguousarray(
                    numpy.asarray(getattr(state, attribute)[component])
                ).view(numpy.float32).ravel().copy()
    return words


def _differing(left, right):
    assert set(left) == set(right), (sorted(left), sorted(right))
    return {name: int((left[name] != right[name]).sum())
            for name in sorted(left)
            if not numpy.array_equal(left[name].view(numpy.uint32),
                                     right[name].view(numpy.uint32))}


def _driver_order(fields, layer, sources, when, dt):
    """The driver's D-seam order, verbatim (driver.py:3300-3313)."""
    stepping.step_D(fields, layer)
    for source in sources:
        source.inject(fields, when + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, layer)


@pytest.mark.parametrize("cell", ((1.2, 1.2, 0.0), (1.0, 1.0, 1.0)),
                         ids=("2d", "3d"))
@pytest.mark.parametrize("conductive", (False, True))
def test_the_two_slots_consulted_as_the_driver_consults_them_match_the_array_path(
        cell, conductive):
    """THE ONE THAT MATTERS, and the only numeric claim this file makes.

    Two independently seeded copies of the same COMPLEX-STORAGE, NO-ABSORBER,
    DISPERSIVE state. One is stepped in the DRIVER's order. The other is stepped the
    way a fused pair owning both consults is: the whole seam closed at the first
    consult, the driver's own in-seam passes run (all three of which are no-ops on this
    grid, which is exactly the claim ``REPLACES`` makes), and the cheap sentinel at the
    second. Every word of every array is compared as raw bytes, so "close" is not a
    passing answer.
    """
    grid_a, fields_a = _protocol_fixture(cell=cell, conductive=conductive)
    grid_b, fields_b = _protocol_fixture(cell=cell, conductive=conductive)
    assert fields_b.Dz.dtype == numpy.complex64, "the fixture is not complex storage"
    assert not _differing(_state_words(fields_a), _state_words(fields_b)), (
        "the two copies did not start equal")

    plans, selected = {}, {}
    inner = _ArrayPair(fields_b, None, product.REPLACES)
    fused_pairs._install_fused_pair(
        plans, selected, fields_b, None, (), "D", "step_D", "update_E",
        "complex no-absorber fused electric pair", inner)
    assert plans["step_D"] is inner
    assert isinstance(plans["update_E"], NoopPlan)

    dt = grid_a.dt
    for step in range(8):
        _driver_order(fields_a, None, (), step * dt, dt)
        stepping.update_P(fields_a, None)
        plans["step_D"].run()
        stepping.fill_symmetry_bc_D(fields_b)
        stepping.zero_metal_D(fields_b)
        stepping.fill_folded_far_ghosts_D(fields_b)
        plans["update_E"].run()
        stepping.update_P(fields_b, None)

    differing = _differing(_state_words(fields_a), _state_words(fields_b))
    assert not differing, differing
    assert inner.runs == 8
    # THE CASE MOVED SOMETHING. A run whose fields never changed would match whatever
    # the protocol did, which is the vacuity this whole track refuses.
    fresh = _protocol_fixture(cell=cell, conductive=conductive)[1]
    assert _differing(_state_words(fresh), _state_words(fields_b)), (
        "eight steps changed no word; the comparison above proves nothing")


def test_the_seam_the_pair_swallows_really_is_empty_on_an_admitted_run(xp):
    """``REPLACES`` claims the two consults are CONTIGUOUS. Measured: on a run this
    predicate admits, each of the three in-seam driver passes leaves every word alone,
    so there is nothing between the halves for the launch to have skipped."""
    _grid, fields = _protocol_fixture()
    before = _state_words(fields)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    assert not _differing(before, _state_words(fields))


def test_a_deposit_that_reached_this_pair_anyway_would_diverge():
    """The null control on the source refusal. Without it, "an electric source is
    refused" would be a clause with no consequence measured behind it."""
    grid_a, fields_a = _protocol_fixture()
    grid_b, fields_b = _protocol_fixture()
    src_a = [electric_source(grid_a)]
    src_b = [electric_source(grid_b)]
    assert src_b[0]._n_source_points, "the case deposits nothing and cannot discriminate"
    inner = _ArrayPair(fields_b, None, product.REPLACES)
    dt = grid_a.dt
    for step in range(8):
        _driver_order(fields_a, None, src_a, step * dt, dt)
        inner.run()
        for source in src_b:
            source.inject(fields_b, step * dt + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields_b)
        stepping.zero_metal_D(fields_b)
        stepping.fill_folded_far_ghosts_D(fields_b)
    differing = _differing(_state_words(fields_a), _state_words(fields_b))
    assert differing, ("a fused launch spanning an electric deposit matched the "
                       "driver, so this product's source refusal has no consequence")


def test_the_second_slot_is_the_cheap_sentinel_and_never_a_repair_plan(xp):
    """``CARRIES_DEPOSIT_REPAIR`` is False and nothing brackets this launch, so the
    installer must leave the ``NoopPlan`` in the second slot on every configuration the
    predicate admits -- which the source matrix above showed is every configuration
    with an empty D seam."""
    _grid, fields = _protocol_fixture()
    plans, selected = {}, {}
    label = "complex no-absorber fused electric pair"
    fused_pairs._install_fused_pair(
        plans, selected, fields, None, (), "D", "step_D", "update_E", label,
        _ArrayPair(fields, None, product.REPLACES))
    assert isinstance(plans["update_E"], NoopPlan)
    assert not isinstance(plans["step_D"], deposit_repair.LeadingRepairPlan)
    assert selected["step_D"] == selected["update_E"] == label


# --------------------------------------------------------------------------
# 7. THE BLOCK, DRIVEN THROUGH THE COMPOSER
# --------------------------------------------------------------------------

@pytest.mark.parametrize("conductive", (False, True))
def test_the_seam_fuses_and_no_other_product_claims_it(xp, conductive):
    """Through ``plan_step(fuse=True)``, which is the only route a caller has.

    THE OTHER THREE PRODUCTS' REFUSALS ARE READ, not assumed: two admitters leave the
    seam UNFUSED naming both, so a composition that fused would still be worth checking
    against the reasons the losers gave.
    """
    fields, layer, grid = build(xp, conductive=conductive)
    plan = arms.plan_step(fields, layer, grid,
                          licenses={LICENSE_COMPLEX_NO_PML: LICENCE,
                                    LICENSE_COMPLEX: LICENCE},
                          subnormal_policy=POLICY, sources=(), fuse=True)
    label = "complex no-absorber fused electric pair"
    # SLOT ARBITRATION, 2026-09-02: `update_E` is this pair's SECOND slot and
    # `cuda_complex_no_pml_fused_polarization_pair`'s FIRST, and all FOUR rows of this
    # product's cell (`TestLoadDump.*_3d`) are exactly the four that pair serves at
    # E->P -- measured on the stamped census (`cuda_complex_no_pml` covering
    # `update_P` on every one) and on the pre-existing board's served ledger, which
    # records both at 4. Installing here is +4 at D_to_E and -4 at E_to_P: NET ZERO,
    # paid for by displacing a released product. `fused_pairs._later_seam_claimant`
    # therefore leaves the slot where it was, and the two facts are pinned APART: the
    # PREDICATE still admits this run (its gate certifies the arithmetic), and the
    # COMPOSER gives the seam to the polarization pair.
    #
    # THIS FIXTURE CARRIES A SUSCEPTIBILITY, so the polarization pair admits it and
    # the arbitration fires: the seam goes to the product that was already serving it.
    # THE ASSERTIONS BELOW CHANGED DIRECTION ON 2026-09-02 AND THAT IS THE POINT --
    # before that date this pair took `update_E` and the polarization pair was refused
    # with an "arm table gave it elsewhere" reason, which is how the board came to
    # credit BOTH at 4 on the same four rows.
    #
    # AND CHANGED AGAIN ON 2026-09-04, when the COMPLEX THREE-SLOT WELD landed on
    # exactly these four rows: its span strictly contains this pair's, so
    # `fused_pairs._superseded_by_a_longer_span` refuses THIS pair by name and the
    # weld takes all three slots -- the E->P pair is failed closed out of update_E
    # by `_pair_may_absorb`, and there is no longer a trade to arbitrate. The two
    # facts stay pinned apart: the PREDICATE still admits this run, and the
    # COMPOSER gives the seam to the product that serves both sides of it.
    assert plan.selected.get("step_D") == plan.selected.get("update_E") == \
        plan.selected.get("update_P") == "complex no-absorber three-slot weld"
    assert "strictly contains" in plan.reasons[f"fused_pair_{FAMILY}"][0], (
        plan.reasons.get(f"fused_pair_{FAMILY}"))
    assert covers(fields, layer, grid)[0] is True
    replaced = fused_pairs.replaced_sub_steps(plan.plans)
    assert "zero_metal_D" not in replaced, (
        "neither product carries the wall clear and may not report that it did")
    _ = label
    for family in ("cuda_complex_fused_electric_pair",
                   "cuda_cylindrical_fused_electric_pair"):
        assert plan.reasons[f"fused_pair_{family}"] == (
            "curl half: no active PML layer",)
    assert plan.reasons["fused_pair_cuda_fused_electric_pair"] == (
        "curl half: complex64 storage: the recurrence is the same but the storage is "
        "not",)


def test_fusion_is_opt_in_on_this_seam_too(xp):
    """``fuse`` defaults to off, so the composition the coverage census measured is
    unchanged and this product adds nothing nobody asked for."""
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid,
                          licenses={LICENSE_COMPLEX_NO_PML: LICENCE,
                                    LICENSE_COMPLEX: LICENCE},
                          subnormal_policy=POLICY, sources=())
    assert plan.selected.get("step_D") == "complex no-PML curl"
    assert plan.selected.get("update_E") == "complex no-PML stored E"


def test_the_pair_may_not_absorb_a_slot_a_different_arm_won(xp, monkeypatch):
    """``_pair_may_absorb`` is the shared bound: a fused product may only take slots the
    ARM TABLE gave to exactly the two arms its predicate conjoins.

    THE SLOT NOW HAS A SECOND LEGITIMATE CLAIMANT and the assertion says so
    rather than assuming there is none. Since 2026-09-02 the complex
    no-absorber E->P weld holds ``update_E`` as its own CURL slot, and on a
    fixture with a pole bank it admits -- so with THIS product barred the slot
    goes to that weld, not back to the plain arm. What this test is about is
    unchanged and is asserted directly: the barred product refused BY NAME and
    did not take the slot. The fall-back to the plain arm is then measured with
    BOTH fused claimants barred, which is the shape the claim was always making.
    """
    fields, layer, grid = build(xp)
    # THE THIRD CLAIMANT, 2026-09-04: the complex three-slot weld supersedes this
    # pair by strict containment on this fixture, so it is barred here too -- this
    # test is about THIS pair's absorb clause, and a weld that took the slot for its
    # own reason would say nothing about it.
    monkeypatch.setitem(fused_pairs.FUSED_PAIR_ARMS,
                        "cuda_three_slot_complex_no_pml_dispersive_weld",
                        ("complex no-PML curl", "no-PML null", "complex no-PML ADE"))
    monkeypatch.setitem(fused_pairs.FUSED_PAIR_ARMS, product.FAMILY,
                        ("complex no-PML curl", "no-PML null"))
    plan = arms.plan_step(fields, layer, grid,
                          licenses={LICENSE_COMPLEX_NO_PML: LICENCE,
                                    LICENSE_COMPLEX: LICENCE},
                          subnormal_policy=POLICY, sources=(), fuse=True)
    assert plan.reasons[f"fused_pair_{product.FAMILY}"], plan.reasons
    assert plan.selected.get("update_E") != "complex no-absorber fused electric pair", \
        plan.selected
    assert plan.selected.get("update_E") in (
        "complex no-PML stored E", "complex no-PML fused polarization pair"), \
        plan.selected

    # WITH EVERY FUSED CLAIMANT BARRED the slot falls back to the plain arm --
    # the original statement, now made where nothing else can take it.
    monkeypatch.setitem(fused_pairs.FUSED_PAIR_ARMS,
                        "cuda_complex_no_pml_fused_polarization_pair",
                        ("complex no-PML curl", "no-PML null"))
    plan = arms.plan_step(fields, layer, grid,
                          licenses={LICENSE_COMPLEX_NO_PML: LICENCE,
                                    LICENSE_COMPLEX: LICENCE},
                          subnormal_policy=POLICY, sources=(), fuse=True)
    assert plan.selected.get("update_E") == "complex no-PML stored E"
    assert plan.reasons[f"fused_pair_{product.FAMILY}"], plan.reasons
    assert plan.reasons[
        "fused_pair_cuda_complex_no_pml_fused_polarization_pair"], plan.reasons
