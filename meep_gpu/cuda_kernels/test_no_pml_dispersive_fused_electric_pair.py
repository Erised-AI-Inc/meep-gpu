"""The no-absorber dispersive-store D/E weld: emitter, predicate and the census cell.

WHAT THIS SUITE PINS, and each section says which failure it is the tripwire for:

1. THE LIFT. The emitted kernel is the two CERTIFIED bodies with a named list of edits
   and nothing else. Every anchor is asserted, and a moved anchor must fail the EMIT
   rather than emit a kernel that compiles and is quietly not the certified
   arithmetic.
2. THE SEAM ORDER. The wall clear sits between the curl's register capture and BOTH
   its consumers -- the store and the constitutive read. That order is the whole
   product, and the probe measured it; here it is read off the emitted text so a
   reordering is caught without a device.
3. THE PREDICATE, on real engine objects, including the DISJOINTNESS that keeps this
   product from colliding with the other eleven ``step_D`` products -- an overlap
   would leave the seam UNFUSED and COST coverage rather than add it.
4. THE REPAIR DECLARATION. This is the first CUDA product to declare a non-default
   ``REPAIR_PATHS``, and the declaration is load-bearing: the split-field repair
   inverts a recurrence this cell does not run. The wrong declaration must be a
   REFUSAL, and that is measured here rather than asserted.
5. THE CENSUS CELL. The three corpus rows, their arms, and
   ``CARRIES_DEPOSIT_REPAIR`` as a per-cell MEASUREMENT off the census rather than a
   constant read back to itself.

NO DEVICE IS NEEDED. ``conductive_kernels`` and ``dispersive_kernels`` are importable
without CuPy on this host, so both certified halves' text is the real thing; the
predicate runs on NumPy wearing CuPy's name.
"""

from __future__ import annotations

import json
import pathlib

import numpy
import pytest

from .. import deposit_repair
from ..dispersion import PolarizationState, Susceptibility
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from ..sources import GaussianEnvelope, VolumeSource
from . import fused_pairs
from . import fused_electric_pair as real
from . import no_pml_dispersive_fused_electric_pair as family
from .test_fused_pairs import _NumpyWearingCupysName

#: The census this round's numbers are read off. Named so a re-cut that moved the rows
#: is a NAMED failure here rather than a silently different denominator.
CENSUS = (pathlib.Path(__file__).resolve().parents[2] / "parity" / "meep_gpu"
          / "results" / "cuda_predicate_coverage_2026-09-02_conductive_final")

#: The TWO board cells this ONE product occupies, and what the census says about them.
#: The denominator is THREE and every one of the three carries an electric deposit
#: inside the seam, which is why ``CARRIES_DEPOSIT_REPAIR`` is the product rather than
#: a clause of it.
CELLS = (
    {"seam": "D_to_E", "curl_arm": "cuda_conductive",
     "constitutive_arm": "cuda_no_pml_dispersive", "rows": 2},
    {"seam": "D_to_E", "curl_arm": "cuda_no_pml_curl",
     "constitutive_arm": "cuda_no_pml_dispersive", "rows": 1},
)

#: The three corpus rows, by name and by which curl arm each takes. Spelled out so a
#: census whose cell moved is a named failure listing WHICH row moved rather than a
#: changed integer.
CELL_ROWS = {
    "examples:absorber-1d.py": "cuda_conductive",
    "tests:TestAbsorber.test_absorber": "cuda_conductive",
    "examples:material-dispersion.py": "cuda_no_pml_curl",
}


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


@pytest.fixture
def source():
    """The emitter at the corpus's own conductive mask and pole arity.

    ``(True, True, True)`` with five poles is ``absorber-1d.py`` and
    ``TestAbsorber.test_absorber``; the lossless two-pole build is
    ``material-dispersion.py`` and is asked for separately where it matters.
    """
    return family.no_pml_dispersive_fused_electric_pair_source


# ---------------------------------------------------------------------------
# 1. THE LIFT
# ---------------------------------------------------------------------------

def test_the_curl_stencil_is_reused_unchanged(source):
    """The three curl stencil lines are the certified ones, character for character.

    THIS IS THE POINT OF THE FAMILY: the weld exists to reuse the certified curl, so
    its survival is asserted rather than claimed. A stencil that had been retyped here
    would be a second copy free to drift.
    """
    text = source((True, True, True), (5, 5, 5))
    assert text.count("        float curl = dtdx * ((sf - f1) + (f2 - ss));") == 3


def test_both_branches_of_the_certified_conditional_are_captured(source):
    """Every per-target tail writes a REGISTER, on both the conductive and the plain
    branch of ``conductive_kernels``' own ``#if``.

    CAPTURING ONE BRANCH ONLY is the failure this is the tripwire for: the other would
    store to global memory, the wall clear below could not reach it, and the
    constitutive half would read a pre-clear volume on exactly the runs whose mask
    took that branch.
    """
    text = source((True, False, True), (2, 2, 2))
    for index, axis in enumerate("xyz"):
        target = f"D{axis}"
        assert (f"        d_{axis} = cond_apply_reg({target}, cf{index}, ci{index}, "
                f"idx, curl);") in text
        assert f"        d_{axis} = {target}[idx] - curl;" in text


def test_no_curl_tail_stores_into_the_displacement(source):
    """No ``D*[idx] = `` survives inside the curl body.

    The only store into D is the one :func:`zero_metal_carry` emits AFTER the clear;
    a surviving curl store would race it and the constitutive half could read either.
    """
    text = source((True, True, True), (5, 5, 5))
    body = text.split("float d_z = 0.0f;", 1)[1].split("stepping._zero_metal", 1)[0]
    for axis in "xyz":
        assert f"D{axis}[idx] = " not in body


def test_the_lift_edits_are_declared_and_no_more(source):
    """:data:`LIFT_EDITS` is DATA a gate can assert, and it names four edits.

    A fifth edit landing without a row here is the failure this catches: the list is
    what a reader and a gate are told the weld does to the certified text.
    """
    assert len(family.LIFT_EDITS) == 4
    for edit in family.LIFT_EDITS:
        assert set(edit) == {"what", "from", "to", "why"}
        assert edit["why"]


# ---------------------------------------------------------------------------
# 2. THE SEAM ORDER -- the whole product
# ---------------------------------------------------------------------------

def test_the_clear_sits_between_the_capture_and_both_consumers(source):
    """``curl -> clear -> {store, constitutive read}``, in that order in the text.

    THE ARRAY PATH'S OWN ORDER: ``step_D`` writes D (:3306), ``zero_metal_D`` zeroes
    it at the wall cells (:3310), ``update_E`` reads it (:3313). The fused kernel
    holds one value and both consumers must see the POST-clear one. The probe measured
    that both orderings diverge (``subtract_before_clear`` and ``store_before_clear``
    each armed on 4 of 7 configurations); this reads it off the text.
    """
    text = source((True, True, True), (5, 5, 5))
    capture = text.index("d_z = cond_apply_reg(Dz,")
    clear = text.index("if (wall_z && k == 0)")
    store = text.index("    Dz[idx] = d_z;")
    read = text.index("float s_z = d_z;")
    assert capture < clear < store, "the wall clear must precede the store"
    assert clear < read, "the wall clear must precede the constitutive read"


def test_the_wall_clears_the_off_diagonal_pair_on_each_axis(source):
    """Two D components per walled axis -- the OFF-DIAGONAL complement.

    A weld that reused the B side's DIAGONAL rule would clear ONE wrong component and
    leave two right ones standing, on every walled run. Two of the three corpus rows
    are METALLIC on z, so this is corpus behaviour and not a fixture's.
    """
    text = source((True, True, True), (5, 5, 5))
    assert "if (wall_x && i == 0) { d_y = 0.0f; d_z = 0.0f; }" in text
    assert "if (wall_y && j == 0) { d_x = 0.0f; d_z = 0.0f; }" in text
    assert "if (wall_z && k == 0) { d_x = 0.0f; d_y = 0.0f; }" in text


def test_the_displacement_is_bound_exactly_once(source):
    """D appears once in the signature, and the constitutive half reads no D volume.

    THE ALIASING HAZARD: binding it a second time as ``const __restrict__`` would be
    two restrict pointers to one allocation, which NVRTC miscompiles WITHOUT A
    DIAGNOSTIC -- a silent wrong answer, which is why this is a test and not a note.
    """
    text = source((True, True, True), (5, 5, 5))
    signature = text.split('extern "C" __global__ void', 1)[1].split("\n) {\n", 1)[0]
    for axis in "xyz":
        assert signature.count(f"D{axis}") == 1
    constitutive = text.split("Dz[idx] = d_z;", 1)[1]
    for axis in "xyz":
        assert f"D{axis}[idx]" not in constitutive


def test_the_pole_pointers_are_restrict_and_the_others_are_not(source):
    """Lifted decisions, not decided here.

    Two susceptibilities never share a ``P`` buffer, so those are ``__restrict__``; an
    isotropic run hands ONE inverse-epsilon pointer three times and a shared
    conductivity profile hands one condfac three times, so those are not.
    """
    text = source((True, True, True), (2, 2, 2))
    signature = text.split('extern "C" __global__ void', 1)[1].split("\n) {\n", 1)[0]
    assert "const float* __restrict__ P_Ex_0" in signature
    for name in ("inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez",
                 "cf0", "cf1", "cf2", "ci0", "ci1", "ci2"):
        assert f"const float* {name}" in signature
        assert f"const float* __restrict__ {name}" not in signature


def test_a_zero_arity_component_forms_no_chain(source):
    """At arity 0 the emitted line is the certified NON-dispersive one.

    ``displacement_minus_polarization`` returns the D array ITSELF when nothing drives
    a component (fields.py:1096-1098), so forming an empty chain would be arithmetic
    the array path does not do.
    """
    text = source((True, True, True), (2, 0, 3))
    assert "float src_y = d_y * inv_eps_Ey[idx];" in text
    assert "float s_y = " not in text


# ---------------------------------------------------------------------------
# 3. THE EMITTER'S REFUSALS
# ---------------------------------------------------------------------------

def test_a_moved_certified_anchor_fails_the_emit(monkeypatch):
    """A certified tail that moved must fail the EMIT, by name.

    Splicing around a missing anchor would produce a kernel that compiles and is
    quietly not the certified arithmetic -- the exact failure this campaign's lift
    discipline exists to prevent.
    """
    from . import conductive_kernels

    original = conductive_kernels.kernel_template("step_D_no_pml_conductive")
    broken = original.replace("        Dx[idx] = Dx[idx] - curl;\n",
                              "        Dx[idx] = Dx[idx] - curl; // moved\n", 1)
    monkeypatch.setattr(conductive_kernels, "_TEMPLATES",
                        dict(conductive_kernels._TEMPLATES,
                             step_D_no_pml_conductive=broken))
    with pytest.raises(AssertionError, match="appears 0 times"):
        family.no_pml_dispersive_fused_electric_pair_source((False, False, False),
                                                            (2, 2, 2))


def test_a_short_conductive_mask_is_refused_by_name():
    """A two-flag mask would leave a ``#define`` unset and the kernel would take the
    wrong branch on the third target with no diagnostic at all."""
    with pytest.raises(ValueError, match="triple of per-target conductivity flags"):
        family.no_pml_dispersive_fused_electric_pair_source((True, True), (2, 2, 2))


def test_every_swept_source_emits_and_is_pure_ascii():
    """All 24 bodies (4 masks x 6 arities) emit, and each is distinct.

    A mask or arity that collapsed onto another's text would mean one of the two is
    never actually exercised, and a digest over the set would not notice.
    """
    sources = family.device_sources()
    assert len(sources) == len(family.COND_MASKS_SWEPT) * len(family.swept_arities())
    assert len(set(sources.values())) == len(sources)
    for text in sources.values():
        text.encode("ascii")


# ---------------------------------------------------------------------------
# 4. THE PREDICATE, ON REAL ENGINE OBJECTS
# ---------------------------------------------------------------------------

def electric_source(grid, component="Ez"):
    return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


def build(xp, *, poles=5, conductivity=0.4, boundaries=("periodic", "periodic",
                                                        "metallic"),
          symmetry=(), pml=False):
    """One engine on the PLAIN branch, with LIVE poles.

    ``PML(thickness=0)`` absorbs on no face, so ``stepping._pml_is_active`` is False
    and ``update_E`` takes the pure overwrite at :993 -- the recurrence this family
    exists for. ``pml=True`` builds an ACTIVE layer instead, which every clause below
    expects to be refused.
    """
    rng = numpy.random.default_rng(20260902)
    grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                boundaries=tuple(boundaries),
                symmetry=tuple(Mirror(axis, int(phase)) for axis, phase in symmetry),
                xp=xp, courant=0.5)
    fields = Fields(grid=grid)
    shape = tuple(int(n) for n in grid.shape)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=shape).astype(numpy.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray(
            (numpy.float32(1.0) / values).astype(numpy.float32))
    fields.set_epsilon_volumes(epsilon, inverse)
    if conductivity is not None:
        fields.set_d_conductivity(xp.asarray(
            numpy.full(shape, float(conductivity), dtype=numpy.float32)))
    kinds = ("lorentzian", "drude")
    for index in range(poles):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: 0.35 + 0.04 * index for name in ("Ex", "Ey", "Ez")}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))
    if pml:
        fields.enable_pml_storage()
        layer = PML(grid=grid, thickness=2)
    else:
        fields.enable_field_storage()
        layer = PML(grid=grid, thickness=0)
    # SEEDED NON-ZERO: zero is a fixed point of ``s = s - P``.
    for state in fields.polarizations:
        for store in (state.P, state.P_prev):
            for _component, array in store.items():
                array[...] = xp.asarray(
                    rng.uniform(-0.6, 0.6, size=shape).astype(numpy.float32))
    return fields, grid, layer



def test_the_installer_gives_the_span_to_the_three_slot_weld(xp):
    """SLOT ARBITRATION, and it is a fact about the COMPOSER not about this weld.

    The predicate ADMITS every row of both cells (the tests below measure that) and
    the gate certifies the arithmetic on 14 of 14 cases under both float32 policies.
    This product has never been about to install here, and the reason changed shape
    once:

    * FIRST READING: it owns ``update_E`` as its SECOND slot,
      ``cuda_no_pml_fused_polarization_pair`` owns it as its FIRST, and all THREE
      rows of these two cells are exactly the three that pair serves at E->P.
      Installing here was +3 at D_to_E and -3 at E_to_P -- NET ZERO, paid for by
      displacing a released product -- so ``fused_pairs._later_seam_claimant`` left
      the slot where it was.
    * SECOND READING, and strictly better: ``cuda_three_slot_no_pml_dispersive_weld``
      takes ``step_D``, ``update_E`` AND ``update_P``, so BOTH seams are served and
      there is no trade to arbitrate. This product is refused by
      ``_superseded_by_a_longer_span``, whose reason names the containing span.

    WHAT IS PINNED IS THE INVARIANT BOTH READINGS SHARE, and it is stronger now: this
    product never silently displaces the later seam's product, the refusal is BY
    NAME, and its own predicate still admits the run. Both cells are walked, because
    the two take DIFFERENT curl arms and the weld declares the second in
    ``FUSED_PAIR_EXTRA_ARMS`` exactly as this module's own row does.
    """
    from . import arms  # noqa: PLC0415

    for sigma, poles, boundaries in ((0.4, 5, ("periodic", "periodic", "metallic")),
                                     (None, 2, ("periodic",) * 3)):
        fields, grid, layer = build(xp, poles=poles, conductivity=sigma,
                                    boundaries=boundaries)
        sources = (electric_source(grid),)
        plan = arms.plan_step(fields, layer, grid, fuse=True, sources=sources)
        assert plan.selected["step_D"] == plan.selected["update_E"] == \
            plan.selected["update_P"] == \
            "no-absorber three-slot dispersive weld", (sigma, poles)
        refusal = plan.reasons[
            "fused_pair_cuda_no_pml_dispersive_fused_electric_pair"][0]
        assert "strictly contains" in refusal and "slot arbitration" in refusal
        covered, _why = family.covers_no_pml_dispersive_fused_electric_pair(
            fields, layer, grid, sources)
        assert covered, ("the PREDICATE must still admit it -- the refusal is the "
                         "installer's arbitration, not a verdict about the weld")


def test_the_conductive_corpus_shape_is_admitted(xp):
    """``absorber-1d.py`` / ``TestAbsorber.test_absorber``: an inert layer, a
    conductivity on every D component, five poles, METALLIC on z, a D deposit."""
    fields, grid, layer = build(xp)
    covered, why = family.covers_no_pml_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert covered, why


def test_the_lossless_corpus_shape_is_admitted(xp):
    """``material-dispersion.py``: the SECOND cell, all periodic, two poles, LOSSLESS.

    The curl predicate that admits it is the other arm of the disjunction, which is
    why this is a separate test and not a parameter.
    """
    fields, grid, layer = build(xp, poles=2, conductivity=None,
                                boundaries=("periodic",) * 3)
    covered, why = family.covers_no_pml_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert covered, why


def test_the_two_curl_arms_cannot_both_admit(xp):
    """The disjunction is a LOOKUP, not a widening: the two curl predicates partition.

    ``covers_conductive_curl`` requires a target carrying a sigma and
    ``covers_no_pml_curl`` refuses one, so exactly one answers on every run this
    family admits. A widening here would be a predicate claiming runs no emitted body
    serves.
    """
    conductive, lossless = family._curl_predicates()
    for sigma in (0.4, None):
        fields, grid, layer = build(xp, poles=2, conductivity=sigma,
                                    boundaries=("periodic",) * 3)
        first = conductive(fields, layer, grid, "step_D")[0]
        second = lossless(fields, layer, grid, "step_D")[0]
        assert first is (sigma is not None)
        assert second is (sigma is None)
        assert not (first and second)


def test_an_active_absorber_is_refused(xp):
    """The layer is what separates this family from every other electric product."""
    fields, grid, layer = build(xp, poles=2, pml=True)
    covered, why = family.covers_no_pml_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert not covered
    assert "an active PML layer is installed" in why


def test_a_mirror_plane_is_refused_by_name(xp):
    """The two symmetry fills run INSIDE this seam and this weld carries neither.

    Restated in this predicate rather than inherited: if either half's tranche ever
    admits a fold, the weld would silently swallow two passes that had started doing
    work.
    """
    fields, grid, layer = build(xp, poles=2, boundaries=("periodic",) * 3,
                                symmetry=(("Y", 1),))
    covered, why = family.covers_no_pml_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert not covered
    assert "mirror plane" in why or "folded" in why


def test_an_undeclared_source_set_is_refused(xp):
    """IGNORANCE IS A REFUSAL. ``Fields`` does not hold the source list, so ``None``
    cannot be read as an empty seam -- it is the caller failing to say."""
    fields, grid, layer = build(xp)
    covered, why = family.covers_no_pml_dispersive_fused_electric_pair(
        fields, layer, grid, None)
    assert not covered


def test_it_partitions_against_every_other_step_D_product(xp):
    """No other ``step_D`` product may admit a run this one does.

    ``install_fused_pairs`` leaves a seam UNFUSED when two products admit it, so an
    overlap would COST coverage rather than add it. This asks the REAL twin and the
    dispersive one, the two nearest neighbours, on both corpus shapes.
    """
    from . import dispersive_fused_electric_pair as dispersive

    for sigma, poles in ((0.4, 5), (None, 2)):
        fields, grid, layer = build(xp, poles=poles, conductivity=sigma,
                                    boundaries=("periodic",) * 3)
        sources = (electric_source(grid),)
        assert family.covers_no_pml_dispersive_fused_electric_pair(
            fields, layer, grid, sources)[0]
        assert not real.covers_fused_electric_pair(fields, layer, grid, sources)[0]
        assert not dispersive.covers_dispersive_fused_electric_pair(
            fields, layer, grid, sources)[0]


# ---------------------------------------------------------------------------
# 5. THE REPAIR DECLARATION -- the first non-default one on this track
# ---------------------------------------------------------------------------

def test_the_declared_path_is_the_one_the_layer_selects(xp):
    """``deposit_repair.repair_path_for`` and this module's declaration agree.

    They are asked from opposite ends: the module DECLARES which recurrence it
    inverts, and ``repair_path_for`` reads the same ``stepping._pml_is_active`` that
    ``update_E`` branches on. A disagreement is a mis-repair waiting to happen.
    """
    assert family.REPAIR_PATHS == (deposit_repair.PLAIN_PATH,)
    _fields, _grid, layer = build(xp)
    assert deposit_repair.repair_path_for(layer) == deposit_repair.PLAIN_PATH


def test_declaring_the_split_field_repair_is_refused_not_mis_applied(xp):
    """THE MEASUREMENT BEHIND THE DECLARATION, and the reason it is load-bearing.

    The split-field repair inverts ``field + kps*fresh - kms*fw_prev``. This cell runs
    a PURE OVERWRITE with no ``f_w`` at all, so applying that inverse would write an
    answer the step never computed. ``deposit_repair.repairable`` must REFUSE it by
    name -- and does, which is what makes a wrong declaration safe.
    """
    fields, grid, layer = build(xp)
    covered, why = deposit_repair.repairable(
        fields, "D", pml=layer, paths=(deposit_repair.SPLIT_FIELD_PATH,))
    assert not covered
    # ``why`` IS A TUPLE of reasons, not a string: the refusal names the layer AND
    # the missing f_w, and both are the evidence that the split-field inverse has
    # nothing here to invert.
    joined = " | ".join(why)
    assert "absorbs on none of its six faces" in joined
    assert "f_w_Ex is not allocated" in joined
    covered, _why = deposit_repair.repairable(
        fields, "D", pml=layer, paths=family.REPAIR_PATHS)
    assert covered
    _ = grid


def test_the_installer_reads_the_declaration_off_this_module():
    """``install_fused_pairs`` hands the module's OWN paths to the bracket.

    Threaded on 2026-09-02. A default here would refuse every row this product exists
    for -- the failure would look like "the product serves nothing", not like a
    wiring bug, which is why it is pinned.
    """
    product = fused_pairs.FUSED_PRODUCTS[
        "cuda_no_pml_dispersive_fused_electric_pair"]
    assert fused_pairs._repair_paths_of(product) == (deposit_repair.PLAIN_PATH,)
    # AND EVERY PRODUCT WRITTEN BEFORE THE PARAMETER KEEPS ITS EXACT ANSWER.
    assert fused_pairs._repair_paths_of(
        fused_pairs.FUSED_PRODUCTS["cuda_fused_electric_pair"]) == (
            deposit_repair.SPLIT_FIELD_PATH,)


def test_the_flag_reaches_the_shared_clause_with_the_declared_paths(xp):
    """``CARRIES_DEPOSIT_REPAIR`` is passed to the clause, not just declared.

    A product that declared the flag without routing it would refuse every in-seam
    source while claiming to carry them.
    """
    assert family.CARRIES_DEPOSIT_REPAIR is True
    fields, grid, layer = build(xp)
    words = {"undeclared": "not declared",
             "refusal": lambda index, source: f"source {index} is in the seam"}
    assert not deposit_repair.seam_source_reasons(
        fields, (electric_source(grid),), "D", pml=layer,
        carries_repair=True, repair_paths=family.REPAIR_PATHS, **words)
    assert deposit_repair.seam_source_reasons(
        fields, (electric_source(grid),), "D", pml=layer,
        carries_repair=False, repair_paths=family.REPAIR_PATHS, **words)


# ---------------------------------------------------------------------------
# 6. THE CENSUS CELL -- measured, not transcribed
# ---------------------------------------------------------------------------

def _census_rows():
    """Every row of the named census, by ``row`` key."""
    rows = {}
    for leg in ("examples.jsonl", "tests.jsonl"):
        path = CENSUS / leg
        # A MISSING CENSUS IS A FAILURE, NOT A SKIP. This suite's cell facts -- which
        # rows the product serves, and that every one of them carries a deposit -- are
        # MEASUREMENTS off that file. Skipping would turn the round's central claim
        # into a silent coverage gap, which is exactly what the no-silent-skip policy
        # is for.
        assert path.exists(), (
            f"the census leg {path} has not been cut; this suite reads the cell's "
            f"membership and its deposit denominator off it and cannot assert either "
            f"without it")
        # THE LEG PREFIX IS THIS SUITE'S, NOT THE CENSUS'S. The census stores a bare
        # `row` ("absorber-1d.py", "TestAbsorber.test_absorber"); the BOARD keys its
        # instances by "<leg>:<row>", which is the name a reader of
        # taxonomy.instances sees and the name CELL_ROWS spells. Prefixing here means
        # the two vocabularies meet in one place instead of a test quietly comparing
        # a bare name against a prefixed one and reporting a moved cell.
        leg_name = leg.split(".", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows[f"{leg_name}:{record.get('row')}"] = record
    return rows


def test_the_census_names_the_three_rows_and_this_product_admits_them_all():
    """The cell's membership is READ, not spelled: each row's curl arm is the one
    ``CELL_ROWS`` names and its constitutive arm is ``cuda_no_pml_dispersive``."""
    rows = _census_rows()
    missing = [name for name in CELL_ROWS if name not in rows]
    assert not missing, f"the census does not carry {missing}"
    for name, expected_arm in CELL_ROWS.items():
        record = rows[name]
        constitutive = record["cuda_no_pml_dispersive"]["update_E"]
        assert constitutive["covered_modulo_backend"], (
            f"{name}: the no-PML dispersive constitutive arm does not admit it")
        arm = record[expected_arm]["step_D"]
        assert arm["covered_modulo_backend"], (
            f"{name}: {expected_arm} does not admit step_D")


def test_every_row_of_the_cell_carries_an_electric_deposit():
    """CARRIES_DEPOSIT_REPAIR IS THE PRODUCT, measured off the census.

    At False this module would compile, pass every arithmetic leg and serve ZERO of
    the three rows -- so the flag is asserted against what the corpus actually
    declares rather than read back from the module that sets it.
    """
    rows = _census_rows()
    carrying = [name for name in CELL_ROWS
                if "D" in (rows[name]["configuration"].get("source_field_types") or ())]
    assert len(carrying) == len(CELL_ROWS), (
        f"only {carrying} declare a D-seam deposit; if that is right, "
        f"CARRIES_DEPOSIT_REPAIR is no longer worth the whole cell and the module's "
        f"docstring says it is")
    assert family.CARRIES_DEPOSIT_REPAIR is True


def test_no_row_of_the_cell_is_folded_or_absorbing():
    """The two clauses this weld refuses by name are refused by the CORPUS too.

    A row that acquired a mirror plane or an active layer would stop being served,
    and this is where that shows up as a named failure rather than as a quiet drop in
    the board's served count.
    """
    rows = _census_rows()
    for name in CELL_ROWS:
        configuration = rows[name]["configuration"]
        assert not configuration["has_symmetry"], name
        assert not configuration["pml_active"], name
        assert configuration["stores_E"], name
        assert configuration["n_polarizations"] > 0, name


def test_the_product_column_admits_exactly_the_cell_rows():
    """The battery asks the PREDICATE, and the predicate admits exactly these three.

    THE DISJOINTNESS MEASURED OVER THE WHOLE CORPUS rather than argued on fixtures: no
    other row in either leg is admitted, so this weld's arithmetic claims its two
    cells and nothing else.

    WHAT THIS COLUMN DOES NOT SAY IS THAT THE COMPOSER INSTALLS IT. It does not, on
    any of the three: each is a row ``cuda_no_pml_fused_polarization_pair`` already
    serves at the E->P seam, and ``fused_pairs._later_seam_claimant`` leaves
    ``update_E`` with that product because the trade is one seam-instance for one.
    The two facts live apart on purpose -- the predicate is about the weld, the
    installer is about the slot -- and
    ``test_the_installer_leaves_update_E_with_the_polarization_pair`` pins the second.
    """
    rows = _census_rows()
    column = "cuda_no_pml_dispersive_fused_electric_pair"
    assert column in next(iter(rows.values())), (
        "this census predates the product column; the battery edit and the census "
        "cut belong to the same round")
    admitted = {name for name, record in rows.items()
                if (record.get(column) or {}).get("covered_modulo_backend")}
    assert admitted == set(CELL_ROWS), (
        f"admitted {sorted(admitted)}, expected {sorted(CELL_ROWS)}")
