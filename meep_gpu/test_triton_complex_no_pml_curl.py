"""Merge-bar tests for :mod:`meep_gpu.triton_kernels.complex_no_pml_curl`.

WHAT A LAPTOP CAN AND CANNOT SETTLE FOR THIS FAMILY. The family ships no kernel:
it re-uses ``complex_fields.bloch_pml_curl_step`` under a no-absorber binding (the
absorber-free chunk record and unit columns). So the byte-identity claim belongs to
a device — ``parity/meep_gpu/gate_triton_complex_no_pml_curl.py``. What is
settleable here, and is settled here, is everything the 2026-08-16 group-(J)
artifact said the closing round still owed:

* the PREDICATE — that it admits group (J)'s four slots and refuses everything
  adjacent, on both no-absorber shapes the engine can present (``pml=None`` and
  an inert ``PML(thickness=0)``);
* DISJOINTNESS — that exactly one curl family admits each slot. Two admitters
  leave a slot UNSELECTED and it falls silently back to the array path, which is
  a coverage LOSS and not an error, so it is measured across every curl predicate
  in the package rather than argued;
* the BUILDER — that it returns None where it refuses and RAISES NOWHERE. This is
  the regression test for the measured failure mode: widening the predicate
  without a builder raises ``ValueError: expected a complex64 volume, got dtype
  None`` behind an inert layer and ``AttributeError: 'NoneType' object has no
  attribute 'kms_x_h'`` behind no layer at all;
* the BINDING — that the plan binds the absorber-free record (no membership run,
  the grid's presence word), so the kernel's dsig-only form steps every cell, and
  that this form on unit coefficients IS ``stepping._apply_curl``'s
  ``target -= curl`` on physical bands with a vacuity floor and on all 36 special
  pairs; the previous binding (the recurrence everywhere with a zeroed auxiliary)
  is pinned as the mutation it now is, losing exactly ``(-0, +0)``;
* that ``update_E`` on group (J)'s shape STILL REFUSES.

Every assertion here fails on the pre-change tree — the module does not exist
there and ``_complex_grid_reasons`` takes no ``require_active_pml``.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import pathlib

import numpy
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import complex_fields as complex_module
from meep_gpu.triton_kernels.launch import SUB_STEPS

MODULE_NAME = "meep_gpu.triton_kernels.complex_no_pml_curl"

SEED = 20260816
CURLS = ("step_B", "step_D")

#: Group (J)'s two rows, transcribed from
#: ``parity/meep_gpu/results/group_j_2026-08-16/PROVENANCE.md``.
J3 = {"cell_size": (2.5, 2.5, 2.5), "dimensions": 3,
      "k_point": (0.23, -0.17, 0.35)}
J2 = {"cell_size": (2.5, 2.5, 0.0), "dimensions": 2,
      "k_point": (0.3892, 0.1597, 0.0)}
J_ROWS = pytest.mark.parametrize("row", [J3, J2], ids=["J3_matgrid_3d",
                                                       "J2_subpixel_smoothing"])


@pytest.fixture(scope="module")
def family():
    """The shipped module itself — never a test-local stub."""
    return importlib.import_module(MODULE_NAME)


class _CupyNamed:
    """A NumPy that answers to the name 'cupy'.

    Clause 1 asks ``grid.xp.__name__`` and nothing else, and every question this
    file settles is host-side Python. Shimming the NAME keeps those questions
    reachable on a laptop without pretending a device is present: no test here
    launches a kernel, and the byte claim is explicitly left to the gate.
    """

    __name__ = "cupy"

    def __getattr__(self, name):
        return getattr(numpy, name)


def _build(*, cell_size=(2.5, 2.5, 2.5), dimensions=3, boundaries="periodic",
           k_point=(0.23, -0.17, 0.35), complex_storage=True, offdiag=True,
           pml_thickness=0, pml_storage=False, cupy_name=True, eps=2.25,
           **grid_kwargs):
    """A real Grid/Fields/PML triple on NumPy, defaulting to group (J)'s 3-D row.

    Every refusal test below perturbs exactly one clause off this configuration,
    so a refusal that appears is attributable to the clause the test moved.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, k_point=k_point, xp=numpy, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    shape = tuple(grid.shape)
    if offdiag:
        # A full inverse-permittivity ROW, as subpixel smoothing of a curved
        # interface installs one. This is what forces stores_E on group (J).
        rng = numpy.random.default_rng(SEED)
        diagonal = {c: numpy.full(shape, eps, dtype=numpy.float32)
                    for c in ("Ex", "Ey", "Ez")}
        inverse = {c: numpy.full(shape, numpy.float32(1.0 / eps),
                                 dtype=numpy.float32) for c in ("Ex", "Ey", "Ez")}
        rows = {"Ex": {"Ey": rng.uniform(-0.08, 0.08, shape).astype(numpy.float32)},
                "Ey": {"Ez": rng.uniform(-0.08, 0.08, shape).astype(numpy.float32)},
                "Ez": {"Ex": rng.uniform(-0.08, 0.08, shape).astype(numpy.float32)}}
        fields.set_epsilon_volumes(diagonal, inverse, rows)
    else:
        fields.set_background_eps(eps)
    if pml_storage or pml_thickness:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    if cupy_name:
        grid.xp = _CupyNamed()
    return fields, PML(grid=grid, thickness=pml_thickness)


def _complex_from_halves(real, imaginary):
    """``real + 1j*imaginary`` WITHOUT losing a signed zero in either half.

    ``values + 1j * imaginary`` promotes every ``-0.0`` to ``+0.0`` — the audited
    defect-4 spelling. Assigning the halves preserves both, which is the whole
    point of a fixture whose needle IS the signed zero.
    """
    out = numpy.empty(numpy.shape(real), dtype=numpy.complex64)
    out.real = real
    out.imag = imaginary
    return out


@pytest.fixture(scope="module", autouse=True)
def _policy():
    """Install the 'keep' policy clause 13 requires, exactly as the device leg did."""
    from meep_gpu.subnormal_policy import install_subnormal_policy

    install_subnormal_policy("keep")


def _reasons(family, fields, pml, sub_step):
    return list(family.complex_no_pml_curl_coverage(
        fields, pml, sub_step).reasons)


def _words(array):
    return numpy.ascontiguousarray(array).view(numpy.float32).view(numpy.uint32)


# ---------------------------------------------------------------------------
# The four slots, on both no-absorber shapes
# ---------------------------------------------------------------------------

@J_ROWS
@pytest.mark.parametrize("sub_step", CURLS)
@pytest.mark.parametrize("drop_layer", [False, True], ids=["inert_layer", "no_layer"])
def test_group_j_slots_are_covered_with_zero_reasons(family, row, sub_step,
                                                     drop_layer):
    """The four slots the device measured IDENTICAL, on both shapes.

    ``pml=None`` and ``PML(thickness=0)`` are BOTH tested because the engine can
    present either and a builder that survives one may raise on the other — which
    is exactly what the pre-change measurement found.
    """
    fields, pml = _build(**row)
    coverage = family.complex_no_pml_curl_coverage(
        fields, None if drop_layer else pml, sub_step)
    assert coverage.covered, coverage.reasons
    assert coverage.reasons == ()


@J_ROWS
@pytest.mark.parametrize("sub_step", CURLS)
def test_the_builder_returns_a_plan_and_raises_on_neither_shape(family, row,
                                                                sub_step):
    """The regression test for the measured raise. None is the only refusal."""
    fields, pml = _build(**row)
    for layer in (pml, None):
        plan = family.plan_complex_no_pml_curl(fields, layer, sub_step)
        assert plan is not None
        assert plan.sub_step == sub_step
        assert plan.shape == tuple(fields.grid.shape)


@J_ROWS
@pytest.mark.parametrize("sub_step", CURLS)
def test_the_shipped_split_field_builder_still_refuses_these_slots(row, sub_step):
    """DISJOINTNESS, direction one: the incumbent must not also admit.

    The refusal is asserted BY ITS CLAUSE, not merely as a False. Measured: the
    incumbent refuses these slots on TWO independent clauses — the absorber
    clause and ``fu_* is not allocated`` — so a bare ``not covered`` still holds
    with clause 3 deleted, and would report the partition intact while it was
    gone. The clause named here is the one the partition rests on.
    """
    fields, pml = _build(**row)
    for layer in (pml, None):
        verdict = complex_module.complex_pml_curl_coverage(
            fields, layer, sub_step)
        assert not verdict.covered
        assert any("no active PML layer" in r for r in verdict.reasons), \
            verdict.reasons


@pytest.mark.parametrize("sub_step", CURLS)
def test_an_active_layer_is_refused_here_and_admitted_by_the_incumbent(family,
                                                                       sub_step):
    """DISJOINTNESS, direction two — and the partition has no gap between them."""
    fields, pml = _build(cell_size=(2.5, 2.5, 2.5), pml_thickness=2)
    assert pml.is_active
    reasons = _reasons(family, fields, pml, sub_step)
    assert any("an active PML layer is installed" in r for r in reasons), reasons
    assert complex_module.complex_pml_curl_coverage(
        fields, pml, sub_step).covered


@pytest.mark.parametrize("sub_step", CURLS)
@pytest.mark.parametrize("thickness", [0, 2], ids=["inert", "active"])
def test_exactly_one_of_the_two_complex_curl_families_admits(family, sub_step,
                                                             thickness):
    """The partition, measured as a partition: never both, never neither."""
    fields, pml = _build(pml_thickness=thickness)
    inverted = family.complex_no_pml_curl_coverage(fields, pml, sub_step)
    incumbent = complex_module.complex_pml_curl_coverage(fields, pml, sub_step)
    assert sum((inverted.covered, incumbent.covered)) == 1, \
        (inverted.reasons, incumbent.reasons)
    # ... and the ABSORBER clause is what separates them, in whichever direction.
    refused = inverted if not inverted.covered else incumbent
    assert any("PML layer" in r for r in refused.reasons), refused.reasons


def test_no_other_curl_family_in_the_package_admits_group_j(family):
    """Disjointness against EVERY curl predicate, not only the obvious twin.

    An UNSELECTED slot is a silent coverage loss, so the separating clause is
    measured here for each family rather than reasoned about in a comment.
    """
    from meep_gpu.triton_kernels import bfast_curl, coverage as base
    from meep_gpu.triton_kernels import no_pml, special_kz, symmetry

    fields, pml = _build()
    others = [
        ("coverage.pml_curl", lambda s: base.pml_curl_coverage(fields, pml, s)),
        ("no_pml.plain_curl", lambda s: no_pml.plain_curl_coverage(fields, pml, s)),
    ]
    for module, name in ((bfast_curl, "bfast"), (special_kz, "special_kz"),
                         (symmetry, "symmetry")):
        for attribute in dir(module):
            if attribute.endswith("_curl_coverage"):
                others.append((f"{name}.{attribute}",
                               lambda s, m=module, a=attribute:
                               getattr(m, a)(fields, pml, s)))
    assert len(others) >= 3, [n for n, _ in others]
    for sub_step in CURLS:
        assert family.complex_no_pml_curl_coverage(
            fields, pml, sub_step).covered
        for name, call in others:
            try:
                verdict = call(sub_step)
            except TypeError:
                continue  # a different signature is a different product
            assert not verdict.covered, f"{name} also admits {sub_step}"


# ---------------------------------------------------------------------------
# update_E must STAY refused
# ---------------------------------------------------------------------------

def test_update_e_on_group_j_stays_refused_by_every_constitutive_family():
    """The device measured 93750 / 281250 differing. Nothing may take this slot."""
    from meep_gpu.triton_kernels import no_pml_constitutive

    fields, pml = _build()
    complex_side = complex_module.complex_constitutive_coverage(
        fields, pml, "E")
    assert not complex_side.covered
    assert any("off-diagonal chi1inv row" in r for r in complex_side.reasons)
    assert not no_pml_constitutive.null_constitutive_coverage(
        fields, pml, "E").covered


def test_this_family_ships_no_constitutive_predicate_at_all(family):
    """The narrowest possible statement of the same guarantee."""
    assert [name for name in dir(family) if "constitutive" in name] == []
    assert set(family.__all__) == {
        "ComplexNoPmlCurlPlan", "complex_no_pml_curl_coverage",
        "plan_complex_no_pml_curl", "plan_complex_no_pml_curl_from_arrays",
        "source_arrays", "unit_coefficient_columns"}


# ---------------------------------------------------------------------------
# The clauses this family did NOT invert must still fire
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", CURLS)
def test_real_storage_is_still_refused(family, sub_step):
    """Clause 2 is shared, NOT inverted: a real run belongs to no_pml.plain_curl."""
    fields, pml = _build(complex_storage=False, k_point=(0.0, 0.0, 0.0))
    reasons = _reasons(family, fields, pml, sub_step)
    assert any("storage is real float32" in r for r in reasons), reasons


@pytest.mark.parametrize("sub_step", CURLS)
def test_a_mirror_plane_is_still_refused(family, sub_step):
    """Clause 4 is shared: a folded run belongs to folded_complex."""
    # Grid refuses a mirrored axis carrying a Bloch k, so clause 2 is satisfied
    # the other way here: force_complex_fields rather than a nonzero k_point.
    fields, pml = _build(symmetry=("x",), k_point=(0.0, 0.0, 0.0))
    reasons = _reasons(family, fields, pml, sub_step)
    assert any("mirror plane" in r or "folded by a mirror" in r
               for r in reasons), reasons


@pytest.mark.parametrize("sub_step", CURLS)
def test_a_conductivity_is_still_refused(family, sub_step):
    """Clause 6 is shared: conductive complex stepping is a separate product."""
    fields, pml = _build()
    shape = tuple(fields.grid.shape)
    fields.set_d_conductivity(numpy.full(shape, numpy.float32(0.3),
                                         dtype=numpy.float32))
    reasons = _reasons(family, fields, pml, sub_step)
    assert any("conductivity is installed" in r for r in reasons), reasons


@pytest.mark.parametrize("sub_step", CURLS)
def test_unstored_e_is_refused_and_is_not_this_products_derive_path(family,
                                                                   sub_step):
    """Clause 9 does REAL work here: unstored E is no_pml.py's DERIVE product."""
    fields, pml = _build(offdiag=False)
    fields._stored_E = False
    reasons = _reasons(family, fields, pml, sub_step)
    assert any("recomputed from D rather than stored" in r for r in reasons), reasons


@pytest.mark.parametrize("sub_step", CURLS)
def test_pml_storage_behind_an_inert_layer_is_refused(family, sub_step):
    """Clause 3b: get_H would serve a stored H that update_H never writes."""
    fields, pml = _build(pml_storage=True)
    assert not pml.is_active
    reasons = _reasons(family, fields, pml, sub_step)
    assert any("PML storage enabled while the layer is inert" in r
               for r in reasons), reasons


def test_an_unknown_sub_step_raises_rather_than_quietly_refusing(family):
    with pytest.raises(ValueError):
        family.complex_no_pml_curl_coverage(*_build(), "update_E")
    with pytest.raises(ValueError):
        family.plan_complex_no_pml_curl(*_build(), "update_E")


def test_a_fields_without_a_grid_is_refused(family):
    class _NoGrid:
        grid = None

    verdict = family.complex_no_pml_curl_coverage(_NoGrid(), None, "step_B")
    assert not verdict.covered
    assert verdict.reasons == ("fields carries no grid",)


# ---------------------------------------------------------------------------
# The binding
# ---------------------------------------------------------------------------

@J_ROWS
def test_the_synthesized_unit_columns_are_bit_identical_to_an_inert_layers(family,
                                                                          row):
    """The synthesized binding must BE the one the device measured, not a second one.

    The group-(J) leg bound ``PML(thickness=0)``'s own tables. On a genuinely
    layer-less run there are none to bind, so this family synthesizes them — and
    that is only the same measurement if the bytes agree.
    """
    fields, pml = _build(**row)
    columns = family.unit_coefficient_columns(numpy, tuple(fields.grid.shape))
    for axis in "xyz":
        for stem in ("kms", "sinv"):
            synthesized = columns[f"{stem}_{axis}"]
            for suffix in ("", "_h"):
                engine = getattr(pml, f"{stem}_{axis}{suffix}")
                assert synthesized.shape == engine.shape
                assert synthesized.dtype == engine.dtype
                assert numpy.array_equal(_words(synthesized), _words(engine))


def _special_pairs():
    specials = {"-0.0": numpy.float32(-0.0), "+0.0": numpy.float32(0.0),
                "subnormal+": numpy.float32(1e-45),
                "subnormal-": numpy.float32(-1e-45),
                "normal+": numpy.float32(0.375), "normal-": numpy.float32(-0.375)}
    pairs = list(itertools.product(specials, repeat=2))
    target_re = numpy.array([specials[a] for a, _ in pairs], numpy.float32)
    curl_re = numpy.array([specials[b] for _, b in pairs], numpy.float32)
    zeros = numpy.zeros(len(pairs), numpy.float32)
    field = _complex_from_halves(target_re, zeros).reshape(-1, 1, 1)
    curl = _complex_from_halves(curl_re, zeros).reshape(-1, 1, 1)
    return pairs, field, curl


@pytest.mark.parametrize("band", [(1e-6, 1e-3), (1e-3, 1.0), (1.0, 1e3)],
                         ids=["1e-6..1e-3", "1e-3..1e0", "1e0..1e3"])
def test_the_dsig_only_form_on_unit_columns_is_target_minus_curl(band):
    """The kernel's form on every cell of this binding IS ``target -= curl``.

    ``stepping._apply_dsig_only_update`` is the array path's dsig-only form, the one
    the kernel selects on a cell outside every membership run. A PHYSICAL BAND fill
    with a VACUITY FLOOR: a no-op agreeing with a no-op measures nothing.
    """
    low, high = band
    rng = numpy.random.default_rng(SEED)
    shape = (12, 11, 10)
    parts = (numpy.exp(rng.uniform(numpy.log(low), numpy.log(high), (4,) + shape))
             * rng.choice([-1.0, 1.0], (4,) + shape)).astype(numpy.float32)
    field = (parts[0] + 1j * parts[1]).astype(numpy.complex64)
    curl = (parts[2] + 1j * parts[3]).astype(numpy.complex64)

    plain = field.copy()
    plain -= curl                                        # stepping._apply_curl
    moved = int((_words(field) != _words(plain)).sum())
    assert moved == field.size * 2, f"VACUOUS: only {moved} words moved"

    dsig_only = field.copy()
    one = numpy.float32(1.0)
    stepping._apply_dsig_only_update(dsig_only, curl, one, one)
    differing = int((_words(plain) != _words(dsig_only)).sum())
    assert differing == 0, f"{differing} / {field.size * 2} words differ"


def test_the_dsig_only_form_on_unit_columns_keeps_every_special_pair():
    """All 36 (target, curl) pairs of the special values, the signed zeros included:
    the binding has no residual, so the gate's planted leg requires identity."""
    pairs, field, curl = _special_pairs()
    plain = field.copy()
    plain -= curl
    dsig_only = field.copy()
    one = numpy.float32(1.0)
    stepping._apply_dsig_only_update(dsig_only, curl, one, one)
    a, b = _words(plain).ravel(), _words(dsig_only).ravel()
    divergent = {pairs[n] for n in range(len(pairs))
                 if a[2 * n] != b[2 * n] or a[2 * n + 1] != b[2 * n + 1]}
    assert divergent == set(), sorted(divergent)


def test_the_previous_binding_loses_exactly_the_pattern_the_gate_plants():
    """The previous binding — the recurrence on every cell with a zeroed auxiliary —
    is the gate's ``previous_binding`` mutation, and it is caught ONLY by the planted
    leg, so its divergent set is pinned exactly: ``(-0.0, +0.0)`` and nothing else.
    The auxiliary reaches ``+0.0 - (+0.0) = +0.0`` and ``-0.0 + (+0.0)`` is ``+0.0``.
    """
    pairs, field, curl = _special_pairs()
    plain = field.copy()
    plain -= curl
    previous = field.copy()
    one = numpy.float32(1.0)
    stepping._apply_pml_update(previous, curl, one, one, one, one,
                               numpy.zeros(field.shape, numpy.complex64))
    a, b = _words(plain).ravel(), _words(previous).ravel()
    divergent = {pairs[n] for n in range(len(pairs)) if a[2 * n] != b[2 * n]}
    assert divergent == {("-0.0", "+0.0")}, sorted(divergent)


def test_no_choice_of_auxiliary_zero_sign_repairs_the_previous_binding():
    """Why the previous binding could not be repaired by seeding, and the form select
    was the fix: seeding the auxiliary at ``-0.0`` moves the sign loss from
    ``field += fu`` to ``field -= fu_previous`` and leaves the divergent set where it
    was."""
    specials = (numpy.float32(-0.0), numpy.float32(0.0), numpy.float32(0.375))
    pairs = list(itertools.product(range(len(specials)), repeat=2))
    target_re = numpy.array([specials[a] for a, _ in pairs], numpy.float32)
    curl_re = numpy.array([specials[b] for _, b in pairs], numpy.float32)
    zeros = numpy.zeros(len(pairs), numpy.float32)
    field = _complex_from_halves(target_re, zeros).reshape(-1, 1, 1)
    curl = _complex_from_halves(curl_re, zeros).reshape(-1, 1, 1)
    one = numpy.float32(1.0)

    plain = field.copy()
    plain -= curl

    divergent = {}
    for label, auxiliary in (
            ("+0.0", numpy.zeros(field.shape, numpy.complex64)),
            ("-0.0", _complex_from_halves(
                numpy.full(field.shape, numpy.float32(-0.0)),
                numpy.full(field.shape, numpy.float32(-0.0))))):
        previous = field.copy()
        stepping._apply_pml_update(previous, curl, one, one, one, one, auxiliary)
        a, b = _words(plain).ravel(), _words(previous).ravel()
        divergent[label] = {(specials[pairs[n][0]].item(), specials[pairs[n][1]].item())
                            for n in range(len(pairs)) if a[2 * n] != b[2 * n]}
    assert divergent["+0.0"] == divergent["-0.0"], divergent
    assert len(divergent["+0.0"]) == 1, divergent


@J_ROWS
@pytest.mark.parametrize("sub_step", CURLS)
def test_the_plan_binds_the_absorber_free_record(family, row, sub_step):
    """No membership run on either sub-lattice of any axis, and the grid's presence.

    That record is what makes the kernel's dsig-only form the only one; a run would
    put the recurrence (and a read of the placeholder auxiliary) on its cells.
    """
    from meep_gpu.chunk_map import RECORD_PRESENCE_WORD

    fields, pml = _build(**row)
    grid = fields.grid
    plan = family.plan_complex_no_pml_curl(fields, pml, sub_step)
    record = numpy.asarray(plan._inner.chunk_bounds)
    assert not record[:RECORD_PRESENCE_WORD].any(), record.tolist()
    expected = sum(1 << axis for axis in range(3) if not grid.is_invariant(axis))
    assert int(record[RECORD_PRESENCE_WORD]) == expected
    assert expected == (7 if row is J3 else 3)


@pytest.mark.parametrize("sub_step", CURLS)
def test_the_auxiliary_is_never_touched_at_launch(family, sub_step):
    """The placeholder auxiliary is the kernel's argument only: ``run`` neither zeroes
    nor reads it (the record has no recurrence cell), so a value a launch could not
    have left there survives every launch of the recorder."""
    fields, pml = _build()
    plan = family.plan_complex_no_pml_curl(fields, pml, sub_step)
    auxiliaries = plan._auxiliaries
    assert len(auxiliaries) == 3
    for auxiliary in auxiliaries:
        auxiliary[...] = numpy.complex64(-1.5 - 0.25j)

    seen = []

    class _Recorder:
        sub_step = "recorder"

        def run(self, guard=None):
            seen.append([int(numpy.count_nonzero(a != numpy.complex64(-1.5 - 0.25j)))
                         for a in auxiliaries])

    plan._inner = _Recorder()
    for _ in range(3):
        plan.run()
    assert seen == [[0, 0, 0]] * 3, seen


@pytest.mark.parametrize("sub_step", CURLS)
def test_the_auxiliary_is_this_products_own_and_never_the_engines(family, sub_step):
    """The engine's ``fu_*`` must not be bound even where a Fields happens to own one."""
    fields, pml = _build()
    # A Fields that HAPPENS to own fu_* volumes while the layer is inert. The
    # engine will not build this, and that is the point: the binding must be this
    # product's own by construction, not by the engine's arrays being absent.
    planted = {}
    for name in ("fu_" + t for t in SUB_STEPS[sub_step]["targets"]):
        planted[name] = numpy.full(tuple(fields.grid.shape), numpy.complex64(9 + 9j))
        setattr(fields, name, planted[name])
    plan = family.plan_complex_no_pml_curl(fields, pml, sub_step)
    assert plan is not None
    for auxiliary in plan._auxiliaries:
        assert all(auxiliary is not array for array in planted.values())
        assert not numpy.any(auxiliary)          # this product's own, zero at build


@J_ROWS
def test_step_D_sources_are_the_B_arrays_the_accessor_serves(family, row):
    """Not substituted: ``get_H`` returns the B array itself without PML."""
    fields, _ = _build(**row)
    assert fields.Hx is None                     # the attribute is absent ...
    sources = family.source_arrays(fields, "step_D")
    assert [id(a) for a in sources] == [id(fields.Bx), id(fields.By), id(fields.Bz)]


@J_ROWS
def test_step_B_sources_are_the_stored_E_arrays(family, row):
    fields, _ = _build(**row)
    sources = family.source_arrays(fields, "step_B")
    assert [id(a) for a in sources] == [id(fields.Ex), id(fields.Ey), id(fields.Ez)]


# ---------------------------------------------------------------------------
# The shared-clause edit must not have moved the incumbent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("thickness", [0, 2], ids=["inert", "active"])
@pytest.mark.parametrize("sub_step", CURLS)
def test_the_default_of_require_active_pml_reproduces_the_incumbent(thickness,
                                                                    sub_step):
    """The parameter added to the shared clause list is inert by default."""
    fields, pml = _build(pml_thickness=thickness)
    explicit = complex_module._complex_grid_reasons(
        fields, pml, fields.grid, require_active_pml=True)
    default = complex_module._complex_grid_reasons(fields, pml, fields.grid)
    assert explicit == default


@pytest.mark.parametrize("thickness", [0, 2], ids=["inert", "active"])
def test_clause_three_is_the_only_clause_the_inversion_moves(thickness):
    """Exactly ONE numbered question separates the two arms, measured as a set diff."""
    fields, pml = _build(pml_thickness=thickness)
    active = set(complex_module._complex_grid_reasons(
        fields, pml, fields.grid, require_active_pml=True))
    inert = set(complex_module._complex_grid_reasons(
        fields, pml, fields.grid, require_active_pml=False))
    assert active ^ inert <= {
        "no active PML layer (this product implements the complex "
        "split-field path only)",
        "an active PML layer is installed (that is the complex "
        "split-field product's path, not this one)"}
    assert len(active ^ inert) == 1


# ---------------------------------------------------------------------------
# The gate's release contract
# ---------------------------------------------------------------------------
#
# The gate itself needs a device. Its RELEASE CONTRACT does not, and a contract
# that cannot refuse is the same hazard as a test that cannot fail — so it is
# mutation-tested here, on a laptop, before any device time is spent on it.

GATE_PATH = (pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
             / "gate_triton_complex_no_pml_curl.py")


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location(
        "gate_triton_complex_no_pml_curl", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _released_payload():
    """The shape a fully successful device run writes."""
    cases = []
    for tag in ("J3", "J2"):
        for sub_step in CURLS:
            for shape_tag in ("inert_layer", "no_layer"):
                cases.append({"case": f"{tag}_{sub_step}_{shape_tag}",
                              "prediction": "IDENTICAL", "verdict": "IDENTICAL"})
    for name in ("J3_step_B_PHASEOFF_CONTROL", "J3_step_D_PHASEOFF_CONTROL",
                 "J3_step_B_STALESOURCE_CONTROL"):
        cases.append({"case": name, "prediction": "DIVERGENT",
                      "verdict": "DIVERGENT"})
    for name in ("J3_step_B_MUTATION_record_with_a_run",
                 "J2_step_B_MUTATION_presence_all_present",
                 "J3_step_D_MUTATION_record_with_a_run",
                 "J2_step_D_MUTATION_presence_all_present",
                 "J3_step_B_MUTATION_unit_column_one_ulp"):
        cases.append({"case": name, "prediction": "DIVERGENT",
                      "verdict": "DIVERGENT"})
    for name in ("J3_step_B_NULL_aux_poisoned", "J3_step_D_NULL_aux_poisoned"):
        cases.append({"case": name, "prediction": "IDENTICAL",
                      "verdict": "IDENTICAL"})
    planted = []
    for tag in ("J3", "J2"):
        for sub_step in CURLS:
            planted.append({"case": f"{tag}_{sub_step}_minus_zero_target_zero_curl:shipped",
                            "binding": "shipped", "prediction": "IDENTICAL",
                            "verdict": "IDENTICAL", "differing": 0, "compared": 11250,
                            "negative_zero_words_stored": 2352})
            planted.append({"case": f"{tag}_{sub_step}_minus_zero_target_zero_curl:"
                                    f"previous_binding",
                            "binding": "previous_binding", "prediction": "DIVERGENT",
                            "verdict": "DIVERGENT", "differing": 1152, "compared": 11250,
                            "negative_zero_words_stored": 2352})
    return {
        "subnormal_policy_installed": "keep",
        "cases": cases,
        "planted": {"cases": planted},
        "refusals": {"disjoint": True, "all_agree": True},
    }


def test_a_fully_successful_run_is_released(gate):
    verdict = gate.validate_payload(_released_payload())
    assert verdict["released"], verdict["reasons"]


@pytest.mark.parametrize("break_it, needle", [
    (lambda p: p.update(subnormal_policy_installed="flush"), "keep"),
    (lambda p: p.update(cases=[]), "no cases ran"),
    (lambda p: p["cases"].__setitem__(0, dict(p["cases"][0], verdict="DIVERGENT")),
     "verdict"),
    (lambda p: p["cases"].__setitem__(8, dict(p["cases"][8], verdict="IDENTICAL")),
     "control did not diverge"),
    (lambda p: p["cases"].__setitem__(11, dict(p["cases"][11], verdict="IDENTICAL")),
     "mutation was NOT caught"),
    (lambda p: p["cases"].__setitem__(16, dict(p["cases"][16], verdict="DIVERGENT")),
     "declared null moved a byte"),
    (lambda p: p.update(planted={"cases": []}), "planted"),
    (lambda p: p["planted"]["cases"].__setitem__(
        0, {"case": "x", "binding": "shipped"}), "no differing count"),
    (lambda p: p["planted"]["cases"].__setitem__(
        0, dict(p["planted"]["cases"][0], verdict="DIVERGENT", differing=4)),
     "planted verdict"),
    (lambda p: p["planted"]["cases"].__setitem__(
        1, dict(p["planted"]["cases"][1], verdict="IDENTICAL", differing=0)),
     "planted verdict"),
    (lambda p: p["planted"]["cases"].__setitem__(
        0, dict(p["planted"]["cases"][0], negative_zero_words_stored=0)),
     "no -0 word was stored"),
    (lambda p: p["planted"].update(cases=p["planted"]["cases"][:3]),
     "it requires 4 each"),
    (lambda p: p.update(refusals={"disjoint": False, "all_agree": True}),
     "UNSELECTED"),
    (lambda p: p.update(refusals={"disjoint": True, "all_agree": False}),
     "disagreed"),
    (lambda p: p.update(cases=[c for c in p["cases"]
                               if "CONTROL" not in c["case"]]), "controls ran"),
    (lambda p: p.update(cases=[c for c in p["cases"]
                               if "MUTATION" not in c["case"]]), "mutations ran"),
    (lambda p: p.update(cases=[c for c in p["cases"]
                               if "NULL" not in c["case"]]), "declared nulls ran"),
])
def test_the_release_contract_refuses_each_way_it_can_be_broken(gate, break_it,
                                                                needle):
    """Each mutation of a released payload must be REFUSED, and by name."""
    payload = _released_payload()
    break_it(payload)
    verdict = gate.validate_payload(payload)
    assert not verdict["released"], payload
    assert any(needle in reason for reason in verdict["reasons"]), verdict["reasons"]


def test_the_contract_requires_the_shipped_planted_rows_identical(gate):
    """The pattern the previous binding lost is now REQUIRED identical: a shipped
    planted row that differs refuses the release, by name."""
    payload = _released_payload()
    payload["planted"]["cases"][2]["differing"] = 86425
    payload["planted"]["cases"][2]["verdict"] = "DIVERGENT"
    verdict = gate.validate_payload(payload)
    assert not verdict["released"]
    assert any("planted verdict" in reason for reason in verdict["reasons"])


def test_the_gate_header_reports_the_device_result_and_known_residual(gate):
    """The harness header must agree with the readable certification record: the
    2026-08-17 release ran on the PREVIOUS binding, and its residual is that
    binding's, which the present gate arms as a mutation."""
    text = GATE_PATH.read_text(encoding="utf-8")
    assert "DEVICE RESULT (2026-08-17), ON THE PREVIOUS BINDING" in text
    assert "triton_complex_no_pml_curl_device_gate" in text
    assert "86,450/281,250" in text and "3,552/11,250" in text
    assert "NO LEG HAS BEEN RUN" not in text
    assert "previous_binding" in text and "leg_planted" in text
