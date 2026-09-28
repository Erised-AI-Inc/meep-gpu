"""Merge bar for the Metal NO-PML CONSTITUTIVE family.

THE FAMILY SHIPS NO KERNEL, so there is no arithmetic a laptop cannot hold and no
part of this file needs a GPU. That is not a shortcut — it is the family's one
distinguishing property, and it is asserted here rather than assumed: the covered
arm compiles nothing, launches nothing and binds nothing, so every test below runs
on a host with no MPS device.

WHAT IS PINNED, in the order the file runs it:

1. **the predicate is the SHARED function object**, not a copy. Which
   configurations are null is decided in ``triton_kernels.no_pml_constitutive``
   and nowhere else;
2. **the clauses**, each one, including the four CONSERVATIVE ones that refuse a
   configuration on which the null would be byte-exact — CONSTRUCTED AND CALLED
   through the Metal wrapper, with the table's ``moved_words`` re-measured against
   ``stepping`` rather than read back from itself;
3. **the FIFTEEN non-clauses** this family deliberately omits and every kernel
   predicate in the package carries, measured on a folded, a cylindrical, a
   complex, a Bloch, a beta, a BFAST, a conductive, a non-contiguous, a chi3 and
   two registered-pole configurations — and the sibling predicate is asserted to
   refuse each one BY NAME, because every one of those grids is already refused by
   "no active PML layer" and "it refused" alone certifies nothing;
3b. **the measurement all fifteen rest on** — every allocated volume rebound to an
   object that raises on any access, and the real ``stepping`` sub-steps shown to
   RETURN anyway, with three refused configurations as the controls that must
   RAISE. That converts "reads no pointer" from a reading of stepping.py:944-945
   into something executed. And **the slot->side table**, which was a second
   literal beside the imported one and a measured byte-visible hole: swapped, the
   composer put a null in the ``update_E`` slot of a run where ``update_E`` writes;
4. **the plan protocol** — ``run(contract=...)`` under every mode, empty
   ``variants``, empty ``volumes``, and the run counter a null family's gate has
   instead of a launch counter;
5. **the refusal battery**: every path the Metal entry point must NOT cover,
   constructed as an object and called — an unreadable layer, an unreadable
   ``stores_E`` / ``has_nonlinearity`` / ``has_offdiagonal_epsilon``, a
   non-iterable ``driven()``, a driving polarization, chi2/chi3, an off-diagonal
   row, an active layer, a stored E, and a ``Fields`` with no grid — paired with
   the control that the same entry point still ADMITS the covered configurations;
6. **the residency model**, including a REGRESSION TEST for the defect this port
   exists to close: declared ``planned``, the verdict refuses the family's own
   configuration naming arrays a no-PML run does not allocate; declared ``null``,
   it holds. The unallocated count is COUNTED (six of nine per side), not stated;
7. **identity against ``stepping.py`` itself**, over multiple full cycles, with
   dense subnormals and signed zeros seeded — and with the two vacuity floors that
   make it mean something: the census must be non-empty, and the CONTROLS must
   move bytes and be refused.

THE CHARACTERISTIC FAILURE OF A NULL FAMILY is that it certifies by agreeing with
a no-op, so "the bytes matched" is satisfied just as well by a run in which
NOTHING COULD HAVE MOVED. Every identity test here therefore carries a census
floor, and :func:`test_the_controls_move_bytes_and_are_refused` is the other half:
on configurations the predicate refuses, the array path must be shown to actually
write. Without that pair the whole file would pass against an empty ``Fields``.

WHAT THE 2026-08-15 AUDIT FOUND IN THIS FILE, because the same failure mode
applies to a test that measures nothing:

* ``test_the_covered_arm_needs_no_torch_and_no_mps_device`` ended in
  ``if "torch" not in sys.modules: assert "torch" not in sys.modules`` — a
  tautology, and measured, the branch was never even entered because an earlier
  test in this file calls a kernel predicate and imports torch. It asserted
  NOTHING. It is now a fresh-interpreter measurement;
* ``test_every_conservative_clause_says_so_where_it_fires`` fired no clause: it
  read ``moved_words`` out of the table and compared it to itself. Its name also
  claimed a property of the reason STRINGS that is FALSE for two of the four rows.
  Split into a table-shape test and a per-clause measurement;
* the breadth tests asserted only ``not sibling.covered``, which holds on every one
  of those grids by the no-PML clause alone;
* three entries in ``METAL_NON_CLAUSES`` had no case anywhere.

AND WHAT THE SECOND PASS FOUND, both of which this file could not have caught as it
stood:

* the ``METAL_NON_CLAUSES`` totality assertion ran ONE DIRECTION — every table key
  must have a case. An OMITTED entry has no key to orphan, so four clauses the
  siblings really fire on configurations this family ADMITS were simply not written
  down: ``chi2/chi3 is installed`` (H side), ``<volume> is not allocated``,
  ``kind 'sellmeier' is outside (...)`` and ``a susceptibility is registered``. The
  gate's ``over_coverage`` leg runs clause -> table on every cut; this file pins the
  key set and both directions of the case mapping;
* ``NULL_SLOTS`` was a second literal. Swapped, ``launch.plan_step`` filled the
  ``update_E`` SLOT with a null on a run where ``stepping.update_E`` moves 3240
  words, and every one of the 93 tests here stayed green.
"""

from __future__ import annotations

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import coverage as metal_coverage
from meep_gpu.metal_kernels import no_pml_constitutive as family
from meep_gpu.metal_kernels import subnormal
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import no_pml_constitutive as shared

#: NON-POWER-OF-TWO FIRST, as every sweep in this package requires. Recorded as a
#: PREDICTED NULL for the covered arm with its reason: the covered arm performs no
#: floating-point operation, so no association or FMA hazard exists for a Courant
#: number to expose. It is load-bearing on the CONTROL cases, where the array path
#: really runs the dsigw accumulation, and that is where it bites.
COURANT_NP2 = 0.35
COURANT_P2 = 0.5

#: Every field volume a seeded run carries. ``f_w_*`` and the stored ``E``/``H``
#: only exist on some of them, which is itself part of the finding — a no-PML run
#: allocates neither, and the residency defect this family closes was a refusal
#: that demanded mirrors of exactly those.
STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

IDENTITY_CYCLES = 3


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class UnrecognisedPole:
    """A registered pole of a kind no kernel here transcribes, DRIVING NOTHING.

    Drives nothing deliberately: a pole that drove an E component is refused by the
    null's own clause 3, and what these two objects measure is the pair of clauses
    the family DROPS — ``susceptibility_kind`` (9b) and
    ``susceptibility_registration`` (the sibling's E-side registration clause).
    """

    class _Kind:
        kind = "sellmeier"

    susceptibility = _Kind()

    def driven(self):
        return ()

    def drives(self, component):
        return False


class QuietLorentzianPole(UnrecognisedPole):
    """A pole of a COVERED kind that drives nothing — the discriminating one.

    ``lorentzian`` silences clause 9b, so the only thing left for the sibling to
    refuse is that a polarization is REGISTERED AT ALL. That is the clause this
    family deliberately does not carry: it asks ``driven()``, because a pole whose
    sigma is identically zero does not switch storage on (driver.py:1542-1547) and
    ``stepping.update_E`` really does return at :954 with a pole in the register.
    """

    class _Kind:
        kind = "lorentzian"

    susceptibility = _Kind()


def build(cell=(1.2, 1.0, 0.9), boundaries="periodic", courant=COURANT_NP2,
          pml_cells=0, seed=20260815, complex_fields=False, symmetry=(),
          cylindrical=False, m=0, k_point=(0.0, 0.0, 0.0), beta=0.0,
          bfast=(0.0, 0.0, 0.0), dimensions=3, storage=False, layer=True,
          pml_storage=False, conductivity=None, noncontiguous=(), nonlinear=False,
          poles=()):
    """A real Grid/Fields/PML triple, seeded with subnormals and signed zeros.

    ``layer=False`` builds a run with NO ``PML`` OBJECT AT ALL, which is a
    different configuration from an inert layer even though both reach the same
    ``stepping`` return. Both are exercised, and each case records which it was.

    ``conductivity`` and ``noncontiguous`` serve the two non-clauses that had no
    case anywhere until this round. ``noncontiguous`` rebinds a volume to a reversed
    VIEW of its own buffer: same bytes, ``_volume_reasons`` reports "is not
    C-contiguous", and ``stepping`` returns before forming an index into it.
    """
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                dimensions=dimensions, courant=courant, k_point=k_point,
                symmetry=symmetry, cylindrical=cylindrical, m=m, beta=beta,
                bfast_scaled_k=bfast, xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    pml = None
    if layer:
        thickness = tuple((pml_cells, pml_cells) if grid.shape[a] >= 2 * pml_cells + 2
                          else (0, 0) for a in range(3))
        pml = PML(grid=grid, thickness=thickness)
        if pml_cells or pml_storage:
            fields.enable_pml_storage()
    if storage:
        fields.enable_field_storage()
    seed_state(fields, grid, seed, complex_fields)
    if nonlinear:
        fields.set_nonlinear_volumes({n: 0.0 for n in ("Ex", "Ey", "Ez")},
                                     {n: 0.4 for n in ("Ex", "Ey", "Ez")})
    for state in tuple(poles):
        fields.polarizations.append(state)
    if conductivity is not None:
        fields.set_d_conductivity(
            np.full(grid.shape, np.float32(conductivity), dtype=np.float32))
    for name in tuple(noncontiguous):
        setattr(fields, name, getattr(fields, name)[::-1])
    return grid, fields, pml


def seed_state(fields, grid, seed: int, complex_fields: bool) -> None:
    """Fill every allocated volume non-degenerately.

    THREE CLASSES, DELIBERATELY, and each one is a pass condition somewhere below:
    ordinary normals, a ``+-0`` lattice, and DENSE SUBNORMALS. The subnormals are
    the class that matters most on this backend — the MPS executor flushes float32
    subnormals natively and has no lever, so any Metal code that touched these
    arrays would destroy them. A byte-identical result with a nonzero subnormal
    census is a direct measurement that nothing on the device touched them.
    """
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = (rng.standard_normal(grid.shape) * 0.37).astype(np.float32)
        flat = host.reshape(-1)
        flat[::17] = np.float32(-0.0)
        flat[7::23] = np.float32(0.0)
        # Dense subnormals: 1e-40 is inside the band and is representable exactly
        # enough to survive the cast on a keep-policy host, which arm64 is.
        flat[3::11] = np.float32(1e-40)
        flat[5::13] = np.float32(-3e-41)
        if complex_fields:
            array[...] = host + 1j * host[::-1, ::-1, ::-1]
        else:
            array[...] = host


def words(array):
    return np.ascontiguousarray(array).reshape(-1).view(np.uint32)


def snapshot(fields):
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def differing(left, right) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def total_differing(before, after) -> int:
    return sum(differing(before[name], after[name]) for name in before)


def census_of(state) -> int:
    return sum(subnormal.census(array) for array in state.values())


def signed_zeros_of(state) -> int:
    return sum(subnormal.signed_zero_census(array)["negative_zero"]
               for array in state.values())


# ---------------------------------------------------------------------------
# 1. One definition
# ---------------------------------------------------------------------------

def test_the_predicate_is_the_shared_function_object():
    """Not a copy, not a wrapper with its own clauses: the same object.

    Which configurations are null is decided in ``triton_kernels`` and nowhere
    else. A COPY would be the over-covering failure this package's coverage module
    exists to prevent — a clause added on one side and not the other, with nothing
    to fail.
    """
    assert family.null_constitutive_coverage is shared.null_constitutive_coverage
    assert family.NULL_SIDES is shared.NULL_SIDES
    assert family.CONSERVATIVE_CLAUSES is shared.CONSERVATIVE_CLAUSES
    assert family.STORED_E_ARM is shared.STORED_E_ARM


def test_the_metal_wrapper_adds_no_clause_of_its_own():
    """``metal_null_constitutive_coverage`` must agree with the shared predicate.

    Over a matrix that includes every argument the Metal signature adds. If the
    wrapper ever grew a clause, this is what would catch it — and the clause would
    then have to be justified in :data:`family.METAL_NON_CLAUSES`, which is why
    that table is data rather than prose.
    """
    grid, fields, pml = build()
    for side in ("H", "E"):
        for residency in (None, object(), metal_coverage):
            for variants in ((), ("off",), ("off", "fast")):
                got = family.metal_null_constitutive_coverage(
                    fields, pml, side, residency, variants)
                want = shared.null_constitutive_coverage(fields, pml, side)
                assert got.covered == want.covered
                assert got.reasons == want.reasons


# ---------------------------------------------------------------------------
# 2. The clauses
# ---------------------------------------------------------------------------

def test_an_inert_layer_admits_both_sides():
    _, fields, pml = build()
    assert not pml.is_active
    for side in ("H", "E"):
        verdict = family.metal_null_constitutive_coverage(fields, pml, side)
        assert verdict.covered, verdict.reasons


def test_no_layer_at_all_admits_both_sides():
    """A different configuration from an inert layer, reaching the same return."""
    _, fields, pml = build(layer=False)
    assert pml is None
    for side in ("H", "E"):
        assert family.metal_null_constitutive_coverage(fields, pml, side).covered


def test_an_active_layer_refuses_both_sides_by_name():
    _, fields, pml = build(pml_cells=2)
    assert pml.is_active
    for side in ("H", "E"):
        verdict = family.metal_null_constitutive_coverage(fields, pml, side)
        assert not verdict.covered
        assert any("active PML" in reason for reason in verdict.reasons), verdict


def test_stored_e_refuses_the_e_side_only():
    """``stepping.py:983`` branches on exactly ``stores_E``; H is never stored."""
    _, fields, pml = build(storage=True)
    assert fields.stores_E
    assert family.metal_null_constitutive_coverage(fields, pml, "H").covered
    verdict = family.metal_null_constitutive_coverage(fields, pml, "E")
    assert not verdict.covered
    assert any("stores_E" in reason for reason in verdict.reasons), verdict
    assert any("STORE arm" in reason for reason in verdict.reasons), verdict


def test_an_unreadable_flag_is_refused_rather_than_admitted():
    """The two load-bearing flags must not default to the ADMITTING answer.

    This is the defect ``_flag`` exists to close, and it is pinned on the Metal
    side too because this package's composer consults the predicate directly: a
    ``Fields`` proxy whose ``stores_E`` raised was once covered with NO reasons
    while the run underneath it wrote 4618 words.
    """
    _, fields, pml = build()

    class Unreadable:
        grid = fields.grid
        polarizations = ()

        @property
        def stores_E(self):
            raise RuntimeError("cannot answer")

        has_nonlinearity = False
        has_offdiagonal_epsilon = False

    verdict = family.metal_null_constitutive_coverage(Unreadable(), pml, "E")
    assert not verdict.covered
    assert any("stores_E" in reason for reason in verdict.reasons), verdict


def conservative_needle(name):
    """The exact object each :data:`shared.CONSERVATIVE_CLAUSES` row names.

    CONSTRUCTED AND CALLED, because reading a table is not evidence that the
    predicate does what the table says.
    """
    if name == "pml_storage_behind_an_inert_layer":
        return build(pml_storage=True)[1:]
    if name == "chi2_chi3_installed_without_stored_e":
        _, fields, pml = build()
        fields.set_nonlinear_volumes({n: 0.0 for n in ("Ex", "Ey", "Ez")},
                                     {n: 0.4 for n in ("Ex", "Ey", "Ez")})
        return fields, pml
    if name == "magnetic_susceptibility_on_the_h_side":
        class _MagneticState:
            def driven(self):
                return ()

            def drives(self, component):
                return component in ("Bx", "Hx")

        _, fields, pml = build()
        fields.polarizations.append(_MagneticState())
        return fields, pml
    if name == "a_polarization_that_cannot_answer_driven":
        class _Opaque:
            pass

        _, fields, pml = build()
        fields.polarizations.append(_Opaque())
        return fields, pml
    raise AssertionError(f"no needle for {name}")


def test_the_conservative_clause_table_has_the_shape_it_claims():
    """The table only. The MEASUREMENT is the next test, and that split is the point.

    THIS TEST USED TO BE CALLED ``..._says_so_where_it_fires`` AND FIRED NOTHING.
    It read ``moved_words`` out of the table and asserted the table's own number
    against itself — no needle built, no predicate called, no ``stepping`` run —
    while its name claimed a property of the reason STRINGS. Measured 2026-08-15,
    that claimed property is also FALSE for two of the four rows:
    ``magnetic_susceptibility_on_the_h_side`` and
    ``a_polarization_that_cannot_answer_driven`` do not contain the word
    "conservative" anywhere in the reason they emit. So the name has been corrected
    to what this test does, and what the file actually needs — the four clauses
    re-measured through the METAL wrapper — is next door.
    """
    assert len(shared.CONSERVATIVE_CLAUSES) == 4
    for name, entry in shared.CONSERVATIVE_CLAUSES.items():
        assert entry["moved_words"] == 0, (name, entry)
        assert entry["why_kept"], name
        assert entry["side"] in ("H", "E"), (name, entry)
        assert entry["needle"], name


@pytest.mark.parametrize("name", sorted(shared.CONSERVATIVE_CLAUSES))
def test_every_conservative_clause_is_re_measured_through_the_metal_wrapper(name):
    """"Conservative" is TWO claims at once and both are measured here.

    The predicate REFUSES, and the real ``stepping`` sub-step moves ZERO words on
    the same seeded object — i.e. the null would have been byte-exact and the
    refusal costs coverage rather than preventing a wrong answer. The table's own
    ``moved_words`` is compared against what ``stepping`` actually did rather than
    read back, so a row that stops being conservative fails HERE.

    Called through ``metal_null_constitutive_coverage`` deliberately. The predicate
    is the shared object and that is pinned above, but the Metal composer calls the
    WRAPPER, and a wrapper that grew a clause would change these verdicts.
    """
    row = shared.CONSERVATIVE_CLAUSES[name]
    fields, pml = conservative_needle(name)
    side = row["side"]
    verdict = family.metal_null_constitutive_coverage(fields, pml, side)
    assert not verdict.covered, (name, verdict.reasons)
    assert verdict.reasons, name

    state = snapshot(fields)
    assert census_of(state) > 0 and signed_zeros_of(state) > 0, (name, len(state))
    before = snapshot(fields)
    (stepping.update_H if side == "H" else stepping.update_E)(fields, pml)
    moved = total_differing(before, snapshot(fields))
    assert moved == row["moved_words"] == 0, (
        f"{name}: stepping moved {moved} words, so this clause is NOT conservative "
        f"— it is preventing a wrong answer, and the table says otherwise")


def test_pml_storage_behind_an_inert_layer_is_refused_conservatively():
    """The clause tranche-1's gate leg 8 recorded as the reason the null sat out."""
    _, fields, pml = build()
    fields.enable_pml_storage()
    for side in ("H", "E"):
        verdict = family.metal_null_constitutive_coverage(fields, pml, side)
        assert not verdict.covered
        assert any("conservatively" in reason for reason in verdict.reasons), verdict


# ---------------------------------------------------------------------------
# 3. The non-clauses — the family's one distinguishing claim
# ---------------------------------------------------------------------------

#: Each row is (name, kwargs, non-clause key, sibling predicate, the clause string
#: that sibling MUST report). ``metallic_walls`` carries ``None`` for the last two:
#: metallic is inside ``COVERED_BOUNDARIES``, so it exercises no omitted clause and
#: is the CONTROL that shows the named-clause assertion discriminates.
#:
#: ``cylindrical_m0``, ``d_conductivity`` and ``noncontiguous_Bx`` were ADDED
#: 2026-08-15: ``METAL_NON_CLAUSES`` carried ``cylindrical_axis``, ``conductivity``
#: and ``layout`` with no case anywhere in this suite or the gate.
#: The sixth element is the SIDES this row claims. ``chi3_h_side_only`` claims one:
#: the null admits H (update_H returns at :916-917 whatever is installed) and is
#: REFUSED on E, conservatively. Declared per row so the assertion can stay strict
#: for the other rows instead of being weakened for all of them.
BREADTH = (
    ("fold_y", dict(cell=(1.2, 1.6, 1.4), symmetry=("y",)),
     "fold", ("constitutive", "H"), "is folded by a mirror plane", ("H", "E")),
    ("complex_storage", dict(complex_fields=True),
     "storage_width", ("constitutive", "H"), "force_complex_fields=True", ("H", "E")),
    ("bloch_k", dict(complex_fields=True, k_point=(0.4, -1.3, 0.7)),
     "bloch_phase", ("constitutive", "H"), "is not exactly zero", ("H", "E")),
    ("beta_kz", dict(cell=(1.2, 1.0, 0.0), dimensions=2, beta=0.2,
                     complex_fields=True),
     "beta", ("constitutive", "H"), "special_kz beta=", ("H", "E")),
    ("bfast", dict(cell=(0.4, 0.4, 1.2), bfast=(0.31, 0.0, 0.0)),
     "bfast", ("constitutive", "H"), "BFAST is active", ("H", "E")),
    ("cylindrical_m0", dict(complex_fields=True, cylindrical=True, m=0,
                            dimensions=2, cell=(1.2, 0.0, 1.4)),
     "cylindrical_axis", ("constitutive", "H"),
     "cylindrical (Dcyl) coordinates are not carried", ("H", "E")),
    ("d_conductivity", dict(conductivity=0.5),
     "conductivity", ("curl", "step_D"), "a conductivity is installed on Dx",
     ("H", "E")),
    ("noncontiguous_Bx", dict(noncontiguous=("Bx",)),
     "layout", ("constitutive", "H"), "Bx is not C-contiguous", ("H", "E")),
    # ADDED 2026-08-15: three clauses the siblings fire on a configuration this
    # family ADMITS, for which METAL_NON_CLAUSES had no entry at all.
    ("chi3_h_side_only", dict(nonlinear=True),
     "nonlinearity", ("constitutive", "H"), "chi2/chi3 is installed", ("H",)),
    ("unrecognised_pole", dict(poles=(UnrecognisedPole(),)),
     "susceptibility_kind", ("constitutive", "H"), "kind 'sellmeier' is outside",
     ("H", "E")),
    ("quiet_lorentzian_pole", dict(poles=(QuietLorentzianPole(),)),
     "susceptibility_registration", ("constitutive", "E"),
     "a susceptibility is registered", ("H", "E")),
    ("metallic_walls", dict(boundaries="metallic"), None,
     ("constitutive", "H"), None, ("H", "E")),
)

#: Subtracted before a sibling's residual is read. EVERY grid in ``BREADTH`` has an
#: inactive absorber, so ``_grid_reasons`` clause 3 refuses all of them whatever the
#: dropped clause does — which is what made the old ``assert not covered`` vacuous.
UNIVERSAL_SIBLING_CLAUSE = "no active PML layer"

GRID_FEATURE_MARKERS = ("force_complex_fields", "mirror plane", "folded by",
                        "cylindrical", "r = 0 axis", "k_point", "BFAST",
                        "special_kz beta", "chi2/chi3", "boundary")

BREADTH_IDS = [row[0] for row in BREADTH]


def sibling_verdict(kind, argument, fields, pml):
    if kind == "constitutive":
        return metal_coverage.constitutive_coverage(fields, pml, argument, object())
    return metal_coverage.pml_curl_coverage(fields, pml, argument, object())


@pytest.mark.parametrize("name,kwargs,non_clause,sibling,clause,sides", BREADTH,
                         ids=BREADTH_IDS)
def test_the_null_is_admitted_where_every_kernel_predicate_refuses(
        name, kwargs, non_clause, sibling, clause, sides):
    """The claim no other predicate in this package makes.

    A sub-step that returns before its first statement reads no pointer, forms no
    index and rounds no float — so the fold, the Bloch phase, the storage width,
    beta, BFAST, the cylindrical axis, a conductivity, a non-contiguous volume, a
    chi3 factor on the H side and an unrecognised pole have nothing to be wrong
    about. Measured here rather than argued, and the SIDE the row does NOT claim is
    asserted REFUSED rather than left unmentioned.
    """
    _, fields, pml = build(**kwargs)
    for side in ("H", "E"):
        verdict = family.metal_null_constitutive_coverage(fields, pml, side)
        if side in sides:
            assert verdict.covered, (name, side, verdict.reasons)
        else:
            assert not verdict.covered, (
                f"{name}: side {side} is not claimed by this row and was admitted")


@pytest.mark.parametrize("name,kwargs,non_clause,sibling,clause,sides", BREADTH,
                         ids=BREADTH_IDS)
def test_the_kernel_predicates_refuse_what_the_null_admits_BY_NAME(
        name, kwargs, non_clause, sibling, clause, sides):
    """The other half, and it has to name the clause or it measures nothing.

    This test asserted only ``not covered`` for a round. Measured 2026-08-15: every
    grid here already carries ``_grid_reasons`` clause 3 ("no active PML layer"),
    the residency clause and six "is not allocated" clauses, so ``not covered``
    would have held with the fold, complex, Bloch, beta and BFAST clauses ALL
    deleted from the sibling. The universal clause is subtracted and the named one
    must appear in what remains — the Metal analogue of the Triton suite's
    ``_residual()``.
    """
    _, fields, pml = build(**kwargs)
    verdict = sibling_verdict(sibling[0], sibling[1], fields, pml)
    assert not verdict.covered, (name, verdict.reasons)
    residual = [r for r in verdict.reasons if UNIVERSAL_SIBLING_CLAUSE not in r]
    if clause is None:
        leaked = [r for r in residual
                  if any(marker in r for marker in GRID_FEATURE_MARKERS)]
        assert not leaked, (
            f"{name} is the CONTROL and must exercise no omitted clause: {leaked}")
    else:
        assert any(clause in r for r in residual), (name, clause, residual)


@pytest.mark.parametrize("name,kwargs,non_clause,sibling,clause,sides", BREADTH,
                         ids=BREADTH_IDS)
def test_the_array_path_moves_nothing_on_every_breadth_grid(
        name, kwargs, non_clause, sibling, clause, sides):
    """The byte half of the breadth claim, on ``stepping`` itself.

    "Admitted" and "refused elsewhere" are both verdicts. This is the measurement
    behind them: on each of these grids the real ``stepping`` sub-steps move ZERO
    words, with a nonzero-word floor so a degenerate fixture cannot pass.
    """
    _, fields, pml = build(**kwargs)
    before = snapshot(fields)
    nonzero = sum(int(np.count_nonzero(words(a))) for a in before.values())
    assert nonzero > 0, f"{name}: VACUOUS — every seeded word is zero"
    for _ in range(IDENTITY_CYCLES):
        stepping.update_H(fields, pml)
        stepping.update_E(fields, pml)
        assert total_differing(before, snapshot(fields)) == 0, name


def test_the_non_clause_table_names_a_reason_for_every_omission():
    """Every clause this family drops must be recorded WITH the reason it is false.

    ``no_pml.py`` clause 3b was rewritten because a refusal fired where its reason
    was untrue. The inverse rule applies to an OMISSION: a clause every sibling
    carries and this one does not is a claim, and a claim needs a reason a reader
    can check.
    """
    assert set(family.METAL_NON_CLAUSES) == {
        "metal_backend", "residency_declaration", "subnormal_policy",
        "storage_width", "fold", "cylindrical_axis", "bloch_phase", "beta",
        "bfast", "conductivity", "layout",
        # ADDED 2026-08-15. All four fire on a configuration this family ADMITS
        # and had no entry; the gate's `over_coverage` leg is what found them, by
        # running the direction the totality assertion never ran.
        "nonlinearity", "volume_allocation", "susceptibility_kind",
        "susceptibility_registration"}
    for name, reason in family.METAL_NON_CLAUSES.items():
        assert len(reason) > 40, name


#: The non-clauses no GRID can exercise, and the test in this file that does.
NON_CLAUSE_CHECKS = {
    "metal_backend": "test_the_covered_arm_imports_no_torch_in_a_fresh_interpreter",
    "residency_declaration": "test_an_undeclared_residency_is_admitted_here_and_refused_next_door",
    "subnormal_policy": "test_the_keep_policy_refuses_the_siblings_and_not_this_arm",
    # No GRID can single this one out: it fires on EVERY no-PML configuration at
    # once, which is the same fact the residency model counts as six-of-nine.
    "volume_allocation": "test_the_siblings_refuse_an_unallocated_volume_and_this_arm_does_not",
}


def test_every_non_clause_has_a_case_or_a_named_check():
    """The table's own rule, enforced: an entry with no case is a claim nothing measured.

    ``METAL_NON_CLAUSES`` said the gate's breadth leg iterated it. It did not, and
    six of the eleven entries had no case anywhere. This assertion is what makes
    adding a twelfth entry without a home a red test rather than a silent claim.
    """
    homed = {row[2] for row in BREADTH if row[2]} | set(NON_CLAUSE_CHECKS)
    orphans = sorted(set(family.METAL_NON_CLAUSES) - homed)
    assert not orphans, orphans
    for check in NON_CLAUSE_CHECKS.values():
        assert check in globals(), check
    # THE OTHER DIRECTION, and it is the one that found the last four entries. A
    # row naming a key the table does not carry is a case measuring a claim nobody
    # made; the gate's `over_coverage` leg runs the full clause -> table sweep.
    unknown = sorted({row[2] for row in BREADTH if row[2]}
                     - set(family.METAL_NON_CLAUSES))
    assert not unknown, unknown


def test_the_covered_arm_imports_no_torch_in_a_fresh_interpreter():
    """The only family in this package whose gate needs no device — MEASURED.

    THIS TEST WAS A TAUTOLOGY. It ended in ``if "torch" not in sys.modules: assert
    "torch" not in sys.modules``, which cannot fail; and measured, ``torch`` IS in
    ``sys.modules`` by the time it runs, because the breadth tests above call a
    kernel predicate, so the branch was not even taken. It asserted nothing at all.

    The claim is only measurable in a process that has not already imported torch
    for some other reason, so the whole covered path — predicate, builder, plan,
    launch — runs in a FRESH interpreter and reports whether torch arrived. The
    ``runs`` count is carried so a subprocess that imported nothing because it did
    nothing cannot pass.
    """
    import json
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    program = f"""
import json, sys
sys.path.insert(0, {root!r})
import numpy as np
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import no_pml_constitutive as family
from meep_gpu.pml import PML

grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries="periodic",
            dimensions=3, courant=0.35, xp=np)
fields = Fields(grid=grid)
pml = PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
step = family.plan_metal_null_constitutive_step(
    fields, pml, live=("step_B", "update_H", "step_D", "update_E"))
step.run()
step.run("fast")
print("VERDICT " + json.dumps({{
    "torch": "torch" in sys.modules,
    "runs": sum(p.runs for p in step.plans.values()),
    "covered": list(step.covered)}}))
"""
    completed = subprocess.run([sys.executable, "-c", program],
                               capture_output=True, text=True, timeout=300,
                               check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    marked = [line for line in completed.stdout.splitlines()
              if line.startswith("VERDICT ")]
    assert len(marked) == 1, completed.stdout
    verdict = json.loads(marked[0][len("VERDICT "):])
    assert verdict["covered"] == ["update_H", "update_E"], verdict
    assert verdict["runs"] == 4, verdict
    assert verdict["torch"] is False, verdict


def test_an_undeclared_residency_is_admitted_here_and_refused_next_door():
    """The ``residency_declaration`` non-clause, both halves, called.

    Every kernel predicate refuses a plan built with no residency. This one accepts
    the argument and ignores it, and that is CORRECT rather than lax because it
    binds no buffer — so there is no private copy for the refusal to be about.
    """
    _, fields, pml = build()
    for side in ("H", "E"):
        assert family.metal_null_constitutive_coverage(
            fields, pml, side, residency=None).covered
    sibling = metal_coverage.constitutive_coverage(fields, pml, "H", None)
    assert not sibling.covered
    assert any("residency was not declared" in r for r in sibling.reasons), sibling


def test_the_siblings_refuse_an_unallocated_volume_and_this_arm_does_not():
    """The ``volume_allocation`` non-clause, both halves, CALLED.

    Every kernel predicate binds one pointer per volume and refuses an unallocated
    one. This arm binds none — and the refusal would fire on SIX of the nine
    volumes per side on the family's OWN configuration, which is the same
    measurement the residency ``null`` set exists for.
    """
    _, fields, pml = build()
    sibling = metal_coverage.constitutive_coverage(fields, pml, "H", object())
    allocation = [r for r in sibling.reasons if "is not allocated" in r]
    assert allocation, "the clause this non-clause is about did not fire"
    for slot in ("update_H", "update_E"):
        missing = [n for n in metal_coverage.sub_step_volumes(slot)
                   if getattr(fields, n, None) is None]
        assert len(missing) == 6, (slot, missing)
    for side in ("H", "E"):
        assert family.metal_null_constitutive_coverage(fields, pml, side).covered


def test_the_keep_policy_refuses_the_siblings_and_not_this_arm():
    """The ``subnormal_policy`` non-clause: the clause is shown to FIRE, here it is unreached.

    ``coverage._metal_backend_reasons`` clause 4 refuses a run resolved to ``keep``
    because the MPS executor flushes natively and has no lever. This arm has no
    operand, result or intermediate, so both policies are unreached rather than
    merely indistinguishable — and "unreached" is only interesting if the clause
    itself is demonstrated to exist and fire.
    """
    keep = subnormal.mps_policy_reasons("keep")
    flush = subnormal.mps_policy_reasons("flush")
    assert keep, "the keep clause did not fire, so dropping it certifies nothing"
    assert not flush, flush
    _, fields, pml = build()
    for side in ("H", "E"):
        assert family.metal_null_constitutive_coverage(fields, pml, side).covered


# ---------------------------------------------------------------------------
# 3b. The measurement every non-clause rests on, and the slot->side table
# ---------------------------------------------------------------------------

#: Poisoned in :func:`poison_every_volume`. ``scratch`` is included deliberately:
#: the PML constitutive path hands it to ``_apply_constitutive_pml``, so leaving it
#: real would give the ACTIVE-PML control one untouched array to work through.
POISON_VOLUMES = STATE + ("scratch",)


class PoisonedVolume:
    """Not ``None`` — so an allocation check passes — and unusable as an array.

    ``shape`` is a real class attribute so ``__getattr__`` does not intercept it: a
    predicate reading a shape is not the array ACCESS this is about. Everything
    else raises, unknown attributes included, because ``_apply_constitutive_pml``
    reaches for ``xp`` before it multiplies.
    """

    shape = (1, 1, 1)

    def __init__(self):
        self.dtype = np.dtype(np.float32)

    @staticmethod
    def _touched(how):
        raise AssertionError(f"the sub-step TOUCHED a field array ({how})")

    def __getattr__(self, name):
        self._touched(f"attribute {name!r}")

    def __getitem__(self, key):
        self._touched("__getitem__")

    def __setitem__(self, key, value):
        self._touched("__setitem__")

    def __array__(self, *args, **kwargs):
        self._touched("__array__")

    def __mul__(self, other):
        self._touched("__mul__")

    __rmul__ = __add__ = __radd__ = __sub__ = __rsub__ = __mul__

    def __iter__(self):
        self._touched("__iter__")


def poison_every_volume(fields):
    replaced = []
    for name in POISON_VOLUMES:
        if getattr(fields, name, None) is not None:
            setattr(fields, name, PoisonedVolume())
            replaced.append(name)
    return replaced


POISON = (
    ("covered_no_pml_H", dict(), "H", False),
    ("covered_no_pml_E", dict(), "E", False),
    ("covered_no_layer_E", dict(layer=False), "E", False),
    ("CONTROL_active_pml_H", dict(pml_cells=2), "H", True),
    ("CONTROL_active_pml_E", dict(pml_cells=2), "E", True),
    ("CONTROL_stored_e_E", dict(storage=True), "E", True),
)


@pytest.mark.parametrize("label,kwargs,side,must_raise", POISON,
                         ids=[row[0] for row in POISON])
def test_the_covered_sub_step_touches_no_field_array_at_all(
        label, kwargs, side, must_raise):
    """EVERY entry in ``METAL_NON_CLAUSES`` rests on this, and it was a READING.

    "A sub-step that returns before its first statement reads no pointer, forms no
    index and rounds no float" was transcribed from stepping.py:944-945 / :983-984
    and nothing executed it. Rebinding every allocated volume to an object that
    raises on any access and calling the real sub-step makes it a MEASUREMENT: if
    the call returns, the sub-step demonstrably touched no field array, and every
    clause about strides, indices, ghost rules and rounding modes is a clause about
    something that did not happen.

    THE CONTROLS ARE WHAT STOP IT BEING A TAUTOLOGY: on three configurations this
    family REFUSES, the same poison must be detected. Without them a poison nothing
    can trip would pass the first three rows.
    """
    _, fields, pml = build(**kwargs)
    verdict = family.metal_null_constitutive_coverage(fields, pml, side)
    replaced = poison_every_volume(fields)
    assert replaced, f"{label}: nothing was rebound, so nothing could be caught"
    call = stepping.update_H if side == "H" else stepping.update_E
    if must_raise:
        assert not verdict.covered, (label, verdict.reasons)
        with pytest.raises(Exception):
            call(fields, pml)
    else:
        assert verdict.covered, (label, verdict.reasons)
        call(fields, pml)  # must not raise: it may touch nothing


def test_the_slot_side_table_is_derived_and_not_a_second_literal():
    """One fact, one home. The literal was a MEASURED, byte-visible hole.

    ``NULL_SLOTS`` used to be spelled out beside the imported ``NULL_SIDES``, free
    to drift from it. Measured with the two entries swapped: ``launch.plan_step``
    on a no-PML ``stores_E`` run filled the ``update_E`` SLOT with a null plan whose
    own ``sub_step`` said ``update_H``, on a run where ``stepping.update_E`` moves
    thousands of words — and the whole suite and every gate leg stayed green.
    """
    assert family.NULL_SLOTS == {
        spec["sub_step"]: side for side, spec in shared.NULL_SIDES.items()}
    for slot, side in family.NULL_SLOTS.items():
        assert shared.NULL_SIDES[side]["sub_step"] == slot
        assert family.slot_side_reasons(slot) == ()
    assert family.slot_side_reasons("step_B"), "an unknown slot must be refused"


def test_a_crossed_slot_side_table_is_refused_rather_than_composed(monkeypatch):
    """The fail-closed clause, CONSTRUCTED AND CALLED through all three entry points.

    The byte visibility is the point: on a no-PML run with ``stores_E`` the E side
    is NOT null — ``stepping.update_E`` writes at :993 — so a composition that put
    a null in that slot would delete every one of those words. Measured here, and
    the refusal is asserted at the coverage entry point, at the builder and at the
    step composer, because each is reached by a different caller.
    """
    from meep_gpu.metal_kernels import arms, launch  # noqa: F401 - registers arms

    _, fields, pml = build(storage=True)
    before = snapshot(fields)
    stepping.update_E(fields, pml)
    moved = total_differing(before, snapshot(fields))
    assert moved > 0, "the fixture cannot show byte visibility"

    monkeypatch.setattr(family, "NULL_SLOTS", {"update_H": "E", "update_E": "H"})
    for slot in ("update_H", "update_E"):
        assert family.slot_side_reasons(slot), slot
    _, crossed_fields, crossed_pml = build(storage=False)
    context = arms.StepContext(crossed_fields, crossed_pml, object(), ("off",))
    for slot in ("update_H", "update_E"):
        verdict = family._arm_coverage(context, slot)
        assert not verdict.covered, (slot, verdict.reasons)
        assert any("sub_step" in r for r in verdict.reasons), verdict.reasons
        assert family._arm_plan(context, slot) is None, slot
    step = family.plan_metal_null_constitutive_step(
        crossed_fields, crossed_pml, live=LIVE_STEP)
    assert step.covered == (), step.covered
    assert set(step.refusals) == {"update_H", "update_E"}


def test_the_crossed_table_leaves_the_slot_on_the_array_path(monkeypatch):
    """Fail-closed all the way out to ``plan_step``: the slot is UNFILLED.

    ``_select_slot`` turns a refusing predicate into an unfilled slot, which is the
    array path, which is always correct. The shipped table is asserted in the same
    test so "nothing was filled" cannot pass by the composer being broken.
    """
    from meep_gpu.metal_kernels import launch
    from meep_gpu.metal_kernels.device import Residency

    # This test isolates the two NULL arms.  A stored-E no-PML run now has its
    # own non-null update_E owner, so choosing it here would test that new family
    # instead of the deliberately crossed null table below.
    _, fields, pml = build(storage=False)
    shipped = launch.plan_step(fields, pml, residency=Residency(), sources=())
    # On an MPS host the independent no-PML curl family owns B/D.  On a host
    # without MPS it refuses by its backend clause, while the null H plan remains
    # valid because it compiles and launches nothing.  The crossed-table invariant
    # must hold in both environments rather than making this host-only test claim
    # a device plan exists.
    curls_available = not metal_coverage._metal_backend_reasons(fields.grid)
    expected_shipped = (("step_B", "update_H", "step_D", "update_E")
                        if curls_available else ("update_H", "update_E"))
    assert shipped.replaces == expected_shipped, shipped.replaces
    assert "update_E" in shipped.plans

    monkeypatch.setattr(family, "NULL_SLOTS", {"update_H": "E", "update_E": "H"})
    _, crossed_fields, crossed_pml = build(storage=False)
    crossed = launch.plan_step(crossed_fields, crossed_pml,
                               residency=Residency(), sources=())
    # Both null slots are refused after the crossing; the independent curls, if
    # this host can build them, remain correctly selected.
    expected_crossed = ("step_B", "step_D") if curls_available else ()
    assert crossed.replaces == expected_crossed, crossed.replaces
    assert "update_H" not in crossed.plans
    assert "update_E" not in crossed.plans


def test_the_arm_gate_is_an_optimisation_and_never_a_verdict():
    """The claim ``_arm_gate``'s docstring makes, MEASURED in both halves.

    "The gate is an optimisation and never the verdict — if the two ever disagreed
    the predicate would win." Measured by computing every registered arm's verdict
    with the gate honoured and with it forced open: the ADMITTED set must be
    identical on every configuration, while the CONSULTED count must really differ
    somewhere, or the gate is not there at all.
    """
    from meep_gpu.metal_kernels import arms, launch  # noqa: F401 - registers arms

    configurations = (("inert", dict()), ("active", dict(pml_cells=2)),
                      ("stored_e", dict(storage=True)),
                      ("walls", dict(boundaries="metallic")),
                      ("no_layer", dict(layer=False)))
    suppressed = 0
    for name, kwargs in configurations:
        _, fields, pml = build(**kwargs)
        context = arms.StepContext(fields, pml, object(), ("off",))
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            gated, forced = [], []
            for spec in arms.registered(slot):
                admits = spec.coverage(context, slot).covered
                if admits:
                    forced.append(spec.label)
                if spec.gate is not None and not spec.gate(context):
                    suppressed += 1
                    continue
                if admits:
                    gated.append(spec.label)
            assert gated == forced, (name, slot, gated, forced)
    assert suppressed > 0, (
        "the gate suppressed no consultation anywhere, so 'the arm gate is an "
        "optimisation' is a claim about a gate that is not there")


# ---------------------------------------------------------------------------
# 4. The plan protocol
# ---------------------------------------------------------------------------

def test_the_plan_accepts_every_contraction_mode_and_launches_nothing():
    """A kernel plan RAISES on a mode it was not built with. A null must not.

    The raise is the right contract for a kernel plan — a guard selector that
    silently fell back to the pinned source would make the gate's guard leg
    vacuous. It is the wrong contract here: there is no floating-point operation
    to contract, so there is no mode under which the result could differ, and a
    ``KeyError`` would be a refusal to perform a sub-step the array path performs
    by returning.
    """
    _, fields, pml = build()
    plan = family.plan_metal_null_constitutive(fields, pml, "E")
    for mode in (None, "off", "fast", "a-mode-no-kernel-was-ever-built-with"):
        plan.run(mode)
    assert plan.runs == 4
    assert plan.modes_requested == (
        "off", "off", "fast", "a-mode-no-kernel-was-ever-built-with")


def test_the_plan_reports_no_variants_and_no_volumes():
    """The EMPTY TUPLES ARE THE ANSWER, not missing ones.

    ``variants`` empty because the plan holds no source string, so there is no
    compiled variant of one to report; an artifact showing ``('off',)`` would be
    claiming a compiled guard that does not exist. ``volumes`` empty because the
    plan binds no buffer, which is the residency consequence.
    """
    _, fields, pml = build()
    plan = family.plan_metal_null_constitutive(fields, pml, "H")
    assert plan.variants == ()
    assert plan.volumes == ()
    assert plan.performs_device_work is False
    assert plan.sub_step == "update_H"
    assert plan.return_site == "stepping.py:944-945"


def test_the_run_counter_is_what_a_null_family_has_instead_of_a_launch_count():
    """A byte comparison is trivially satisfied by a plan that was never invoked."""
    _, fields, pml = build()
    step = family.plan_metal_null_constitutive_step(
        fields, pml, live=("step_B", "update_H", "step_D", "update_E"))
    assert step.covered == ("update_H", "update_E")
    assert all(plan.runs == 0 for plan in step.plans.values())
    step.run()
    assert all(plan.runs == 1 for plan in step.plans.values())


def test_the_builder_refuses_out_of_coverage_rather_than_raising():
    _, fields, pml = build(pml_cells=2)
    assert family.plan_metal_null_constitutive(fields, pml, "H") is None
    assert family.plan_metal_null_constitutive(fields, pml, "E") is None


def test_an_unknown_side_raises_rather_than_refusing_quietly():
    _, fields, pml = build()
    with pytest.raises(ValueError):
        family.metal_null_constitutive_coverage(fields, pml, "Q")
    with pytest.raises(ValueError):
        family.MetalNullConstitutivePlan("Q")


# ---------------------------------------------------------------------------
# 4b. The refusal battery — every path that must NOT be covered, CALLED
# ---------------------------------------------------------------------------
#
# THE PREDICATE IS THE SHARED OBJECT and the Triton suite exercises these, so this
# battery is not re-deriving the verdicts. What it pins is that the METAL ENTRY
# POINT reaches every one of them: `metal_null_constitutive_coverage` is what
# `arms.py` calls, and a wrapper that swallowed an exception, defaulted an
# unreadable flag or short-circuited on `residency` would leave the Triton suite
# green while this package over-covered. Constructed and called, because reading
# the predicate is not evidence.


def _grid_of(fields):
    return fields.grid


def refusal_battery():
    """(label, fields, pml, side, a substring the refusal must contain)."""
    _, base, pml = build()
    grid = base.grid

    class NoGrid:
        grid = None
        polarizations = ()
        stores_E = False
        has_nonlinearity = False
        has_offdiagonal_epsilon = False

    class UnreadableLayer:
        @property
        def is_active(self):
            raise RuntimeError("cannot answer")

    class UnreadableNonlinearity:
        polarizations = ()
        stores_E = False
        has_offdiagonal_epsilon = False

        def __init__(self, grid):
            self.grid = grid

        @property
        def has_nonlinearity(self):
            raise RuntimeError("cannot answer")

    class UnreadableOffdiagonal:
        polarizations = ()
        stores_E = False
        has_nonlinearity = False

        def __init__(self, grid):
            self.grid = grid

        @property
        def has_offdiagonal_epsilon(self):
            raise RuntimeError("cannot answer")

    class BadDriven:
        def driven(self):
            return 17

    class DrivesEz:
        def driven(self):
            return ("Ez",)

        def drives(self, component):
            return component == "Ez"

    _, driven_fields, driven_pml = build()
    driven_fields.polarizations.append(DrivesEz())
    _, bad_fields, bad_pml = build()
    bad_fields.polarizations.append(BadDriven())
    _, nonlinear_fields, nonlinear_pml = build()
    nonlinear_fields.set_nonlinear_volumes(
        {n: 0.0 for n in ("Ex", "Ey", "Ez")},
        {n: 0.4 for n in ("Ex", "Ey", "Ez")})
    shape = grid.shape
    _, offdiag_fields, offdiag_pml = build()
    offdiag_fields.set_epsilon_volumes(
        {n: np.full(shape, np.float32(2.25), dtype=np.float32)
         for n in ("Ex", "Ey", "Ez")},
        {n: np.full(shape, np.float32(1.0 / 2.25), dtype=np.float32)
         for n in ("Ex", "Ey", "Ez")},
        chi1inv_offdiagonal={"Ez": {"Ex": np.full(shape, np.float32(0.05),
                                                  dtype=np.float32)}})
    _, active_fields, active_pml = build(pml_cells=2)
    _, stored_fields, stored_pml = build(storage=True)

    return (
        ("no grid, H", NoGrid(), pml, "H", "carries no grid"),
        ("no grid, E", NoGrid(), pml, "E", "carries no grid"),
        ("layer cannot answer is_active, H", base, UnreadableLayer(), "H",
         "does not report is_active"),
        ("layer cannot answer is_active, E", base, UnreadableLayer(), "E",
         "does not report is_active"),
        ("an active layer, H", active_fields, active_pml, "H", "active PML layer"),
        ("an active layer, E", active_fields, active_pml, "E", "active PML layer"),
        ("stores_E", stored_fields, stored_pml, "E", "stores_E is True"),
        ("a polarization drives Ez", driven_fields, driven_pml, "E", "drives Ez"),
        ("driven() is not iterable", bad_fields, bad_pml, "E",
         "does not report driven()"),
        ("chi2/chi3 installed", nonlinear_fields, nonlinear_pml, "E", "chi2/chi3"),
        ("an off-diagonal row", offdiag_fields, offdiag_pml, "E", "off-diagonal"),
        ("has_nonlinearity unreadable", UnreadableNonlinearity(grid), pml, "E",
         "does not report has_nonlinearity"),
        ("has_offdiagonal_epsilon unreadable", UnreadableOffdiagonal(grid), pml, "E",
         "does not report has_offdiagonal_epsilon"),
    )


BATTERY = refusal_battery()


@pytest.mark.parametrize("label,fields,pml,side,marker", BATTERY,
                         ids=[row[0] for row in BATTERY])
def test_the_metal_entry_point_refuses_every_uncovered_path_by_name(
        label, fields, pml, side, marker):
    verdict = family.metal_null_constitutive_coverage(fields, pml, side)
    assert not verdict.covered, (label, verdict.reasons)
    assert any(marker in reason for reason in verdict.reasons), (label,
                                                                 verdict.reasons)


def test_the_battery_is_not_a_predicate_that_refuses_everything():
    """The other half. A predicate that refused every input would pass the battery.

    The admitting configurations are the control, and they are asserted from the
    SAME entry point in the same session.
    """
    _, fields, pml = build()
    for side in ("H", "E"):
        assert family.metal_null_constitutive_coverage(fields, pml, side).covered
    _, no_layer, absent = build(layer=False)
    for side in ("H", "E"):
        assert family.metal_null_constitutive_coverage(no_layer, absent, side).covered


# ---------------------------------------------------------------------------
# 5. The residency model — the regression this port exists to close
# ---------------------------------------------------------------------------

LIVE_STEP = ("step_B", "update_H", "step_D", "update_E")


def test_declaring_a_null_slot_PLANNED_refuses_the_familys_own_configuration():
    """THE MEASURED DEFECT, pinned so it cannot come back.

    This is what tranche 1 did, having no other set to put a null in, and the
    verdict it produced on the one configuration this family exists for. SIX of the
    nine names it demands per side are ``None`` on such a run — this docstring said
    FOUR, and the count is now measured rather than remembered, by
    :func:`test_the_unallocated_count_is_six_of_nine_per_side` below: H is never
    stored without PML, E is served on demand as ``D * inv_eps`` and is not stored
    either, and ``f_w_*`` is allocated only by ``enable_pml_storage``.
    """
    verdict = metal_coverage.residency_coverage(
        mirrored=(), planned=("update_H", "update_E"), live=LIVE_STEP)
    assert not verdict.covered
    assert any("carry no mirror" in reason for reason in verdict.reasons), verdict


def test_the_unallocated_count_is_six_of_nine_per_side():
    """The number three files state about the false refusal, COUNTED.

    ``no_pml_constitutive.py``, ``coverage.residency_reasons`` and this file's own
    docstring all stated FOUR of nine. Measured over the whole demanded set through
    ``sub_step_volumes`` on the fixture the gate builds: SIX, both sides. A number
    nobody counts is a number that drifts, and this is what counts it.
    """
    _, fields, _ = build()
    for slot in ("update_H", "update_E"):
        demanded = metal_coverage.sub_step_volumes(slot)
        missing = [n for n in demanded if getattr(fields, n, None) is None]
        assert len(demanded) == 9, (slot, demanded)
        assert len(missing) == 6, (slot, missing)
    assert [n for n in metal_coverage.sub_step_volumes("update_H")
            if getattr(fields, n, None) is not None] == ["Bx", "By", "Bz"]
    assert [n for n in metal_coverage.sub_step_volumes("update_E")
            if getattr(fields, n, None) is not None] == ["Dx", "Dy", "Dz"]


def test_declaring_the_same_slot_NULL_holds():
    """The fix, and it is the same call with the slot in the right set."""
    verdict = metal_coverage.residency_coverage(
        mirrored=(), planned=(), live=LIVE_STEP, null=("update_H", "update_E"))
    assert verdict.covered, verdict.reasons


def test_a_null_slot_does_not_stale_a_mirror_either():
    """The other half of the invariant. A null writes nothing, so nothing goes stale.

    The mirror set is exactly ``step_B``'s own volumes plus the ``E`` the null
    ``update_E`` would write. Undeclared, the composer refuses — ``update_H`` reads
    mirrored ``B`` and ``update_E`` writes mirrored ``E``, both on the array path
    as far as it knows. Declared null, both are known to touch nothing.
    """
    mirrored = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez")
    stale = metal_coverage.residency_coverage(
        mirrored=mirrored, planned=("step_B",), live=LIVE_STEP)
    assert not stale.covered, "update_H/update_E must stale these mirrors undeclared"
    assert any("update_E runs on the array path" in reason
               for reason in stale.reasons), stale.reasons
    held = metal_coverage.residency_coverage(
        mirrored=mirrored, planned=("step_B",), live=LIVE_STEP,
        null=("update_H", "update_E"))
    assert held.covered, held.reasons


def test_planned_and_null_together_is_a_contradiction_and_is_refused():
    verdict = metal_coverage.residency_coverage(
        mirrored=(), planned=("update_H",), live=LIVE_STEP, null=("update_H",))
    assert not verdict.covered
    assert any("both planned and null" in reason for reason in verdict.reasons)


def test_a_null_declaration_for_a_sub_step_that_never_runs_is_refused():
    verdict = metal_coverage.residency_coverage(
        mirrored=(), planned=(), live=("step_B", "step_D"), null=("update_P",))
    assert not verdict.covered
    assert any("not live" in reason for reason in verdict.reasons), verdict


def test_an_undeclared_live_set_is_still_a_refusal_on_the_null_path():
    """``None`` is a REFUSAL for every declared set, never an empty one."""
    step = family.plan_metal_null_constitutive_step(
        build()[1], build()[2], live=None)
    assert not step.residency.covered
    assert any("live sub-step set was not declared" in reason
               for reason in step.residency.reasons)


def test_the_step_plan_answers_the_residency_question_correctly():
    """An EMPTY mirror registry is a real answer here; ``None`` is still a refusal.

    The distinction is the one ``residency_reasons`` is built on. A composition
    whose only plans are nulls legitimately mirrors NOTHING — and it has to say so
    by handing over a registry, because "no mirrors were declared" and "the mirror
    set was not declared" are different claims and only the first is safe.
    """
    from meep_gpu.metal_kernels.device import Residency

    _, fields, pml = build()
    step = family.plan_metal_null_constitutive_step(
        fields, pml, residency=Residency(), live=LIVE_STEP)
    assert step.covered == ("update_H", "update_E")
    assert step.residency.covered, step.residency.reasons
    assert "residency=held" in repr(step)


def test_an_undeclared_mirror_set_is_a_refusal_even_for_a_null_composition():
    _, fields, pml = build()
    step = family.plan_metal_null_constitutive_step(fields, pml, live=LIVE_STEP)
    assert not step.residency.covered
    assert any("mirror set was not declared" in reason
               for reason in step.residency.reasons)


# ---------------------------------------------------------------------------
# 6. Identity against stepping.py, with its floors and its controls
# ---------------------------------------------------------------------------

IDENTITY = (
    ("inert_layer_np2", dict(courant=COURANT_NP2)),
    ("inert_layer_p2", dict(courant=COURANT_P2)),
    ("no_layer_at_all", dict(layer=False)),
    ("metallic_walls", dict(boundaries="metallic")),
    ("fold_y", dict(cell=(1.2, 1.6, 1.4), symmetry=("y",))),
    ("complex_storage", dict(complex_fields=True)),
)


@pytest.mark.parametrize("name,kwargs", IDENTITY, ids=[n for n, _ in IDENTITY])
def test_the_covered_arm_leaves_every_live_array_byte_unchanged(name, kwargs):
    """MULTIPLE FULL CYCLES, not one call, against the ORIGINAL snapshot.

    A sub-step that wrote back exactly what it read would pass a single-call
    identity test and drift on the second. The census floors below are what stop
    this from passing against an empty ``Fields``: a zero-init run satisfies "no
    bytes moved" trivially, which is THE failure mode for a null family.
    """
    _, fields, pml = build(**kwargs)
    for side in ("H", "E"):
        assert family.metal_null_constitutive_coverage(fields, pml, side).covered

    before = snapshot(fields)
    assert before, f"{name}: no arrays were allocated at all"
    nonzero = sum(int(np.count_nonzero(words(a))) for a in before.values())
    assert nonzero > 0, f"{name}: VACUOUS — every seeded word is zero"

    step = family.plan_metal_null_constitutive_step(fields, pml, live=LIVE_STEP)
    for _ in range(IDENTITY_CYCLES):
        step.run()
        after = snapshot(fields)
        assert total_differing(before, after) == 0, name

    assert all(plan.runs == IDENTITY_CYCLES for plan in step.plans.values())


@pytest.mark.parametrize("name,kwargs", IDENTITY, ids=[n for n, _ in IDENTITY])
def test_the_array_path_itself_moves_nothing_on_the_covered_arm(name, kwargs):
    """The claim is about ``stepping.py``, so ``stepping.py`` is what is called.

    The plan reproducing a no-op is only interesting if the ARRAY PATH is a no-op
    here too — that is the substitution's whole content.
    """
    _, fields, pml = build(**kwargs)
    before = snapshot(fields)
    for _ in range(IDENTITY_CYCLES):
        stepping.update_H(fields, pml)
        stepping.update_E(fields, pml)
    assert total_differing(before, snapshot(fields)) == 0, name


def test_the_seeded_subnormal_and_signed_zero_classes_are_non_vacuous():
    """A census of ZERO is VACUOUS, not passed.

    AND ON THIS BACKEND THE SUBNORMAL CLASS IS THE POINTED ONE. The MPS executor
    flushes float32 subnormals natively and exposes no lever, so any Metal code
    that touched these arrays would destroy them. Byte-identity WITH a nonzero
    subnormal census is a direct measurement that nothing on the device touched
    them — a stronger statement here than the same leg makes on a keep host.
    """
    _, fields, _ = build()
    state = snapshot(fields)
    assert census_of(state) > 0, "no subnormal words were seeded"
    assert signed_zeros_of(state) > 0, "no negative zeros were seeded"


def test_subnormals_survive_the_covered_arm_bit_for_bit():
    _, fields, pml = build()
    before = snapshot(fields)
    seeded = census_of(before)
    step = family.plan_metal_null_constitutive_step(fields, pml, live=LIVE_STEP)
    for _ in range(IDENTITY_CYCLES):
        step.run()
    after = snapshot(fields)
    assert census_of(after) == seeded > 0
    assert total_differing(before, after) == 0


CONTROLS = (
    ("active_pml_np2", dict(pml_cells=2, courant=COURANT_NP2), ("H", "E")),
    ("active_pml_p2", dict(pml_cells=2, courant=COURANT_P2), ("H", "E")),
    ("no_pml_stored_e", dict(storage=True), ("E",)),
)


@pytest.mark.parametrize("name,kwargs,must_move", CONTROLS,
                         ids=[n for n, _, _ in CONTROLS])
def test_the_controls_move_bytes_and_are_refused(name, kwargs, must_move):
    """BOTH HALVES ARE PASS CONDITIONS. Either one alone measures nothing.

    An identity leg with no failing control is decorative: "no bytes moved" would
    be satisfied by a harness that never called anything. So each control asserts
    the predicate REFUSES *and* that the array path really does write on the named
    side.
    """
    _, fields, pml = build(**kwargs)
    for side in must_move:
        assert not family.metal_null_constitutive_coverage(
            fields, pml, side).covered, (name, side)

    for side in must_move:
        before = snapshot(fields)
        (stepping.update_H if side == "H" else stepping.update_E)(fields, pml)
        moved = total_differing(before, snapshot(fields))
        assert moved > 0, (
            f"{name}/{side}: the control moved NO bytes, so the identity leg it "
            f"exists to make non-vacuous is still unproven")


def test_the_stored_e_record_tracks_the_built_sibling_not_a_stale_refusal():
    """Arm S is built in its sibling module; the shared record must say so."""
    assert family.STORED_E_ARM["built"] is True
    reasons = family.stored_e_constitutive_reasons(*build(storage=True)[1:])
    assert reasons and not any("not built" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# 7. Composition and disjointness
# ---------------------------------------------------------------------------

def test_exactly_one_null_arm_is_registered_per_constitutive_slot():
    """Two arms under one label would make every constitutive slot ambiguous.

    Tranche 1 registered the null arm from ``launch.py`` directly against the
    Triton plan builder. This family owns that row now; a leftover second
    registration would not raise — it would silently turn both constitutive slots
    into a two-admitter refusal, which reads as "nothing covers this".
    """
    from meep_gpu.metal_kernels import arms, launch  # noqa: F401 - import registers

    for slot in ("update_H", "update_E"):
        nulls = [spec for spec in arms.registered(slot)
                 if spec.label == "no-PML null"]
        assert len(nulls) == 1, [repr(spec) for spec in nulls]
        assert nulls[0].family == family.FAMILY


def test_no_two_arms_admit_one_slot_on_any_configuration():
    """The partition is BY CONSTRUCTION — both sides ask ``pml.is_active`` — and
    it is measured rather than asserted."""
    from meep_gpu.metal_kernels import arms, launch  # noqa: F401 - import registers

    configurations = (
        ("inert", dict()),
        ("active", dict(pml_cells=2)),
        ("stored_e", dict(storage=True)),
        ("walls", dict(boundaries="metallic")),
        ("no_layer", dict(layer=False)),
    )
    for name, kwargs in configurations:
        _, fields, pml = build(**kwargs)
        context = arms.StepContext(fields, pml, object(), ("off",))
        for slot in ("update_H", "update_E", "step_B", "step_D"):
            admitting = [
                spec.label for spec in arms.registered(slot)
                if (spec.gate is None or spec.gate(context))
                and spec.coverage(context, slot).covered]
            assert len(admitting) <= 1, (name, slot, admitting)
