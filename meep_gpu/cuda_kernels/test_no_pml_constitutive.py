"""The hand-CUDA NO-PML constitutive arm: every clause, its refusal, and its bytes.

THIS FILE IS THE WHOLE MERGE BAR FOR THIS FAMILY, and unlike every other test in
this directory it is not half of one. The other slices push their bit-identity
question onto a device because a kernel has to be launched to be compared; this arm
has no kernel, so the question it has to answer -- "does the array path move
anything on the configurations the predicate admits?" -- is answered by calling
``stepping.update_H`` / ``stepping.update_E`` directly and comparing uint32 words.
That runs here, on a laptop, with no CuPy and no NVRTC.

WHAT THE DEVICE LEG THEN ADDS, and why it is still owed: the same assertions on real
device arrays, a whole-driver engine leg, and the mutation battery under a CUDA
context (which is the one host where a locale reset and a shared compile cache can
break a gate that compiles nothing --
``triton_kernels/no_pml_constitutive`` measured both). It is
``parity/meep_gpu/gate_cuda_no_pml_null_constitutive.py``, and
:data:`no_pml_constitutive.NULL_CONSTITUTIVE_CONFIRMATION` is where its numbers
land. ``test_confirmation_record_is_all_or_nothing`` refuses a half-filled record,
so a green suite can never be mistaken for a device verdict.

FIVE KINDS OF ASSERTION LIVE HERE:

1. **the verdict table** -- one row per clause, per side, pinned to the EXACT refusal
   string. In a first-refusal predicate the clause ORDER is part of the contract:
   a configuration that trips two clauses must name the one a caller can act on, and
   a table of ``(covered, reason)`` pairs is the only way to pin that;
2. **the transcription pins** -- the tables against ``coverage``'s, and every cited
   ``stepping.py`` line read back out of ``stepping.py``. A citation that has drifted
   into fiction is exactly the defect the validate-against-the-reference rule exists to prevent, and it is
   checkable rather than reviewable;
3. **the identity property** -- for every admitted fixture the real sub-step moves
   ZERO words over multiple full cycles, and for every refused control it moves
   words (or raises). An identity assertion with no failing control measures
   nothing, so both halves are here;
4. **the evidence table re-measured** -- :data:`CLAUSE_EVIDENCE` records a number per
   clause, and this file recomputes every one of them. The adjectives
   ("conservative", "shadowed") are derived from the numbers rather than asserted;
5. **the partition and the mutations** -- this arm and the three shipped constitutive
   predicates must never both admit one slot, and every clause must be shown to
   change a verdict when it is deleted from the REAL source text.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import types

import numpy
import pytest

try:
    import cupy
except ImportError:  # pragma: no cover - exercised on any NumPy-only host
    cupy = None

from .. import stepping
from ..dispersion import PolarizationState, Susceptibility
from ..fields import Fields
from ..grid import Grid
from ..pml import PML
from . import coverage
from . import no_pml_constitutive as null
from .no_pml_constitutive import covers_no_pml_null_constitutive as covers

HERE = pathlib.Path(__file__).parent
MODULE_PATH = HERE / "no_pml_constitutive.py"
STEPPING_PATH = HERE.parent / "stepping.py"

SIDES = ("H", "E")

#: How many full ``update_H``+``update_E`` cycles the identity property runs. One is
#: not enough: a sub-step that wrote back exactly what it read would pass a single
#: call and drift on the second, and a null that is only idempotent is not a null.
IDENTITY_CYCLES = 3

#: Every array a ``Fields`` may own that a constitutive sub-step could conceivably
#: write, read by name and skipped when unallocated -- WHICH OF THEM EXIST is itself
#: configuration-dependent (a no-PML run owns no H and no ``f_w``).
CANDIDATE_ARRAYS = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
)


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in.

    Carried even though THIS predicate has no backend clause, because the partition
    tests below call ``coverage.covers_real_pml_constitutive``, which does: on a
    NumPy grid it would refuse everything for the backend and the partition would be
    vacuously satisfied.
    """

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


_BACKENDS = [pytest.param(_NumpyWearingCupysName(), id="numpy-as-cupy")]
if cupy is not None:  # pragma: no cover - only on a CUDA host
    _BACKENDS.append(pytest.param(cupy, id="cupy"))


@pytest.fixture(params=_BACKENDS)
def xp(request):
    return request.param


# ---------------------------------------------------------------------------
# Fixtures: REAL Grid / Fields / PML objects
#
# The predicate asks a ``Fields`` and a ``PML`` two questions each. A dictionary
# would answer them too, and would answer them the way the test author expected
# rather than the way the engine does -- which is how ``stores_E`` and
# ``_pml_active`` came to be entangled in the first place (enable_pml_storage calls
# enable_field_storage, fields.py:700). So the objects are the engine's own, and the
# duck-typed stand-ins appear ONLY where they stand in for something the engine
# cannot build: an attribute that raises, a magnetic susceptibility.
# ---------------------------------------------------------------------------

def build(xp, *, pml_thickness=None, storage=False, pml_storage=False,
          cell=(8.0, 8.0, 8.0), **grid_kwargs):
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=("periodic",) * 3,
                xp=xp, **grid_kwargs)
    fields = Fields(grid=grid)
    if storage:
        fields.enable_field_storage()
    if pml_storage:
        fields.enable_pml_storage()
    layer = None if pml_thickness is None else PML(grid=grid, thickness=pml_thickness)
    return fields, layer, grid


class _Raising:
    """An object whose named property raises, delegating everything else.

    The two clauses this arm rests on are read through ``_flag`` for exactly this
    object: a ``Fields`` that cannot answer ``stores_E`` is not a covered run, and
    the sibling track measured what happens when it is treated as one.
    """

    def __init__(self, wrapped, attribute):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_attribute", attribute)

    def __getattr__(self, item):
        if item == object.__getattribute__(self, "_attribute"):
            raise RuntimeError(f"{item} is not answerable on this object")
        return getattr(object.__getattribute__(self, "_wrapped"), item)


class _MagneticState:
    """A duck-typed polarization that claims to drive an H component.

    The engine refuses magnetic susceptibilities at the driver, so this cannot be
    built from ``add_susceptibility``. It is the needle for the one H-side feature
    clause, and the clause is CONSERVATIVE: ``update_H`` returns at stepping.py:944
    whatever is registered, so the null is byte-exact here and refused anyway.
    """

    def drives(self, component):
        return component in ("Hx", "Hy", "Hz", "Bx", "By", "Bz")

    def driven(self):
        return ("Hx",)


class _MuteState:
    """A polarization with no ``drives()`` at all: unreadable, therefore refused."""


class _RaisingState:
    """A polarization whose ``drives()`` raises: a raise is not a refusal unless caught."""

    def drives(self, component):
        raise RuntimeError("this susceptibility cannot say what it drives")


def _electric_state(grid, sigma=0.35):
    return PolarizationState(Susceptibility(frequency=1.1, gamma=0.05),
                             grid.xp.full(grid.shape, numpy.float32(sigma)),
                             grid, numpy.float32)


# ---------------------------------------------------------------------------
# 1. THE VERDICT TABLE
#
# One entry per clause the predicate can state, per side. This is a
# SPECIFICATION, not a record of a previous implementation: the reason string is
# what a caller reads when a slot unexpectedly stays on the array path, and a
# refusal that fires for a different clause than the one intended is a refusal
# nobody can act on.
# ---------------------------------------------------------------------------

INERT_LAYER = "an inert PML object (thickness 0) rather than no layer at all"

_COVERED = (True, "covered")

_ACTIVE_LAYER_REFUSAL = (
    "an active PML layer is installed: update_H/update_E run the dsigw "
    "accumulation (stepping.py:946-951, :1014-1018), which is not a no-op")
_UNREADABLE_LAYER_REFUSAL = (
    "the layer does not report is_active: stepping._pml_is_active "
    "(stepping.py:2498-2506) branches on exactly that attribute")
_STORED_E_REFUSAL = (
    "fields.stores_E is True: update_E writes E[...] = (D - sum P) * inv_eps "
    "(stepping.py:1022) instead of returning at :983 -- that is the STORE arm, "
    "which this package has not built (STORED_E_ARM_STATUS)")
_UNREADABLE_STORES_E_REFUSAL = (
    "fields does not report stores_E: stepping.py:983 branches on exactly that "
    "attribute")
_PML_STORAGE_REFUSAL = (
    "Fields has PML storage enabled while the layer is inert: get_H would serve a "
    "stored H that update_H never writes (fields.py:695-702, :1189-1191) -- refused "
    "conservatively to match the curl families, not because the null would be wrong")
_NO_GRID_REFUSAL = "no grid: this is not a run"


def _cases(xp):
    """Every fixture the verdict table names, built fresh for one backend."""
    plain = build(xp)
    inert = build(xp, pml_thickness=0)
    active = build(xp, pml_thickness=2, storage=True, pml_storage=True)
    stored = build(xp, storage=True)
    pml_storage_inert = build(xp, pml_storage=True, pml_thickness=0)

    fields_pol, _, grid_pol = build(xp)
    fields_pol.polarizations.append(_electric_state(grid_pol))

    fields_mag, _, grid_mag = build(xp)
    fields_mag.polarizations.append(_MagneticState())

    fields_mute, _, grid_mute = build(xp)
    fields_mute.polarizations.append(_MuteState())

    fields_raise, _, grid_raise = build(xp)
    fields_raise.polarizations.append(_RaisingState())

    fields_chi2, _, grid_chi2 = build(xp)
    fields_chi2.set_nonlinear_volumes(
        {"Ex": xp.full(grid_chi2.shape, numpy.float32(0.2))}, {})

    fields_offdiag, _, grid_offdiag = build(xp)
    fields_offdiag.set_epsilon_volumes(
        {c: xp.full(grid_offdiag.shape, numpy.float32(2.25))
         for c in ("Ex", "Ey", "Ez")},
        {c: xp.full(grid_offdiag.shape, numpy.float32(1 / 2.25))
         for c in ("Ex", "Ey", "Ez")},
        {"Ex": {"Ey": xp.full(grid_offdiag.shape, numpy.float32(0.1))}})

    return {
        "plain": plain,
        "inert_layer": inert,
        "active_layer": active,
        # THE MUTATION BATTERY'S FIXTURE, and it is a different object on purpose.
        # ``active_layer`` also has PML storage switched on, so deleting the layer
        # clause from the source leaves clause 5 to refuse it -- the mutation would
        # be "caught" by a clause that is not the one under test, which measures
        # nothing about the clause that was deleted. This one trips the layer clause
        # and nothing else, on either side.
        "active_layer_bare": build(xp, pml_thickness=2),
        "unreadable_layer": (plain[0], _Raising(inert[1], "is_active"), plain[2]),
        "stored_e": stored,
        "unreadable_stores_e": (_Raising(plain[0], "stores_E"), None, plain[2]),
        "driven_polarization": (fields_pol, None, grid_pol),
        "magnetic_susceptibility": (fields_mag, None, grid_mag),
        "mute_polarization": (fields_mute, None, grid_mute),
        "raising_polarization": (fields_raise, None, grid_raise),
        "nonlinearity": (fields_chi2, None, grid_chi2),
        "offdiagonal": (fields_offdiag, None, grid_offdiag),
        "pml_storage_behind_inert": pml_storage_inert,
        "no_grid": (plain[0], None, None),
    }


_GOLDEN = {
    "H": {
        "plain": _COVERED,
        "inert_layer": _COVERED,
        "active_layer": (False, _ACTIVE_LAYER_REFUSAL),
        "active_layer_bare": (False, _ACTIVE_LAYER_REFUSAL),
        "unreadable_layer": (False, _UNREADABLE_LAYER_REFUSAL),
        # H IS NEVER STORED WITHOUT PML (fields.py:664-665), so every clause that
        # exists because update_E writes E is silent on this side. That is the
        # per-sub-step split doing its job, and it is measured below: the array path
        # moves 0 of 4608 words on the stored_e fixture's H side.
        "stored_e": _COVERED,
        "unreadable_stores_e": _COVERED,
        "driven_polarization": _COVERED,
        "magnetic_susceptibility": (
            False, "polarization 0 drives magnetic component Hx: the H-side "
                   "polarization slot is empty in this engine "
                   "(stepping.py:1406-1407) and a null cannot report coverage of a "
                   "sub-step that grew a term"),
        "mute_polarization": (
            False, "polarization 0 (_MuteState) does not report drives(); an "
                   "unreadable susceptibility is not a covered one"),
        "raising_polarization": (
            False, "polarization 0 (_RaisingState) does not report drives(); an "
                   "unreadable susceptibility is not a covered one"),
        "nonlinearity": _COVERED,
        "offdiagonal": _COVERED,
        "pml_storage_behind_inert": (False, _PML_STORAGE_REFUSAL),
        "no_grid": (False, _NO_GRID_REFUSAL),
    },
    "E": {
        "plain": _COVERED,
        "inert_layer": _COVERED,
        "active_layer": (False, _ACTIVE_LAYER_REFUSAL),
        "active_layer_bare": (False, _ACTIVE_LAYER_REFUSAL),
        "unreadable_layer": (False, _UNREADABLE_LAYER_REFUSAL),
        "stored_e": (False, _STORED_E_REFUSAL),
        "unreadable_stores_e": (False, _UNREADABLE_STORES_E_REFUSAL),
        "driven_polarization": (
            False, "polarization 0 drives Ex: update_E's source becomes (D - sum P) "
                   "and storage is switched on (driver.py:1543-1549)"),
        # A magnetic state answers drives('Ex') False, so the E side admits it. The
        # H side is where it is refused, and that asymmetry is the point of asking
        # per side rather than per run.
        "magnetic_susceptibility": _COVERED,
        "mute_polarization": (
            False, "polarization 0 (_MuteState) does not report drives(); an "
                   "unreadable susceptibility is not a covered one"),
        "raising_polarization": (
            False, "polarization 0 (_RaisingState) does not report drives(); an "
                   "unreadable susceptibility is not a covered one"),
        "nonlinearity": (
            False, "instantaneous chi2/chi3 is installed: update_E's product for "
                   "those components is the Pade factor (stepping.py:999-1000). "
                   "CONSERVATIVE where it fires alone -- driver.set_chi2/set_chi3 "
                   "force stored E (driver.py:2023-2029) so a driver-built run has "
                   "already been refused above, but Fields.set_nonlinear_volumes "
                   "does not, and on that object update_E returns at "
                   "stepping.py:983 and the null is byte-exact"),
        # SHADOWED, and the shadow is what the table records: set_epsilon_volumes
        # switched storage on, so the stores_E clause answers first. The off-diagonal
        # refusal is unreachable from a real Fields and is pinned on a duck-typed one
        # by test_the_offdiagonal_clause_fires_when_the_shadow_is_removed.
        "offdiagonal": (False, _STORED_E_REFUSAL),
        "pml_storage_behind_inert": (False, _STORED_E_REFUSAL),
        "no_grid": (False, _NO_GRID_REFUSAL),
    },
}


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("case", sorted(_GOLDEN["H"]))
def test_every_clause_by_its_own_refusal_string(xp, side, case):
    fields, layer, grid = _cases(xp)[case]
    assert covers(fields, layer, grid, side) == _GOLDEN[side][case]


#: The two refusals a REAL ``Fields`` cannot reach, because ``set_epsilon_volumes``
#: switches storage on first. They are pinned by their own tests below on duck-typed
#: objects; they are named here so that
#: :func:`test_the_table_covers_every_clause_the_source_can_state` can account for
#: every ``return False`` in the shipped source without a reader keeping a list.
_OFFDIAG_REFUSAL = (
    "an off-diagonal chi1inv row is installed: the row product reads the other "
    "components' volumes (stepping.py:1001-1008) and belongs to "
    "covers_real_pml_offdiag_constitutive")
_UNREADABLE_OFFDIAG_REFUSAL = (
    "fields does not report has_offdiagonal_epsilon; a run that cannot say whether "
    "update_E forms a row product is not a covered one")

#: How many ``return False`` sites the predicate has. A TRIPWIRE, not a statistic: a
#: clause added without a row in :data:`_GOLDEN` (or a dedicated test) fails here, in
#: a first-refusal predicate where a new clause can silently take precedence over
#: every clause below it.
_REFUSAL_SITES = 13


def test_the_table_covers_every_clause_the_source_can_state():
    """No refusal the predicate can emit may go unpinned.

    Counted from the source text rather than from a list a reader maintains. The
    accounting is: 13 sites in the source; 11 distinct reason strings pinned by the
    verdict table (two sites emit a per-component / per-index format string that the
    table reaches with more than one fixture, and two sites are unreachable from a
    real ``Fields``); the last two are the off-diagonal pair, named above and pinned
    on duck-typed objects by their own tests.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    body = source[source.index("def covers_no_pml_null_constitutive"):]
    assert len(re.findall(r"return False, ", body)) == _REFUSAL_SITES
    stated = {reason for table in _GOLDEN.values()
              for covered, reason in table.values() if not covered}
    assert len(stated) == 11
    for unreachable in (_OFFDIAG_REFUSAL, _UNREADABLE_OFFDIAG_REFUSAL):
        assert unreachable not in stated
        # ...and is really the text the predicate emits, not a paraphrase of it.
        assert unreachable.split(":")[0].split(";")[0] in " ".join(body.split())


def test_an_unknown_side_raises_rather_than_refusing(xp):
    fields, layer, grid = build(xp)
    with pytest.raises(ValueError, match="side must be one of"):
        covers(fields, layer, grid, "B")


def test_the_grid_is_read_for_its_presence_and_nothing_else(xp):
    """Every OTHER predicate here reads the grid's backend, shape and boundaries.

    This one reads none of them, and that is the family's distinguishing claim
    rather than an oversight -- so it is measured on a grid that would raise if any
    of them were touched, not argued from the absence of the lines.
    """

    class _Landmine:
        def __getattr__(self, item):
            raise AssertionError(f"the null predicate read grid.{item}")

    fields, layer, _ = build(xp)
    for side in SIDES:
        assert covers(fields, layer, _Landmine(), side) == _COVERED


def test_a_fields_that_cannot_answer_anything_splits_by_side(xp):
    """H admits it, E refuses it, and BOTH match what ``stepping`` does to it.

    ``update_H`` returns before it touches ``fields`` at all, so a bare object is a
    run on which it genuinely does nothing; ``update_E`` reads ``stores_E`` at
    stepping.py:983 and raises. Admitting the first and refusing the second is not an
    inconsistency -- it is the predicate agreeing with the array path twice.
    """
    _, _, grid = build(xp)
    assert covers(object(), None, grid, "H") == _COVERED
    assert covers(object(), None, grid, "E") == (False, _UNREADABLE_STORES_E_REFUSAL)
    stepping.update_H(object(), None)  # returns; nothing to assert but the absence
    with pytest.raises(AttributeError):
        stepping.update_E(object(), None)


def test_the_offdiagonal_clause_fires_when_the_shadow_is_removed(xp):
    """The clause is SHADOWED on a real ``Fields``, not dead. Pinned on a stand-in.

    ``Fields.set_epsilon_volumes`` switches storage on for a surviving row
    (fields.py:1253-1255), so ``stores_E`` refuses first and this refusal is
    unreachable from the engine's own constructor today. It stays in the predicate
    because the day that coupling changes, a first-refusal chain that had deleted it
    would ADMIT the row product -- and :data:`CLAUSE_EVIDENCE` records that the array
    path moves 1450 words on the real route, so the null would be wrong rather than
    merely unbuilt-for.
    """

    class _OffdiagonalNotStored:
        stores_E = False
        polarizations = ()
        has_offdiagonal_epsilon = True

    _, _, grid = build(xp)
    covered, reason = covers(_OffdiagonalNotStored(), None, grid, "E")
    assert (covered, reason) == (False, _OFFDIAG_REFUSAL)


def test_an_unreadable_offdiagonal_flag_is_refused_by_name(xp):
    class _Mute:
        stores_E = False
        polarizations = ()
        has_offdiagonal_epsilon = None

    _, _, grid = build(xp)
    assert covers(_Mute(), None, grid, "E") == (False, _UNREADABLE_OFFDIAG_REFUSAL)


def test_chi3_alone_is_refused_although_has_nonlinearity_would_say_no(xp):
    """The one place this predicate is deliberately WIDER than its Triton sibling.

    ``Fields.has_nonlinearity`` reads ``_chi2_components`` ALONE (fields.py:966-967),
    so a chi3-only object answers False through the property. Reading both maps by
    name is what ``covers_real_pml_constitutive`` already does, and the assertion
    below is on the property, not on a belief about it.
    """

    class _Chi3Only:
        stores_E = False
        polarizations = ()
        has_offdiagonal_epsilon = False
        _chi2_components = ()
        _chi3_components = ("Ez",)

        @property
        def has_nonlinearity(self):
            return bool(self._chi2_components)

    _, _, grid = build(xp)
    subject = _Chi3Only()
    assert subject.has_nonlinearity is False
    covered, reason = covers(subject, None, grid, "E")
    assert covered is False
    assert reason.startswith("instantaneous chi2/chi3 is installed")


# ---------------------------------------------------------------------------
# 2. TRANSCRIPTION PINS
# ---------------------------------------------------------------------------

def test_the_side_table_is_keyed_as_the_kernel_families_are():
    assert set(null.NULL_CONSTITUTIVE_SIDES) == set(coverage.CONSTITUTIVE_SIDES)


def test_the_component_tuples_match_the_kernel_families_targets():
    assert null.NULL_ELECTRIC_COMPONENTS == coverage.CONSTITUTIVE_SIDES["E"]["targets"]
    assert null.NULL_MAGNETIC_COMPONENTS == coverage.CONSTITUTIVE_SIDES["H"]["targets"]


def test_the_named_sub_steps_are_the_real_stepping_functions():
    for side, spec in null.NULL_CONSTITUTIVE_SIDES.items():
        assert callable(getattr(stepping, spec["sub_step"]))
    assert null.NULL_CONSTITUTIVE_SIDES["H"]["sub_step"] == "update_H"
    assert null.NULL_CONSTITUTIVE_SIDES["E"]["sub_step"] == "update_E"


def test_every_cited_return_site_is_read_back_out_of_stepping():
    """The validate-against-the-reference rule, made checkable.

    The two ``return`` statements this whole family rests on are cited by line. A
    citation is only worth something if it still points at the statement it names,
    and that is a thing a test can read rather than a thing a reviewer has to.
    """
    lines = STEPPING_PATH.read_text(encoding="utf-8").splitlines()
    for side, spec in null.NULL_CONSTITUTIVE_SIDES.items():
        first, last = spec["return_lines"]
        assert spec["return_site"] == f"stepping.py:{first}-{last}"
        guard = lines[first - 1].strip()
        statement = lines[last - 1].strip()
        assert guard.startswith("if not"), (side, guard)
        assert statement.startswith("return"), (side, statement)
    # And each guard is the sub-step's OWN, not the other's.
    assert "_pml_is_active(pml)" in lines[943]
    assert "not pml_active and not fields.stores_E" in lines[982]


def test_the_inactive_layer_test_is_the_one_stepping_branches_on():
    """``_pml_is_active`` is ``pml is not None and pml.is_active`` -- read, not recalled.

    This is what makes the arm's admitted set the exact complement of the kernel
    predicates' rather than an approximation: both ask the same attribute.
    """
    source = STEPPING_PATH.read_text(encoding="utf-8")
    body = source[source.index("def _pml_is_active"):]
    body = body[:body.index("\ndef ")]
    assert "return pml is not None and pml.is_active" in body


def test_an_all_zero_face_layer_is_inactive_and_therefore_ours(xp):
    """``PML(thickness=0)`` is the case pml.py:427-434 exists to name.

    A layer object present in the driver but absorbing nowhere takes the NO-PML path
    in ``stepping``, so it belongs to this arm and not to the kernels'. Both halves
    are asserted: the layer really is inert, and both predicates agree about who
    owns it.
    """
    fields, layer, grid = build(xp, pml_thickness=0, storage=True)
    assert layer is not None and layer.is_active is False
    assert covers(fields, layer, grid, "H") == _COVERED
    assert coverage.covers_real_pml_constitutive(fields, layer, grid, "H") == (
        False, "no active PML layer")


# ---------------------------------------------------------------------------
# 3. THE IDENTITY PROPERTY, with its failing control
# ---------------------------------------------------------------------------

def _words(array):
    host = numpy.asarray(getattr(array, "get", lambda: array)()
                         if cupy is not None and isinstance(array, getattr(cupy, "ndarray", ()))
                         else array)
    return numpy.ascontiguousarray(host).view(numpy.uint32).ravel().copy()


def _snapshot(fields):
    out = {}
    for name in CANDIDATE_ARRAYS:
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = _words(array)
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for label in ("P", "P_prev"):
            table = getattr(state, label, None) or {}
            for component, array in table.items():
                if array is not None:
                    out[f"pol{index}.{label}[{component}]"] = _words(array)
    return out


def _moved(before, after):
    total = 0
    for name in sorted(set(before) | set(after)):
        if name not in before or name not in after:
            total += 1
            continue
        total += int(numpy.count_nonzero(before[name] != after[name]))
    return total


def _seed(fields, salt=0):
    """Uniform values, a plane of NEGATIVE ZEROS, and dense subnormals.

    The signed-zero plane is the class a random seed provably cannot reach
    (``rng.uniform`` never returns one) and the exact class that separates a plain
    store from an accumulation; the subnormals are the class a flush-to-zero policy
    would move. Neither is relevant to a sub-step that performs no operation --
    which is precisely the claim, so they are seeded rather than assumed away.
    """
    rng = numpy.random.default_rng(20260820 + salt)
    xp = fields.grid.xp
    shape = tuple(fields.grid.shape)
    size = int(numpy.prod(shape))
    arrays = [getattr(fields, n, None) for n in CANDIDATE_ARRAYS]
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for label in ("P", "P_prev"):
            arrays.extend((getattr(state, label, None) or {}).values())
    for array in arrays:
        if array is None:
            continue
        host = rng.uniform(-1.0, 1.0, size=size).astype(numpy.float32)
        host[3::11] = numpy.float32(-0.0)
        host[5::23] = numpy.float32(1e-45)
        host[7::29] = numpy.float32(-3e-44)
        array[...] = xp.asarray(numpy.ascontiguousarray(host.reshape(shape)))


def _census(fields):
    nonzero = signed_zero = subnormal = 0
    for name in CANDIDATE_ARRAYS:
        array = getattr(fields, name, None)
        if array is None:
            continue
        w = _words(array)
        nonzero += int(numpy.count_nonzero(w != numpy.uint32(0)))
        signed_zero += int(numpy.count_nonzero(w == numpy.uint32(0x80000000)))
        exponent = (w >> numpy.uint32(23)) & numpy.uint32(0xFF)
        mantissa = w & numpy.uint32(0x7FFFFF)
        subnormal += int(numpy.count_nonzero((exponent == 0) & (mantissa != 0)))
    return nonzero, signed_zero, subnormal


def _cycles(fields, layer, cycles=IDENTITY_CYCLES):
    """Run N full ``update_H``+``update_E`` cycles, comparing against the ORIGINAL."""
    before = _snapshot(fields)
    worst = 0
    for _ in range(cycles):
        stepping.update_H(fields, layer)
        stepping.update_E(fields, layer)
        worst = max(worst, _moved(before, _snapshot(fields)))
    return worst


ADMITTED_ON_BOTH_SIDES = ("plain", "inert_layer")


@pytest.mark.parametrize("case", ADMITTED_ON_BOTH_SIDES)
def test_an_admitted_configuration_moves_no_word_over_several_cycles(xp, case):
    fields, layer, grid = _cases(xp)[case]
    _seed(fields)
    nonzero, signed_zero, subnormal = _census(fields)
    assert (nonzero, bool(signed_zero), bool(subnormal)) > (0, False, False), (
        "VACUOUS: a state with no nonzero, no signed-zero and no subnormal word "
        "satisfies 'nothing moved' trivially")
    assert covers(fields, layer, grid, "H") == _COVERED
    assert covers(fields, layer, grid, "E") == _COVERED
    assert _cycles(fields, layer) == 0


def test_the_control_that_makes_that_a_measurement(xp):
    """The same assertion where the predicate REFUSES: the array path must move.

    An identity leg with no failing control measures nothing -- it is satisfied just
    as well by a fixture in which nothing could have moved.
    """
    fields, layer, grid = _cases(xp)["active_layer"]
    _seed(fields)
    for side in SIDES:
        assert covers(fields, layer, grid, side)[0] is False
    assert _cycles(fields, layer) > 0

    stored_fields, stored_layer, stored_grid = _cases(xp)["stored_e"]
    _seed(stored_fields)
    assert covers(stored_fields, stored_layer, stored_grid, "E")[0] is False
    assert covers(stored_fields, stored_layer, stored_grid, "H") == _COVERED
    before = _snapshot(stored_fields)
    stepping.update_H(stored_fields, stored_layer)
    assert _moved(before, _snapshot(stored_fields)) == 0   # H is never stored
    stepping.update_E(stored_fields, stored_layer)
    assert _moved(before, _snapshot(stored_fields)) > 0    # E is


# ---------------------------------------------------------------------------
# 4. THE EVIDENCE TABLE, RE-MEASURED
# ---------------------------------------------------------------------------

_EVIDENCE_NEEDLES = {
    "active_layer": ("active_layer", ("H", "E")),
    "stored_e": ("stored_e", ("E",)),
    "driven_polarization": ("driven_polarization", ("E",)),
    "nonlinearity": ("nonlinearity", ("E",)),
    "offdiagonal": ("offdiagonal", ("E",)),
    "magnetic_susceptibility": ("magnetic_susceptibility", ("H",)),
    "pml_storage_behind_an_inert_layer": ("pml_storage_behind_inert", ("H", "E")),
    "unreadable_polarization": ("mute_polarization", ("H", "E")),
    # The grid is not an argument ``stepping`` takes, so the needle for the
    # ``no_grid`` clause is the ordinary covered fixture: the clause refuses a run
    # the array path would have been a no-op on, which is what makes it conservative.
    "no_grid": ("plain", ("H", "E")),
}


def test_every_numeric_evidence_row_has_a_needle():
    """A row with a number in it must be a row this file re-measures.

    Without this, a clause could be added to :data:`CLAUSE_EVIDENCE` with a
    plausible ``moved_words`` that nothing ever recomputes -- which is precisely the
    failure mode the table exists to prevent, reintroduced one level up.
    """
    numeric = {name for name, entry in null.CLAUSE_EVIDENCE.items()
               if any(isinstance(v, int) for v in entry["moved_words"].values())}
    assert numeric == set(_EVIDENCE_NEEDLES), numeric ^ set(_EVIDENCE_NEEDLES)


@pytest.mark.parametrize("clause", sorted(_EVIDENCE_NEEDLES))
def test_the_recorded_word_counts_are_re_measured(clause):
    """Recompute every number :data:`CLAUSE_EVIDENCE` records, on its own needle.

    The table's adjectives are derived from these numbers, so a row whose
    measurement stops agreeing has to fail HERE rather than quietly reclassify a
    clause from load-bearing to conservative in a comment. Measured on NumPy only:
    the counts are a property of ``stepping``'s control flow, not of the backend,
    and the device gate re-measures them on CuPy.
    """
    case, sides = _EVIDENCE_NEEDLES[clause]
    entry = null.CLAUSE_EVIDENCE[clause]
    for side in sides:
        fields, layer, _ = _cases(_NumpyWearingCupysName())[case]
        _seed(fields)
        before = _snapshot(fields)
        (stepping.update_H if side == "H" else stepping.update_E)(fields, layer)
        moved = _moved(before, _snapshot(fields))
        assert moved == entry["moved_words"][side], (clause, side, moved)
        if "total_words" in entry:
            assert sum(int(v.size) for v in before.values()) == entry["total_words"][side]


def test_the_roles_follow_from_the_numbers_rather_than_the_prose():
    for name, entry in null.CLAUSE_EVIDENCE.items():
        moved = entry["moved_words"]
        numeric = [v for v in moved.values() if isinstance(v, int)]
        if entry["role"] == "load_bearing" and numeric:
            assert any(v > 0 for v in numeric), name
        if entry["role"] == "conservative":
            assert all(v == 0 for v in numeric), name
        assert set(entry["sides"]) <= set(SIDES)
        assert set(moved) <= set(SIDES)


def test_the_two_unreadable_clauses_really_do_face_a_raising_array_path(xp):
    """``role: load_bearing`` with ``moved_words: "array path raises"`` -- measured.

    Both entries claim the array path cannot even run on their needle. That is a
    stronger claim than "it moves words" and it is the reason the two flags are read
    through ``_flag`` rather than through ``getattr(..., False)``.
    """
    fields, inert, grid = build(xp, pml_thickness=0)
    with pytest.raises(RuntimeError):
        stepping.update_H(fields, _Raising(inert, "is_active"))
    with pytest.raises(RuntimeError):
        stepping.update_E(_Raising(fields, "stores_E"), None)


# ---------------------------------------------------------------------------
# 5. THE PARTITION, and the mutations
# ---------------------------------------------------------------------------

_SIBLING_CONSTITUTIVE = {
    "covers_real_pml_constitutive":
        lambda f, p, g, side: coverage.covers_real_pml_constitutive(f, p, g, side),
    "covers_real_pml_offdiag_constitutive":
        lambda f, p, g, side: (coverage.covers_real_pml_offdiag_constitutive(f, p, g)
                               if side == "E" else (False, "H side has no offdiag kernel")),
    "covers_real_pml_complex_constitutive":
        lambda f, p, g, side: coverage.covers_real_pml_complex_constitutive(
            f, p, g, side, "NAIVE"),
}


def _sibling_verdict(sibling, fields, layer, grid, side):
    """A sibling predicate's ``covered``, with a RAISE recorded as "did not admit".

    THE RAISE IS A REAL MEASUREMENT AND IT IS NOT SWALLOWED HERE -- it is pinned by
    :func:`test_a_sibling_predicate_raises_where_this_one_refuses`, which is where
    the finding lives. Collapsing it to False for the partition arithmetic is sound
    (a predicate that raised certainly did not admit the slot) and keeps the
    partition question separate from the fail-closed-ness question.
    """
    try:
        return bool(_SIBLING_CONSTITUTIVE[sibling](fields, layer, grid, side)[0])
    except Exception:  # noqa: BLE001 - measured, and pinned by its own test
        return False


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("case", sorted(_GOLDEN["H"]))
@pytest.mark.parametrize("sibling", sorted(_SIBLING_CONSTITUTIVE))
def test_no_slot_is_ever_claimed_twice(xp, side, case, sibling):
    """The partition, measured on every fixture rather than argued from construction.

    Two predicates admitting one slot is a fail-closed composer's worst input: it
    has to pick, and any rule for picking is a rule nobody measured. It cannot
    happen here because both sides ask ``pml.is_active`` and this arm requires the
    answer the others refuse -- and that is exactly the kind of "cannot happen" this
    package measures instead of asserting.
    """
    fields, layer, grid = _cases(xp)[case]
    mine = covers(fields, layer, grid, side)[0]
    theirs = _sibling_verdict(sibling, fields, layer, grid, side)
    assert not (mine and theirs), (case, side, sibling)


def test_a_sibling_predicate_raises_where_this_one_refuses(xp):
    """A MEASURED FINDING ABOUT ``coverage.py``, pinned here rather than fixed here.

    ``covers_real_pml_constitutive`` reads the layer as
    ``getattr(pml, "is_active", False)``. ``getattr`` with a
    default swallows a MISSING attribute, not a raising one -- so on a layer object
    whose ``is_active`` property raises, that predicate PROPAGATES the exception
    instead of refusing. Its own sibling in the same file already treats this as a
    defect class and catches it by name ("A RAISE IS NOT A REFUSAL unless it is
    caught here", ``covers_real_pml_ade_component``), and a
    predicate a planner consults turns an escaping exception into a crashed run
    where a fail-closed "no" was the correct answer.

    This arm reads the same attribute through ``_flag`` and refuses by name. The
    asymmetry is measured here so that it is a recorded fact rather than an
    impression, and so that hardening ``coverage.py`` later FAILS this test and
    forces the record to be updated in the same change.

    Not fixed in this round: ``coverage.py``'s digest is under a live census
    manifest.
    """
    fields, inert, grid = build(xp, pml_thickness=0, storage=True)
    unreadable = _Raising(inert, "is_active")
    assert covers(fields, unreadable, grid, "H") == (False, _UNREADABLE_LAYER_REFUSAL)
    with pytest.raises(RuntimeError):
        coverage.covers_real_pml_constitutive(fields, unreadable, grid, "H")


def test_the_partition_is_exhaustive_on_the_layer_question(xp):
    """Not merely disjoint: between them the two real-storage arms answer every layer.

    Disjointness alone would be satisfied by two predicates that both refuse
    everything. What makes the pair a PARTITION of the layer clause is that on a run
    the kernels refuse ONLY for the layer, this arm admits -- so no real-storage
    no-PML slot is left with nobody able to say what happens there.
    """
    fields, layer, grid = build(xp, cell=(8.0, 8.0, 8.0))
    for side in SIDES:
        kernel = coverage.covers_real_pml_constitutive(fields, layer, grid, side)
        assert kernel == (False, "no active PML layer")
        assert covers(fields, layer, grid, side) == _COVERED


def _mutant(text_edit, label):
    """Compile a TEXT-EDITED copy of the shipped module into a fresh module object.

    Importing the shipped predicate under a mutant's name is the kernel-less version
    of a stale binary served from a compile cache, and it is closed the same way: the
    mutant is compiled from characters that differ, and the test asserts the edit
    actually applied before it believes the verdict.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    mutated, count = text_edit(source)
    assert count == 1, f"{label}: the mutation matched {count} sites, expected 1"
    spec = importlib.util.spec_from_loader(f"_mutant_{label}", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(MODULE_PATH)
    exec(compile(mutated, f"<mutant {label}>", "exec"), module.__dict__)
    return module


def _drop(marker, replacement=""):
    def edit(source):
        if source.count(marker) != 1:
            return source, source.count(marker)
        return source.replace(marker, replacement), 1
    return edit


_MUTATIONS = {
    # The load-bearing pair. Each of these, uncaught, is silent corruption: a
    # sub-step the composer skips while the array path had 2715 or 1287 words of
    # work to do.
    "admit_an_active_layer": (
        _drop("        if active:\n", "        if False:\n"),
        "active_layer_bare", ("H", "E")),
    "admit_a_stored_e_run": (
        _drop("        if stored:\n", "        if False:\n"), "stored_e", ("E",)),
    "default_an_unreadable_flag_to_admit": (
        _drop("        return _UNREADABLE\n    if value is None:",
              "        return False\n    if value is None:"),
        "unreadable_layer", ("H", "E")),
    # The conservative clauses. Uncaught, each is a coverage claim over a
    # configuration nobody built for -- byte-exact today and unguarded tomorrow.
    "admit_a_magnetic_susceptibility": (
        _drop("                if driven:\n                    return False, (\n"
              "                        f\"polarization {index} drives magnetic component \"",
              "                if False:\n                    return False, (\n"
              "                        f\"polarization {index} drives magnetic component \""),
        "magnetic_susceptibility", ("H",)),
    "admit_pml_storage_behind_an_inert_layer": (
        _drop('    if bool(getattr(fields, "_pml_active", False)):',
              "    if False:"),
        "pml_storage_behind_inert", ("H",)),
    "admit_a_nonlinearity": (
        _drop('        if (getattr(fields, "_chi2_components", None)\n'
              '                or getattr(fields, "_chi3_components", None)):',
              "        if False:"),
        "nonlinearity", ("E",)),
    "read_the_property_instead_of_both_maps": (
        _drop('        if (getattr(fields, "_chi2_components", None)\n'
              '                or getattr(fields, "_chi3_components", None)):',
              '        if _flag(fields, "has_nonlinearity") is True:'),
        None, None),
}


@pytest.mark.parametrize("label", sorted(k for k in _MUTATIONS if _MUTATIONS[k][1]))
def test_every_mutation_is_caught(label):
    """Delete a clause from the REAL source and require the verdict to change.

    A clause nobody can show changing a verdict is a clause nobody is testing, and in
    a first-refusal predicate it is also a clause that may be permanently shadowed
    without anyone noticing.
    """
    edit, case, sides = _MUTATIONS[label]
    module = _mutant(edit, label)
    xp = _NumpyWearingCupysName()
    for side in sides:
        fields, layer, grid = _cases(xp)[case]
        shipped = covers(fields, layer, grid, side)
        mutated = module.covers_no_pml_null_constitutive(fields, layer, grid, side)
        assert shipped[0] is False, (label, side, "the shipped verdict was not a refusal")
        assert mutated[0] is True, (label, side, "the mutation changed nothing")


def test_the_chi3_only_mutation_is_caught_by_the_property_it_replaces():
    """The one mutation that is a WIDENING rather than a deletion.

    Reading ``has_nonlinearity`` instead of the two maps is the sibling track's
    spelling, and it is not wrong for a null -- so it cannot be caught on a real
    ``Fields``. It is caught on the object that separates them, which is the only
    object on which the two spellings disagree.
    """
    module = _mutant(_MUTATIONS["read_the_property_instead_of_both_maps"][0],
                     "chi3_property")

    class _Chi3Only:
        stores_E = False
        polarizations = ()
        has_offdiagonal_epsilon = False
        has_nonlinearity = False
        _chi2_components = ()
        _chi3_components = ("Ez",)

    _, _, grid = build(_NumpyWearingCupysName())
    subject = _Chi3Only()
    assert covers(subject, None, grid, "E")[0] is False
    assert module.covers_no_pml_null_constitutive(subject, None, grid, "E")[0] is True


# ---------------------------------------------------------------------------
# 6. THE RECORD
# ---------------------------------------------------------------------------

def test_the_stored_e_arm_is_recorded_as_unbuilt_here():
    """The refusal and the record must agree about who owns ``stores_E`` True."""
    assert null.STORED_E_ARM_STATUS["built_for_cuda"] is False
    assert null.STORED_E_ARM_STATUS["refused_here_as"] == "fields.stores_E is True"
    xp = _NumpyWearingCupysName()
    fields, layer, grid = _cases(xp)["stored_e"]
    assert covers(fields, layer, grid, "E")[1].startswith(
        null.STORED_E_ARM_STATUS["refused_here_as"])


def test_the_reach_record_adds_up_and_says_what_it_is():
    """The slot delta is arithmetic a reader can check, and a BOUND a reader is told about.

    Two things go wrong with a coverage number in this package and both are checkable
    here: a per-side split that does not sum to the headline (which is how a number
    survives a change to one side), and a bound quietly reported as a measurement.
    The census itself is re-derivable by
    ``parity/meep_gpu/replay_null_constitutive_slots.py``; what this pins is that the
    record cannot drift internally.
    """
    reach = null.NULL_CONSTITUTIVE_REACH
    assert reach["by_side"]["H"] + reach["by_side"]["E"] == reach["slots_this_arm_adds"]
    assert reach["slots_claimed_twice"] == 0
    # Every no-PML row gains update_H; only the unstored ones gain update_E.
    assert reach["by_side"]["H"] == reach["no_pml_rows"]
    assert reach["by_side"]["E"] == reach["no_pml_rows"] - reach["no_pml_rows_with_stored_e"]
    assert reach["rows_taken_at_both_constitutive_sub_steps"] == reach["by_side"]["E"]
    # A SUB-STEP GAINED IS NOT A RUN TAKEN OVER, and the record says so in a field
    # rather than in a sentence somebody has to read.
    assert reach["rows_taken_at_every_sub_step"] == 0
    assert reach["is_an_upper_bound"] is True
    assert reach["slots_this_arm_adds"] < reach["constitutive_slots_in_the_record"]


def test_confirmation_record_is_all_or_nothing():
    """A half-filled device record is worse than an empty one: it reads as current.

    Either the gate has not run -- in which case every measured field is None and a
    reader can see that at a glance -- or it has, and then the artifact directory it
    names has to exist. There is no third state in which some numbers are from a
    device and the rest are from an intention.
    """
    record = null.NULL_CONSTITUTIVE_CONFIRMATION
    measured = ("artifacts", "recorded_utc", "host", "device", "gpu_index",
                "identity_cases", "identity_bit_identical", "breadth_cases",
                "control_cases", "controls_that_moved", "mutation_legs",
                "engine_steps")
    filled = [name for name in measured if record[name] is not None]
    assert filled == [] or len(filled) == len(measured), (
        f"partially filled device record: {filled}")
    if not filled:
        return
    # THE ARTIFACT DIRECTORY IS NOT ASSERTED TO EXIST, and that is deliberate rather
    # than lax: ``parity/meep_gpu/results/`` is gitignored, so a test that required
    # the directory would pass only on the machine that cut it and fail on every
    # clone. The RECORD is what ships; the artifact is what the record was read out
    # of. What is checked here is that the record names a path in the right tree and
    # that its numbers cannot disagree with each other.
    assert record["artifacts"].startswith("parity/meep_gpu/results/")
    assert record["identity_bit_identical"] == record["identity_cases"]
    assert record["controls_that_moved"] == record["control_cases"]
    assert record["differing_words_per_policy"] == 0
    assert record["words_compared_per_policy"] > 0
    assert record["slots_claimed_twice"] == 0
    # A run that certifies "nothing moved" must also prove something COULD have:
    # the driven and source-free engine arms are required to differ.
    assert record["engine_driven_vs_source_free_differing_words"] > 0
    assert set(record["policies_released"]) == {"keep", "flush"}
    # The gpu index this stream owns, recorded so a run on another one is visible.
    assert isinstance(record["gpu_index"], int)


def test_the_device_record_and_the_reach_record_describe_the_same_arm():
    """The two records are independent measurements and must not contradict.

    The reach replay says the arm serves 14 no-PML rows; the device record says the
    breadth leg admitted 8 configurations and the identity leg 7. Neither number
    constrains the other -- but both are claims about an arm that takes NO row whole
    step, and a record that quietly acquired a whole-step claim on one side while the
    other still denied it is the kind of drift that outlives the round it started in.
    """
    assert null.NULL_CONSTITUTIVE_REACH["rows_taken_at_every_sub_step"] == 0
    assert "0 of the 14 rows" in null.NULL_CONSTITUTIVE_CONFIRMATION["limits"]
