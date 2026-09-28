"""The real-field PML curl slice: its coverage predicate, and the mutations that pin it.

THE BIT-IDENTITY GATE LIVES ELSEWHERE — it needs a device and a sweep, so it is
``parity/meep_gpu/probe_fused_kernel_bit_identity.py`` (``--experiments
pml,multistep``) with ``run_pml_gate_mutations.py`` driving its mutation legs.
What is pinned HERE is the other half of the slice's silent-failure surface: the
predicate that decides whether the kernel may run at all.

WHY THE PREDICATE NEEDS MUTATIONS OF ITS OWN. Every configuration it refuses
produces a plausible, smooth, WRONG field if it leaks — not a crash, not a NaN.
A predicate that no mutation exercises is indistinguishable from one that
returns True, and the five mutations below are the difference:

  * WIDEN IT BY ONE AXIS — let a mirrored grid through — and a configuration
    with an entirely different ghost rule becomes "covered".
  * DROP ONE COVERED-CONFIGURATION CHECK — the float32 dtype — and complex
    storage becomes "covered".
  * WIDEN THE CONDUCTIVITY LOOP back over both sub-steps' targets and the
    over-refusal this file's 2026-08-15 round removed comes straight back.
  * DROP ``stores_E`` and an E that ``update_E`` never wrote is "covered" —
    the invariant admitting dispersion rests on.
  * DROP THE INT32 BOUND and a volume the kernel's ``int idx`` cannot address
    is "covered". Two of the five leak only when a second clause goes with
    them, and each records which survivor caught it.

Every one is applied to the REAL function's source, not to a hand-written
stand-in, so they cannot drift away from what ships.

AND THE SAME ARGUMENT RUN BACKWARDS, because this file's round WIDENED two
fail-closed clauses rather than only narrowing coverage: the pre-widening
verdicts are carried as data and asserted ABSENT, so a predicate drifting back
to over-refusing fails a test instead of quietly costing seven corpus slots. The
two widenings are then measured on the ARRAY PATH itself — a live pole changes
neither curl's bits while changing ``update_E``'s, and a D conductivity changes
``step_D``'s and not ``step_B``'s — because a verdict table only says what the
predicate returns, never whether it describes the engine.

COVERAGE IS PER SUB-STEP HERE TOO. ``covers_real_pml_curl`` takes a required
``sub_step``, so every verdict below is measured at both ``step_B`` and
``step_D``. They are not always equal: a D conductivity refuses ``step_D`` and
admits ``step_B``, which is a configuration the pre-2026-08-15 predicate could
not express at all.

THIS FILE USED TO BE ONE SKIP. ``step_curl_kernels`` imports ``cupy`` at module
scope, and while the predicate lived beside the kernel strings the whole file
collapsed to a single sanctioned ``requires_resource('cupy')`` skip on any host
without a GPU — the one part of the track that is pure decision logic, pinned
only on the one machine that can launch a kernel. The predicate now lives in
``coverage.py``, which imports ``__future__`` and ``typing`` and nothing else, so
everything below runs at the merge bar. The three tests that genuinely need CuPy
— the two coefficient-table tests and the boundary-code refusal, all of which
reach through ``step_curl_kernels`` — are marked and skip on their own.

THE BACKEND STAND-IN, AND WHAT IT IS NOT. Off device the fixture hands ``Grid``
a module that is NumPy wearing CuPy's ``__name__``, because the predicate's
first question is whether the backend is CuPy at all. Everything the predicate
reads — dtype, shape, contiguity, the PML vectors, the grid's own resolution of
its boundary kinds — is exercised on REAL ``Grid``/``Fields``/``PML`` objects,
which is what catches an attribute rename. What it does not exercise is
CuPy-specific array behaviour; on a CUDA host the same tests run again against
real CuPy arrays, which is where that is covered.
"""

from __future__ import annotations

import pathlib
import re
import types

import numpy
import pytest

try:
    import cupy
except ImportError:  # pragma: no cover - exercised on any NumPy-only host
    cupy = None

from ..device_identity import weld_survives_edit
from ..fields import Fields
from ..grid import Grid
from ..pml import PML
from . import coverage


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__``.

    The predicate refuses any backend whose module is not named "cupy", which is
    the one thing about the real device library that cannot be reproduced off
    device — and the only thing this stands in for. Every array it hands back is
    a real NumPy array with a real dtype, shape, stride set and flags.
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
    """The array module the grid is built on, named "cupy" either way."""
    return request.param


def build(xp, boundaries=("periodic", "periodic", "periodic"), symmetry=(),
          force_complex_fields=False, cell=(8.0, 8.0, 8.0), **grid_kwargs):
    """A frozen (fields, pml, grid) triple with PML storage allocated.

    The layer skips any axis too thin to hold it — an invariant axis stands for
    a whole infinite direction and has no wall to absorb at — and skips the low
    face of a mirrored axis, where cell 0 is the plane rather than a wall.
    """
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=symmetry, xp=xp, **grid_kwargs)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


#: The two sub-steps the certified pair serves — the argument the predicate takes,
#: and the axis every verdict below is measured along.
SUB_STEPS = ("step_B", "step_D")


# --------------------------------------------------------------------------
# The verdict table the extraction had to preserve.
#
# Every pair below was produced by the PRE-EXTRACTION function: its source was
# pulled textually out of ``step_curl_kernels.py`` at sha256 dc4c0856… — the
# exact bytes the 2026-08-09/10 bit-identity gate compiled, per
# ``certification.json`` — and exec'd over these configurations. So this is not
# a table of what the predicate is believed to do; it is what it DID, before the
# move, case by case. The move is a pure relocation only if every pair still
# reproduces, reason string included: the reason is the part a caller reads when
# a configuration unexpectedly stays on the array path, and a refusal that fires
# for a different clause than the one intended is a refusal nobody can act on.
#
# TWO VERDICTS HAVE SINCE MOVED, DELIBERATELY, and they are the only two: see
# ``_WIDENED_ON_2026_08_15`` below, which carries what each one used to be and
# the ``stepping.py`` line that says the old answer was wrong. Everything else
# still reproduces the pre-extraction function exactly.
# --------------------------------------------------------------------------

_GOLDEN = {
    "covered_periodic": (True, "covered"),
    "covered_m_m_p": (True, "covered"),
    "covered_m_m_m": (True, "covered"),
    "covered_m_p_m": (True, "covered"),
    "complex_storage": (
        False, "complex64 storage: the recurrence is the same but the storage is not"),
    # ADMITTED since 2026-08-20 at BOTH terminations, each on its own device
    # measurement. See ``_WIDENED_ON_2026_08_20``.
    "mirror_plane": (True, "covered"),
    "mirror_plane_metallic": (True, "covered"),
    # THE PLANE CAP, RAISED 2 -> 3 ON 2026-08-20 by re-running the gate with three
    # ``fold_XYZ_*`` specs at both terminations and both of them in the mutation
    # plan: 456 cases per policy, all identical, every mutation as required
    # (``CURL_FOLD_ADMISSION["planes_round_artifacts"]``). Three is every plane a
    # 3-D grid has, so the clause below the cap is now unreachable from a real
    # ``Grid`` -- which is why it appears in ``_UNREACHED_REFUSALS`` and why
    # ``test_the_fold_plane_cap_is_the_record_s_number`` lowers the record instead
    # of building a four-plane grid nothing can build.
    "three_mirror_planes": (True, "covered"),
    # THE TWO ROUTES TO THE FOLD'S TERMINATION, disagreeing and unanswerable. A
    # real ``Grid`` cannot produce either — ``stored_cells`` adds its extra slot
    # exactly when the axis is mirrored and not metallic — which is the reason
    # both are built here from a proxy rather than left as prose in a comment.
    "fold_routes_disagree": (
        False, "axis 1 is folded with stored_cells > owned_cells True and "
               "is_metallic True; the two routes to the fold's termination "
               "disagree and neither can be trusted"),
    "fold_unanswerable": (
        False, "axis 1 is folded and this grid cannot say whether it stores the "
               "slot past MEEP's owned window; the top-plane mask cannot be "
               "decided from is_metallic alone"),
    "cylindrical": (
        False, "cylindrical (Dcyl): prefix-sum radial derivative and axis-row rules"),
    "special_kz": (
        False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"),
    "inactive_layer": (False, "no active PML layer"),
    "numpy_backend": (False, "backend is not CuPy"),
    # A ``Fields`` with no arrays at all now stops on the EARLIER of its two true
    # refusals — it stores no E either. ``component_not_allocated`` is what keeps
    # ``_array_problem``'s allocation branch exercised.
    "no_storage": (False, "E is recomputed from D rather than stored"),
    "component_not_allocated": (False, "Ex is not allocated"),
    "bloch_k": (
        False, "nonzero Bloch k: the wrapped plane carries a phase real storage "
               "cannot hold"),
    "bfast": (False, "BFAST: a second additive term on every curl target"),
    # ADMITTED since 2026-08-15 — the curl differences the STORED E and H arrays
    # and reads no polarization at all. See ``_WIDENED_ON_2026_08_15``.
    "polarizations": (True, "covered"),
    # ADMITTED since 2026-08-20. The refusal said "a Pade factor on the
    # constitutive product", which is true of ``update_E`` and false of this
    # sub-step: ``calc_nonlinear_u`` is applied where E is recovered from D, and
    # ``step_B``/``step_D`` difference the STORED E and H arrays. Retired on a
    # static scan of ``stepping`` performed rather than quoted, a device leg
    # (112/112 per policy on grids carrying a live chi) and the premise armed as
    # ``chi_pade_scale_the_curl`` and CAUGHT. See
    # :data:`coverage.CURL_NONLINEAR_ADMISSION`.
    "chi2": (True, "covered"),
    "chi3": (True, "covered"),
    "no_stored_E": (False, "E is recomputed from D rather than stored"),
    "wrong_dtype": (False, "Bz is float64, not float32"),
    "wrong_shape": (False, "Dy has shape (2, 2, 2), not the grid's (8, 8, 8)"),
    "not_contiguous": (False, "fu_Bx is not C-contiguous"),
    "pml_vector_missing": (False, "pml.sinv_y_h is missing"),
    "pml_vector_dtype": (False, "pml.kms_z is float64, not float32"),
    "pml_vector_size": (False, "pml.sinv_x has 7 entries, not the axis's 8"),
}

#: The cases whose verdict DEPENDS on which curl is being asked about. A
#: conductivity is read per curl term (``fields.condfac_for(term.target)``,
#: stepping.py:508), so a sigma installed on the D components disqualifies
#: ``step_D`` and leaves ``step_B`` an ordinary curl — and the reverse for a B
#: sigma, which in practice means an ``mp.Absorber``'s magnetic half.
_GOLDEN_BY_SUB_STEP = {
    "conductivity": {
        "step_B": (True, "covered"),
        "step_D": (False, "Dx carries a conductivity: routes to the three-history "
                          "conductive-PML recurrence"),
    },
    "conductivity_b": {
        "step_B": (False, "Bx carries a conductivity: routes to the three-history "
                          "conductive-PML recurrence"),
        "step_D": (True, "covered"),
    },
}

#: What the pre-2026-08-15 predicate returned for the two verdicts that moved,
#: and the reason each move is a repair rather than a relaxation. Kept as data so
#: the widening cannot be undone quietly: the test below asserts the predicate
#: does NOT say these things any more, which is the direction that matters — a
#: fail-closed predicate drifting back to over-refusing costs coverage silently
#: and looks exactly like a legitimate refusal in a plan log.
_WIDENED_ON_2026_08_15 = {
    ("polarizations", "step_B"): (
        False, "dispersion: E is (D - sum P) * inv_eps and update_P closes the step"),
    ("polarizations", "step_D"): (
        False, "dispersion: E is (D - sum P) * inv_eps and update_P closes the step"),
    ("conductivity", "step_B"): (
        False, "Dx carries a conductivity: routes to the three-history "
               "conductive-PML recurrence"),
}

#: The fold, retired in TWO measured steps and kept here for the same reason as
#: the block above: so the widening cannot be undone quietly, and so a reader can
#: see what each half of it cost to establish.
#:
#: ``mirror_plane_metallic`` moved on 2026-08-19 with NO DEVICE CODE WRITTEN — the
#: shipped pair was already bit-identical on a folded METALLIC axis, 48/48 at one
#: launch and at 60 under both float32 subnormal policies. ``mirror_plane`` (a
#: folded PERIODIC axis) moved on 2026-08-20, and only because a KERNEL BRANCH was
#: added: the same gate had measured it diverging 0/64 with all 19,649 differing
#: words on one plane — the last stored slot, for the components whose Yee shift
#: on the folded axis is 1 — and none elsewhere. ``BC_MIRROR_PERIODIC`` is that
#: plane's mask and nothing else.
_WIDENED_ON_2026_08_20 = {
    ("mirror_plane", "step_B"): (
        False, "axis 1 is folded PERIODIC: it stores one slot past MEEP's owned "
               "window and this kernel has no top-plane mask"),
    ("mirror_plane", "step_D"): (
        False, "axis 1 is folded PERIODIC: it stores one slot past MEEP's owned "
               "window and this kernel has no top-plane mask"),
    ("mirror_plane_metallic", "step_B"): (
        False, "mirror symmetry: different ghost rule, a parity mask and two fill "
               "passes"),
    ("mirror_plane_metallic", "step_D"): (
        False, "mirror symmetry: different ghost rule, a parity mask and two fill "
               "passes"),
}


def golden(case: str, sub_step: str):
    """The specified verdict for one case at one sub-step."""
    if case in _GOLDEN_BY_SUB_STEP:
        return _GOLDEN_BY_SUB_STEP[case][sub_step]
    return _GOLDEN[case]


#: Every case with a builder, in one place, so the parametrisations agree.
_ALL_CASES = tuple(sorted(set(_GOLDEN) | set(_GOLDEN_BY_SUB_STEP)))


class _GridWhoseFoldRoutesDisagree:
    """A folded PERIODIC grid that also claims the folded axis is metallic.

    THE TWO ROUTES ARE INDEPENDENT AND MUST AGREE. ``Grid.stored_cells`` exceeds
    ``owned_cells`` exactly when an axis is mirrored and not metallic, so this
    object is unbuildable from a real ``Grid`` — which is why the predicate's
    cross-check would otherwise be a clause nothing exercises. The consequence of
    trusting the wrong route is a top-plane mask applied where MEEP steps the
    plane: a wrong answer on one plane of one component, not a crash.
    """

    def __init__(self, grid):
        self._grid = grid

    def is_metallic(self, axis):
        return True if self._grid.is_mirrored(axis) else self._grid.is_metallic(axis)

    def __getattr__(self, item):
        return getattr(self._grid, item)


class _GridThatCannotSayIfItStoresPastOwned:
    """A folded grid with no ``owned_cells``, the shape ``test_stepping``'s stubs have.

    ``stepping._stored_past_owned`` answers False for such a grid on purpose. Doing
    the same HERE would classify a folded PERIODIC axis as metallic and dispatch it
    to a kernel with no top-plane mask, so ``coverage._stored_past_owned_from``
    answers None instead and the fold clause refuses by name.
    """

    def __init__(self, grid):
        self._grid = grid

    def __getattr__(self, item):
        if item == "owned_cells":
            raise AttributeError("owned_cells")
        return getattr(self._grid, item)


def configuration(case: str, xp):
    """Build one named configuration. Every refusal clause in the predicate that
    a configuration can reach has a case here; the three that cannot be reached
    (a mirrored or cylindrical axis behind the whole-grid refusals, and a
    boundary kind with no code) are refused earlier and pinned by their own
    tests below."""
    if case == "covered_periodic":
        return build(xp)
    if case.startswith("covered_"):
        kinds = {"m": "metallic", "p": "periodic"}
        return build(xp, boundaries=tuple(kinds[c] for c in case.split("_")[1:]))
    if case == "complex_storage":
        return build(xp, force_complex_fields=True)
    if case == "mirror_plane":
        return build(xp, symmetry=("y",))
    if case == "mirror_plane_metallic":
        return build(xp, symmetry=("y",),
                     boundaries=("periodic", "metallic", "periodic"))
    if case == "three_mirror_planes":
        return build(xp, symmetry=("x", "y", "z"))
    if case == "fold_routes_disagree":
        fields, layer, grid = build(xp, symmetry=("y",))
        return fields, layer, _GridWhoseFoldRoutesDisagree(grid)
    if case == "fold_unanswerable":
        fields, layer, grid = build(xp, symmetry=("y",))
        return fields, layer, _GridThatCannotSayIfItStoresPastOwned(grid)
    if case == "cylindrical":
        grid = Grid(resolution=1.0, cell_size=(8.0, 0.0, 8.0), cylindrical=True,
                    m=0, xp=xp)
        # phi is the one-cell invariant axis and r's low side is the axis itself,
        # so the layer goes on the high r face and both z faces.
        layer = PML(grid=grid, thickness={"x": (0, 2), "z": 2})
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, layer, grid
    if case == "special_kz":
        return build(xp, cell=(8.0, 8.0, 0.0), dimensions=2, beta=0.25)
    if case == "conductivity":
        fields, layer, grid = build(xp)
        fields.set_d_conductivity(xp.full(grid.shape, 0.4, dtype=xp.float32))
        return fields, layer, grid
    if case == "conductivity_b":
        # The magnetic half of an ``mp.Absorber``, which is the only thing that
        # sets one (``set_b_conductivity``, fields.py:776). It reaches step_B and
        # nothing else.
        fields, layer, grid = build(xp)
        fields.set_b_conductivity(xp.full(grid.shape, 0.4, dtype=xp.float32))
        return fields, layer, grid
    if case == "inactive_layer":
        grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=xp)
        layer = PML(grid=grid, thickness=0)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, layer, grid
    if case == "numpy_backend":
        grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=numpy)
        layer = PML(grid=grid, thickness=2)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, layer, grid
    if case == "no_storage":
        grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=xp)
        layer = PML(grid=grid, thickness=2)
        return Fields(grid=grid), layer, grid  # no enable_*_storage(): no arrays
    if case == "component_not_allocated":
        # Storage enabled, then one volume taken away: what ``no_storage`` used to
        # reach before the ``stores_E`` clause landed in front of it.
        fields, layer, grid = build(xp)
        fields.Ex = None
        return fields, layer, grid
    if case == "bloch_k":
        return build(xp, k_point=(0.25, 0.0, 0.0))
    if case == "bfast":
        return build(xp, bfast_scaled_k=(0.1, 0.0, 0.0))
    if case == "polarizations":
        fields, layer, grid = build(xp)
        fields.polarizations.append(object())
        return fields, layer, grid
    if case == "no_stored_E":
        # The invariant the dispersion admission rests on, and the one case a real
        # PML run cannot produce: ``enable_pml_storage`` forces stored E
        # (fields.py:678). Standing it up by hand is the only way to exercise the
        # clause, and the clause is the only thing between an unstored E and a
        # kernel binding ``fields.Ex`` as though update_E had written it.
        fields, layer, grid = build(xp)

        class _Recomputed:
            stores_E = False

            def __getattr__(self, item):
                return getattr(fields, item)

        return _Recomputed(), layer, grid
    if case in ("chi2", "chi3"):
        fields, layer, grid = build(xp)
        setattr(fields, f"_{case}_components", {"Ex": 0.5})
        return fields, layer, grid
    if case == "wrong_dtype":
        fields, layer, grid = build(xp)
        fields.Bz = fields.Bz.astype(xp.float64)
        return fields, layer, grid
    if case == "wrong_shape":
        fields, layer, grid = build(xp)
        fields.Dy = xp.zeros((2, 2, 2), dtype=xp.float32)
        return fields, layer, grid
    if case == "not_contiguous":
        fields, layer, grid = build(xp)
        fields.fu_Bx = xp.asfortranarray(fields.fu_Bx)
        return fields, layer, grid
    if case == "pml_vector_missing":
        fields, layer, grid = build(xp)
        layer.sinv_y_h = None
        return fields, layer, grid
    if case == "pml_vector_dtype":
        fields, layer, grid = build(xp)
        layer.kms_z = layer.kms_z.astype(xp.float64)
        return fields, layer, grid
    if case == "pml_vector_size":
        fields, layer, grid = build(xp)
        layer.sinv_x = layer.sinv_x.reshape(-1)[:-1]
        return fields, layer, grid
    raise AssertionError(f"no builder for {case!r}")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("case", _ALL_CASES)
def test_the_predicate_states_the_verdict_the_table_specifies(case, sub_step, xp):
    """Verdict AND reason, case by case, sub-step by sub-step.

    For every case but the two in ``_WIDENED_ON_2026_08_15`` this is still the
    relocation test it started as: the pair is what the pre-extraction function
    returned on the same configuration.
    """
    fields, layer, grid = configuration(case, xp)
    assert coverage.covers_real_pml_curl(fields, layer, grid, sub_step) \
        == golden(case, sub_step), (
            f"{case}/{sub_step}: the predicate's verdict is not the specified one")


@pytest.mark.parametrize("case,sub_step", sorted(_WIDENED_ON_2026_08_15))
def test_the_widened_clauses_have_not_drifted_back_to_over_refusing(case, sub_step, xp):
    """THE TEST THAT FAILS IF THE PREDICATE GOES BACK TO REFUSING THESE.

    Two clauses were narrowed on 2026-08-15 because they refused a sub-step for a
    fact belonging to a different one: the dispersion refusal (an ``update_E``
    fact, three corpus rows x two sub-steps) and the conductivity loop over all
    six curl targets (one corpus row at ``step_B``). Measured, the pair cost seven
    sub-steps of the 759-slot corpus denominator against the sibling track's
    per-sub-step predicate, in the direction that loses coverage silently.

    A fail-closed predicate re-acquiring a refusal is the failure this catches,
    and it is a quiet one: the sub-step just stays on the array path and the plan
    log records a plausible-looking reason. So the OLD verdict is carried here as
    data and asserted against, rather than the new one merely being asserted
    somewhere.
    """
    fields, layer, grid = configuration(case, xp)
    now = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    assert now != _WIDENED_ON_2026_08_15[(case, sub_step)], (
        f"{case}/{sub_step} is refused again with the pre-2026-08-15 reason; the "
        f"widening has been reverted and seven corpus sub-steps went with it")
    assert now == golden(case, sub_step)


def test_an_unnamed_sub_step_raises_rather_than_refusing(xp):
    """A typo must not read as a legitimate "not covered".

    ``covers_real_pml_curl(fields, pml, grid, "stepB")`` returning ``(False, ...)``
    would look exactly like any other refusal in a plan log and the sub-step would
    stay on the array path forever. The constitutive predicate's ``side`` argument
    raises for the same reason, and this is the same argument.
    """
    fields, layer, grid = build(xp)
    for name in ("stepB", "B", "update_H", None):
        with pytest.raises(ValueError, match="sub_step must be one of"):
            coverage.covers_real_pml_curl(fields, layer, grid, name)


def test_the_sub_step_targets_are_steppings_own_curl_terms(xp):
    """``CURL_SUB_STEPS`` is a transcription, and this is what diffs it.

    The whole per-sub-step conductivity narrowing rests on which components each
    curl WRITES. Getting that table wrong by one component admits a conductive
    sub-step the three-history recurrence owns — a smooth, absorbing, wrong field
    rather than a crash.
    """
    from .. import stepping  # noqa: PLC0415

    assert coverage.CURL_SUB_STEPS == {
        "step_B": tuple(term.target for term in stepping.B_CURL_TERMS),
        "step_D": tuple(term.target for term in stepping.D_CURL_TERMS),
    }


# Every ``return False`` in the shipped curl predicate that NO configuration in
# the table reaches, identified by a distinctive fragment of its own source line.
#
# IT IS EMPTY, and as of 2026-08-20 that is a fact rather than an aspiration: every
# refusal this predicate can state is reached by a table case or by a hand-built
# triple from the constitutive slice's ``_synthetic``, which is where the objects
# that cannot answer live. The list held THREE entries until the fold was admitted:
#
#   * ``f"axis {axis} is mirrored"`` no longer exists at all. The fold clause is now
#     a plane cap and a termination cross-check, both reached — by
#     ``three_mirror_planes``, ``fold_routes_disagree`` and ``fold_unanswerable``.
#   * the boundary-kind refusal moved into ``_real_curl_boundary_codes_from``,
#     whose single ``return False, no_codes`` site the same three cases reach.
#   * ``f"axis {axis} is the cylindrical r = 0 axis"`` was unreachable only while
#     the blanket mirror refusal sat above it and fired first. With the fold
#     admitted, ``_synthetic``'s ``grid_reports_the_axis_rule_without_cylindrical``
#     — an axis rule declared with no ``cylindrical`` flag — walks straight into it.
#     MEASURED, not argued: that case is the one this list was checked against.
#
# An empty list is the strong form of this test. Adding an entry means declaring a
# refusal nothing can reach, and it has to be justified where it is added.
#: Refusal lines no configuration in this slice can reach, each with the reason.
#: THE LIST IS NOT A WAIVER: a clause on it must be unreachable as a matter of
#: arithmetic, not merely unbuilt, and the test below fails if one of them becomes
#: reachable again.
#:
#: * the plane cap -- ``len(folded) > CURL_FOLD_ADMISSION["folded_planes_swept"]``
#:   with the cap at 3 and ``folded`` drawn from ``range(3)``. A grid has three
#:   axes, so the comparison cannot be true. The clause is kept rather than
#:   deleted because it is what makes the predicate read the record's number
#:   instead of restating it, and lowering that number must still refuse -- which
#:   ``test_the_fold_plane_cap_is_the_record_s_number`` measures directly.
_UNREACHED_REFUSALS = ("mirror planes at once",)


def _fields_that_cannot_answer_about_conductivity(xp):
    """A (fields, pml, grid) triple whose ``fields`` has no ``condfac_for``.

    The curl's own fail-closed read, and one no grid stands in for: measured
    escaping the predicate as an ``AttributeError`` before 2026-08-15.
    """
    fields, layer, grid = build(xp)

    class _Mute:
        def __getattr__(self, item):
            if item == "condfac_for":
                raise AttributeError("condfac_for")
            return getattr(fields, item)

    return _Mute(), layer, grid


def _refusal_lines() -> dict:
    """Absolute line number -> source text, for every ``return False`` site."""
    import inspect

    lines, start = inspect.getsourcelines(coverage.covers_real_pml_curl)
    return {start + offset: text.strip()
            for offset, text in enumerate(lines)
            if text.strip().startswith("return False")}


def _lines_executed(function, *args) -> set:
    """The line numbers ``function`` actually ran for these arguments."""
    import sys

    code = function.__code__
    seen: set = set()

    def tracer(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == "line":
            seen.add(frame.f_lineno)
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        function(*args)
    finally:
        sys.settrace(previous)
    return seen


def test_the_golden_table_covers_every_refusal_the_predicate_can_state(xp):
    """A verdict table that has drifted away from the predicate pins nothing.

    NOT A COUNT — A TRACE, which is what the constitutive slice's equivalent
    already was. The count version compared the table's distinct reasons against
    ``len(re.findall("return False"))`` minus a hard-coded 3, so it moved for
    reasons that had nothing to do with coverage: consolidating the per-array
    dtype/shape/contiguity checks into one fail-closed helper changed the number
    without changing a single verdict. A trace says which clause went unreached
    and names it.
    """
    from .test_constitutive_pml_real import _SYNTHETIC_CASES, _synthetic

    refusals = _refusal_lines()
    reached: set = set()
    # BOTH sub-steps, because a clause reached only for one of them is a clause
    # half the shipped kernel pair runs without.
    for sub_step in SUB_STEPS:
        for case in _ALL_CASES:
            fields, layer, grid = configuration(case, xp)
            reached |= _lines_executed(coverage.covers_real_pml_curl,
                                       fields, layer, grid, sub_step)
        for case in _SYNTHETIC_CASES:
            fields, layer, grid = _synthetic(case, "H", xp)
            reached |= _lines_executed(coverage.covers_real_pml_curl,
                                       fields, layer, grid, sub_step)
        # ...plus the one object this predicate asks that no grid answers, and that
        # the constitutive predicate never reads at all: ``fields.condfac_for``.
        reached |= _lines_executed(
            coverage.covers_real_pml_curl,
            *_fields_that_cannot_answer_about_conductivity(xp), sub_step)

    unreached = sorted(text for line, text in refusals.items() if line not in reached)
    admitted = sorted(text for text in unreached
                      if any(fragment in text for fragment in _UNREACHED_REFUSALS))
    assert unreached == admitted, {
        "unreached and not admitted": sorted(set(unreached) - set(admitted))}
    assert len(unreached) == len(_UNREACHED_REFUSALS), {
        "admitted but now reachable": sorted(
            fragment for fragment in _UNREACHED_REFUSALS
            if not any(fragment in text for text in unreached))}


def test_every_swept_boundary_combination_is_covered(xp):
    """The four combinations the bit-identity gate sweeps must all be admitted.

    A gate that measures configurations the predicate then refuses is measuring
    nothing that can ever run. The four are read off ``certification.json``'s
    ``bit_identity_gate.swept.boundary_sets``.
    """
    for boundaries in (("periodic", "periodic", "periodic"),
                       ("metallic", "metallic", "periodic"),
                       ("metallic", "metallic", "metallic"),
                       ("metallic", "periodic", "metallic")):
        fields, layer, grid = build(xp, boundaries=boundaries)
        for sub_step in SUB_STEPS:
            covered, reason = coverage.covers_real_pml_curl(
                fields, layer, grid, sub_step)
            assert covered, f"{boundaries} at {sub_step} was refused: {reason}"
        assert coverage.real_pml_boundary_kinds(grid) == boundaries


def test_the_boundary_spellings_are_steppings_own():
    """"Same precedence, same spellings" — measured against the four constants.

    ``coverage.py`` emitted ``"cyl_axis"`` where ``stepping.CYL_AXIS`` is
    ``"axis"``. No verdict moved (neither string is in ``BC_CODES``, and the
    ``cylindrical`` and ``is_axis`` clauses fire first), but it made this module
    disagree with ``stepping._boundary_kinds`` on 19 of the 186 measured corpus
    rows while its docstring claimed agreement — a false-alarm column in the
    record, and a transcription that nothing checked.
    """
    from .. import stepping  # noqa: PLC0415

    assert coverage.PERIODIC == stepping.PERIODIC
    assert coverage.MIRROR == stepping.MIRROR
    assert coverage.METALLIC == stepping.METALLIC
    assert coverage.CYL_AXIS == stepping.CYL_AXIS
    assert set(coverage.BC_CODES) == {stepping.PERIODIC, stepping.METALLIC}
    # THE CURL PAIR'S OWN MAP IS A SUPERSET, AND THE EXTRA KEY IS NOT A STEPPING
    # SPELLING. ``stepping._boundary_kinds`` answers "mirror" at both terminations;
    # "mirror_periodic" is this pair's name for the half that stores the slot past
    # MEEP's owned window. Keeping it OUT of ``BC_CODES`` is what stops
    # ``complex_pml_kernels``, which refuses any kind outside that dict, from
    # inheriting an admission it has no branch for.
    assert coverage.MIRROR_PERIODIC not in (stepping.PERIODIC, stepping.METALLIC,
                                            stepping.MIRROR, stepping.CYL_AXIS)
    assert set(coverage.REAL_CURL_BC_CODES) == set(coverage.BC_CODES) | {
        coverage.MIRROR_PERIODIC}
    for kind, code in coverage.BC_CODES.items():
        assert coverage.REAL_CURL_BC_CODES[kind] == code, kind
    assert len(set(coverage.REAL_CURL_BC_CODES.values())) == 3


def test_the_boundary_resolution_agrees_with_steppings_on_every_shaped_grid(xp):
    """Not the spellings — the whole RESOLUTION, against the function it copies.

    ``real_pml_boundary_kinds`` exists because ``coverage.py`` imports nothing;
    that makes it a transcription, and a transcription with no comparison is a
    second implementation nobody diffs. Compared here on every grid this slice
    builds, including the folded and cylindrical ones the predicate refuses —
    which are exactly the grids whose spelling was wrong.
    """
    from .. import stepping  # noqa: PLC0415

    for case in _ALL_CASES:
        fields, layer, grid = configuration(case, xp)
        assert coverage.real_pml_boundary_kinds(grid) == \
            stepping._boundary_kinds(grid, layer if layer.is_active else None), case


def test_absorbing_axes_still_resolve_to_periodic(xp):
    """An axis that absorbs WRAPS. Calling it metallic is a perfect mirror where
    MEEP has a wrap — measured at 6.31e-04 against CPU MEEP where the wrap gives
    4.00e-07, and the reason the complex kernels' hard-coded layout is wrong."""
    fields, layer, grid = build(xp)
    assert layer.is_active
    assert coverage.real_pml_boundary_kinds(grid) == ("periodic",) * 3


# --------------------------------------------------------------------------
# The structural property the extraction bought, pinned so it cannot be undone.
# --------------------------------------------------------------------------

def test_the_predicate_module_pulls_in_no_device_dependency():
    """One ``import cupy`` in ``coverage.py`` and this whole file is a skip again.

    That is not hypothetical: it is exactly the state this file was in until the
    predicate moved out of ``step_curl_kernels``. The check is textual and on the
    shipped source, so it fails on the import line rather than on the machine
    that happens not to have the library.
    """
    import pathlib

    from .test_constitutive_pml_real import STDLIB_ONLY, device_free_import_report

    source = (pathlib.Path(__file__).parent / "coverage.py").read_text(encoding="utf-8")
    report = device_free_import_report(source)
    assert report["absolute"] <= STDLIB_ONLY, (
        f"coverage.py imports {sorted(report['absolute'] - STDLIB_ONLY)}; the "
        f"predicate has to stay evaluable on a host with no GPU and no optional "
        f"dependency, which is the only reason it lives in its own module")
    # RELATIVE AND DYNAMIC TOO. This check used to filter ``node.level == 0`` and
    # so was blind to every ``from . import ...`` — a spelling that reaches CuPy
    # through any sibling in this package, and that the Triton track's own
    # coverage module uses.
    assert report["relative"] == [] and report["dynamic"] == set(), report


def test_the_kernel_module_re_exports_the_predicate_rather_than_copying_it():
    """Two copies of a fail-closed predicate is one predicate nobody maintains.

    Read textually so it holds without CuPy: ``step_curl_kernels`` must import
    the three names from ``coverage`` and must not define any of them itself.
    """
    import ast
    import pathlib

    source = (pathlib.Path(__file__).parent / "step_curl_kernels.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    names = {"covers_real_pml_curl", "real_pml_boundary_kinds", "BC_CODES"}

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("coverage"):
            imported.update(alias.name for alias in node.names)
    assert names <= imported, f"not imported from coverage: {sorted(names - imported)}"

    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    defined |= {target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    assert not (names & defined), (
        f"step_curl_kernels re-defines {sorted(names & defined)}; the predicate "
        f"has drifted back into the module that cannot be imported off device")


def test_the_probe_can_still_load_the_kernel_module_by_path():
    """The bit-identity probe imports this module BY PATH, outside the package.

    ``probe_fused_kernel_bit_identity.load_kernel_module`` calls
    ``spec_from_file_location``, so ``from .coverage import …`` raises
    ``ImportError`` there and the by-path fallback beside it is what keeps the
    probe working. That fallback has no other caller and would otherwise be
    exercised for the first time on the one host that runs the gate, hours into
    a device run. Here it is exercised on every laptop, with a stand-in for the
    one import the probe's host really does have.
    """
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(__file__).parent / "step_curl_kernels.py"
    saved = sys.modules.get("cupy")
    if cupy is None:
        sys.modules["cupy"] = types.ModuleType("cupy")
    try:
        spec = importlib.util.spec_from_file_location("probe_step_curl_kernels_test",
                                                      path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
    finally:
        if cupy is None:
            sys.modules.pop("cupy", None)
        if saved is not None:
            sys.modules["cupy"] = saved

    # Loaded outside the package the fallback execs coverage.py into a module of
    # its own, so the function OBJECT differs by construction. What must hold is
    # that it came out of the same file: a fallback that quietly resolved a
    # second copy — or a stale one on sys.path — is the failure this catches.
    assert (module.covers_real_pml_curl.__code__.co_filename
            == coverage.covers_real_pml_curl.__code__.co_filename), (
        "the by-path load resolved a predicate from a different file than the "
        "package import does; the probe would be gating a second copy")
    assert module.BC_CODES == coverage.BC_CODES
    assert module.real_pml_boundary_kinds.__code__.co_filename \
        == coverage.real_pml_boundary_kinds.__code__.co_filename


# --------------------------------------------------------------------------
# Refusals. Each of these is a SILENT WRONG ANSWER if it leaks.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_fold_is_covered_at_both_terminations_with_the_codes_that_split_them(
        sub_step, xp):
    """The fold is admitted, and the two terminations get DIFFERENT codes.

    Admission alone is not the claim: a predicate that said "covered" and a
    launcher that handed both terminations the same code would step a plane MEEP
    does not own on every folded PERIODIC run, silently, on one plane of one
    component. So this asserts the split as well as the verdict.
    """
    for boundaries, expected in (
            (("periodic", "periodic", "periodic"),
             coverage.REAL_CURL_BC_CODES[coverage.MIRROR_PERIODIC]),
            (("periodic", "metallic", "periodic"),
             coverage.REAL_CURL_BC_CODES["metallic"])):
        fields, layer, grid = build(xp, symmetry=("y",), boundaries=boundaries)
        assert grid.has_symmetry() and grid.is_mirrored(1)
        covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
        assert covered, (boundaries, reason)
        codes, refusal = coverage.real_curl_boundary_codes(grid)
        assert refusal is None, refusal
        assert codes[1] == expected, (boundaries, codes)
        # The unfolded axes are untouched by the fold: same codes as ever.
        assert codes[0] == codes[2] == coverage.REAL_CURL_BC_CODES["periodic"]


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_third_simultaneous_fold_plane_is_admitted_on_its_own_measurement(
        sub_step, xp):
    """Three planes were refused until the gate swept three, and then admitted.

    THE CAP MOVED BECAUSE A MEASUREMENT MOVED, and both halves are pinned here:
    the grid is admitted, and the record says the number came from a run at three
    planes rather than from the spec tuple that used to supply it. The companion
    test lowers the record and shows the refusal comes back, which is what stops
    this from being a test that the clause was deleted.
    """
    fields, layer, grid = build(xp, symmetry=("x", "y", "z"))
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    assert covered, reason
    assert reason == "covered"
    table = coverage.CURL_FOLD_ADMISSION
    assert table["folded_planes_swept"] == 3, table["folded_planes_swept"]
    assert table["planes_round_max_folded_planes_scored"] == 3
    # BOTH terminations reached the count the cap names. A cap resting on the
    # metallic arm alone would be the same over-claim, one plane further out.
    by_arm = table["planes_round_folded_planes_by_arm"]
    for arm in ("folded_all_metallic", "folded_all_periodic"):
        assert max(by_arm[arm]) == 3, (arm, by_arm[arm])


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_fold_plane_cap_is_the_record_s_number(sub_step, xp):
    """Lower the record and the refusal comes back -- the clause reads the record.

    With the cap at 3 and a grid at three axes, ``len(folded) > cap`` cannot be
    true, so the clause is unreachable from any real ``Grid`` (it is on
    ``_UNREACHED_REFUSALS`` for exactly that reason). What can still be measured,
    and is what the clause exists for, is that the number DECIDING the verdict is
    the one in the admission record: put 2 back and the three-plane grid is
    refused by name.

    This replaces the old drop-the-clause mutation, which pinned nothing once the
    clause became unreachable: deleting a dead branch changes no verdict.
    """
    fields, layer, grid = build(xp, symmetry=("x", "y", "z"))
    assert coverage.covers_real_pml_curl(fields, layer, grid, sub_step)[0]

    table = coverage.CURL_FOLD_ADMISSION
    shipped = table["folded_planes_swept"]
    try:
        table["folded_planes_swept"] = 2
        covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
        assert not covered, (
            "lowering the record's plane count did not change the verdict, so the "
            "predicate is not reading the record and the cap is unpinned")
        assert "3 mirror planes at once" in reason, reason
        assert "swept 2 simultaneous" in reason, reason
    finally:
        table["folded_planes_swept"] = shipped
    assert coverage.covers_real_pml_curl(fields, layer, grid, sub_step)[0]


def test_the_stored_past_owned_transcription_is_steppings_own(xp):
    """``coverage`` transcribes ``_stored_past_owned``; a transcription needs a diff.

    ``coverage.py`` imports nothing, so this fact is copied rather than called —
    the same discipline, and the same hazard, as ``real_pml_boundary_kinds``, whose
    copy WAS measured to have drifted (it emitted ``"cyl_axis"``). Compared on every
    grid this slice builds, folded and flat.
    """
    from .. import stepping  # noqa: PLC0415

    for case in _ALL_CASES:
        if case in ("fold_routes_disagree", "fold_unanswerable"):
            continue  # proxies built precisely to make the two answers differ
        _fields, _layer, grid = configuration(case, xp)
        assert coverage._stored_past_owned_from(grid) == tuple(
            stepping._stored_past_owned(grid, axis) for axis in range(3)), case


def test_the_two_routes_to_the_folds_termination_must_agree(xp):
    """A grid whose ``is_metallic`` and stored extent disagree is refused, not guessed.

    Unbuildable from a real ``Grid``, which is the point: without this the clause
    would be prose. The proxy is what makes "neither can be trusted" a measured
    refusal.
    """
    fields, layer, grid = configuration("fold_routes_disagree", xp)
    for sub_step in SUB_STEPS:
        covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
        assert not covered
        assert "disagree" in reason, reason
    assert coverage.real_curl_boundary_codes(grid)[0] is None


def test_a_fold_on_a_grid_that_cannot_answer_is_refused_not_assumed_metallic(xp):
    """No ``owned_cells`` -> None -> refusal, where ``stepping`` answers False.

    The deliberate divergence from the function it transcribes, and the one that
    matters: ``stepping``'s False is safe because nothing downstream of it masks;
    here it would classify a folded PERIODIC axis as metallic and drop the
    top-plane mask.
    """
    fields, layer, grid = configuration("fold_unanswerable", xp)
    assert coverage._stored_past_owned_from(grid) == (None, None, None)
    for sub_step in SUB_STEPS:
        covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
        assert not covered
        assert "cannot say whether it stores" in reason, reason


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_an_inactive_layer_is_refused(sub_step, xp):
    """A zero-thickness layer steps the no-PML path bit-identically, and that
    path is not this kernel."""
    fields, layer, grid = configuration("inactive_layer", xp)
    assert not layer.is_active
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    assert not covered
    assert "PML" in reason


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_numpy_backend_is_refused(sub_step):
    fields, layer, grid = configuration("numpy_backend", numpy)
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    assert not covered
    assert "CuPy" in reason


# --------------------------------------------------------------------------
# The CuPy-only remainder: the launch-side helpers, which hold device views and
# reach through ``step_curl_kernels`` — the module the predicate no longer
# lives in.
# --------------------------------------------------------------------------

@pytest.fixture
def kernels():
    if cupy is None:
        pytest.skip("the launch-side helpers hold CuPy views and compile options")
    from . import step_curl_kernels

    return step_curl_kernels


@pytest.mark.requires_resource("cupy")
def test_the_launcher_raises_on_a_grid_the_predicate_refuses(kernels, xp):
    """The launch-side resolver and the predicate answer about the SAME grid.

    ``real_pml_boundary_codes(kinds)`` was replaced by ``real_curl_boundary_codes
    (grid)`` because a fold's two terminations resolve to one kind string and need
    two codes. What is pinned here is the pairing: a grid the predicate refuses has
    no code triple, and a grid it admits has one whose folded axis carries the code
    the fold's termination calls for.
    """
    # A CYLINDRICAL GRID, not a three-plane fold. Three planes was the refused
    # grid here until 2026-08-20, when the gate swept three and the predicate
    # admitted them -- and a test that kept using it would have been asserting the
    # launcher raises on a grid the predicate now covers, which is the pairing
    # this test exists to forbid.
    fields, layer, grid = configuration("cylindrical", xp)
    assert not coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]
    with pytest.raises(ValueError, match="no boundary-code triple"):
        kernels.real_curl_boundary_codes(grid)

    fields, layer, grid = build(xp, symmetry=("y",))
    assert coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]
    codes = kernels.real_curl_boundary_codes(grid)
    assert [int(c) for c in codes] == [0, 2, 0], codes

    # AND THE THREE-PLANE FOLD, which the predicate now admits, must HAVE a code
    # triple -- the same pairing read from the widened side.
    fields, layer, grid = configuration("three_mirror_planes", xp)
    assert coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]
    assert len(kernels.real_curl_boundary_codes(grid)) == 3


@pytest.mark.requires_resource("cupy")
def test_coefficient_tables_are_views_not_copies(kernels):
    """Flattened once at plan time; ``reshape(-1)`` on an (n,1,1) array is a view."""
    fields, layer, grid = build(cupy)
    for half_integer, first in ((True, "kms_x_h"), (False, "kms_x")):
        tables = kernels.real_pml_curl_tables(layer, half_integer=half_integer)
        assert set(tables) == {f"{label}_{axis}"
                               for axis in "xyz" for label in ("kms", "sinv")}
        for axis, size in zip("xyz", grid.shape):
            assert tables[f"kms_{axis}"].size == size
            assert tables[f"kms_{axis}"].flags.c_contiguous
            assert tables[f"kms_{axis}"].dtype == cupy.float32
        assert tables["kms_x"].data.ptr == getattr(layer, first).data.ptr


@pytest.mark.requires_resource("cupy")
def test_the_two_sub_lattices_are_different_tables(kernels):
    """B reads half-integer, D integer. Backwards is a half-cell error, not a crash."""
    fields, layer, grid = build(cupy)
    half = kernels.real_pml_curl_tables(layer, half_integer=True)
    whole = kernels.real_pml_curl_tables(layer, half_integer=False)
    assert not bool(cupy.array_equal(half["kms_x"], whole["kms_x"])), (
        "the integer and half-integer coefficient sets are identical, so no test "
        "in this suite can tell them apart")


# --------------------------------------------------------------------------
# The two required predicate mutations (R4). Applied to the REAL source.
# --------------------------------------------------------------------------

def mutated_predicate(pattern: str, replacement: str):
    """Recompile ``covers_real_pml_curl`` with one refusal edited out.

    Textual, against the shipping source, and it raises if the pattern matches
    nothing — so a mutation that has drifted away from the code fails loudly
    instead of quietly testing an unmutated function.
    """
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(coverage.covers_real_pml_curl))
    mutated, count = re.subn(pattern, replacement, source)
    if count == 0:
        raise AssertionError(
            f"predicate mutation {pattern!r} matched nothing; the mutation and "
            f"the predicate have drifted apart and it is pinning nothing.")
    namespace = dict(vars(coverage))
    namespace["__builtins__"] = __builtins__
    module = types.ModuleType("mutated_predicate")
    module.__dict__.update(namespace)
    exec(compile(mutated, "<mutated covers_real_pml_curl>", "exec"),
         module.__dict__)
    return module.covers_real_pml_curl


def test_mutation_drop_the_chi_admission_is_caught(xp):
    """Put the chi2/chi3 refusal back and the two corpus nonlinear rows are refused.

    The clause is GONE from the predicate, so the mutation runs the other way:
    re-inserting it must change a verdict, or the admission is not what decides
    these grids and ``CURL_NONLINEAR_ADMISSION`` is describing a widening that
    bought nothing. The four slots it moves are ``step_B``/``step_D`` on
    ``3rd-harm-1d.py`` and ``Test3rdHarm1d.test_3rd_harm_1d``.
    """
    fields, layer, grid = configuration("chi3", xp)
    assert coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]

    narrowed = mutated_predicate(
        r'    shape = facts\["shape"\]\n',
        '    if getattr(fields, "_chi2_components", None) or getattr(\n'
        '            fields, "_chi3_components", None):\n'
        '        return False, "instantaneous chi2/chi3"\n'
        '    shape = facts["shape"]\n')
    covered, reason = narrowed(fields, layer, grid, "step_B")
    assert not covered, (
        "re-inserting the chi2/chi3 refusal did not change the verdict, so the "
        "admission is not what decides a nonlinear grid and this suite is pinning "
        "nothing about it")
    assert "chi2/chi3" in reason, reason


def test_mutation_drop_the_two_route_cross_check_is_caught(xp):
    """Remove the cross-check and a grid whose two routes disagree is admitted.

    The second edge, and the one whose consequence is a wrong ANSWER rather than a
    wrong row count: with the check gone, ``is_metallic`` alone decides the code,
    so a folded axis that stores the extra slot but calls itself metallic gets
    ``BC_METALLIC`` — no top-plane mask — and steps a plane MEEP does not own.
    """
    fields, layer, grid = configuration("fold_routes_disagree", xp)
    assert not coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]

    # ALL THREE FUNCTIONS GO INTO ONE NAMESPACE. Only the first is edited; the
    # other two are re-exec'd unchanged so that their name lookups resolve inside
    # the mutated module. Recompiling the leaf alone leaves the predicate calling
    # the SHIPPED one through its own ``__globals__``, and the mutation measures
    # nothing — the same trap ``_recompiled``'s docstring records for the array
    # checks that moved into ``_array_problem``.
    module = _recompiled(
        "no-cross-check",
        _fold_termination_problem=(
            r"    if bool\(past_owned\) == metallic:\n"
            r"(?:.*\n)*?"
            r"                f\"the fold's termination disagree and neither can be "
            r"trusted\"\)\n",
        ),
        _real_curl_boundary_codes_from=(),
        covers_real_pml_curl=(),
    )
    covered, reason = module.covers_real_pml_curl(fields, layer, grid, "step_B")
    assert covered, (
        "the widened predicate still refused the disagreeing grid, so this mutation "
        "is not exercising the cross-check and the cross-check is unpinned")
    assert reason == "covered"


def _recompiled(label: str, **mutations):
    """A module holding the named shipped functions with refusals edited out.

    MORE THAN ONE FUNCTION, because a refusal may not live in the predicate any
    more: the array dtype/shape/contiguity checks moved into ``_array_problem``
    when the reads went fail-closed, and a mutation that only rewrote
    ``covers_real_pml_curl`` would silently match nothing there. Every mutated
    source is exec'd into one namespace seeded from the real module, so a mutated
    helper is what the mutated predicate calls.

    Raises if any pattern matches other than once — a mutation that has drifted
    away from the code fails loudly instead of quietly testing an unmutated
    function.
    """
    import inspect
    import textwrap

    namespace = dict(vars(coverage))
    namespace["__builtins__"] = __builtins__
    module = types.ModuleType("mutated_predicate")
    module.__dict__.update(namespace)
    for name, patterns in mutations.items():
        source = textwrap.dedent(inspect.getsource(getattr(coverage, name)))
        for pattern in patterns:
            source, count = re.subn(pattern, "", source)
            assert count == 1, (
                f"{name}: {pattern!r} matched {count} times, expected 1; the "
                f"mutation and the predicate have drifted apart")
        exec(compile(source, f"<{label} {name}>", "exec"), module.__dict__)
    return module


# ``_also_drop_per_axis_mirror`` lived here until 2026-08-20. It widened the
# predicate by deleting the blanket mirror refusal, which is a mutation with
# nothing left to defeat now that the fold is ADMITTED. Its role — pinning where
# the fold admission stops — passed to the two mutations above, which delete the
# plane cap and the two-route cross-check instead. Deleted rather than left
# raising "matched 0 times", which is a red test that says nothing.


def test_mutation_drop_the_dtype_check_is_caught(xp):
    """Drop the float32 check and complex64 storage becomes 'covered'.

    The recurrence is the same algebra in complex; the STORAGE is not, and a
    float32 kernel reading a complex64 buffer reads interleaved real and
    imaginary parts as neighbouring cells.
    """
    fields, layer, grid = build(xp, force_complex_fields=True)
    assert not coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]

    dropped = mutated_predicate(
        r"    if getattr\(fields, \"force_complex_fields\", False\):\n"
        r"        return False, \"complex64 storage[^\"]*\"\n", "")
    covered, reason = dropped(fields, layer, grid, "step_B")
    # The per-array dtype loop is the second line of defence and SHOULD still
    # catch it. That is the finding this mutation records: the dtype refusal is
    # stated twice, so dropping either one alone does not leak.
    assert not covered, "dropping the flag check leaked; nothing else caught it"
    assert "not float32" in reason, (
        f"the survivor refused for the wrong reason ({reason!r}); the second "
        f"line of defence is not the dtype check")

    # BOTH refusals removed — and the second one now lives in ``_array_problem``,
    # which is why this goes through the multi-function recompiler. A mutation
    # that still targeted only the predicate would match nothing here and, before
    # the drift assertion, would have reported a pass for an edit never applied.
    blind = _recompiled(
        "dtype-blind",
        covers_real_pml_curl=(
            r"    if getattr\(fields, \"force_complex_fields\", False\):\n"
            r"        return False, \"complex64 storage[^\"]*\"\n",),
        _array_problem=(
            r"        if array\.dtype != xp\.float32:\n"
            r"            return f\"\{label\} is \{array\.dtype\}, not float32\"\n",),
    )
    covered, reason = blind.covers_real_pml_curl(fields, layer, grid, "step_B")
    assert covered, (
        "with BOTH dtype refusals removed the predicate still refused complex "
        "storage, so neither of them is the check that stops it")


# --------------------------------------------------------------------------
# The three clauses the 2026-08-15 sub-step round touched, each with a mutation
# that makes the leak visible. A widening is a fail-closed refusal being
# RELAXED, so it needs more than a verdict table saying the new answer: it needs
# the old answer's absence pinned (above), the stepping.py fact behind it
# measured (below), and the surviving clause proved load-bearing (here).
# --------------------------------------------------------------------------

def test_mutation_widen_the_conductivity_loop_back_to_all_six_targets_is_caught(xp):
    """Charge every target's sigma to both curls and step_B is refused again.

    This is the over-refusal as it stood, reconstructed from the shipped source:
    the loop ranges over ``CURL_SUB_STEPS[sub_step]``, and substituting the union
    of both sub-steps' targets is exactly what the pre-2026-08-15 predicate did.
    """
    fields, layer, grid = configuration("conductivity", xp)
    assert coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]

    widened = mutated_predicate(
        r"for component in CURL_SUB_STEPS\[sub_step\]",
        'for component in ("Dx", "Dy", "Dz", "Bx", "By", "Bz")')
    covered, reason = widened(fields, layer, grid, "step_B")
    assert not covered and "conductivity" in reason, (
        "the sub-step-blind conductivity loop no longer refuses step_B on a D "
        "conductivity, so the narrowing is not what admits that row")


def test_mutation_drop_the_stored_E_clause_is_caught(xp):
    """Drop it and an E that update_E never wrote becomes 'covered'.

    The clause is what the dispersion admission stands on. With a live pole and
    an unstored E, ``get_E`` returns a freshly computed ``D * inv_eps`` one
    polarization out of date (fields.py:1008-1021) while the kernel binds
    ``fields.Ex`` — which is either a zeroed array or a stale one, and smooth
    either way.
    """
    fields, layer, grid = configuration("no_stored_E", xp)
    assert not coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]

    dropped = mutated_predicate(
        r"    if not getattr\(fields, \"stores_E\", False\):\n"
        r"        return False, \"E is recomputed from D rather than stored\"\n", "")
    covered, reason = dropped(fields, layer, grid, "step_B")
    assert covered, (
        f"with the stores_E clause removed the predicate still refused ({reason!r}), "
        f"so nothing here is pinning the invariant dispersion is admitted on")


def test_mutation_drop_the_int32_clause_is_caught(xp):
    """Drop it and a volume the kernel's ``int idx`` cannot address is 'covered'.

    2**31 cells wraps the index rather than failing the launch, so the leak is a
    silently corrupted volume. The clause was carried by the constitutive
    predicate from the start and by this one only from 2026-08-15.
    """
    from .test_constitutive_pml_real import _synthetic

    fields, layer, grid = _synthetic("cells_exceed_int32", "H", xp)
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, "step_B")
    assert not covered and "int32 index range" in reason, reason

    dropped = mutated_predicate(
        r"    if cells >= 2 \*\* 31:\n"
        r"        return False, f\"\{cells\} cells exceeds the kernel's int32 "
        r"index range\"\n", "")
    covered, reason = dropped(fields, layer, grid, "step_B")
    # The survivor is the per-array SHAPE check — and it survives only because
    # this fixture is a grid stand-in whose volumes are still 8x8x8. A real
    # 2**31-cell grid carries arrays of that shape, so on the configuration the
    # clause exists for there is no second line of defence at all. That is the
    # finding, and it is why the leg below removes the shape checks too.
    assert not covered and "not the grid's" in reason, (
        f"the survivor refused for an unexpected reason ({reason!r})")

    blind = _recompiled(
        "int32-blind",
        covers_real_pml_curl=(
            r"    if cells >= 2 \*\* 31:\n"
            r"        return False, f\"\{cells\} cells exceeds the kernel's int32 "
            r"index range\"\n",),
        _array_problem=(
            r"        if tuple\(array\.shape\) != shape:\n"
            r"            return f\"\{label\} has shape \{tuple\(array\.shape\)\}, "
            r"not the grid's \{shape\}\"\n",),
        _coefficient_vector_problem=(
            r"        if vector\.size != shape\[axis\]:\n"
            r"            return \(f\"pml\.\{label\} has \{vector\.size\} entries, "
            r"not the axis's \"\n"
            r"                    f\"\{shape\[axis\]\}\"\)\n",
            r"        if tuple\(vector\.shape\) not in \(expected, \(shape\[axis\],\)\):\n"
            r"            return \(f\"pml\.\{label\} has shape \{tuple\(vector\.shape\)\}, "
            r"not axis \"\n"
            r"                    f\"\{axis\}'s broadcast shape \{expected\}\"\)\n",),
    )
    covered, reason = blind.covers_real_pml_curl(fields, layer, grid, "step_B")
    assert covered, (
        f"with the int32 clause and every shape comparison removed the predicate "
        f"still refused ({reason!r}), so the int32 clause is not what stops a "
        f"grid the kernel's int index cannot address")


# --------------------------------------------------------------------------
# THE WIDENINGS, MEASURED ON THE ARRAY PATH ITSELF.
#
# The two clauses above were narrowed on the strength of a claim about
# ``stepping.py``: that a curl reads no polarization, and that it reads only its
# OWN targets' conductivity. Neither is checked by any verdict table — a verdict
# table only says what the predicate returns. These run the real sub-steps on
# this host's NumPy and compare bytes, so the claims are measured rather than
# read. A gate leg on device is still owed for the kernel side (disposition
# §8.2a); this is the array-path half, and it is the half that says whether the
# predicate is describing the engine correctly.
# --------------------------------------------------------------------------

def _seeded(xp, **build_kwargs):
    """A built triple whose eighteen volumes hold physical-band values, not zeros.

    Zero is a fixed point of every sub-step here, so a comparison of two
    all-zero runs passes for any pair of implementations. Each test below
    asserts its own vacuity floor on top of this.
    """
    fields, layer, grid = build(xp, **build_kwargs)
    state = numpy.random.RandomState(20260815)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        values = state.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32)
        getattr(fields, name)[...] = xp.asarray(values)
    return fields, layer, grid


def _stepped(xp, sub_step: str, prepare=None, **build_kwargs) -> dict:
    """Run one real curl sub-step and return copies of everything it wrote."""
    from .. import stepping  # noqa: PLC0415

    fields, layer, grid = _seeded(xp, **build_kwargs)
    if prepare is not None:
        prepare(fields, grid, xp)
    getattr(stepping, sub_step)(fields, layer)
    written = coverage.CURL_SUB_STEPS[sub_step]
    return {name: numpy.asarray(getattr(fields, name)).copy()
            for name in written + tuple("fu_" + n for n in written)}


def _stored_E(xp, prepare=None, **build_kwargs) -> dict:
    """Run the real ``update_E`` on the same seeded fields — the control leg."""
    from .. import stepping  # noqa: PLC0415

    fields, layer, grid = _seeded(xp, **build_kwargs)
    if prepare is not None:
        prepare(fields, grid, xp)
    stepping.update_E(fields, layer)
    return {name: numpy.asarray(getattr(fields, name)).copy()
            for name in ("Ex", "Ey", "Ez")}


def _identical(left: dict, right: dict) -> bool:
    return all(numpy.array_equal(left[name].view(numpy.uint32),
                                 right[name].view(numpy.uint32))
               for name in left)


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_registered_polarization_changes_neither_curl_bit_for_bit(sub_step, xp):
    """THE MEASUREMENT BEHIND ADMITTING DISPERSION.

    Two runs whose fields hold identical values, one with a live Lorentzian
    driving all three E components. ``step_B`` differences the stored E arrays
    and ``step_D`` the stored H arrays; if either ever consulted P, or swapped in
    ``D - sum P``, these would differ. They are compared as raw bits, not to a
    tolerance, because the predicate's whole currency is bit-identity.
    """
    from ..dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    def register(fields, grid, module):
        state = PolarizationState(
            Susceptibility(frequency=0.3, gamma=0.05), 1.0, grid, module.float32)
        assert state.driven() == ("Ex", "Ey", "Ez"), (
            "the susceptibility drives nothing, so this configuration is not "
            "dispersive and the comparison below is vacuous")
        for name in state.P:
            state.P[name][...] = module.asarray(
                numpy.full(grid.shape, 0.25, dtype=numpy.float32))
            state.P_prev[name][...] = module.asarray(
                numpy.full(grid.shape, 0.125, dtype=numpy.float32))
        fields.polarizations.append(state)

    plain = _stepped(xp, sub_step)
    dispersive = _stepped(xp, sub_step, prepare=register)

    # VACUITY FLOOR: the sub-step has to have written something, or two runs of
    # nothing would compare equal.
    assert any(numpy.any(values != 0.0) for values in plain.values()), (
        f"{sub_step} wrote only zeros; this comparison would pass for any pair")
    # THE CONTROL, and without it this test passes on a polarization that failed
    # to install. The SAME pole, on the SAME seeded fields, must change what
    # ``update_E`` writes — measured at 7.05e-01 on all three E components — so
    # "the curls are unchanged" is a statement about the curls and not about a
    # susceptibility that turned out to be inert.
    assert not _identical(_stored_E(xp), _stored_E(xp, prepare=register)), (
        "the registered polarization did not change update_E either, so it is "
        "not live and this comparison measures nothing")
    assert _identical(plain, dispersive), (
        f"{sub_step} produced different bits with a polarization registered, so "
        f"the curl DOES read one and admitting dispersion is wrong")


def test_a_one_sided_conductivity_reaches_one_curl_and_not_the_other(xp):
    """THE MEASUREMENT BEHIND CHARGING A SIGMA TO ONE SUB-STEP.

    A D conductivity must change ``step_D``'s output and leave ``step_B``'s bits
    untouched — which is what ``_apply_curl`` reading ``condfac_for(term.target)``
    per term (stepping.py:508) means in practice. The second half is the control:
    without it, a test that only checked step_B was unchanged would pass on a
    sigma that had silently failed to install.
    """
    def install(fields, grid, module):
        sigma = module.asarray(numpy.full(grid.shape, 0.4, dtype=numpy.float32))
        fields.set_d_conductivity(sigma)
        assert fields.has_conductivity

    for sub_step, must_change in (("step_B", False), ("step_D", True)):
        plain = _stepped(xp, sub_step)
        conductive = _stepped(xp, sub_step, prepare=install)
        assert any(numpy.any(values != 0.0) for values in plain.values()), (
            f"{sub_step} wrote only zeros; this comparison would pass for any pair")
        assert _identical(plain, conductive) is (not must_change), (
            f"{sub_step}: a D conductivity "
            f"{'did not change' if must_change else 'changed'} its output, which "
            f"is the opposite of what the per-sub-step conductivity clause claims")


# --------------------------------------------------------------------------
# THE INSTANTANEOUS chi2/chi3 ADMISSION (2026-08-20)
#
# The refusal said "a Pade factor on the constitutive product", which is true of
# ``update_E`` and false of this sub-step. Three things are pinned: the READING
# is re-performed here rather than quoted, the RECORD is bound to the artifact
# that produced it, and the DISJOINTNESS with the nonlinear constitutive family
# is re-measured on a grid carrying a chi.
# --------------------------------------------------------------------------

CHI_NEEDLES = ("chi2", "chi3", "nonlinear")


def _stepping_chi_scan():
    """Re-perform the gate's static scan of ``stepping``, here, at the merge bar.

    THE GATE RAN THIS ON A DEVICE HOST AGAINST THE FILE AS IT WAS THEN. What this
    copy adds is that the reading is re-checked against the file as it is NOW, on
    every test run: a helper that gained a chi lookup after the gate ran would
    make the admission false, and nothing else in this suite would notice.

    Docstrings are excluded, and the exclusion is what makes the scan usable: two
    functions inside the curl's closure mention the nonlinearity only in prose,
    and counting prose would report the null as false.
    """
    import ast  # noqa: PLC0415

    source = (pathlib.Path(__file__).resolve().parents[1] / "stepping.py"
              ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, node)

    def symbols(node):
        prose = {id(child.value) for child in ast.walk(node)
                 if isinstance(child, ast.Expr)
                 and isinstance(child.value, ast.Constant)
                 and isinstance(child.value.value, str)}
        out = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                out.add(child.id)
            elif isinstance(child, ast.Attribute):
                out.add(child.attr)
            elif (isinstance(child, ast.Constant) and isinstance(child.value, str)
                  and id(child) not in prose):
                out.add(child.value)
        return out

    names_chi = {name for name, node in functions.items()
                 if any(needle in symbol.lower() for symbol in symbols(node)
                        for needle in CHI_NEEDLES)}
    calls = {name: {called for called in
                    (child.func.id if isinstance(child.func, ast.Name)
                     else getattr(child.func, "attr", None)
                     for child in ast.walk(node) if isinstance(child, ast.Call))
                    if called in functions}
             for name, node in functions.items()}

    def closure(root):
        seen, stack = set(), [root]
        while stack:
            current = stack.pop()
            if current in seen or current not in functions:
                continue
            seen.add(current)
            stack.extend(calls.get(current, ()))
        return seen

    return names_chi, closure


def test_the_curl_call_closure_names_no_chi_and_update_E_s_does():
    """THE READING THE ADMISSION RESTS ON, re-performed rather than quoted.

    BOTH HALVES ARE THE TEST. The negative -- neither curl sub-step's transitive
    call closure names a chi2/chi3/nonlinear symbol -- is the claim. The POSITIVE
    CONTROL is what makes the negative mean anything: the same scan must find them
    inside ``update_E``'s closure, or it is a scan that finds nothing anywhere and
    reports an empty intersection for everything it is asked about.
    """
    names_chi, closure = _stepping_chi_scan()
    assert names_chi, "the scan found no chi symbol anywhere in stepping.py"
    for root in ("step_B", "step_D"):
        touched = sorted(closure(root) & names_chi)
        assert not touched, (
            f"{root}'s call closure now names {touched}; the array path's curl CAN "
            f"see the nonlinearity and coverage.CURL_NONLINEAR_ADMISSION is false")
    control = sorted(closure("update_E") & names_chi)
    assert control, (
        "update_E's closure names no chi symbol, so this scan cannot find one "
        "anywhere and its silence about the curl says nothing")
    record = coverage.CURL_NONLINEAR_ADMISSION
    assert sorted(record["stepping_functions_naming_chi"]) == sorted(names_chi), (
        f"the record names {record['stepping_functions_naming_chi']} and the live "
        f"scan finds {sorted(names_chi)}")
    assert tuple(record["chi_in_step_B_closure"]) == ()
    assert tuple(record["chi_in_step_D_closure"]) == ()


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_chi_grid_is_admitted_by_the_curl_and_refused_by_the_constitutive_pair(
        sub_step, xp):
    """The two families stay DISJOINT on a grid carrying a nonlinearity.

    The curl admits it (the Pade factor is not this sub-step's), the certified
    constitutive pair refuses it by name, and the nonlinear constitutive family is
    the one that serves those two sub-steps. Two families on one slot is what the
    union census reports as a FINDING, so the widening is worth re-measuring here
    rather than assuming an E-side clause the chi never touched still fires.
    """
    from . import nonlinear_constitutive  # noqa: PLC0415

    fields, layer, grid = configuration("chi3", xp)
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    assert covered, reason
    for side in ("H", "E"):
        plain = coverage.covers_real_pml_constitutive(fields, layer, grid, side)
        assert not plain[0] and "chi2/chi3" in plain[1], plain
        nonlinear = nonlinear_constitutive.covers_real_pml_nonlinear_constitutive(
            fields, layer, grid, side)
        assert not (plain[0] and nonlinear[0]), (side, plain, nonlinear)


def test_the_chi_admission_is_the_run_that_produced_it():
    """The record's numbers must be the artifact's, and the verdict must flip.

    ``results/`` is gitignored, so this is a declared skip where the bytes are
    absent. Where they are present, the four clauses that make this a MEASUREMENT
    rather than a definition are re-read: identity under both policies, the armed
    premise caught, a non-empty LIVE-chi arm on each sub-step, and a falsification
    run whose release verdict came out False.
    """
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    record = coverage.CURL_NONLINEAR_ADMISSION
    root = pathlib.Path(__file__).resolve().parents[2] / record["artifacts"]
    if not root.is_dir():
        pytest.skip(f"{record['artifacts']} is not on this checkout (gitignored)")
    for relative, digest in record["artifact_sha256"].items():
        path = root / relative
        assert path.is_file(), f"{relative} is missing from {root}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
            f"{relative} is not the bytes the record names")
    for relative, digest in record["subject_sha256"].items():
        subject = pathlib.Path(__file__).resolve().parents[2] / relative
        live = hashlib.sha256(subject.read_bytes()).hexdigest()
        if live == digest:
            continue
        # ONE HOME FOR THE RULE (device_identity.py:209): an edit that provably
        # cannot reach the compiled program leaves this admission describing the
        # kernel that ships. The helper returns False for anything it cannot
        # establish -- this record carries no device_sha256/code_sha256 today --
        # so the byte rule below still stands until those are backfilled.
        if weld_survives_edit(subject, record, relative):
            continue
        assert live == digest, f"{relative} has moved since the gate ran"
    for policy_leg in ("keep", "flush"):
        payload = json.loads((root / policy_leg / "gate.json").read_text(
            encoding="utf-8"))
        summary = payload["summary"]
        assert summary["released"], (policy_leg, summary["reasons"])
        assert summary["scored_cases"] == record["cases_per_policy"]
        assert summary["single_launch_identical"] == record["identical_single_launch"]
        assert (summary["multi_step_identical"]
                == record["identical_at_sixty_launches"])
        # THE NON-VACUITY CLAUSE: a live chi on each sub-step, or the identity is
        # about a grid whose nonlinearity does nothing.
        assert summary["chi_arms"]["chi_live"] == \
            record["cases_with_a_measurably_live_chi"], policy_leg
        assert summary["chi_live_by_sub_step"] == record["live_cases_per_sub_step"]
        assert summary["chi_words_moved_in_update_E"]["min_on_the_live_arm"] > 0
        # THE PREMISE, ARMED AND CAUGHT, on BOTH device strings.
        for sub_step in SUB_STEPS:
            leg = payload["source_mutations"][f"{sub_step}:chi_pade_scale_the_curl"]
            assert leg["verdict"] == "CAUGHT", (policy_leg, sub_step)
            assert leg["caught"] == leg["ran"] > 0
            assert leg["kernel_constructions_from_mutated_bytes"] > 0, (
                "no kernel was built from the mutated bytes, so this leg measured "
                "the SHIPPED kernel and its verdict is about nothing")
    falsified = json.loads((root / "falsify" / "gate.json").read_text(
        encoding="utf-8"))["summary"]
    assert falsified["released"] is False
    assert falsified["falsification"]["verdict_flipped"] is True
