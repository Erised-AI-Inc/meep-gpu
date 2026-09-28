"""``grid.beta`` under complex64 storage: its coefficient, its deltas, its behaviour.

WHAT IS PINNED HERE AND WHAT IS NOT. The BYTE-IDENTITY VERDICT needs a device and
lives in ``parity/meep_gpu/gate_cuda_complex_beta.py``. What a laptop can reach --
and every one of these is a plausible, smooth, WRONG field if it leaks -- is:

* THE COEFFICIENT, against ``stepping._special_kz_beta_term``'s own arithmetic on a
  real ``Grid``/``Fields`` pair, not against a restatement of it. Its real word is a
  SIGNED zero that Python's complex multiply produces and the kernel must be handed
  rather than synthesise, and that asymmetry is MEASURED here over the corpus's own
  betas rather than described.
* THE DELTAS, line by line against the FOLDED complex source this family transforms,
  so "the fold plus one statement per beta target" is a claim a diff checks.
* THE BEHAVIOUR, on beta grids folded and unfolded, against ``stepping.step_B`` /
  ``step_D`` as uint32 words -- with FIVE defects each shown to diverge, including
  the composition needle a fold makes reachable: the beta insert moved BELOW the
  masks, which is invisible in the interior and wrong on the mirror plane and on the
  far-ghost plane.
* THE PREDICATE, whose admissions must partition against the certified complex pair
  AND against ``complex_folded_kernels`` -- twelve of this family's sixteen census
  slots are folded, so the two are one wrong clause away from claiming the same
  rows.

WHAT THE NUMPY LEG CANNOT SEE, stated rather than papered over: it compiles nothing,
so no NVRTC contraction and no expansion arm; and on NumPy the complex multiply is
bitwise commutative, so the COEFFICIENT-LEFT orientation that ``FMA_V1`` makes
normative cannot be discriminated here at all. Those are the device gate's.
"""

from __future__ import annotations

import math

import numpy
import pytest

from .. import stepping
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import complex_beta_kernels as beta_family
from . import complex_emitter, complex_folded_kernels as folded, coverage
from .test_complex_folded import (  # the ONE transcription, not a second copy
    BC_METALLIC, BC_MIRROR_PERIODIC, CURL_ARRAYS, LICENCE, POLICY,
    _GridWithCupysName, boundary_codes_for, build, build_spec, differing_words,
    phase_arguments, restore, run_kernel_numpy, seed, snapshot, tables_for)

#: The corpus's own beta values, plus the two REAL-storage ones for contrast. The
#: signed-zero measurement below runs over all of them because the sign of the
#: PRODUCT ``sign * 2*pi*beta*dt`` is what decides the zero's sign, and a table
#: sampled only at positive beta would miss half the states.
CORPUS_BETAS = (0.2, -0.6850526103319672, -0.9120991827764708,
                -0.39073112848927377, 0.3321611318837033)

#: Beta fixtures. ``Grid._resolve_beta`` admits an effective 2-D Cartesian grid and
#: nothing else, so every one of these is nz == 1 -- which is not a choice, and is
#: why this family's z ghost rule is exercised at one cell only.
BETA_SPECS = (
    # TestSpecialKz.test_special_kz's class: unfolded, an in-plane kx, negative beta.
    {"label": "beta_plain_kx", "cell": (16.0, 8.0, 0.0), "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "k": (0.31, 0.0, 0.0),
     "beta": -0.39073112848927377},
    {"label": "beta_plain_metallic_x", "cell": (16.0, 8.0, 0.0), "axes": "",
     "phase": 1, "boundaries": ("metallic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0), "beta": 0.3321611318837033},
    # TestSpecialKz.test_eigsrc_kz__idx0's class: Mirror(Y) PERIODIC, k = 0.
    {"label": "beta_fold_Y_periodic", "cell": (8.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0), "beta": 0.2},
    # TestEigCoeffs.test_binary_grating_special_kz's class: Mirror(Y) PERIODIC with
    # an IN-PLANE kx on the PML'd X axis, negative beta. The one configuration where
    # the fold, the Bloch rotation and the beta insert are all live at once.
    {"label": "beta_fold_Y_periodic_kx", "cell": (8.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.27, 0.0, 0.0), "beta": -0.6850526103319672},
    # The OTHER half of the fold split, which no corpus row carries: a family must
    # not release on the half it happens to need.
    {"label": "beta_fold_Y_metallic", "cell": (8.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "metallic", "periodic"),
     "k": (0.0, 0.0, 0.0), "beta": 0.2},
    # A fold on the SLOWEST-STRIDE axis, and an odd full count on it.
    {"label": "beta_fold_X_periodic_odd_count", "cell": (15.0, 8.0, 0.0),
     "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "k": (0.0, 0.0, 0.0),
     "beta": -0.9120991827764708},
)


def build_beta(spec, courant=0.5):
    return build_spec(spec, courant=courant, beta=spec["beta"])


def coefficients_for(sub_step, grid):
    return beta_family.beta_curl_coefficients(
        grid.beta, grid.dt, beta_family.MAGNETIC[sub_step])


# ---------------------------------------------------------------------------
# THE COEFFICIENT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", CORPUS_BETAS)
@pytest.mark.parametrize("magnetic", (True, False))
def test_the_coefficient_is_steppings_own_word_for_word(beta, magnetic):
    """Against ``stepping._special_kz_beta_term`` on a real grid, not a restatement.

    The oracle returns ``-(complex64(coefficient) * partner)`` for a whole volume,
    so this multiplies by the SAME partner and compares uint32 words: that catches a
    wrong sign, a wrong rounding, a missing ``+-1j`` and a lost signed zero in one
    comparison, where checking the scalar alone would only catch the first three.
    """
    fields, _layer, grid = build(
        (16.0, 8.0, 0.0), ("periodic",) * 3, (), dimensions=2, beta=beta)
    rng = numpy.random.default_rng(20260820)
    partner = (rng.uniform(-1.0, 1.0, size=grid.shape)
               + 1j * rng.uniform(-1.0, 1.0, size=grid.shape)).astype(numpy.complex64)
    for index, sign in enumerate((1.0, -1.0)):
        oracle = stepping._special_kz_beta_term(fields, partner, sign,
                                                magnetic=magnetic)
        word = beta_family.beta_curl_coefficients(
            grid.beta, grid.dt, magnetic)[index]
        mine = -(numpy.complex64(complex(*word)) * partner)
        assert oracle.dtype == mine.dtype == numpy.complex64
        assert numpy.array_equal(
            numpy.ascontiguousarray(oracle).view(numpy.float32).view(numpy.uint32),
            numpy.ascontiguousarray(mine).view(numpy.float32).view(numpy.uint32)), (
            beta, magnetic, sign)


def test_the_real_word_is_a_signed_zero_exactly_where_python_puts_one():
    """MEASURED, and it is why the word is passed through instead of written as 0.0f.

    ``coefficient * 1j`` is ``(coefficient*0.0 - 0.0*1.0, coefficient*1.0)``, so the
    REAL word is ``-0.0`` exactly when ``coefficient`` is negative AND the ``+1j``
    (magnetic) branch is taken, and ``+0.0`` in every other combination. This walks
    the corpus betas at two courants and asserts the whole table, because a kernel
    that synthesised the zero would be right on three quarters of the states and
    wrong on the rest -- and ``fma(x, y, -0.0f)`` differs from ``fma(x, y, +0.0f)``
    wherever ``x*y`` is exactly zero, which a thin absorber reaches from all-zero
    state.
    """
    negative_zeros = 0
    total = 0
    for beta in CORPUS_BETAS:
        for dt in (0.0125, 0.00875):
            for magnetic in (True, False):
                pair = beta_family.beta_curl_coefficients(beta, dt, magnetic)
                for index, sign in enumerate((1.0, -1.0)):
                    coefficient = sign * 2.0 * math.pi * beta * dt
                    real = numpy.float32(pair[index][0])
                    bits = int(numpy.float32(real).view(numpy.uint32))
                    expected = 0x80000000 if (coefficient < 0.0 and magnetic) else 0
                    assert bits == expected, (beta, dt, magnetic, sign, hex(bits))
                    negative_zeros += int(bits == 0x80000000)
                    total += 1
    assert total == 40
    assert negative_zeros == 10, (
        "the negative-zero states vanished from the table; the measurement this "
        "clause rests on no longer holds and the pass-through may be pointless")


def test_the_imaginary_word_carries_the_whole_magnitude_and_both_signs():
    """The ``+-1j`` is not decoration: swapping it swaps which sub-step gets which.

    ``step_B`` takes ``+1j`` and ``step_D`` ``-1j`` (stepping.py:798-799), so the two
    sub-steps' coefficients are exact negatives of each other. A kernel that used one
    for both would be a plausible run with the out-of-plane coupling reversed on half
    the fields.
    """
    magnetic = beta_family.beta_curl_coefficients(0.2, 0.0125, True)
    electric = beta_family.beta_curl_coefficients(0.2, 0.0125, False)
    for index in (0, 1):
        assert magnetic[index][1] == -electric[index][1] != 0.0
    assert magnetic[0][1] == -magnetic[1][1]
    assert beta_family.MAGNETIC == {"step_B": True, "step_D": False}


# ---------------------------------------------------------------------------
# THE DELTAS
# ---------------------------------------------------------------------------

def test_the_only_difference_from_the_folded_source_is_the_deltas():
    """This family IS the folded kernel plus one statement per beta target."""
    for sub_step in ("step_B", "step_D"):
        for arm in sorted(complex_emitter.EXPANSIONS):
            base = folded.folded_source(sub_step, arm).split("\n")
            new = beta_family.beta_source(sub_step, arm).split("\n")
            added_ok, removed_ok = set(), set()
            for _label, old, replacement, _sites in beta_family._deltas_for(sub_step):
                added_ok.update(replacement.split("\n"))
                removed_ok.update(old.split("\n"))
            added = [line for line in new if line not in base]
            removed = [line for line in base if line not in new]
            assert all(line in added_ok for line in added), (
                sub_step, arm, [line for line in added if line not in added_ok])
            assert all(line in removed_ok for line in removed), (
                sub_step, arm, [line for line in removed if line not in removed_ok])
            assert added, (sub_step, arm)


def test_a_moved_anchor_fails_the_transform_instead_of_dropping_a_beta_term():
    """A site count is a WELD. Two beta targets, and a kernel missing one compiles."""
    original = folded.folded_source
    for label, old, _new, _sites in beta_family._deltas_for("step_D"):
        def broken(sub_step, expansion, _old=old):
            return original(sub_step, expansion).replace(_old, "/* moved */")

        beta_family.reset_kernel_sources()
        folded.folded_source = broken
        try:
            with pytest.raises(RuntimeError, match=r"matched 0 sites"):
                beta_family.beta_source("step_D", "FMA_V1")
        finally:
            folded.folded_source = original
        assert label


def test_the_beta_statement_lands_on_two_targets_and_takes_the_centre_register():
    """Two terms, the right registers, and NOTHING on the third component.

    MEEP's ``cc`` loop runs over ``d_c`` in {X, Y} only (step_db.cpp:148-176), which
    is why Bz and Dz get no term at all -- and why a kernel that added one to all
    three would look entirely plausible. The registers are ``f_2`` for target 0 and
    ``f_1`` for target 1, the UNSHIFTED centre operands the stencil already loaded;
    an ``sf``/``ss`` there is the partner defect.
    """
    for sub_step in ("step_B", "step_D"):
        source = beta_family.beta_source(sub_step, "FMA_V1")
        assert source.count("curl = cf_sub(curl, mul_imag_coefficient_left(") == 2
        assert "mul_imag_coefficient_left(bp, f_2)" in source
        assert "mul_imag_coefficient_left(bm, f_1)" in source
        assert "mul_imag_coefficient_left(bp, sf" not in source
        assert "mul_imag_coefficient_left(bm, ss" not in source
        # The third target's block must be untouched: split at its own comment.
        third = source[source.index("// Target 2:"):]
        assert "mul_imag_coefficient_left" not in third


def test_the_beta_statement_precedes_every_mask_line_in_both_kernels():
    """Position, read out of the emitted text -- the composition needle's other half.

    The array path adds the increment BEFORE ``_mask_non_owned_cells`` (:361/:363
    then :369; :443/:445 then :450). Under a fold that mask has two arms and both
    reach a beta target, so an insert below them leaves a live increment on the
    mirror plane and on the far-ghost plane. The behaviour leg measures that; this
    reads the ordering directly, so the two cannot both be satisfied by an accident
    of one fixture.
    """
    for sub_step in ("step_B", "step_D"):
        source = beta_family.beta_source(sub_step, "FMA_V1")
        for target in ("// Target 0:", "// Target 1:"):
            block = source[source.index(target):]
            block = block[:block.index("pml_apply(")]
            insert = block.index("mul_imag_coefficient_left(")
            first_mask = block.index("curl = cf_zero();")
            assert insert < first_mask, (sub_step, target)


def test_the_coefficient_left_product_is_a_name_over_the_arms_own_body():
    """One arithmetic, not two: the helper is a single call and has no arm branch.

    ``rotate_field_left`` already fuses the FIRST operand's product under FMA_V1 and
    rounds both separately under NAIVE, and "left" means "first operand" in every
    orientation this family carries. A second body would be a second spelling of one
    rule with two arms to keep in step by hand -- so the emitted helper is pinned to
    its single statement, and pinned to be IDENTICAL under both arms.
    """
    bodies = set()
    for arm in sorted(complex_emitter.EXPANSIONS):
        source = beta_family.beta_source("step_B", arm)
        start = source.index("__device__ __forceinline__ cf mul_imag_coefficient_left")
        body = source[start:source.index("\n}\n", start) + 3]
        bodies.add(body)
        assert body.count("return rotate_field_left(c, z);") == 1
        assert "__fmaf_rn" not in body and "*" not in body.split("{")[1]
    assert len(bodies) == 1, "the helper grew an arm branch"


def test_the_signature_carries_four_beta_words_and_the_folded_arguments():
    """Appended, never inserted: the argument list is the folded one plus four."""
    for sub_step in ("step_B", "step_D"):
        source = beta_family.beta_source(sub_step, "FMA_V1")
        folded_source = folded.folded_source(sub_step, "FMA_V1")
        start = source.index(f"void {beta_family.BETA_KERNELS[sub_step]}(")
        signature = source[start:source.index(") {", start)]
        base_start = folded_source.index(
            f"void {folded.FOLDED_KERNELS[sub_step]}(")
        base = folded_source[base_start:folded_source.index(") {", base_start)]
        base_arguments = base.split("(", 1)[1]
        assert base_arguments.rstrip().rstrip(",") in signature.replace(
            "float pzr, float pzi\n", "float pzr, float pzi,\n")
        assert "float bp_re, float bp_im, float bm_re, float bm_im" in signature
        assert beta_family.BETA_KERNELS[sub_step] in source
        assert folded.FOLDED_KERNELS[sub_step] not in source


# ---------------------------------------------------------------------------
# THE BEHAVIOUR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", BETA_SPECS, ids=lambda s: s["label"])
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_complex_beta_curl_reproduces_stepping_word_for_word(spec, sub_step):
    """THE leg. Transcribed device tree against ``stepping`` on a beta grid.

    NON-VACUITY FLOORS: the array path must move state, the absorber must not be the
    identity, and the BETA TERM ITSELF must reach an output word -- the last checked
    by re-running with a zero coefficient and requiring the answers to differ. On a
    beta grid whose coupling happened to cancel, the case would measure the certified
    complex curl and report it as a beta verdict.
    """
    fields, layer, grid = build_beta(spec)
    rng = numpy.random.default_rng(20260823)
    arrays = (tuple(CURL_ARRAYS[sub_step]["targets"])
              + tuple(CURL_ARRAYS[sub_step]["aux"])
              + tuple(CURL_ARRAYS[sub_step]["sources"]))
    outputs = (tuple(CURL_ARRAYS[sub_step]["targets"])
               + tuple(CURL_ARRAYS[sub_step]["aux"]))
    seed(fields, grid, arrays, rng)
    frozen = snapshot(fields, arrays)

    profile = numpy.concatenate([
        numpy.asarray(getattr(layer, f"{stem}_{axis}{suffix}")).ravel()
        for axis in "xyz" for stem in ("kms", "sinv") for suffix in ("", "_h")])
    assert float(numpy.max(numpy.abs(profile - 1.0))) > 1e-3, (
        "the absorber is the identity; this case cannot see a coefficient-index "
        "error")

    getattr(stepping, sub_step)(fields, layer)
    oracle = snapshot(fields, arrays)
    assert differing_words(frozen, oracle, outputs) > 0, "the array path moved nothing"

    codes = boundary_codes_for(grid)
    flags, phases = phase_arguments(grid, CURL_ARRAYS[sub_step]["backward"])
    tables = tables_for(sub_step, layer)
    dtdx = grid.dt / grid.dx

    restore(fields, frozen)
    run_kernel_numpy(sub_step, fields, tables, codes, dtdx, flags, phases,
                     beta=coefficients_for(sub_step, grid))
    mine = snapshot(fields, arrays)
    assert differing_words(oracle, mine, outputs) == 0, (
        f"{spec['label']} {sub_step}: "
        f"{differing_words(oracle, mine, outputs)} words differ")

    restore(fields, frozen)
    run_kernel_numpy(sub_step, fields, tables, codes, dtdx, flags, phases,
                     beta=((0.0, 0.0), (0.0, 0.0)))
    without = snapshot(fields, arrays)
    assert differing_words(oracle, without, outputs) > 0, (
        f"{spec['label']} {sub_step}: zeroing the beta coefficient changed nothing, "
        f"so this case measures the certified complex curl and not the beta insert")


#: THE FIVE DEFECTS, and what each one is. Every entry must CHANGE an output word on
#: at least one sub-step of its fixture, or the branch it targets is untested.
BETA_DEFECTS = ("dropped", "signs_swapped", "added_not_subtracted",
                "after_mask", "partner_shifted")


def _mutate(defect, sub_step, grid, kwargs):
    plus, minus = coefficients_for(sub_step, grid)
    if defect == "dropped":
        return dict(kwargs, beta=((0.0, 0.0), (0.0, 0.0)))
    if defect == "signs_swapped":
        return dict(kwargs, beta=(minus, plus))
    if defect == "added_not_subtracted":
        # curl + (c*g) instead of curl - (c*g): the same magnitude, the wrong side.
        return dict(kwargs, beta=((-plus[0], -plus[1]), (-minus[0], -minus[1])))
    if defect == "after_mask":
        return dict(kwargs, beta=(plus, minus), beta_after_mask=True)
    if defect == "partner_shifted":
        return dict(kwargs, beta=(plus, minus), beta_partner_shifted=True)
    raise AssertionError(defect)


@pytest.mark.parametrize("defect", BETA_DEFECTS)
def test_each_beta_defect_changes_an_output_word_somewhere(defect):
    """A branch never shown to change an answer has been executed, not tested.

    ``after_mask`` is the one a FOLD makes reachable and an unfolded grid cannot see
    at all -- on an all-periodic beta grid there is no mask on the beta targets, so
    the insert's position is genuinely free there. That is why the sweep runs every
    fixture and asks only that SOME fixture catches each defect, and why the fold
    fixtures are in the list.

    ``partner_shifted`` is the one the REAL special_kz family measured as a
    structural NULL: every beta partner's only shifted companion is the neighbour
    along Z, and every beta grid has nz == 1, so the two are the same word. It is
    kept here rather than deleted, and the assertion below records which way it
    came out on THIS family's fixtures.
    """
    caught = {}
    for spec in BETA_SPECS:
        for sub_step in ("step_B", "step_D"):
            fields, layer, grid = build_beta(spec)
            rng = numpy.random.default_rng(20260824)
            arrays = (tuple(CURL_ARRAYS[sub_step]["targets"])
                      + tuple(CURL_ARRAYS[sub_step]["aux"])
                      + tuple(CURL_ARRAYS[sub_step]["sources"]))
            outputs = (tuple(CURL_ARRAYS[sub_step]["targets"])
                       + tuple(CURL_ARRAYS[sub_step]["aux"]))
            seed(fields, grid, arrays, rng)
            frozen = snapshot(fields, arrays)
            getattr(stepping, sub_step)(fields, layer)
            oracle = snapshot(fields, arrays)
            restore(fields, frozen)
            codes = boundary_codes_for(grid)
            flags, phases = phase_arguments(grid, CURL_ARRAYS[sub_step]["backward"])
            kwargs = _mutate(defect, sub_step, grid, {})
            run_kernel_numpy(sub_step, fields, tables_for(sub_step, layer), codes,
                             grid.dt / grid.dx, flags, phases, **kwargs)
            mine = snapshot(fields, arrays)
            caught[(spec["label"], sub_step)] = differing_words(oracle, mine, outputs)
    live = {key: value for key, value in caught.items() if value}
    if defect == "partner_shifted":
        # A MEASURED NULL, recorded rather than deleted: on nz == 1 the shifted
        # companion of every beta partner IS the partner, so this defect cannot be
        # seen by any grid beta is legal on. The REAL special_kz family measured the
        # same null (0/12 CAUGHT across both policies) and kept it for the same
        # reason. If it ever becomes detectable, this assertion is what says so.
        assert not live, (
            f"partner_shifted became detectable on {sorted(live)}; the structural "
            f"argument (every beta grid has nz == 1) no longer holds and this "
            f"family needs a real partner mutation")
        return
    assert live, f"{defect} changed no output word on any of {len(caught)} cases"
    if defect != "after_mask":
        assert len(live) == len(caught), (
            f"{defect} was invisible on {sorted(set(caught) - set(live))}; a defect "
            f"in the term itself must reach a word on every beta fixture")
        return
    # AFTER_MASK IS THE COMPOSITION NEEDLE, and how many words it moves is decided
    # by HOW MANY MASK ARMS reach a beta target. Measured on these fixtures:
    #   all-periodic unfolded : 0   -- no mask on a beta target, the position is free
    #   one non-PERIODIC code : 64  -- the cell-0 arm only (metallic, or a folded
    #                                 METALLIC axis, which inherits that same arm)
    #   a folded PERIODIC axis: 128 -- BOTH arms, cell 0 and the top plane
    # The 0/64/128 ladder IS the claim that a fold doubles this defect's reach, so
    # it is asserted rather than summarised as "some fixture caught it".
    for (label, sub_step), moved in sorted(caught.items()):
        spec = next(s for s in BETA_SPECS if s["label"] == label)
        codes = boundary_codes_for(build_beta(spec)[2])
        folded_periodic = BC_MIRROR_PERIODIC in codes
        walled = any(code != 0 for code in codes)
        if not walled:
            expected = 0
        elif folded_periodic:
            expected = 64
        else:
            expected = 32
        assert moved == expected, (
            label, sub_step, f"moved {moved} words, expected {expected} for "
            f"codes={codes}")


# ---------------------------------------------------------------------------
# THE PREDICATE
# ---------------------------------------------------------------------------

def _ask(predicate, fields, layer, grid, key):
    return predicate(fields, layer, grid, key, license=LICENCE,
                     subnormal_policy=POLICY)


@pytest.mark.parametrize("spec", BETA_SPECS, ids=lambda s: s["label"])
def test_the_three_complex_families_partition_a_beta_row(spec):
    """Exactly one arm may serve each slot of a beta row -- folded or not.

    TWELVE OF THIS FAMILY'S SIXTEEN CENSUS SLOTS ARE ALSO FOLDED, so the beta family
    and the folded family are one wrong clause away from claiming the same rows,
    which the union census reports as a FINDING rather than as coverage.
    """
    fields, layer, grid = build_beta(spec)
    proxy = _GridWithCupysName(grid)
    for sub_step in ("step_B", "step_D"):
        mine, reason = _ask(beta_family.covers_complex_beta_curl,
                            fields, layer, proxy, sub_step)
        assert mine, (spec["label"], sub_step, reason)
        base, base_reason = _ask(coverage.covers_real_pml_complex_curl,
                                 fields, layer, proxy, sub_step)
        assert not base, (spec["label"], sub_step)
        other, other_reason = _ask(folded.covers_complex_folded_curl,
                                   fields, layer, proxy, sub_step)
        assert not other, (spec["label"], sub_step,
                           "the folded family also admits this beta slot")
        assert "special_kz" in base_reason or "mirror symmetry" in base_reason
        # The folded family's refusal names WHY, and the why differs by fixture: a
        # FOLDED beta row is refused for beta (it inherits the certified predicate's
        # clause), an UNFOLDED one for having no fold at all. Both are partitions;
        # asserting one string for both would be asserting the wrong thing on half
        # the fixtures.
        assert ("special_kz" in other_reason if spec["axes"]
                else "no mirror fold is active" in other_reason), other_reason
    for side in ("H", "E"):
        mine, reason = _ask(beta_family.covers_complex_beta_constitutive,
                            fields, layer, proxy, side)
        assert mine, (spec["label"], side, reason)
        base, _ = _ask(coverage.covers_real_pml_complex_constitutive,
                       fields, layer, proxy, side)
        assert not base, (spec["label"], side)
        other, _ = _ask(folded.covers_complex_folded_constitutive,
                        fields, layer, proxy, side)
        assert not other, (spec["label"], side)


def test_a_beta_free_complex_run_is_refused_by_name():
    """The other half of the partition: this family may not take a beta = 0 row."""
    fields, layer, grid = build(
        (12.0, 8.0, 0.0), ("periodic",) * 3, (), k_point=(0.21, 0.0, 0.0),
        dimensions=2)
    proxy = _GridWithCupysName(grid)
    for sub_step in ("step_B", "step_D"):
        covered, reason = _ask(beta_family.covers_complex_beta_curl,
                               fields, layer, proxy, sub_step)
        assert not covered
        assert "grid.beta is zero" in reason


def test_a_real_storage_beta_run_is_handed_back_to_special_kz_by_name():
    """A refusal that NAMES the family that owns the configuration, not the storage.

    ``special_kz_curl`` is certified for exactly this row class. A refusal reading
    "float32" would leave a reader looking for a missing kernel; one that names the
    owner is the difference between a gap and a partition.
    """
    grid = Grid(resolution=1.0, cell_size=(16.0, 8.0, 0.0),
                boundaries=("periodic",) * 3, xp=numpy, courant=0.5,
                dimensions=2, beta=0.3321611318837033)
    layer = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    covered, reason = _ask(beta_family.covers_complex_beta_curl,
                           fields, layer, _GridWithCupysName(grid), "step_B")
    assert not covered
    assert "special_kz_curl" in reason


def test_a_beta_run_with_a_phase_on_the_folded_axis_is_refused():
    """The fold clause is ASKED, not re-derived: this family's kernels are the fold's.

    ``_beta_reasons`` delegates to ``complex_folded_kernels._fold_reasons`` on a
    folded grid, so a fold the folded family could not serve is one this cannot
    serve either -- and the two cannot drift apart, which they would if the clause
    were spelled twice.
    """
    from .test_complex_folded import _StubGrid

    grid = _StubGrid(mirrored=(False, True, False),
                     stored_past_owned=(False, True, False),
                     k_point=(0.0, 0.37, 0.0),
                     phases=(None, complex(0.5, 0.5), None))
    grid.beta = 0.2

    class _Complex:
        force_complex_fields = True

    reason = beta_family._beta_reasons(_Complex(), grid)
    assert reason is not None and "is folded AND carries k component" in reason


def test_a_cylindrical_or_three_dimensional_beta_grid_is_refused_by_name():
    """RESTATED, never inherited from ``Grid``'s constructor guard.

    A predicate that reasons "the constructor would have refused it" admits by
    argument from absence -- and these two clauses are the ones MEEP itself aborts
    on (fields.cpp:546-547), so they must be refusals with a line behind them.
    """
    from .test_complex_folded import _StubGrid

    for attribute, value, needle in (("cylindrical", True, "cylindrical (Dcyl)"),
                                     ("dimensions", 3, "dimensions=3")):
        grid = _StubGrid()
        grid.beta = 0.2
        setattr(grid, attribute, value)

        class _Complex:
            force_complex_fields = True

        reason = beta_family._beta_reasons(_Complex(), grid)
        assert reason is not None and needle in reason, (attribute, reason)


def test_the_admission_record_claims_nothing_it_has_not_measured():
    """A family record that says "passed" before its gate ran is worse than none.

    RELEASED ON DEVICE 2026-09-01, and the pins below moved WITH the release --
    the two-part-edit discipline this test exists for. Until that run this test
    pinned ``released`` False and empty ``artifacts``; now it pins the device
    leg's own numbers, the REPORTED platform finding, and the partition move
    that landed in the same edit.
    """
    record = beta_family.BETA_COMPLEX_ADMISSION
    assert record["released"] is True, (
        "released became True on the 2026-09-01 device leg; if it is False again "
        "the record was rolled back without rolling back this pin")
    assert record["artifacts"] == (
        "parity/meep_gpu/results/cuda_complex_beta_2026-09-01/keep/gate.json",
        "parity/meep_gpu/results/cuda_complex_beta_2026-09-01/flush/gate.json",
    )
    assert record["gate"].endswith("gate_cuda_complex_beta.py")
    assert any("special_kz_curl" in line
               for line in record["what_it_does_not_license"])
    # THE PARTITION MOVED WITH THE RELEASE, or "certified" is a word someone typed.
    assert beta_family.CERTIFIED_KERNELS == ("step_B_pml_complex_beta",
                                             "step_D_pml_complex_beta")
    assert beta_family.UNCERTIFIED_KERNELS == {}
    # THE DEVICE LEG'S NUMBERS MUST BE INTERNALLY CONSISTENT.
    device = record["device_leg"]
    assert (device["folded_cases"] + device["unfolded_cases"]
            == device["curl_cases_scored"]
            == device["curl_single_launch_identical"]
            == device["curl_multi_step_identical"] == 48)
    assert device["constitutive_cases_scored"] == {"H": 24, "E": 24}
    assert device["constitutive_identical"] == {"H": 24, "E": 24}, (
        "the certified complex constitutive pair's admission on a beta run is "
        "this family's to measure, and it must have measured it identical")
    assert (device["mutation_legs_caught"] + device["mutation_legs_null_confirmed"]
            + device["mutation_legs_reported"] == device["mutation_legs"])
    assert device["policies"] == ("ieee_keep_ftz_stripped", "meep_x86_flush")
    # THE ONE UNCATCHABLE LEG MUST STAY VISIBLE IN EVERY PLACE THAT SPEAKS OF IT.
    # On device it is REPORTED -- the leg's own contract calls an all-legs-uncaught
    # outcome a platform finding, and the reason (mul_imag_coefficient_left reads
    # the imaginary word alone) is recorded beside the verdict, not argued.
    assert device["synthesise_the_zero_real_word"].startswith("REPORTED")
    assert device["mutation_legs_reported"] == 1
    assert any("SIGNED ZERO real word is load-bearing" in line
               and "REPORTED" in line
               for line in record["what_it_does_not_license"]), (
        "the finding has to reach the does-not-license list in its measured form: "
        "nothing may cite the pass-through as load-bearing")
    # THE HOST LEG STAYS, as the transcription record it always was.
    host = record["host_leg"]
    assert host["artifact"].endswith("gate.json")
    assert (host["folded_cases"] + host["unfolded_cases"]
            == host["curl_cases_scored"] == host["curl_single_launch_identical"])
    assert (host["mutation_legs_caught"] + host["mutation_legs_null_confirmed"]
            + host["mutation_legs_uncaught"] == host["mutation_legs"])
    assert host["synthesise_the_zero_real_word"].startswith("UNCAUGHT")
    assert host["mutation_legs_uncaught"] == 1
    assert record["slots"] == 16 and record["rows"] == 4
