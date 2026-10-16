"""``grid.beta`` under complex64 storage: its coefficient, its deltas, its predicate.

WHAT IS PINNED HERE AND WHAT IS NOT. The BYTE-IDENTITY VERDICT needs a device and
lives in ``parity/meep_gpu/gate_cuda_complex_beta.py``. What a laptop can reach --
and every one of these is a plausible, smooth, WRONG field if it leaks -- is:

* THE COEFFICIENT, against ``stepping._special_kz_beta_term``'s own arithmetic on a
  real ``Grid``/``Fields`` pair, not against a restatement of it: one real float32
  word per call site, the D side's negated in double before its one rounding, and the
  increment ``-quarter_turn(c, g)`` formed part by part.
* THE DELTAS, line by line against the FOLDED complex source this family transforms,
  so "the fold plus MEEP's beta pass after targets 0 and 1" is a claim a diff checks.
* THE PREDICATE, whose admissions must partition against the certified complex pair
  AND against ``complex_folded_kernels`` -- twelve of this family's sixteen census
  slots are folded, so the two are one wrong clause away from claiming the same
  rows.

THE BEHAVIOUR -- the emitted text executed against ``stepping.step_B`` / ``step_D`` as
uint32 words on beta grids folded and unfolded, with planted defects each shown to
diverge -- is ``test_fold_complex_chunk_forms.py``'s: it runs the shipped device text
itself through the complex interpreter rather than a transcription of it.
"""

from __future__ import annotations

import math
import re

import numpy
import pytest

from .. import stepping
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import complex_beta_kernels as beta_family
from . import complex_folded_kernels as folded, coverage, folded_pass
from .test_complex_folded import (  # the ONE fixture builder, not a second copy
    _GridWithCupysName, build, build_spec)

#: The corpus's own beta values, plus the two REAL-storage ones for contrast, over
#: both signs: the D side's word is the B side's negated, and a table sampled only at
#: positive beta would see half the states.
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


# ---------------------------------------------------------------------------
# THE COEFFICIENT
# ---------------------------------------------------------------------------

def _beta_increment(word, partner):
    """The kernel's ``beta_increment(c, g)``, transcribed in float32 part by part:
    ``(c * g.im, -(c * g.re))`` -- ``-quarter_turn(c, g)`` with no complex product."""
    word = numpy.float32(word)
    out = numpy.empty(partner.shape, dtype=numpy.complex64)
    out.real = (word * partner.imag).astype(numpy.float32)
    out.imag = (-(word * partner.real)).astype(numpy.float32)
    return out


@pytest.mark.parametrize("beta", CORPUS_BETAS)
@pytest.mark.parametrize("magnetic", (True, False))
def test_the_coefficient_is_steppings_own_word_for_word(beta, magnetic):
    """Against ``stepping._special_kz_beta_term`` on a real grid, not a restatement.

    The oracle returns the negated quarter turn of the call site's coefficient for a
    whole volume, so this forms the kernel's increment from the SAME partner and
    compares uint32 words: that catches a wrong sign, a wrong rounding, the two parts
    exchanged and a lost signed zero in one comparison, where checking the scalar alone
    would only catch the first two.
    """
    fields, _layer, grid = build(
        (16.0, 8.0, 0.0), ("periodic",) * 3, (), dimensions=2, beta=beta)
    rng = numpy.random.default_rng(20260820)
    partner = numpy.empty(grid.shape, dtype=numpy.complex64)
    partner.real = rng.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32)
    partner.imag = rng.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32)
    # Signed zeros in both parts, so the increment's own zeros carry a sign to compare.
    partner.real[0, ::2] = numpy.float32(-0.0)
    partner.imag[1, ::3] = numpy.float32(-0.0)
    for index, sign in enumerate((1.0, -1.0)):
        oracle = stepping._special_kz_beta_term(fields, partner, sign,
                                                magnetic=magnetic)
        word = beta_family.beta_curl_coefficients(grid.beta, grid.dt, magnetic)[index]
        mine = _beta_increment(word, partner)
        assert oracle.dtype == mine.dtype == numpy.complex64
        assert numpy.array_equal(
            numpy.ascontiguousarray(oracle).view(numpy.float32).view(numpy.uint32),
            numpy.ascontiguousarray(mine).view(numpy.float32).view(numpy.uint32)), (
            beta, magnetic, sign)


def test_each_call_site_word_is_one_rounding_of_the_signed_double():
    """ONE float32 word per call site: ``sign * 2*pi*beta*dt`` in double, negated on the
    D side before the round (stepping's ``coefficient if magnetic else -coefficient``),
    rounded ONCE. No complex coefficient -- and so no signed-zero real word -- exists
    to be handed over."""
    total = 0
    for beta in CORPUS_BETAS:
        for dt in (0.0125, 0.00875):
            for magnetic in (True, False):
                pair = beta_family.beta_curl_coefficients(beta, dt, magnetic)
                assert len(pair) == 2
                for index, sign in enumerate((1.0, -1.0)):
                    coefficient = sign * 2.0 * math.pi * beta * dt
                    expected = numpy.float32(coefficient if magnetic else -coefficient)
                    word = numpy.float32(pair[index])
                    assert word.view(numpy.uint32) == expected.view(numpy.uint32), (
                        beta, dt, magnetic, sign)
                    total += 1
    assert total == 40


def test_the_sub_steps_and_the_call_sites_are_exact_negatives():
    """``step_B`` takes ``+1j`` and ``step_D`` ``-1j``, so the two sub-steps' words are
    exact negatives of each other; and the two call sites (sign +1 and -1) are too. A
    kernel that used one for both would be a plausible run with the out-of-plane
    coupling reversed on half the fields."""
    magnetic = beta_family.beta_curl_coefficients(0.2, 0.0125, True)
    electric = beta_family.beta_curl_coefficients(0.2, 0.0125, False)
    for index in (0, 1):
        assert magnetic[index] == -electric[index] != 0.0
    assert magnetic[0] == -magnetic[1]
    assert beta_family.MAGNETIC == {"step_B": True, "step_D": False}


# ---------------------------------------------------------------------------
# THE DELTAS
# ---------------------------------------------------------------------------

#: The lines one beta pass adds after its target's ``pml_apply`` call
#: (``folded_pass.complex_beta_insert``), as code: the ownership flag, the curl's own
#: mask lines as that flag, the increment and the ladder.
_PASS_LINES = (
    re.compile(r" *int owned_cell = 1;"),
    re.compile(r" *if \(.*\) owned_cell = 0;"),
    re.compile(r" *cf inc_beta = owned_cell \? beta_increment\(b[pm], f_[12]\) "
               r": cf_zero\(\);"),
    re.compile(r" *cfold_apply\(f[01], u[01], idx, inc_beta, .*\);"),
    re.compile(r" *//.*"),
)


def test_the_only_difference_from_the_folded_source_is_the_deltas():
    """This family IS the folded kernel plus MEEP's beta pass after targets 0 and 1."""
    helpers = set(folded_pass.COMPLEX_FOLD_HELPERS.split("\n"))
    for sub_step in ("step_B", "step_D"):
        base = folded.folded_source(sub_step).split("\n")
        new = beta_family.beta_source(sub_step).split("\n")
        renamed = (f'extern "C" __global__ void '
                   f'{beta_family.BETA_KERNELS[sub_step]}(')
        added = [line for line in new if line not in base]
        removed = [line for line in base if line not in new]
        unexplained = [line for line in added
                       if line not in helpers and line != renamed
                       and line != "    float bp, float bm,"
                       and not any(p.fullmatch(line) for p in _PASS_LINES)]
        assert not unexplained, (sub_step, unexplained)
        assert removed == [f'extern "C" __global__ void '
                           f'{folded.FOLDED_KERNELS[sub_step]}('], sub_step
        assert "    float bp, float bm," in added


@pytest.mark.parametrize("anchor,moved", [
    ("    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,\n"
     "    const int* __restrict__ cb\n) {",
     "    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,\n"
     "    const int* cb\n) {"),
    ("        cf f_2 = cwindow(g1, idx, k, ph_z, pz);\n",
     "        cf f_2 = cshift_up(g1, idx, k, nz, sz, bc_z);\n"),
])
def test_a_moved_anchor_fails_the_transform_instead_of_dropping_a_beta_pass(anchor,
                                                                           moved):
    """A site count is a WELD. Two beta targets, and a kernel missing one compiles."""
    base = folded.folded_source("step_B")
    assert base.count(anchor) == 1
    with pytest.raises((RuntimeError, AssertionError)):
        beta_family.beta_source("step_B",
                                folded=base.replace(anchor, moved, 1))


def test_the_beta_pass_lands_on_two_targets_and_takes_the_centre_register():
    """Two passes, the right registers, and NOTHING on the third component.

    MEEP's ``cc`` loop runs over ``d_c`` in {X, Y} only (step_db.cpp:161-175), which
    is why Bz and Dz get no pass at all -- and why a kernel that added one to all
    three would look entirely plausible. The registers are ``f_2`` for target 0 and
    ``f_1`` for target 1, the UNSHIFTED centre operands the stencil already loaded;
    an ``sf``/``ss`` there is the partner defect.
    """
    for sub_step in ("step_B", "step_D"):
        source = beta_family.beta_source(sub_step)
        assert source.count("cf inc_beta = owned_cell ? beta_increment(") == 2
        assert "beta_increment(bp, f_2)" in source
        assert "beta_increment(bm, f_1)" in source
        assert "beta_increment(bp, sf" not in source
        assert "beta_increment(bm, ss" not in source
        third = source[source.index("// Target 2:"):]
        assert "beta_increment(" not in third and "cfold_apply(" not in third


def test_the_beta_pass_follows_its_own_curl_and_is_guarded_to_owned_cells():
    """Position, read out of the emitted text. MEEP's pass runs after the curl
    (step_db.cpp:161-175) and visits owned cells only, so per target the curl's
    ``pml_apply`` comes first, then the guarded increment, then its ladder."""
    for sub_step in ("step_B", "step_D"):
        source = beta_family.beta_source(sub_step)
        for target in ("// Target 0:", "// Target 1:"):
            block = source[source.index(target):]
            block = block[:block.index("\n    }\n")]
            curl = block.index("pml_apply(")
            increment = block.index("cf inc_beta = owned_cell ? beta_increment(")
            ladder = block.index("cfold_apply(")
            assert curl < increment < ladder, (sub_step, target)


def test_the_beta_increment_is_one_part_wise_body():
    """One arithmetic: the helper forms ``-quarter_turn(c, g)`` from two real products
    and two sign flips, with no complex product."""
    source = beta_family.beta_source("step_B")
    start = source.index("__device__ __forceinline__ cf beta_increment(")
    body = source[start:source.index("\n}\n", start) + 3]
    assert "    o.re = c * g.im;\n    o.im = -(c * g.re);\n" in body
    assert "__fmaf_rn" not in body


def test_the_signature_carries_two_beta_words_before_the_chunk_record():
    """The argument list is the folded one with the two words inserted before ``cb``,
    which stays last."""
    for sub_step in ("step_B", "step_D"):
        source = beta_family.beta_source(sub_step)
        folded_source = folded.folded_source(sub_step)
        start = source.index(f"void {beta_family.BETA_KERNELS[sub_step]}(")
        signature = source[start:source.index(") {", start)]
        base_start = folded_source.index(f"void {folded.FOLDED_KERNELS[sub_step]}(")
        base = folded_source[base_start:folded_source.index(") {", base_start)]
        expected = base.split("(", 1)[1].replace(
            "    const int* __restrict__ cb\n",
            "    float bp, float bm,\n    const int* __restrict__ cb\n")
        assert signature.split("(", 1)[1] == expected
        assert beta_family.BETA_KERNELS[sub_step] in source
        assert folded.FOLDED_KERNELS[sub_step] not in source


# ---------------------------------------------------------------------------
# THE PREDICATE
# ---------------------------------------------------------------------------

def _ask(predicate, fields, layer, grid, key):
    return predicate(fields, layer, grid, key)


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
