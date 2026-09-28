"""Laptop tests for the Metal chi2/chi3 (Pade) ``update_E`` family.

THE MERGE BAR, NOT THE CERTIFICATION. These are what must stay green on every
change, and they are chosen for the defects a byte gate would catch LATE or not at
all — plus this family's OWN hazards, which are not the off-diagonal family's:

* **the divide.** This is the first Metal family in the package that divides, and
  ``shaders.py``'s transcription rules say in as many words that the certified pair
  was chosen partly BECAUSE it does not. The spelling is pinned here — plain ``/``,
  never ``fast::divide`` — against the measurement in
  ``parity/meep_gpu/probe_metal_divide_and_negation.py`` that licensed it;
* **the Pade groupings**, character for character against ``calc_nonlinear_u``;
* **the four-corner association**, which is the SHIFTED-PAIR one and not MEEP C's
  left-to-right sum, and whose two shifts go in OPPOSITE directions;
* **the magnitude ceiling**, which is a coverage refusal and not a tolerance — with
  a case on each side of it, because a ceiling only ever tested from one side is a
  constant rather than a clause;
* **the scalar/volume chi arms**, and the fact that a LINEAR component in a partly
  nonlinear run compiles to the certified plain body verbatim;
* the BINDING COUNT against Metal's measured 31-buffer ceiling — this family is
  four bindings from it, and the packed scalar buffer exists because of it;
* the registration, because a family that quietly changed slots or became unwired
  would change what ``plan_step`` launches.

Everything that needs a device is guarded and skipped, so this file is the merge
bar on a host with no MPS as well as on this one.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import (
    arms,
    coverage,
    nonlinear_update_e as nonlinear,
    registry,
    shaders,
    subnormal,
)
from meep_gpu.pml import PML


def _torch_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available())


needs_mps = pytest.mark.skipif(not _torch_mps(),
                               reason="no MPS device on this host")

ALL_LIVE = (1, 1, 1)
VOLUMES = (1, 1, 1)
PERIODIC3 = (shaders.PERIODIC,) * 3
METALLIC3 = (shaders.METALLIC,) * 3


# ---------------------------------------------------------------------------
# The constant tables, pinned against the engine's own
# ---------------------------------------------------------------------------

def test_e_terms_match_steppings_constitutive_order():
    """The kernel writes the components ``stepping`` writes, in its order, and each
    one's OWN axis is what the half-integer coefficient index uses."""
    assert tuple((t[0], t[1]) for t in nonlinear.E_TERMS) == tuple(
        (c, s) for c, s, _axis in stepping.E_CONSTITUTIVE_TERMS)
    for component, _source, axis in nonlinear.E_TERMS:
        assert stepping.AXIS_NAMES[axis] == component[1]


def test_transverse_partners_follow_meeps_cycle_direction():
    """``(own_axis + offset) % 3`` for offset 1 then 2 — stepping.py:1183-1185.

    Derived from ``stepping``'s own expression rather than compared against a copy
    of the table, so the two cannot drift together.
    """
    for index, (_component, _source, own_axis) in enumerate(nonlinear.E_TERMS):
        expected = tuple((own_axis + offset) % 3 for offset in (1, 2))
        assert nonlinear.TRANSVERSE_PARTNERS[index] == expected


def test_half_integer_is_the_e_side_sub_lattice():
    """stepping.py:1015 asks for ``half_integer=True`` on the E side. A swap is the
    silent half-cell absorber error, not a crash."""
    assert nonlinear.HALF_INTEGER is True


def test_the_binding_count_stays_under_metals_ceiling():
    """31 is a measured platform limit; a 32nd binding is a COMPILE ERROR.

    Counted off the emitted source rather than trusted from the constant, so an
    added buffer moves the measurement and not only the comment.
    """
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, PERIODIC3)
    indices = sorted(int(n) for n in re.findall(r"\[\[buffer\((\d+)\)\]\]", source))
    assert indices == list(range(len(indices))), indices
    assert len(indices) == nonlinear.BINDING_COUNT
    assert nonlinear.BINDING_COUNT <= MAX_BUFFER_BINDINGS


def test_the_packed_scalar_buffer_is_what_keeps_the_family_under_the_ceiling():
    """Six separate ``constant float&`` chi scalars would not compile at all.

    24 volume/coefficient buffers + 6 scalars + 4 shape scalars = 34 > 31. This is
    the design decision the platform limit FORCED, and the arithmetic is asserted
    so a future edit that "simplifies" the packing rediscovers the reason here.
    """
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    unpacked = nonlinear.BINDING_COUNT - 1 + nonlinear.CHI_SCALAR_SLOTS
    assert unpacked > MAX_BUFFER_BINDINGS, unpacked


# ---------------------------------------------------------------------------
# The emitter's arithmetic — this family's own hazards
# ---------------------------------------------------------------------------

def test_the_pade_quotient_is_the_plain_divide_and_never_the_fast_one():
    """MEASURED, NOT PREFERRED, and the measurement is the reason the plain
    operator is legal here at all.

    ``probe_metal_divide_and_negation.py`` on this host: plain ``/`` agrees with
    ``numpy.float32`` division on 0 differing normal-result words over five operand
    classes and over the whole Pade expression; ``fast::divide`` diverges by up to
    2 ulp on 102,578 words. The sibling Triton track carries ``tl.math.div_rn``
    because ITS plain ``/`` is ``div.full.f32`` — a fact about that compiler, not
    this one, and deliberately not ported.
    """
    source = nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, PERIODIC3)
    assert "fast::divide" not in source
    assert "precise::divide" not in source
    for component in range(3):
        assert f"float u{component} = num{component} / den{component};" in source


def test_the_pade_groupings_are_calc_nonlinear_us_own():
    """``calc_nonlinear_u`` (stepping.py:1054-1056) associates left to right::

        c2 = (di * chi2) * (chi1inv * chi1inv)
        c3 = (dsqr * chi3) * ((chi1inv * chi1inv) * chi1inv)
        u  = ((1 + c2) + 2*c3) / ((1 + 2*c2) + 3*c3)

    Every paren decides bits. Flattening any of them is a different float32 number,
    which is why they are pinned as text rather than described.
    """
    source = nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, PERIODIC3)
    for c in range(3):
        assert f"float us_sq{c} = us{c} * us{c};" in source
        assert f"float us_cu{c} = (us{c} * us{c}) * us{c};" in source
        assert f"float c2_{c} = (gs{c} * chi2_{c}) * us_sq{c};" in source
        assert f"float c3_{c} = (dsqr{c} * chi3_{c}) * us_cu{c};" in source
        assert f"float num{c} = (1.0f + c2_{c}) + 2.0f * c3_{c};" in source
        assert f"float den{c} = (1.0f + 2.0f * c2_{c}) + 3.0f * c3_{c};" in source
        # Row THEN scale — stepping.py:1107-1109, `row * calc_nonlinear_u(...)`.
        assert f"float src{c} = (gs{c} * us{c}) * u{c};" in source


def test_dsqr_scales_the_transverse_sum_of_squares_by_one_sixteenth_last():
    """``gs*gs + 0.0625*(g1s*g1s + g2s*g2s)`` — stepping.py:1147.

    ``0.0625 = (1/4)^2`` turns each UNNORMALIZED four-point sum back into a mean
    before it is squared. It scales the SUM and is applied last.
    """
    source = nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, PERIODIC3)
    for c in range(3):
        assert (f"float dsqr{c} = gs{c} * gs{c} + 0.0625f * "
                f"((sum_{c}0 * sum_{c}0) + (sum_{c}1 * sum_{c}1));") in source


def test_the_four_corner_sum_is_the_pair_association_not_meep_cs_left_to_right():
    """``(g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1])`` — the SHIFTED-PAIR association.

    ``stepping._nonlinear_transverse_sums`` (:1158-1162) forms the near pair, then
    adds the pair shifted; MEEP C sums left to right (step_generic.cpp:646-648).
    The byte arbiter is stepping.py, so a `near`/`far` split with one add between
    them is the shape, and a flattened four-term sum is a different number.
    """
    source = nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, PERIODIC3)
    for c in range(3):
        for offset in (0, 1):
            tag = f"{c}{offset}"
            assert f"float sum_{tag} = near_{tag} + far_{tag};" in source


@pytest.mark.parametrize("component", (0, 1, 2))
def test_the_two_shifts_go_in_opposite_directions(component):
    """Half a cell DOWN the partner's axis, half a cell UP the component's OWN axis.

    Taking both the same way round is the half-cell registration error that
    survives every scalar test (stepping.py:1160-1171). Checked by reading the
    emitted index expressions: the near term must carry the partner axis's DOWN
    name and the far term the own axis's UP name, and the corner must carry BOTH.
    """
    source = nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, METALLIC3)
    own_axis = nonlinear.E_TERMS[component][2]
    up_name = nonlinear._TWO_WAY_NAMES["xyz"[own_axis]][1]
    up_valid = nonlinear._TWO_WAY_NAMES["xyz"[own_axis]][3]
    for offset in (0, 1):
        partner_axis = nonlinear.TRANSVERSE_PARTNERS[component][offset]
        down_name = nonlinear._TWO_WAY_NAMES["xyz"[partner_axis]][0]
        down_valid = nonlinear._TWO_WAY_NAMES["xyz"[partner_axis]][2]
        tag = f"{component}{offset}"
        block = source.split(f"float near_{tag}")[1].split(f"float sum_{tag}")[0]
        # The near pair reads the PARTNER axis down, guarded by its own flag.
        assert f"({down_valid} ? g{partner_axis}[" in block
        assert down_name in block
        # The far pair reads the OWN axis up, and the corner ANDs the two flags.
        assert f"({up_valid} ? g{partner_axis}[" in block
        assert up_name in block
        assert f"({up_valid} && {down_valid})" in block
        # And the two axes are genuinely different, or "opposite directions" is
        # not a statement about anything.
        assert own_axis != partner_axis


def test_a_linear_component_emits_the_certified_plain_product():
    """MEEP's ``else if (u)`` branch (stepping.py:1100-1102) is ``gs * us``.

    That is the certified constitutive kernel's component body verbatim, which is
    what makes a partly nonlinear run's linear components byte-identical to it
    rather than merely close.
    """
    source = nonlinear.nonlinear_source((0, 0, 1), (0, 0, 1), (0, 0, 1), PERIODIC3)
    assert "float src0 = gs0 * us0;" in source
    assert "float src1 = gs1 * us1;" in source
    assert "float src2 = (gs2 * us2) * u2;" in source
    # No Pade machinery at all on the linear components.
    for dead in ("dsqr0", "dsqr1", "num0", "num1", "u0 =", "u1 ="):
        assert dead not in source, dead


@pytest.mark.parametrize("volume,expected", ((1, "q22[ii]"), (0, "chis[2]")),
                         ids=("volume", "scalar"))
def test_the_chi_arm_selects_the_volume_load_or_the_packed_scalar_slot(volume,
                                                                       expected):
    """One arm reads a per-cell volume, the other a slot in the packed buffer.

    Both are real: ``mp.Medium(chi3=...)`` usually produces a uniform float, and a
    materials map produces a volume. The scalar arm is admitted because a Python
    float meeting a float32 array is a weak scalar rounded ONCE to float32 —
    measured, 0 differing words over seven scalars spanning 1e-30 to 3.4e38.
    """
    source = nonlinear.nonlinear_source((0, 0, 1), (0, 0, volume), (0, 0, volume),
                                        PERIODIC3)
    assert f"float chi2_2 = {expected};" in source


def test_no_min_or_max_builtin_and_no_negation_is_emitted():
    """Metal's ``min``/``max`` return the SECOND operand when both are zeros and
    drop NaN, disagreeing with ``numpy.minimum``/``maximum`` (shaders.py:41-46).

    Negation is also absent: the Pade factor is multiplies, adds and one divide,
    and the tail SUBTRACTS rather than negating. Asserted so that a future edit
    which introduces either has to come back and re-measure it, rather than
    inheriting another backend's idiom.
    """
    source = nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, METALLIC3)
    body = source.split("uint idx [[thread_position_in_grid]])")[1]
    code = "\n".join(line for line in body.splitlines()
                     if not line.strip().startswith("//"))
    assert not re.search(r"\b(min|max|fmin|fmax)\s*\(", code)
    assert "-1.0f" not in code
    assert "0.0f -" not in code


def test_the_emitted_source_carries_the_contraction_directive_exactly_once():
    """Without it the compiler contracts the numerator and denominator sums into
    fmas BEFORE the divide runs — measured on this expression as 10,770 differing
    words. The directive has ONE spelling, in ``shaders.py``."""
    source = nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, PERIODIC3)
    assert source.count("pragma clang fp contract") == 1
    assert shaders.contraction_pragma(shaders.CONTRACT_OFF) in source


def test_no_substitution_placeholder_survives_in_any_specialisation():
    """A placeholder left in the source would compile to something, and a mutation
    leg would then report an uncaught defect it never planted."""
    for live, second, third, codes in nonlinear.specialisations()[::37]:
        source = nonlinear.nonlinear_source(live, second, third, codes)
        assert "__" not in source.replace("__CONTRACT__", ""), (live, codes)


# ---------------------------------------------------------------------------
# The emitter's refusals
# ---------------------------------------------------------------------------

def test_an_all_linear_specialisation_is_refused_by_name():
    """That configuration IS the certified plain constitutive kernel's, and
    emitting this source for it would overlap the two families."""
    with pytest.raises(ValueError, match="no component is nonlinear"):
        nonlinear.nonlinear_source((0, 0, 0), (0, 0, 0), (0, 0, 0), PERIODIC3)


def test_a_boundary_code_that_is_neither_periodic_nor_metallic_is_refused():
    with pytest.raises(ValueError, match="neither PERIODIC nor METALLIC"):
        nonlinear.nonlinear_source(ALL_LIVE, VOLUMES, VOLUMES, (0, 0, 7))


def test_a_chi_arm_flag_on_a_linear_component_is_canonicalised_away():
    """Two keys that compile the same string must BE the same key.

    Otherwise the source-keyed compile memo hides the disagreement while the
    fingerprint enumeration counts it twice, and the corpus digest stops meaning
    what it says.
    """
    left = nonlinear.nonlinear_source((0, 0, 1), (1, 1, 1), (1, 1, 1), PERIODIC3)
    right = nonlinear.nonlinear_source((0, 0, 1), (0, 0, 1), (0, 0, 1), PERIODIC3)
    assert left == right


def test_the_specialisation_corpus_is_the_size_the_family_claims():
    """5 arms per component (linear, or nonlinear x two chi flags), minus the
    all-linear triple this family refuses, times 8 boundary triples."""
    corpus = nonlinear.specialisations()
    assert len(corpus) == (5 ** 3 - 1) * 8 == 992
    assert len(set(corpus)) == len(corpus)


# ---------------------------------------------------------------------------
# Predicate fixtures
# ---------------------------------------------------------------------------

def _grid(boundaries="periodic"):
    return Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries=boundaries,
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)


def _fields(grid, chi2=0.1, chi3=0.2, components=("Ez",), inverse=0.5,
            scalar=False, complex_storage=False):
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    shape = grid.shape
    epsilon = np.full(shape, np.float32(1.0 / inverse), dtype=np.float32)
    inverse_volume = np.full(shape, np.float32(inverse), dtype=np.float32)
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse_volume for c in ("Ex", "Ey", "Ez")})

    def value(magnitude):
        return (np.float32(magnitude) if scalar
                else np.full(shape, np.float32(magnitude), dtype=np.float32))

    fields.set_nonlinear_volumes({c: value(chi2) for c in components},
                                 {c: value(chi3) for c in components})
    return fields


def _pml(grid):
    return PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))


class _Residency:
    """The minimum a predicate needs to see: something exposing ``mirror``."""

    def mirror(self, name, host, constant=False, dtype=None):  # pragma: no cover
        raise AssertionError("the predicate must not mirror anything")


def _verdict(fields, grid, residency=None):
    return nonlinear.nonlinear_constitutive_coverage(
        fields, _pml(grid), _Residency() if residency is None else residency)


# ---------------------------------------------------------------------------
# The predicate — the inverted clause, and the clauses that stay
# ---------------------------------------------------------------------------

def test_a_linear_run_is_refused_toward_the_plain_kernel():
    """THE INVERTED CLAUSE, in its own direction. MEEP DELETES the trivial pair
    (structure.cpp:822-826, fields.py:879-880), which is the only reason a
    zero-chi run reproduces the linear engine bit for bit."""
    grid = _grid()
    fields = _fields(grid, chi2=0.0, chi3=0.0)
    verdict = _verdict(fields, grid)
    assert not verdict.covered
    assert any("no chi2/chi3 is installed" in r for r in verdict.reasons), \
        verdict.reasons


def test_a_nonlinear_run_is_admitted(monkeypatch):
    """The other direction of the same clause — and a predicate that refused
    everything would pass every refusal test above while carrying nothing.

    THE POLICY IS SET HERE RATHER THAN INHERITED. Every test that asserts an
    ADMISSION needs the flush policy, because ``_metal_backend_reasons`` refuses
    outright under any other — and on this arm64 host the process default resolves
    to ``keep`` (``match_meep`` measures MEEP, whose ``set_zero_subnormals`` is a
    no-op behind ``#if HAVE_IMMINTRIN_H``). Without this line the test passes only
    when some earlier test in the same process happened to leave the variable set,
    which is a pass that depends on collection order rather than on the predicate.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    grid = _grid()
    verdict = _verdict(_fields(grid), grid)
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("kwargs,needle", (
    ({"complex_storage": True}, "DOCMP split"),
), ids=("complex_storage",))
def test_the_clauses_that_keep_the_uncarried_rows_uncarried(kwargs, needle):
    """Clause 2 stays UN-inverted, and the row it refuses is named in UNCARRIED.

    A row that leaves ``UNCARRIED`` must do so in the change that makes its family
    real; these are the ones that did NOT, and this is where their refusal is
    pinned so "uncarried" stays a measured claim.
    """
    grid = _grid()
    fields = _fields(grid, **kwargs)
    verdict = _verdict(fields, grid)
    assert not verdict.covered
    assert any(needle in r for r in verdict.reasons), verdict.reasons


def test_an_offdiagonal_row_is_refused_because_meeps_general_case_is_a_third_body():
    """MEEP scales the WHOLE row product by the Pade factor (stepping.py:1095-1099).

    ``offdiag_update_e`` refuses a nonlinearity through the shared clause 10, so a
    run carrying both is uncovered WITH A REASON FROM BOTH SIDES — which is the
    composer's fail-closed answer, not a silent gap.
    """
    from meep_gpu.metal_kernels import offdiag_update_e as offdiag

    grid = _grid()
    fields = _fields(grid)
    shape = grid.shape
    rng = np.random.default_rng(11)
    epsilon = np.full(shape, np.float32(2.0), dtype=np.float32)
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    row = {"Ex": {"Ey": (0.03 * rng.standard_normal(shape)).astype(np.float32)}}
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")}, row)

    mine = _verdict(fields, grid)
    assert not mine.covered
    assert any("most-general case" in r for r in mine.reasons), mine.reasons

    theirs = offdiag.offdiag_constitutive_coverage(fields, _pml(grid),
                                                   _Residency())
    assert not theirs.covered
    assert any("chi2/chi3" in r for r in theirs.reasons), theirs.reasons


def test_an_inactive_absorber_is_refused_because_update_e_is_a_different_sub_step():
    """Without a PML ``update_E`` is ``field[...] = constitutive``
    (stepping.py:1019-1022) — no dsigw accumulation and no ``f_w`` at all."""
    grid = _grid()
    fields = _fields(grid)
    verdict = nonlinear.nonlinear_constitutive_coverage(
        fields, PML(grid=grid, thickness=0), _Residency())
    assert not verdict.covered
    assert any("no active PML" in r for r in verdict.reasons), verdict.reasons


def test_a_plan_built_with_no_residency_is_refused_by_name():
    grid = _grid()
    verdict = nonlinear.nonlinear_constitutive_coverage(_fields(grid), _pml(grid),
                                                        None)
    assert not verdict.covered
    assert any("residency" in r for r in verdict.reasons), verdict.reasons


# ---------------------------------------------------------------------------
# The magnitude ceiling — a coverage refusal, from BOTH sides
# ---------------------------------------------------------------------------

def test_the_magnitude_ceiling_admits_below_and_refuses_above(monkeypatch):
    """A CEILING TESTED FROM ONE SIDE IS A CONSTANT, NOT A CLAUSE.

    With ``chi1inv = 0.5``, ``|chi1inv|^3 = 0.125``, so the clause quantity is
    ``chi3 * 0.125``. The pair below and above 1e29 is therefore chi3 = 4e29
    (5.0e28, admitted) and chi3 = 1.6e30 (2.0e29, refused) — computed against the
    installed epsilon rather than picked to look big, which is the mistake the
    composition matrix's first ceiling row made and the sweep caught.

    The ADMITTED half needs the flush policy set explicitly; see
    :func:`test_a_nonlinear_run_is_admitted` for why inheriting it is a pass that
    depends on collection order.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    grid = _grid()
    below = _fields(grid, chi3=4e29, inverse=0.5)
    assert _verdict(below, grid).covered, _verdict(below, grid).reasons

    above = _fields(grid, chi3=1.6e30, inverse=0.5)
    verdict = _verdict(above, grid)
    assert not verdict.covered
    reason, = [r for r in verdict.reasons if "chi3[Ez]" in r]
    assert "2.000e+29" in reason, reason
    assert "ceiling" in reason and "1e+29" in reason


def test_the_ceiling_is_per_cell_and_not_a_product_of_two_maxima(monkeypatch):
    """``chi`` and ``chi1inv`` live on the same grid, so the right question is
    ``max_i(|chi_i| * |chi1inv_i|^p)``.

    A run whose large chi and large epsilon are in DIFFERENT cells is admitted; the
    looser ``max|chi| * max|chi1inv|^p`` form would refuse it. Built so the two
    forms disagree.

    THE VALUES ARE PICKED WITH float32 ROUNDING IN MIND, because the first spelling
    of this test did not: ``np.float32(8e29) * 0.125`` is 1.0000000150e29, which is
    ABOVE a ceiling of exactly 1e29, so the "admitted" half failed on a rounding
    step rather than on the clause. 7.9e29 leaves real headroom.

    The ADMITTED half needs the flush policy set explicitly; see
    :func:`test_a_nonlinear_run_is_admitted` for why inheriting it is a pass that
    depends on collection order.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    grid = _grid()
    shape = grid.shape
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    inverse[0, 0, 0] = np.float32(1.0)          # |chi1inv|^3 = 1.0 here
    epsilon = (1.0 / inverse).astype(np.float32)
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")})
    chi3 = np.full(shape, np.float32(7.9e29), dtype=np.float32)  # * 0.125 = 9.9e28
    chi3[0, 0, 0] = np.float32(9e28)                             # * 1.0   = 9.0e28
    fields.set_nonlinear_volumes({"Ez": np.float32(0.1)}, {"Ez": chi3})

    loose = float(np.max(np.abs(chi3))) * float(np.max(np.abs(inverse)) ** 3)
    assert loose > nonlinear.CHI_MAGNITUDE_CEILING, loose
    tight = float(np.max(np.abs(chi3) * np.abs(inverse) ** 3))
    assert tight <= nonlinear.CHI_MAGNITUDE_CEILING, tight
    assert _verdict(fields, grid).covered, _verdict(fields, grid).reasons


def test_the_ceiling_is_evaluated_in_float64_so_it_cannot_overflow_to_a_refusal():
    """``chi3 * |chi1inv|^3`` at the ceiling OVERFLOWS float32.

    A check that overflowed to ``inf`` and then refused everything would be a
    different clause wearing this one's name, and it would look like a working
    ceiling from every test that only probes above it.
    """
    inverse = np.full((2, 2, 2), np.float32(1.0), dtype=np.float32)
    chi3 = np.full((2, 2, 2), np.float32(1e29), dtype=np.float32)
    with np.errstate(over="ignore"):
        overflowed = np.float32(1e29) * np.float32(1e29)
    assert np.isinf(overflowed), \
        "the float32 overflow this test guards against did not occur"
    magnitude = nonlinear._magnitude(chi3, inverse, 3)
    assert magnitude is not None and np.isfinite(magnitude)
    assert magnitude == pytest.approx(1e29, rel=1e-6)


def test_a_chi_whose_magnitude_cannot_be_evaluated_is_refused_rather_than_assumed():
    """FAIL-CLOSED. An unevaluable operand is not a passing one."""
    assert nonlinear._magnitude(object(), np.ones((2, 2, 2), np.float32), 3) is None
    assert nonlinear._magnitude(np.full((2, 2, 2), np.inf, np.float32),
                                np.ones((2, 2, 2), np.float32), 3) is None


# ---------------------------------------------------------------------------
# Registration and wiring
# ---------------------------------------------------------------------------

def test_the_nonlinear_spine_is_disjoint_from_the_incumbent_but_covers_the_three_unaffected_substeps(
        monkeypatch):
    """The χ²/χ³ factor belongs only to ``update_E``.

    ``step_B``, ``step_D``, and ``update_H`` reuse the certified ordinary Metal
    bodies under a separate, inverted admission; the incumbent must continue to
    refuse this nonlinear configuration so the planner never resolves an overlap
    by arm ordering.  ``update_E`` is deliberately refused by the spine because
    its Pade body is the dedicated nonlinear shader.
    """
    monkeypatch.setattr(coverage, "_metal_backend_reasons", lambda grid: [])
    fields = _fields(_grid())
    pml = _pml(fields.grid)
    residency = _Residency()

    for slot in ("step_B", "step_D"):
        incumbent = coverage.pml_curl_coverage(fields, pml, slot, residency)
        spine = nonlinear.nonlinear_run_pml_curl_coverage(
            fields, pml, slot, residency)
        assert not incumbent.covered
        assert any("chi2/chi3" in reason for reason in incumbent.reasons)
        assert spine.covered, spine.reasons

    incumbent_h = coverage.constitutive_coverage(fields, pml, "H", residency)
    spine_h = nonlinear.nonlinear_run_constitutive_coverage(
        fields, pml, "H", residency)
    assert not incumbent_h.covered
    assert any("chi2/chi3" in reason for reason in incumbent_h.reasons)
    assert spine_h.covered, spine_h.reasons

    spine_e = nonlinear.nonlinear_run_constitutive_coverage(
        fields, pml, "E", residency)
    assert not spine_e.covered
    assert any("side='E'" in reason for reason in spine_e.reasons)


def test_the_nonlinear_spine_requires_a_real_nonlinear_material(monkeypatch):
    """The inverse clause prevents it overlapping ordinary linear PML plans."""
    monkeypatch.setattr(coverage, "_metal_backend_reasons", lambda grid: [])
    grid = _grid()
    fields = _fields(grid, chi2=0.0, chi3=0.0)
    pml = _pml(grid)
    for slot in ("step_B", "step_D"):
        verdict = nonlinear.nonlinear_run_pml_curl_coverage(
            fields, pml, slot, _Residency())
        assert not verdict.covered
        assert any("no chi2/chi3" in reason for reason in verdict.reasons)
    verdict = nonlinear.nonlinear_run_constitutive_coverage(
        fields, pml, "H", _Residency())
    assert not verdict.covered
    assert any("no chi2/chi3" in reason for reason in verdict.reasons)


def test_the_nonlinear_arms_are_registered_and_wired_on_exactly_their_four_slots():
    """A family that quietly changed slots or became unwired would change what
    ``plan_step`` launches."""
    registered = [spec for spec in arms.registered()
                  if spec.family == nonlinear.FAMILY]
    assert [(spec.slot, spec.label, spec.wired) for spec in registered] == [
        ("step_B", "nonlinear PML curl", True),
        ("step_D", "nonlinear PML curl", True),
        ("update_H", "nonlinear PML magnetic", True),
        ("update_E", "nonlinear", True),
    ]


def test_the_family_module_is_in_the_registry_list():
    """A family in the tree but NOT in ``FAMILY_MODULES`` is invisible to
    ``plan_step`` — a silent coverage loss rather than an error.

    Asserted against the package CONTENTS, not against a copy of the tuple, so a
    module that registers an arm and is left out of the list fails here.
    """
    assert "nonlinear_update_e" in registry.FAMILY_MODULES
    assert nonlinear.__name__.rsplit(".", 1)[-1] == "nonlinear_update_e"


def test_re_registering_the_same_arm_is_refused():
    """A duplicate would make WHICH KERNEL RUNS depend on import order."""
    with pytest.raises(ValueError, match="already registered"):
        arms.register(family=nonlinear.FAMILY, slot=nonlinear.SLOT,
                      label="nonlinear",
                      coverage=lambda *a: None, plan=lambda *a: None)


def test_the_plan_builder_returns_none_out_of_coverage():
    """None means REFUSED. A covered verdict must never meet a raise, and a
    refused one must never produce a plan."""
    grid = _grid()
    fields = _fields(grid, chi2=0.0, chi3=0.0)
    assert nonlinear.plan_nonlinear_constitutive(fields, _pml(grid),
                                                 _Residency()) is None


# ---------------------------------------------------------------------------
# Device legs
# ---------------------------------------------------------------------------

def _seed_spine_pair(actual, reference, seed=29):
    """Put identical non-fixed-point state into the three unaffected sub-steps."""
    rng = np.random.default_rng(seed)
    names = (
        "Bx", "By", "Bz", "Dx", "Dy", "Dz",
        "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
        "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
        "f_w_Hx", "f_w_Hy", "f_w_Hz",
    )
    for name in names:
        values = rng.uniform(-0.35, 0.35, actual.grid.shape).astype(np.float32)
        getattr(actual, name)[...] = values
        getattr(reference, name)[...] = values


@needs_mps
@pytest.mark.parametrize(
    "slot, watched",
    (
        ("step_B", ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")),
        ("step_D", ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")),
        ("update_H", ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")),
    ),
)
def test_nonlinear_spine_substeps_are_byte_identical_to_stepping(
        monkeypatch, slot, watched):
    """The reused Metal shader must reproduce its nonlinear engine context exactly.

    This is the narrow device claim that licenses the new admissions: the array
    path performs the named sub-step on a nonlinear material, the corresponding
    plan runs over persistent mirrors, and every affected float32 word must match.
    ``update_E`` is intentionally absent because it reads the Pade factor and has
    its own dedicated shader and gate.
    """
    from meep_gpu.metal_kernels.device import Residency

    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    grid = _grid("metallic")
    actual = _fields(grid, components=("Ex", "Ey", "Ez"))
    reference = _fields(_grid("metallic"), components=("Ex", "Ey", "Ez"))
    actual_pml = _pml(grid)
    reference_pml = _pml(reference.grid)
    _seed_spine_pair(actual, reference)
    before = {name: np.array(getattr(actual, name), copy=True) for name in watched}

    if slot == "update_H":
        stepping.update_H(reference, reference_pml)
        plan = nonlinear.plan_nonlinear_run_constitutive(
            actual, actual_pml, "H", Residency())
    else:
        getattr(stepping, slot)(reference, reference_pml)
        plan = nonlinear.plan_nonlinear_run_pml_curl(
            actual, actual_pml, slot, Residency())
    assert plan is not None
    residency = plan.residency
    residency.sync_in()
    plan.run()
    residency.sync_out()
    assert plan.launches == 1

    moved = sum(int(np.count_nonzero(
        before[name].view(np.uint32) != getattr(reference, name).view(np.uint32)))
        for name in watched)
    assert moved > 0, f"VACUOUS {slot}: the array-path sub-step moved no words"
    differing = sum(int(np.count_nonzero(
        getattr(actual, name).view(np.uint32) != getattr(reference, name).view(np.uint32)))
        for name in watched)
    assert differing == 0, f"{slot}: {differing} differing words (moved {moved})"


@needs_mps
@pytest.mark.parametrize("boundaries", ("periodic", "metallic"))
@pytest.mark.parametrize("scalar", (False, True), ids=("volume", "scalar"))
@pytest.mark.parametrize("components", (("Ez",), ("Ex", "Ey", "Ez")),
                         ids=("one", "all"))
def test_one_update_e_is_byte_identical_to_stepping(monkeypatch, boundaries,
                                                    scalar, components):
    """THE SUB-STEP CLAIM, as words and with a vacuity floor.

    Zero init is a FIXED POINT of the constitutive sub-step, so the volumes are
    seeded with physical-band values and the leg asserts the array path actually
    MOVED state — a no-op agreeing with a no-op is trivially identical.
    """
    from meep_gpu.metal_kernels.device import Residency

    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    assert not subnormal.mps_policy_reasons(), (
        "the executor refuses the resolved policy, so this leg would be measuring "
        "the refusal rather than the kernel")

    grid = _grid(boundaries)
    fields = _fields(grid, components=components, scalar=scalar)
    pml = _pml(grid)
    rng = np.random.default_rng(3)
    for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.standard_normal(
            grid.shape).astype(np.float32)

    watched = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")
    before = {n: getattr(fields, n).copy() for n in watched}
    stepping.update_E(fields, pml)
    reference = {n: getattr(fields, n).copy() for n in watched}
    moved = sum(int(np.count_nonzero(reference[n].view(np.uint32)
                                     != before[n].view(np.uint32)))
                for n in watched)
    assert moved > 0, "VACUOUS: the array path moved nothing"

    for name, values in before.items():
        getattr(fields, name)[...] = values
    residency = Residency()
    plan = nonlinear.plan_nonlinear_constitutive(fields, pml, residency)
    assert plan is not None
    residency.sync_in()
    plan.run()
    residency.sync_out()
    assert plan.launches == 1

    differing = sum(int(np.count_nonzero(getattr(fields, n).view(np.uint32)
                                         != reference[n].view(np.uint32)))
                    for n in watched)
    assert differing == 0, f"{differing} differing words (moved {moved})"


@needs_mps
def test_asking_for_an_unbuilt_contract_variant_raises(monkeypatch):
    """A guard argument that quietly did nothing would make a gate's most
    important leg vacuous."""
    from meep_gpu.metal_kernels.device import Residency

    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    grid = _grid()
    plan = nonlinear.plan_nonlinear_constitutive(_fields(grid), _pml(grid),
                                                 Residency())
    assert plan is not None
    with pytest.raises(KeyError, match="holds no 'fast' variant"):
        plan.run(shaders.CONTRACT_FAST)


@needs_mps
def test_the_keep_policy_refuses_the_plan_at_build_time(monkeypatch):
    """Metal FLUSHES subnormals and exposes no lever, so ``keep`` is not
    offerable and a run resolved to it is refused rather than silently stepped
    under a policy the device was never in."""
    from meep_gpu.metal_kernels.device import Residency

    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "keep")
    grid = _grid()
    verdict = nonlinear.nonlinear_constitutive_coverage(_fields(grid), _pml(grid),
                                                        Residency())
    assert not verdict.covered
    assert any("subnormal policy" in r for r in verdict.reasons), verdict.reasons
