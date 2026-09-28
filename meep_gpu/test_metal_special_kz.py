"""Merge-bar tests for the Metal special_kz (``grid.beta``) curls.

WHAT IS PINNED HERE THAT NO GATE CAN PIN, because a gate needs a GPU and this file
must pass on any host:

1. **The coefficient, against ``stepping`` ITSELF.** ``beta_curl_coefficients`` is
   a transcription of ``stepping._special_kz_beta_term``'s host arithmetic, and the
   two are compared BYTE FOR BYTE by calling the array path's own function. A gate
   comparing kernel output to array-path output would agree even if both used the
   same wrong coefficient; this is the only place the coefficient itself is the
   thing under test.
2. **The three defects that are byte-invisible in this family**, pinned by
   SOURCE-TEXT assertion instead — the discipline the BFAST gate established for
   stating a negative result. Each names the measurement that showed it invisible.
3. **The probe contract**, in every direction: no artifact, another backend's
   artifact, a record missing the fifth pattern, a ``NEITHER`` verdict, and an
   all-ambiguous record. Four refusals and one licence.
4. **The refusals**, over a duck-typed configuration matrix that needs no GPU.
5. **The 31-binding ceiling**, asserted against the complex signature's own count:
   a re/im-split build would need nine more buffers and is a COMPILE ERROR, not a
   slower kernel, so the constraint is proven from the shipped source rather than
   left for a future edit to rediscover.

The device tests at the end are skipped without MPS. They are the smallest
non-vacuous version of the byte gate — one full cycle per storage family — so a red
merge bar names the defect without a GPU run.
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from meep_gpu.metal_kernels import special_kz, subnormal, templates

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "metal_kernels")


def _mps_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch.backends, "mps", None), "is_available",
                        lambda: False)())


requires_mps = pytest.mark.skipif(not _mps_available(),
                                  reason="no MPS device on this host")

BETA_KZ2D = 0.3321611318837033
BETA_GRATING = -0.6850526103319672


# ---------------------------------------------------------------------------
# 1. The optional-dependency contract
# ---------------------------------------------------------------------------

def test_the_family_module_imports_no_torch_at_module_level():
    """A predicate that could only be read on a Mac is a predicate nobody reviews."""
    path = os.path.join(PACKAGE_DIR, "special_kz.py")
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders.extend(alias.name for alias in node.names
                             if alias.name.split(".")[0] in ("torch", "numpy"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] \
                in ("torch", "numpy"):
            offenders.append(node.module)
    assert not offenders, f"module-level heavy imports: {offenders}"


# ---------------------------------------------------------------------------
# 2. The coefficient, against stepping.py itself
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, beta, dt):
        self.beta = beta
        self.dt = dt


class _Fields:
    def __init__(self, beta, dt, offdiagonal=False):
        self.grid = _Grid(beta, dt)
        self.has_offdiagonal_epsilon = offdiagonal


@pytest.mark.parametrize("beta,dt", [(BETA_KZ2D, 1.0 / 24.0),
                                     (BETA_GRATING, 0.034),
                                     (-0.3907, 0.0233),
                                     (0.2, 0.35 / 10.0)])
@pytest.mark.parametrize("magnetic", [True, False])
@pytest.mark.parametrize("complex_storage", [True, False])
def test_the_coefficient_matches_steppings_own_rounding(beta, dt, magnetic,
                                                        complex_storage):
    """Byte-identical to what ``stepping._special_kz_beta_term`` puts in the product.

    The array path computes ``sign*2*pi*beta*dt`` in f64, multiplies by ±1j under
    complex storage, and rounds ONCE through ``partner.dtype.type`` before the
    multiply. This calls that function on a unit partner and reads the coefficient
    straight back out of the returned product — so a rounding moved, doubled or
    dropped anywhere in the transcription fails here rather than in a gate run.
    """
    from meep_gpu import stepping

    dtype = np.complex64 if complex_storage else np.float32
    fields = _Fields(beta, dt)
    ours = special_kz.beta_curl_coefficients(beta, dt, magnetic, complex_storage)
    for index, sign in enumerate((1.0, -1.0)):
        # THE PRODUCT IS COMPARED, NOT THE COEFFICIENT UN-NEGATED. `-(x)` on a
        # complex flips BOTH words, so "negate stepping's answer back" would move
        # the signed zero this coefficient exists to carry. Forming the same
        # product from our own words and comparing bit for bit asks the question
        # the kernel actually asks.
        partner = np.ones((1,), dtype=dtype)
        theirs = stepping._special_kz_beta_term(fields, partner, sign,
                                                magnetic=magnetic)
        if complex_storage:
            mine = np.array([complex(ours[index][0], ours[index][1])],
                            dtype=np.complex64)
        else:
            mine = np.array([ours[index]], dtype=np.float32)
        product = -(mine * partner)
        assert not np.count_nonzero(
            product.view(np.uint32) != np.asarray(theirs).view(np.uint32)), (
            f"{product!r} != {theirs!r}")


@pytest.mark.parametrize("magnetic", [True, False])
@pytest.mark.parametrize("complex_storage", [True, False])
def test_the_coefficient_matches_stepping_on_numpy_typed_grid_scalars(
        magnetic, complex_storage):
    """S:770 IS THE EXPRESSION, on the objects the array path holds — not on
    ``float()`` copies of them.

    THE PATH IS LIVE. ``Grid._resolve_beta`` coerces ``beta`` to a Python float;
    NOTHING coerces ``dt``, which is ``courant / resolution``, so
    ``Grid(courant=numpy.float32(...))`` leaves ``grid.dt`` a ``numpy.float32`` and
    hands it straight to both ``stepping._special_kz_beta_term`` and
    ``beta_curl_coefficients``. Under NEP 50 a numpy-typed operand rounds the
    product chain EARLY, so a transcription that widens through ``float()`` first
    computes the whole chain in f64 and rounds once — a different float32 word.

    MEASURED 2026-08-15 before the casts came out: 14,926 of 40,000 random
    ``numpy.float32`` (beta, dt, sign) draws produced a different coefficient. This
    test is the pin; the four rows above cover the Python-float case, which never
    discriminated because there both spellings are the same arithmetic.

    THE SCALARS COME FROM A REAL ``Grid``, not from hand-made numpy values, so the
    test is a statement about the engine rather than about a constructed hazard —
    and note that ``grid.beta`` is a Python float (``_resolve_beta`` coerces it)
    while ``grid.dt`` is not, which is the exact mixture the two spellings disagree
    on. ``beta = BETA_GRATING`` with ``courant = 0.34`` is the byte gate's own
    ``r2`` pair and it DISCRIMINATES on BOTH signs: ``float()`` gives
    ``-0.14634662866592407``, the literal spelling ``-0.14634664356708527``.
    Measured over the corpus betas crossed with fourteen Courants and two
    resolutions: 76 of 168 rows discriminate, so the pin is not a lucky draw.

    The ``type(grid.dt)`` assertion is deliberate — if ``Grid`` ever coerces
    ``dt``, this test must be re-examined rather than pass silently on a hazard that
    has moved.
    """
    from meep_gpu import stepping
    from meep_gpu.grid import Grid

    grid = Grid(resolution=10.0, cell_size=(2.0, 1.6, 0.0), boundaries="periodic",
                dimensions=2, courant=np.float32(0.34), k_point=(0.0, 0.0, 0.0),
                beta=BETA_GRATING, xp=np)
    assert isinstance(grid.dt, np.float32), (
        f"grid.dt is {type(grid.dt).__name__}: Grid now coerces it, so this test "
        f"no longer reaches the NEP-50 rounding hazard it was written for and the "
        f"transcription's literal spelling needs a new pin")

    dtype = np.complex64 if complex_storage else np.float32
    fields = _Fields(grid.beta, grid.dt)
    ours = special_kz.beta_curl_coefficients(grid.beta, grid.dt, magnetic,
                                             complex_storage)
    for index, sign in enumerate((1.0, -1.0)):
        partner = np.ones((1,), dtype=dtype)
        theirs = stepping._special_kz_beta_term(fields, partner, sign,
                                                magnetic=magnetic)
        mine = (np.array([complex(ours[index][0], ours[index][1])],
                         dtype=np.complex64) if complex_storage
                else np.array([ours[index]], dtype=np.float32))
        product = -(mine * partner)
        assert not np.count_nonzero(
            product.view(np.uint32) != np.asarray(theirs).view(np.uint32)), (
            f"numpy-typed dt: {product!r} != {theirs!r}. The transcription is "
            f"normalising operands stepping.py:797 does not normalise")


def test_the_complex_coefficient_carries_a_signed_zero_real_word():
    """The ±0 real word is Python's own complex multiply and is PASSED THROUGH.

    Synthesizing it as ``+0.0`` would be wrong on exactly the cells where it
    matters, and it is what makes the beta product's expansion arm ambiguous — a
    fact the probe records and two gate rows depend on.
    """
    (plus, minus) = special_kz.beta_curl_coefficients(BETA_KZ2D, 1.0 / 24.0,
                                                      magnetic=True,
                                                      complex_storage=True)
    signs = {np.float32(plus[0]).view(np.uint32).item(),
             np.float32(minus[0]).view(np.uint32).item()}
    assert signs & {0x80000000}, (
        "no coefficient carried a NEGATIVE zero real word; Python's complex "
        "multiply produces one for one of the two signs and passing it through "
        "is the whole reason the real word is not synthesized")


def test_the_term_carries_no_dtdx():
    """An analytic derivative, not a finite difference (stepping.py:758-762)."""
    source = special_kz.beta_curl_source((0, 0, 0), False, True)
    beta_lines = [line for line in source.splitlines() if "beta_plus" in line
                  and "curl0" in line]
    assert beta_lines, "the beta insert is missing from the shipped source"
    assert all("dtdx" not in line for line in beta_lines), beta_lines


# ---------------------------------------------------------------------------
# 3. The three byte-invisible defects, pinned by source text
# ---------------------------------------------------------------------------

def test_the_beta_partner_is_the_unshifted_centre_operand():
    """Measured invisible on the z axis, so pinned here.

    ``beta`` is legal only on an effective-2-D grid, so the invariant axis has
    extent 1 and ``b_z`` IS ``b``: the gate measured that mutation at 0/2 caught
    and records it as a predicted null. The property still has to hold, so it is
    asserted on the SOURCE: the term reads ``b`` and ``a``, never a shifted name.
    """
    for backward in (False, True):
        source = special_kz.beta_curl_source((0, 0, 0), backward, True)
        assert "curl0 = curl0 - (beta_plus * b);" in source
        assert "curl1 = curl1 - (beta_minus * a);" in source
        for shifted in ("b_x", "b_z", "a_y", "a_z", "c_x", "c_y"):
            assert f"beta_plus * {shifted}" not in source
            assert f"beta_minus * {shifted}" not in source


def test_the_real_arm_spells_subtraction_rather_than_a_negation():
    """The zero-sign question does not arise where no negation is spelled.

    The gate measured ``a + (0.0f - b)`` at 0/2 caught in this arm and records the
    reason: the arm spells IEEE subtraction. The spelling hazard lives in the
    complex helper and is measured there at the product layer (36/512 words).
    """
    source = special_kz.beta_curl_source((0, 0, 0), False, True)
    assert "0.0f -" not in source
    assert "* -1.0" not in source


def test_the_beta_product_puts_the_coefficient_on_the_left():
    """S:784 is ``dtype.type(coefficient) * partner_values``.

    Byte-invisible for THIS product — the coefficient's ±0 real word makes both
    fusions exact, which the gate records as a predicted null and the probe as
    AMBIGUOUS_BOTH — so the orientation is pinned on the source instead, and the
    rule itself is measured on the Bloch rotation where it does bite.
    """
    source = special_kz.beta_bloch_curl_source((0, 0, 0), False, (1, 0, 0),
                                               "FMA_V1", True)
    assert "c_mul(float2(bpr, bpi), b)" in source
    assert "c_mul(b, float2(bpr, bpi))" not in source
    # ...and the phase rotation keeps the FIELD on the left (S:1862's `*=`).
    assert "c_mul(b_x, float2(pxr, pxi))" in source


# ---------------------------------------------------------------------------
# 4. Source specialisation
# ---------------------------------------------------------------------------

def test_the_beta_arm_is_compiled_in_or_out_and_not_branched():
    with_beta = special_kz.beta_curl_source((0, 0, 0), False, True)
    without = special_kz.beta_curl_source((0, 0, 0), False, False)
    assert "beta_plus * b" in with_beta
    assert "beta_plus * b" not in without
    assert "HAS_BETA = 0" in without
    assert templates.source_sha256(with_beta) != templates.source_sha256(without)


def test_the_contraction_directive_appears_exactly_once_per_source():
    for source in special_kz.enumerate_sources().values():
        assert source.count("pragma clang fp") == 1


#: Every (boundaries, k) a beta run can actually be BUILT with on this engine.
#: ``Grid`` refuses a metallic invariant axis and a k component on it, so the
#: matrix below is the whole reachable space of ghost rules and phases, not a
#: sample. Measured with :func:`meep_gpu.grid.Grid` 2026-08-15.
_REACHABLE = [
    ("periodic", (0.0, 0.0, 0.0)),
    ("periodic", (0.3, 0.0, 0.0)),
    ("periodic", (0.0, 0.2, 0.0)),
    ("periodic", (0.3, 0.2, 0.0)),
    (("metallic", "periodic", "periodic"), (0.0, 0.0, 0.0)),
    (("metallic", "periodic", "periodic"), (0.0, 0.2, 0.0)),
    (("periodic", "metallic", "periodic"), (0.0, 0.0, 0.0)),
    (("periodic", "metallic", "periodic"), (0.3, 0.0, 0.0)),
    (("metallic", "metallic", "periodic"), (0.0, 0.0, 0.0)),
]


@pytest.mark.parametrize("boundaries,k", _REACHABLE)
def test_every_source_a_reachable_grid_emits_is_enumerated(boundaries, k):
    """The WELD MUST COVER WHAT THE GATE LAUNCHES, and once it did not.

    ``enumerate_sources`` is what the gate fingerprints into ``provenance.json``.
    Its complex half used to be a hand-written pair of boundary triples and three
    phase triples, and it had fallen behind the emitter: the gate's OWN
    ``c3_brillouin_edge_metallic_y`` case resolves codes ``(0, 1, 0)``, whose source
    sha256 appeared in NO provenance row on either sub-step. Measured 2026-08-15: 32
    reachable complex specialisations were missing.

    This walks the reachable (boundary, k) space through the SAME calls the plan
    builders make — ``stepping._boundary_kinds`` then
    ``special_kz.bloch_phase_words`` then the emitter — and asserts the result is in
    the enumeration. It needs no GPU: a source is a string.
    """
    import hashlib

    from meep_gpu import stepping
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    catalogue = {hashlib.sha256(source.encode("utf-8")).hexdigest(): label
                 for label, source in special_kz.enumerate_sources().items()}
    complex_storage = any(float(component) != 0.0 for component in k)
    grid = Grid(resolution=10.0, cell_size=(2.0, 1.6, 0.0), boundaries=boundaries,
                dimensions=2, courant=0.34, k_point=k, beta=BETA_KZ2D, xp=np)
    pml = PML(grid=grid, thickness=tuple((2, 2) if grid.shape[a] >= 6 else (0, 0)
                                         for a in range(3)))
    kinds = stepping._boundary_kinds(grid, pml)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    for sub_step, spec in special_kz.SUB_STEPS.items():
        for has_beta in (True, False):
            if complex_storage:
                phased, _ = special_kz.bloch_phase_words(
                    grid, kinds, bool(spec["backward"]))
                for expansion in sorted(templates.EXPANSIONS):
                    source = special_kz.beta_bloch_curl_source(
                        codes, spec["backward"], phased, expansion, has_beta)
                    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                    assert digest in catalogue, (
                        f"{boundaries}/{k}/{sub_step}/{expansion}/beta{int(has_beta)}"
                        f" emits codes={codes} phased={phased}, which is NOT in "
                        f"enumerate_sources() — the gate would launch a "
                        f"specialisation it never fingerprinted")
            # The real arm's emitter is boundary-only and is exercised on every row.
            source = special_kz.beta_curl_source(codes, spec["backward"], has_beta)
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            assert digest in catalogue, (codes, sub_step, has_beta)


def test_the_enumeration_is_a_superset_of_what_the_emitter_can_produce():
    """...and the derived set is not merely large, it CONTAINS the emittable one."""
    emittable = set(special_kz._emittable_bloch_specialisations())
    for codes, phased in emittable:
        for axis in range(3):
            if codes[axis] == templates.METALLIC:
                assert phased[axis] == 0, (codes, phased)
        assert phased[2] == 0, (codes, phased)
    # eight boundary triples, and every phase combination each of them permits
    assert len(emittable) == 18, sorted(emittable)
    assert len(special_kz.enumerate_sources()) == 32 + len(emittable) * 8


def test_the_mask_masks_both_planes_on_the_complex_arm():
    """A complex zero is assigned to BOTH words (stepping.py:1943)."""
    source = special_kz.beta_bloch_curl_source((1, 1, 1), True, (0, 0, 0),
                                               "FMA_V1", True)
    assert source.count("float2(0.0f, 0.0f) : curl") == 6


def test_the_complex_signature_fits_under_the_31_binding_ceiling():
    """A measured platform limit, not a style rule.

    Metal's buffer attribute indices run 0..30; a 32nd is a COMPILE ERROR. This
    kernel binds 30. A re/im-split build would need nine more and could not be
    built at all, which is why the complex volumes are ``float2`` — forced, not
    preferred.
    """
    from meep_gpu.metal_kernels import device

    source = special_kz.beta_bloch_curl_source((0, 0, 0), False, (1, 1, 0),
                                               "FMA_V1", True)
    indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
    assert indices == sorted(indices)
    assert max(indices) < device.MAX_BUFFER_BINDINGS, indices
    assert len(indices) == 30, indices
    assert source.count("device float2*") + source.count("device const float2*") == 9


def test_an_unknown_expansion_arm_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        special_kz.beta_bloch_curl_source((0, 0, 0), False, (0, 0, 0), "GUESS", True)
    with pytest.raises(ValueError):
        special_kz.beta_curl_source((2, 0, 0), False, True)


# ---------------------------------------------------------------------------
# 5. The probe contract
# ---------------------------------------------------------------------------

def _record(**overrides):
    patterns = {name: "AMBIGUOUS_BOTH" for name in special_kz.BETA_PROBE_PATTERNS}
    patterns["c8_mul_c8"] = "FMA_V1"
    patterns.update(overrides.pop("patterns", {}))
    record = {"backend": special_kz.PROBE_BACKEND, "patterns": patterns}
    record.update(overrides)
    return record


def test_a_measured_record_licenses_exactly_one_arm():
    assert special_kz.beta_expansion_from_probe(_record()) == "FMA_V1"


def test_a_record_without_the_fifth_pattern_licenses_nothing():
    """An artifact cut before this tranche does not license this family."""
    record = _record()
    del record["patterns"][special_kz.BETA_PROBE_PATTERN]
    assert special_kz.beta_expansion_from_probe(record) is None


def test_a_neither_verdict_is_a_refusal_by_name():
    assert special_kz.beta_expansion_from_probe(
        _record(patterns={"c8_mul_c8": "NEITHER"})) is None


def test_an_all_ambiguous_record_licenses_nothing():
    """Nothing discriminated, so an arm chosen under it would be a default."""
    assert special_kz.beta_expansion_from_probe(
        _record(patterns={"c8_mul_c8": "AMBIGUOUS_BOTH"})) is None


def test_disagreeing_patterns_license_nothing():
    assert special_kz.beta_expansion_from_probe(
        _record(patterns={"c8_mul_c8": "FMA_V1",
                          "python_float_left": "NAIVE"})) is None


def test_another_backends_artifact_licenses_nothing():
    assert special_kz.beta_expansion_from_probe(_record(backend="cupy")) is None


# ---------------------------------------------------------------------------
# 6. Refusals, over a duck-typed matrix
# ---------------------------------------------------------------------------

_STUB_SHAPE = (8, 6, 1)


class _StubGrid:
    """A duck-typed grid the predicate can actually RESOLVE.

    ``is_metallic`` used to be missing, which made ``stepping._boundary_kinds``
    raise and the coverage wrapper answer ``None`` — so every stub row below
    refused with "boundary kinds could not be resolved for this grid" ON TOP of
    whatever clause it meant to test, and no stub configuration could ever be
    ADMITTED. A refusal matrix with no reachable positive control is satisfied by a
    predicate that refuses everything, which is the vacuity these tests exist to
    catch elsewhere. :func:`test_the_stub_baseline_is_ADMITTED` is the control.
    """

    def __init__(self, **kwargs):
        self.xp = np
        self.shape = _STUB_SHAPE
        self.dimensions = 2
        self.beta = BETA_KZ2D
        self.k_point = (0.0, 0.0, 0.0)
        self.has_bloch = False
        self.cylindrical = False
        self.bfast_active = False
        self.dt = 0.035
        self.dx = 0.1
        self._metallic = (False, False, False)
        self.__dict__.update(kwargs)

    def has_symmetry(self):
        return getattr(self, "_symmetry", False)

    def is_mirrored(self, axis):
        return False

    def is_metallic(self, axis):
        return bool(self._metallic[axis])

    def is_axis(self, axis):
        return False

    def bloch_phase(self, axis):
        return None


class _StubResidency:
    def mirror(self, *args, **kwargs):
        return None


class _StubPML:
    """An active absorber carrying the per-axis vectors the layout clause reads."""

    is_active = True

    def __init__(self, shape=_STUB_SHAPE):
        for axis, extent in zip("xyz", shape):
            for stem in ("kms", "sinv", "kps"):
                for suffix in ("", "_h"):
                    setattr(self, f"{stem}_{axis}{suffix}",
                            np.ones(extent, dtype=np.float32))

    def is_mirrored(self, axis):
        return False


class _StubFields:
    def __init__(self, grid, dtype=np.float32, **kwargs):
        self.grid = grid
        self.force_complex_fields = dtype is np.complex64
        self.stores_E = True
        self.has_nonlinearity = False
        self.has_offdiagonal_epsilon = False
        self.has_polarizations = False
        self.polarizations = ()
        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                     "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
                     "fu_Dx", "fu_Dy", "fu_Dz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                     "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            setattr(self, name, np.zeros(grid.shape, dtype=dtype))
        self.__dict__.update(kwargs)

    def condfac_for(self, component):
        return None

    def inverse_epsilon_for(self, component):
        return np.ones(self.grid.shape, dtype=np.float32)


def test_the_stub_baseline_is_ADMITTED(monkeypatch):
    """THE POSITIVE CONTROL for every refusal row below.

    Without it a predicate that refused every configuration for one unrelated
    reason would satisfy the whole parametrised matrix, because each row only
    asserts that ITS clause appears among the reasons — and the reasons accumulate.

    THE POLICY IS PINNED, as every other family's admission test pins it. This
    arm64 host's DEFAULT resolves to ``keep`` and the MPS executor cannot honour
    keep, so the predicate refuses; a baseline that read the ambient environment
    passed only because the shell that ran it exported ``flush``, which makes the
    positive control a property of the shell rather than of the code. MEASURED:
    green under an exported ``flush``, red under ``keep`` and red with the variable
    unset — the same three tests, the same tree.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    verdict = special_kz.beta_pml_curl_coverage(
        _StubFields(_StubGrid()), _StubPML(), "step_B", _StubResidency())
    assert verdict.covered, verdict.reasons


def test_the_stub_baseline_is_REFUSED_under_keep(monkeypatch):
    """...and the pinning above is not a way to make the refusal unreachable.

    The companion of the positive control: under ``keep`` the same stub must be
    refused BY NAME, so the pin is a declared precondition rather than a green
    light nobody can revoke.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.KEEP)
    verdict = special_kz.beta_pml_curl_coverage(
        _StubFields(_StubGrid()), _StubPML(), "step_B", _StubResidency())
    assert not verdict.covered
    assert any("subnormal policy" in reason for reason in verdict.reasons), \
        verdict.reasons


def test_the_complex_stub_baseline_is_ADMITTED(monkeypatch):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    grid = _StubGrid(has_bloch=True, k_point=(0.3, 0.0, 0.0))
    fields = _StubFields(grid, dtype=np.complex64)
    verdict = special_kz.beta_bloch_pml_curl_coverage(
        fields, _StubPML(), "step_B", _StubResidency(), _record())
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("label,grid_kwargs,fields_kwargs,fragment", [
    ("beta_zero", {"beta": 0.0}, {}, "grid.beta is zero"),
    ("three_d", {"dimensions": 3}, {}, "not the effective-2-D grid"),
    ("cylindrical", {"cylindrical": True}, {}, "cylindrical"),
    ("bfast", {"bfast_active": True}, {}, "BFAST"),
    ("k_point", {"k_point": (0.3, 0.0, 0.0)}, {}, "not exactly zero"),
    ("complex_storage", {}, {"force_complex_fields": True},
     "belongs to the complex beta arm"),
    ("nonlinearity", {}, {"has_nonlinearity": True}, "chi2/chi3"),
    ("offdiagonal", {}, {"has_offdiagonal_epsilon": True},
     "implicit-i trick no longer cancels"),
    ("no_stored_E", {}, {"stores_E": False}, "recomputed from D"),
    ("mirror_plane", {"_symmetry": True}, {}, "a mirror plane is active"),
    ("shape_not_3d", {"shape": (8, 6)}, {}, "not three-dimensional"),
])
def test_the_real_predicate_refuses_by_name(label, grid_kwargs, fields_kwargs,
                                            fragment):
    """Every refusal names its clause. A refusal with no reason is not a refusal."""
    grid = _StubGrid(**grid_kwargs)
    fields = _StubFields(grid, **fields_kwargs)
    verdict = special_kz.beta_pml_curl_coverage(fields, _StubPML(), "step_B",
                                                _StubResidency())
    assert not verdict.covered
    assert any(fragment in reason for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("label,grid_kwargs,fields_kwargs,fragment", [
    ("beta_zero", {"beta": 0.0}, {}, "grid.beta is zero"),
    ("three_d", {"dimensions": 3}, {}, "not the effective-2-D grid"),
    ("cylindrical", {"cylindrical": True}, {}, "cylindrical"),
    ("bfast", {"bfast_active": True}, {}, "BFAST"),
    ("k_on_the_invariant_axis", {"k_point": (0.3, 0.0, 0.4)}, {},
     "rides the invariant axis"),
    ("nonlinearity", {}, {"has_nonlinearity": True}, "chi2/chi3"),
    ("no_stored_E", {}, {"stores_E": False}, "recomputed from D"),
    ("dispersion", {}, {"has_polarizations": True},
     "complex-storage ADE is a future tranche"),
    ("shape_not_3d", {"shape": (8, 6)}, {}, "not three-dimensional"),
])
def test_the_complex_predicate_refuses_by_name(label, grid_kwargs, fields_kwargs,
                                               fragment):
    """The complex arm's own matrix, against its own admitted baseline.

    It had none: every complex refusal was checked by the byte gate's refusals leg
    on a GPU host, so a laptop merge bar could not see a widened complex clause at
    all.
    """
    defaults = {"has_bloch": True, "k_point": (0.3, 0.0, 0.0)}
    defaults.update(grid_kwargs)
    grid = _StubGrid(**defaults)
    fields = _StubFields(grid, dtype=np.complex64, **fields_kwargs)
    verdict = special_kz.beta_bloch_pml_curl_coverage(
        fields, _StubPML(), "step_B", _StubResidency(), _record())
    assert not verdict.covered
    assert any(fragment in reason for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("arm", ["real", "complex"])
def test_an_unreadable_layout_is_refused_rather_than_admitted(arm):
    """A volume that does not report its layout is NOT a contiguous one.

    MEASURED admitting on the complex arm before this clause was fixed: the
    contiguity test read ``flags is not None and not getattr(flags,
    "c_contiguous", True)``, so a volume exposing only ``dtype`` and ``shape`` was
    ADMITTED with an EMPTY reason tuple. The real arm always failed closed here;
    the two now agree, and both are pinned.
    """
    dtype = np.complex64 if arm == "complex" else np.float32

    class _NoFlags:
        def __init__(self):
            self.dtype = np.dtype(dtype)
            self.shape = _STUB_SHAPE

    grid = _StubGrid(**({"has_bloch": True, "k_point": (0.3, 0.0, 0.0)}
                        if arm == "complex" else {}))
    fields = _StubFields(grid, dtype=dtype, Bx=_NoFlags())
    verdict = (special_kz.beta_bloch_pml_curl_coverage(
        fields, _StubPML(), "step_B", _StubResidency(), _record())
        if arm == "complex" else
        special_kz.beta_pml_curl_coverage(fields, _StubPML(), "step_B",
                                          _StubResidency()))
    assert not verdict.covered
    assert any("not C-contiguous" in reason for reason in verdict.reasons), \
        verdict.reasons


def test_a_non_three_dimensional_shape_is_refused_on_the_complex_arm():
    """The plan multiplies ``shape[0..2]``; the predicate must refuse first.

    MEASURED admitting: with a 2-D ``grid.shape`` AND 2-D volumes the per-volume
    clause is silent (they agree), ``_coefficient_reasons`` is skipped by its own
    ``len(shape) == 3`` guard, and the verdict was ``covered=True`` with NO reasons
    — an ``IndexError`` waiting in ``BetaBlochPmlCurlPlan.__init__``.
    """
    grid = _StubGrid(shape=(8, 6), has_bloch=True, k_point=(0.3, 0.0, 0.0))
    fields = _StubFields(grid, dtype=np.complex64)
    verdict = special_kz.beta_bloch_pml_curl_coverage(
        fields, _StubPML(), "step_B", _StubResidency(), _record())
    assert not verdict.covered
    assert any("not three-dimensional" in reason for reason in verdict.reasons), \
        verdict.reasons


def test_an_unreadable_conductivity_table_is_refused_rather_than_admitted():
    grid = _StubGrid()
    fields = _StubFields(grid)
    del fields.__class__.condfac_for  # noqa: B010 - restored below
    try:
        verdict = special_kz.beta_pml_curl_coverage(fields, _StubPML(), "step_B",
                                                    _StubResidency())
        assert not verdict.covered
        assert any("not an absent one" in reason for reason in verdict.reasons)
    finally:
        _StubFields.condfac_for = lambda self, component: None


def test_the_real_constitutive_companion_is_admitted_and_inherits_the_clauses(
        monkeypatch):
    """It delegates the ARITHMETIC; it must not delegate the admission.

    Policy pinned for the same reason as the two baselines above: the admission
    half of this test measures the shell's environment unless it is.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    grid = _StubGrid()
    admitted = special_kz.beta_run_constitutive_coverage(
        _StubFields(grid), _StubPML(), "H", _StubResidency())
    assert admitted.covered, admitted.reasons
    for label, fields_kwargs, fragment in (
            ("offdiagonal", {"has_offdiagonal_epsilon": True},
             "implicit-i trick no longer cancels"),
            ("dispersion_on_E", {"has_polarizations": True},
             "update_E's source is")):
        verdict = special_kz.beta_run_constitutive_coverage(
            _StubFields(_StubGrid(), **fields_kwargs), _StubPML(), "E",
            _StubResidency())
        assert not verdict.covered, label
        assert any(fragment in reason for reason in verdict.reasons), \
            (label, verdict.reasons)
    zero = special_kz.beta_run_constitutive_coverage(
        _StubFields(_StubGrid(beta=0.0)), _StubPML(), "H", _StubResidency())
    assert not zero.covered
    assert any("grid.beta is zero" in reason for reason in zero.reasons)


def test_the_complex_constitutive_admits_a_complex_beta_run(monkeypatch):
    """THIS TEST USED TO PIN THE REFUSAL, and the inversion is the whole change.

    Until 2026-08-19 ``beta_run_complex_constitutive_coverage`` refused every
    configuration it was shown and this asserted the refusal string. It is a real
    restatement now — ``complex_fields``' clause set with clause 12 inverted — so the
    positive control comes first: the complex stub baseline is ADMITTED on both
    sides. The refusal rows that follow are then about clauses rather than about a
    predicate that says no to everything.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    grid = _StubGrid(has_bloch=True, k_point=(0.3, 0.0, 0.0))
    fields = _StubFields(grid, dtype=np.complex64)
    for side in ("H", "E"):
        verdict = special_kz.beta_run_complex_constitutive_coverage(
            fields, _StubPML(), side, _StubResidency(), _record())
        assert verdict.covered, (side, verdict.reasons)


@pytest.mark.parametrize("label,grid_kwargs,fields_kwargs,side,fragment", [
    # Clause 12, INVERTED: this product exists only for a beta run.
    ("beta_zero", {"beta": 0.0}, {}, "E", "grid.beta is zero"),
    # Clause 2, INVERTED: real storage belongs to the real companion. Both halves
    # of the disjunction have to be off — a Bloch k alone puts a run here.
    ("real_storage", {"has_bloch": False}, {"force_complex_fields": False}, "H",
     "storage is real float32"),
    # The fold is folded_beta's, on the constitutive pair as on the curls.
    ("folded", {"_symmetry": True}, {}, "H", "a mirror plane is active"),
    # E-side only: the off-diagonal row product is not element-wise. The CURL
    # admits the same configuration, which the gate's refusals leg measures.
    ("offdiagonal_E", {}, {"has_offdiagonal_epsilon": True}, "E",
     "off-diagonal chi1inv row"),
    # Module-wide clause 7: complex-storage ADE is a later tranche.
    ("susceptibility", {}, {"has_polarizations": True}, "E",
     "complex-storage ADE"),
    ("nonlinearity", {}, {"has_nonlinearity": True}, "H", "chi2/chi3"),
    ("bfast", {"bfast_active": True}, {}, "H", "BFAST"),
    ("k_on_the_invariant_axis", {"k_point": (0.3, 0.0, 0.2)}, {}, "H",
     "rides the invariant axis"),
])
def test_the_complex_constitutive_refuses_each_clause_by_name(
        monkeypatch, label, grid_kwargs, fields_kwargs, side, fragment):
    """Each clause of the restatement, refused BY NAME against the admitted stub.

    The baseline above is what makes these rows mean anything: each asserts only
    that ITS fragment appears among the reasons, and the reasons accumulate, so a
    predicate that refused everything would satisfy the whole matrix.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    kwargs = {"has_bloch": True, "k_point": (0.3, 0.0, 0.0)}
    kwargs.update(grid_kwargs)
    grid = _StubGrid(**kwargs)
    fields = _StubFields(grid, dtype=np.complex64, **fields_kwargs)
    verdict = special_kz.beta_run_complex_constitutive_coverage(
        fields, _StubPML(), side, _StubResidency(), _record())
    assert not verdict.covered, label
    assert any(fragment in reason for reason in verdict.reasons), \
        (label, verdict.reasons)


def test_an_unknown_sub_step_raises_rather_than_refusing_quietly():
    grid = _StubGrid()
    fields = _StubFields(grid)
    with pytest.raises(ValueError):
        special_kz.beta_pml_curl_coverage(fields, None, "update_H")


def test_the_family_registers_eight_wired_arms_over_all_four_slots():
    """WIRED as of tranche 2, and SYMMETRIC as of 2026-08-19.

    Eight rows: a real and a complex arm on each of the four slots — two curl arms
    per curl slot, two constitutive arms per side. The count did not change when the
    complex constitutive companion landed, and that is the point of pinning it here:
    the two complex constitutive arms were already registered while their predicate
    refused unconditionally, so a composer could say WHY ``update_H`` was on the
    array path. Building the product changed what those arms ANSWER, not how many
    there are.
    """
    from meep_gpu.metal_kernels import arms

    assert all(spec.wired for spec in special_kz.ARMS)
    assert len(special_kz.ARMS) == 8
    registered = {(spec.family, spec.slot) for spec in arms.registered()}
    assert ("special_kz_real", "step_B") in registered
    assert ("special_kz_real", "update_E") in registered
    assert ("special_kz_complex", "step_D") in registered
    assert ("special_kz_complex", "update_H") in registered
    assert {spec.slot for spec in special_kz.ARMS} == {
        "step_B", "step_D", "update_H", "update_E"}


def test_the_complex_constitutive_plan_is_the_CERTIFIED_class(monkeypatch):
    """The delegation, pinned as a type rather than described in a docstring.

    The whole claim of this companion is that it costs no kernel: the plan it
    returns must BE ``complex_fields``' certified plan class, built by that module's
    own function table. A plan of any other type would mean a second complex
    constitutive body had appeared in the tree, which is the outcome the docstring
    says cannot happen.
    """
    from meep_gpu.metal_kernels import complex_fields

    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    grid = _StubGrid(has_bloch=True, k_point=(0.3, 0.0, 0.0))
    fields = _StubFields(grid, dtype=np.complex64)
    for side in ("H", "E"):
        plan = special_kz.plan_beta_run_complex_constitutive(
            fields, _StubPML(), side, _StubResidency(), probe=_record())
        assert isinstance(plan, complex_fields.ComplexConstitutivePlan), (side, plan)


def test_the_complex_constitutive_refuses_without_a_probe_artifact():
    """A missing or unlicensable probe is a refusal, never a defaulted arm.

    The EXPANSION arm is a measured property of this host's reference. The
    constitutive companion binds it exactly as the curl does, so it must refuse the
    same unlicensable record rather than inheriting whatever the curl resolved.
    """
    grid = _StubGrid(has_bloch=True, k_point=(0.3, 0.0, 0.0))
    fields = _StubFields(grid, dtype=np.complex64)
    verdict = special_kz.beta_run_complex_constitutive_coverage(
        fields, _StubPML(), "H", _StubResidency(), probe={})
    assert not verdict.covered
    assert any("expansion probe" in reason for reason in verdict.reasons), \
        verdict.reasons
    assert special_kz.plan_beta_run_complex_constitutive(
        fields, _StubPML(), "H", _StubResidency(), probe={}) is None


# ---------------------------------------------------------------------------
# 7. The arithmetic, in process, against stepping.py
# ---------------------------------------------------------------------------

def _engine_case(complex_storage, beta=BETA_KZ2D, courant=0.34,
                 k=(0.0, 0.0, 0.0), boundaries="periodic", seed=20260815):
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(2.0, 1.6, 0.0), boundaries=boundaries,
                dimensions=2, courant=courant, k_point=k, beta=beta, xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    eps = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(eps, (np.float32(1.0) / eps).astype(np.float32))
    pml = PML(grid=grid, thickness=tuple((2, 2) if grid.shape[a] >= 6 else (0, 0)
                                         for a in range(3)))
    rng = np.random.default_rng(seed)
    for name in _STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        real = (rng.standard_normal(grid.shape) * 0.37).astype(np.float32)
        if complex_storage:
            imag = (rng.standard_normal(grid.shape) * 0.29).astype(np.float32)
            out = np.zeros(grid.shape, dtype=np.complex64)
            view = out.reshape(-1).view(np.float32).reshape(-1, 2)
            view[:, 0] = real.reshape(-1)
            view[:, 1] = imag.reshape(-1)
            array[...] = out
        else:
            array[...] = real
    return grid, fields, pml


_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
          "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")


def _words(array):
    return np.ascontiguousarray(array).reshape(-1).view(np.uint32)


@requires_mps
@pytest.mark.parametrize("complex_storage,k", [(False, (0.0, 0.0, 0.0)),
                                               (True, (0.9205, 0.0, 0.0))])
def test_one_beta_curl_pair_is_byte_identical_to_stepping(monkeypatch,
                                                          complex_storage, k):
    """Both sub-steps on the device against both on the array path, in one process.

    uint32 word equality on targets AND auxiliaries, with an assertion that the
    step MOVED STATE — a no-op agreeing with a no-op is trivially identical.
    """
    from meep_gpu import stepping
    from meep_gpu.metal_kernels import device, subnormal

    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    grid, fields, pml = _engine_case(complex_storage, k=k, courant=0.35)
    codes = [1 if kind == "metallic" else 0
             for kind in stepping._boundary_kinds(grid, pml)]
    probe = {"backend": special_kz.PROBE_BACKEND,
             "patterns": dict({name: "AMBIGUOUS_BOTH"
                               for name in special_kz.BETA_PROBE_PATTERNS},
                              c8_mul_c8="FMA_V1")}

    for sub_step in ("step_B", "step_D"):
        spec = special_kz.SUB_STEPS[sub_step]
        before = {n: np.array(getattr(fields, n), copy=True) for n in _STATE}
        residency = device.Residency()
        plan = (special_kz.plan_beta_bloch_pml_curl(fields, pml, sub_step,
                                                    residency, probe=probe)
                if complex_storage else
                special_kz.plan_beta_pml_curl(fields, pml, sub_step, residency))
        assert plan is not None, "the engine route refused a covered configuration"
        plan.run()
        residency.sync_out()
        got = {n: np.array(getattr(fields, n), copy=True) for n in _STATE}

        for name, value in before.items():
            getattr(fields, name)[...] = value
        getattr(stepping, sub_step)(fields, pml)

        names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
        moved = sum(int(np.count_nonzero(_words(getattr(fields, n))
                                         != _words(before[n]))) for n in names)
        assert moved > 0, "VACUOUS: the array path moved no state"
        differing = {n: int(np.count_nonzero(_words(got[n])
                                             != _words(getattr(fields, n))))
                     for n in names}
        assert not {n: c for n, c in differing.items() if c}, differing


@requires_mps
def test_a_beta_zero_build_reproduces_the_certified_plain_curl(monkeypatch):
    """The identity arm: with the term compiled out, this IS the certified kernel."""
    from meep_gpu.metal_kernels import device, launch, subnormal

    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu import stepping

    grid, fields, pml = _engine_case(False, beta=0.0, courant=0.35)
    codes = [1 if kind == "metallic" else 0
             for kind in stepping._boundary_kinds(grid, pml)]
    spec = special_kz.SUB_STEPS["step_B"]
    arrays = {n: np.array(getattr(fields, n), copy=True)
              for n in tuple(spec["targets"]) + tuple(spec["sources"])}
    arrays.update({"fu_" + n: np.array(getattr(fields, "fu_" + n), copy=True)
                   for n in spec["targets"]})
    flat = {f"{stem}_{axis}": np.asarray(
        getattr(pml, f"{stem}_{axis}{spec['suffix']}")).reshape(-1).astype(np.float32)
        for axis in "xyz" for stem in ("kms", "sinv")}

    beta_arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
    residency = device.Residency()
    special_kz.plan_beta_pml_curl_from_arrays(
        "step_B", beta_arrays, flat, codes, float(grid.dt / grid.dx), 0.0, 0.0,
        residency, has_beta=False).run()
    residency.sync_out()

    plain_arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
    residency = device.Residency()
    launch.plan_from_arrays("step_B", plain_arrays, flat, codes,
                            float(grid.dt / grid.dx), residency).run()
    residency.sync_out()

    names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
    moved = sum(int(np.count_nonzero(_words(plain_arrays[n]) != _words(arrays[n])))
                for n in names)
    assert moved > 0, "VACUOUS: the certified curl moved no state"
    for name in names:
        assert not np.count_nonzero(_words(beta_arrays[name])
                                    != _words(plain_arrays[name])), name
