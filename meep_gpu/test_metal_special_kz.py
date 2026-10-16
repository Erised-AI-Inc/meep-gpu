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


def _kernel_increment(word, partner, complex_storage):
    """The negated increment the emitted pass forms from one beta word, in numpy.

    Real: ``bi = -(w * g)``. Complex: ``bq = (-(w * g.im), w * g.re)``, then
    ``bi = -bq`` per part -- the quarter turn and the outer negation, each a sign-bit
    negation or one float32 product, exactly the statements the pass emits.
    """
    w = np.float32(word)
    if not complex_storage:
        return -(w * partner)
    out = np.empty_like(partner)
    out.real = -(-(w * partner.imag))
    out.imag = -(w * partner.real)
    return out


def _partner(complex_storage):
    """Unit, signed-zero and random partner words, so a sign of zero is exercised."""
    values = np.array([0.0, -0.0, 1.0, -1.0, 1e-30, -2.5, 3.7e5], dtype=np.float32)
    rng = np.random.default_rng(20261008)
    real = np.concatenate([np.repeat(values, values.size),
                           rng.standard_normal(512).astype(np.float32)])
    if not complex_storage:
        return real
    imag = np.concatenate([np.tile(values, values.size),
                           rng.standard_normal(512).astype(np.float32)])
    out = np.empty(real.size, dtype=np.complex64)
    out.real = real
    out.imag = imag
    return out


@pytest.mark.parametrize("beta,dt", [(BETA_KZ2D, 1.0 / 24.0),
                                     (BETA_GRATING, 0.034),
                                     (-0.3907, 0.0233),
                                     (0.2, 0.35 / 10.0)])
@pytest.mark.parametrize("magnetic", [True, False])
@pytest.mark.parametrize("complex_storage", [True, False])
def test_the_coefficient_matches_steppings_own_rounding(beta, dt, magnetic,
                                                        complex_storage):
    """Byte-identical to the increment ``stepping._special_kz_beta_term`` returns.

    The array path forms ``sign*2*pi*beta*dt`` in f64 and rounds it ONCE to the
    storage's realnum; real storage multiplies the partner by it, complex storage
    takes a quarter turn of the partner's parts by it (``-it`` on the D side), and
    the increment is returned negated. This forms the same increment from this
    module's words with the kernel's own statements (:func:`_kernel_increment`) and
    compares every word, over partners that include signed zeros, so a rounding
    moved, doubled or dropped anywhere in the transcription fails here rather than in
    a gate run.
    """
    from meep_gpu import stepping

    fields = _Fields(beta, dt)
    ours = special_kz.beta_curl_coefficients(beta, dt, magnetic, complex_storage)
    partner = _partner(complex_storage)
    for index, sign in enumerate((1.0, -1.0)):
        assert isinstance(ours[index], float), ours
        theirs = stepping._special_kz_beta_term(fields, partner, sign,
                                                magnetic=magnetic)
        mine = _kernel_increment(ours[index], partner, complex_storage)
        assert not np.count_nonzero(
            mine.view(np.uint32) != np.asarray(theirs).view(np.uint32)), (
            f"{mine[:4]!r} != {theirs[:4]!r}")


@pytest.mark.parametrize("magnetic", [True, False])
@pytest.mark.parametrize("complex_storage", [True, False])
def test_the_coefficient_matches_stepping_on_numpy_typed_grid_scalars(
        magnetic, complex_storage):
    """The expression is the array path's, on the objects the array path holds --
    not on ``float()`` copies of them.

    THE PATH IS LIVE. ``Grid._resolve_beta`` coerces ``beta`` to a Python float;
    NOTHING coerces ``dt``, which is ``courant / resolution``, so
    ``Grid(courant=numpy.float32(...))`` leaves ``grid.dt`` a ``numpy.float32`` and
    hands it straight to both ``stepping._special_kz_beta_term`` and
    ``beta_curl_coefficients``. Under NEP 50 a numpy-typed operand rounds the
    product chain EARLY, so a transcription that widens through ``float()`` first
    computes the whole chain in f64 and rounds once -- a different float32 word.

    MEASURED 2026-08-15 before the casts came out: 14,926 of 40,000 random
    ``numpy.float32`` (beta, dt, sign) draws produced a different coefficient.
    ``beta = BETA_GRATING`` with ``courant = 0.34`` DISCRIMINATES on BOTH signs.

    The ``type(grid.dt)`` assertion is deliberate -- if ``Grid`` ever coerces
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

    fields = _Fields(grid.beta, grid.dt)
    ours = special_kz.beta_curl_coefficients(grid.beta, grid.dt, magnetic,
                                             complex_storage)
    partner = _partner(complex_storage)
    for index, sign in enumerate((1.0, -1.0)):
        theirs = stepping._special_kz_beta_term(fields, partner, sign,
                                                magnetic=magnetic)
        mine = _kernel_increment(ours[index], partner, complex_storage)
        assert not np.count_nonzero(
            mine.view(np.uint32) != np.asarray(theirs).view(np.uint32)), (
            f"numpy-typed dt: {mine[:4]!r} != {theirs[:4]!r}. The transcription is "
            f"normalising operands stepping does not normalise")


def test_the_complex_words_are_one_real_word_per_sign_and_the_d_side_negates():
    """A quarter turn by a REAL coefficient, never a complex pair.

    ``part_products.quarter_turn`` scales the partner's parts by one realnum and
    negates one product, so the kernel binds one float per sign. The D side's word is
    the B side's negated (``quarter_turn(-coefficient, g)``), and f64 negation
    commutes with the f32 rounding exactly, so the two words differ in the sign bit
    alone.
    """
    for beta, dt in ((BETA_KZ2D, 1.0 / 24.0), (BETA_GRATING, 0.034)):
        b_side = special_kz.beta_curl_coefficients(beta, dt, magnetic=True,
                                                   complex_storage=True)
        d_side = special_kz.beta_curl_coefficients(beta, dt, magnetic=False,
                                                   complex_storage=True)
        real = special_kz.beta_curl_coefficients(beta, dt, magnetic=False,
                                                 complex_storage=False)
        for b_word, d_word, r_word in zip(b_side, d_side, real):
            assert isinstance(b_word, float) and isinstance(d_word, float)
            assert (np.float32(b_word).view(np.uint32) ^ np.uint32(0x80000000)
                    == np.float32(d_word).view(np.uint32))
            assert np.float32(b_word).view(np.uint32) == np.float32(r_word).view(
                np.uint32)


def _pass_block(source):
    """The folded pass's text, between its two marker lines."""
    from meep_gpu.metal_kernels import folded_pass

    start = source.index(folded_pass.FOLDED_PASS_BEGIN)
    return source[start:source.index(folded_pass.FOLDED_PASS_END, start)]


def test_the_term_carries_no_dtdx():
    """An analytic derivative, not a finite difference (``_special_kz_beta_term``)."""
    for backward in (False, True):
        block = _pass_block(special_kz.beta_curl_source((0, 0, 0), backward, True))
        assert "beta_plus" in block and "beta_minus" in block
        assert "dtdx" not in block, block


def test_the_pass_runs_after_the_forms_and_before_every_store():
    """MEEP's beta pass follows every component's curl (step_db.cpp:161-175).

    The pass reads the forms' ``n``/``v`` registers and runs in front of the first
    guarded ``fu`` store, so both stores carry the pass; the curl's own statements are
    the certified emitter's, unchanged.
    """
    from meep_gpu.metal_kernels import folded_pass, shaders

    for backward in (False, True):
        for codes in ((0, 0, 0), (1, 1, 0), (1, 0, 1)):
            source = special_kz.beta_curl_source(codes, backward, True)
            certified = shaders.curl_source(codes, backward)
            start = source.index(folded_pass.FOLDED_PASS_BEGIN)
            end = source.index(folded_pass.FOLDED_PASS_END)
            assert source.index("float v2 = ") < start
            assert end < source.index(folded_pass.FOLDED_PASS_ANCHOR)
            stripped = (source[:start] + source[end:].split("\n", 2)[2]).replace(
                "beta_pml_curl_step(", "pml_curl_step(")
            stripped = stripped.replace(
                "    constant float&     beta_plus  [[buffer(20)]],\n"
                "    constant float&     beta_minus [[buffer(21)]],\n"
                "    constant ChunkBounds& cb [[buffer(22)]],\n",
                "    constant ChunkBounds& cb [[buffer(20)]],\n")
            assert stripped == certified, (codes, backward)


# ---------------------------------------------------------------------------
# 3. The three byte-invisible defects, pinned by source text
# ---------------------------------------------------------------------------

def test_the_beta_partner_is_the_unshifted_centre_operand():
    """Measured invisible on the z axis, so pinned here.

    ``beta`` is legal only on an effective-2-D grid, so the invariant axis has
    extent 1 and ``b_z`` IS ``b``: the gate measured that mutation at 0/2 caught
    and records it as a predicted null. The property still has to hold, so it is
    asserted on the SOURCE: the pass reads the stored centres ``b`` and ``a``, never a
    shifted or Bloch-connected name.
    """
    for backward in (False, True):
        block = _pass_block(special_kz.beta_curl_source((0, 0, 0), backward, True))
        assert "float bi0 = -(beta_plus * b);" in block
        assert "float bi1 = -(beta_minus * a);" in block
        cblock = _pass_block(special_kz.beta_bloch_curl_source(
            (0, 0, 0), backward, (2, 1, 0), True))
        assert "float2(-(beta_plus * b.y), beta_plus * b.x)" in cblock
        assert "float2(-(beta_minus * a.y), beta_minus * a.x)" in cblock
        for shifted in ("b_x", "b_z", "a_y", "a_z", "c_x", "c_y",
                        "b_dx", "b_dz", "a_dy", "a_dz"):
            assert f"* {shifted}" not in block + cblock


def test_the_real_arm_spells_subtraction_rather_than_a_negation():
    """No ``0.0f - x`` and no ``x * -1.0`` anywhere in the pass.

    The increment's one negation is ``-(...)``, a sign-bit operation on this host;
    ``0.0f - x`` misses 36/512 on the exhaustive signed-zero table.
    """
    for complex_storage in (False, True):
        source = (special_kz.beta_bloch_curl_source((0, 0, 0), False, (0, 0, 0), True)
                  if complex_storage
                  else special_kz.beta_curl_source((0, 0, 0), False, True))
        block = _pass_block(source)
        assert "0.0f -" not in block
        assert "* -1.0" not in block


def test_the_complex_pass_forms_no_complex_product():
    """``part_products.quarter_turn``: two real products and an exact negation.

    The complex pass calls no complex helper at all -- no ``c_mul`` of the beta word
    and the partner, which would round the cross terms the array path never forms.
    """
    for backward in (False, True):
        block = _pass_block(special_kz.beta_bloch_curl_source(
            (0, 0, 0), backward, (1, 0, 0), True))
        assert "c_mul" not in block, block
        assert "float2 bi0 = -bq0;" in block and "float2 bi1 = -bq1;" in block


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
    ``complex_fields.phase_arguments`` then the emitter — and asserts the result is in
    the enumeration. It needs no GPU: a source is a string.
    """
    import hashlib

    from meep_gpu import stepping
    from meep_gpu.grid import Grid
    from meep_gpu.metal_kernels import complex_fields
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
                phased, _ = complex_fields.phase_arguments(
                    complex_fields.bloch_phase_table(grid, kinds),
                    backward=bool(spec["backward"]))
                source = special_kz.beta_bloch_curl_source(
                    codes, spec["backward"], phased, has_beta)
                digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                assert digest in catalogue, (
                    f"{boundaries}/{k}/{sub_step}/beta{int(has_beta)}"
                    f" emits codes={codes} phased={phased}, which is NOT in "
                    f"enumerate_sources() — the gate would launch a "
                    f"specialisation it never fingerprinted")
            # The real arm's emitter is boundary-only and is exercised on every row.
            source = special_kz.beta_curl_source(codes, spec["backward"], has_beta)
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            assert digest in catalogue, (codes, sub_step, has_beta)


def test_the_enumeration_is_a_superset_of_what_the_emitter_can_produce():
    """...and the derived set is not merely large, it CONTAINS the emittable one."""
    from meep_gpu.metal_kernels import complex_fields

    emittable = set(special_kz._emittable_bloch_specialisations())
    for codes, phased in emittable:
        for axis in range(3):
            if codes[axis] == templates.METALLIC:
                assert phased[axis] == complex_fields.CONNECT_NONE, (codes, phased)
        assert phased[2] == complex_fields.CONNECT_NONE, (codes, phased)
    kinds = len(complex_fields.CONNECT_KINDS)
    # per invariant-axis code: (P, P) kinds**2, (P, M) and (M, P) kinds, (M, M) one
    assert len(emittable) == 2 * (kinds * kinds + 2 * kinds + 1), sorted(emittable)
    # two sides, both HAS_BETA arms
    assert len(special_kz.enumerate_sources()) == 32 + len(emittable) * 2 * 2


def test_the_mask_masks_both_planes_on_the_complex_arm():
    """A complex zero is assigned to BOTH words (stepping.py:1943)."""
    source = special_kz.beta_bloch_curl_source((1, 1, 1), True, (0, 0, 0), True)
    assert source.count("float2(0.0f, 0.0f) : curl") == 6


def test_the_complex_signature_fits_under_the_31_binding_ceiling():
    """A measured platform limit, not a style rule.

    Metal's buffer attribute indices run 0..30; a 32nd is a COMPILE ERROR. This
    kernel binds the certified complex curl's 24 and the two beta words. A re/im-split
    build would need nine more volumes, which is why the complex volumes are
    ``float2`` -- forced, not preferred.
    """
    from meep_gpu.metal_kernels import device

    source = special_kz.beta_bloch_curl_source((0, 0, 0), False, (1, 1, 0), True)
    indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
    assert indices == list(range(len(indices)))
    assert max(indices) < device.MAX_BUFFER_BINDINGS, indices
    assert len(indices) == 26, indices
    assert f"constant ChunkBounds& cb [[buffer({special_kz.COMPLEX_RECORD_BUFFER})]]" \
        in source
    assert source.count("device float2*") + source.count("device const float2*") == 9


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
    device = "cpu"

    def mirror(self, *args, **kwargs):
        return None


def _stub_record():
    """A valid chunk record for the stub shape: no layer runs, x and y resolved."""
    record = np.zeros(25, dtype=np.int32)
    record[24] = 0b011
    return record


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
        fields, _StubPML(), "step_B", _StubResidency())
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
        fields, _StubPML(), "step_B", _StubResidency())
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
        fields, _StubPML(), "step_B", _StubResidency())
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
        fields, _StubPML(), "step_B", _StubResidency())
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
            fields, _StubPML(), side, _StubResidency())
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
        fields, _StubPML(), side, _StubResidency())
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
            fields, _StubPML(), side, _StubResidency(),
            chunk_bounds=_stub_record())
        assert isinstance(plan, complex_fields.ComplexConstitutivePlan), (side, plan)


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

    for sub_step in ("step_B", "step_D"):
        spec = special_kz.SUB_STEPS[sub_step]
        before = {n: np.array(getattr(fields, n), copy=True) for n in _STATE}
        residency = device.Residency()
        record = _one_rank_record(fields, pml)
        plan = (special_kz.plan_beta_bloch_pml_curl(fields, pml, sub_step,
                                                    residency, chunk_bounds=record)
                if complex_storage else
                special_kz.plan_beta_pml_curl(fields, pml, sub_step, residency,
                                              chunk_bounds=record))
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
        residency, has_beta=False, chunk_bounds=_one_rank_record(fields, pml)).run()
    residency.sync_out()

    plain_arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
    residency = device.Residency()
    launch.plan_from_arrays("step_B", plain_arrays, flat, codes,
                            float(grid.dt / grid.dx), residency,
                            chunk_bounds=_one_rank_record(fields, pml)).run()
    residency.sync_out()

    names = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
    moved = sum(int(np.count_nonzero(_words(plain_arrays[n]) != _words(arrays[n])))
                for n in names)
    assert moved > 0, "VACUOUS: the certified curl moved no state"
    for name in names:
        assert not np.count_nonzero(_words(beta_arrays[name])
                                    != _words(plain_arrays[name])), name


def _one_rank_record(fields, pml):
    """The record of the one-rank division ``stepping`` steps these ``Fields`` with."""
    from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record

    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))
