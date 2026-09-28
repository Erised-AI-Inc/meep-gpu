"""Merge-bar tests for the Metal CYLINDRICAL COMPLEX family (Dcyl, complex64
storage at every m — m = 0 since 2026-09-04).

THE FAMILY IS ``complex_fields`` PLUS FIVE CYLINDRICAL ADDITIONS, so this file does
not re-test complex storage: that is byte-certified elsewhere. It tests the five
additions, the two things this family gets to assume, and the two things it costs.

**1. THE FIVE ADDITIONS, AGAINST ``stepping.py`` DIRECTLY.** The radial prefix, the
i*m/r coupling, the |m| = 1 axis-row replacement, the per-|m| near-axis rules and
the m = 0 arm (no coupling, the m = 0 axis rules as post-recurrence edits) are
compared as WORDS against the array path over the three m classes, both z
terminations, both branches of ``accurate_fields_near_cylorigin`` and multi-launch
runs. Every leg asserts a VACUITY FLOOR: a plan that wrote nothing would otherwise
agree with anything, and half this family's work is writing zeros.

**2. THE COMPARISON HAS TEETH, AND THAT IS MEASURED RATHER THAN HOPED.** Five
mutations are planted in the shipped source and each must be CAUGHT. Without them
"0 differing words" is compatible with a plan that never launched, a reference that
never moved, and a kernel whose cylindrical blocks are all dead.

**3. THE TWO THINGS THE KERNEL ASSUMES ABOUT THE r AXIS.** It compiles ``BCX =
METALLIC`` where ``_boundary_kinds`` says ``'axis'``. The FAR ghost half is a SOURCE
IDENTITY and is asserted as one, by reading ``stepping``'s branch text — so a future
split of that branch fails here rather than in a field. The NEAR ghost half is NOT an
identity and is MEASURED on the array path, in this file, at both m classes.

**4. THE ONE THING IT COSTS.** The prefix is a sequential HOST scan, so every launch
syncs one source volume out and the prefix in. That is counted, and the counter is
ARMED: a plan whose prefix is not refreshed after its source changes is shown to
diverge, so ``prefix_syncs == launches`` is evidence rather than bookkeeping.

**5. THE INVERSIONS.** Six families to stay clear of, and the separations are pinned
as predicate outcomes rather than as prose — including the one against the
cylindrical REAL product, which is clause 2 (storage) inverted in BOTH directions:
a complex64 m = 0 run is ADMITTED here and refused there by name, a float32 run is
refused here by name. Until 2026-09-04 clause 14 refused m = 0 here as well, and
that left the complex64 m = 0 configuration refused by both families; the test that
pinned that refusal now pins the admission.

Device-touching tests are skipped without MPS. Nothing in this file asserts anything
about GENERATED CODE: ``torch.mps.compile_shader`` exposes no AIR and no ISA, so
every behavioural leg catches a wrong answer and never a wrong instruction.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pytest

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "parity", "meep_gpu")
_PARITY = os.path.abspath(_PARITY)
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms, complex_fields, cylindrical_complex as cyl, device, launch,
    preconditions, registry, shaders, templates,
)

ENVIRONMENT = matrix.prepare_environment()

#: The expansion arm every source leg builds with. Read from the SAME artifact the
#: predicates read, so a test cannot certify a body the family would not launch.
PROBE = cyl.load_expansion_probe()
EXPANSION = cyl.expansion_from_probe(PROBE) if PROBE else None

requires_probe = pytest.mark.skipif(
    EXPANSION is None,
    reason="no cylindrical-complex expansion probe artifact on this host")

VOLUMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
           "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
           "fu_Dx", "fu_Dy", "fu_Dz")


def _mps_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch.backends, "mps", None), "is_available",
                        lambda: False)())


requires_mps = pytest.mark.skipif(not _mps_available(),
                                  reason="no MPS device on this host")


def _words(array) -> np.ndarray:
    return np.ascontiguousarray(array).view(np.float32).reshape(-1).view(np.uint32)


def _differing(left, right) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def build(m: int = 1, z_kind: str = "metallic", accurate: bool = False,
          seed: int = 11, nr: int = 18, nz: int = 22, courant: float = 0.37,
          complex_storage: bool = True, r_pml=(0, 4)):
    """A Dcyl ``(Fields, PML)`` pair seeded in the PHYSICAL BAND.

    NOT COSMETIC. Zero-init is a fixed point of the near-axis zeroing and of half the
    recurrence, so a plan built on zeros is a no-op that agrees with a no-op. Every
    volume is seeded and every leg asserts words actually moved.
    """
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=1.0, cell_size=(float(nr), 0.0, float(nz)),
                cylindrical=True, m=int(m), boundaries={"z": z_kind},
                courant=float(courant),
                accurate_fields_near_cylorigin=bool(accurate), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    shape = tuple(grid.shape)
    fields.set_isotropic_epsilon_volume(np.full(shape, np.float32(2.25)),
                                        np.full(shape, np.float32(1.0 / 2.25)))
    fields.enable_pml_storage()
    rng = np.random.default_rng(seed)
    for name in VOLUMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if array.dtype.kind == "c":
            array.real[...] = rng.standard_normal(array.shape).astype(np.float32)
            array.imag[...] = rng.standard_normal(array.shape).astype(np.float32)
        else:
            array[...] = rng.standard_normal(array.shape).astype(array.dtype)
    return fields, PML(grid=grid, thickness={"x": tuple(r_pml), "z": 4})


# ---------------------------------------------------------------------------
# Source structure: what is specialised, what is a runtime uniform, and how many
# ---------------------------------------------------------------------------

@requires_probe
def test_the_specialisation_enumeration_is_CLOSED_at_twelve():
    """``12``, not ``12 x UNBOUNDED``. This is the whole point of the round.

    Triton carries ``ZERO_ROWS`` as a ``tl.constexpr`` equal to ``abs(m)``, so its
    variant count cannot be enumerated by any author. Here it is a
    ``constant uint&`` uniform — licensed by the threshold measurement in
    ``probe_metal_dynamic_loop_arity`` on THIS host — and what remains is
    ``BACKWARD x BCZ x M_CLASS``, three m classes since the ``M_ZERO`` arm landed
    (eight sources before 2026-09-04).
    """
    sources = cyl.enumerate_cylindrical_sources(EXPANSION)
    assert len(sources) == 12, sorted(sources)
    assert len(set(sources.values())) == 12, "two specialisations emit one source"
    assert sum("/m0/" in label for label in sources) == 4
    for label, source in sources.items():
        assert "__" not in "".join(
            line for line in source.splitlines() if line.strip().startswith("__")), (
            f"{label} still holds a placeholder")


@requires_probe
def test_the_m_zero_body_carries_no_coupling_and_the_m_zero_axis_rules():
    """The ``M_ZERO`` arm is a THIRD BODY, transcribed from the array path's own
    ``grid.m != 0`` / ``m == 0`` branches, not the |m| >= 1 body at m = 0.

    Absent: the i*m/r block (a zero coefficient row through ``c_mul`` would still
    turn a ``-0.0`` curl word into ``+0.0``), the |m| = 1 increment, the |m| >= 2
    hold. Present, AFTER the recurrence and on the FIELD registers only: ``Bx[r=0]
    = 0`` on the B side; ``Dz[r=0] += axis_coef * Hp[r=0]`` as a post-add and then
    ``Dy[r=0] = 0`` on the D side.
    """
    b_side = cyl.cylindrical_curl_source(templates.METALLIC, False, cyl.M_ZERO,
                                         EXPANSION)
    d_side = cyl.cylindrical_curl_source(templates.METALLIC, True, cyl.M_ZERO,
                                         EXPANSION)
    for source in (b_side, d_side):
        assert "float2 q0 = c0[i];" not in source
        assert "c_mul(q0, c)" not in source and "c_mul(q2, a)" not in source
        assert "curl0 = at_x ? -inc : curl0;" not in source
        assert "curl1 = at_x ? -inc : curl1;" not in source
        assert "bool near = (i < int(zrows));" not in source
        # The rows and the |m| = 1 scalars stay BOUND (one signature, three arms).
        assert "device const float2* c0        [[buffer(10)]]," in source
        assert "constant float&      axis_coef [[buffer(26)]]," in source
    assert "v0 = at_x ? float2(0.0f, 0.0f) : v0;" in b_side
    assert "inc0" not in b_side
    assert "float2 inc0 = c_mul_coefficient_left(axis_coef, b);" in d_side
    assert "v2 = at_x ? (v2 + inc0) : v2;" in d_side
    assert "v1 = at_x ? float2(0.0f, 0.0f) : v1;" in d_side
    assert "n1 = at_x" not in d_side and "n2 = at_x" not in d_side, (
        "the m = 0 rules touch the FIELDS only, never fu_ (stepping :586-587, :661)")
    # The post-add sits AFTER the recurrence's `v2` and BEFORE the store.
    assert (d_side.index("float2 v2 = ") < d_side.index("v2 = at_x ? (v2 + inc0)")
            < d_side.index("f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;"))
    # And the |m| >= 1 bodies still carry the coupling.
    for arm in (cyl.M_ONE, cyl.M_MANY):
        assert "float2 q0 = c0[i];" in cyl.cylindrical_curl_source(
            templates.METALLIC, True, arm, EXPANSION)


def test_the_m_zero_axis_coefficient_is_the_array_paths_own_scalar():
    """``4.0 * (dt/dx)`` formed in float64, rounded ONCE at the binding — the word
    NumPy forms when it casts the Python float to complex64 before the multiply."""
    for dtdx in (0.37, 0.5, 0.2718281828, 0.25):
        coefficient = cyl.axis_coefficient(dtdx)
        assert coefficient == 4.0 * dtdx
        assert np.float32(coefficient) == np.complex64(4.0 * dtdx).real
        # Scaling by four is exact, so the in-kernel spelling `4.0f * dtdx` is the
        # same word; the gate carries that as a null control.
        assert np.float32(coefficient) == np.float32(4.0) * np.float32(dtdx)


@requires_probe
def test_zero_rows_is_a_BOUND_UNIFORM_and_never_a_baked_literal():
    """The near-axis count must not appear in the source at all.

    A family that baked it would be back to an unbounded variant count, and the
    regression would be invisible — the sources would still compile and still be
    correct, one per m. The check is on the SOURCE TEXT because that is the only
    place this backend lets a claim about specialisation be held.
    """
    sources = cyl.enumerate_cylindrical_sources(EXPANSION)
    many = [s for label, s in sources.items() if "/mN/" in label]
    assert many, "no M_MANY specialisation was emitted"
    for source in many:
        assert "int(zrows)" in source
        assert "constant uint&       zrows" in source
    for m, accurate, expected in ((0, False, 0), (0, True, 0), (1, False, 0),
                                  (-1, False, 0), (2, False, 2),
                                  (3, False, 3), (3, True, 1), (5, False, 5),
                                  (-5, False, 5)):
        assert cyl.zero_rows(m, accurate) == expected, (m, accurate)


@requires_probe
def test_the_binding_count_is_what_the_signature_actually_binds():
    """``CURL_BINDINGS`` is CHECKED against the emitted source, not asserted.

    The ``float2`` design rests on the margin under Metal's 31-buffer ceiling, and
    the number used to live as a literal in three places on the complex family. One
    home, checked here, so a binding added to the signature is a failing test rather
    than a compile error a reader has to attribute.
    """
    source = cyl.cylindrical_curl_source(templates.METALLIC, False, cyl.M_ONE,
                                         EXPANSION)
    bound = {int(n) for n in re.findall(r"\[\[buffer\((\d+)\)\]\]", source)}
    assert bound == set(range(cyl.CURL_BINDINGS)), sorted(bound)
    assert cyl.CURL_BINDINGS <= device.MAX_BUFFER_BINDINGS
    # The certified complex curl binds 23; this one binds four more (the prefix, the
    # two i*m/r rows and, since 2026-09-04, the m = 0 axis coefficient) plus three
    # scalars. Stated as a relation so a change to either is visible here.
    assert cyl.CURL_BINDINGS == complex_fields.CURL_BINDINGS + 4 == 27


@requires_probe
def test_the_contraction_directive_is_present_and_the_two_modes_differ():
    """The pragma is mandatory and is a property of the SOURCE on this backend.

    Not a style check: without it the compiler contracts and byte-identity fails
    (measured by the real tranche at 37,483-37,568 differing words). Spelled exactly
    once per source, from ``shaders``' single home.
    """
    off = cyl.cylindrical_curl_source(templates.METALLIC, False, cyl.M_ONE,
                                      EXPANSION, shaders.CONTRACT_OFF)
    fast = cyl.cylindrical_curl_source(templates.METALLIC, False, cyl.M_ONE,
                                       EXPANSION, shaders.CONTRACT_FAST)
    assert off.count("#pragma clang fp contract") == 1
    assert "#pragma clang fp contract(off)" in off
    assert "#pragma clang fp contract(fast)" in fast
    assert off != fast


@requires_probe
def test_the_kernel_never_divides_and_never_takes_a_max():
    """Two Metal facts forced the i*m/r row onto the host; this is the consequence.

    ``fast::divide`` is measured-divergent on this platform and ``max`` returns the
    SECOND operand when both are zeros, disagreeing with ``numpy.maximum``. The row's
    clamp and divide therefore stay in ``imr_coefficient_row``, on the host, in
    float64, rounded once. If either ever appears in the kernel the reason has been
    forgotten.
    """
    for source in cyl.enumerate_cylindrical_sources(EXPANSION).values():
        body = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("//"))
        assert "divide" not in body
        assert not re.search(r"\bmax\s*\(", body)
        assert not re.search(r"\bmin\s*\(", body)


@requires_probe
def test_negation_is_unary_minus_and_not_the_triton_workaround():
    """Measured on THIS backend, not inherited from the Triton track's opposite.

    Triton spells negated addends ``x * -1.0`` because TRITON lowers ``-x`` as
    ``0.0 - x`` and canonicalizes signed zeros. The family probe's ``negation`` leg
    compiled all three spellings here: ``-x`` and ``x * -1.0f`` both matched
    ``numpy.negative`` 0/1024 in BOTH contraction modes and ``0.0f - x`` missed one
    word — the ``+0.0`` corner. The array path performs a sign flip, so ``-x`` is the
    transcription and the refuted spelling must not appear.
    """
    for source in cyl.enumerate_cylindrical_sources(EXPANSION).values():
        body = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("//"))
        assert "0.0f - " not in body


# ---------------------------------------------------------------------------
# The r axis: one reading, one measurement
# ---------------------------------------------------------------------------

def test_the_FAR_ghost_is_a_source_identity_in_stepping():
    """``_shift_up`` has ONE branch over MIRROR, METALLIC and CYL_AXIS.

    THIS IS THE HALF THAT IS NOT MEASURED, because it is not a measurement — it is a
    reading, and this test is what keeps the reading true. The kernel compiles
    ``BCX = METALLIC`` on an axis ``_boundary_kinds`` calls ``'axis'``; if that
    branch is ever split, the compile-time choice silently stops matching the array
    path and no byte leg on an admitted configuration would say so.
    """
    import inspect

    source = inspect.getsource(stepping._shift_up)
    assert "if boundary in (MIRROR, METALLIC, CYL_AXIS):" in source, (
        "stepping._shift_up no longer serves CYL_AXIS from the METALLIC branch; the "
        "cylindrical kernel's BCX = METALLIC specialisation rests on that identity")


@pytest.mark.parametrize("m,z_kind", ((1, "metallic"), (3, "periodic")),
                         ids=("m1-metallic", "m3-periodic"))
def test_the_NEAR_ghost_is_unobservable_on_the_array_path(m, z_kind):
    """Forcing ``boundaries[0] = 'metallic'`` must change NOT ONE WORD.

    ``_shift_down``'s CYL_AXIS branch writes the ``r_to_minus_r`` image where
    METALLIC writes zero, so this is a real difference in the shifted buffer. The
    kernel's claim is that ``_mask_non_owned_cells`` zeroes the only row that can
    consume it. That is a claim about ``stepping.py`` and it is measured here rather
    than argued — a NONZERO count would mean this family must carry the near ghost,
    which is a legitimate outcome and not a test to be relaxed.

    The full sweep (six m values x both terminations x both sub-steps, 0 differing
    over 24 rows) lives in ``probe_metal_cylindrical_complex.py``; two rows are kept
    here so the merge bar carries it.
    """
    original = stepping._boundary_kinds
    for sub_step, runner in (("step_B", stepping.step_B),
                             ("step_D", stepping.step_D)):
        fields_a, pml_a = build(m, z_kind, seed=23)
        fields_b, pml_b = build(m, z_kind, seed=23)
        before = {n: np.array(getattr(fields_a, n), copy=True) for n in VOLUMES}
        runner(fields_a, pml_a)

        def forced(grid, pml, _original=original):
            return (stepping.METALLIC,) + tuple(_original(grid, pml)[1:])

        stepping._boundary_kinds = forced
        try:
            runner(fields_b, pml_b)
        finally:
            stepping._boundary_kinds = original

        differing = sum(_differing(getattr(fields_a, n), getattr(fields_b, n))
                        for n in VOLUMES)
        moved = sum(_differing(before[n], getattr(fields_a, n)) for n in VOLUMES)
        assert moved > 0, f"{sub_step} moved nothing: the leg is vacuous"
        assert differing == 0, (
            f"{sub_step} m={m} z={z_kind}: the r = 0 near ghost IS observable "
            f"({differing} words). The kernel compiles BCX = METALLIC and must not")


# ---------------------------------------------------------------------------
# The host-built constants, against the array path's own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", (1, -1, 2, -2, 3, 5))
def test_the_imr_row_reproduces_steppings_own_factor(m):
    """``imr_coefficient_row`` x partner == ``-_cylindrical_imr_term(...)``, in WORDS.

    The transcription is pinned against the array path rather than against a second
    transcription of the same formula. ``_cylindrical_imr_term`` returns the term in
    CURL SIGN CONVENTION (negated once, :724), so the row's product is its negation.
    """
    fields, _pml = build(m, seed=31)
    rows = int(fields.grid.shape[0])
    dtdx = fields.grid.dt / fields.grid.dx
    for sub_step, terms in cyl.IMR_TERMS.items():
        spec = cyl.SUB_STEPS[sub_step]
        partner_names = {0: spec["sources"][2], 2: spec["sources"][0]}
        for index, _register, sign in terms:
            target = spec["targets"][index]
            partner = getattr(fields, partner_names[index])
            reference = stepping._cylindrical_imr_term(fields, target, partner, sign)
            row = cyl.imr_coefficient_row(np, target, sign, m, dtdx, rows,
                                          np.complex64)
            ours = -(row * partner)
            assert _differing(ours, reference) == 0, (sub_step, target, sign)
            # The laundering claim the module docstring rests on: `(-1j) * X` puts
            # +0.0 in EVERY real word, at both signs of m and of `sign`.
            assert np.all(_words(np.ascontiguousarray(row).view(np.complex64).real)
                          == np.uint32(0)), (target, sign, m)


@pytest.mark.parametrize("m", (1, -1))
def test_the_axis_increment_scalar_carries_a_SIGNED_ZERO_real_word(m):
    """``1j * (m*dtdx)`` gives ``-0.0`` at m < 0 — the opposite of the i*m/r row.

    The difference is the SIGN OF THE UNIT, not the operand order, and it is why both
    scalars are host-rounded and passed through rather than synthesised in-kernel.
    """
    dtdx = 0.37
    minus, (real, imag) = cyl.axis_increment_scalars(m, dtdx)
    assert minus == float(np.float32(-dtdx))
    word = _words(np.float32([real]))[0]
    assert word == (np.uint32(0x80000000) if m < 0 else np.uint32(0)), hex(int(word))
    assert imag == float(np.float32(m * dtdx))


def test_m_class_has_three_arms_and_splits_at_zero_and_two():
    assert cyl.m_class(0) == cyl.M_ZERO
    assert cyl.m_class(1) == cyl.M_ONE
    assert cyl.m_class(-1) == cyl.M_ONE
    for m in (2, -2, 3, 5, -7):
        assert cyl.m_class(m) == cyl.M_MANY
    assert cyl.M_ARMS == (cyl.M_ZERO, cyl.M_ONE, cyl.M_MANY)
    assert len({cyl.M_ZERO, cyl.M_ONE, cyl.M_MANY}) == 3
    with pytest.raises(ValueError, match="m_arm must be"):
        cyl.cylindrical_curl_source(templates.METALLIC, False, 7, EXPANSION or "FMA_V1")


@pytest.mark.parametrize("sub_step,extra", (("step_B", 1), ("step_D", 0)))
def test_the_prefix_shape_matches_the_array_paths_own(sub_step, extra):
    """B extends Ep by ONE ZERO WALL ROW; D does not. Both are ``stepping``'s."""
    fields, _pml = build(1)
    shape = tuple(fields.grid.shape)
    assert cyl.prefix_shape(sub_step, shape) == (shape[0] + extra,) + shape[1:]
    sources = {name: getattr(fields, name)
               for name in cyl.SUB_STEPS[sub_step]["sources"]}
    prefix = cyl.cylindrical_prefix(np, sub_step, sources)
    assert prefix.shape == cyl.prefix_shape(sub_step, shape)
    assert prefix.dtype == np.complex64


# ---------------------------------------------------------------------------
# The predicate: the inversions, by name
# ---------------------------------------------------------------------------

@requires_probe
def test_a_real_dcyl_run_is_admitted_on_every_slot():
    fields, pml = build(1)
    residency = device.Residency()
    for sub_step in ("step_B", "step_D"):
        verdict = cyl.cylindrical_complex_pml_curl_coverage(
            fields, pml, sub_step, residency, PROBE)
        assert verdict.covered, verdict.reasons
    for side in ("H", "E"):
        verdict = cyl.cylindrical_complex_constitutive_coverage(
            fields, pml, side, residency, PROBE)
        assert verdict.covered, verdict.reasons


@requires_probe
def test_a_complex_m_zero_run_is_ADMITTED_here_and_refused_by_the_real_family():
    """The partition against the cylindrical REAL product is STORAGE, both ways.

    Until 2026-09-04 clause 14 refused m = 0 here by name, on the reading that
    ``stepping.py:731-736`` (complex storage mandatory at |m| >= 1) made clause 2 a
    mere engine coupling. That reading was wrong about the partition: the real
    product refuses complex64 storage, so a complex64 m = 0 run — constructible,
    and the corpus row ``dipole_in_vacuum_cyl_off_axis.py`` — was refused by BOTH
    families and stepped by the array path. The ``M_ZERO`` arm carries it here, and
    the real family's clause 2 still refuses it by name, so no slot is co-admitted.
    """
    from meep_gpu.metal_kernels import cylindrical_real

    fields, pml = build(m=0, complex_storage=True)
    residency = device.Residency()
    for sub_step in ("step_B", "step_D"):
        verdict = cyl.cylindrical_complex_pml_curl_coverage(
            fields, pml, sub_step, residency, PROBE)
        assert verdict.covered, verdict.reasons
        real = cylindrical_real.cylindrical_real_curl_coverage(
            fields, pml, sub_step, residency)
        assert not real.covered
        assert any("force_complex_fields=True" in reason for reason in real.reasons)
    for side in ("H", "E"):
        verdict = cyl.cylindrical_complex_constitutive_coverage(
            fields, pml, side, residency, PROBE)
        assert verdict.covered, verdict.reasons
    plan = cyl.plan_cylindrical_complex_pml_curl(fields, pml, "step_D", residency,
                                                 probe=PROBE)
    assert plan is not None
    assert plan.m_arm == cyl.M_ZERO and plan.zero_rows == 0
    assert plan.axis_coef == cyl.axis_coefficient(fields.grid.dt / fields.grid.dx)
    # THE OTHER DIRECTION, by name: float32 storage at m = 0 is the real family's.
    real_fields, real_pml = build(m=0, complex_storage=False)
    verdict = cyl.cylindrical_complex_pml_curl_coverage(
        real_fields, real_pml, "step_B", device.Residency(), PROBE)
    assert not verdict.covered
    assert any("force_complex_fields is not set" in reason
               and "REAL product" in reason for reason in verdict.reasons)


@requires_probe
@pytest.mark.parametrize("mutate,fragment", (
    (lambda f, p: (matrix.cart()[0], matrix.cart()[1]),
     "grid is not cylindrical"),
    (lambda f, p: (build(1, complex_storage=False)[0], p),
     "force_complex_fields is not set"),
    (lambda f, p: (f, None), "no active PML layer"),
), ids=("cartesian", "real-storage", "no-absorber"))
def test_each_inverted_clause_refuses_by_name(mutate, fragment):
    fields, pml = build(1)
    fields, pml = mutate(fields, pml)
    verdict = cyl.cylindrical_complex_pml_curl_coverage(
        fields, pml, "step_B", device.Residency(), PROBE)
    assert not verdict.covered
    assert any(fragment in reason for reason in verdict.reasons), verdict.reasons


@requires_probe
def test_a_single_radial_row_is_refused_because_the_axis_increment_reads_row_one():
    """nr < 2 is a SILENT out-of-bounds read on the GPU, not a loud one.

    Measured by the Triton tranche on a real ``(1, 1, 20)`` grid: ``numpy.take``
    RAISES IndexError there and ``cupy.take`` silently returns row 0, so the two
    backends do not agree that the configuration is steppable. The kernel's matching
    load is live on every lane and would read a plane past the end.
    """
    # The r PML is dropped rather than thinned: a 1-cell radial axis cannot hold one
    # at all, which is the platform saying the same thing this clause says.
    fields, pml = build(1, nr=1, r_pml=(0, 0))
    verdict = cyl.cylindrical_complex_pml_curl_coverage(
        fields, pml, "step_B", device.Residency(), PROBE)
    assert not verdict.covered
    assert any("radial extent" in reason for reason in verdict.reasons), \
        verdict.reasons


def test_a_missing_probe_is_a_named_refusal_and_never_a_default_arm():
    fields, pml = build(1)
    residency = device.Residency()
    assert cyl.expansion_from_probe({}) is None
    verdict = cyl.cylindrical_complex_pml_curl_coverage(
        fields, pml, "step_B", residency, probe={})
    assert not verdict.covered
    assert any("expansion probe" in reason for reason in verdict.reasons)
    assert cyl.plan_cylindrical_complex_pml_curl(
        fields, pml, "step_B", residency, probe={}) is None


def test_a_probe_missing_only_this_familys_two_patterns_licenses_nothing():
    """An artifact cut for the unfolded complex family may not licence this kernel.

    The base five are the same call sites; the two this family adds are calls
    ``complex_fields``' artifact never made. Reading its record would licence an arm
    from evidence about five of seven multiplies.
    """
    inherited = {"backend": cyl.PROBE_BACKEND,
                 "patterns": {name: complex_fields.AMBIGUOUS_BOTH
                              for name in complex_fields.PROBE_PATTERNS}}
    inherited["patterns"]["c8_mul_c8"] = "FMA_V1"
    assert complex_fields.expansion_from_probe(inherited) == "FMA_V1"
    assert cyl.expansion_from_probe(inherited) is None


def test_the_residency_must_be_declared():
    fields, pml = build(1)
    verdict = cyl.cylindrical_complex_pml_curl_coverage(
        fields, pml, "step_B", None, PROBE)
    assert not verdict.covered
    assert any("residency was not declared" in reason for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_the_family_is_in_FAMILY_MODULES_and_the_list_matches_the_PACKAGE():
    """A family in the tree but NOT in ``FAMILY_MODULES`` is INVISIBLE to ``plan_step``.

    That is a SILENT coverage loss rather than an error — the slot simply falls to
    the array path with nobody named — so the list is asserted against the PACKAGE
    CONTENTS rather than eyeballed: every module in ``metal_kernels`` that defines
    ``register_arms`` must be named, and a new family that forgets is a failing test
    on the day it lands.
    """
    assert "cylindrical_complex" in registry.FAMILY_MODULES
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal_kernels")
    registering = set()
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py") or name.startswith("test_"):
            continue
        with open(os.path.join(here, name), "r", encoding="utf-8") as handle:
            if "def register_arms(" in handle.read():
                registering.add(name[:-3])
    missing = registering - set(registry.FAMILY_MODULES)
    assert not missing, (
        f"{sorted(missing)} register arms but are absent from FAMILY_MODULES, so "
        f"plan_step never imports them and their slots fall silently to the array "
        f"path")


def test_the_family_registers_on_exactly_four_slots_and_touches_no_others():
    """Four slots; the independent ADE family owns ``update_P``."""
    slots = {spec.slot for spec in arms.registered()
             if spec.family == cyl.FAMILY}
    assert slots == {"step_B", "step_D", "update_H", "update_E"}
    assert "update_P" not in {spec.slot for spec in arms.registered()
                              if spec.family == cyl.FAMILY}
    labels = {spec.label for spec in arms.registered() if spec.family == cyl.FAMILY}
    assert labels == {"cylindrical complex"}
    assert all(spec.wired for spec in arms.registered() if spec.family == cyl.FAMILY)


@requires_probe
def test_plan_step_composes_all_four_slots_and_nothing_else_co_admits():
    fields, pml = build(1)
    residency = device.Residency()
    plan = launch.plan_step(fields, pml, residency, sources=(),
                            cylindrical_complex_probe=PROBE)
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert plan.selected.get(slot) == "cylindrical complex", (
            slot, plan.reasons.get(slot))
    assert "refuses rather than picking by table order" not in "".join(
        reason for reasons in plan.reasons.values() for reason in reasons)


@requires_probe
def test_the_constitutive_plan_IS_the_certified_complex_one():
    """No new arithmetic on ``update_H``/``update_E`` — only the admission is new.

    Checked as a TYPE and a SOURCE, not as a comment: the plan object is
    ``complex_fields``' own and the compiled entry point is that family's
    ``bloch_constitutive_step``. A divergence would then have to come from the
    predicate, which is the only thing this family contributes on these two slots.
    """
    fields, pml = build(1)
    residency = device.Residency()
    for side in ("H", "E"):
        plan = cyl.plan_cylindrical_complex_constitutive(fields, pml, side,
                                                         residency, probe=PROBE)
        assert isinstance(plan, complex_fields.ComplexConstitutivePlan), side
        assert plan.side == side


# ---------------------------------------------------------------------------
# The device legs: the four additions, against stepping.py
# ---------------------------------------------------------------------------

def _run_pair(m, z_kind, sub_step, accurate=False, steps=1, courant=0.37,
              functions=None, seed=11, route="engine"):
    """Array path and Metal plan on identical state; returns (differing, moved, plan).

    ``route`` selects which BUILDER assembles the plan — the engine's (a real
    ``Fields``/``PML`` through the predicate) or the gate's (bare host arrays, no
    predicate). Both must produce the same bytes, which is what makes the mutation
    legs' route trustworthy. Passing ``functions`` forces the bare-array route and is
    the MUTATION SEAM.
    """
    fields_a, pml_a = build(m, z_kind, accurate, seed=seed, courant=courant)
    fields_b, pml_b = build(m, z_kind, accurate, seed=seed, courant=courant)
    spec = cyl.SUB_STEPS[sub_step]
    touched = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
    before = {n: np.array(getattr(fields_a, n), copy=True) for n in touched}

    runner = stepping.step_B if sub_step == "step_B" else stepping.step_D
    for _ in range(steps):
        runner(fields_a, pml_a)

    residency = device.Residency()
    if functions is None and route == "engine":
        plan = cyl.plan_cylindrical_complex_pml_curl(fields_b, pml_b, sub_step,
                                                     residency, probe=PROBE)
        assert plan is not None, cyl.cylindrical_complex_pml_curl_coverage(
            fields_b, pml_b, sub_step, residency, PROBE).reasons
    else:
        grid = fields_b.grid
        kinds = stepping._boundary_kinds(grid, pml_b)
        bcz = (templates.METALLIC if kinds[2] == "metallic"
               else templates.PERIODIC)
        names = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
                 + tuple(spec["sources"]))
        plan = cyl.plan_cylindrical_complex_pml_curl_from_arrays(
            sub_step, {n: getattr(fields_b, n) for n in names},
            {f"{stem}_{axis}": getattr(pml_b, f"{stem}_{axis}{spec['suffix']}")
             for axis in "xyz" for stem in ("kms", "sinv")},
            bcz, int(grid.m), bool(grid.accurate_fields_near_cylorigin),
            grid.dt / grid.dx, EXPANSION, residency, functions=functions)
    for _ in range(steps):
        plan.run()
    residency.sync_out()

    differing = sum(_differing(getattr(fields_a, n), getattr(fields_b, n))
                    for n in touched)
    moved = sum(_differing(before[n], getattr(fields_a, n)) for n in touched)

    # THE CHECKED SUBNORMAL PRECONDITION, over the WINDOW this comparison covers.
    # MPS flushes float32 subnormals natively and exposes no lever, so `keep` is not
    # offerable and every byte claim on this backend rides on the band being empty.
    # A census at step 0 of a 6-step run certifies step 0, which is why the window
    # carries its first and last step rather than a scalar count.
    window = preconditions.SubnormalWindow(0, steps, per_array_words=64)
    for name in touched + tuple(spec["sources"]):
        window.observe(name, getattr(fields_a, name), step=steps)
    preconditions.assert_clean_or_refuse(
        window, f"cylindrical m={m} z={z_kind} {sub_step} x{steps}")
    return differing, moved, plan


@requires_mps
@requires_probe
@pytest.mark.parametrize("m", (0, 1, -1, 2, -2, 3, 5))
@pytest.mark.parametrize("z_kind", ("metallic", "periodic"))
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_curl_reproduces_stepping_word_for_word(m, z_kind, sub_step):
    differing, moved, plan = _run_pair(m, z_kind, sub_step)
    assert moved > 0, "the array path moved nothing: this comparison is vacuous"
    assert differing == 0, (
        f"m={m} z={z_kind} {sub_step}: {differing} differing uint32 words "
        f"({moved} moved by the reference)")
    assert plan.launches == 1
    assert plan.prefix_syncs == 1


@requires_mps
@requires_probe
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_accurate_branch_binds_a_DIFFERENT_zero_row_count_at_the_same_m(sub_step):
    """``accurate_fields_near_cylorigin`` selects row 0 alone where the default takes
    ``|m|`` rows — the same specialisation, a different bound uniform. If the count
    were a compile-time constant this would be a different kernel; it is not.
    """
    differing, moved, plan = _run_pair(3, "metallic", sub_step, accurate=True,
                                       courant=0.25)
    assert moved > 0
    assert differing == 0, differing
    assert plan.zero_rows == 1
    assert cyl.zero_rows(3, False) == 3


@requires_mps
@requires_probe
@pytest.mark.parametrize("m,sub_step", ((1, "step_B"), (1, "step_D"),
                                        (3, "step_B"), (3, "step_D")))
def test_six_launches_stay_identical_the_accumulating_auxiliary_class(m, sub_step):
    """``fu_*`` is STATE. A kernel right for one launch and wrong forever after is
    indistinguishable from a correct one in a single-launch comparison, and the
    |m| >= 2 rule zeroes the auxiliaries AFTER the recurrence, which is exactly the
    kind of write a single launch cannot separate from a no-op.
    """
    differing, moved, plan = _run_pair(m, "metallic", sub_step, steps=6)
    assert moved > 0
    assert differing == 0, differing
    assert plan.launches == 6
    assert plan.prefix_syncs == 6


@requires_mps
@requires_probe
def test_the_prefix_MUST_be_refreshed_and_the_counter_is_armed():
    """The residency cost this family pays, shown to be load-bearing.

    The prefix is a HOST scan of a volume the device owns between launches, so every
    launch syncs one source out and the prefix in. This asserts the counter AND arms
    it: with the source changed under the plan, a launch that reuses the previous
    prefix produces DIFFERENT bytes from one that refreshes. Without the arming,
    ``prefix_syncs == launches`` would be bookkeeping rather than evidence.
    """
    fields, pml = build(1, seed=17)
    residency = device.Residency()
    plan = cyl.plan_cylindrical_complex_pml_curl(fields, pml, "step_B", residency,
                                                 probe=PROBE)
    assert plan is not None
    plan.run()
    residency.sync_out()
    assert plan.prefix_syncs == 1

    # Change the prefix SOURCE (Ey) on both host and device, then launch twice from
    # the same state: once with the stale prefix still bound, once refreshed.
    rng = np.random.default_rng(99)
    fields.Ey.real[...] = rng.standard_normal(fields.Ey.shape).astype(np.float32)
    fields.Ey.imag[...] = rng.standard_normal(fields.Ey.shape).astype(np.float32)
    residency.sync_in(("Ey",))
    snapshot = {n: np.array(getattr(fields, n), copy=True)
                for n in ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")}

    stale_args = plan._args  # noqa: SLF001 - the arming needs the bound tuple
    plan._functions[shaders.CONTRACT_OFF](*stale_args)   # NO refresh
    residency.sync_out()
    stale = {n: np.array(getattr(fields, n), copy=True) for n in snapshot}

    for name, values in snapshot.items():
        getattr(fields, name)[...] = values
    residency.sync_in(tuple(snapshot))
    plan.run()                                            # WITH refresh
    residency.sync_out()

    differing = sum(_differing(stale[n], getattr(fields, n)) for n in snapshot)
    assert differing > 0, (
        "a launch on the STALE prefix produced identical bytes to a refreshed one; "
        "the refresh is then unmeasured and prefix_syncs proves nothing")
    assert plan.prefix_syncs == 2


# ---------------------------------------------------------------------------
# The mutations: does the comparison have teeth?
# ---------------------------------------------------------------------------

#: ``(id, sub_step, m, needle, replacement)``. Each is a defect a reader might plant
#: by accident, and each must be CAUGHT by the byte comparison above. A mutation that
#: is NOT caught is a leg that certifies nothing, which is why they are here rather
#: than in a comment about coverage.
MUTATIONS = (
    ("imr_sign_flipped", "step_B", 1,
     "curl0 = curl0 - m0;", "curl0 = curl0 + m0;"),
    ("zero_rows_off_by_one", "step_D", 3,
     "bool near = (i < int(zrows));", "bool near = (i <= int(zrows));"),
    ("axis_increment_not_negated", "step_B", 1,
     "curl0 = at_x ? -inc : curl0;", "curl0 = at_x ? inc : curl0;"),
    ("prefix_row_stride_wrong", "step_B", 1,
     "float2 pu = pfx[ii + nyz];", "float2 pu = pfx[ii + 1];"),
    ("near_axis_auxiliary_not_zeroed", "step_B", 3,
     "        n0 = near ? float2(0.0f, 0.0f) : n0;", "        // n0 kept"),
    # THE m = 0 ARM'S OWN RULES, each a transcription slip a reader could make.
    ("m_zero_B_axis_clear_dropped", "step_B", 0,
     "    v0 = at_x ? float2(0.0f, 0.0f) : v0;", "    // MUTANT: Bx[r=0] kept"),
    ("m_zero_D_post_add_dropped", "step_D", 0,
     "        v2 = at_x ? (v2 + inc0) : v2;", "        // MUTANT: no post-add"),
    ("m_zero_D_Dy_axis_clear_dropped", "step_D", 0,
     "        v1 = at_x ? float2(0.0f, 0.0f) : v1;", "        // MUTANT: Dy[r=0] kept"),
    ("m_zero_D_post_add_also_writes_the_auxiliary", "step_D", 0,
     "        v2 = at_x ? (v2 + inc0) : v2;",
     "        v2 = at_x ? (v2 + inc0) : v2;\n        n2 = at_x ? (n2 + inc0) : n2;"),
)


@requires_mps
@requires_probe
def test_the_m_zero_post_add_is_not_a_curl_fold():
    """The one m = 0 defect the needle table above cannot spell in place.

    ``stepping.py:575-580`` records the fold as MEASURED WRONG under PML (Er 2.7e-01
    / Hp 4.5e-01 against the post-add's 3.6e-07): Dz's split-field ladder is R, and
    routing the increment through the recurrence instead of adding it to the stored
    value gives a different ``fu_Dz`` and a different ``Dz``. Planted here as the
    fold, and it must be CAUGHT.
    """
    from meep_gpu.metal_kernels.device import compile_source

    source = cyl.cylindrical_curl_source(templates.METALLIC, True, cyl.M_ZERO,
                                         EXPANSION)
    mutant = source.replace(
        "        v2 = at_x ? (v2 + inc0) : v2;\n", "")
    mutant = mutant.replace(
        "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);\n",
        "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);\n"
        "    curl2 = at_x ? -(c_mul_coefficient_left(axis_coef, b)) : curl2;\n", 1)
    assert mutant != source and mutant.count("axis_coef, b") == 2
    functions = {shaders.CONTRACT_OFF:
                 compile_source(mutant).cyl_complex_pml_curl_step}
    differing, moved, plan = _run_pair(0, "metallic", "step_D", functions=functions)
    assert moved > 0 and plan.launches == 1
    assert differing > 0, "the curl fold reproduced the post-add; the rule is unmeasured"


@requires_mps
@requires_probe
@pytest.mark.parametrize("label,sub_step,m,needle,replacement", MUTATIONS,
                         ids=[row[0] for row in MUTATIONS])
def test_every_planted_mutation_is_caught(label, sub_step, m, needle, replacement):
    from meep_gpu.metal_kernels.device import compile_source

    backward = bool(cyl.SUB_STEPS[sub_step]["backward"])
    arm = cyl.m_class(m)
    source = cyl.cylindrical_curl_source(templates.METALLIC, backward, arm, EXPANSION)
    assert source.count(needle) == 1, (
        f"the {label} needle is not uniquely present; a mutation that does not "
        f"apply reports the defect as UNCAUGHT")
    mutant = source.replace(needle, replacement)
    assert mutant != source
    functions = {shaders.CONTRACT_OFF:
                 compile_source(mutant).cyl_complex_pml_curl_step}
    differing, moved, plan = _run_pair(m, "metallic", sub_step,
                                       functions=functions)
    assert moved > 0
    assert plan.launches == 1, "the mutant never ran; the leg is disarmed"
    assert differing > 0, f"{label} was NOT caught: the byte comparison has no teeth"


@requires_mps
@requires_probe
def test_the_from_arrays_route_reproduces_the_engine_route():
    """The gate's route and the engine's route must build the SAME plan.

    They differ only in where the arrays come from and how the coefficient mirrors
    are named; everything that decides a BIT is assembled in one place. Checked
    because the mutation legs above run through the bare-array route, and a route
    that quietly bound a different m class or a different uniform would report every
    mutation as caught for the wrong reason.
    """
    differing, moved, plan = _run_pair(3, "periodic", "step_D", route="arrays")
    assert moved > 0
    assert differing == 0, differing
    assert plan.m_arm == cyl.M_MANY
    assert plan.zero_rows == 3
    assert plan.bcz == templates.PERIODIC


def test_the_subnormal_precondition_is_DEMONSTRATED_TO_FIRE():
    """A precondition never shown to fire is decorative.

    Every byte leg above asserts the band is empty. This is the other half: a volume
    scaled into the band must make the SAME census refuse. Without it, the legs would
    prove only that the physical band is clean, which was never in doubt.
    """
    fields, _pml = build(1, seed=41)
    clean = preconditions.SubnormalWindow(0, 1, per_array_words=64)
    clean.observe("Ey", fields.Ey, step=1)
    preconditions.assert_clean_or_refuse(clean, "control/clean")

    banded = preconditions.SubnormalWindow(0, 1, per_array_words=64)
    scaled = (fields.Ey * np.float32(1e-40)).astype(np.complex64)
    banded.observe("Ey_scaled", scaled, step=1)
    preconditions.demonstrate_firing(banded, "control/banded")

    empty = preconditions.SubnormalWindow(0, 1, per_array_words=64)
    with pytest.raises(AssertionError, match="VACUOUS"):
        preconditions.assert_clean_or_refuse(empty, "control/empty")


@requires_mps
@requires_probe
def test_a_mode_the_plan_was_not_built_with_RAISES():
    """The variant selector is a contract, not a convenience.

    A guard argument that quietly launched the pinned variant would make a
    contraction leg vacuous. Inherited from ``plans.KernelPlan`` and checked here
    because this family OVERRIDES ``run`` — the override must not have loosened it.
    """
    fields, pml = build(1)
    plan = cyl.plan_cylindrical_complex_pml_curl(fields, pml, "step_B",
                                                 device.Residency(), probe=PROBE)
    assert plan is not None
    with pytest.raises(KeyError, match="holds no 'fast' variant"):
        plan.run(shaders.CONTRACT_FAST)


# ---------------------------------------------------------------------------
# The axis-row rule is a REPLACEMENT, and that is a signed-zero fact
# ---------------------------------------------------------------------------

def _plant_axis_seam(fields, lane: int = 5) -> None:
    """Build the ONE state in which REPLACE and ACCUMULATE differ.

    The masked axis row is exactly ``+0.0`` and ``+0.0 + x == x`` for every x EXCEPT
    ``-0.0``, so the two spellings diverge only where the increment's real word is an
    exact ``+0.0`` AND the recurrence operand it meets is an exact ``-0.0``. Neither
    a random seed nor a signed-zero scatter produces that pair — measured 0 of 5 in
    the family gate before this construction existed — so it is built:

    * ``Ey[0,0,k] = -0.0 + 1j`` beside ``Ey[0,0,k+1] = +0.0 + 0.5j`` makes the z
      difference's real word ``-0.0``, which ``(-dtdx) * d`` turns into ``+0.0``;
    * ``Ez[1,0,k] = 0`` makes the ``(1j*m*dtdx) * Ez[r+1]`` real word ``+0.0``, so
      the increment's real word is ``+0.0 - +0.0 = +0.0``;
    * ``fu_Bx[0,0,k] = -0.0 + 1j`` makes the recurrence minuend's real word ``-0.0``,
      the only operand that can tell ``-(+0.0)`` from ``+0.0 - (+0.0)``.
    """
    fields.Ey[0, 0, lane] = complex(-0.0, 1.0)
    fields.Ey[0, 0, lane + 1] = complex(0.0, 0.5)
    fields.Ez[1, 0, lane] = complex(0.0, 0.0)
    fields.fu_Bx[0, 0, lane] = complex(-0.0, 1.0)
    fields.Bx[0, 0, lane] = complex(0.25, -0.5)


@requires_mps
@requires_probe
def test_the_axis_increment_REPLACES_and_never_accumulates():
    """The shipped kernel is exact on the constructed state; an accumulator is not.

    Two halves and both are required. Without the first, the mutation could be
    passing because the state is pathological rather than because the spelling is
    right; without the second, ``curl0 = at_x ? -inc : curl0`` would be a comment
    about signed zeros rather than a measured choice.
    """
    from meep_gpu.metal_kernels.device import compile_source

    fields_a, pml_a = build(1, seed=11)
    fields_b, pml_b = build(1, seed=11)
    _plant_axis_seam(fields_a)
    _plant_axis_seam(fields_b)
    touched = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")
    before = {n: np.array(getattr(fields_a, n), copy=True) for n in touched}
    stepping.step_B(fields_a, pml_a)
    moved = sum(_differing(before[n], getattr(fields_a, n)) for n in touched)
    assert moved > 0

    residency = device.Residency()
    plan = cyl.plan_cylindrical_complex_pml_curl(fields_b, pml_b, "step_B",
                                                 residency, probe=PROBE)
    assert plan is not None
    plan.run()
    residency.sync_out()
    assert sum(_differing(getattr(fields_a, n), getattr(fields_b, n))
               for n in touched) == 0

    shipped = cyl.cylindrical_curl_source(templates.METALLIC, False, cyl.M_ONE,
                                          EXPANSION)
    mutant = shipped.replace("curl0 = at_x ? -inc : curl0;",
                             "curl0 = at_x ? (curl0 - inc) : curl0;")
    assert mutant != shipped
    fields_c, pml_c = build(1, seed=11)
    _plant_axis_seam(fields_c)
    spec = cyl.SUB_STEPS["step_B"]
    names = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
             + tuple(spec["sources"]))
    mutant_residency = device.Residency()
    mutant_plan = cyl.plan_cylindrical_complex_pml_curl_from_arrays(
        "step_B", {n: getattr(fields_c, n) for n in names},
        {f"{stem}_{axis}": getattr(pml_c, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")},
        templates.METALLIC, 1, False, fields_c.grid.dt / fields_c.grid.dx,
        EXPANSION, mutant_residency,
        functions={shaders.CONTRACT_OFF:
                   compile_source(mutant).cyl_complex_pml_curl_step})
    mutant_plan.run()
    mutant_residency.sync_out()
    assert sum(_differing(getattr(fields_a, n), getattr(fields_c, n))
               for n in touched) > 0, (
        "an ACCUMULATING axis-row increment reproduced the array path on the state "
        "built to separate the two; the REPLACE spelling is then unmeasured")


# ---------------------------------------------------------------------------
# The COMPLETE step: the classes a single-launch comparison cannot see
# ---------------------------------------------------------------------------

#: One complete driver step on a WALLED Dcyl run — six passes, with the wall clear
#: sitting between the curl and the constitutive pass (driver.py:3281-3287,
#: :3292-3302; the two fold passes are inert on a Dcyl grid). No Metal product
#: carries ``zero_metal_*``, so it runs on the HOST and must be bracketed.
WALLED_PASSES = ("step_B", "zero_metal_B", "update_H",
                 "step_D", "zero_metal_D", "update_E")

_ARRAY_PASS = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
}


def _complete_step_pair(steps: int = 4, bracket: bool = True):
    """Array path and composed Metal plan, walked per COMPLETE STEP.

    ``bracket=False`` runs the host wall pass WITHOUT its ``sync_out``/``sync_in``,
    which is the arming: the engine holds NumPy and the device mirror is
    authoritative between launches, so an unbracketed host write goes nowhere.
    """
    fields, pml = build(1, z_kind="metallic", seed=11)
    reference, reference_pml = build(1, z_kind="metallic", seed=11)
    assert launch.live_sub_steps(fields, pml, ()) is not None
    assert tuple(n for n in launch.live_sub_steps(fields, pml, ())
                 if n in _ARRAY_PASS) == WALLED_PASSES

    residency = device.Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=(),
                            synced=("zero_metal_B", "zero_metal_D"),
                            cylindrical_complex_probe=PROBE)
    assert set(plan.plans) == {"step_B", "update_H", "step_D", "update_E"}, \
        sorted(plan.plans)
    residency.sync_in()
    for _ in range(steps):
        for slot in WALLED_PASSES:
            _ARRAY_PASS[slot](reference, reference_pml)
            if slot in plan.plans:
                plan.plans[slot].run()
            elif bracket:
                residency.sync_out()
                _ARRAY_PASS[slot](fields, pml)
                residency.sync_in()
            else:
                _ARRAY_PASS[slot](fields, pml)
    residency.sync_out()
    differing = sum(_differing(getattr(fields, n), getattr(reference, n))
                    for n in VOLUMES)
    evolved = sum(_differing(np.zeros_like(getattr(reference, n)),
                             getattr(reference, n)) for n in VOLUMES)
    return differing, evolved, plan


@requires_mps
@requires_probe
def test_a_complete_walled_step_reproduces_the_array_path():
    """Six green sub-step comparisons say nothing about the object the engine runs.

    A stale mirror, the ``zero_metal_*`` seam and the accumulating ``fu_*`` state are
    all invisible to a single-launch comparison by construction. This walks four
    complete steps in the driver's own pass order and compares EVERY stored volume.
    """
    differing, evolved, plan = _complete_step_pair()
    assert evolved > 0, "the reference never evolved: this comparison is vacuous"
    assert differing == 0, differing
    for slot in ("step_B", "update_H", "step_D", "update_E"):
        assert plan.plans[slot].launches == 4, (slot, plan.plans[slot].launches)
    for slot in ("step_B", "step_D"):
        assert plan.plans[slot].prefix_syncs == 4, (
            slot, "the prefix must be refreshed once per launch; a plan that "
            "refreshed it once is byte-perfect on step 0 and wrong after")


@requires_mps
@requires_probe
def test_the_host_wall_pass_MUST_be_bracketed_by_the_residency_syncs():
    """The arming for the test above, and the whole stale-mirror class in one line.

    ``zero_metal_B`` clears stored cell 0 of the walled axis on the HOST. Without the
    sync bracket it writes an array the device never reads, so the wall keeps the
    field the reference just cleared. Without this, the bracket in the test above
    would be ceremony rather than a measured requirement.
    """
    differing, evolved, _plan = _complete_step_pair(bracket=False)
    assert evolved > 0
    assert differing > 0, (
        "dropping the sync bracket around the host wall pass changed NOTHING; the "
        "residency clause is then unmeasured and every array-path pass in a "
        "composed step is running on trust")
