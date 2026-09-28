"""Merge-bar tests for the Metal cylindrical m = 0 (real storage) family.

This family's substance is three substitutions on a certified body, and every one of
them fails SILENTLY — a wrong radial weight, a missing wall row or a dropped axis
rule all produce a smooth, plausible, entirely wrong field rather than an error. So
what is pinned here is chosen by which hazards a byte gate cannot reach:

**1. THE PAIRINGS THAT ARE HALF-CELL ERRORS.** ``ir0`` is 0.0 on the B side and 0.5
on the D side because Ep sits at the node and Hp half a cell out
(stepping.py:1297-1299); the wall row is the B side's only. Swapping either is a
half-cell error in the radial ladder, and it is invisible in every test that runs
one sub-step. They are pinned as TABLES against ``stepping``'s own call sites.

**2. THE EMITTER REUSE, AGAINST THE FUNCTION AND NOT AGAINST A READING OF IT.** The
whole design rests on compiling the r axis as METALLIC so that
``shaders.ownership_mask`` and ``shaders.ghost`` need no cylindrical arm. That is a
claim about which CELLS get zeroed, so it is checked against
``stepping._mask_non_owned_cells`` running on a real Dcyl grid.

**3. THE CLAUSE INVERSIONS**, because a missed inversion costs slots QUIETLY: two
admitters leave a slot unselected and it falls to the array path with no error at
all. Clause 6 inverted (cylindrical REQUIRED) and clause 14 new (m == 0 REQUIRED)
are asserted to actually refuse the configurations they name.

**4. THE STRUCTURAL CONTRACTS** that would otherwise be found at compile time or,
worse, at run time: the 31-binding ceiling, the two-launch plan's ORDER, and the
no-module-level-torch rule that keeps the predicate readable on a host with no GPU.

WHAT IS MEASURED ELSEWHERE, and named so this file is not mistaken for the whole
claim. ``parity/meep_gpu/probe_metal_cylindrical_real.py`` carries the five
measurements this family was built on — Metal's ``/`` against NumPy's divide, the
column-serial scan against ``stepping.cylindrical_rderiv_prefix``, the negation
spellings, the r-axis near ghost on the array path, and the emitter agreement. The
device-touching tests here are the smallest non-vacuous version, so a red merge bar
names the defect without a GPU run; they skip without MPS.
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.metal_kernels import cylindrical_real as cyl
from meep_gpu.metal_kernels import shaders
from meep_gpu.triton_kernels.launch import SUB_STEPS

MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "metal_kernels", "cylindrical_real.py")

#: One representative configuration: r is the axis (compiled METALLIC), phi wraps,
#: z is a wall — the corpus's own commonest cylindrical shape.
CODES = (shaders.METALLIC, shaders.PERIODIC, shaders.METALLIC)


def _has_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available())


requires_mps = pytest.mark.skipif(not _has_mps(), reason="no MPS device")


def _engine(shape=(20, 1, 40), courant=0.3141592653589793, m=0,
            z_kind="metallic", complex_storage=False):
    """A real Dcyl ``Grid``/``Fields``/``PML`` — the same shape the probe builds."""
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=int(m), boundaries={"z": z_kind},
                courant=float(courant), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    epsilon = np.full(tuple(grid.shape), np.float32(2.0))
    fields.set_isotropic_epsilon_volume(epsilon,
                                        (1.0 / epsilon).astype(np.float32))
    fields.enable_pml_storage()
    return grid, fields, PML(grid=grid,
                             thickness={"x": (0, max(2, shape[0] // 4)),
                                        "z": max(2, shape[2] // 4)})


class _Residency:
    """The smallest object ``_residency_declaration_reasons`` accepts."""

    def mirror(self, *args, **kwargs):  # pragma: no cover - never called here
        raise AssertionError("the predicate must not build mirrors")


# ---------------------------------------------------------------------------
# 1. The pairings that are half-cell errors
# ---------------------------------------------------------------------------

def test_ir0_is_zero_on_the_b_side_and_a_half_on_the_d_side():
    """``ir0`` is half the phi component's r-Yee shift, per sub-step.

    Not transcribed from the comment: rebuilt from ``fields.IYEE_SHIFTS`` for the
    component each sub-step's scan actually reads. Ep (= Ey) has r-shift 0 so ir0 is
    0.0; Hp (= Hy) has r-shift 1 so ir0 is 0.5 (stepping.py:1297-1299). Swapping
    them is a half-cell error in the radial weights and produces a smooth wrong
    field, which is exactly the class no single-sub-step test can see.
    """
    from meep_gpu.fields import IYEE_SHIFTS

    for sub_step, expected_source in (("step_B", "Ey"), ("step_D", "Hy")):
        scanned = SUB_STEPS[sub_step]["sources"][1]
        assert scanned == expected_source, (
            f"{sub_step}'s scan reads {scanned!r}; the ir0 table is keyed on the "
            f"phi component and this pairing decides the radial weights")
        assert cyl.PREFIX_IR0[sub_step] == 0.5 * IYEE_SHIFTS[scanned][0]


def test_only_the_b_side_carries_the_zero_wall_row():
    """The wall row exists because Bz's curl is a FORWARD difference of the prefix.

    ``step_D``'s Dz reads rows i and i-1 (a backward difference) and never looks
    past the top, so a wall row there would add a scan row nothing consumes. The
    table is asserted against the sub-step's own DIRECTION rather than spelled
    twice.
    """
    for sub_step, spec in SUB_STEPS.items():
        forward = not spec["backward"]
        assert cyl.PREFIX_WALL_ROW[sub_step] is forward
        assert cyl.scan_rows(sub_step, 20) == 20 + (1 if forward else 0)


def test_the_row_vectors_come_from_steppings_own_builder():
    """``prefix_row_vectors`` must not re-derive the ladder in float32.

    ``stepping._cylindrical_rderiv_weights`` builds both vectors in FLOAT64 and
    rounds to float32 exactly once (stepping.py:1339-1341). Recomputing the divisor
    as ``float32(i + ir0) - 0.5f`` would round twice and is a half-ulp error along
    the whole radial ladder — invisible in any tolerance and visible in every word.
    """
    for sub_step in SUB_STEPS:
        rows = cyl.scan_rows(sub_step, 17)
        weights, divisor = cyl.prefix_row_vectors(sub_step, 17)
        expected_w, expected_d = stepping._cylindrical_rderiv_weights(
            np, rows, cyl.PREFIX_IR0[sub_step], np.float32)
        assert weights.shape == (rows,) and divisor.shape == (rows - 1,)
        assert np.array_equal(weights.view(np.uint32),
                              expected_w.reshape(-1).view(np.uint32))
        assert np.array_equal(divisor.view(np.uint32),
                              expected_d.reshape(-1).view(np.uint32))

    # THE DOUBLE ROUNDING THIS GUARDS AGAINST, exhibited rather than described: a
    # divisor formed as float32(i + ir0) - 0.5f instead of float32((i + ir0) - 0.5)
    # rounds twice. At ir0 = 0.0 and 0.5 the ladder values happen to be exactly
    # representable and the two orders agree, so this loop searches for an ir0 where
    # they do NOT — which is what makes the pin above load-bearing rather than a
    # restatement of a coincidence.
    disagreeing = []
    for ir0 in (0.1, 0.3, 1.0 / 3.0, 0.7):
        counts = np.arange(64, dtype=np.float64) + ir0
        twice = (counts.astype(np.float32)[1:] - np.float32(0.5)).astype(np.float32)
        once = (counts[1:] - 0.5).astype(np.float32)
        if not np.array_equal(twice.view(np.uint32), once.view(np.uint32)):
            disagreeing.append(ir0)
    assert disagreeing, (
        "no ir0 in the probe set separates the two rounding orders, so this test "
        "cannot demonstrate the defect it names")


def test_the_axis_coefficient_is_host_rounded_from_the_array_paths_expression():
    """``4.0 * (dt/dx)`` in float64, rounded ONCE — stepping.py:586.

    The kernel binds this scalar rather than forming ``4.0f * dtdx`` itself. Scaling
    by four is exact so the two agree, and this test PINS that they agree rather
    than leaving the transcription resting on the coincidence.
    """
    for dtdx in (0.3141592653589793, 0.5, 0.37, 0.6, 1.0 / 3.0):
        host = np.float32(4.0 * dtdx)
        in_kernel = np.float32(4.0) * np.float32(dtdx)
        assert host.view(np.uint32) == in_kernel.view(np.uint32)


# ---------------------------------------------------------------------------
# 2. The emitter reuse, checked against the function
# ---------------------------------------------------------------------------

def test_the_r_axis_maps_to_metallic_and_the_emitter_refuses_anything_else():
    """'axis' compiles as METALLIC, and a PERIODIC r axis is refused BY NAME.

    A periodic r axis would wrap the far face onto the axis row — a silent wrong
    answer — so the source emitter refuses rather than emitting it.
    """
    assert cyl.boundary_codes(("axis", "periodic", "metallic")) == CODES
    assert cyl.boundary_codes(("axis", "periodic", "periodic")) == (
        shaders.METALLIC, shaders.PERIODIC, shaders.PERIODIC)
    with pytest.raises(ValueError, match="r axis must compile as METALLIC"):
        cyl.cylindrical_curl_source("step_B", (shaders.PERIODIC,
                                               shaders.PERIODIC,
                                               shaders.METALLIC))


def test_the_certified_ownership_mask_reproduces_the_is_axis_clause():
    """``shaders.ownership_mask`` with r METALLIC masks what the ARRAY PATH masks.

    The claim the family rests on, checked against
    ``stepping._mask_non_owned_cells`` itself on a real Dcyl grid rather than
    against a reading of it. The array path masks r row 0 through its ``is_axis``
    clause (stepping.py:1945-1949); the emitter masks it through its METALLIC
    clause; the two must land on the same cells.
    """
    shape = (20, 1, 40)
    grid, fields, pml = _engine(shape)
    kinds = stepping._boundary_kinds(grid, pml)
    assert kinds[0] == "axis", kinds
    codes = cyl.boundary_codes(kinds)

    for terms, backward in ((stepping.B_CURL_TERMS, False),
                            (stepping.D_CURL_TERMS, True)):
        emitted = cyl.templates.ownership_mask(codes, backward)
        flags = {"at_x": (0, slice(None), slice(None)),
                 "at_y": (slice(None), 0, slice(None)),
                 "at_z": (slice(None), slice(None), 0)}
        order = [term.target for term in terms]
        kernel = {term.target: np.zeros(shape, dtype=bool) for term in terms}
        for line in emitted.splitlines():
            line = line.strip()
            if not line.startswith("curl"):
                continue
            flag = line.split("=", 1)[1].split("?", 1)[0].strip()
            kernel[order[int(line[4])]][flags[flag]] = True

        masked_total = 0
        for term in terms:
            probe = np.ones(shape, dtype=np.float32)
            stepping._mask_non_owned_cells(probe, grid, term.iyee)
            array_path = probe == 0.0
            masked_total += int(array_path.sum())
            assert np.array_equal(array_path, kernel[term.target]), (
                f"{term.target}: the emitter and _mask_non_owned_cells disagree "
                f"about which cells are unowned")
        # A census of zero would mean the case never constructed the class.
        assert masked_total > 0


# ---------------------------------------------------------------------------
# 3. The clause inversions
# ---------------------------------------------------------------------------

def _reasons(fields, pml, sub_step="step_B"):
    return cyl.cylindrical_real_curl_coverage(fields, pml, sub_step,
                                              _Residency()).reasons


def test_clause_6_is_inverted_a_cartesian_grid_is_refused_by_name():
    """The ONE inversion that separates this family from every Cartesian one."""
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 0.0), dimensions=2,
                courant=0.35, xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    epsilon = np.full(tuple(grid.shape), np.float32(2.0))
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(np.float32))
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
    reasons = _reasons(fields, pml)
    assert any("grid.cylindrical is False" in r for r in reasons), reasons
    assert any("is not the cylindrical r = 0 axis" in r for r in reasons), reasons


def test_clause_14_refuses_nonzero_m_by_name_not_by_omission():
    """m != 0 is refused EXPLICITLY, naming what it needs and who owns it.

    The separation from the |m| >= 1 family must not rest on the engine coupling
    that makes complex storage mandatory there: a coupling is not a clause, and a
    residual separated only by one is planted and measured, never assumed.
    """
    grid, fields, pml = _engine(m=2, complex_storage=True)
    reasons = _reasons(fields, pml)
    named = [r for r in reasons if "grid.m = 2" in r]
    assert named, reasons
    assert "i*m/r" in named[0]
    # And clause 2 refuses the SAME configuration independently, which is the
    # deliberate belt-and-braces on the only pair whose inversion is internal.
    assert any("complex-storage cylindrical" in r for r in reasons), reasons


def test_clause_2_refuses_complex_storage_at_m_zero_independently_of_clause_14():
    """A complex-storage m = 0 run is constructible, and is refused on storage.

    ``force_complex_fields`` is an independent switch, so clause 14 alone would let
    this configuration through to a kernel that reads the wrong stride.
    """
    grid, fields, pml = _engine(m=0, complex_storage=True)
    reasons = _reasons(fields, pml)
    assert any("complex-storage cylindrical" in r for r in reasons), reasons
    assert not any("grid.m" in r for r in reasons), reasons


def test_clause_4_admits_the_axis_kind_on_the_r_axis_only():
    """The widening is PER AXIS: an axis-kind z would mean the grid disagrees.

    Spelled per axis rather than by adding 'axis' to the covered set, which would
    silently compile the wrong ghost onto two axes.
    """
    grid, fields, pml = _engine()
    assert not _reasons(fields, pml), _reasons(fields, pml)

    class _Swapped:
        """A grid whose PHI axis claims to be the cylindrical axis."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def is_axis(self, axis):
            return axis == 1

    fields.grid = _Swapped(grid)
    try:
        reasons = _reasons(fields, pml)
    finally:
        fields.grid = grid
    assert any("axis 0 is not the cylindrical r = 0 axis" in r for r in reasons)
    assert any("axis 1 is declared the cylindrical r = 0 axis" in r
               for r in reasons), reasons


def test_a_real_m0_dcyl_run_is_admitted_on_all_four_slots():
    """The non-vacuity floor for this file: the predicate must ADMIT something.

    Every refusal test above is satisfied by a predicate that refuses everything.
    """
    grid, fields, pml = _engine()
    for sub_step in ("step_B", "step_D"):
        verdict = cyl.cylindrical_real_curl_coverage(fields, pml, sub_step,
                                                     _Residency())
        assert verdict.covered, verdict.reasons
    for side in ("H", "E"):
        verdict = cyl.cylindrical_real_constitutive_coverage(fields, pml, side,
                                                             _Residency())
        assert verdict.covered, verdict.reasons


def test_a_conductivity_is_refused_on_every_curl_target():
    """This kernel is the PLAIN split-field recurrence; a conductivity is a different one."""
    grid, fields, pml = _engine()
    shape = tuple(grid.shape)
    fields.set_d_conductivity({name: np.full(shape, np.float32(0.3))
                               for name in ("Dx", "Dy", "Dz")})
    reasons = _reasons(fields, pml, "step_D")
    assert any("a conductivity is installed on D" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# 4. Structural contracts
# ---------------------------------------------------------------------------

def test_the_curl_kernel_fits_under_the_buffer_ceiling():
    """31 is a MEASURED platform limit and a 32nd binding is a compile error."""
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = cyl.cylindrical_curl_source("step_D", CODES)
    indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
    assert indices == sorted(indices), "the binding indices are out of order"
    assert indices == list(range(len(indices))), "a binding index is skipped"
    assert len(indices) == 22, len(indices)
    assert max(indices) < MAX_BUFFER_BINDINGS


def test_the_scan_kernel_binds_its_two_row_vectors_and_nothing_else():
    source = cyl.cylindrical_prefix_source("step_B")
    indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
    assert indices == list(range(8)), indices
    assert "weights[" in source and "divisor[" in source


def test_the_contraction_directive_is_present_in_both_kernels_and_spelled_once():
    """Mandatory, and reached through ``templates`` so the package has ONE spelling."""
    for source in cyl.enumerate_cylindrical_real_sources().values():
        assert source.count("#pragma clang fp contract(off)") == 1, source[:200]
        assert "fast_math" not in source
    for source in cyl.enumerate_cylindrical_real_sources(
            shaders.CONTRACT_FAST).values():
        assert source.count("#pragma clang fp contract(fast)") == 1


def _code_lines(source: str) -> str:
    """The source with ``//`` comment text removed.

    The spelling assertions below are about what the COMPILER sees. Matching the
    whole file would let a comment that NAMES a refuted spelling — which the shipped
    sources deliberately do, so a reader learns why — fail the test that forbids it,
    and the obvious repair is to delete the explanation.
    """
    kept = []
    for line in source.splitlines():
        head = line.split("//", 1)[0]
        if head.strip():
            kept.append(head)
    return "\n".join(kept)


def test_the_divide_is_the_ieee_one_and_never_fast_divide():
    """``fast::divide`` was MEASURED to miss 625 of the 2,046 words of the divisor
    ladder this kernel walks, so its absence is a pinned property, not a style rule.
    """
    for source in cyl.enumerate_cylindrical_real_sources().values():
        code = _code_lines(source)
        assert "fast::" not in code
        assert "1.0f / divisor" not in code, (
            "x * (1/y) is a different float32 number from x / y")


def test_no_unary_minus_and_no_refuted_negation_spelling():
    """``0.0f - x`` was MEASURED to miss 257/8,462 words on this backend.

    The kernels negate nothing, so the strongest pin is that the refuted spelling
    does not appear in the code at all.
    """
    for source in cyl.enumerate_cylindrical_real_sources().values():
        code = _code_lines(source)
        assert "0.0f -" not in code
        assert "* -1.0f" not in code


def test_the_module_imports_no_torch_at_module_scope():
    """A predicate-only host with no GPU must still be able to import this package."""
    with open(MODULE, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(name.split(".")[0] in ("torch", "numpy") for name in names), (
            f"{names} is imported at module scope; torch and numpy imports belong "
            f"inside the function that needs them")


def test_the_enumerated_sources_are_the_reachable_ones_only():
    """Ten sources: two scans and eight curls, with the r axis pinned METALLIC.

    Enumerating an unreachable specialisation would put strings in the fingerprint
    file that no configuration can launch, which reads as a claim that they ship.
    """
    labels = sorted(cyl.enumerate_cylindrical_real_sources())
    assert len(labels) == 10, labels
    assert sum(label.startswith("cyl_rderiv_prefix/") for label in labels) == 2
    curls = [label for label in labels if label.startswith("cyl_pml_curl_step/")]
    assert len(curls) == 8
    assert all(label.rsplit("/", 1)[1][0] == str(shaders.METALLIC)
               for label in curls), curls


def test_the_bz_substitution_is_one_multiply_and_one_subtract():
    """stepping.py:375 — NOT the four-operand grouping, which is a different number.

    Pinned as source text because the two spellings agree at every power-of-two
    Courant number: the mutation that distributes ``dtdx`` over the difference was
    MEASURED uncaught at Courant 0.5 and caught (4,390 words) at 0.3141592653589793.
    A byte gate can hold this only if its case matrix carries a non-power-of-two
    Courant, so the source text is pinned here as well.
    """
    source = cyl.cylindrical_curl_source("step_B", CODES)
    assert "float curl2 = dtdx * (pfx_up - pfx_here);" in source
    assert "(dtdx * pfx_up)" not in source

    d_source = cyl.cylindrical_curl_source("step_D", CODES)
    # The D side keeps the four-operand grouping and swaps only the OPERAND.
    assert "float curl2 = dtdx * ((pb_x - pb) + (a - a_y));" in d_source
    # Dx keeps the RAW Hp: stepping.py:454-455 substitutes for the Dz term only.
    assert "float curl0 = dtdx * ((c_y - c) + (b - b_z));" in d_source


def test_the_axis_rules_are_on_the_right_side_and_in_the_array_paths_order():
    """m = 0: Bx zeroed on the B side; Dz post-added then Dy zeroed on the D side."""
    b_source = cyl.cylindrical_curl_source("step_B", CODES)
    assert "v0 = at_x ? 0.0f : v0;" in b_source
    assert "v1 = at_x ? 0.0f : v1;" not in b_source
    assert "v2 = at_x ?" not in b_source
    # `axis_coef` is BOUND on the B side (one binding layout for both sub-steps, the
    # discipline `constitutive_source` follows for its inverse-epsilon buffers) and
    # must never be READ there: the B side has no on-axis increment.
    assert "axis_coef" in b_source, "the binding layout is meant to be shared"
    assert "axis_coef *" not in _code_lines(b_source), (
        "the B side has no on-axis increment; a read here would be an increment "
        "the array path never applies")

    d_source = cyl.cylindrical_curl_source("step_D", CODES)
    post_add = d_source.index("v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;")
    zero_dp = d_source.index("v1 = at_x ? 0.0f : v1;")
    assert post_add < zero_dp, (
        "stepping.py:586-587 does the Dz post-add first; the order is transcribed")
    assert "v0 = at_x ? 0.0f : v0;" not in d_source
    # The post-add reads the RAW stored Hp (fields.get_H('Hy') = g1), never the prefix.
    assert "axis_coef * g1[ii]" in d_source
    assert "axis_coef * pfx" not in d_source


# ---------------------------------------------------------------------------
# 5. The plan — two launches, and their ORDER
# ---------------------------------------------------------------------------

@requires_mps
def test_the_plan_launches_the_scan_then_the_curl_exactly_once_each():
    """A slot may not pass by NOT EXECUTING, and the order is the contract.

    The curl reads the prefix the scan writes, and the prefix is a function of THIS
    sub-step's source volume — so a scan hoisted out of the loop feeds the curl the
    previous timestep's radial derivative, which is a plausible wrong field rather
    than an error.
    """
    from meep_gpu.metal_kernels.device import Residency

    grid, fields, pml = _engine()
    residency = Residency()
    order = []
    for sub_step in ("step_B", "step_D"):
        plan = cyl.plan_cylindrical_real_curl(fields, pml, sub_step, residency)
        if plan is None:
            pytest.skip("the subnormal policy on this host refuses the family; "
                        "run with MEEP_GPU_SUBNORMAL_POLICY=flush")
        for name, table in (("scan", plan._prefix_functions),
                            ("curl", plan._curl_functions)):
            inner = table["off"]
            table["off"] = (lambda *a, _n=name, _f=inner:
                            (order.append(_n), _f(*a))[1])
        residency.sync_in()
        plan.run()
        assert plan.prefix_launches == 1 and plan.curl_launches == 1
        assert plan.launches == 2
    assert order == ["scan", "curl", "scan", "curl"], order


@requires_mps
def test_rebuilding_a_plan_reuses_the_scratch_rather_than_refusing_it():
    """This family OWNS three volumes, which the earlier families never did.

    Every volume they mirror belongs to the ENGINE, so the builder always has the
    array in hand and one name always means one array. The radial prefix and its two
    row vectors are allocated in the BUILDER, so a second build handed ``mirror`` a
    different array under the same name and it refused — correctly by its own
    contract and wrongly for this case, since the prefix is fully overwritten by the
    scan before the curl reads it and the row vectors are grid invariants.
    """
    from meep_gpu.metal_kernels.device import Residency

    grid, fields, pml = _engine()
    residency = Residency()
    first = cyl.plan_cylindrical_real_curl(fields, pml, "step_B", residency)
    if first is None:
        pytest.skip("the subnormal policy on this host refuses the family")
    second = cyl.plan_cylindrical_real_curl(fields, pml, "step_B", residency)
    assert second is not None
    for name in ("cyl_pfx:step_B", "cyl_weights:step_B", "cyl_divisor:step_B"):
        assert name in residency.names
    # The SAME device buffer, not two copies: a second allocation would leave the
    # first plan launching against a prefix the second plan's scan never fills.
    assert second._prefix_args[0] is first._prefix_args[0]
    assert second.prefix_host is first.prefix_host
    assert residency.host("cyl_pfx:step_B") is first.prefix_host
    # And an unregistered name answers None rather than raising: "not yet" is the
    # ordinary first-build answer.
    assert residency.host("cyl_pfx:step_D") is None


@requires_mps
def test_the_plan_variants_are_the_intersection_of_both_kernels():
    """A variant the plan cannot launch on BOTH kernels must not be reported.

    A plan holding a ``fast`` scan and no ``fast`` curl would report a variant whose
    guard leg would then measure the shipped curl while believing it had removed the
    guard.
    """
    from meep_gpu.metal_kernels.device import Residency

    grid, fields, pml = _engine()
    plan = cyl.plan_cylindrical_real_curl(
        fields, pml, "step_B", Residency(),
        contract_variants=(shaders.CONTRACT_OFF, shaders.CONTRACT_FAST))
    if plan is None:
        pytest.skip("the subnormal policy on this host refuses the family")
    assert plan.variants == (shaders.CONTRACT_FAST, shaders.CONTRACT_OFF)
    plan._curl_functions.pop(shaders.CONTRACT_FAST)
    assert plan.variants == (shaders.CONTRACT_OFF,)
    with pytest.raises(KeyError, match="holds no 'fast' curl variant"):
        plan.run(shaders.CONTRACT_FAST)


@requires_mps
def test_the_scan_reproduces_steppings_own_prefix_word_for_word():
    """The smallest non-vacuous version of the probe's 48-case prefix leg."""
    from meep_gpu.metal_kernels.device import compile_source

    import torch

    rng = np.random.default_rng(2026)
    for sub_step in ("step_B", "step_D"):
        shape = (13, 1, 9)
        source = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        rows = cyl.scan_rows(sub_step, shape[0])
        scanned = source
        if cyl.PREFIX_WALL_ROW[sub_step]:
            scanned = np.zeros((rows,) + shape[1:], dtype=np.float32)
            scanned[0:shape[0]] = source
        expected = stepping.cylindrical_rderiv_prefix(
            np, scanned, cyl.PREFIX_IR0[sub_step])

        weights, divisor = cyl.prefix_row_vectors(sub_step, shape[0])
        entry = compile_source(
            cyl.cylindrical_prefix_source(sub_step)).cyl_rderiv_prefix

        def upload(array):
            return torch.from_numpy(
                np.ascontiguousarray(array, dtype=np.float32).reshape(-1)).to("mps")

        # A sentinel fill, so a kernel that never ran cannot compare equal.
        out = upload(np.full((rows,) + shape[1:], np.float32(-7.5)))
        entry(out, upload(source), upload(weights), upload(divisor),
              rows, shape[1], shape[2], shape[1] * shape[2])
        torch.mps.synchronize()
        got = out.cpu().numpy().reshape(expected.shape)
        assert np.count_nonzero(got.view(np.uint32) != np.float32(-7.5).view(
            np.uint32)) > 0, "VACUOUS: the scan moved nothing"
        assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), (
            f"{sub_step}: the device scan is not "
            f"stepping.cylindrical_rderiv_prefix's answer")


# ---------------------------------------------------------------------------
# What the byte gate's NULL CONTROLS rest on
# ---------------------------------------------------------------------------
#
# `gate_metal_cylindrical_real.py` carries two mutations that must NOT be caught —
# the r = 0 near ghost restored on `c_x` and on `pb_x` — and a null control is only
# worth recording if the operand it edits is LIVE. These two tests pin the source
# facts those controls depend on, so an edit that quietly invalidates a null fails
# HERE, in the suite, rather than by turning a gate row green for the wrong reason.


def test_b_x_is_dead_on_both_sub_steps_so_a_null_planted_there_would_be_vacuous():
    """``b_x`` — ``g1``'s shift-down/up along r — is consumed by NO curl.

    THIS IS A CORRECTION PINNED AS A TEST. The build round's mutation table restored
    the CYL_AXIS near ghost on ``b_x`` and recorded the resulting null as evidence
    that compiling the r axis METALLIC is equivalent. It is not evidence of anything:
    on the B side Bz's curl is the prefix difference (the emitted source says so in
    as many words) and on the D side Dz's takes the PREFIXED ``pb_x``, so ``b_x`` is
    initialised and never read on either. The gate re-arms that null on ``c_x`` and
    ``pb_x``, which ARE read.

    If a future edit starts consuming ``b_x``, this test fails and the gate's null
    controls have to be re-derived rather than silently changing meaning.
    """
    import re  # noqa: PLC0415

    # A WORD BOUNDARY, not a substring: `pb_x` — the PREFIXED operand, which IS read —
    # contains `b_x`, and a naive `in` match would report the dead operand as live and
    # fail for the opposite of the right reason.
    token = re.compile(r"(?<![A-Za-z0-9_])b_x(?![A-Za-z0-9_])")
    for sub_step in SUB_STEPS:
        for codes in ((shaders.METALLIC, shaders.PERIODIC, shaders.METALLIC),
                      (shaders.METALLIC, shaders.PERIODIC, shaders.PERIODIC)):
            source = cyl.cylindrical_curl_source(sub_step, codes)
            body = [line for line in source.splitlines()
                    if token.search(line) and not line.strip().startswith("//")]
            assert len(body) == 1, (
                f"{sub_step}/{codes}: b_x appears on {len(body)} live lines, not just "
                f"its initialiser — it is now READ somewhere, and the gate's "
                f"near-ghost null controls rest on it being dead: {body}")
            assert "float b_x =" in body[0], body[0]
            # NON-VACUITY: the sibling operand this one is contrasted with must BE
            # read, or the test is pinning a source that carries no live neighbour.
            live = [line for line in source.splitlines()
                    if "pb_x" in line and not line.strip().startswith("//")]
            expected_live = 2 if sub_step == "step_D" else 0
            assert len(live) == expected_live, (sub_step, codes, live)


def test_the_prefix_row_zero_is_an_exact_positive_zero_which_is_load_bearing():
    """Prefix row 0 is ``+0.0`` exactly, and that — not the mask — zeroes Dz at r = 0.

    ``cylindrical_rderiv_prefix`` starts from ``zeros_like`` (stepping.py:1314), so
    every word of the scan's first row is ``0x00000000``. The gate measured what that
    buys: dropping the r ownership mask on **Dz** changes nothing (0/8), because that
    curl's radial pair at r = 0 is ``(prefix[0], the metallic zero ghost)`` and its
    phi pair is a self-difference on the one-cell invariant axis — the value is
    already the word the mask would write. On **Dy** the mask IS load-bearing (11-40
    words per case), because its radial pair is ``(Hz[0], zero)``.

    Pinned here because two of the gate's null controls are explained by this zero
    row rather than by the ownership mask, and a scan that ever started row 0 at
    anything else would change which explanation is true.
    """
    rng = np.random.default_rng(4242)
    for sub_step in SUB_STEPS:
        for radial in (9, 13, 20, 32):
            shape = (radial, 1, 12)
            source = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            prefix = stepping.cylindrical_rderiv_prefix(
                np, source, cyl.PREFIX_IR0[sub_step])
            row_zero = np.ascontiguousarray(prefix[0]).reshape(-1).view(np.uint32)
            assert np.all(row_zero == np.uint32(0)), (
                f"{sub_step}/nr={radial}: prefix row 0 is not an exact +0.0 "
                f"({sorted(set(int(w) for w in row_zero))[:4]}); the gate's Dz "
                f"near-ghost and mask nulls are explained by this zero and would "
                f"have to be re-derived")
            # NON-VACUITY: the rest of the scan must not be zero too, or the row
            # above is a fact about an empty array.
            rest = np.ascontiguousarray(prefix[1:]).reshape(-1).view(np.uint32)
            assert np.count_nonzero(rest) > 0, (
                f"{sub_step}/nr={radial}: the whole prefix is zero, so row 0 being "
                f"zero says nothing")
