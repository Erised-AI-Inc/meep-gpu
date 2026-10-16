"""Merge-bar tests for the Metal cylindrical m = 0 (real storage) family.

This family's substance is three substitutions on a certified body, and every one of
them fails SILENTLY — a wrong radial weight, a missing wall row or a dropped axis
rule all produce a smooth, plausible, entirely wrong field rather than an error. So
what is pinned here is chosen by which hazards a byte gate cannot reach:

**1. THE PAIRINGS THAT ARE HALF-CELL ERRORS.** ``ir0`` is 0.0 on the B side and 0.5
on the D side because Ep sits at the node and Hp half a cell out
(``stepping.step_B`` / ``step_D``); the wall row is the B side's only. Swapping either is a
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

**5. THE PER-CHUNK RADIAL SUM.** MEEP sums ``f_rderiv_int`` in every structure chunk
(``stepping.cylindrical_rderiv_difference``); the chunk starts the scan binds are the
array path's own, and the column walk's restart rule is pinned against one, two and
three r chunks on both sides.

The device-touching tests here are the smallest non-vacuous version, so a red merge bar
names the defect without a GPU run; they skip without MPS. The gate
(``parity/meep_gpu/gate_metal_cylindrical_real.py``) carries the case matrix.
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


#: A float32 word that is not a power of two: the D side's on-axis product with Courant
#: is MEEP's double product, which this backend forms in float32 only for such a word.
COURANT = float(np.float32(0.3141592653589793))


def _engine(shape=(20, 1, 40), courant=COURANT, m=0,
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
    """``prefix_row_vectors`` is ``stepping._cylindrical_rderiv_rows``, word for word.

    ``weights[g] = realnum(g + ir0)`` and ``rinv[g-1] = realnum(1.0 / ((g + ir0) -
    0.5))``: the quotient in double, rounded once, as MEEP stores it.
    """
    for sub_step in SUB_STEPS:
        rows = cyl.scan_rows(sub_step, 17)
        weights, rinv = cyl.prefix_row_vectors(sub_step, 17)
        expected_w, expected_r = stepping._cylindrical_rderiv_rows(
            np, rows, cyl.PREFIX_IR0[sub_step], np.float32)
        assert weights.shape == (rows,) and rinv.shape == (rows - 1,)
        assert np.array_equal(weights.view(np.uint32),
                              expected_w.reshape(-1).view(np.uint32))
        assert np.array_equal(rinv.view(np.uint32),
                              expected_r.reshape(-1).view(np.uint32))


def test_the_axis_coefficient_is_the_float32_word_of_four_courant():
    """MEEP's on-axis ``g[i] * (Courant * 4)`` is a double product rounded once.

    For a float32-word Courant the kernel's float32 product with ``4 * Courant`` is that
    word on every operand (two float32 operands multiply exactly in double); for any
    other Courant it is not, which is why the D-side predicate refuses one.
    """
    rng = np.random.default_rng(7)
    hp = rng.uniform(-1.0, 1.0, 20000).astype(np.float32)
    for courant in (COURANT, 0.5, 0.375, 0.25):
        assert float(np.float32(courant)) == courant
        factor = cyl.axis_coefficient(courant)
        assert factor == 4.0 * courant
        double = (hp.astype(np.float64) * factor).astype(np.float32)
        single = np.float32(factor) * hp
        assert np.array_equal(double.view(np.uint32), single.view(np.uint32))
    courant = 0.3
    factor = cyl.axis_coefficient(courant)
    double = (hp.astype(np.float64) * factor).astype(np.float32)
    single = np.float32(factor) * hp
    assert np.count_nonzero(double.view(np.uint32) != single.view(np.uint32)) > 0, (
        "no operand separates the double product from the float32 one at Courant 0.3, "
        "so the refusal below would guard nothing")


def test_the_d_side_refuses_a_courant_that_is_not_a_float32_word():
    """Courant 0.3 is refused on step_D by name and admitted on step_B (no product)."""
    grid, fields, pml = _engine(courant=0.3)
    d_reasons = _reasons(fields, pml, "step_D")
    assert any("is not a float32 word" in r for r in d_reasons), d_reasons
    b_reasons = _reasons(fields, pml, "step_B")
    assert not any("float32 word" in r for r in b_reasons), b_reasons
    grid, fields, pml = _engine()
    assert not any("float32 word" in r for r in _reasons(fields, pml, "step_D"))


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
    assert len(indices) == 23, len(indices)
    assert max(indices) < MAX_BUFFER_BINDINGS
    assert f"constant ChunkBounds& cb [[buffer({cyl.CURL_RECORD_BUFFER})]]" in source


def test_the_scan_kernel_binds_its_row_vectors_and_the_chunk_starts():
    for sub_step in SUB_STEPS:
        source = cyl.scan_source(sub_step)
        indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
        assert indices == list(range(9)), indices
        assert "weights[" in source and "rinv[" in source and "starts[" in source


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


def test_the_scan_multiplies_rinv_and_never_divides():
    """MEEP stores ``rinv`` and MULTIPLIES by it; a divide is another float32 word.

    The only ``/`` left is the integer index decode of the column.
    """
    for sub_step in SUB_STEPS:
        code = _code_lines(cyl.scan_source(sub_step))
        divides = [line for line in code.splitlines() if "/" in line]
        assert divides == ["    int j = int(idx) / nzi;"], divides
        assert "fast::" not in code and "rinv[" in code


def test_no_unary_minus_and_no_refuted_negation_spelling():
    """``0.0f - x`` was MEASURED to miss 257/8,462 words on this backend.

    The one negation is the certified curl's ``(-dtdx)``, a sign flip; the refuted
    spellings do not appear in the code at all.
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
    assert sum(label.startswith("cyl_rderiv_difference/") for label in labels) == 2
    curls = [label for label in labels if label.startswith("cyl_pml_curl_step/")]
    assert len(curls) == 8
    assert all(label.rsplit("/", 1)[1][0] == str(shaders.METALLIC)
               for label in curls), curls


def test_the_target_2_curl_is_one_product_of_the_radial_difference_on_both_sides():
    """``part_products.scale(dtdx, radial)``: Bz and Dz differentiate the per-chunk sum.

    The phi partner is the derivative MEEP drops (its i*m/r part is a pass of its own),
    so neither side keeps the certified four-operand stencil on target 2.
    """
    for sub_step in SUB_STEPS:
        source = cyl.cylindrical_curl_source(sub_step, CODES)
        assert source.count(cyl.CYLINDRICAL_CURL2) == 1
        assert "float curl2 = has_x" not in source


def test_the_curl_is_the_certified_curl_with_exactly_four_edits():
    """Undo the four declared edits and the certified ``shaders.curl_source`` remains.

    Targets 0 and 1 (MEEP's left-to-right bracket with the phi partner absent), the
    bounds block, the ghost rule, the ownership mask and the per-chunk forms are
    therefore the certified text, byte for byte.
    """
    for sub_step, backward in (("step_B", False), ("step_D", True)):
        for codes in ((shaders.METALLIC, shaders.PERIODIC, shaders.METALLIC),
                      (shaders.METALLIC, shaders.PERIODIC, shaders.PERIODIC)):
            source = cyl.cylindrical_curl_source(sub_step, codes)
            undone = source.replace("kernel void cyl_pml_curl_step(",
                                    "kernel void pml_curl_step(")
            undone = undone.replace(cyl._CURL_EXTRA_BINDINGS, "")
            undone = undone.replace(
                f"constant ChunkBounds& cb [[buffer({cyl.CURL_RECORD_BUFFER})]]",
                f"constant ChunkBounds& cb [[buffer({shaders.CURL_RECORD_BUFFER})]]")
            undone = undone.replace(cyl.CYLINDRICAL_CURL2, cyl._CERTIFIED_CURL2)
            undone = undone.replace(cyl._AXIS_TAILS[sub_step] + "\n", "")
            assert undone == shaders.curl_source(codes, backward), sub_step


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
# 5. The per-chunk radial sum: the chunk starts and the column walk
# ---------------------------------------------------------------------------

def _record(grid, pml):
    from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record

    return chunk_bounds_record(ChunkLayout.one_rank(grid, pml))


def _split_r(layout, cut):
    """The division with every chunk that straddles doubled r ``cut`` cut there."""
    from meep_gpu.chunk_map import ChunkLayout

    boxes, flags = [], []
    for (lo, hi), flag in zip(layout.boxes, layout.pml_flags):
        if lo[0] < cut < hi[0]:
            boxes += [(lo, (cut,) + tuple(hi[1:])), ((cut,) + tuple(lo[1:]), hi)]
            flags += [flag, flag]
        else:
            boxes.append((lo, hi))
            flags.append(flag)
    return ChunkLayout(boxes, flags, layout.frames, layout.faces)


def _divisions(grid, pml):
    """One, two and three r chunks: no layer, the high-r layer, and a cut below it."""
    from meep_gpu.chunk_map import ChunkLayout

    two = ChunkLayout.one_rank(grid, pml)
    return {"one": ChunkLayout.one_rank(grid, None), "two": two,
            "three": _split_r(two, 2 * (grid.shape[0] // 3))}


def _starts(fields, pml, layout, sub_step):
    fields.install_chunk_layout(layout)
    try:
        return stepping._cylindrical_scan_starts(fields, pml, cyl.SCAN_TARGETS[sub_step])
    finally:
        fields.install_chunk_layout(None)


def test_the_chunk_starts_of_one_two_and_three_r_chunks_are_accepted():
    """The array path's own starts, for every division the walk serves, on both sides."""
    grid, fields, pml = _engine()
    rows = grid.shape[0]
    seen = {}
    for name, layout in _divisions(grid, pml).items():
        for sub_step in SUB_STEPS:
            line = cyl.scan_starts(sub_step, _starts(fields, pml, layout, sub_step), rows)
            assert line.dtype == np.int32 and line.shape == (rows,)
            seen[(name, sub_step)] = sorted(set(int(v) for v in line))
    assert seen[("one", "step_B")] == [0] and seen[("one", "step_D")] == [0]
    assert len(seen[("two", "step_B")]) == 2 and len(seen[("three", "step_D")]) == 3, seen


def test_the_chunk_starts_are_refused_where_the_walk_cannot_restart():
    """A full volume, a wrong length, a first chunk off the axis, an unreachable change."""
    rows = 8
    with pytest.raises(ValueError, match="differ along z"):
        cyl.scan_starts("step_B", np.zeros((rows, 1, 3), dtype=np.int64), rows)
    with pytest.raises(ValueError, match="hold 7 rows"):
        cyl.scan_starts("step_B", np.zeros((7, 1, 1), dtype=np.int64), rows)
    with pytest.raises(ValueError, match="not at the axis"):
        cyl.scan_starts("step_B", np.full((rows, 1, 1), 2, dtype=np.int64), rows)
    forward = np.array([0, 0, 0, 3, 3, 3, 3, 3]).reshape(-1, 1, 1)
    assert cyl.scan_starts("step_B", forward, rows).tolist() == forward.reshape(-1).tolist()
    with pytest.raises(ValueError, match="only for a chunk starting at row 2"):
        cyl.scan_starts("step_D", forward, rows)
    backward = np.array([0, 0, 0, 0, 3, 3, 3, 3]).reshape(-1, 1, 1)
    assert cyl.scan_starts("step_D", backward, rows).tolist() == backward.reshape(-1).tolist()
    with pytest.raises(ValueError, match="only for a chunk starting at row 4"):
        cyl.scan_starts("step_B", backward, rows)


def _column_walk(sub_step, source, starts):
    """The scan kernel's column loop, transcribed in NumPy float32 (host twin)."""
    rows = source.shape[0]
    weights, rinv = cyl.prefix_row_vectors(sub_step, rows)
    f32 = np.float32
    out = np.full(source.shape, f32(-7.5), dtype=np.float32)
    for j in range(source.shape[1]):
        for k in range(source.shape[2]):
            col = source[:, j, k]
            w_prev = col[0] * weights[0]
            acc = f32(0.0)
            owner = 0
            if sub_step == "step_B":
                for g in range(rows):
                    if starts[g] != owner:
                        acc, owner = f32(0.0), starts[g]
                    w = (col[g + 1] if g + 1 < rows else f32(0.0)) * weights[g + 1]
                    nxt = acc + rinv[g] * (w - w_prev)
                    out[g, j, k] = nxt - acc
                    acc, w_prev = nxt, w
            else:
                out[0, j, k] = f32(0.0)
                for g in range(1, rows):
                    if starts[g] != owner:
                        acc, owner = f32(0.0), starts[g]
                    w = col[g] * weights[g]
                    nxt = acc + rinv[g - 1] * (w - w_prev)
                    out[g, j, k] = acc - nxt
                    acc, w_prev = nxt, w
    return out


def _array_path_difference(sub_step, source, starts):
    rows = source.shape[0]
    if sub_step == "step_B":
        extended = np.zeros((rows + 1,) + source.shape[1:], dtype=source.dtype)
        extended[:rows] = source
        return stepping.cylindrical_rderiv_difference(
            np, extended, 0.0, starts, rows, backward=False)
    return stepping.cylindrical_rderiv_difference(np, source, 0.5, starts, rows,
                                                  backward=True)


def test_the_column_walk_is_the_array_paths_per_chunk_difference_on_the_host():
    """The kernel's loop, transcribed, against ``cylindrical_rderiv_difference`` called.

    One, two and three r chunks, both sides, random and signed-zero-rich columns. A
    one-sum walk over a divided column is the control and must differ.
    """
    grid, fields, pml = _engine(shape=(21, 1, 7))
    rng = np.random.default_rng(31)
    for sub_step in SUB_STEPS:
        for name, layout in _divisions(grid, pml).items():
            starts = _starts(fields, pml, layout, sub_step)
            line = cyl.scan_starts(sub_step, starts, grid.shape[0])
            for values in (rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32),
                           np.where(rng.uniform(size=grid.shape) < 0.5, np.float32(-0.0),
                                    np.float32(0.0)).astype(np.float32)):
                expected = _array_path_difference(sub_step, values, starts)
                walked = _column_walk(sub_step, values, line)
                assert np.array_equal(walked.view(np.uint32), expected.view(np.uint32)), (
                    sub_step, name)
                if name != "one":
                    single = _column_walk(sub_step, values, np.zeros_like(line))
                    if np.any(values):
                        assert not np.array_equal(single.view(np.uint32),
                                                  expected.view(np.uint32)), (
                            "a one-sum walk over a divided column must differ")


def test_the_backward_difference_row_zero_is_an_exact_positive_zero():
    """Row 0 of Dz's difference is ``+0`` (the axis; the per-m rules own it)."""
    rng = np.random.default_rng(4242)
    for radial in (9, 13, 20):
        source = rng.uniform(-1.0, 1.0, size=(radial, 1, 12)).astype(np.float32)
        out = _array_path_difference("step_D", source, None)
        assert np.all(np.ascontiguousarray(out[0]).view(np.uint32) == 0)
        assert np.count_nonzero(np.ascontiguousarray(out[1:]).view(np.uint32)) > 0


# ---------------------------------------------------------------------------
# 6. The plan — two launches, and their ORDER
# ---------------------------------------------------------------------------

@requires_mps
def test_the_plan_launches_the_scan_then_the_curl_exactly_once_each():
    """A slot may not pass by NOT EXECUTING, and the order is the contract."""
    from meep_gpu.metal_kernels.device import Residency

    grid, fields, pml = _engine()
    residency = Residency()
    order = []
    for sub_step in ("step_B", "step_D"):
        plan = cyl.plan_cylindrical_real_curl(fields, pml, sub_step, residency,
                                              chunk_bounds=_record(grid, pml))
        if plan is None:
            pytest.skip("the subnormal policy on this host refuses the family; "
                        "run with MEEP_GPU_SUBNORMAL_POLICY=flush")
        for name, table in (("scan", plan._scan_functions),
                            ("curl", plan._curl_functions)):
            inner = table["off"]
            table["off"] = (lambda *a, _n=name, _f=inner:
                            (order.append(_n), _f(*a))[1])
        residency.sync_in()
        plan.run()
        assert plan.scan_launches == 1 and plan.curl_launches == 1
        assert plan.launches == 2
    assert order == ["scan", "curl", "scan", "curl"], order


@requires_mps
def test_rebuilding_a_plan_reuses_the_scratch_rather_than_refusing_it():
    """The radial difference and its two row vectors are plan-owned and shared."""
    from meep_gpu.metal_kernels.device import Residency

    grid, fields, pml = _engine()
    residency = Residency()
    record = _record(grid, pml)
    first = cyl.plan_cylindrical_real_curl(fields, pml, "step_B", residency,
                                           chunk_bounds=record)
    if first is None:
        pytest.skip("the subnormal policy on this host refuses the family")
    second = cyl.plan_cylindrical_real_curl(fields, pml, "step_B", residency,
                                            chunk_bounds=record)
    assert second is not None
    for name in ("cyl_radial:step_B", "cyl_weights:step_B", "cyl_rinv:step_B"):
        assert name in residency.names
    assert second._scan_args[0] is first._scan_args[0]
    assert second.radial_host is first.radial_host
    assert residency.host("cyl_radial:step_B") is first.radial_host
    assert residency.host("cyl_radial:step_D") is None


@requires_mps
def test_the_plan_variants_are_the_intersection_of_both_kernels():
    """A variant the plan cannot launch on BOTH kernels must not be reported."""
    from meep_gpu.metal_kernels.device import Residency

    grid, fields, pml = _engine()
    plan = cyl.plan_cylindrical_real_curl(
        fields, pml, "step_B", Residency(),
        contract_variants=(shaders.CONTRACT_OFF, shaders.CONTRACT_FAST),
        chunk_bounds=_record(grid, pml))
    if plan is None:
        pytest.skip("the subnormal policy on this host refuses the family")
    assert plan.variants == (shaders.CONTRACT_FAST, shaders.CONTRACT_OFF)
    plan._curl_functions.pop(shaders.CONTRACT_FAST)
    assert plan.variants == (shaders.CONTRACT_OFF,)
    with pytest.raises(KeyError, match="holds no 'fast' curl variant"):
        plan.run(shaders.CONTRACT_FAST)


def test_a_plan_without_the_chunk_record_is_refused_by_name():
    """No default division: the record is a required keyword, refused when None."""
    from meep_gpu.chunk_map import ChunkLayoutRefusal

    grid, fields, pml = _engine()
    with pytest.raises(TypeError):
        cyl.plan_cylindrical_real_curl(fields, pml, "step_B", _Residency())  # noqa
    with pytest.raises(ChunkLayoutRefusal, match="no chunk record"):
        cyl._build_curl_plan("step_B", grid.shape, 0.5, 0.5, CODES, _Residency(), {},
                             [], None, None, "test", None, None, ("off",))


@requires_mps
def test_the_scan_reproduces_the_array_paths_per_chunk_difference_word_for_word():
    """The device scan against ``stepping.cylindrical_rderiv_difference``, 1-3 chunks."""
    from meep_gpu.metal_kernels.device import compile_source

    import torch

    grid, fields, pml = _engine(shape=(21, 1, 9))
    rng = np.random.default_rng(2026)

    def upload(array, dtype=np.float32):
        return torch.from_numpy(
            np.ascontiguousarray(array, dtype=dtype).reshape(-1)).to("mps")

    for sub_step in ("step_B", "step_D"):
        entry = compile_source(cyl.scan_source(sub_step)).cyl_rderiv_difference
        weights, rinv = cyl.prefix_row_vectors(sub_step, grid.shape[0])
        for name, layout in _divisions(grid, pml).items():
            starts = _starts(fields, pml, layout, sub_step)
            line = cyl.scan_starts(sub_step, starts, grid.shape[0])
            source = rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
            expected = _array_path_difference(sub_step, source, starts)
            out = upload(np.full(grid.shape, np.float32(-7.5)))
            entry(out, upload(source), upload(weights), upload(rinv),
                  upload(line, np.int32), grid.shape[0], grid.shape[1], grid.shape[2],
                  grid.shape[1] * grid.shape[2])
            torch.mps.synchronize()
            got = out.cpu().numpy().reshape(expected.shape)
            assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), (
                sub_step, name)


@requires_mps
def test_one_curl_sub_step_is_byte_identical_to_stepping():
    """The plan against ``stepping.step_B`` / ``step_D`` on a random full-volume state."""
    from meep_gpu.metal_kernels.device import Residency

    names = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
             "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")
    for sub_step, oracle in (("step_B", stepping.step_B), ("step_D", stepping.step_D)):
        reference = _engine()
        device = _engine()
        rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
        for (_g, f, _p), rng in ((reference, rng_a), (device, rng_b)):
            for name in names:
                getattr(f, name)[...] = rng.uniform(
                    -1.0, 1.0, size=getattr(f, name).shape).astype(np.float32)
        oracle(reference[1], reference[2])
        residency = Residency()
        plan = cyl.plan_cylindrical_real_curl(device[1], device[2], sub_step, residency,
                                              chunk_bounds=_record(device[0], device[2]))
        if plan is None:
            pytest.skip("the subnormal policy on this host refuses the family")
        residency.sync_in()
        plan.run()
        residency.sync_out()
        for name in names:
            want = np.ascontiguousarray(getattr(reference[1], name)).view(np.uint32)
            got = np.ascontiguousarray(getattr(device[1], name)).view(np.uint32)
            assert np.array_equal(want, got), (sub_step, name,
                                               int(np.count_nonzero(want != got)))
