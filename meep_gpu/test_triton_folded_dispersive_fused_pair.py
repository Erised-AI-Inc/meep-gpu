"""Tests for the folded DISPERSIVE fused electric pair — ``step_D`` into ``update_E``.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity per COMPLETE driver step against the array path and against the three
separately certified products this launch replaces — lives in
``parity/meep_gpu/probe_triton_folded_dispersive_fused_pair.py``. A green suite here
is not a certification, and the module under test says so too.

What IS pinned here:

* the DISJOINTNESS this product rests on, in both directions and on the same two
  grids: ``folded_fused_pair`` refuses a registered susceptibility by name and this
  product's E half REQUIRES one. Two predicates admitting one slot leaves
  ``_select_slot`` unable to choose and the slot UNSELECTED — a silent coverage LOSS,
  not an error — so it is measured rather than argued;
* the transcription, read off the shipped source: the curl half, the two fills, the
  wall clear and the constitutive grouping must be the RELEASED
  ``folded_fused_pair`` kernel's, and the pole chain must be
  ``dispersive_update_e.constitutive_step_dispersive``'s eight arms;
* THE ONE SUBSTITUTION, in both of its two places, and that the ghost's chain is
  indexed at ``dst`` — the defect this product could have and the one no comparison
  restricted to owned cells would ever see;
* every clause of the seam predicate, in both directions, including
  ``CARRIES_DEPOSIT_REPAIR`` — which on this cell is the product rather than one
  clause of it, since all four corpus rows declare an electric source;
* the deferral — nothing in ``launch.py`` or ``fastpath.py`` names this module;
* the gate's own no-device legs, which run here;
* that every armed mutation is scored on a grid that ENTERS the branch it rewrites,
  including the pole count.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import pathlib
import sys

import numpy
import pytest

from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields, IYEE_SHIFTS
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import dispersive_update_e as dispersive
from meep_gpu.triton_kernels import folded_dispersive_fused_pair as product
from meep_gpu.triton_kernels import folded_dispersive_update_e as constitutive_half
from meep_gpu.triton_kernels import folded_fused_pair as plain
from meep_gpu.triton_kernels import symmetry

MODULE_NAME = "meep_gpu.triton_kernels.folded_dispersive_fused_pair"
PACKAGE_DIR = pathlib.Path(product.__file__).parent
MODULE_PATH = PACKAGE_DIR / "folded_dispersive_fused_pair.py"
API_ROOT = PACKAGE_DIR.parents[1]
GATE = (API_ROOT / "parity" / "meep_gpu"
        / "probe_triton_folded_dispersive_fused_pair.py")

D_COMPONENTS = ("Dx", "Dy", "Dz")
E_COMPONENTS = ("Ex", "Ey", "Ez")


def build(counts=(2, 1, 1), cell=(1.6, 3.0, 1.0), boundaries="metallic",
          mirrors=(("Y", 1),), thickness=0.2, seed=17):
    """A real Grid/Fields/PML triple on NumPy carrying ``counts[c]`` poles.

    Built the way the engine builds one: a pole whose sigma is exactly zero on a
    component does not drive it (``sigma_is_trivial``, dispersion.py:600), so a
    ``(2, 1, 1)`` configuration is an ordinary anisotropic material rather than a
    harness contrivance — and it is the ASYMMETRIC one, which is the harder fixture
    for a per-component defect.
    """
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    fields.enable_field_storage()
    shape = grid.shape
    for index in range(max(counts) if counts else 0):
        term = Susceptibility(frequency=1.0, gamma=0.1)
        sigmas = {name: (0.3 + 0.05 * index) if counts[axis] > index else 0.0
                  for axis, name in enumerate(E_COMPONENTS)}
        fields.polarizations.append(
            PolarizationState(term, sigmas, grid, numpy.float32))
    rng = numpy.random.default_rng(seed)
    for state in fields.polarizations:
        for component in E_COMPONENTS:
            for slot in ("P", "P_prev"):
                buffer = getattr(state, slot, {}).get(component)
                if buffer is not None:
                    buffer[...] = rng.uniform(
                        -0.2, 0.2, size=shape).astype(buffer.dtype)
    return fields, PML(grid=grid, thickness=thickness)


def residual(verdict):
    """The reasons that are not the NumPy-host backend clause."""
    return [reason for reason in verdict.reasons if "cupy" not in reason]


def electric_deposit(fields):
    """A REAL source that publishes the index the injection writes.

    :class:`Source` below carries a ``field_type`` and nothing else: enough to PLACE
    a source in a seam, and deliberately not enough to CARRY one — the repair refuses
    it by name for publishing no deposit index. A case about the carry needs the
    engine's own source, and one that deposits nothing would make the admitting
    assertion pass by measuring an empty scatter.
    """
    from meep_gpu import deposit_repair
    from meep_gpu.sources import GaussianEnvelope, VolumeSource

    source = VolumeSource(grid=fields.grid, component="Ez",
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert source._n_source_points, "the case deposits nothing and measures nothing"
    assert deposit_repair._deposit_index(source) is not None, (
        "the case cannot exercise the repair: this source publishes no deposit index")
    return source


class Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "probe_folded_dispersive_fused_D", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shipped(name: str) -> str:
    """One shipped function's source text, docstring removed."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(found for found in ast.walk(tree)
                if isinstance(found, ast.FunctionDef) and found.name == name)
    segment = ast.get_source_segment(text, node)
    lines = segment.splitlines()
    body = node.body[0]
    if (isinstance(body, ast.Expr) and isinstance(body.value, ast.Constant)
            and isinstance(body.value.value, str)):
        start = body.lineno - node.lineno
        end = (body.end_lineno or body.lineno) - node.lineno
        lines = lines[:start] + lines[end + 1:]
    return "\n".join(line.split("#", 1)[0].strip() for line in lines
                     if line.split("#", 1)[0].strip())


# ---------------------------------------------------------------------------
# The disjointness this product rests on
# ---------------------------------------------------------------------------

def test_a_poled_fold_is_THIS_products_and_a_bare_fold_is_the_plain_pairs():
    """Two predicates admitting one slot is how a wrong body gets chosen at random.

    ``launch._select_slot`` fails CLOSED on that: the slot is left UNSELECTED and the
    row falls back to the array path — a silent coverage LOSS, which is worse than an
    error. Both directions, on the same two grids.
    """
    poled, pml = build(counts=(2, 1, 1))
    assert residual(product.folded_dispersive_fused_pair_coverage(
        poled, pml, ())) == []
    assert any("susceptibility is registered" in reason for reason in
               plain.folded_fused_pair_coverage(poled, pml, ()).reasons)

    bare, pml = build(counts=())
    assert residual(plain.folded_fused_pair_coverage(bare, pml, ())) == []
    assert any("no susceptibility is registered" in reason for reason in
               product.folded_dispersive_fused_pair_coverage(bare, pml, ()).reasons)


def test_the_disjointness_clause_is_restated_here_and_not_only_inherited():
    """The E half already requires a pole; this product says so in its OWN reasons.

    Inheriting it would be enough for the answer and not for the reader: the clause
    that keeps this product out of ``folded_fused_pair``'s cell has to be legible in
    the predicate that competes for it.
    """
    bare, pml = build(counts=())
    reasons = product.folded_dispersive_fused_pair_coverage(bare, pml, ()).reasons
    own = [r for r in reasons if r.startswith("no susceptibility is registered")]
    half = [r for r in reasons
            if r.startswith("folded dispersive constitutive half:")
            and "no susceptibility" in r]
    assert own, reasons
    assert half, reasons


def test_the_two_halves_are_the_named_certified_predicates():
    """The conjunction is of SHIPPED predicates, not of a restatement of them."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP)" in source
    assert "folded_dispersive_constitutive_coverage(fields, pml)" in source
    assert callable(symmetry.folded_composition_curl_coverage)
    assert callable(constitutive_half.folded_dispersive_constitutive_coverage)


def test_the_predicate_reports_both_halves_reasons_with_their_side_named():
    fields, pml = build(counts=(2, 1, 1), mirrors=())
    reasons = product.folded_dispersive_fused_pair_coverage(fields, pml, ()).reasons
    assert any(r.startswith("folded curl half: ") for r in reasons), reasons
    assert any(r.startswith("folded dispersive constitutive half: ")
               for r in reasons), reasons


# ---------------------------------------------------------------------------
# The seam clauses
# ---------------------------------------------------------------------------

def test_a_folded_metallic_grid_is_admitted_modulo_the_numpy_host():
    fields, pml = build()
    assert residual(product.folded_dispersive_fused_pair_coverage(
        fields, pml, ())) == []


def test_a_folded_periodic_grid_is_admitted_because_the_far_carry_is_inline():
    fields, pml = build(boundaries="periodic")
    assert residual(product.folded_dispersive_fused_pair_coverage(
        fields, pml, ())) == []


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    """``Fields`` does not hold the source list, so ignorance is never an empty set."""
    fields, pml = build()
    reasons = product.folded_dispersive_fused_pair_coverage(
        fields, pml, None).reasons
    assert any("was not declared" in reason for reason in reasons), reasons


def test_a_magnetic_source_is_admitted_and_an_electric_DEPOSIT_is_carried():
    """The polarity, both ways, and the CARRY — which on this cell is the product.

    All four corpus rows of ``(folded PML, folded dispersive)`` declare an electric
    source, so with ``CARRIES_DEPOSIT_REPAIR`` at False this arm would admit NOTHING.
    """
    fields, pml = build()
    assert residual(product.folded_dispersive_fused_pair_coverage(
        fields, pml, (Source("B"),))) == []
    assert residual(product.folded_dispersive_fused_pair_coverage(
        fields, pml, (electric_deposit(fields),))) == []


def test_an_electric_source_that_publishes_no_deposit_index_is_REFUSED():
    """FAIL CLOSED. A source the repair cannot save is refused BY NAME."""
    fields, pml = build()
    reasons = product.folded_dispersive_fused_pair_coverage(
        fields, pml, (Source("D"),)).reasons
    assert any("does not publish the index it writes" in reason
               for reason in reasons), reasons


def test_the_flag_is_declared_and_is_passed_to_the_clause_that_reads_it():
    """A flag the seam clause is not handed is a claim nothing acts on."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert product.CARRIES_DEPOSIT_REPAIR is True
    assert "CARRIES_DEPOSIT_REPAIR = True" in source
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source


def test_the_flag_at_False_would_cost_the_whole_cell(monkeypatch):
    """MEASURED, not asserted: with the flag off, an electric deposit is refused.

    This is the trap the round before this one paid for on the Dcyl pair — a module
    that compiles, passes every other leg and serves nothing at all.
    """
    fields, pml = build()
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    reasons = product.folded_dispersive_fused_pair_coverage(
        fields, pml, (electric_deposit(fields),)).reasons
    assert any("is electric" in reason for reason in reasons), reasons


def test_an_unfolded_grid_is_refused_because_this_is_a_composition_product():
    fields, pml = build(mirrors=())
    reasons = product.folded_dispersive_fused_pair_coverage(fields, pml, ()).reasons
    assert any("mirror plane" in reason for reason in reasons), reasons


def test_an_offdiagonal_row_is_refused_by_name_as_the_stencil_it_is():
    """``has_offdiagonal_epsilon`` is a PROPERTY over the installed rows, so the row
    is installed rather than the flag forced: a forced flag would test the clause
    against a state the engine cannot be in."""
    fields, pml = build()
    shape = fields.grid.shape
    fields._chi1inv_offdiagonal = {
        "Ex": {"Ey": numpy.full(shape, 0.05, dtype=numpy.float32)}}
    assert fields.has_offdiagonal_epsilon
    reasons = product.folded_dispersive_fused_pair_coverage(fields, pml, ()).reasons
    assert any("STENCIL" in reason for reason in reasons), reasons


def test_more_poles_than_the_compiled_slots_is_a_refusal_and_not_a_ValueError():
    """An over-ceiling configuration must be refused BY NAME rather than raise at
    plan time, and the from-arrays route's ceiling and this one must not drift."""
    fields, pml = build(counts=(dispersive.MAX_POLES + 1, 1, 1))
    reasons = product.folded_dispersive_fused_pair_coverage(fields, pml, ()).reasons
    assert any(f"MAX_POLES={dispersive.MAX_POLES}" in reason
               for reason in reasons), reasons
    assert product.plan_folded_dispersive_fused_pair(fields, pml, ()) is None


def test_the_builder_refuses_every_configuration_the_predicate_refuses():
    """``None`` is the only refusal: a raise reaches a caller that would otherwise
    have stepped correctly."""
    for kwargs in ({"mirrors": ()}, {"counts": ()},
                   {"counts": (dispersive.MAX_POLES + 1, 1, 1)}):
        fields, pml = build(**kwargs)
        assert not product.folded_dispersive_fused_pair_coverage(
            fields, pml, ()).covered
        assert product.plan_folded_dispersive_fused_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# The transcription: this body is a COPY, plus one substitution
# ---------------------------------------------------------------------------

CURL_LINES = (
    "curl0 = dtdx * ((c_y - c) + (b - b_z))",
    "curl1 = dtdx * ((a_z - a) + (c - c_x))",
    "curl2 = dtdx * ((b_x - b) + (a - a_y))",
    "n0 = ((p0 * km_y) - curl0) * si_y",
    "n1 = ((p1 * km_z) - curl1) * si_z",
    "n2 = ((p2 * km_x) - curl2) * si_x",
)


def plain_kernel_text() -> str:
    path = PACKAGE_DIR / "folded_fused_pair.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(found for found in ast.walk(tree)
                if isinstance(found, ast.FunctionDef)
                and found.name == "folded_fused_curl_constitutive_D")
    return "\n".join(
        line.split("#", 1)[0].strip()
        for line in ast.get_source_segment(text, node).splitlines()
        if line.split("#", 1)[0].strip())


@pytest.mark.parametrize("line", CURL_LINES)
def test_every_curl_line_is_the_released_bodys_own(line):
    """The parenthesisation decides float32 bits; a reformat is a different number."""
    assert line in shipped("folded_dispersive_fused_curl_constitutive_D")
    assert line in plain_kernel_text()


def test_the_owned_cells_source_is_D_minus_the_pole_sum_and_not_D():
    text = shipped("folded_dispersive_fused_curl_constitutive_D")
    for index, register in ((0, "v0"), (1, "v1"), (2, "v2")):
        assert f"s{index} = _subtract_poles_at({register}, " in text
        assert f"src{index} = s{index} * tl.load(ie{index} + idx" in text
        # ...and the PLAIN product must be GONE.
        assert f"src{index} = {register} * tl.load(ie{index} + idx" not in text


def test_the_ghost_subtracts_its_OWN_cells_poles():
    """The array path's fills write D and never P (stepping.py:1497-1498, :1529-1532),
    so ``update_E`` at an imaged cell reads THAT cell's poles. Imaging the source
    lane's would be a smooth, plausible, wrong field on every folded plane, and no
    comparison restricted to owned cells would ever see it."""
    text = shipped("_carry_ghost_E_dispersive")
    assert "s = _subtract_poles_at(ghost, p0, p1, p2, p3, p4, p5, p6, p7," in text
    assert "dst, mask, NP)" in text
    assert "src = s * tl.load(ie + dst, mask=mask, other=0.0)" in text


def shipped_from(path: pathlib.Path, name: str) -> str:
    """One function's source text from ANY file, docstring and comments removed."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(found for found in ast.walk(tree)
                if isinstance(found, ast.FunctionDef) and found.name == name)
    lines = ast.get_source_segment(text, node).splitlines()
    body = node.body[0]
    if (isinstance(body, ast.Expr) and isinstance(body.value, ast.Constant)
            and isinstance(body.value.value, str)):
        start = body.lineno - node.lineno
        end = (body.end_lineno or body.lineno) - node.lineno
        lines = lines[:start] + lines[end + 1:]
    return "\n".join(line.split("#", 1)[0].strip() for line in lines
                     if line.split("#", 1)[0].strip())


def test_the_carry_is_the_certified_one_with_only_its_source_line_split():
    """Every statement the released ``_carry_ghost_E`` carries must still be here."""
    certified = shipped_from(PACKAGE_DIR / "folded_fused_pair.py",
                             "_carry_ghost_E").splitlines()
    mine = shipped("_carry_ghost_E_dispersive")
    checked = 0
    for line in certified:
        if line.startswith("src = ghost * ") or line.startswith("def "):
            continue    # the ONE licensed difference, and the signature
        assert line in mine, line
        checked += 1
    assert checked == 7, checked


def test_the_pole_chain_is_the_certified_eight_arms_in_order():
    """The order is bit-load-bearing: pre-summing was caught 20/20 at two poles."""
    text = shipped("_subtract_poles_at")
    for k in range(8):
        assert f"if NP > {k}:" in text
        assert (f"base = base - tl.load(p{k} + where, mask=mask, other=0.0)") in text
    subtractions = [line for line in text.splitlines()
                    if line.startswith("base = base - ")]
    assert len(subtractions) == 8, subtractions
    # ONE load per arm. Two on one line would be a pre-summed pair, which is a
    # different float32 number at two or more poles.
    assert all(line.count("tl.load(") == 1 for line in subtractions), subtractions


def test_the_certified_source_of_the_pole_chain_still_spells_it_the_same_way():
    """If ``dispersive_update_e`` moves, this transcription's source has moved."""
    path = PACKAGE_DIR / "dispersive_update_e.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(found for found in ast.walk(tree)
                if isinstance(found, ast.FunctionDef)
                and found.name == "constitutive_step_dispersive")
    certified = ast.get_source_segment(text, node)
    for k in range(8):
        assert f"if NP0 > {k}:" in certified
        assert f"s0 = s0 - tl.load(a{k} + idx, mask=live, other=0.0)" in certified


def test_every_carry_passes_its_own_components_pole_group_and_count():
    """A carry that handed component 0's poles to component 2 would be invisible
    wherever the two pole counts agree."""
    text = shipped("folded_dispersive_fused_curl_constitutive_D")
    lines = text.splitlines()
    for index, group in enumerate(("a0, a1, a2, a3, a4, a5, a6, a7,",
                                   "b0, b1, b2, b3, b4, b5, b6, b7,",
                                   "c0, c1, c2, c3, c4, c5, c6, c7,")):
        calls = sum(1 for line in lines if line.startswith(
            f"_carry_ghost_E_dispersive(f{index}, w{index}, e{index}, ie{index},"))
        assert calls == 7, (index, calls)
        # the signature, the owned cell's chain, and one continuation per carry
        assert sum(1 for line in lines if group in line) == calls + 2
        assert sum(1 for line in lines if f", NP{index})" in line) == calls + 1


def test_the_top_plane_mask_is_the_D_familys_three_lines_and_not_the_Bs_six():
    text = shipped("folded_dispersive_fused_curl_constitutive_D")
    masks = [line for line in text.splitlines()
             if line.startswith("curl") and "tl.where(last_" in line]
    assert len(masks) == 3, masks


def test_no_carry_open_codes_the_ghost_store():
    text = shipped("folded_dispersive_fused_curl_constitutive_D")
    for index in (0, 1, 2):
        assert f"tl.store(f{index} + dst" not in text


def test_the_backward_constexpr_is_step_Ds_and_only_step_Ds():
    from meep_gpu.triton_kernels.launch import SUB_STEPS
    assert product.BACKWARD == SUB_STEPS[product.CURL_SUB_STEP]["backward"] == 1


def test_the_fill_geometry_is_read_off_the_engines_own_Yee_table():
    """NEAR on the two axes that are NOT the component's own, FAR on its own."""
    for index, name in enumerate(D_COMPONENTS):
        assert product.NEAR_FILL_AXES[index] == tuple(
            axis for axis in range(3) if IYEE_SHIFTS[name][axis] == 0)
        assert product.FAR_FILL_AXES[index] == tuple(
            axis for axis in range(3) if IYEE_SHIFTS[name][axis] == 1)
        assert set(product.NEAR_FILL_AXES[index]) & set(
            product.FAR_FILL_AXES[index]) == set()


def test_mirror_phases_matches_the_folded_pairs():
    """The two bodies are pinned equal; a shared helper would make this unnecessary
    and is not available without pulling a Triton kernel definition onto the laptop."""
    for boundaries in ("metallic", "periodic"):
        for mirrors in ((("Y", 1),), (("Y", -1),), (("X", 1), ("Y", -1))):
            fields, _pml = build(cell=(3.0, 3.0, 1.0), boundaries=boundaries,
                                 mirrors=mirrors)
            assert (product.mirror_phases(fields.grid)
                    == plain.mirror_phases(fields.grid))


def test_the_declared_passes_are_the_five_the_driver_calls_in_this_seam():
    assert product.REPLACES == ("step_D", "fill_D", "zero_metal_D",
                                "fill_folded_far_ghosts_D", "update_E")
    assert product.REPLACES == plain.REPLACES


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_refuses_a_fold_that_also_carries_a_wall_or_a_bad_phase():
    counts = (2, 1, 1)
    fields, pml = build(counts=counts)
    order = dispersive.poles_per_component(fields)
    binding = dispersive.LivePoleBinding(fields, order)
    common = dict(shape=(6, 6, 6), dtdx=0.3, block=64,
                  targets=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
                  auxiliaries=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
                  sources=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
                  curl_coefficients=[numpy.zeros(6, dtype=numpy.float32)] * 6,
                  e_targets=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
                  e_aux=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
                  inverse_epsilon=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
                  e_coefficients=[numpy.zeros(6, dtype=numpy.float32)] * 6,
                  poles=binding)
    with pytest.raises(ValueError, match="no axis is folded"):
        product.FoldedDispersiveFusedPairPlan(
            bc=(0, 0, 0), zero_metal=(False,) * 3, phases=(0, 0, 0), **common)
    with pytest.raises(ValueError, match="fold and a wall clear"):
        product.FoldedDispersiveFusedPairPlan(
            bc=(symmetry.CODE_MIRROR_METALLIC, 0, 0), zero_metal=(True, False, False),
            phases=(1, 0, 0), **common)
    with pytest.raises(ValueError, match="not [+]1 or -1"):
        product.FoldedDispersiveFusedPairPlan(
            bc=(symmetry.CODE_MIRROR_METALLIC, 0, 0),
            zero_metal=(False,) * 3, phases=(0, 0, 0), **common)
    with pytest.raises(ValueError, match="outside"):
        product.FoldedDispersiveFusedPairPlan(
            bc=(symmetry.CODE_MIRROR_PERIODIC, 0, 0), zero_metal=(False,) * 3,
            phases=(1, 0, 0), reflect=(9, None, None), **common)


def test_the_plan_refuses_a_pole_count_past_the_compiled_slots():
    fields, _pml = build(counts=(2, 1, 1))
    order = dispersive.poles_per_component(fields)
    binding = dispersive.LivePoleBinding(fields, order)
    binding.counts = (dispersive.MAX_POLES + 1, 1, 1)

    class _Over:
        counts = (dispersive.MAX_POLES + 1, 1, 1)

        def arrays(self):  # pragma: no cover - the plan raises before this
            return ((), (), ())

    with pytest.raises(ValueError, match="MAX_POLES"):
        product.FoldedDispersiveFusedPairPlan(
            shape=(6, 6, 6), dtdx=0.3, bc=(symmetry.CODE_MIRROR_METALLIC, 0, 0),
            zero_metal=(False,) * 3, phases=(1, 0, 0), block=64,
            targets=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
            auxiliaries=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
            sources=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
            curl_coefficients=[numpy.zeros(6, dtype=numpy.float32)] * 6,
            e_targets=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
            e_aux=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
            inverse_epsilon=[numpy.zeros((6, 6, 6), dtype=numpy.float32)] * 3,
            e_coefficients=[numpy.zeros(6, dtype=numpy.float32)] * 6,
            poles=_Over())


def test_the_plan_binds_a_LIVE_pole_binding_and_never_a_snapshot():
    """``PolarizationState.update`` rotates P/P_prev/scratch every step, so a pointer
    captured at plan time is one timestep stale and stale in a way that still
    computes."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "LivePoleBinding(fields, poles_per_component(fields))" in source
    assert "groups = self._poles.arrays()" in source


def test_the_plan_declares_the_five_passes_it_replaces():
    assert product.FoldedDispersiveFusedPairPlan.replaces == product.REPLACES


# ---------------------------------------------------------------------------
# The optional-import contract and the deferral
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    assert MODULE_NAME not in sys.modules


def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.startswith("triton")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.import_module(MODULE_NAME)
    fields, pml = build()
    assert isinstance(reloaded.folded_dispersive_fused_pair_coverage(
        fields, pml, ()).covered, bool)
    assert reloaded.plan_folded_dispersive_fused_pair(fields, pml, ()) is None
    with pytest.raises(ImportError):
        reloaded.folded_dispersive_fused_curl_constitutive_D_kernel()


def test_the_composer_routes_this_product_and_dispatch_admits_it():
    """ROUTED 2026-09-02 by the installer wave, RELEASED at dispatch — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this test was named
    ``..._and_dispatch_still_refuses_it`` for as long as the label sat outside
    ``fastpath.RELEASED_FUSED_ARMS``. This product spent that time refused for a
    reason none of its unreleased siblings carried: it WAS driven. On 2026-09-11
    ``dispatch_fused_route_2026-09-11_realarms`` took it through the driver's own
    consults on ``folded_dispersive_2d``, where it dispatched at 7 of 7 slots and
    every ``step()`` checkpoint read fused==array to 1600 steps — and the route
    gate's SECOND consult site was read as refusing it: after
    ``synchronize_magnetic_fields``, ``fused_vs_array`` False AND
    ``unfused_vs_array`` False on a leg that installs NO fused product.

    THAT REFUSAL WAS A MISATTRIBUTION, which is why this test asserts the release
    rather than the hold. Re-driven on the real route gate with the arm admitted
    in-process, three times in fresh processes over the full 1600-step ladder, the
    site reads PASS with ``first_divergent_checkpoint`` null and 0 of 41 arrays
    differing over 200,080 words on BOTH comparisons, and the armed nulls still
    fire. The signature belonged to the subnormal policy installing LAZILY at the
    first dispatch's plan freeze: CuPy kernels compiled before it keep their own
    ``-ftz=true``, so the ARRAY leg flushes subnormals the dispatch legs keep —
    which is why both legs appeared to diverge by identical amounts, and why no
    mechanism was ever found (this arm spans ``step_D``/``update_E``, while
    ``synchronize_magnetic_fields`` consults only the magnetic slots). The engine-
    side ordering fix is a separate round.

    THE PARTITION IS STILL EXACTLY ONE OF TWO, in the other direction now: the
    label is in ``ARM_CERTIFICATION`` and out of ``PENDING_DEVICE_GATE_ARMS``.
    Both are asserted rather than one, because a label released while the pending
    rung still held it would be a plan claiming a certification the ladder denies.
    The certification is asserted by KEY rather than by presence — the ledger entry
    ``triton_folded_dispersive_fused_pair_device_gate`` existed before the hold was
    lifted, and it is what a dispatching run names.

    WHAT THE RELEASE ROW DOES NOT COVER is asserted too: its
    ``susceptibilities`` axis is pinned at ONE pole, because the route gate's
    Triton envelope witness is the same fold at FIVE and stops witnessing if any
    folded row admits it.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_folded_dispersive_fused_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["folded_dispersive_fused_pair"]
    assert row["module"] == "folded_dispersive_fused_pair"
    assert row["builder"] == "plan_folded_dispersive_fused_pair"
    label = row["label"]
    assert label == 'fused pair D (folded dispersive)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert "folded_dispersive_2d" in _fastpath.RELEASED_FUSED_ARMS[label]
    assert _fastpath.ARM_CERTIFICATION[label] == (
        "folded_dispersive_fused_pair",
        "triton_folded_dispersive_fused_pair_device_gate")
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
    axes = dict((axis, required)
                for axis, required, _why in _fastpath.FUSED_RELEASE_ARM_AXES[label])
    assert axes["susceptibilities"] == 1, axes
    assert axes["folded"] == "required", axes
def test_the_module_is_declared_ROUTED_rather_than_NOT_AN_ARM():
    """It LEFT ``test_triton_planner_composition.NOT_AN_ARM`` on 2026-09-02.

    Both directions, because both failures are wrong in opposite ways: a routed
    module still named in that tuple would tell a reader the planner cannot reach
    it, and a module missing from ``SUPPORT_MODULES`` would break the lazy-import
    seam this package rests on.
    """
    from meep_gpu.test_triton_planner_composition import NOT_AN_ARM  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    assert "folded_dispersive_fused_pair" not in NOT_AN_ARM
    assert "folded_dispersive_fused_pair" in _launch.SUPPORT_MODULES
def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    from meep_gpu.test_fused_pair_deposit_wiring import WIRED_FOR_THE_REPAIR
    key = "triton_kernels/folded_dispersive_fused_pair.py"
    assert key in WIRED_FOR_THE_REPAIR


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" \
        in source
    assert source.count("kernel[self._grid](") == 1


# ---------------------------------------------------------------------------
# The gate's own no-device legs
# ---------------------------------------------------------------------------

def test_the_gate_exists_and_names_the_product_it_measures():
    assert GATE.exists()
    text = GATE.read_text(encoding="utf-8")
    assert "meep_gpu.triton_kernels.folded_dispersive_fused_pair" in text
    assert "folded_dispersive_fused_curl_constitutive_D" in text


def test_the_gates_transcription_leg_traces_every_line_to_its_source():
    row = load_gate().transcription_leg()
    assert row["passed"], row["findings"]
    assert row["pole_arms_checked"] == 8
    assert row["curl_lines_checked"] == 6


def test_the_gates_predicate_leg_admits_every_case_it_will_score():
    row = load_gate().predicate_leg()
    assert row["passed"], row["findings"]
    assert row["cases"], "the predicate leg scored no case"


def test_the_gates_corpus_leg_lands_on_the_cell_the_board_scored():
    gate = load_gate()
    row = gate.corpus_admission_leg()
    assert row["passed"], row["findings"]
    assert row["seam_instances_gained"] == 4, row["funnel"]
    assert row["board_rows"] == row["funnel"]["cell_rows"]
    # ...and EVERY row of the cell carries an electric deposit, which is why the
    # deposit repair is this product rather than one clause of it.
    assert (row["funnel"]["carrying_an_electric_deposit"]
            == row["funnel"]["both"] == 4)


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    import meep_gpu.driver as driver_module
    gate = load_gate()
    for name in gate.SEAM_PASSES:
        assert hasattr(driver_module, name), name


def test_every_mutation_is_scored_on_a_grid_that_enters_the_branch_it_rewrites():
    gate = load_gate()
    for name, _why, _expectation, _rewrite in gate.mutation_table():
        index, case = gate.mutation_case_for(name)
        assert gate.CASES[index] is case


def test_the_order_mutations_are_scored_where_every_component_carries_two_poles():
    """A swap or a pre-sum on a one-pole component reads a DEAD slot instead, which
    is a different defect scored under this one's name."""
    gate = load_gate()
    for name in gate.MUTATION_REQUIRES_TWO_POLES_EVERYWHERE:
        _index, case = gate.mutation_case_for(name)
        assert min(gate._pole_counts(case[4])) >= 2, case[0]


def test_the_parity_mutations_are_scored_on_an_ODD_plane():
    gate = load_gate()
    for name, axes in gate.MUTATION_REQUIRES_ODD_PARITY.items():
        _index, case = gate.mutation_case_for(name)
        phases = {axis.upper(): int(phase) for axis, phase in case[3]}
        for axis in axes:
            assert phases.get(axis.upper()) == -1, (name, case[0])


def test_the_guard_REFUSES_a_pairing_that_does_not_enter_the_branch():
    """The dead-branch trap, exercised: a mutation re-pointed at a grid that does not
    carry what it rewrites must RAISE rather than run."""
    gate = load_gate()
    original = dict(gate.MUTATION_CASE)
    try:
        gate.MUTATION_CASE["m17_far_reflect_row_is_n_minus_two"] = \
            "x_fold_periodic_even"
        with pytest.raises(AssertionError, match="even full count"):
            gate.mutation_case_for("m17_far_reflect_row_is_n_minus_two")
        gate.MUTATION_CASE["mp2_first_two_arms_swapped"] = "one_pole_y_fold"
        with pytest.raises(AssertionError, match="DEAD slot"):
            gate.mutation_case_for("mp2_first_two_arms_swapped")
        gate.MUTATION_CASE["m1_parity_dropped"] = "x_fold_periodic_even"
        with pytest.raises(AssertionError, match="DEAD BRANCH|ODD"):
            gate.mutation_case_for("m1_parity_dropped")
    finally:
        gate.MUTATION_CASE.clear()
        gate.MUTATION_CASE.update(original)


def test_the_case_table_reaches_both_mirror_codes_and_three_dimensions():
    """A table that stopped at folded METALLIC 2-D cases would release a far carry and
    a whole z half that no leg executed."""
    gate = load_gate()
    periodic = [case for case in gate.CASES
                if any(case[2][axis.lower()] == "periodic"
                       for axis, _phase in case[3])]
    metallic = [case for case in gate.CASES
                if any(case[2][axis.lower()] == "metallic"
                       for axis, _phase in case[3])]
    three_d = [case for case in gate.CASES if float(case[1][2]) > 0.0]
    assert periodic and metallic and three_d
    # ...and the pole sets sweep one, two-asymmetric and two-everywhere.
    counts = {gate._pole_counts(case[4]) for case in gate.CASES}
    assert (2, 1, 1) in counts and (2, 2, 2) in counts and (1, 0, 1) in counts


def test_the_carry_family_and_its_null_control_cover_both_mirror_codes():
    """A bracket that changes nothing is not load-bearing, so every carry case is run
    twice — and the unbracketed twin must diverge."""
    gate = load_gate()
    assert gate.CARRY_CASES
    text = GATE.read_text(encoding="utf-8")
    assert "bracket=False" in text
    assert "require_identical=False" in text
    for name in gate.CARRY_CASES:
        assert gate.CASES[gate.case_index(name)]
    # A null control that agreed must FAIL, and one that ran no fused launch must
    # fail too — a leg that stops early still has to have measured the fused path.
    assert gate.verdict_of({"first_divergence": None, "fused_kernel_launches": 3},
                           require_identical=False,
                           require_launches_at_least=1, require_moved=False)[0] is False
    assert gate.verdict_of({"first_divergence": {"array": "Ex"},
                            "fused_kernel_launches": 0},
                           require_identical=False,
                           require_launches_at_least=1, require_moved=False)[0] is False
    assert gate.verdict_of({"first_divergence": {"array": "Ex"},
                            "fused_kernel_launches": 1},
                           require_identical=False,
                           require_launches_at_least=1, require_moved=False)[0] is True


def test_the_gates_private_scratch_rule_is_the_leading_underscore():
    """``_fmp_scratch`` is allocated by whichever route calls
    ``Fields.displacement_minus_polarization`` — the ARRAY path always, the fused one
    only on a carry leg — so its presence differs by route while it carries no state
    either route reads across a step. It is out of the comparison for that reason and
    the rule is the underscore, not the name; this is the tripwire on the rule."""
    gate = load_gate()
    assert gate.PRIVATE_SCRATCH == ("_fmp_scratch",)
    source = GATE.read_text(encoding="utf-8")
    assert 'if name.startswith("_"):' in source
    assert "private_scratch_on_the_array_route" in source
    # ...and a PUBLIC asymmetry still fails the leg.
    assert gate.verdict_of({"inventory_asymmetry_vs_array": ["Ex"]})[0] is False


def test_the_gate_pins_the_files_this_product_actually_rests_on():
    gate = load_gate()
    pinned = set(gate.source_hashes())
    for name in ("meep_gpu/triton_kernels/folded_dispersive_fused_pair.py",
                 "meep_gpu/triton_kernels/folded_fused_pair.py",
                 "meep_gpu/triton_kernels/folded_dispersive_update_e.py",
                 "meep_gpu/triton_kernels/dispersive_update_e.py",
                 "meep_gpu/deposit_repair.py",
                 "meep_gpu/test_triton_folded_dispersive_fused_pair.py"):
        assert name in pinned, name


def test_the_gate_has_not_been_run_or_names_the_run_it_took():
    """DEVICE STATUS, read off the gate rather than believed.

    While the docstring says UNRUN this asserts exactly that; when an artifact lands
    the docstring must name it, and this is the seam where a stale "UNRUN" fails.
    """
    text = GATE.read_text(encoding="utf-8")
    header = text.split('"""')[1]
    assert "DEVICE STATUS" in header
    if "UNRUN" in header:
        assert "may cite it as a release" in header
    else:
        assert "results/" in header
