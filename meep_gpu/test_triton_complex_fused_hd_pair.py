"""Laptop contracts for the Triton COMPLEX H->D weld: complex ``update_H`` welded
into complex ``step_D``.

WHAT THIS SUITE IS FOR. The device gate
(``parity/meep_gpu/gate_triton_complex_fused_hd_pair.py``) is what certifies the bytes;
this suite is the merge bar, and it holds the things that would otherwise only be true
on a machine nobody runs on the way in:

* the LIFT -- both halves are the certified COMPLEX kernels' own text plus the declared
  edits, and every parser that establishes that RAISES on a changed spelling rather
  than silently redirecting a tap. Each parser is exercised against a PLANTED change,
  so "it would raise" is measured rather than claimed;
* the ARITHMETIC FACTS the complex path turns on, as properties of the shipped text:
  which multiply helpers this kernel may reach, and the divide-free count with a
  planted divide that must move it;
* the predicate's refusals BY NAME, including the fold refusal that names the two
  folded-complex cells this product deliberately does not widen into;
* the CELL, joined off the board artifact rather than transcribed -- 17 rows, none
  carrying a standing in-seam withdraw, and the folded neighbour's DIFFERENT
  ``update_H`` arm, which is what makes "not a second variant of this weld on this
  backend" a measurement instead of a sentence;
* the declarations that DESCRIBE the wiring rather than request it, read off the tree
  so that a later round which lands the missing label finds this suite red instead of
  finding a stale sentence.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pathlib

import numpy
import pytest

from .fields import Fields
from .grid import Grid
from .pml import PML

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "triton_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
RESULTS = PARITY / "results"
MODULE = "meep_gpu.triton_kernels.complex_fused_hd_pair"
FAMILY = "complex_fused_hd_pair"

#: The board this product's cell is priced on, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows a predicate must refuse.
BOARD = RESULTS / "fusion_matrix_triton_2026-09-07_cyl" / "fusion_matrix.json"
SEAM_RECORD = RESULTS / "h_to_d_seam_2026-09-04" / "h_to_d_seam.jsonl"

#: The cell, and the two numbers the module's docstring states. Asserted against the
#: artifact rather than trusted: a board re-cut that moved either one must fail here.
CELL_ARMS = ("complex", "complex PML")
CELL_ROWS = 17
CELL_ROWS_WITH_A_STANDING_WITHDRAW = 0

#: The folded-complex neighbour, which on THIS backend is not a variant of this weld.
#: Two cells, not one, and neither shares this product's ``update_H`` arm.
FOLDED_NEIGHBOUR_CELLS = {
    ("folded complex", "folded complex PML"): 2,
    ("folded complex", "folded complex beta PML"): 3,
}


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def certified():
    return importlib.import_module("meep_gpu.triton_kernels.complex_fields")


# ---------------------------------------------------------------------------
# Fixtures -- real Grid/Fields/PML on NumPy, complex storage
# ---------------------------------------------------------------------------

def build(thickness=0.2, boundaries=None, symmetry=(), complex_storage=True,
          sigma=None):
    from .grid import Mirror

    grid = Grid(resolution=10.0, cell_size=(1.2, 1.2, 0.0), dimensions=2,
                boundaries=boundaries or {"x": "metallic", "y": "metallic",
                                          "z": "periodic"},
                symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if sigma is not None:
        fields.set_d_conductivity(
            numpy.full(grid.shape, float(sigma), dtype=numpy.float32))
    if thickness:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    faces = {}
    if thickness:
        for axis, letter in enumerate("xyz"[:2]):
            faces[letter] = ({"high": thickness} if grid.is_mirrored(axis)
                             else thickness)
    pml = PML(grid=grid, thickness=faces or 0)
    return fields, pml


#: The two clauses every predicate on a NumPy host fires for reasons that have nothing
#: to do with any configuration under test. Subtracted BY NAME rather than by position,
#: so a clause that stopped firing does not quietly take another with it.
HOST_CLAUSES = ("not cupy", "expansion probe artifact")


def _reasons(verdict):
    return [reason for reason in verdict.reasons
            if not any(clause in reason for clause in HOST_CLAUSES)]


class _Electric:
    """A source with a standing INTEGRATED electric withdraw."""

    from . import sources as _sources_module

    withdraw = _sources_module.VolumeSource.withdraw
    field_type = "D"
    component = "Ez"
    is_integrated = True
    _n_source_points = 3
    _applied_dipole = 0j


class _NonIntegratedElectric(_Electric):
    is_integrated = False


class _Magnetic(_Electric):
    field_type = "B"
    component = "Hz"


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_the_product_declares_the_seam_it_replaces(product):
    """``REPLACES`` names the driver call sites ONE launch performs, in order."""
    assert product.REPLACES == ("update_H", "step_D")
    assert product.SLOT == "update_H"
    assert product.CONSTITUTIVE_SIDE == "H"
    assert product.CURL_SUB_STEP == "step_D"
    assert product.ARMS == CELL_ARMS
    assert product.BACKWARD == 1
    assert product.LAUNCHES_PER_RUN == 1
    assert product.ROTATED == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    assert product.IN_PLACE == ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")
    assert product.CONSTITUTIVE_SOURCES == ("Bx", "By", "Bz")


def test_the_volume_lists_are_the_real_twins_rather_than_a_second_spelling(product):
    """The storage dtype does not change which volumes the seam touches.

    Re-spelling the three tuples would be a second list to keep in step with the
    engine's attribute names, and the failure mode of a drifted copy here is a launch
    bound to the wrong volume rather than an error.
    """
    from .triton_kernels import fused_hd_pair as real

    assert product.ROTATED is real.ROTATED
    assert product.IN_PLACE is real.IN_PLACE
    assert product.CONSTITUTIVE_SOURCES is real.CONSTITUTIVE_SOURCES


def test_the_volume_lists_are_the_shipped_tables_own(product):
    """...and those tables are the coverage module's, not a fourth copy."""
    from .triton_kernels import coverage as tcoverage

    side = tcoverage.CONSTITUTIVE_SIDES[product.CONSTITUTIVE_SIDE]
    assert product.ROTATED == tuple(side["targets"]) + tuple(side["aux"])
    assert product.CONSTITUTIVE_SOURCES == tuple(side["sources"])


def test_replaces_is_the_drivers_own_two_adjacent_consults(product):
    """The two names are consults the driver makes back to back, in that order."""
    from . import driver as driver_module

    source = inspect.getsource(driver_module.FdtdDriver.step)
    order = [line.split('fast.dispatch("', 1)[1].split('"', 1)[0]
             for line in source.splitlines() if 'fast.dispatch("' in line]
    first = order.index("update_H")
    assert order[first:first + 2] == list(product.REPLACES)


def test_the_only_statement_in_the_seam_is_the_electric_withdraw(product):
    """Nothing is INJECTED between the two consults, so the deposit repair is a fact."""
    from . import driver as driver_module

    source = inspect.getsource(driver_module.FdtdDriver.step)
    between = source.split('fast.dispatch("update_H"', 1)[1]
    between = between.split('fast.dispatch("step_D"', 1)[0]
    statements = [line.strip() for line in between.splitlines()
                  if line.strip() and not line.strip().startswith("#")
                  and "update_H(self.fields" not in line]
    assert any("withdraw" in line for line in statements), statements
    assert not any("inject" in line for line in statements), statements
    assert product.CARRIES_DEPOSIT_REPAIR is False
    assert product.REPAIR_PATHS == ()


def test_the_seam_name_is_the_withdraw_hoists_and_the_span_is_its_own(product):
    from . import withdraw_hoist

    assert product.SEAM == withdraw_hoist.SEAM
    assert product.REPLACES == withdraw_hoist.SEAM_SPAN


def test_the_two_flags_and_the_wiring_move_together(product):
    """``HOISTS_THE_WITHDRAW`` False is entailed by ``INSTALLABLE`` False."""
    assert product.INSTALLABLE is False
    assert product.HOISTS_THE_WITHDRAW is False
    reason = product.INSTALLABLE_REASON
    assert "4 - (installed pairs)" in reason
    assert "arbitration_over_the_driven_rows" in reason
    assert "GAIN where NEITHER does" in reason
    # THE THIRD FACT IS THIS ARM'S OWN and must be stated: under `flush` no complex arm
    # is selected at all, so the arbitration question does not even arise there.
    assert "'keep'" in reason and "flush" in reason
    # AND THE REASON NAMES THE ARRAY PATH AS THE GATE'S REFERENCE while unwired, which
    # is the honest reading of an unregistered product's evidence.
    assert "AWAITS WIRING" in reason


def test_the_default_block_is_the_certified_modules_own(product, certified):
    assert product.DEFAULT_BLOCK == certified.DEFAULT_BLOCK


def test_the_expansion_facts_are_restated_from_the_certified_module(product, certified):
    """The pattern set and the policy are the complex tranche's, not this product's."""
    assert product.PROBE_PATTERNS == certified.PROBE_PATTERNS
    assert product.CERTIFIED_UNDER_SUBNORMAL_POLICY == \
        certified.CERTIFIED_UNDER_SUBNORMAL_POLICY
    assert product.CERTIFIED_UNDER_SUBNORMAL_POLICY == "keep"


def test_backward_and_the_sub_lattice_are_read_from_the_shipped_tables(product):
    """The shared ``kms`` group is only correct while the two halves agree."""
    curl, constitutive = product._sub_lattice_suffixes()
    assert curl == constitutive == ""


# ---------------------------------------------------------------------------
# The lift -- the constitutive half
# ---------------------------------------------------------------------------

def test_the_constitutive_half_is_the_certified_complex_body(product):
    """A LIFT, not a transcription: the two cuts are the SAME STRING."""
    assert product.certified_constitutive_tail() == product.lifted_constitutive_tail()


def test_every_constitutive_lift_edit_is_declared_with_its_reason(product):
    edits = product.CONSTITUTIVE_LIFT_EDITS
    assert len(edits) == 8
    for edit in edits:
        assert set(edit) == {"line", "became", "why"}
        assert edit["why"].strip()
    # NO ARITHMETIC IN THE TABLE. Every entry is a pointer spelling, a resolved
    # constexpr branch, a store that moves, or the decode. An edit that touched an
    # operand would be a transcription wearing a lift's name.
    joined = " ".join(edit["line"] + edit["became"] for edit in edits)
    for arithmetic in ("_mul_coefficient_left(kp", "a_re + t_re", "a_re - t_re"):
        assert arithmetic not in joined


def test_the_lifted_constitutive_carries_no_store_and_no_scale_arm(product):
    tail = product.lifted_constitutive_tail()
    assert "tl.store(" not in tail
    assert "SCALE" not in tail
    # ...and it reads only the PRE-LAUNCH volumes, never the curl's own pointers.
    for stem in ("f0 +", "f1 +", "f2 +", "w0 +", "w1 +", "w2 +",
                 "g0 +", "g1 +", "g2 +", "e0 +", "e1 +", "e2 +"):
        assert stem not in tail


def test_the_constitutive_lift_raises_when_a_needle_stops_matching(product, tmp_path):
    """The lift RAISES on a drifted spelling rather than emitting a stale read.

    ARMED, not asserted: the certified module is copied with one load respelled and
    the cut is required to refuse it.
    """
    source = (PACKAGE_DIR / "complex_fields.py").read_text(encoding="utf-8")
    planted = tmp_path / "complex_fields.py"
    planted.write_text(
        source.replace("prev_re = tl.load(w0 + 2 * idx",
                       "prev_re = tl.load(w0 + idx * 2", 1), encoding="utf-8")
    tail = product._real._cut(
        product._real._source_of("bloch_constitutive_step", planted),
        product.DECODE_END)
    with pytest.raises(AssertionError, match="matches 0 times"):
        product._real.needle(tail, "prev_re = tl.load(w0 + 2 * idx",
                             "prev_re = tl.load(wi0 + 2 * idx")


# ---------------------------------------------------------------------------
# The lift -- the curl half
# ---------------------------------------------------------------------------

def test_the_curl_half_is_the_certified_complex_body_with_the_reads_redirected(product):
    assert product.certified_curl_tail() == product.lifted_curl_tail()


def test_the_curl_lift_is_twelve_generated_edits(product):
    """Six own-cell word loads and six shifted PAIRS. Nothing else moves."""
    edits = product.curl_lift_edits()
    assert len(edits) == 12
    assert len(product.OWN_LOAD_EDITS) == 6
    assert len(product.HALO_TAPS) == 6
    for old, new in edits:
        assert old != new
        assert "tl.load(g" in old


def test_every_halo_tap_is_parsed_from_the_emitters_own_lines(product):
    """WHICH cell and WHICH guard come from the certified text, not from this file."""
    tail = product._real._cut(
        product.function_source("bloch_pml_curl_step"), product.DECODE_END)
    taps = product.halo_taps(tail)
    offsets = product.offset_coordinates(tail)
    assert set(taps) == {stem for stem, _ in product.HALO_TAPS}
    assert set(offsets) == {"ox", "oy", "oz"}
    # The two word planes of each operand agree on component, cell and guard -- which
    # is what makes the pair ONE complex value rather than two words.
    for stem, planes in taps.items():
        assert planes["re"] == planes["im"]
    assert {stem: planes["re"][1] for stem, planes in taps.items()} == {
        "a_y": "oy", "a_z": "oz", "b_x": "ox", "b_z": "oz",
        "c_x": "ox", "c_y": "oy"}
    assert {stem: planes["re"][2] for stem, planes in taps.items()} == {
        "a_y": "vy", "a_z": "vz", "b_x": "vx", "b_z": "vz",
        "c_x": "vx", "c_y": "vy"}


def test_the_tap_parser_refuses_a_pair_whose_planes_disagree(product):
    """ARMED. A pair that read two different cells is not one complex operand."""
    tail = product._real._cut(
        product.function_source("bloch_pml_curl_step"), product.DECODE_END)
    planted = tail.replace("a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)",
                           "a_y_im = tl.load(g0 + 2 * oz + 1, mask=vy, other=0.0)", 1)
    assert planted != tail
    with pytest.raises(AssertionError, match="not one complex operand"):
        product.halo_taps(planted)


def test_the_tap_parser_refuses_a_changed_word_pair_spelling(product):
    """ARMED. A load that stopped being `gN + 2 * offset[ + 1]` is a changed layout."""
    tail = product._real._cut(
        product.function_source("bloch_pml_curl_step"), product.DECODE_END)
    planted = tail.replace("b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)",
                           "b_x_re = tl.load(g1 + ox * 2, mask=vx, other=0.0)", 1)
    assert planted != tail
    with pytest.raises(AssertionError, match="word-pair addressing"):
        product.halo_taps(planted)


def test_the_tap_parser_refuses_a_tap_that_appeared_or_vanished(product):
    """ARMED. The declared six are the emitter's six, in both directions."""
    tail = product._real._cut(
        product.function_source("bloch_pml_curl_step"), product.DECODE_END)
    planted = tail.replace(
        "c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)\n"
        "c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)\n", "", 1)
    assert planted != tail
    with pytest.raises(AssertionError, match="shifted magnetic operands"):
        product.halo_taps(planted)


def test_the_welded_curl_reads_no_magnetic_pointer(product):
    tail = product.lifted_curl_tail()
    for stem in ("g0 +", "g1 +", "g2 +"):
        assert stem not in tail
    # ...and every foreign read goes through the ONE recompute helper.
    assert tail.count("_h_tap_complex(") == 6


def test_the_bloch_rotation_is_left_standing_and_is_outside_the_recompute(product):
    """The phase is applied to the LOADED register, on the wrapped lane only.

    A weld that rotated inside the recompute would rotate every unwrapped lane too,
    which is a smooth wrong answer rather than an error.
    """
    curl = product.lifted_curl_tail()
    assert curl.count("_rotate_field_left(") == 6
    assert curl.count("tl.where(w") == 12
    tap = product.function_source("_h_tap_complex", pathlib.Path(product.__file__))
    cell = product.function_source("_h_cell_complex", pathlib.Path(product.__file__))
    assert "_rotate_field_left" not in tap
    assert "_rotate_field_left" not in cell


def test_the_tap_serves_an_exact_zero_where_the_certified_load_masked_off(product):
    """``other=0.0`` on BOTH planes is what the metallic ghost is."""
    tap = product.function_source("_h_tap_complex", pathlib.Path(product.__file__))
    assert "return tl.where(valid, v_re, 0.0), tl.where(valid, v_im, 0.0)" in tap


# ---------------------------------------------------------------------------
# The arithmetic facts, as properties of the shipped text
# ---------------------------------------------------------------------------

def test_this_kernel_reaches_only_the_three_certified_multiply_helpers(product):
    """The pattern set is a claim about the text, so the text is scanned.

    A fourth orientation would need its own probe pattern; a kernel that grew one
    silently would be licensed by an artifact that never measured it.
    """
    called = set()
    for name in product.DEVICE_FUNCTIONS:
        tree = ast.parse(product.function_source(
            name, pathlib.Path(product.__file__)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            leaf = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if leaf and (leaf.startswith("_mul_") or leaf.startswith("_rotate_")
                         or leaf.startswith("_div_")):
                called.add(leaf)
    # EQUALITY IN BOTH DIRECTIONS: a fourth helper fails here, and so does a
    # declaration that names a helper the kernel stopped reaching.
    assert called == set(product.MULTIPLY_HELPERS)
    assert called == {"_rotate_field_left", "_mul_field_left",
                      "_mul_coefficient_left"}


def test_the_constitutive_half_launches_only_the_coefficient_left_product(product):
    """The ``SCALE`` arm that would bring ``inv_eps`` in belongs to the E side."""
    cell = product.function_source("_h_cell_complex", pathlib.Path(product.__file__))
    assert "_mul_coefficient_left(" in cell
    assert "_mul_field_left(" not in cell
    assert "inv_eps" not in cell and "SCALE" not in cell


def test_this_seam_divides_nowhere(product):
    evidence = product.divide_free_evidence()
    assert evidence["divide_free"] is True
    assert evidence["true_divisions"] == 0
    assert evidence["div_rn_calls"] == 0
    assert evidence["reciprocal_calls"] == 0
    # The floor divisions ARE the certified index decode, one per decoding function.
    assert evidence["floor_divisions"] == 2


def test_the_divide_free_count_bites_on_a_planted_divide(product, tmp_path):
    """THE NULL CONTROL. A zero nobody has shown can move is worth nothing."""
    source = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    planted = tmp_path / "planted.py"
    planted.write_text(
        source.replace("v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)",
                       "v0_re, v0_im = (r_re / si_z, r_im / si_z)", 1),
        encoding="utf-8")
    evidence = product.divide_free_evidence(planted)
    assert evidence["divide_free"] is False
    assert evidence["true_divisions"] == 2


def test_the_divide_free_count_bites_on_a_planted_div_rn(product, tmp_path):
    """...and on the correctly rounded spelling too, which a text scan for `/` misses."""
    source = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    planted = tmp_path / "planted_div_rn.py"
    planted.write_text(
        source.replace("si_z = tl.load(sinvz + k, mask=live, other=0.0)",
                       "si_z = tl.math.div_rn(1.0, tl.load(sinvz + k, mask=live, "
                       "other=0.0))", 1),
        encoding="utf-8")
    evidence = product.divide_free_evidence(planted)
    assert evidence["divide_free"] is False
    assert evidence["div_rn_calls"] == 1


def test_the_divide_facts_are_recorded_with_where_each_is_true(product):
    """The Dcyl round's measured spellings are carried as inapplicable, not repeated."""
    facts = product.DIVIDE_FACTS
    assert len(facts) == 4
    for fact in facts:
        assert set(fact) == {"claim", "where_it_is_true", "why_it_is_moot_here"}
        assert fact["claim"].strip() and fact["where_it_is_true"].strip()
    joined = " ".join(fact["claim"] for fact in facts)
    assert "arithmetic.h:96-110" in joined      # CuPy's scaled algorithm
    assert "div.full.f32" in joined             # Triton's approximate divide
    assert "Metal complex verdict" in joined    # the sibling backend's prescription


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_the_predicate_refuses_fields_with_no_grid(product):
    class _Bare:
        grid = None

    verdict = product.complex_fused_hd_pair_coverage(_Bare(), None, ())
    assert verdict.covered is False
    assert verdict.reasons == ("fields carries no grid",)


def test_an_undeclared_source_list_is_a_refusal_not_an_empty_set(product):
    fields, pml = build()
    reasons = _reasons(product.complex_fused_hd_pair_coverage(fields, pml, None))
    assert any("source set was not declared" in reason for reason in reasons), reasons


def test_a_standing_integrated_electric_withdraw_is_refused_by_name(product):
    fields, pml = build()
    reasons = _reasons(
        product.complex_fused_hd_pair_coverage(fields, pml, (_Electric(),)))
    assert any("standing integrated" in reason and "3313-3314" in reason
               for reason in reasons), reasons


def test_a_non_integrated_electric_source_is_not_this_seams_business(product):
    fields, pml = build()
    reasons = _reasons(product.complex_fused_hd_pair_coverage(
        fields, pml, (_NonIntegratedElectric(),)))
    assert not any("standing integrated" in reason for reason in reasons), reasons


def test_a_magnetic_source_is_not_this_seams_business(product):
    """The magnetic injection is one seam EARLIER (driver.py:3283-3284)."""
    fields, pml = build()
    reasons = _reasons(
        product.complex_fused_hd_pair_coverage(fields, pml, (_Magnetic(),)))
    assert not any("standing integrated" in reason for reason in reasons), reasons


def test_a_folded_grid_is_refused_and_the_refusal_names_the_folded_complex_arms(product):
    fields, pml = build(symmetry=(("X", 1),))
    reasons = product.complex_fused_hd_pair_coverage(fields, pml, ()).reasons
    fold = [reason for reason in reasons if "is folded" in reason]
    assert fold, reasons
    # BOTH halves already refuse a mirror plane, and this product RE-CHECKS rather
    # than reading its coverage off another module's guard -- so the list carries the
    # halves' own reasons AND this one, which is the only one that names the cells.
    own = [reason for reason in fold if "`folded complex` arms" in reason]
    assert own, fold
    assert "their own ghost-map measurement" in own[0]


def test_real_storage_is_refused_by_the_two_complex_halves(product):
    fields, pml = build(complex_storage=False)
    reasons = product.complex_fused_hd_pair_coverage(fields, pml, ()).reasons
    assert any(reason.startswith("complex constitutive half:") for reason in reasons)
    assert any(reason.startswith("complex curl half:") for reason in reasons)


def test_the_expansion_probe_clause_is_inherited_rather_than_restated(product):
    """An absent probe is a REFUSAL, and it arrives from the halves' own predicates."""
    fields, pml = build()
    reasons = product.complex_fused_hd_pair_coverage(fields, pml, ()).reasons
    probe = [reason for reason in reasons if "expansion probe artifact" in reason]
    assert len(probe) == 2, probe
    assert probe[0].startswith("complex constitutive half:")
    assert probe[1].startswith("complex curl half:")


def test_a_missing_rotating_volume_is_refused_by_name(product):
    fields, pml = build()
    fields.Hy = None
    reasons = product.complex_fused_hd_pair_coverage(fields, pml, ()).reasons
    assert any("Hy is not allocated" in reason and "rotates it" in reason
               for reason in reasons), reasons


def test_the_refusals_are_ordered_constitutive_first(product):
    """The driver reaches ``update_H`` first, so a reader sees that half's reason first."""
    fields, pml = build()
    reasons = product.complex_fused_hd_pair_coverage(fields, pml, ()).reasons
    first_curl = next(index for index, reason in enumerate(reasons)
                      if reason.startswith("complex curl half:"))
    first_constitutive = next(index for index, reason in enumerate(reasons)
                              if reason.startswith("complex constitutive half:"))
    assert first_constitutive < first_curl


def test_the_plan_builder_refuses_rather_than_raising_on_this_host(product):
    fields, pml = build()
    assert product.plan_complex_fused_hd_pair(fields, pml, ()) is None


def test_explain_is_the_same_verdict_under_the_name_a_report_reads(product):
    fields, pml = build()
    assert (product.explain_complex_fused_hd_pair(fields, pml, ())
            == product.complex_fused_hd_pair_coverage(fields, pml, ()))


# ---------------------------------------------------------------------------
# The absent backend
# ---------------------------------------------------------------------------

def test_the_module_imports_without_triton(product):
    """The merge bar is a laptop, and every host clause above ran on one."""
    assert product.FAMILY == FAMILY


def test_the_kernel_accessor_names_the_missing_import(product):
    if product.triton is not None:  # pragma: no cover - a CUDA host
        pytest.skip("Triton is importable here; the refusal branch is the laptop's")
    with pytest.raises(ImportError, match="needs Triton to launch"):
        product.fused_complex_constitutive_curl_H_to_D_kernel()


# ---------------------------------------------------------------------------
# The cell, joined off the artifact
# ---------------------------------------------------------------------------

def _board():
    if not BOARD.is_file():
        pytest.skip(f"{BOARD} is not in this checkout")
    return json.loads(BOARD.read_text(encoding="utf-8"))


def test_the_cell_is_seventeen_rows_and_none_of_them_carries_a_withdraw(product):
    """The two numbers the module's docstring states, READ OFF THE BOARD.

    A board re-cut that moved either number must fail here rather than leave a
    docstring quietly wrong about what this predicate reaches.
    """
    instances = _board()["aggregate"]["h_to_d_seam"]["instances"]
    mine = [row for row in instances
            if (row["update_H"], row["step_D"]) == tuple(product.ARMS)]
    assert len(mine) == CELL_ROWS
    assert all(row["bucket"] == "buildable_not_built" for row in mine)
    withdrawing = [row for row in mine if row["withdraw_in_seam"]]
    assert len(withdrawing) == CELL_ROWS_WITH_A_STANDING_WITHDRAW, withdrawing
    assert all(row["integrated_electric_sources"] == 0 for row in mine)
    assert all(row["served_by"] is None for row in mine)


def test_the_seam_record_agrees_with_the_board_about_this_cell(product):
    """TWO WITNESSES. The board is a join; the seam record is what it joined."""
    if not SEAM_RECORD.is_file():
        pytest.skip(f"{SEAM_RECORD} is not in this checkout")
    rows = [json.loads(line) for line in
            SEAM_RECORD.read_text(encoding="utf-8").splitlines() if line.strip()]
    mine = [row for row in rows
            if ((row.get("arms") or {}).get("triton") or {}).get("update_H")
            == product.ARMS[0]
            and ((row.get("arms") or {}).get("triton") or {}).get("step_D")
            == product.ARMS[1]]
    assert len(mine) == CELL_ROWS
    assert not [row for row in mine
                if (row.get("h_to_d_seam") or {}).get("withdraw_in_seam")]


def test_the_folded_complex_neighbour_is_not_a_variant_of_this_weld_here():
    """THE FINDING, AS A MEASUREMENT. Two cells, not one, and a different H arm.

    ``cuda_kernels/complex_fused_hd_pair`` covers its folded neighbour as a second
    variant of one weld because on that backend the folded rows select the PLAIN
    complex arm at ``update_H``. On this backend they do not: the census records
    ``folded complex``, a distinct predicate, and the folded rows are split across
    two step_D arms rather than one. If a later census changes either fact, the
    "second product, not a widening" ruling should be revisited -- and this is what
    makes that happen.
    """
    instances = _board()["aggregate"]["h_to_d_seam"]["instances"]
    counted = {}
    for row in instances:
        key = (row["update_H"], row["step_D"])
        if key[0].startswith("folded complex"):
            counted[key] = counted.get(key, 0) + 1
    for cell, rows in FOLDED_NEIGHBOUR_CELLS.items():
        assert counted.get(cell) == rows, counted
    assert all(cell[0] != CELL_ARMS[0] for cell in counted), counted


def test_the_folded_complex_constitutive_is_its_own_predicate_on_this_backend():
    """...and the arm label is not a relabelling of the same coverage function."""
    from .triton_kernels import complex_fields, folded_complex

    assert (folded_complex.folded_complex_constitutive_coverage
            is not complex_fields.complex_constitutive_coverage)
    # It launches the SAME certified kernel, which is exactly why the CUDA sibling
    # could share a half and this one still cannot share a cell.
    doc = folded_complex.folded_complex_constitutive_coverage.__doc__ or ""
    assert "bloch_constitutive_step" in doc


# ---------------------------------------------------------------------------
# Not wired, read off the tree
# ---------------------------------------------------------------------------

#: The label the composer writes for this product once it is routed. One spelling,
#: read by both tests below and by the wiring itself.
LABEL = "fused pair H->D (complex)"


def test_the_composer_carries_this_product_and_refuses_it_by_name(product):
    """ROUTED, NOT INSTALLED. The seam loop is offered this product and refuses it.

    THIS TEST WAS THE OPPOSITE ASSERTION UNTIL THE WIRING LANDED, and that is the
    point of writing it that way: the unwired version asserted the composer named this
    module NOWHERE, so the round that routed it found this file red instead of finding
    a stale sentence. What replaces it is the stronger claim -- the product has a row,
    an arm pair and a label, and ``_declared_uninstallable`` refuses it on every
    configuration -- because a routed product whose refusal stopped firing would be
    INSTALLED, which no composition gate has driven.
    """
    from .triton_kernels import launch as launch_module

    assert FAMILY in launch_module.CERTIFIED_FUSED_PRODUCTS
    row = launch_module.CERTIFIED_FUSED_PRODUCTS[FAMILY]
    assert row["curl_slot"] == product.SLOT
    assert row["module"] == FAMILY
    assert row["label"] == LABEL
    assert launch_module.CERTIFIED_FUSED_PAIR_ARMS[FAMILY] == product.ARMS
    assert FAMILY in launch_module.SUPPORT_MODULES
    refusal = launch_module._declared_uninstallable(  # noqa: SLF001
        {FAMILY: product}, FAMILY, row)
    assert refusal and "INSTALLABLE = False" in refusal


def test_the_dispatcher_carries_the_label_and_leaves_it_unreleased(product):
    """The label is in the see-through and in the pending list, and in NO envelope.

    A label ``plan_step`` can write must also be declared in the dispatcher's own
    tables; a label in ``RELEASED_FUSED_ARMS`` would be dispatched. Both directions
    are asserted, so neither the wiring nor a later release can move silently.
    """
    from . import fastpath

    assert fastpath.FUSED_ARM_CONSTITUENTS[LABEL] == product.ARMS
    assert LABEL in fastpath.PENDING_DEVICE_GATE_ARMS
    assert LABEL not in fastpath.RELEASED_FUSED_ARMS
    assert "triton_complex_fused_hd_pair_2026-09-07" in \
        fastpath.PENDING_DEVICE_GATE_ARMS[LABEL]


def test_this_module_does_not_claim_the_planner_knows_about_it(product):
    """The module may not assert a present-tense fact about a file it does not own."""
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    for claim in ("is consulted by ``launch.plan_step``",
                  "``launch.plan_step`` installs this",
                  "RELEASED_FUSED_ARMS carries"):
        assert claim not in text


def test_the_module_states_the_reference_it_actually_has(product):
    """UNREGISTERED means the oracle is the array path, and the artifact must say so."""
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "composition-installed reference awaits wiring" in text.lower()
