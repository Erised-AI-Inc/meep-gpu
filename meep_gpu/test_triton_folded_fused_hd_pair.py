"""Laptop contracts for the Triton FOLDED H->D weld: folded ``update_H`` welded
into folded ``step_D``.

WHAT THIS SUITE IS FOR. The device gate
(``parity/meep_gpu/gate_triton_folded_fused_hd_pair.py``) is what certifies the bytes;
this suite is the merge bar, and it holds the things that would otherwise only be true
on a machine nobody runs on the way in:

* the LIFT -- the curl half is :func:`.symmetry.pml_curl_step_folded`'s own text plus
  the nine generated redirects, the constitutive half is IMPORTED from
  :mod:`.fused_hd_pair` rather than copied, and every parser that establishes either
  RAISES on a changed spelling instead of quietly redirecting a tap;
* THE CODES, which are this family's single point of failure. The plain product's
  ``[1 if kind == "metallic" else 0]`` mapping is a valid triple that is WRONG on a
  fold, and the two triples are measured DIFFERENT here rather than argued;
* the predicate's refusals BY NAME, and the DISJOINTNESS from :mod:`.fused_hd_pair`
  driven in both directions on the same fixtures rather than asserted;
* the declarations that DESCRIBE the wiring rather than request it;
* the composer's two brakes, DRIVEN against an injected row;
* the standing measurement this family's gate rests on -- the Metal sibling's
  whole-step mask attribution -- re-read from that artifact, so a claim in this
  module's docstring cannot outlive the record it cites.
"""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
import sys

import numpy
import pytest

from .fields import Fields
from .grid import Grid
from .pml import PML

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "triton_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
RESULTS = PARITY / "results"
MODULE = "meep_gpu.triton_kernels.folded_fused_hd_pair"
PLAIN_MODULE = "meep_gpu.triton_kernels.fused_hd_pair"
FAMILY = "folded_fused_hd_pair"

#: The board this family's cell is priced on, and the seam record the gate lifts from.
BOARD = "fusion_matrix_triton_2026-09-07_cyl"
SEAM_RECORD = "h_to_d_seam_2026-09-04"

#: The Metal sibling's artifact, whose whole-step mask attribution this family's
#: docstring cites as a MEASUREMENT rather than as a prediction.
METAL_ARTIFACT = "metal_folded_fused_hd_pair_2026-09-07_wired"


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def plain():
    return importlib.import_module(PLAIN_MODULE)


@pytest.fixture(scope="module")
def symmetry():
    return importlib.import_module("meep_gpu.triton_kernels.symmetry")


@pytest.fixture(scope="module")
def gate():
    for path in (str(PARITY), str(HERE.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("gate_triton_folded_fused_hd_pair")


# ---------------------------------------------------------------------------
# Fixtures -- real Grid/Fields/PML on NumPy, folded and unfolded
# ---------------------------------------------------------------------------

def build(symmetry=(("Y", 1),), boundaries=None, thickness=0.2,
          complex_storage=False, sigma=None, nonlinear=False, cell=(1.2, 1.2, 0.0)):
    """A real folded (or unfolded) 2-D grid on NumPy.

    THE ABSORBER GOES ON THE FAR FACE OF A FOLDED AXIS and on both faces of every
    other one, which is what a folded run actually carries: the mirror plane is a
    boundary MEEP does not absorb at. Spelling it here rather than passing a scalar
    is what makes ``folded_axis_kinds`` resolve at all -- a layer on the plane face
    is one of its named refusals.
    """
    from .grid import Mirror

    grid = Grid(resolution=10.0, cell_size=cell, dimensions=2,
                boundaries=boundaries or {"x": "metallic", "y": "metallic",
                                          "z": "periodic"},
                symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if nonlinear:
        shape = grid.shape
        fields.set_nonlinear_volumes(
            {"Ez": numpy.full(shape, 0.1, dtype=numpy.float32)},
            {"Ez": numpy.full(shape, 0.2, dtype=numpy.float32)})
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


def _reasons(verdict):
    """The refusal reasons MINUS the array-module clause.

    Every clause here is asked on a NumPy host, where the folded predicates' first
    clause always fires; subtracting it is what lets the other clauses be asserted at
    all. It is subtracted by NAME rather than by position, so a clause that stopped
    firing does not quietly take another with it.
    """
    return [reason for reason in verdict.reasons if "not cupy" not in reason]


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


class _GridView:
    """The real grid with one attribute overridden, for a refusal stand-in."""

    def __init__(self, grid, **overrides):
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_grid"), name)


class _FieldsView:
    def __init__(self, fields, grid):
        self._fields = fields
        self.grid = grid

    def __getattr__(self, name):
        return getattr(self._fields, name)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_the_product_declares_the_seam_it_replaces(product):
    """``REPLACES`` names the driver call sites ONE launch performs, in order."""
    assert product.REPLACES == ("update_H", "step_D")
    assert product.SLOT == "update_H"
    assert product.CONSTITUTIVE_SIDE == "H"
    assert product.CURL_SUB_STEP == "step_D"
    assert product.FAMILY == FAMILY


def test_the_arms_are_the_folded_pair_the_composer_writes(product):
    """``ARMS`` are the labels ``launch._select_slot`` writes on these two slots.

    READ OFF ``launch.py``'s own arm tables rather than spelled: an arm renamed there
    while this tuple stood would make every board join and every ledger row point at a
    cell that no longer exists.
    """
    text = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    assert product.ARMS == ("folded", "folded PML")
    assert '_Arm("folded", has_fold,' in text
    assert '_Arm("folded PML", has_fold,' in text


def test_replaces_is_the_drivers_own_two_adjacent_consults(product):
    """READ OFF ``driver.py``, so a driver move fails here rather than reading true."""
    text = (HERE / "driver.py").read_text(encoding="utf-8")
    body = text.split("    def step(self", 1)[1].split("\n    def ", 1)[0]
    order = [line.split('fast.dispatch("', 1)[1].split('"', 1)[0]
             for line in body.splitlines() if 'fast.dispatch("' in line]
    first = order.index(product.REPLACES[0])
    assert order[first + 1] == product.REPLACES[1], order


def test_the_only_statement_in_the_seam_is_the_electric_withdraw(product):
    """On a FOLD as much as off one: every fill closes before or opens after."""
    text = (HERE / "driver.py").read_text(encoding="utf-8")
    body = text.split("    def step(self", 1)[1].split("\n    def ", 1)[0]
    between = body.split('fast.dispatch("update_H"', 1)[1]
    between = between.split('fast.dispatch("step_D"', 1)[0]
    # THE TWO FRAGMENTS THE SPLIT LEAVES are the tail and the head of the two consult
    # lines themselves; they are dropped by name so what remains is statements.
    statements = [line.strip() for line in between.splitlines()
                  if line.strip() and not line.strip().startswith("#")
                  and "update_H(self.fields" not in line
                  and line.strip() not in (", self.fields):",
                                           "if fast is None or not")]
    assert statements, "nothing at all stands between the two consults"
    assert all("withdraw" in line or line.startswith("for source")
               for line in statements), statements
    # THE FILLS, BY NAME: none of the three folded-relevant passes is in the seam.
    for pass_name in ("fill_symmetry_bc_B", "fill_symmetry_bc_D", "zero_metal_B",
                      "zero_metal_D", "fill_folded_far_ghosts_B",
                      "fill_folded_far_ghosts_D"):
        assert pass_name not in "".join(statements), pass_name


def test_the_seam_name_is_the_withdraw_hoists_and_the_span_is_its_own(product):
    from . import withdraw_hoist

    assert product.SEAM == withdraw_hoist.SEAM
    assert product.REPLACES == withdraw_hoist.SEAM_SPAN


def test_the_three_flags_are_facts_or_measured_verdicts(product):
    """Each flag, with the fact or the verdict that makes it false."""
    from .triton_kernels import launch as triton_launch

    # A FACT ABOUT THE DRIVER: nothing is injected in this seam.
    assert product.CARRIES_DEPOSIT_REPAIR is False
    assert product.REPAIR_PATHS == ()
    # A CONSEQUENCE OF THE NEXT ONE: the hoist branch is unreachable while the
    # product is uninstallable, so declaring True would be a claim about wiring that
    # cannot fire.
    assert product.HOISTS_THE_WITHDRAW is False
    assert product.INSTALLABLE is False
    assert FAMILY not in triton_launch.CERTIFIED_FUSED_PRODUCTS
    assert FAMILY not in triton_launch.CERTIFIED_FUSED_PAIR_ARMS
    # THE REASON NAMES BOTH HALVES.
    reason = product.INSTALLABLE_REASON
    assert "label" in reason and "ARBITRATION" in reason
    assert "78" in reason


def test_every_non_ascii_line_is_the_lifted_bodys_own(product):
    """ASCII everywhere this module writes prose; the lifted body keeps its bytes.

    ``symmetry.pml_curl_step_folded``'s mask comments carry four em dashes, and this
    module's kernel body is that text CHARACTER FOR CHARACTER -- so an ASCII-only
    rule here would be a rule against the lift. The clause that is actually true is
    asserted instead: every non-ASCII line in this file is a line of the certified
    body it lifts.
    """
    source = (PACKAGE_DIR / "folded_fused_hd_pair.py").read_text(encoding="utf-8")
    certified = {line.strip() for line in product.certified_curl_tail().splitlines()}
    strays = [line.strip() for line in source.splitlines()
              if not line.isascii() and line.strip() not in certified]
    assert strays == [], strays
    assert not source.splitlines()[0].startswith(" ")


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def test_the_curl_half_is_the_folded_body_with_nine_redirected_reads(product):
    """The shipped kernel's curl body IS ``pml_curl_step_folded``'s, plus the edits."""
    certified = product.certified_curl_tail()
    lifted = product.lifted_curl_tail()
    assert certified == lifted, (len(certified), len(lifted))
    assert len(product.folded_curl_lift_edits()) == 9
    statements = [line for line in certified.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    assert len(statements) >= 40, len(statements)


def test_the_lift_cuts_the_folded_kernel_and_not_the_plain_one(product, plain):
    """The two certified bodies are DIFFERENT text, and this family cuts the fold's.

    A copy that cut ``kernels.pml_curl_step`` instead would compile, launch and
    converge, and would wrap on every folded axis. So the two raw tails are asserted
    unequal and the fold's own two deltas are asserted present in ours and absent in
    the plain one.
    """
    folded_raw = product.raw_folded_curl_tail()
    plain_raw = plain._cut(plain._source_of("pml_curl_step"), plain.DECODE_END)
    assert folded_raw != plain_raw
    # DELTA 1: the ghost branch is written the other way round.
    assert "if BCX == PERIODIC:" in folded_raw
    assert "if BCX == METALLIC:" in plain_raw
    # DELTA 2: the cell-0 mask is widened to "not periodic".
    assert "if BCY != PERIODIC:" in folded_raw
    assert "!= PERIODIC" not in plain_raw
    # DELTA 3: the top-plane mask exists only on the fold.
    assert "MIRROR_PERIODIC" in folded_raw
    assert "last_x, last_y, last_z" in folded_raw
    assert "MIRROR_PERIODIC" not in plain_raw
    # ...and all three survive into the welded text, which is where they matter.
    welded = product.certified_curl_tail()
    assert "if BCX == PERIODIC:" in welded
    assert "if BCY != PERIODIC:" in welded
    assert welded.count("MIRROR_PERIODIC") == folded_raw.count("MIRROR_PERIODIC")


def test_no_magnetic_pointer_survives_below_the_seam(product):
    """In this signature ``gN`` does not exist; every read is a register or a tap."""
    welded = product.certified_curl_tail()
    for stem in ("g0", "g1", "g2", "e0", "e1", "e2"):
        assert f"{stem} +" not in welded, stem
    assert welded.count("_h_tap(") == 6
    assert welded.count("own0") >= 1


def test_the_constitutive_half_is_imported_and_not_copied(product):
    """A second copy of the lifted body would be a second place for it to drift."""
    tree = ast.parse((PACKAGE_DIR / "folded_fused_hd_pair.py").read_text(
        encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)}
    assert "_h_cell" not in defined
    assert "_h_tap" not in defined
    source = (PACKAGE_DIR / "folded_fused_hd_pair.py").read_text(encoding="utf-8")
    assert "from .fused_hd_pair import _h_cell, _h_tap" in source


def test_the_tap_table_is_parsed_from_the_folded_emitters_own_lines(product, plain):
    """WHICH cell and WHICH guard each redirect uses come from the folded text."""
    tail = product.raw_folded_curl_tail()
    taps = plain.halo_taps(tail)
    offsets = plain.offset_coordinates(tail)
    assert set(taps) == {name for name, _ in plain.HALO_TAPS}
    assert {name: value[0] for name, value in taps.items()} == dict(plain.HALO_TAPS)
    assert set(offsets) == {"ox", "oy", "oz"}
    # The folded emitter's own guards, not the plain one's names.
    assert {value[2] for value in taps.values()} == {"vx", "vy", "vz"}
    # Each redirect carries the coordinates the emitter's own index line composed.
    for register, component in plain.HALO_TAPS:
        _component, offset, mask = taps[register]
        coordinates = ", ".join(offsets[offset])
        needle = f"{register} = _h_tap({component}, {coordinates}, {mask},"
        assert needle in product.certified_curl_tail(), needle


def test_the_lift_raises_on_a_changed_folded_spelling(product, plain):
    """A drifted anchor RAISES rather than emitting an unchecked kernel."""
    with pytest.raises(AssertionError):
        plain._cut(product.raw_folded_curl_tail(), "        i = plane // nope\n")
    with pytest.raises(AssertionError):
        plain._source_of("pml_curl_step_folded_that_does_not_exist",
                         product.SYMMETRY_PATH)
    with pytest.raises(AssertionError):
        plain.needle(product.raw_folded_curl_tail(), "curl0", "curl0")


def test_the_folded_decode_anchor_is_the_deeper_one(product, plain):
    """Eight spaces, not four: the folded kernel lives inside a guard block."""
    assert product.FOLDED_DECODE_END.startswith("        i = plane")
    assert plain.DECODE_END.startswith("    i = plane")
    assert product.FOLDED_DECODE_END != plain.DECODE_END


# ---------------------------------------------------------------------------
# The codes -- this family's single point of failure
# ---------------------------------------------------------------------------

def test_the_codes_come_from_folded_axis_kinds_and_the_plain_mapping_is_wrong(
        product, symmetry):
    """MEASURED, not argued: the two triples DIFFER on every folded fixture.

    ``plan_fused_hd_pair`` builds its triple as ``[1 if kind == "metallic" else 0]``
    over ``_boundary_kinds``. On a folded axis that reports ``"mirror"``, which the
    expression maps to 0 = PERIODIC: the ghost would wrap to the far plane and the
    cell-0 mask would not be emitted at all. Nothing in the kernel can catch it,
    because 0 is a valid code -- so the difference is measured here and the wrong
    mapping is armed as a gate mutation.
    """
    from .triton_kernels import coverage as tcoverage

    seen = 0
    for keywords in ({}, {"boundaries": {"x": "metallic", "y": "periodic",
                                         "z": "periodic"}},
                     {"symmetry": (("X", 1), ("Y", 1))}):
        fields, pml = build(**keywords)
        codes, reasons = symmetry.folded_axis_kinds(fields.grid, pml)
        assert codes is not None, reasons
        kinds = tcoverage._boundary_kinds(fields.grid, pml)
        plain_codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
        assert tuple(codes) != plain_codes, (codes, plain_codes, kinds)
        # ...and the difference is exactly the folded axes, each of which the plain
        # mapping calls PERIODIC.
        for axis, kind in enumerate(kinds):
            if kind == "mirror":
                assert codes[axis] in (symmetry.CODE_MIRROR_METALLIC,
                                       symmetry.CODE_MIRROR_PERIODIC)
                assert plain_codes[axis] == symmetry.CODE_PERIODIC
                seen += 1
    assert seen >= 4, seen


def test_both_folded_terminations_are_reachable_from_this_suites_fixtures(
        product, symmetry):
    """MIRROR_METALLIC and MIRROR_PERIODIC are different kernels; both are built."""
    metallic, pml_m = build()
    periodic, pml_p = build(boundaries={"x": "metallic", "y": "periodic",
                                        "z": "periodic"})
    assert symmetry.folded_axis_kinds(metallic.grid, pml_m)[0][1] == \
        symmetry.CODE_MIRROR_METALLIC
    assert symmetry.folded_axis_kinds(periodic.grid, pml_p)[0][1] == \
        symmetry.CODE_MIRROR_PERIODIC


def test_the_plan_refuses_a_code_outside_the_four_ghost_rules(product):
    """A floor, not the rule: 0 is a valid code and a wrong one is the mutation."""
    shape = (4, 5, 6)
    arrays, flat, holder = _bare_arrays(product, shape)
    with pytest.raises(ValueError):
        product.plan_folded_fused_hd_pair_from_arrays(arrays, flat, 0.35, (1, 4, 0),
                                                      holder)
    # ...and each of the four is accepted.
    for code in (0, 1, 2, 3):
        product.plan_folded_fused_hd_pair_from_arrays(arrays, flat, 0.35,
                                                      (code, 2, 0), holder)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_a_folded_pml_run_is_admitted(product):
    fields, pml = build()
    assert _reasons(product.folded_fused_hd_pair_coverage(fields, pml, ())) == []


def test_an_undeclared_source_list_is_refused_by_name(product):
    """Ignorance is never an empty set."""
    fields, pml = build()
    reasons = _reasons(product.folded_fused_hd_pair_coverage(fields, pml, None))
    assert any("was not declared" in reason for reason in reasons), reasons


def test_a_standing_integrated_electric_withdraw_is_refused_by_name(product):
    """The 3 of this cell's 78 rows the board files under ``withdraw_seam``."""
    fields, pml = build()
    reasons = _reasons(
        product.folded_fused_hd_pair_coverage(fields, pml, (_Electric(),)))
    assert any("standing integrated" in reason for reason in reasons), reasons
    assert any("HOISTS_THE_WITHDRAW = False" in reason for reason in reasons), reasons


def test_a_non_integrated_electric_source_is_not_this_seams_business(product):
    fields, pml = build()
    assert _reasons(product.folded_fused_hd_pair_coverage(
        fields, pml, (_NonIntegratedElectric(),))) == []


def test_a_magnetic_source_is_withdrawn_one_seam_earlier(product):
    fields, pml = build()
    assert _reasons(product.folded_fused_hd_pair_coverage(
        fields, pml, (_Magnetic(),))) == []


def test_an_unfolded_grid_is_refused_naming_the_plain_product(product):
    """The exact inverse of ``fused_hd_pair``'s own folded clause."""
    fields, pml = build(symmetry=())
    reasons = _reasons(product.folded_fused_hd_pair_coverage(fields, pml, ()))
    assert any("no axis is folded" in reason for reason in reasons), reasons
    assert any("fused_hd_pair" in reason for reason in reasons), reasons


def test_the_two_h_to_d_products_are_disjoint_in_both_directions(product, plain):
    """DRIVEN on the same fixtures, not argued from branch order.

    Exactly one of the two admits each configuration: the fold is required here and
    refused there, so a composer that offered both could not pick the wrong one by
    asking in the wrong order.
    """
    for keywords in ({}, {"boundaries": {"x": "metallic", "y": "periodic",
                                         "z": "periodic"}},
                     {"symmetry": (("X", 1), ("Y", 1))}, {"symmetry": ()}):
        fields, pml = build(**keywords)
        folded_ok = not _reasons(
            product.folded_fused_hd_pair_coverage(fields, pml, ()))
        plain_ok = not _reasons(plain.fused_hd_pair_coverage(fields, pml, ()))
        assert folded_ok != plain_ok, (keywords, folded_ok, plain_ok)
        assert folded_ok == bool(keywords.get("symmetry", (("Y", 1),)))


def test_an_inactive_absorber_is_named_by_the_constitutive_half_first(product):
    """THE ORDER IS THE DRIVER'S: ``update_H`` runs first, so it speaks first."""
    fields, pml = build(thickness=0.0)
    verdict = product.folded_fused_hd_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert verdict.reasons[0].startswith("folded constitutive half"), verdict.reasons


def test_a_conductivity_is_named_by_the_curl_half(product):
    """The constitutive half deliberately does NOT refuse it; the conjunction does."""
    fields, pml = build(sigma=0.4)
    reasons = _reasons(product.folded_fused_hd_pair_coverage(fields, pml, ()))
    assert reasons and all(reason.startswith("folded curl half")
                           for reason in reasons), reasons
    assert any("conductivity" in reason for reason in reasons), reasons


def test_complex_storage_is_refused_by_both_halves(product):
    fields, pml = build(complex_storage=True)
    reasons = _reasons(product.folded_fused_hd_pair_coverage(fields, pml, ()))
    assert any("complex64 storage is not carried" in reason
               for reason in reasons), reasons


def test_a_nonlinearity_is_refused(product):
    fields, pml = build(nonlinear=True)
    reasons = _reasons(product.folded_fused_hd_pair_coverage(fields, pml, ()))
    assert any("chi2/chi3" in reason for reason in reasons), reasons


def test_a_cylindrical_grid_is_refused(product):
    fields, pml = build()
    view = _FieldsView(fields, _GridView(fields.grid, cylindrical=True))
    reasons = _reasons(product.folded_fused_hd_pair_coverage(view, pml, ()))
    assert any("cylindrical" in reason for reason in reasons), reasons


def test_an_unresolvable_fold_is_refused_by_name(product):
    """``folded_axis_kinds``' own refusals reach the seam with their reasons."""
    fields, pml = build()
    view = _FieldsView(fields, _GridView(fields.grid,
                                         mirror_phase=lambda axis: 0))
    reasons = _reasons(product.folded_fused_hd_pair_coverage(view, pml, ()))
    assert any("mirror phase" in reason for reason in reasons), reasons


def test_the_builder_returns_none_rather_than_raising_on_a_refused_run(product):
    fields, pml = build(symmetry=())
    assert product.plan_folded_fused_hd_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _bare_arrays(product, shape):
    arrays = {name: numpy.zeros(shape, dtype=numpy.float32)
              for name in (product.ROTATED + product.IN_PLACE
                           + product.CONSTITUTIVE_SOURCES)}
    for name in product.ROTATED:
        arrays["scratch_" + name] = numpy.zeros(shape, dtype=numpy.float32)
    flat = {f"{stem}_{axis}": numpy.zeros(n, dtype=numpy.float32)
            for stem in ("kms", "sinv", "kps")
            for axis, n in zip("xyz", shape)}

    class _Holder:
        pass

    holder = _Holder()
    for name in product.ROTATED:
        setattr(holder, name, arrays[name])
    return arrays, flat, holder


def test_the_plan_rotates_exactly_the_six_magnetic_volumes(product):
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan

    assert issubclass(product.FoldedFusedHdPairPlan, ScratchWeldPairPlan)
    assert product.FoldedFusedHdPairPlan.replaces == product.REPLACES
    assert product.ROTATED == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")


def test_the_plan_is_a_sibling_rather_than_a_subclass_of_the_plain_one(product,
                                                                      plain):
    """A subclass would inherit ``_launch`` and fire the UNFOLDED kernel."""
    assert not issubclass(product.FoldedFusedHdPairPlan, plain.FusedHdPairPlan)
    assert "fused_constitutive_curl_H_to_D_folded_kernel" in \
        (PACKAGE_DIR / "folded_fused_hd_pair.py").read_text(encoding="utf-8")


def test_an_aliased_scratch_pair_is_refused_rather_than_launched(product):
    shape = (4, 5, 6)
    arrays, flat, holder = _bare_arrays(product, shape)
    product.plan_folded_fused_hd_pair_from_arrays(arrays, flat, 0.35, (1, 2, 0),
                                                  holder)
    arrays["scratch_Hx"] = arrays["Hx"]
    with pytest.raises((ValueError, RuntimeError)):
        plan = product.plan_folded_fused_hd_pair_from_arrays(
            arrays, flat, 0.35, (1, 2, 0), holder)
        plan._resolve()


def test_an_input_that_aliases_an_output_is_refused(product):
    shape = (4, 5, 6)
    arrays, flat, holder = _bare_arrays(product, shape)
    arrays["Bx"] = arrays["Dx"]
    with pytest.raises(ValueError):
        product.plan_folded_fused_hd_pair_from_arrays(arrays, flat, 0.35, (1, 2, 0),
                                                      holder)


def test_the_module_is_importable_without_triton(product):
    if product.triton is not None:  # pragma: no cover - a device host
        pytest.skip("Triton is importable here; the absence path is the laptop's")
    with pytest.raises(ImportError):
        product.fused_constitutive_curl_H_to_D_folded_kernel()


def test_the_four_ghost_codes_are_the_symmetry_modules_own(product, symmetry):
    """A plan built here and a plan built there index the same table."""
    source = (PACKAGE_DIR / "folded_fused_hd_pair.py").read_text(encoding="utf-8")
    for name in ("CODE_PERIODIC", "CODE_METALLIC", "CODE_MIRROR_METALLIC",
                 "CODE_MIRROR_PERIODIC"):
        assert f"_symmetry.{name}" in source, name
    assert (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC,
            symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC) == \
        (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# The composer
# ---------------------------------------------------------------------------

def test_the_composer_is_not_offered_this_product_and_the_reason_is_a_file_boundary():
    """Read off the tree, so a round that lands the label finds this red."""
    from .triton_kernels import launch as triton_launch

    assert FAMILY not in triton_launch.CERTIFIED_FUSED_PRODUCTS
    fastpath = (HERE / "fastpath.py").read_text(encoding="utf-8")
    assert "fused pair H->D (folded)" not in fastpath
    assert FAMILY not in fastpath


def test_the_composer_refuses_this_product_on_both_brakes(product):
    """DRIVEN against an injected row, on the real composer, on a folded fixture."""
    from .triton_kernels import launch as triton_launch

    products = triton_launch.CERTIFIED_FUSED_PRODUCTS
    arms = triton_launch.CERTIFIED_FUSED_PAIR_ARMS
    products[FAMILY] = {"curl_slot": "update_H", "module": FAMILY,
                        "coverage": "folded_fused_hd_pair_coverage",
                        "builder": "plan_folded_fused_hd_pair",
                        "label": "fused pair H->D (folded)"}
    arms[FAMILY] = product.ARMS
    try:
        uninstallable = triton_launch._declared_uninstallable(
            {FAMILY: product}, FAMILY, products[FAMILY])
        assert uninstallable is not None
        assert "INSTALLABLE = False" in uninstallable
        # The incumbent holds update_H, so the absorb brake refuses too.
        selected = {"step_B": "fused pair B (folded)",
                    "update_H": "fused pair B (folded)",
                    "step_D": "folded PML", "update_E": "folded"}
        assert triton_launch._pair_may_absorb(
            selected, "update_H", "step_D", product.ARMS) is not None
    finally:
        products.pop(FAMILY, None)
        arms.pop(FAMILY, None)


def test_the_h_to_d_seam_row_is_last_and_that_is_load_bearing():
    from . import withdraw_hoist
    from .triton_kernels import launch as triton_launch

    seams = triton_launch.CERTIFIED_FUSED_PAIR_SEAMS
    assert seams["update_H"] == ("step_D", withdraw_hoist.SEAM)
    assert list(seams)[-1] == "update_H"


# ---------------------------------------------------------------------------
# The gate's own host legs, and the records it rests on
# ---------------------------------------------------------------------------

def test_the_gates_host_legs_pass_here(gate):
    """A transcription drift fails at the merge bar, not at the far end of a queue."""
    for leg in (gate.leg_driver_order, gate.leg_transcription,
                gate.leg_ghost_observability):
        result = leg()
        assert result["passed"], result["findings"]


def test_every_mutation_needle_resolves_exactly_once(gate, product, plain):
    """A needle matching nothing is a DISARMED leg wearing a pass."""
    texts = {
        "folded": (PACKAGE_DIR / "folded_fused_hd_pair.py").read_text(
            encoding="utf-8"),
        "plain": (PACKAGE_DIR / "fused_hd_pair.py").read_text(encoding="utf-8"),
    }
    armed = 0
    for tag, spec in gate.MUTATIONS.items():
        if spec["target"] != "kernel":
            continue
        armed += 1
        text = texts[spec.get("module", "folded")]
        # A MUTATION MAY CARRY SEVERAL EDITS -- the paired control that earns a null
        # is two -- and every needle of every one must resolve exactly once.
        for old_text, _new_text in (spec.get("edits")
                                    or ((spec["old"], spec["new"]),)):
            assert text.count(old_text) == 1, (tag, spec.get("module"),
                                               old_text.strip()[:50])
    assert armed >= 12, armed
    assert texts[gate.BYTE_NEUTRAL.get("module", "folded")].count(
        gate.BYTE_NEUTRAL["old"]) == 1


def test_the_gates_cell_is_the_boards_cell(gate, product):
    assert tuple(gate.CELL_ARMS) == product.ARMS
    assert gate.CELL_ARMS == ("folded", "folded PML")


def test_the_lift_basis_is_the_cell_the_seam_record_prices(gate):
    """78 rows, 3 of them declined by name -- read from the record, not listed."""
    if not (RESULTS / gate.CENSUS).is_dir() or not (RESULTS / gate.SEAM_RECORD).is_dir():
        pytest.skip("the census/seam records are not in this checkout")
    rows, facts = gate.lift_basis(RESULTS)
    assert facts["rows_in_cell"] == 78, facts
    assert len(facts["rows_with_a_standing_withdraw"]) == 3, facts
    assert set(facts["rows_with_a_standing_withdraw"]) == {
        "examples:absorbed_power_density.py", "examples:finite_grating.py",
        "examples:mie_scattering.py"}
    assert all(tuple(row["arms"].values()) == gate.CELL_ARMS for row in rows)


def test_the_boards_cell_is_seventy_eight_and_seventy_five_are_unbuilt():
    """The number in the module docstring, re-read from the board it cites."""
    path = RESULTS / BOARD / "fusion_matrix.json"
    if not path.is_file():
        pytest.skip("the board is not in this checkout")
    board = json.loads(path.read_text(encoding="utf-8"))
    seam = board["aggregate"]["h_to_d_seam"]
    cell = [row for row in seam["instances"]
            if (row["update_H"], row["step_D"]) == ("folded", "folded PML")]
    assert len(cell) == 78
    buckets = {}
    for row in cell:
        buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1
    assert buckets == {"buildable_not_built": 75, "withdraw_seam": 3}, buckets
    assert board["aggregate"]["denominator"] == 597


def test_the_arbitration_join_in_the_reason_is_the_boards_own(product):
    """The loss/tie/gain split is a MEASUREMENT off the board, re-derived here."""
    path = RESULTS / BOARD / "fusion_matrix.json"
    if not path.is_file():
        pytest.skip("the board is not in this checkout")
    board = json.loads(path.read_text(encoding="utf-8"))
    seam = board["aggregate"]["h_to_d_seam"]
    cell = {row["row"] for row in seam["instances"]
            if (row["update_H"], row["step_D"]) == ("folded", "folded PML")}
    served = {"B->H": 0, "D->E": 0}
    for entry in board["seam_instances"]:
        if entry["row"] in cell and entry["seam"] in served:
            if entry.get("predicate_admits") and entry.get("product"):
                served[entry["seam"]] += 1
    assert served["B->H"] == 78, served
    assert served["D->E"] == 67, served
    # LOSS on 67, TIE on 11, GAIN on 0 -- and the reason says exactly that.
    assert "LOSS on 67" in product.INSTALLABLE_REASON
    assert "TIE on 11" in product.INSTALLABLE_REASON
    assert "GAIN on ZERO" in product.INSTALLABLE_REASON


def test_the_mask_attribution_this_family_cites_is_still_what_the_record_says():
    """The Metal sibling's measured fu_D attribution, re-read from its artifact.

    This module's docstring and this family's gate both rest on it: at WHOLE-STEP
    granularity the two mask drops move words in ``fu_D`` and none in ``D``. A
    comparison that carried only the primaries would certify a mask-less folded curl
    as correct, so the citation is checked against the record rather than remembered.
    """
    path = RESULTS / METAL_ARTIFACT / "gate.json"
    if not path.is_file():
        pytest.skip("the Metal sibling's artifact is not in this checkout")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    row = next(entry for entry in artifact["rows"] if entry["leg"] == "mutation")
    finding = row["predicted_null"]["the_masks_were_predicted_null_at_whole_step_and_are_not"]
    measured = finding["measured"]
    assert measured["drop_cell0_mask"] == {"words_in_D": 0, "words_in_fu_D": 540}
    assert measured["drop_top_plane_mask"] == {"words_in_D": 0, "words_in_fu_D": 264}
    assert finding["prediction_held"] is False
    source = (PACKAGE_DIR / "folded_fused_hd_pair.py").read_text(encoding="utf-8")
    assert "264" in source and "540" in source


def test_the_gate_binds_the_folded_sources_it_depends_on(gate):
    """A driver, composer or symmetry move must withhold this credit, not hide."""
    for name in ("meep_gpu/driver.py", "meep_gpu/stepping.py",
                 "meep_gpu/triton_kernels/symmetry.py",
                 "meep_gpu/triton_kernels/folded_fused_hd_pair.py",
                 "meep_gpu/triton_kernels/fused_hd_pair.py",
                 "meep_gpu/triton_kernels/launch.py",
                 "parity/meep_gpu/gate_triton_folded_fused_hd_pair.py"):
        assert name in gate.SOURCES, name
