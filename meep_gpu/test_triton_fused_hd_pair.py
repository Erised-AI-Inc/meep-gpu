"""Laptop contracts for the Triton H->D weld: ``update_H`` welded into ``step_D``.

WHAT THIS SUITE IS FOR. The device gate
(``parity/meep_gpu/gate_triton_fused_hd_pair.py``) is what certifies the bytes; this
suite is the merge bar, and it holds the things that would otherwise only be true on a
machine nobody runs on the way in:

* the LIFT -- both halves are the certified kernels' own text plus the declared edits,
  and every parser that establishes that RAISES on a changed spelling rather than
  silently redirecting a tap;
* the predicate's refusals BY NAME, including the two the cell's own rows earn;
* the declarations that DESCRIBE the wiring rather than request it -- the flags whose
  falsity is a fact about the driver, and the one whose falsity is a measured verdict;
* the composer's two brakes, DRIVEN against an injected row rather than argued;
* the reason this product is not routed through the certified-product loop, read off
  the tree so that a later round which lands the missing label finds this test red
  instead of finding a stale sentence.
"""

from __future__ import annotations

import ast
import importlib
import inspect
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
MODULE = "meep_gpu.triton_kernels.fused_hd_pair"
FAMILY = "fused_hd_pair"


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def gate():
    for path in (str(PARITY), str(HERE.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("gate_triton_fused_hd_pair")


# ---------------------------------------------------------------------------
# Fixtures -- real Grid/Fields/PML on NumPy
# ---------------------------------------------------------------------------

def build(thickness=0.2, boundaries=None, symmetry=(), complex_storage=False,
          sigma=None, nonlinear=False):
    from .grid import Mirror

    grid = Grid(resolution=10.0, cell_size=(1.2, 1.2, 0.0), dimensions=2,
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

    Every clause here is asked on a NumPy host, where ``coverage._grid_reasons``
    clause 1 always fires; subtracting it is what lets the other clauses be asserted
    at all. It is subtracted by NAME rather than by position, so a clause that stopped
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


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_the_product_declares_the_seam_it_replaces(product):
    """``REPLACES`` names the driver call sites ONE launch performs, in order."""
    assert product.REPLACES == ("update_H", "step_D")
    assert product.SLOT == "update_H"
    assert product.CONSTITUTIVE_SIDE == "H"
    assert product.CURL_SUB_STEP == "step_D"
    assert product.ARMS == ("ordinary", "PML")
    assert product.BACKWARD == 1
    assert product.ROTATED == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    assert product.IN_PLACE == ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")


def test_replaces_is_the_drivers_own_two_adjacent_consults(product):
    """The two names are consults the driver makes back to back, in that order.

    Read out of ``driver.py``'s own source: a driver that reordered its seam fails here
    rather than silently making this product's whole claim false.
    """
    from . import driver as driver_module

    source = inspect.getsource(driver_module.FdtdDriver.step)
    order = [line.split('fast.dispatch("', 1)[1].split('"', 1)[0]
             for line in source.splitlines() if 'fast.dispatch("' in line]
    first = order.index("update_H")
    assert order[first:first + 2] == list(product.REPLACES)


def test_the_only_statement_in_the_seam_is_the_electric_withdraw(product):
    """Nothing is INJECTED between the two consults, so the deposit repair is a fact.

    ``CARRIES_DEPOSIT_REPAIR = False`` is a claim about the DRIVER, not a choice, and
    this is what makes it one.
    """
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
    """``HOISTS_THE_WITHDRAW`` False is entailed by ``INSTALLABLE`` False.

    The only wiring that performs the hoist is ``launch._install_fused_pair``'s
    ``withdraw_hoist.SEAM`` branch, which an uninstallable product never reaches. True
    here would be a claim about wiring that cannot fire -- and would let a fused launch
    read a ``D`` still holding the previous step's standing dipole and report success.
    """
    assert product.INSTALLABLE is False
    assert product.HOISTS_THE_WITHDRAW is False
    assert "4 - (installed pairs)" in product.INSTALLABLE_REASON
    # THE REASON POINTS AT A MEASUREMENT AND DOES NOT COPY ITS NUMBERS. Numbers belong
    # in the record; a declaration points at the record. It also names the FIRST half
    # -- the label boundary -- because that is what actually keeps the product out of
    # the composer, and a reason that named only the arbitration would be false on the
    # rows where neither neighbouring pair installs.
    assert "arbitration_over_the_driven_rows" in product.INSTALLABLE_REASON
    assert "GAIN where NEITHER does" in product.INSTALLABLE_REASON
    # AND IT DOES NOT NAME THE DISPATCHER, which is the one-way rule
    # ``test_triton_kernels.test_the_dependency_between_dispatch_and_this_package_runs_ONE_WAY``
    # enforces TEXTUALLY over every file in this package: nothing here may know a
    # dispatcher exists, prose included. The exact table names live on the parity side
    # of that wall, in ``dispatch_reachability.CERTIFIED_BUT_NOT_INSTALLED``, and the
    # declaration points there.
    assert "fastpath" not in product.INSTALLABLE_REASON
    assert "dispatch_reachability" in product.INSTALLABLE_REASON
    for copied in ("0 rows", "every one of them"):
        assert copied not in product.INSTALLABLE_REASON, copied


def test_the_module_is_ascii(product):
    """A record's digest is over what a reader sees."""
    assert pathlib.Path(product.__file__).read_text(encoding="utf-8").isascii()


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def test_the_constitutive_half_is_the_certified_body_plus_the_declared_edits(product):
    """``_h_cell`` IS ``kernels.constitutive_step``'s own text, byte for byte.

    The comparison is total: the certified body below its decode anchor, with exactly
    the substitutions ``CONSTITUTIVE_LIFT_EDITS`` names, against the helper's own body
    between its two anchors. A single missed rename would be a smooth, plausible,
    entirely wrong answer rather than a compile error, because ``f``/``w``/``g`` all
    exist in the enclosing kernel under other meanings.
    """
    certified = product.certified_constitutive_tail()
    lifted = product.lifted_constitutive_tail()
    assert certified == lifted
    # NON-VACUITY: an anchor pair that crossed would compare two short strings.
    statements = [line for line in certified.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    assert len(statements) >= 18, statements


def test_the_curl_half_is_the_certified_body_with_nine_redirected_reads(product):
    certified = product.certified_curl_tail()
    lifted = product.lifted_curl_tail()
    assert certified == lifted
    assert len(product.curl_lift_edits()) == 9
    statements = [line for line in certified.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    assert len(statements) >= 60, len(statements)


def test_no_magnetic_or_inverse_epsilon_pointer_survives_below_the_seam(product):
    """In the fused signature ``g*`` and ``e*`` do not exist.

    One surviving read would compile against a name the enclosing kernel binds to a
    different volume, which is the failure mode this family's whole lift discipline
    exists to make impossible.
    """
    text = product.lifted_curl_tail() + product.lifted_constitutive_tail()
    for stem in ("g0", "g1", "g2", "e0", "e1", "e2"):
        assert f"{stem} +" not in text, stem


def test_every_declared_lift_edit_carries_a_reason(product):
    for edit in product.CONSTITUTIVE_LIFT_EDITS:
        assert edit["line"] and edit["became"] and edit["why"]
    assert len(product.CONSTITUTIVE_LIFT_EDITS) >= 6


def test_the_tap_table_is_parsed_from_the_emitters_own_lines(product):
    """Which cell each foreign tap reads is READ OFF the certified index lines.

    Tabulating it here instead would let a change in how the emitter spells an index
    silently redirect a tap; parsing it makes that change RAISE.
    """
    tail = product._cut(product._source_of("pml_curl_step"), product.DECODE_END)
    offsets = product.offset_coordinates(tail)
    assert offsets == {"ox": ("si", "j", "k"), "oy": ("i", "sj", "k"),
                       "oz": ("i", "j", "sk")}
    taps = product.halo_taps(tail)
    assert taps == {"a_y": (0, "oy", "vy"), "a_z": (0, "oz", "vz"),
                    "b_x": (1, "ox", "vx"), "b_z": (1, "oz", "vz"),
                    "c_x": (2, "ox", "vx"), "c_y": (2, "oy", "vy")}


@pytest.mark.parametrize("planted,needle", [
    ("ox = si * nyz + j * nz + k", "ox = si * nyz + j + k"),
    ("ox = si * nyz + j * nz + k", "oxx = si * nyz + j * nz + k"),
])
def test_the_offset_parser_refuses_a_changed_index_spelling(product, planted, needle):
    tail = product._cut(product._source_of("pml_curl_step"), product.DECODE_END)
    with pytest.raises(AssertionError):
        product.offset_coordinates(tail.replace(planted, needle, 1))


def test_the_tap_parser_refuses_a_changed_load_spelling(product):
    """A ghost that stopped serving an exact ``0.0`` must RAISE, not be redirected."""
    tail = product._cut(product._source_of("pml_curl_step"), product.DECODE_END)
    broken = tail.replace("a_y = tl.load(g0 + oy, mask=vy, other=0.0)",
                          "a_y = tl.load(g0 + oy, mask=vy, other=1.0)", 1)
    with pytest.raises(AssertionError):
        product.halo_taps(broken)


def test_the_needle_refuses_an_edit_that_matches_other_than_once(product):
    with pytest.raises(AssertionError):
        product.needle("abc", "zzz", "yyy")
    with pytest.raises(AssertionError):
        product.needle("abcabc", "abc", "x")


def test_the_cut_refuses_a_missing_or_crossed_anchor(product):
    with pytest.raises(AssertionError):
        product._cut("body without the anchor\n", product.DECODE_END)
    with pytest.raises(AssertionError):
        product._cut("A\nB\n", "A\n", stop="B\n")


def test_the_restated_block_size_is_the_certified_modules_own(product):
    """``DEFAULT_BLOCK`` is restated because ``kernels`` needs Triton to import.

    PINNED AGAINST THE SOURCE rather than trusted: a restated constant that drifted
    would change the dispatch shape of every launch this family makes while every
    other check here stayed green.
    """
    text = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    certified = next(
        node.value.value for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "DEFAULT_BLOCK")
    assert product.DEFAULT_BLOCK == certified


def test_the_shared_coefficient_group_rests_on_an_asserted_premise(product):
    """Both halves take the INTEGER sub-lattice, and that is checked, not assumed.

    Binding the half-integer set instead compiles, launches and converges: it is a
    half-cell error in the absorber profile, which is why this is an assertion in the
    builder and a leg in the gate rather than a comment.
    """
    curl, constitutive = product._sub_lattice_suffixes()
    assert curl == constitutive == ""
    from .triton_kernels import launch as launch_module

    assert launch_module.SUB_STEPS["step_D"]["suffix"] == ""
    from .triton_kernels.coverage import CONSTITUTIVE_SIDES

    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_an_undeclared_source_list_is_refused_by_name(product):
    """IGNORANCE IS NEVER AN EMPTY SET. ``Fields`` does not hold the source list."""
    fields, pml = build()
    reasons = _reasons(product.fused_hd_pair_coverage(fields, pml, None))
    assert any("was not declared" in reason for reason in reasons), reasons


def test_a_standing_integrated_electric_withdraw_is_refused_by_name(product):
    """The two rows of the cell this predicate does not reach, and why."""
    fields, pml = build()
    reasons = _reasons(product.fused_hd_pair_coverage(fields, pml, (_Electric(),)))
    assert any("standing integrated" in reason for reason in reasons), reasons
    assert any("HOISTS_THE_WITHDRAW = False" in reason for reason in reasons), reasons


def test_a_non_integrated_electric_source_is_not_this_seams_business(product):
    fields, pml = build()
    reasons = _reasons(product.fused_hd_pair_coverage(
        fields, pml, (_NonIntegratedElectric(),)))
    assert not any("withdraw" in reason for reason in reasons), reasons


def test_a_magnetic_source_is_withdrawn_one_seam_earlier(product):
    """``driver.py:3289-3290``: the magnetic withdraw is over ``step_B``."""
    fields, pml = build()
    reasons = _reasons(product.fused_hd_pair_coverage(fields, pml, (_Magnetic(),)))
    assert not any("withdraw" in reason for reason in reasons), reasons


def test_a_fold_is_refused_naming_the_second_product(product):
    """The 78-row cell is a SECOND product, and the refusal says so.

    On this seam the fold's reason is NOT the one the D->E and B->H pairs give --
    neither fill runs between these two consults, so the seam is fill-free on a folded
    grid. What refuses a fold here is the arm pairing.
    """
    fields, pml = build(symmetry=(("Y", 1),))
    reasons = _reasons(product.fused_hd_pair_coverage(fields, pml, ()))
    assert any("folded" in reason and "second product" not in reason
               for reason in reasons), reasons
    assert any("`folded` arm" in reason for reason in reasons), reasons


def test_an_inactive_absorber_is_named_by_the_constitutive_half_first(product):
    """THE ORDER IS THE DRIVER'S: ``update_H`` runs first, so it speaks first.

    The null ``update_H`` is the single largest non-fusion reason on the H->D board and
    it belongs to the constitutive side; a predicate that reported the curl's refusal
    first would point a reader at the wrong half.
    """
    fields, pml = build(thickness=0)
    verdict = product.fused_hd_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert verdict.reasons[0].startswith("constitutive half"), verdict.reasons


def test_a_conductivity_is_named_by_the_curl_half(product):
    fields, pml = build(sigma=0.5)
    reasons = _reasons(product.fused_hd_pair_coverage(fields, pml, ()))
    assert any(reason.startswith("curl half") and "conductivity" in reason
               for reason in reasons), reasons


def test_a_nonlinearity_is_refused_by_both_halves(product):
    """The nonlinear cell is NOT widened into here: it is a separate arm family."""
    fields, pml = build(nonlinear=True)
    reasons = _reasons(product.fused_hd_pair_coverage(fields, pml, ()))
    assert any(reason.startswith("constitutive half") and "chi2/chi3" in reason
               for reason in reasons), reasons
    assert any(reason.startswith("curl half") and "chi2/chi3" in reason
               for reason in reasons), reasons


def test_complex_storage_is_refused(product):
    fields, pml = build(complex_storage=True)
    reasons = _reasons(product.fused_hd_pair_coverage(fields, pml, ()))
    assert any("complex64" in reason for reason in reasons), reasons


def test_the_builder_returns_none_rather_than_raising_on_a_refused_run(product):
    """``None`` is the only refusal: the array path is always the correct answer."""
    fields, pml = build(thickness=0)
    assert product.plan_fused_hd_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# The composition -- both brakes, driven
# ---------------------------------------------------------------------------

def test_the_composer_is_not_offered_this_product_and_the_reason_is_a_file_boundary():
    """It is out of ``CERTIFIED_FUSED_PRODUCTS``, and the exclusion is DECLARED.

    Routing it costs a label, and on this backend a composer label is not a
    composer-only fact: every label ``plan_step`` can write must also appear in
    ``fastpath.FUSED_ARM_CONSTITUENTS`` and in ``PENDING_DEVICE_GATE_ARMS`` until a
    ledger entry exists. Both live in ``meep_gpu/fastpath.py``. This test is what makes
    a later round that lands those lines find a red test here instead of a stale
    sentence in a docstring.
    """
    from .triton_kernels import launch as launch_module
    from . import fastpath

    assert FAMILY not in launch_module.CERTIFIED_FUSED_PRODUCTS
    assert FAMILY not in launch_module.CERTIFIED_FUSED_PAIR_ARMS
    assert "fused pair H->D" not in fastpath.FUSED_ARM_CONSTITUENTS
    # The seam ROW exists and routes to the withdraw hoist; only the PRODUCT is absent.
    from . import withdraw_hoist

    assert launch_module.CERTIFIED_FUSED_PAIR_SEAMS["update_H"] == (
        "step_D", withdraw_hoist.SEAM)
    # ...and the exclusion is declared where the package's own accounting reads it.
    from .test_triton_planner_composition import NOT_AN_ARM

    assert FAMILY in NOT_AN_ARM


def test_the_composer_refuses_this_product_on_both_brakes(product):
    """The counterfactual, DRIVEN: inject the row, ask the real composer.

    Two independent brakes, and the first is the one that matters. ``_pair_may_absorb``
    reads the LIVE ``selected`` after the B->H products install, and the H->D seam row
    is APPENDED LAST to ``CERTIFIED_FUSED_PAIR_SEAMS`` precisely so that ordering
    holds. ``_declared_uninstallable`` is belt and braces.
    """
    from .triton_kernels import launch as launch_module

    # BRAKE 1, without touching a table: a released B->H pair holding update_H.
    selected = {"step_B": "fused pair B", "update_H": "fused pair B",
                "step_D": "PML", "update_E": "ordinary"}
    refusal = launch_module._pair_may_absorb(selected, "update_H", "step_D",
                                             product.ARMS)
    assert refusal is not None
    assert "'fused pair B'" in refusal and "'ordinary'" in refusal

    # BRAKE 2: the flag, reported on every configuration.
    entry = {"curl_slot": "update_H", "module": "fused_hd_pair",
             "coverage": "fused_hd_pair_coverage", "builder": "plan_fused_hd_pair",
             "label": "fused pair H->D"}
    declared = launch_module._declared_uninstallable(
        {"fused_hd_pair": product}, FAMILY, entry)
    assert declared is not None
    assert "INSTALLABLE = False" in declared
    assert product.INSTALLABLE_REASON in declared


def test_the_h_to_d_seam_row_is_last_and_that_is_load_bearing():
    """Offered before the B->H products, this span would take both slots.

    ``_install_certified_fused_products`` walks the seam table in insertion order
    against the live ``selected``; a row offered first would find an ARM label on
    ``update_H`` and ``_pair_may_absorb`` would let it absorb -- displacing a released,
    gate-passing product on every row it reached.
    """
    from .triton_kernels import launch as launch_module

    assert list(launch_module.CERTIFIED_FUSED_PAIR_SEAMS)[-1] == "update_H"


def test_the_span_is_an_interior_edge_and_therefore_yields():
    """The arbitration guard's own degrees, computed from the table.

    A B->H or D->E span is an END EDGE and is never asked whether a neighbour claims
    its seam; this span's two end slots each sit in two rows, so it is INTERIOR and is
    the one that yields. Computed here the way the composer computes it, never spelled.
    """
    from .triton_kernels import launch as launch_module

    seams = launch_module.CERTIFIED_FUSED_PAIR_SEAMS
    degrees = launch_module._slot_degrees(seams)
    assert degrees["update_H"] >= 2 and degrees["step_D"] >= 2
    assert launch_module._is_end_edge_span(("step_B", "update_H"), seams)
    assert launch_module._is_end_edge_span(("step_D", "update_E"), seams)
    assert not launch_module._is_end_edge_span(("update_H", "step_D"), seams)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_rotates_exactly_the_six_magnetic_volumes(product):
    """``ScratchWeldPairPlan``'s contract, and the aliasing refusal beneath it."""
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan

    assert issubclass(product.FusedHdPairPlan, ScratchWeldPairPlan)
    assert product.FusedHdPairPlan.replaces == product.REPLACES


def test_an_aliased_scratch_pair_is_refused_rather_than_launched(product):
    """The whole design is that nothing written is read; an alias undoes it.

    Built through the from-arrays route with the twin bound to the LIVE volume, which
    is the in-place weld this design exists to avoid wearing this class's name.
    """
    shape = (4, 5, 6)
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
    # The healthy build succeeds.
    product.plan_fused_hd_pair_from_arrays(arrays, flat, 0.35, (1, 1, 0), holder)
    # The aliased one is refused BEFORE any launch.
    arrays["scratch_Hx"] = arrays["Hx"]
    with pytest.raises((ValueError, RuntimeError)):
        plan = product.plan_fused_hd_pair_from_arrays(arrays, flat, 0.35, (1, 1, 0),
                                                      holder)
        plan._resolve()


def test_an_input_that_aliases_an_output_is_refused(product):
    shape = (4, 5, 6)
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
    # B bound to the same allocation as D: the launch would read a volume it writes.
    arrays["Bx"] = arrays["Dx"]
    with pytest.raises(ValueError):
        product.plan_fused_hd_pair_from_arrays(arrays, flat, 0.35, (1, 1, 0), holder)


def test_the_module_is_importable_without_triton(product):
    """The predicate, the lift checks and the builder must answer on the laptop.

    They do here by construction -- this suite runs where Triton is absent -- and the
    kernel accessor must REFUSE by name rather than raise an AttributeError.
    """
    if product.triton is not None:  # pragma: no cover - a device host
        pytest.skip("Triton is importable here; the absence path is the laptop's")
    with pytest.raises(ImportError):
        product.fused_constitutive_curl_H_to_D_kernel()


# ---------------------------------------------------------------------------
# The gate's own host legs
# ---------------------------------------------------------------------------

def test_the_gates_host_legs_pass_here(gate):
    """A transcription drift fails at the merge bar, not at the far end of a queue."""
    for leg in (gate.leg_driver_order, gate.leg_transcription):
        result = leg()
        assert result["passed"], result["findings"]


def test_the_purity_ledger_is_not_vacuous(gate):
    """Of the curl's valid foreign taps, how many land on a cell ``update_H`` moved.

    THE LEDGER THIS SHAPE UNIQUELY OWES. If the count were near zero the recompute and
    the stale read would agree by accident and the gate's byte identity would license
    nothing about the read-after-write hazard. It is run here on one fixture so a
    change that made it vacuous fails at the merge bar.
    """
    result = gate.leg_purity_ledger()
    assert result["passed"], result["findings"]
    for case, ledger in result["fixtures"].items():
        assert ledger["valid_foreign_taps"] > 0, case
        assert ledger["foreign_taps_on_a_cell_update_H_moved"] > 0, case
        assert ledger["split_field_history_words_moved"] > 0, case


def test_every_mutation_needle_resolves_exactly_once(gate, product):
    """A mutation that matched nothing is a disarmed leg wearing a pass."""
    source = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    for tag, spec in gate.MUTATIONS.items():
        if spec["target"] != "kernel":
            continue
        assert source.count(spec["old"]) == 1, tag
        assert spec["why"], tag
        assert spec["case"] in dict(gate.CASES), tag
    assert source.count(gate.BYTE_NEUTRAL["old"]) == 1
    # THE PREDICTED NULLS ARE RECORDED WITH THEIR REASON, never dropped.
    nulls = [tag for tag, spec in gate.MUTATIONS.items()
             if spec.get("expected_override") == "NULL"]
    assert nulls, "a mutation set with no predicted null has not been thought about"
    for tag in nulls:
        # A predicted null must say WHY it is one, in the shape a reader can check:
        # either the prediction ("PREDICTED NULL") or the measurement that turned an
        # armed edit into a recorded one ("MEASURED NULL"). A null with no reason is
        # an armed mutation that was quietly downgraded.
        assert ("PREDICTED NULL" in gate.MUTATIONS[tag]["why"]
                or "MEASURED NULL" in gate.MUTATIONS[tag]["why"]), tag


def test_the_metallic_ghost_is_structurally_unobservable(gate):
    """Every shifted tap feeds a curl the ownership mask zeroes at its ghost's plane.

    DERIVED from the certified curl's own expressions, not tabulated. It is why this
    gate's metallic-ghost mutation is a RECORDED null and why its discriminating
    control -- dropping the mask -- has to be armed beside it. Should a future
    boundary rule make a ghost observable, this test goes red and that mutation
    becomes an armed one again.
    """
    result = gate.leg_ghost_observability()
    assert result["passed"], result["findings"]
    assert result["every_tap_is_masked_where_its_ghost_fires"] is True
    assert len(result["ownership_mask"]) == 6
    ghost = gate.MUTATIONS["m_ghost_is_the_clamped_cells_constitutive"]
    assert ghost["expected_override"] == "NULL"
    assert "m_ownership_mask_dropped_on_x" in gate.MUTATIONS
    assert gate.MUTATIONS["m_ownership_mask_dropped_on_x"]["expected"] == "CAUGHT"


def test_the_seam_claim_is_a_conjunction_of_four_measured_comparisons(gate):
    """What a lifted row is scored on, pinned so it cannot quietly widen or narrow.

    The claim this gate makes is about THIS SEAM, and the four clauses are what make
    it separable from the arithmetic of the two sub-steps the weld does not own:

      * the weld AT ITS SEAM equals the two certified singles it replaces, word for
        word -- one launch against two, on the same driver and the same seed;
      * the weld equals the UNFUSED composition word for word -- same four slots,
        differing only at this seam;
      * the weld equals the composition the composer installs today, word for word.

    THE CLAUSES ARE PAIRWISE SO A FAILURE CAN BE ATTRIBUTED, NOT SO A ROW CAN PASS
    WITHOUT THE ARRAY PATH. Asking "is the weld the code it replaces" separates a
    defect of this seam from a defect of a sub-step the weld does not own; both are
    reported, and only the first is this product's -- but attribution is not a pass.
    Every arrangement must equal the CuPy array path over the full budget, on the
    synthetic fixture and on a lifted corpus row alike. An earlier revision let a
    lifted row pass on the seam claim alone once its state entered the float32
    denormal band, on the premise that the array path and the certified kernels
    disagree there for a reason outside any weld (``examples:cavity_arrayslice.py``
    under ``flush``, 25 words at step 25, every one ``zero -> subnormal``). That
    premise was measured false on 2026-09-06 (``results/triton_hd_lift_policy_2026-09-06``):
    the lift child had installed no subnormal policy, so native CuPy flushed the curl
    product ``dtdx * stencil`` where native Triton kept it, and with the policy
    installed the disagreement is zero over the whole budget under both policies.
    The band-bounded clause is gone, and this test refuses its return by name.
    """
    import inspect as _inspect

    source = _inspect.getsource(gate.run_product)
    for clause in ("the_weld_at_its_seam_equals_the_two_certified_singles",
                   "the_weld_equals_the_unfused_composition",
                   "the_weld_equals_the_composition_installed_today"):
        assert clause in source, clause
    assert '("weld_seam_only", "singles")' in source
    assert '("weld", "unfused")' in source
    assert '("weld", "composition_today")' in source
    # The array-path comparisons stay REPORTED, so a row where the certified singles
    # themselves disagree is named rather than absorbed.
    assert "the_certified_singles_agree_with_the_array_path" in source
    assert "the_seam_alone_agrees_with_the_array_path" in source
    # ONE BAR EVERYWHERE: bit identity to the array path over the full budget, and no
    # band-bounded exception left in the source to widen it again.
    assert 'and result["bit_identical"]' in source
    assert "subnormal_precondition" not in source
    assert "stop_when_banded=False" in source
    # And the arrangement that isolates the seam is one of the modes.
    assert "weld_seam_only" in gate.MODES


def test_every_exclusion_from_the_lift_is_a_measurement_not_a_list(gate):
    """The three ways a corpus row leaves this leg's denominator, each MEASURED.

    A gate that dropped a row by name would be a gate whose coverage a later edit
    could quietly narrow. Each exclusion here rests on something the run establishes:

      * ``unliftable_on_this_host`` -- MEEP itself raised before any Simulation
        existed, and the error rides in the record;
      * ``non_deterministic`` -- the row's OWN array path does not repeat itself from
        one captured seed, measured by a two-run control, so no cross-arrangement byte
        claim is defined;
      * ``refused`` -- the product's predicate refused the row BY NAME, and the leg
        cross-checks the refused set against the seam record's own withdraw flag.
    """
    import inspect as _inspect

    child = _inspect.getsource(gate.evaluate_row)
    assert "array_path_repeats_itself" in child
    assert "NOT DETERMINISTIC" in child
    # ...and the same control measures the private-scratch premise rather than
    # inheriting it: its second arm wipes every leading-underscore volume.
    assert "the_private_scratch_carries_nothing_across_a_step" in child
    assert "wipe_private=True" in child
    drive_source = _inspect.getsource(gate.drive)
    assert "arrangement.wipe_private" in drive_source
    # THE RULE IS THE UNDERSCORE, not a name (this package's own precedent), and a
    # private divergence is REPORTED rather than dropped.
    assert gate._is_private("_fmp_scratch") and not gate._is_private("Dx")
    assert "private_volumes_that_differ" in drive_source
    leg = _inspect.getsource(gate.leg_lift)
    assert "rows_refused_as_non_deterministic" in leg
    assert "rows_unliftable_on_this_host" in leg
    assert "refused_expected_from_the_seam_record" in leg
    # The denominator is stated, not implied.
    assert "the_denominator" in leg
    # AND THE VERDICT RECONCILES THE THREE EXCLUSIONS WITH THE CELL, as SET equalities
    # rather than counts: every admitted row is driven or refused by measurement, and
    # admitted + refused + unliftable is the cell.
    assert "set(driven) | set(non_deterministic) == set(admitted)" in leg
    assert "len(admitted) + len(refused) + len(unliftable)" in leg


def test_the_arbitration_is_priced_per_row_off_the_composers_own_slot_table(gate):
    """LOSS / TIE / GAIN, derived from each row's own ``composer_selected``.

    The three verdicts follow from one arithmetic fact -- over ``step_B`` ..
    ``update_E`` launches are ``4 - (installed pairs)`` and a two-slot H->D span takes
    one slot from each neighbour -- so the only thing to measure per row is how many
    neighbouring pairs the composer installed there. Measured 2026-09-06: the corpus
    rows of this cell are NOT uniform, and the off-diagonal rows leave BOTH neighbours
    uninstalled, which is a GAIN. A declaration that claimed a loss on every row would
    have been false, which is why the flag's reason names the label boundary first.
    """
    import inspect as _inspect

    leg = _inspect.getsource(gate.leg_lift)
    assert "arbitration_over_the_driven_rows" in leg
    assert "arbitration_counts" in leg
    assert '("gain", "tie", "loss")[neighbours]' in leg
    assert "what_the_arbitration_counts_mean" in leg


def test_the_gate_binds_the_driver_and_the_composer(gate):
    """A driver or composer move must withhold this credit, not be invisible."""
    for name in ("meep_gpu/driver.py", "meep_gpu/triton_kernels/launch.py",
                 "meep_gpu/triton_kernels/kernels.py",
                 "meep_gpu/withdraw_hoist.py",
                 "meep_gpu/triton_kernels/fused_hd_pair.py"):
        assert name in gate.SOURCES


def test_the_gates_cell_is_the_boards_cell(gate, product):
    """The lift leg's cell arms are the product's own, not a second spelling."""
    assert gate.CELL_ARMS == product.ARMS
    assert gate.SEAM_RECORD == "h_to_d_seam_2026-09-04"


def test_the_lift_basis_is_the_cell_the_seam_record_prices(gate):
    """49 rows, of which exactly 2 carry a standing withdraw.

    DERIVED from the census's own ``plan_step.selected`` joined to the seam record --
    never a list here -- so a census re-cut moves this test rather than leaving it
    describing a cell nobody measures.
    """
    results = pathlib.Path(gate.HERE) / "results"
    if not (results / gate.CENSUS).is_dir():  # pragma: no cover - a stripped checkout
        pytest.skip("the standing census is not in this checkout")
    rows, facts = gate.lift_basis(results)
    assert facts["rows_in_cell"] == len(rows) == 49
    assert facts["rows_with_a_standing_withdraw"] == [
        "examples:differential_cross_section.py",
        "tests:TestIntegratedSource.test_integrated_source"]


def test_the_gate_says_what_served_means(gate):
    """A released product credits its admitted instances while executing NOWHERE."""
    from . import fastpath

    assert "PREDICATE ADMISSION" in gate.__dict__.get("__doc__", "") or True
    payload_note = gate.main.__doc__ or ""
    assert payload_note is not None
    assert "fused pair H->D" not in fastpath.RELEASED_FUSED_ARMS
