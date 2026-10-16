"""Laptop contracts for the Triton Dcyl m = 0 REAL H->D product:
``triton_kernels/cylindrical_real_fused_hd_pair.py``.

The device gate (``parity/meep_gpu/gate_triton_cylindrical_real_fused_hd_pair.py``) is
what certifies the bytes; this suite is the merge bar. It holds what would otherwise be
true only on a machine nobody runs on the way in:

* the declarations DESCRIBE the seam, the oracle and the wiring rather than request
  them -- the radial sum's component and ``ir0`` are read off ``stepping.step_D``'s own
  text and the certified family's table, and the scan's row vectors are the certified
  host bindings;
* launch 1 is the certified Cartesian weld's constitutive block, statement for
  statement, and spells no arithmetic of its own; the constitutive it calls is IMPORTED
  from that weld, whose lift is re-asserted here because this product launches it;
* launches 2 and 3 are the certified ``CylindricalCurlPlan``, built by the certified
  builder over each rotating allocation, and no curl or scan text is copied;
* the predicate's refusals BY NAME, on real Dcyl objects on NumPy;
* the plan's aliasing refusals and its rotation contract;
* the composer is offered this product (routed 2026-09-07: the composer's, the
  dispatcher's and the board's tables all carry it) and refuses it by its own
  ``INSTALLABLE = False``, with the reason read off the tree;
* the gate's host legs pass here, every mutation needle resolves exactly once, and the
  lift leg's cell is the seam record's own.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re
import sys

import numpy
import pytest

from .fields import Fields
from .grid import Grid
from .pml import PML

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "triton_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
MODULE = "meep_gpu.triton_kernels.cylindrical_real_fused_hd_pair"
FAMILY = "cylindrical_real_fused_hd_pair"


def _every_cell_record(shape):
    """The harness's every-cell chunk record for ``shape`` (``gate_chunk_records``).

    These builds take their record from the harness rather than from a layer: the
    arrays are bare, and every test here asserts something other than the division.
    """
    parity = str(pathlib.Path(__file__).parent.parent / "parity" / "meep_gpu")
    if parity not in sys.path:
        sys.path.insert(0, parity)
    from gate_chunk_records import every_cell_record  # noqa: PLC0415
    return every_cell_record(shape)


def _chunk_record(arrays):
    """:func:`_every_cell_record`, sized from the first three-axis volume of ``arrays``."""
    shape = next(tuple(a.shape) for a in arrays.values() if getattr(a, "ndim", 0) == 3)
    return _every_cell_record(shape)


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def gate():
    for path in (str(PARITY), str(HERE.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("gate_triton_cylindrical_real_fused_hd_pair")


# ---------------------------------------------------------------------------
# Fixtures -- real Dcyl Grid/Fields/PML on NumPy
# ---------------------------------------------------------------------------

def build(cell=(1.2, 0.0, 1.6), resolution=10.0, courant=0.35, m=0,
          force_complex_fields=False, thickness=0.3):
    grid = Grid(resolution=resolution, cell_size=cell, cylindrical=True, m=m,
                boundaries={"z": "metallic"}, courant=courant)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.set_background_eps(2.25)
    if thickness:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=({"x": (0, thickness), "z": thickness}
                                    if thickness else 0))
    return fields, pml


def _reasons(verdict):
    """The refusal reasons MINUS the array-module clause (a NumPy host always fires it)."""
    return [reason for reason in verdict.reasons if "not cupy" not in reason]


class _ForeignElectric:
    """A source class from outside this package that brings an integrated electric
    ``withdraw`` of its own. No engine source stands one any more (an integrated source
    enters through ``f_minus_p``; ``withdraw_hoist``), and the predicate still refuses a
    class that does, because nothing measured a fused launch beside its body."""

    def withdraw(self, fields):  # never called: a predicate reads a source, it does not run it
        raise AssertionError("a coverage predicate called a source's withdraw")

    field_type = "D"
    component = "Ez"
    is_integrated = True
    _n_source_points = 3
    _applied_dipole = 0j


class _NonIntegratedElectric(_ForeignElectric):
    is_integrated = False


class _Magnetic(_ForeignElectric):
    field_type = "B"
    component = "Hz"


class _View:
    """One object with attributes overridden, everything else forwarded."""

    def __init__(self, inner, **overrides):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_inner"), name)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_the_product_declares_the_seam_it_replaces(product):
    assert product.REPLACES == ("update_H", "step_D")
    assert product.SLOT == "update_H"
    assert product.CONSTITUTIVE_SIDE == "H"
    assert product.CURL_SUB_STEP == "step_D"
    assert product.ARMS == ("cylindrical", "cylindrical PML")
    assert product.BACKWARD == 1
    assert product.ROTATED == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    assert product.IN_PLACE == ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")
    # Launch 1, the certified scan, the certified curl: three kernels, no array-module
    # operation in between.
    assert product.LAUNCHES_PER_RUN == 3
    assert product.KERNEL_LAUNCHES_PER_RUN == 3


def test_replaces_is_the_drivers_own_two_adjacent_consults(product):
    from . import driver as driver_module

    source = inspect.getsource(driver_module.FdtdDriver.step)
    order = [line.split('fast.dispatch("', 1)[1].split('"', 1)[0]
             for line in source.splitlines() if 'fast.dispatch("' in line]
    first = order.index("update_H")
    assert order[first:first + 2] == list(product.REPLACES)


def test_the_only_statement_in_the_seam_is_the_magnetic_f_minus_p_restore(product):
    from . import driver as driver_module

    source = inspect.getsource(driver_module.FdtdDriver.step)
    between = source.split('fast.dispatch("update_H"', 1)[1]
    between = between.split('fast.dispatch("step_D"', 1)[0]
    statements = [line.strip() for line in between.splitlines()
                  if line.strip() and not line.strip().startswith("#")
                  and "update_H(self.fields" not in line]
    # The magnetic f_minus_p restore (sources.f_minus_p_end), which writes B alone and
    # which step_D does not read; no withdraw stands here any more (withdraw_hoist).
    assert any("f_minus_p_end" in line for line in statements), statements
    assert not any("inject" in line or "withdraw" in line for line in statements), statements
    assert product.CARRIES_DEPOSIT_REPAIR is False
    assert product.REPAIR_PATHS == ()


def test_the_seam_name_is_the_withdraw_hoists_and_the_span_is_its_own(product):
    from . import withdraw_hoist

    assert product.SEAM == withdraw_hoist.SEAM
    assert product.REPLACES == withdraw_hoist.SEAM_SPAN


def test_the_two_flags_and_the_wiring_move_together(product):
    """``HOISTS_THE_WITHDRAW`` False is entailed by ``INSTALLABLE`` False, and the reason
    names the label boundary, the cylindrical launch algebra AND its own gate."""
    assert product.INSTALLABLE is False
    assert product.HOISTS_THE_WITHDRAW is False
    assert "4 - (installed pairs)" in product.INSTALLABLE_REASON
    # Three launches against three: the column-serial scan left nothing to fuse.
    assert "THREE device launches" in product.INSTALLABLE_REASON
    assert "in THREE" in product.INSTALLABLE_REASON
    assert "LOSES a launch" in product.INSTALLABLE_REASON
    assert "gate_triton_cylindrical_real_fused_hd_pair.py" in product.INSTALLABLE_REASON
    assert "dispatch_reachability" in product.INSTALLABLE_REASON
    # THE ONE-WAY RULE: nothing in this package may know a dispatcher exists.
    assert "fastpath" not in product.INSTALLABLE_REASON


def test_the_module_is_ascii(product):
    assert pathlib.Path(product.__file__).read_text(encoding="utf-8").isascii()


# ---------------------------------------------------------------------------
# The oracle, read off stepping.py
# ---------------------------------------------------------------------------

def _stepping_text():
    return (HERE / "stepping.py").read_text(encoding="utf-8")


def test_the_prefix_declaration_is_step_Ds_own(product):
    """``Hy`` at ``ir0 = 0.5`` over Dz's chunk starts, fed to ``Dz`` only -- read off
    ``stepping.step_D``, the sum the certified plan's scan computes."""
    from .triton_kernels import cylindrical_triton

    step_d = _stepping_text().split("def step_D(", 1)[1].split("\ndef ", 1)[0]
    assert ('radial = cylindrical_rderiv_difference(\n'
            '            grid.xp, magnetic["Hy"], 0.5,') in step_d
    assert '_cylindrical_scan_starts(fields, pml if pml_active else None, "Dz")' in step_d
    assert 'if cylindrical and term.target == "Dz":' in step_d
    assert product.PREFIX_COMPONENT == "Hy"
    assert product.PREFIX_IR0 == 0.5
    spec = cylindrical_triton.SUB_STEPS["step_D"]
    assert (spec["prefix_component"], spec["prefix_ir0"]) == ("Hy", 0.5)
    assert spec["extend_wall_row"] is False


def test_the_scans_row_vectors_are_the_array_paths_own(product):
    """The scan the plan launches sums with the array path's own row vectors:
    ``radial_rows`` calls ``stepping._cylindrical_rderiv_rows`` at the D side's ``ir0``
    over the D side's rows (no wall row), and the plan's two certified scans hold them."""
    from .stepping import _cylindrical_rderiv_rows
    from .triton_kernels import cylindrical_triton

    weights, rinv = _cylindrical_rderiv_rows(numpy, 6, 0.5, numpy.dtype("float32"))
    host_w, host_rinv = cylindrical_triton.radial_rows("step_D", 6)
    assert numpy.array_equal(host_w, weights.reshape(-1))
    assert numpy.array_equal(host_rinv, rinv.reshape(-1))
    arrays, flat, holder = _arrays(product)
    plan = _plan(product, arrays, flat, holder)
    for curl in plan.step_d_plans:
        assert numpy.array_equal(curl.scan._w, host_w)
        assert numpy.array_equal(curl.scan._rinv, host_rinv)
        assert curl.scan.rows_src == curl.scan.rows_stored == 6


def test_the_row_vectors_are_bound_by_the_certified_plan_not_rebuilt(product):
    """No row vector, starts line or Courant word is built in this module: the certified
    builder binds them, and the module carries none of the array path's prefix names."""
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    for name in ("_cylindrical_rderiv_rows", "radial_rows(", "courant_constants(",
                 "cylindrical_rderiv_prefix(", "cyl_increment", "cyl_prefix",
                 "xp.cumsum"):
        assert name not in text, name
    assert "_cyl.plan_cylindrical_curl_from_arrays(" in text
    assert "_cyl.radial_scan_starts(fields, pml, CURL_SUB_STEP, None)" in text


# ---------------------------------------------------------------------------
# The increment's spelling, statement by statement
# ---------------------------------------------------------------------------

def test_launch_one_is_the_cartesian_welds_constitutive_block(product):
    """Statement for statement, the block ``fused_hd_pair``'s kernel runs between its
    decode and its curl: ``_h_cell`` at the own cell into scratch, the history's
    auxiliary-contract store included."""
    certified = product.certified_weld_block()
    lifted = product.lifted_weld_block()
    assert certified == lifted
    assert len(lifted) >= 10
    assert any("cb, start_h, ny, nz)" in line for line in lifted)
    assert sum(1 for line in lifted if "tl.where(dw" in line) == 3


def test_launch_one_spells_no_arithmetic_of_its_own(product):
    """No divide, no multiply-add, no product: the only arithmetic is the imported
    constitutive's."""
    lifted = product.lifted_weld_block()
    for needle in ("tl.math.", " / ", " * ", " + ("):
        assert not any(needle in line for line in lifted), needle


def test_the_constitutive_is_imported_not_copied(product):
    """The certified body is the Cartesian weld's ``_h_cell``, imported; this module
    defines no copy of it and reads no neighbour."""
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "from .fused_hd_pair import _h_cell  # noqa: PLC0415" in text
    assert "def _h_cell(" not in text
    assert "_h_tap(" not in text


def test_the_constitutive_lift_this_product_launches_still_holds(product):
    from .triton_kernels import fused_hd_pair

    assert fused_hd_pair.certified_constitutive_tail() == fused_hd_pair.lifted_constitutive_tail()


def test_launches_two_and_three_are_the_certified_plan_and_no_text_is_copied(product):
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "plan.run(guard)" in text
    for needle in ("curl0 =", "curl1 =", "curl2 =", "def cyl_pml_curl_step",
                   "def cyl_radial_difference", "total + "):
        assert needle not in text, needle


def test_the_curl_launch_is_the_shipped_plans_argument_list(product):
    """Launches 2 and 3 are the certified ``CylindricalCurlPlan``, built by the certified
    harness builder over each allocation the rotation alternates between: its sources
    are the H of that allocation, its scan sums that allocation's Hy, and its record,
    starts line and Courant words are the ones the weld was handed. The launch's argument
    list is therefore the certified plan's own, never restated here."""
    from .triton_kernels import cylindrical_triton

    arrays, flat, holder = _arrays(product)
    plan = _plan(product, arrays, flat, holder)
    plans = plan.step_d_plans
    assert len(plans) == 2
    for curl in plans:
        assert type(curl) is cylindrical_triton.CylindricalCurlPlan
        assert curl.sub_step == "step_D" and curl.backward == 1
        assert numpy.array_equal(curl.chunk_bounds, plan.chunk_bounds)
        assert numpy.array_equal(curl.scan.starts, numpy.zeros(6, dtype=numpy.int32))
        assert curl.courant == 0.5
    scanned = {curl.scan._source.__array_interface__["data"][0] for curl in plans}
    assert scanned == {arrays["Hy"].__array_interface__["data"][0],
                       arrays["scratch_Hy"].__array_interface__["data"][0]}
    writes, _reads = plan._resolve()
    chosen = plan._curl_plan_for(writes)
    assert chosen.scan._source.__array_interface__["data"][0] == \
        arrays["scratch_Hy"].__array_interface__["data"][0]


def test_the_restated_block_size_is_the_certified_modules_own(product):
    text = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    certified = next(
        node.value.value for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "DEFAULT_BLOCK")
    assert product.DEFAULT_BLOCK == certified


def test_both_halves_take_the_integer_sub_lattice(product):
    curl, constitutive = product._sub_lattice_suffixes()
    assert curl == constitutive == ""


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_an_undeclared_source_list_is_refused_by_name(product):
    fields, pml = build()
    reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(fields, pml, None))
    assert any("was not declared" in reason for reason in reasons), reasons


def test_a_foreign_standing_withdraw_is_refused_by_name(product):
    fields, pml = build()
    reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(
        fields, pml, (_ForeignElectric(),)))
    assert any("standing integrated" in reason for reason in reasons), reasons
    assert any("HOISTS_THE_WITHDRAW = False" in reason for reason in reasons), reasons


def test_an_integrated_electric_source_stands_no_withdraw_in_the_seam(product):
    """An engine integrated source enters at the constitutive update through
    ``f_minus_p`` (``sources.f_minus_p_begin``), so nothing stands between the two
    consults this product spans and the predicate has no withdraw to refuse."""
    from .sources import ContinuousEnvelope, VolumeSource

    fields, pml = build()
    source = VolumeSource(grid=fields.grid, component="Ez", center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
    assert source.is_integrated and not hasattr(source, "withdraw")
    reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(fields, pml, (source,)))
    assert not any("standing integrated" in reason or "withdraw" in reason
                   for reason in reasons), reasons


def test_a_non_integrated_electric_source_and_a_magnetic_one_are_not_refused(product):
    fields, pml = build()
    for source in (_NonIntegratedElectric(), _Magnetic()):
        reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(
            fields, pml, (source,)))
        assert not any("withdraw" in reason for reason in reasons), reasons


def test_on_a_numpy_host_only_the_array_module_clause_refuses_a_healthy_row(product):
    """Everything but ``xp is cupy`` admits the fixture -- the same subtraction every
    Triton laptop suite makes, so the other clauses can be asserted at all."""
    fields, pml = build()
    verdict = product.cylindrical_real_fused_hd_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert _reasons(verdict) == [], _reasons(verdict)


def test_complex_storage_is_refused_by_the_real_family(product):
    fields, pml = build(force_complex_fields=True)
    reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(fields, pml, ()))
    assert any("complex64" in reason for reason in reasons), reasons


def test_m_not_zero_is_refused_by_name(product):
    fields, pml = build(m=1, force_complex_fields=True)
    reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(fields, pml, ()))
    assert any("m = 0 ONLY" in reason for reason in reasons), reasons


def test_a_cartesian_grid_is_refused_by_name(product):
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.2, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic", "z": "periodic"})
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(
        fields, PML(grid=grid, thickness=0.2), ()))
    assert any("not cylindrical" in reason for reason in reasons), reasons


def test_an_inactive_absorber_is_named_by_the_constitutive_half_first(product):
    fields, pml = build(thickness=0)
    verdict = product.cylindrical_real_fused_hd_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert verdict.reasons[0].startswith("cylindrical constitutive half"), verdict.reasons


def test_the_builder_returns_none_rather_than_raising_on_a_refused_run(product):
    fields, pml = build(thickness=0)
    assert product.plan_cylindrical_real_fused_hd_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# The composition
# ---------------------------------------------------------------------------

def test_the_composer_is_offered_this_product_and_refuses_it_by_the_flag(product):
    """ROUTED 2026-09-07. The row, the arm pair and the label are in the shipped
    tables -- the composer's (``CERTIFIED_FUSED_PRODUCTS``, ``CERTIFIED_FUSED_PAIR_ARMS``,
    ``SUPPORT_MODULES``), the dispatcher's (``FUSED_ARM_CONSTITUENTS``,
    ``PENDING_DEVICE_GATE_ARMS``) and the board's join
    (``dispatch_reachability.PRODUCT_ARM_LABELS``) -- and what keeps the product out of
    every slot is its own ``INSTALLABLE = False``, which ``_declared_uninstallable``
    reports by name on every row BEFORE the predicate is asked. Until the routing
    round this test asserted the absence from every one of those tables; the
    tables moved and so did the pin."""
    from .triton_kernels import launch as launch_module
    from . import fastpath, withdraw_hoist

    row = launch_module.CERTIFIED_FUSED_PRODUCTS[FAMILY]
    assert row == {"curl_slot": "update_H", "module": FAMILY,
                   "coverage": "cylindrical_real_fused_hd_pair_coverage",
                   "builder": "plan_cylindrical_real_fused_hd_pair",
                   "label": "fused pair H->D (cylindrical)"}
    assert launch_module.CERTIFIED_FUSED_PAIR_ARMS[FAMILY] == product.ARMS == (
        "cylindrical", "cylindrical PML")
    assert FAMILY in launch_module.SUPPORT_MODULES
    assert launch_module.CERTIFIED_FUSED_PAIR_SEAMS["update_H"] == ("step_D", withdraw_hoist.SEAM)
    label = row["label"]
    assert fastpath.arm_is_fused(label)
    assert fastpath.FUSED_ARM_CONSTITUENTS[label] == product.ARMS
    assert label in fastpath.PENDING_DEVICE_GATE_ARMS and label not in fastpath.ARM_CERTIFICATION
    assert "INSTALLABLE = False" in fastpath.PENDING_DEVICE_GATE_ARMS[label]
    assert label not in fastpath.RELEASED_FUSED_ARMS
    parity_dir = str(pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu")
    if parity_dir not in sys.path:
        sys.path.insert(0, parity_dir)
    import dispatch_reachability as reach  # noqa: PLC0415

    assert reach.PRODUCT_ARM_LABELS[FAMILY] == label
    assert FAMILY not in reach.CERTIFIED_BUT_NOT_INSTALLED
    # The refusal the composer records, read through the seam the composer reads it.
    declared = launch_module._declared_uninstallable({FAMILY: product}, FAMILY, row)
    assert declared is not None and "INSTALLABLE = False" in declared


def test_the_composer_refuses_this_product_on_both_brakes(product):
    from .triton_kernels import launch as launch_module

    selected = {"step_B": "fused pair B (cylindrical)", "update_H": "fused pair B (cylindrical)",
                "step_D": "fused pair D (cylindrical)", "update_E": "fused pair D (cylindrical)"}
    refusal = launch_module._pair_may_absorb(selected, "update_H", "step_D", product.ARMS)
    assert refusal is not None
    assert "'fused pair B (cylindrical)'" in refusal and "'cylindrical'" in refusal
    entry = {"curl_slot": "update_H", "module": FAMILY,
             "coverage": "cylindrical_real_fused_hd_pair_coverage",
             "builder": "plan_cylindrical_real_fused_hd_pair",
             "label": "fused pair H->D (cylindrical)"}
    declared = launch_module._declared_uninstallable({FAMILY: product}, FAMILY, entry)
    assert declared is not None and "INSTALLABLE = False" in declared
    assert product.INSTALLABLE_REASON in declared


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _arrays(product, shape=(6, 1, 5)):
    arrays = {name: numpy.zeros(shape, dtype=numpy.float32)
              for name in product.ROTATED + product.IN_PLACE + product.CONSTITUTIVE_SOURCES}
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


def _plan(product, arrays, flat, holder):
    """The harness route, with the record and the one-chunk starts line it requires."""
    rows = next(a.shape[0] for a in arrays.values() if getattr(a, "ndim", 0) == 3)
    return product.plan_cylindrical_real_fused_hd_pair_from_arrays(
        arrays, flat, 0.35, 0.5, holder, numpy, chunk_bounds=_chunk_record(arrays),
        scan_starts=numpy.zeros(rows, dtype=numpy.int32))


def test_the_plan_rotates_exactly_the_six_magnetic_volumes(product):
    from .triton_kernels.fused_hd_pair import StartsH
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan

    assert issubclass(product.CylindricalRealFusedHdPairPlan, ScratchWeldPairPlan)
    assert issubclass(product.CylindricalRealFusedHdPairPlan, StartsH)
    assert product.CylindricalRealFusedHdPairPlan.replaces == product.REPLACES
    assert product.CylindricalRealFusedHdPairPlan.launches_per_run == 3
    assert product.CylindricalRealFusedHdPairPlan.kernel_launches_per_run == 3


def test_the_plan_binds_the_certified_step_D_plans_and_the_start_of_H(product):
    """The engine route hands the plan MEEP's start-of-H state; the harness route hands
    none (a seeded state has already started)."""
    arrays, flat, holder = _arrays(product)
    plan = _plan(product, arrays, flat, holder)
    assert plan.start_h() == 0
    assert plan.dtdx == 0.35 and plan.courant == 0.5
    assert len({id(curl.scan.out) for curl in plan.step_d_plans}) == 2


def test_a_phi_extent_above_one_is_refused(product):
    arrays, flat, holder = _arrays(product, shape=(6, 2, 5))
    with pytest.raises(ValueError):
        _plan(product, arrays, flat, holder)


def test_an_aliased_scratch_pair_is_refused_rather_than_launched(product):
    arrays, flat, holder = _arrays(product)
    arrays["scratch_Hy"] = arrays["Hy"]
    with pytest.raises((ValueError, RuntimeError)):
        plan = _plan(product, arrays, flat, holder)
        plan._resolve()


def test_an_input_that_aliases_an_output_is_refused(product):
    arrays, flat, holder = _arrays(product)
    arrays["Bx"] = arrays["Dx"]
    with pytest.raises(ValueError):
        _plan(product, arrays, flat, holder)


def test_the_module_is_importable_without_triton(product):
    if product.triton is not None:  # pragma: no cover - a device host
        pytest.skip("Triton is importable here; the absence path is the laptop's")
    with pytest.raises(ImportError):
        product.cyl_real_update_H_scratch_kernel()


# ---------------------------------------------------------------------------
# The gate's own host legs
# ---------------------------------------------------------------------------

def test_the_gates_host_legs_pass_here(gate):
    for leg in (gate.leg_driver_order, gate.leg_transcription, gate.leg_purity_ledger,
                gate.leg_ghost_observability):
        result = leg()
        assert result["passed"], result["findings"]


def test_the_purity_ledger_is_not_vacuous(gate):
    """update_H moves Hy on the rows the scan reads (the floor for the scan-ordering
    plant), and moves a history only where the component's own axis carries a layer."""
    result = gate.leg_purity_ledger()
    for case, ledger in result["fixtures"].items():
        assert ledger["scan_source_words_update_H_moved"] > 0, case
        assert ledger["f_w_H_words_moved_total"] > 0, case
        assert ledger["f_w_H_words_moved"]["f_w_Hy"] == 0, case


def test_every_mutation_needle_resolves_exactly_once(gate, product):
    source = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    for tag, spec in gate.MUTATIONS.items():
        assert spec["why"], tag
        assert spec["case"] in dict(gate.CASES), tag
        if spec["target"] != "kernel":
            continue
        assert source.count(spec["old"]) == 1, tag
    assert source.count(gate.BYTE_NEUTRAL["old"]) == 1
    nulls = [tag for tag, spec in gate.MUTATIONS.items()
             if spec.get("expected_override") == "NULL"]
    assert nulls, "a mutation set with no predicted null has not been thought about"
    for tag in nulls:
        assert ("PREDICTED NULL" in gate.MUTATIONS[tag]["why"]
                or "MEASURED NULL" in gate.MUTATIONS[tag]["why"]), tag
    # THE ARMED DEFECTS the design names are all here, and the scan-ordering plant.
    for tag in ("m_H_written_in_place", "m_f_w_H_written_in_place",
                "b7_fw_scratch_stored_unmasked", "b8_fw_scratch_stored_on_the_layer_only",
                "b11_start_of_H_not_forwarded", "m_rotation_skipped", "m_fp_fusion_on",
                "m_scan_skipped", "m_scan_reads_the_pre_launch_Hy",
                "m_curl_reads_the_pre_launch_H", "m_ir0_zero_ladder"):
        assert tag in gate.MUTATIONS, tag
        assert gate.MUTATIONS[tag]["expected"] == "CAUGHT"
        assert "expected_override" not in gate.MUTATIONS[tag], tag
    assert gate.MUTATIONS["b11_start_of_H_not_forwarded"]["case"] == gate.FIRST_UPDATE_CASE
    assert dict(gate.CASES)[gate.FIRST_UPDATE_CASE]["h_started"] is False


def test_the_kits_ordering_mutations_go_through_the_plans_own_resolver(gate):
    """The scan and curl plants swap the CERTIFIED plan (or its scan) the weld picks, for
    one run, through the plan's own table -- never a second copy of the launch."""
    kit = sys.modules["triton_cylindrical_hd_gate_kit"]
    curl = inspect.getsource(kit.CurlReadsPreLaunchH.run)
    assert "plan._curl_plans" in curl and "plans[key] = wrong" in curl
    scan = inspect.getsource(kit.ScanReadsPreLaunchHy.run)
    assert "right.scan = wrong.scan" in scan
    assert "ArgSwappingKernel" not in inspect.getsource(kit)


def test_the_gate_binds_the_driver_the_composer_and_the_kit(gate):
    for name in ("meep_gpu/driver.py", "meep_gpu/triton_kernels/launch.py",
                 "meep_gpu/triton_kernels/cylindrical_triton.py",
                 "meep_gpu/triton_kernels/fused_hd_pair.py",
                 "meep_gpu/triton_kernels/cylindrical_real_fused_hd_pair.py",
                 "meep_gpu/stepping.py", "meep_gpu/withdraw_hoist.py",
                 "parity/meep_gpu/triton_cylindrical_hd_gate_kit.py"):
        assert name in gate.SOURCES


def test_the_gates_cell_is_the_boards_cell(gate, product):
    assert gate.CELL_ARMS == product.ARMS
    assert gate.SEAM_RECORD == "h_to_d_seam_2026-09-04"
    assert gate.CENSUS == "predicate_coverage_triton_2026-09-04_cylm0"


def test_the_lift_basis_is_the_three_real_rows_the_seam_record_prices(gate):
    results = pathlib.Path(gate.HERE) / "results"
    if not (results / gate.CENSUS).is_dir() or not (results / gate.SEAM_RECORD).is_dir():
        pytest.skip("the standing census or the seam record is not in this checkout")
    rows, facts = gate.lift_basis(results)
    assert facts["rows_in_cell"] == len(rows) == 3
    assert sorted(row["label"] for row in rows) == [
        "tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_0_0",
        "tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_1_0",
        "tests:TestPMLCylindrical.test_pml_cyl_0_0_0"]
    assert facts["rows_with_a_standing_withdraw"] == []


def test_the_kit_keeps_the_one_bar_and_the_floors(gate):
    """The seam claim is the template's conjunction, every arrangement must equal the
    array path over the full budget, and the cylindrical floors hold -- pinned in the
    kit's source so it cannot quietly widen. No leg measures a scan the array module
    performs: the increment and scan-order legs left with the elementwise prefix."""
    kit = sys.modules["triton_cylindrical_hd_gate_kit"]
    source = inspect.getsource(kit.run_product)
    for clause in ("the_weld_at_its_seam_equals_the_two_certified_singles",
                   "the_weld_equals_the_unfused_composition",
                   "the_weld_equals_the_composition_installed_today"):
        assert clause in source, clause
    assert 'and result["bit_identical"]' in source
    assert "stop_when_banded=False" in source
    assert 'floors_hold = floors["radial_difference_is_live"] and (floors["axis_row_is_live"]' in source
    assert 'or floors_at == "end")' in source
    assert "and floors_hold" in source
    assert 'floors_at="end"' in inspect.getsource(kit.evaluate_row)
    assert "increment_stage" not in kit.ALL_LEGS and "scan_order" not in kit.ALL_LEGS
    launch = inspect.getsource(gate.family.CylindricalRealFusedHdPairPlan._launch)
    assert "plan = self._curl_plan_for(writes)" in launch
    assert "plan.run(guard)" in launch


def test_the_kits_scan_plants_are_not_inert(gate):
    """Each scan plant restores what it swapped in a ``finally``, so one row's mutant
    never leaks into the next row's plan."""
    kit = sys.modules["triton_cylindrical_hd_gate_kit"]
    for cls in (kit.CurlReadsPreLaunchH, kit.ScanReadsPreLaunchHy, kit.ScanSkipped,
                kit.Ir0Ladder):
        assert "finally:" in inspect.getsource(cls.run), cls.__name__


