"""Laptop contracts for the Triton Dcyl m = 0 REAL H->D product:
``triton_kernels/cylindrical_real_fused_hd_pair.py``.

The device gate (``parity/meep_gpu/gate_triton_cylindrical_real_fused_hd_pair.py``) is
what certifies the bytes; this suite is the merge bar. It holds what would otherwise be
true only on a machine nobody runs on the way in:

* the declarations DESCRIBE the seam, the oracle and the wiring rather than request
  them -- the pool tags, the ``ir0`` and the row vectors' key are read off
  ``stepping.py``'s own text, the prefix component off the certified family's table;
* the increment's spelling is pinned statement by statement, and the one spelling this
  backend must NOT use (``/``, which lowers to ``div.full.f32``) is refused by name;
* the constitutive half is IMPORTED from the certified Cartesian weld, whose lift is
  re-asserted here because this product launches it;
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

from .fields import Fields, StepScratch
from .grid import Grid
from .pml import PML

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "triton_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
MODULE = "meep_gpu.triton_kernels.cylindrical_real_fused_hd_pair"
FAMILY = "cylindrical_real_fused_hd_pair"


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


class _Electric:
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
    assert product.LAUNCHES_PER_RUN == 3
    assert product.KERNEL_LAUNCHES_PER_RUN == 2


def test_replaces_is_the_drivers_own_two_adjacent_consults(product):
    from . import driver as driver_module

    source = inspect.getsource(driver_module.FdtdDriver.step)
    order = [line.split('fast.dispatch("', 1)[1].split('"', 1)[0]
             for line in source.splitlines() if 'fast.dispatch("' in line]
    first = order.index("update_H")
    assert order[first:first + 2] == list(product.REPLACES)


def test_the_only_statement_in_the_seam_is_the_electric_withdraw(product):
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
    """``HOISTS_THE_WITHDRAW`` False is entailed by ``INSTALLABLE`` False, and the reason
    names the label boundary, the cylindrical launch algebra AND its own gate."""
    assert product.INSTALLABLE is False
    assert product.HOISTS_THE_WITHDRAW is False
    assert "4 - (installed pairs)" in product.INSTALLABLE_REASON
    assert "SEVEN" in product.INSTALLABLE_REASON and "THREE" in product.INSTALLABLE_REASON
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
    """``Hy`` at ``ir0 = 0.5``, fed to ``Dz`` only -- read off ``stepping.step_D``."""
    from .triton_kernels import cylindrical_triton

    step_d = _stepping_text().split("def step_D(", 1)[1].split("\ndef ", 1)[0]
    assert 'prefixed["Hy"] = cylindrical_rderiv_prefix(grid.xp, magnetic["Hy"], 0.5' in step_d
    assert 'if cylindrical and term.target == "Dz":' in step_d
    assert product.PREFIX_COMPONENT == "Hy"
    assert product.PREFIX_IR0 == 0.5
    spec = cylindrical_triton.SUB_STEPS["step_D"]
    assert (spec["prefix_component"], spec["prefix_ir0"]) == ("Hy", 0.5)
    assert spec["extend_wall_row"] is False


def test_the_pool_tags_and_the_row_vector_key_are_the_oracles_own(product):
    """The plan takes the SAME pooled slots and the SAME cached constants the shipped
    ``cylindrical_rderiv_prefix`` uses, so the scan writes where the oracle writes."""
    scan = _stepping_text().split("def cylindrical_rderiv_prefix(", 1)[1].split("\ndef ", 1)[0]
    assert f'scratch.take("{product.INCREMENT_TAG}"' in scan
    assert f'scratch.take("{product.PREFIX_TAG}"' in scan
    assert 'key = ("cyl_rderiv", int(f_p.shape[0]), float(ir0), real_dtype)' in scan
    assert product.rderiv_constant_key(12, numpy.dtype("float32")) == (
        "cyl_rderiv", 12, 0.5, numpy.dtype("float32"))
    # The four elementwise statements the kernel replaces, in the oracle's order.
    order = [scan.index(needle) for needle in (
        "xp.multiply(f_p, weights, out=weighted)",
        "increment[_face(0, 0)] = 0",
        "xp.subtract(weighted[1:], weighted[:-1], out=increment[1:])",
        "increment[1:] /= divisor",
        "xp.cumsum(increment, axis=0,")]
    assert order == sorted(order)


def test_the_row_vectors_are_bound_from_the_pool_not_rebuilt(product):
    scratch = StepScratch(numpy)
    from .stepping import _cylindrical_rderiv_weights

    expected = scratch.constant(product.rderiv_constant_key(9, numpy.dtype("float32")),
                                lambda: _cylindrical_rderiv_weights(numpy, 9, 0.5,
                                                                    numpy.dtype("float32")))
    weights, divisor = product.rderiv_vectors(numpy, scratch, 9, numpy.dtype("float32"))
    assert weights is expected[0] and divisor is expected[1]
    assert weights.shape == (9, 1, 1) and divisor.shape == (8, 1, 1)
    # The D-side ladder: weights (ir + 0.5), divisor ir for ir = 1..nr-1.
    assert numpy.array_equal(weights.reshape(-1), (numpy.arange(9) + 0.5).astype(numpy.float32))
    assert numpy.array_equal(divisor.reshape(-1), numpy.arange(1, 9).astype(numpy.float32))


# ---------------------------------------------------------------------------
# The increment's spelling, statement by statement
# ---------------------------------------------------------------------------

def test_every_increment_statement_appears_exactly_once_in_the_kernel(product):
    statements = product.increment_statements()
    assert len(statements) >= 30
    for tag, needle in product.INCREMENT_SPELLING:
        assert sum(1 for line in statements if needle in line) == 1, tag


def test_the_divide_is_div_rn_and_never_slash(product):
    """Triton's fp32 ``/`` is ``div.full.f32``; the oracle's ``true_divide`` is correctly
    rounded. The correctly rounded intrinsic is the spelling and ``/`` is a mutation."""
    statements = product.increment_statements()
    assert any("value = tl.math.div_rn(diff, d_im1)" in line for line in statements)
    divisions = [line for line in statements
                 if " = " in line and " / " in line.split("#", 1)[0] and "//" not in line]
    assert divisions == [], divisions
    assert "* (1.0 / " not in "".join(statements)


def test_the_field_is_left_of_the_weight_and_the_axis_row_is_an_exact_zero(product):
    statements = product.increment_statements()
    assert "weighted_here = own1 * w_i" in statements
    assert "weighted_below = halo * w_im1" in statements
    assert "value = tl.where(inner, value, 0.0)" in statements
    assert "inner = live & (i >= 1)" in statements


def test_the_halo_is_a_recompute_through_the_certified_body(product):
    """The one foreign value is ``_h_tap`` at ``i - 1`` under the ``inner`` guard, never a
    load of the stored Hy and never a load of the scratch."""
    statements = product.increment_statements()
    joined = "\n".join(statements)
    assert "halo = _h_tap(1, i - 1, j, k, inner," in joined
    assert "tl.load(hi1 + idx - nyz" not in joined
    assert "tl.load(ho1 + idx - nyz" not in joined
    # ...and the constitutive half is the certified body, imported, not copied.
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "from .fused_hd_pair import _h_cell, _h_tap" in text
    assert "def _h_cell(" not in text


def test_the_constitutive_lift_this_product_launches_still_holds(product):
    from .triton_kernels import fused_hd_pair

    assert fused_hd_pair.certified_constitutive_tail() == fused_hd_pair.lifted_constitutive_tail()


def test_launch_two_is_the_certified_curl_object_and_no_curl_text_is_copied(product):
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "_cyl.cylindrical_curl_kernel()" in text
    for needle in ("curl0 =", "curl1 =", "curl2 =", "def cyl_pml_curl_step"):
        assert needle not in text, needle


def test_the_curl_launch_is_the_shipped_plans_argument_list(product):
    """Launch 2's argument order is ``cyl_pml_curl_step``'s signature, read off the
    certified module rather than trusted: targets, auxiliaries, the three stepped H,
    the prefix, Hp for the axis add, the six coefficients, the scalars."""
    from .triton_kernels import cylindrical_triton

    tree = ast.parse(pathlib.Path(cylindrical_triton.__file__).read_text(encoding="utf-8"))
    kernel = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "cyl_pml_curl_step")
    names = [arg.arg for arg in kernel.args.args]
    assert names[:11] == ["f0", "f1", "f2", "u0", "u1", "u2", "g0", "g1", "g2", "pfx", "hp"]
    assert names[11:17] == ["kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz"]
    assert names[17:23] == ["nx", "ny", "nz", "n_elem", "dtdx", "four_dtdx"]
    launch = inspect.getsource(product.CylindricalRealFusedHdPairPlan._launch)
    assert "*self._targets, *self._aux," in launch
    assert "scratch[0], scratch[1], scratch[2]," in launch
    assert "pointer(prefix), scratch[1]," in launch
    assert "nx, ny, nz, self.n_elem, self.dtdx, self.four_dtdx," in launch
    assert "BACKWARD=self.backward," in launch


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


def test_a_standing_integrated_electric_withdraw_is_refused_by_name(product):
    fields, pml = build()
    reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(
        fields, pml, (_Electric(),)))
    assert any("standing integrated" in reason for reason in reasons), reasons
    assert any("HOISTS_THE_WITHDRAW = False" in reason for reason in reasons), reasons


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


def test_a_missing_step_scratch_is_refused_by_name(product):
    fields, pml = build()
    reasons = _reasons(product.cylindrical_real_fused_hd_pair_coverage(
        _View(fields, scratch=None), pml, ()))
    assert any("StepScratch" in reason for reason in reasons), reasons


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


def test_the_plan_rotates_exactly_the_six_magnetic_volumes(product):
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan

    assert issubclass(product.CylindricalRealFusedHdPairPlan, ScratchWeldPairPlan)
    assert product.CylindricalRealFusedHdPairPlan.replaces == product.REPLACES
    assert product.CylindricalRealFusedHdPairPlan.launches_per_run == 3
    assert product.CylindricalRealFusedHdPairPlan.kernel_launches_per_run == 2


def test_the_plan_binds_the_pools_own_slots(product):
    arrays, flat, holder = _arrays(product)
    scratch = StepScratch(numpy)
    plan = product.plan_cylindrical_real_fused_hd_pair_from_arrays(
        arrays, flat, 0.35, holder, numpy, scratch)
    increment = plan._increment()
    assert increment is scratch.take("cyl_increment", (6, 1, 5), numpy.dtype("float32"))
    assert plan._prefix_out() is scratch.take("cyl_prefix", (6, 1, 5), numpy.dtype("float32"))
    assert plan._weights.shape == (6, 1, 1) and plan._divisor.shape == (5, 1, 1)
    assert plan.four_dtdx == float(numpy.float32(4.0 * 0.35))


def test_a_none_pool_and_a_phi_extent_above_one_are_refused(product):
    arrays, flat, holder = _arrays(product)
    with pytest.raises(ValueError):
        product.plan_cylindrical_real_fused_hd_pair_from_arrays(arrays, flat, 0.35, holder,
                                                                numpy, None)
    arrays2, flat2, holder2 = _arrays(product, shape=(6, 2, 5))
    with pytest.raises(ValueError):
        product.plan_cylindrical_real_fused_hd_pair_from_arrays(arrays2, flat2, 0.35, holder2,
                                                                numpy, StepScratch(numpy))


def test_an_aliased_scratch_pair_is_refused_rather_than_launched(product):
    arrays, flat, holder = _arrays(product)
    arrays["scratch_Hy"] = arrays["Hy"]
    with pytest.raises((ValueError, RuntimeError)):
        plan = product.plan_cylindrical_real_fused_hd_pair_from_arrays(
            arrays, flat, 0.35, holder, numpy, StepScratch(numpy))
        plan._resolve()


def test_an_input_that_aliases_an_output_is_refused(product):
    arrays, flat, holder = _arrays(product)
    arrays["Bx"] = arrays["Dx"]
    with pytest.raises(ValueError):
        product.plan_cylindrical_real_fused_hd_pair_from_arrays(arrays, flat, 0.35, holder,
                                                                numpy, StepScratch(numpy))


def test_the_module_is_importable_without_triton(product):
    if product.triton is not None:  # pragma: no cover - a device host
        pytest.skip("Triton is importable here; the absence path is the laptop's")
    with pytest.raises(ImportError):
        product.cyl_real_update_H_increment_kernel()


# ---------------------------------------------------------------------------
# The gate's own host legs
# ---------------------------------------------------------------------------

def test_the_gates_host_legs_pass_here(gate):
    for leg in (gate.leg_driver_order, gate.leg_transcription, gate.leg_purity_ledger,
                gate.leg_ghost_observability):
        result = leg()
        assert result["passed"], result["findings"]


def test_the_purity_ledger_is_not_vacuous(gate):
    result = gate.leg_purity_ledger()
    for case, ledger in result["fixtures"].items():
        assert ledger["halo_source_words_update_H_moved"] > 0, case
        assert ledger["f_w_Hy_words_moved"] > 0, case


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
    # THE ARMED SPELLINGS the design names are all here, and the stored-halo plant.
    for tag in ("m_divide_spelled_with_slash", "m_reciprocal_multiply", "m_fp_fusion_on",
                "m_subtract_reversed", "m_ir0_zero_ladder", "m_halo_reads_the_stored_Hy",
                "m_rotation_skipped", "m_curl_reads_the_pre_launch_H", "m_scan_skipped"):
        assert tag in gate.MUTATIONS, tag
        assert gate.MUTATIONS[tag]["expected"] == "CAUGHT"
        assert "expected_override" not in gate.MUTATIONS[tag], tag
    # The scratch-halo read is a MEASURED null on a one-program fixture and says so.
    assert gate.MUTATIONS["m_halo_reads_the_neighbours_scratch"]["expected_override"] == "NULL"
    assert dict(gate.CASES)[gate.MUTATION_CASE]["shape"][0] * dict(gate.CASES)[gate.MUTATION_CASE]["shape"][1] <= 256


def test_the_increment_variants_carry_the_kernels_own_spelling(gate, product):
    """The ``primary`` variant IS the launch-1 statements (multiply field-left, subtract,
    ``div_rn``), so the increment leg measures the product's arithmetic, not a look-alike."""
    primary = gate.INCREMENT_VARIANTS["primary"]
    assert "tl.math.div_rn(diff, d_im1)" in primary["divide"]
    assert "f_i * w_i" in primary["multiply"] and "f_im1 * w_im1" in primary["multiply"]
    assert primary["expected"] == "IDENTICAL" and primary["also_fusion_on"]
    assert gate.INCREMENT_VARIANTS["slash_div_full"]["divide"] == "value = diff / d_im1"
    for name in ("slash_div_full", "reciprocal_multiply", "distributed_divide"):
        assert gate.INCREMENT_VARIANTS[name]["expected"] == "DIFFERS", name
    assert gate.INCREMENT_VARIANTS["swapped_multiply_operands"]["expected"] == "IDENTICAL"


def test_the_curl_redirect_names_every_magnetic_argument_of_launch_two(gate):
    """``cyl_pml_curl_step(f0..u2, g0, g1, g2, pfx, hp, ...)``: g0..g2 at 6..8 and Hp at 10."""
    assert gate.SPEC.curl_prelaunch_positions() == {6: 0, 7: 1, 8: 2, 10: 1}


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


def test_the_kit_keeps_the_one_bar_and_the_scan_on_the_array_path(gate):
    """The seam claim is the template's conjunction, every arrangement must equal the
    array path over the full budget, and the product leaves ``cumsum`` to the array
    module -- pinned in the kit's source so it cannot quietly widen."""
    kit = sys.modules["triton_cylindrical_hd_gate_kit"]
    source = inspect.getsource(kit.run_product)
    for clause in ("the_weld_at_its_seam_equals_the_two_certified_singles",
                   "the_weld_equals_the_unfused_composition",
                   "the_weld_equals_the_composition_installed_today"):
        assert clause in source, clause
    assert 'and result["bit_identical"]' in source
    assert "stop_when_banded=False" in source
    # Both floors on the seeded fixture; on a lifted row (floors measured on the array
    # path's FINAL state, because the seed is the engine's zeros) the prefix must be
    # live and the axis row's liveness is recorded -- a fact about the row.
    assert 'floors_hold = floors["prefix_is_live"] and (floors["axis_row_is_live"]' in source
    assert 'or floors_at == "end")' in source
    assert "and floors_hold" in source
    assert 'floors_at="end"' in inspect.getsource(kit.evaluate_row)
    scan = inspect.getsource(kit.leg_scan_order)
    assert "serial_device_vs_cupy_cumsum_differing" in scan
    assert "serial_device_vs_numpy_cumsum_differing" in scan
    launch = inspect.getsource(gate.family.CylindricalRealFusedHdPairPlan._launch)
    assert "self.xp.cumsum(increment, axis=0, out=self._prefix_out())" in launch


def test_the_kits_curl_redirect_is_not_inert(gate):
    """An earlier kit set an attribute nothing read; the redirect now installs an
    argument-swapping kernel in the plan's own ``_curl_kernel`` seam and asserts it
    launched exactly once."""
    kit = sys.modules["triton_cylindrical_hd_gate_kit"]
    source = inspect.getsource(kit.CurlReadsPreLaunchH.run)
    assert "ArgSwappingKernel" in source
    assert "plan._curl_kernel = swapper" in source
    assert "swapper.calls != 1" in source
    assert "_launch_curl_sources" not in inspect.getsource(kit)
