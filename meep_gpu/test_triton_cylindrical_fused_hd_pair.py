"""Laptop contracts for the Triton Dcyl COMPLEX H->D product:
``triton_kernels/cylindrical_fused_hd_pair.py``.

The complex twin of ``test_triton_cylindrical_real_fused_hd_pair.py``. What this suite
adds is the complex storage's one fact: launch 1 is the complex Cartesian weld's
constitutive block, statement for statement, on both word planes, through the IMPORTED
``complex_fused_hd_pair._h_cell_complex`` (the root's per-part forms), and nothing in
this product spells a complex multiply, a multiply-add or a divide -- the pre-scan
increment's FMA_V1 multiply and scaled complex divide left with the elementwise prefix.
Launches 2 and 3 are the certified ``CylindricalComplexCurlPlan``.
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
MODULE = "meep_gpu.triton_kernels.cylindrical_fused_hd_pair"
FAMILY = "cylindrical_fused_hd_pair"


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
def gate(monkeypatch_module):
    for path in (str(PARITY), str(HERE.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("gate_triton_cylindrical_fused_hd_pair")


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


# ---------------------------------------------------------------------------
# Fixtures -- complex Dcyl Grid/Fields/PML on NumPy
# ---------------------------------------------------------------------------

def build(cell=(1.2, 0.0, 1.6), resolution=10.0, courant=0.35, m=1,
          force_complex_fields=True, thickness=0.3, z="metallic"):
    grid = Grid(resolution=resolution, cell_size=cell, cylindrical=True, m=m,
                boundaries={"z": z}, courant=courant)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.set_background_eps(2.25)
    if thickness:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    faces = ({"x": (0, thickness), "z": thickness} if z == "metallic"
             else {"x": (0, thickness)})
    pml = PML(grid=grid, thickness=faces if thickness else 0)
    return fields, pml


def _reasons(verdict):
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


class _View:
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
    assert product.ARMS == ("cylindrical complex", "cylindrical complex PML")
    assert product.BACKWARD == 1
    assert product.ROTATED == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    assert product.IN_PLACE == ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")
    assert product.LAUNCHES_PER_RUN == 3 and product.KERNEL_LAUNCHES_PER_RUN == 3
    assert product.PREFIX_COMPONENT == "Hy" and product.PREFIX_IR0 == 0.5
    assert product.MULTIPLY_HELPERS == ()


def test_the_flags_and_the_reason_are_the_real_twins_with_this_gate_named(product):
    from . import withdraw_hoist

    assert product.INSTALLABLE is False
    assert product.HOISTS_THE_WITHDRAW is False
    assert product.CARRIES_DEPOSIT_REPAIR is False and product.REPAIR_PATHS == ()
    assert product.SEAM == withdraw_hoist.SEAM
    assert "gate_triton_cylindrical_fused_hd_pair.py" in product.INSTALLABLE_REASON
    assert "gate_triton_cylindrical_real_fused_hd_pair.py" not in product.INSTALLABLE_REASON
    assert "fastpath" not in product.INSTALLABLE_REASON
    assert "dispatch_reachability" in product.INSTALLABLE_REASON


def test_the_module_is_ascii(product):
    assert pathlib.Path(product.__file__).read_text(encoding="utf-8").isascii()


def test_the_prefix_declaration_is_the_complex_familys_own(product):
    from .triton_kernels import cylindrical_complex

    spec = cylindrical_complex.PREFIX["step_D"]
    assert (spec["component"], spec["ir0"], spec["extend_wall_row"]) == ("Hy", 0.5, False)
    step_d = (HERE / "stepping.py").read_text(encoding="utf-8").split(
        "def step_D(", 1)[1].split("\ndef ", 1)[0]
    assert ('radial = cylindrical_rderiv_difference(\n'
            '            grid.xp, magnetic["Hy"], 0.5,') in step_d


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def test_the_constitutive_half_is_the_certified_complex_body_plus_the_declared_edits(product):
    """The complex constitutive this product launches is the complex Cartesian weld's
    lifted ``bloch_constitutive_step`` body, imported, not copied: its lift holds, and
    launch 1 is that weld's block, statement for statement."""
    from .triton_kernels import complex_fused_hd_pair

    assert (complex_fused_hd_pair.certified_constitutive_tail()
            == complex_fused_hd_pair.lifted_constitutive_tail())
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "from .complex_fused_hd_pair import _h_cell_complex  # noqa: PLC0415" in text
    for name in ("def _h_cell_complex(", "def _h_tap_complex(", "CONSTITUTIVE_LIFT_EDITS",
                 "def certified_constitutive_tail("):
        assert name not in text, name
    assert product.certified_weld_block() == product.lifted_weld_block()
    assert len(product.lifted_weld_block()) >= 20


def test_launch_one_spells_no_arithmetic_of_its_own(product):
    lifted = product.lifted_weld_block()
    for needle in ("_mul_", "_rotate_", "_div_", "_connect", "tl.math.", " / ", "fma"):
        assert not any(needle in line for line in lifted), needle
    # Both word planes of every scratch volume, and the history's auxiliary contract.
    assert sum(1 for line in lifted if line.strip().startswith("tl.store(ho")) == 6
    assert sum(1 for line in lifted if "tl.where(dw" in line) == 6


# ---------------------------------------------------------------------------
# The two spellings that invert, pinned statement by statement
# ---------------------------------------------------------------------------

def test_the_fma_v1_arrangement_is_the_certified_helpers_own_text(product):
    """No multiply-add survives anywhere in the complex cylindrical H->D path: this
    module spells none, and the certified complex helpers it no longer copies have one
    per-part body each, so no multiply-add branch is left to transcribe."""
    from .triton_kernels import complex_fields

    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "tl.math.fma" not in text and "_mul_field_left_fma" not in text
    certified = pathlib.Path(complex_fields.__file__).read_text(encoding="utf-8")
    helper = certified.split("def _mul_field_left(", 1)[1].split("\n@triton.jit", 1)[0]
    assert "tl.math.fma" not in helper


def test_no_neighbour_is_read_and_no_prefix_name_survives(product):
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    for name in ("_h_tap_complex(", "cylindrical_rderiv_prefix", "cyl_increment",
                 "cyl_prefix", "cumsum", "_cylindrical_rderiv_rows", "imr_rinv_row("):
        assert name not in text, name


def test_launch_two_is_the_certified_complex_curl_with_the_shipped_argument_list(product):
    """Launches 2 and 3 are the certified ``CylindricalComplexCurlPlan``, built by the
    certified harness builder over each rotating allocation; its argument list, i*m/r
    rows, axis rules and Courant words are that plan's own, never restated here."""
    from .triton_kernels import cylindrical_complex

    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "_cx.plan_cylindrical_complex_curl_from_arrays(" in text
    assert "plan.run(guard)" in text
    for needle in ("curl0_re =", "curl2_re =", "def cyl_complex_pml_curl_step",
                   "def cyl_radial_difference"):
        assert needle not in text, needle
    arrays, curl_flat, constitutive_flat, holder = _arrays(product)
    plan = _plan(product, arrays, curl_flat, constitutive_flat, holder, m=2, accurate=True)
    plans = plan.step_d_plans
    assert len(plans) == 2
    for curl in plans:
        assert type(curl) is cylindrical_complex.CylindricalComplexCurlPlan
        assert curl.sub_step == "step_D" and curl.backward == 1
        assert curl.m == 2 and curl.m_class == cylindrical_complex.m_class(2)
        assert curl.zero_rows == cylindrical_complex.zero_rows(2, True) == 1
        assert curl.bcz == 1 and curl.courant == 0.5
        assert numpy.array_equal(curl.chunk_bounds, plan.chunk_bounds)
    scanned = {curl.scan._source.__array_interface__["data"][0] for curl in plans}
    assert scanned == {arrays["Hy"].view(numpy.float32).__array_interface__["data"][0],
                       arrays["scratch_Hy"].view(numpy.float32).__array_interface__["data"][0]}


def test_the_restated_block_size_is_the_certified_modules_own(product):
    text = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    certified = next(
        node.value.value for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "DEFAULT_BLOCK")
    assert product.DEFAULT_BLOCK == certified


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_an_undeclared_source_list_and_a_foreign_standing_withdraw_are_refused_by_name(product):
    fields, pml = build()
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(fields, pml, None))
    assert any("was not declared" in reason for reason in reasons), reasons
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(fields, pml, (_ForeignElectric(),)))
    assert any("standing integrated" in reason for reason in reasons), reasons


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
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(fields, pml, (source,)))
    assert not any("standing integrated" in reason or "withdraw" in reason
                   for reason in reasons), reasons


def test_real_storage_is_refused_by_name(product):
    fields, pml = build(force_complex_fields=False, m=0)
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(fields, pml, ()))
    assert any("force_complex_fields is not set" in reason for reason in reasons), reasons


def test_on_a_numpy_host_with_a_probe_only_the_array_module_clause_refuses(product):
    fields, pml = build()
    verdict = product.cylindrical_fused_hd_pair_coverage(fields, pml, ())
    assert not verdict.covered
    residual = [reason for reason in _reasons(verdict)
                if "subnormal policy" not in reason]
    assert residual == [], residual


@pytest.mark.parametrize("m,z", [(0, "metallic"), (1, "periodic"), (-1, "metallic"),
                                 (2, "metallic"), (3, "periodic")])
def test_every_m_class_and_both_z_terminations_reach_the_predicate(product, m, z):
    fields, pml = build(m=m, z=z, courant=0.25)
    verdict = product.cylindrical_fused_hd_pair_coverage(fields, pml, ())
    residual = [reason for reason in _reasons(verdict) if "subnormal policy" not in reason]
    assert residual == [], residual


def test_an_inactive_absorber_is_named_by_the_constitutive_half_first(product):
    fields, pml = build(thickness=0)
    verdict = product.cylindrical_fused_hd_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert verdict.reasons[0].startswith("cylindrical complex constitutive half"), verdict.reasons


def test_the_builder_returns_none_rather_than_raising_on_a_refused_run(product):
    fields, pml = build(thickness=0)
    assert product.plan_cylindrical_fused_hd_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# The composition
# ---------------------------------------------------------------------------

def test_the_composer_is_offered_this_product_and_refuses_it_by_the_flag(product):
    """ROUTED 2026-09-07, on the real sibling's pin: the row, the arm pair and the
    label are in the composer's, the dispatcher's and the board's tables, and the
    product's own ``INSTALLABLE = False`` is what the composer reports on every row
    before its predicate is asked."""
    from .triton_kernels import launch as launch_module
    from . import fastpath, withdraw_hoist

    row = launch_module.CERTIFIED_FUSED_PRODUCTS[FAMILY]
    assert row == {"curl_slot": "update_H", "module": FAMILY,
                   "coverage": "cylindrical_fused_hd_pair_coverage",
                   "builder": "plan_cylindrical_fused_hd_pair",
                   "label": "fused pair H->D (cylindrical complex)"}
    assert launch_module.CERTIFIED_FUSED_PAIR_ARMS[FAMILY] == product.ARMS == (
        "cylindrical complex", "cylindrical complex PML")
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
    declared = launch_module._declared_uninstallable({FAMILY: product}, FAMILY, row)
    assert declared is not None and "INSTALLABLE = False" in declared


def test_the_composer_refuses_this_product_on_both_brakes(product):
    from .triton_kernels import launch as launch_module

    selected = {"step_B": "fused pair B (cylindrical complex)",
                "update_H": "fused pair B (cylindrical complex)",
                "step_D": "fused pair D (cylindrical complex)",
                "update_E": "fused pair D (cylindrical complex)"}
    refusal = launch_module._pair_may_absorb(selected, "update_H", "step_D", product.ARMS)
    assert refusal is not None and "'cylindrical complex'" in refusal
    entry = {"curl_slot": "update_H", "module": FAMILY,
             "coverage": "cylindrical_fused_hd_pair_coverage",
             "builder": "plan_cylindrical_fused_hd_pair",
             "label": "fused pair H->D (cylindrical complex)"}
    declared = launch_module._declared_uninstallable({FAMILY: product}, FAMILY, entry)
    assert declared is not None and "INSTALLABLE = False" in declared


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _arrays(product, shape=(6, 1, 5)):
    arrays = {name: numpy.zeros(shape, dtype=numpy.complex64)
              for name in product.ROTATED + product.IN_PLACE + product.CONSTITUTIVE_SOURCES}
    for name in product.ROTATED:
        arrays["scratch_" + name] = numpy.zeros(shape, dtype=numpy.complex64)
    curl_flat = {f"{stem}_{axis}": numpy.ones(n, dtype=numpy.float32)
                 for stem in ("kms", "sinv") for axis, n in zip("xyz", shape)}
    constitutive_flat = {f"{stem}_{axis}": numpy.ones(n, dtype=numpy.float32)
                         for stem in ("kps", "kms") for axis, n in zip("xyz", shape)}

    class _Holder:
        pass

    holder = _Holder()
    for name in product.ROTATED:
        setattr(holder, name, arrays[name])
    return arrays, curl_flat, constitutive_flat, holder


def _plan(product, arrays, curl_flat, constitutive_flat, holder, m=1, accurate=False,
          bcz=1):
    """The harness route, with the record and the one-chunk starts line it requires."""
    rows = next(a.shape[0] for a in arrays.values() if getattr(a, "ndim", 0) == 3)
    return product.plan_cylindrical_fused_hd_pair_from_arrays(
        arrays, curl_flat, constitutive_flat, 0.35, 0.5, m, accurate, bcz, holder,
        numpy, chunk_bounds=_chunk_record(arrays),
        scan_starts=numpy.zeros(rows, dtype=numpy.int32))


def test_the_plan_rotates_the_six_volumes_and_binds_the_pool(product):
    """The plan rotates the six H volumes (the real family's choreography and its start
    of H), binds the complex constitutive's kps and kms, and holds one certified step_D
    plan per rotating allocation, each with its own scan output."""
    from .triton_kernels.fused_hd_pair import StartsH
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan

    assert issubclass(product.CylindricalFusedHdPairPlan, ScratchWeldPairPlan)
    assert issubclass(product.CylindricalFusedHdPairPlan, StartsH)
    assert product.CylindricalFusedHdPairPlan.launches_per_run == 3
    assert product.CylindricalFusedHdPairPlan.kernel_launches_per_run == 3
    arrays, curl_flat, constitutive_flat, holder = _arrays(product)
    plan = _plan(product, arrays, curl_flat, constitutive_flat, holder, m=2, accurate=True)
    assert plan.rotated_names == product.ROTATED
    assert len(plan._kps) == 3 and len(plan._kms) == 3
    assert plan.bcz == 1 and plan.m == 2
    assert plan.start_h() == 0
    assert len({id(curl.scan.out) for curl in plan.step_d_plans}) == 2
    writes, _reads = plan._resolve()
    assert plan._curl_plan_for(writes) in plan.step_d_plans


def test_real_storage_a_none_pool_and_an_alias_are_refused(product):
    arrays, curl_flat, constitutive_flat, holder = _arrays(product)
    real = {name: (value.real.astype(numpy.float32) if value.dtype == numpy.complex64
                   else value) for name, value in arrays.items()}
    with pytest.raises(ValueError):
        _plan(product, real, curl_flat, constitutive_flat, holder)
    aliased = dict(arrays)
    aliased["scratch_Hy"] = aliased["Hy"]
    with pytest.raises((ValueError, RuntimeError)):
        _plan(product, aliased, curl_flat, constitutive_flat, holder)._resolve()
    arrays["Bx"] = arrays["Dx"]
    with pytest.raises(ValueError):
        _plan(product, arrays, curl_flat, constitutive_flat, holder)


def test_the_module_is_importable_without_triton(product):
    if product.triton is not None:  # pragma: no cover - a device host
        pytest.skip("Triton is importable here; the absence path is the laptop's")
    with pytest.raises(ImportError):
        product.cyl_complex_update_H_scratch_kernel()


# ---------------------------------------------------------------------------
# The gate's own host legs
# ---------------------------------------------------------------------------

def test_the_gates_host_legs_pass_here(gate):
    for leg in (gate.leg_driver_order, gate.leg_transcription, gate.leg_purity_ledger,
                gate.leg_ghost_observability):
        result = leg()
        assert result["passed"], result["findings"]


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
    for tag in ("m_H_written_in_place", "m_imaginary_plane_takes_B",
                "m_f_w_H_written_in_place", "b7_fw_scratch_stored_unmasked",
                "b8_fw_scratch_stored_on_the_layer_only", "b11_start_of_H_not_forwarded",
                "m_rotation_skipped", "m_scan_skipped", "m_scan_reads_the_pre_launch_Hy",
                "m_ir0_zero_ladder", "m_curl_reads_the_pre_launch_H"):
        assert gate.MUTATIONS[tag]["expected"] == "CAUGHT", tag
        assert "expected_override" not in gate.MUTATIONS[tag], tag
    assert gate.MUTATIONS["b11_start_of_H_not_forwarded"]["case"] == gate.FIRST_UPDATE_CASE
    assert dict(gate.CASES)[gate.FIRST_UPDATE_CASE]["h_started"] is False


def test_the_gate_counts_every_kernel_the_weld_can_launch(gate, product):
    """Launch 1 and BOTH certified step_D plans' scan and curl are counted, through the
    plans' own slots, so the unpicked plan's counters are a measured zero: the gate
    names the five slots, and the kit resolves every dotted slot on a real plan."""
    source = inspect.getsource(gate.Spec.default_kernels)
    for needle in ('slots[f"step_d_plans.{index}._kernel"]',
                   'slots[f"step_d_plans.{index}.scan._kernel"]',
                   "family.cyl_complex_update_H_scratch_kernel()"):
        assert needle in source, needle
    kit = sys.modules["triton_cylindrical_hd_gate_kit"]
    arrays, curl_flat, constitutive_flat, holder = _arrays(product)
    plan = _plan(product, arrays, curl_flat, constitutive_flat, holder)
    for index, curl in enumerate(plan.step_d_plans):
        assert kit._slot_owner(plan, f"step_d_plans.{index}._kernel") == (curl, "_kernel")
        assert kit._slot_owner(plan, f"step_d_plans.{index}.scan._kernel") == (
            curl.scan, "_kernel")


def test_both_policies_reach_the_composer(gate):
    for policy in ("flush", "keep"):
        reachable, why = gate.SPEC.policy_reaches_composer(policy)
        assert reachable, (policy, why)


def test_the_gates_cell_is_the_boards_cell(gate, product):
    assert gate.CELL_ARMS == product.ARMS
    assert gate.SEAM_RECORD == "h_to_d_seam_2026-09-04"


def test_the_lift_basis_is_the_complex_cell_the_seam_record_prices(gate):
    results = pathlib.Path(gate.HERE) / "results"
    if not (results / gate.CENSUS).is_dir() or not (results / gate.SEAM_RECORD).is_dir():
        pytest.skip("the standing census or the seam record is not in this checkout")
    rows, facts = gate.lift_basis(results)
    assert facts["rows_in_cell"] == len(rows) == 17
    assert facts["rows_with_a_standing_withdraw"] == [
        "examples:cylinder_cross_section.py", "examples:zone_plate.py"]
    # The m = 0 COMPLEX row (dipole_in_vacuum_cyl_off_axis) sits in the seam record
    # with NO Triton arms (its board cut predates the m = 0 admission), so it is
    # outside this cell by that record and is named here rather than assumed in.
    assert "examples:dipole_in_vacuum_cyl_off_axis.py" not in {row["label"] for row in rows}
