"""Laptop contracts for the Triton Dcyl COMPLEX H->D product:
``triton_kernels/cylindrical_fused_hd_pair.py``.

The complex twin of ``test_triton_cylindrical_real_fused_hd_pair.py``. What this suite
adds is the one thing the complex family gets wrong if it copies either neighbour: the
increment's multiply is the FMA_V1 ``mul_field_left`` arrangement with EXPLICIT fused
multiply-adds and the divide is CuPy's SCALED complex division with every zero-valued
term kept and both reciprocals correctly rounded -- neither numpy's ``a * (1/d)`` (the
sibling backend's verdict), nor the real family's componentwise ``/``, nor the
certified constitutive's own arm-gated helper. Each is pinned by its exact statements,
and the lift of the constitutive half is asserted byte for byte.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys

import numpy
import pytest

from .fields import Fields, StepScratch
from .grid import Grid
from .pml import PML

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "triton_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
MODULE = "meep_gpu.triton_kernels.cylindrical_fused_hd_pair"
FAMILY = "cylindrical_fused_hd_pair"
PROBE = PARITY / "results" / "expansion_probe_2026-08-17" / "expansion_probe_keep.json"


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
    if PROBE.is_file():
        patch.setenv("MEEP_GPU_COMPLEX_EXPANSION_PROBE", str(PROBE))
    yield patch
    patch.undo()


@pytest.fixture
def probe():
    if not PROBE.is_file():
        pytest.skip("the keep-cut expansion probe record is not in this checkout")
    import json

    return json.loads(PROBE.read_text(encoding="utf-8"))


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


class _Electric:
    from . import sources as _sources_module

    withdraw = _sources_module.VolumeSource.withdraw
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
    assert product.LAUNCHES_PER_RUN == 3 and product.KERNEL_LAUNCHES_PER_RUN == 2
    assert product.PREFIX_COMPONENT == "Hy" and product.PREFIX_IR0 == 0.5
    assert product.INCREMENT_TAG == "cyl_increment" and product.PREFIX_TAG == "cyl_prefix"


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
    assert 'prefixed["Hy"] = cylindrical_rderiv_prefix(grid.xp, magnetic["Hy"], 0.5' in step_d


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def test_the_constitutive_half_is_the_certified_complex_body_plus_the_declared_edits(product):
    certified = product.certified_constitutive_tail()
    lifted = product.lifted_constitutive_tail()
    assert certified == lifted
    statements = [line for line in certified.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    assert len(statements) >= 40, len(statements)
    # Every accumulation is the certified helper under the run's arm, in the certified
    # order (kps*src added, then kms*prev subtracted), never regrouped.
    assert certified.count("_mul_coefficient_left(kp_") == 3
    assert certified.count("_mul_coefficient_left(km_") == 3
    assert "a_re = a_re + t_re" in certified and "a_re = a_re - t_re" in certified


def test_no_certified_pointer_or_branch_survives_in_the_lift(product):
    lifted = product.lifted_constitutive_tail()
    for stem in ("f", "w", "g", "e"):
        for target in range(3):
            assert f"{stem}{target} +" not in lifted, f"{stem}{target}"
    assert "SCALE" not in lifted and "tl.store(" not in lifted


def test_every_declared_lift_edit_carries_a_reason_and_none_is_arithmetic(product):
    assert len(product.CONSTITUTIVE_LIFT_EDITS) >= 6
    for edit in product.CONSTITUTIVE_LIFT_EDITS:
        assert edit["line"] and edit["became"] and edit["why"]
        assert "*" not in edit["became"].replace("2 * idx", "").replace("...", "")


def test_a_drifted_certified_spelling_raises_rather_than_lifting(product):
    from .triton_kernels import fused_hd_pair

    tail = product.function_source("bloch_constitutive_step")
    with pytest.raises(AssertionError):
        fused_hd_pair.needle(tail, "this needle matches nothing", "x")
    with pytest.raises(AssertionError):
        fused_hd_pair.needle(tail, "mask=live, other=0.0", "x")


# ---------------------------------------------------------------------------
# The two spellings that invert, pinned statement by statement
# ---------------------------------------------------------------------------

def test_the_divide_is_cupys_scaled_algorithm_with_the_zero_terms_kept(product):
    statements = product.increment_statements()
    for needle in product.DIVIDE_SPELLING:
        assert statements.count(needle) == 1, needle
    assert "q_re, q_im = _div_coefficient_scaled(d_re, d_im, d_im1)" in statements
    # Both reciprocals correctly rounded, and NO `/` anywhere below the decode.
    assert statements.count("oos = tl.math.div_rn(1.0, s)") == 1
    assert statements.count("oos2 = tl.math.div_rn(1.0, s2)") == 1
    divisions = [line for line in statements
                 if " = " in line and " / " in line.split("#", 1)[0] and "//" not in line]
    assert divisions == [], divisions
    # NOT numpy's reciprocal multiply (the Metal verdict) and NOT componentwise.
    joined = "\n".join(statements)
    assert "_mul_field_left_fma(d_re, d_im" not in joined
    assert "tl.math.div_rn(d_re, d_im1)" not in joined


def test_the_multiply_is_the_fma_v1_field_left_arrangement_ungated_by_the_arm(product):
    statements = product.increment_statements()
    for needle in product.MULTIPLY_SPELLING:
        assert statements.count(needle) == 1, needle
    assert "wh_re, wh_im = _mul_field_left_fma(o1_re, o1_im, w_i)" in statements
    assert "wb_re, wb_im = _mul_field_left_fma(h_re, h_im, w_im1)" in statements
    # The arm-gated certified helper is NOT reached by the increment.
    assert not any("_mul_field_left(" in line and "_mul_field_left_fma(" not in line
                   for line in statements)
    # Negation is `* -1.0`, never unary minus (complex_fields' own rule).
    source = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    helper = source.split("def _mul_field_left_fma(", 1)[1].split("@triton.jit", 1)[0]
    assert "* -1.0" in helper and "(-" not in helper.split('"""')[-1]


def test_the_fma_v1_arrangement_is_the_certified_helpers_own_text(product):
    """``_mul_field_left_fma`` spells exactly ``complex_fields._mul_field_left``'s FMA_V1
    branch -- read off that module's source so the two cannot drift."""
    from .triton_kernels import complex_fields

    text = pathlib.Path(complex_fields.__file__).read_text(encoding="utf-8")
    branch = text.split("def _mul_field_left(", 1)[1].split("if EXPANSION == FMA_V1:", 1)[1]
    branch = branch.split("else:", 1)[0]
    for needle in product.MULTIPLY_SPELLING:
        assert needle in branch, needle


def test_every_increment_statement_appears_exactly_once(product):
    statements = product.increment_statements()
    for tag, needle in product.INCREMENT_SPELLING:
        assert sum(1 for line in statements if needle in line) == 1, tag
    assert "q_re = tl.where(inner, q_re, 0.0)" in statements
    assert "q_im = tl.where(inner, q_im, 0.0)" in statements
    assert "d_re = wh_re - wb_re" in statements and "d_im = wh_im - wb_im" in statements


def test_the_halo_is_a_recompute_through_the_lifted_body(product):
    joined = "\n".join(product.increment_statements())
    assert "h_re, h_im = _h_tap_complex(1, i - 1, j, k, inner," in joined
    assert "tl.load(hi1 + 2 * (idx - nyz)" not in joined
    assert "tl.load(ho1 + 2 * (idx - nyz)" not in joined


def test_launch_two_is_the_certified_complex_curl_with_the_shipped_argument_list(product):
    from .triton_kernels import cylindrical_complex

    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    assert "_cx.cyl_complex_pml_curl_step" in text
    for needle in ("curl0_re =", "curl2_re =", "def cyl_complex_pml_curl_step"):
        assert needle not in text, needle
    tree = ast.parse(pathlib.Path(cylindrical_complex.__file__).read_text(encoding="utf-8"))
    kernel = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "cyl_complex_pml_curl_step")
    names = [arg.arg for arg in kernel.args.args]
    assert names[:12] == ["f0", "f1", "f2", "u0", "u1", "u2", "g0", "g1", "g2", "pfx",
                          "c0", "c2"]
    assert names[18:27] == ["nx", "ny", "nz", "n_elem", "dtdx", "minus_dtdx", "inc_b_re",
                            "inc_b_im", "four_dtdx"]
    launch = inspect.getsource(product.CylindricalFusedHdPairPlan._launch)
    assert "pointer(_word_view(prefix)), *self._imr_rows," in launch
    assert "minus_dtdx, inc_b[0], inc_b[1], self.four_dtdx," in launch
    for constexpr in ("BACKWARD=self.backward", "BCZ=self.bcz", "M_CLASS=self.m_class",
                      "ZERO_ROWS=self.zero_rows", "EXPANSION=self.expansion"):
        assert constexpr in launch, constexpr


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

def test_an_undeclared_source_list_and_a_standing_withdraw_are_refused_by_name(product):
    fields, pml = build()
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(fields, pml, None))
    assert any("was not declared" in reason for reason in reasons), reasons
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(fields, pml, (_Electric(),)))
    assert any("standing integrated" in reason for reason in reasons), reasons


def test_real_storage_is_refused_by_name(product):
    fields, pml = build(force_complex_fields=False, m=0)
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(fields, pml, ()))
    assert any("force_complex_fields is not set" in reason for reason in reasons), reasons


def test_a_missing_probe_is_refused_by_name(product, monkeypatch):
    monkeypatch.delenv("MEEP_GPU_COMPLEX_EXPANSION_PROBE", raising=False)
    fields, pml = build()
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(fields, pml, ()))
    assert any("expansion probe" in reason for reason in reasons), reasons


def test_on_a_numpy_host_with_a_probe_only_the_array_module_clause_refuses(product, probe):
    fields, pml = build()
    verdict = product.cylindrical_fused_hd_pair_coverage(fields, pml, (), probe=probe)
    assert not verdict.covered
    residual = [reason for reason in _reasons(verdict)
                if "subnormal policy" not in reason]
    assert residual == [], residual


@pytest.mark.parametrize("m,z", [(0, "metallic"), (1, "periodic"), (-1, "metallic"),
                                 (2, "metallic"), (3, "periodic")])
def test_every_m_class_and_both_z_terminations_reach_the_predicate(product, probe, m, z):
    fields, pml = build(m=m, z=z, courant=0.25)
    verdict = product.cylindrical_fused_hd_pair_coverage(fields, pml, (), probe=probe)
    residual = [reason for reason in _reasons(verdict) if "subnormal policy" not in reason]
    assert residual == [], residual


def test_an_inactive_absorber_is_named_by_the_constitutive_half_first(product):
    fields, pml = build(thickness=0)
    verdict = product.cylindrical_fused_hd_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert verdict.reasons[0].startswith("cylindrical complex constitutive half"), verdict.reasons


def test_a_missing_step_scratch_is_refused_by_name(product):
    fields, pml = build()
    reasons = _reasons(product.cylindrical_fused_hd_pair_coverage(
        _View(fields, scratch=None), pml, ()))
    assert any("StepScratch" in reason for reason in reasons), reasons


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


def _plan(product, arrays, curl_flat, constitutive_flat, holder, scratch, m=1, accurate=False,
          bcz=1):
    return product.plan_cylindrical_fused_hd_pair_from_arrays(
        arrays, curl_flat, constitutive_flat, 0.35, m, accurate, bcz, 1, holder, numpy,
        scratch)


def test_the_plan_rotates_the_six_volumes_and_binds_the_pool(product):
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan
    from .triton_kernels import cylindrical_complex

    assert issubclass(product.CylindricalFusedHdPairPlan, ScratchWeldPairPlan)
    arrays, curl_flat, constitutive_flat, holder = _arrays(product)
    scratch = StepScratch(numpy)
    plan = _plan(product, arrays, curl_flat, constitutive_flat, holder, scratch, m=2,
                 accurate=True)
    assert plan._increment() is scratch.take("cyl_increment", (6, 1, 5),
                                             numpy.dtype("complex64"))
    assert plan.m_class == cylindrical_complex.m_class(2)
    assert plan.zero_rows == cylindrical_complex.zero_rows(2, True) == 1
    assert plan.expansion == 1 and plan.bcz == 1
    assert plan.four_dtdx == cylindrical_complex.four_dtdx_scalar(0.35)
    assert plan.increment_scalars == cylindrical_complex.axis_increment_scalars(2, 0.35)
    assert len(plan._imr_rows) == 2 and len(plan._h_coeff) == 6


def test_real_storage_a_none_pool_and_an_alias_are_refused(product):
    arrays, curl_flat, constitutive_flat, holder = _arrays(product)
    with pytest.raises(ValueError):
        _plan(product, arrays, curl_flat, constitutive_flat, holder, None)
    real = {name: (value.real.astype(numpy.float32) if value.dtype == numpy.complex64
                   else value) for name, value in arrays.items()}
    with pytest.raises(ValueError):
        _plan(product, real, curl_flat, constitutive_flat, holder, StepScratch(numpy))
    arrays["Bx"] = arrays["Dx"]
    with pytest.raises(ValueError):
        _plan(product, arrays, curl_flat, constitutive_flat, holder, StepScratch(numpy))


def test_the_module_is_importable_without_triton(product):
    if product.triton is not None:  # pragma: no cover - a device host
        pytest.skip("Triton is importable here; the absence path is the laptop's")
    with pytest.raises(ImportError):
        product.cyl_complex_update_H_increment_kernel()


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
    for tag in nulls:
        assert ("PREDICTED NULL" in gate.MUTATIONS[tag]["why"]
                or "MEASURED NULL" in gate.MUTATIONS[tag]["why"]), tag
    for tag in ("m_numpy_reciprocal_divide", "m_componentwise_divide",
                "m_first_reciprocal_spelled_with_slash", "m_halo_reads_the_stored_Hy",
                "m_rotation_skipped", "m_scan_skipped", "m_ir0_zero_ladder",
                "m_curl_reads_the_pre_launch_H"):
        assert gate.MUTATIONS[tag]["expected"] == "CAUGHT", tag
        assert "expected_override" not in gate.MUTATIONS[tag], tag
    # numpy's reciprocal coincides with the scaled divide wherever d * (1/d) rounds to
    # exactly 1.0 -- every ladder value below 41 -- so the mutation case must reach it.
    shape = dict(gate.CASES)[gate.MUTATION_CASE]["shape"]
    assert shape[0] > 41 and shape[0] * shape[1] <= 256, shape
    assert gate.MUTATIONS["m_halo_reads_the_neighbours_scratch"]["expected_override"] == "NULL"
    # The four-product multiply is a MEASURED null inside the increment (its signed-zero
    # difference cannot reach the increment through the subtract and the scaled divide,
    # 0 of 8,991,940 words under both policies) and is pinned where it IS byte-visible:
    # the multiply sub-leg's planted row-0 class under flush.
    assert gate.MUTATIONS["m_four_product_multiply_in_the_increment"]["expected_override"] == "NULL"
    assert "MEASURED NULL" in gate.MUTATIONS["m_four_product_multiply_in_the_increment"]["why"]
    assert gate.INCREMENT_VARIANTS["four_product_multiply"]["expected"] == "IDENTICAL"
    assert gate.MULTIPLY_VARIANTS["four_product"]["expected"]["flush"]["planted"] == "DIFFERS"
    assert gate.MULTIPLY_VARIANTS["four_product"]["expected"]["keep"] == "IDENTICAL"


def test_the_increment_variants_carry_the_kernels_own_spelling(gate, product):
    primary = gate.INCREMENT_VARIANTS["primary"]
    for needle in ("tl.math.fma(fr_i, w_i, (fi_i * 0.0) * -1.0)",
                   "tl.math.fma(fr_i, 0.0, fi_i * w_i)"):
        assert needle in primary["multiply"], needle
    for needle in ("s = tl.abs(d_im1) + 0.0", "oos = tl.math.div_rn(1.0, s)",
                   "bis = 0.0 * oos", "s2 = (brs * brs) + (bis * bis)",
                   "oos2 = tl.math.div_rn(1.0, s2)",
                   "q_re = ((ars * brs) + (ais * bis)) * oos2",
                   "q_im = ((ais * brs) - (ars * bis)) * oos2"):
        assert needle in primary["divide"], needle
    for name in ("numpy_reciprocal", "componentwise"):
        assert gate.INCREMENT_VARIANTS[name]["expected"] == "DIFFERS", name


def test_the_curl_redirect_names_the_three_magnetic_arguments_of_launch_two(gate):
    assert gate.SPEC.curl_prelaunch_positions() == {6: 0, 7: 1, 8: 2}


def test_the_flush_policy_does_not_reach_the_composer_and_keep_does(gate, probe):
    reachable, why = gate.SPEC.policy_reaches_composer("flush")
    assert not reachable and "CERTIFIED under the 'keep'" in why
    reachable, why = gate.SPEC.policy_reaches_composer("keep")
    assert reachable, why
    forced = gate.SPEC.forced_expansion()
    assert forced["expansion"] == 1 and forced["arm"] == "FMA_V1"


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
