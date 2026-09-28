"""The host half of the two no-absorber stored-E Metal welds' certification.

THE PRODUCTS weld the no-absorber ``step_D`` (plain and conductive arms, one module
each) to the stored-E ``update_E`` and carry ``zero_metal_D`` between them, with the
in-seam electric deposit carried across the launch by
``deposit_repair.PLAIN_PATH`` — the SECOND repair, and these are the first Metal
products to declare it. The bytes are the gate's
(``parity/meep_gpu/gate_metal_no_pml_fused_electric_pairs.py``, on this Mac's MPS);
what is here is everything answerable without a device run and everything that would
be a silent widening if it drifted: the declarations, the transcription anchors, the
predicate in both directions — including the conductive family's LIFTED rescale
clause (a table-publishing scaled source admitted, the table-less fallback still
refused by name) — the two-arm partition, the absorb rows, and the installer
threading that hands the composed bracket THIS product's repair path rather than
the split-field default.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re
import sys
import textwrap

import numpy
import pytest

from . import deposit_repair
from .metal_kernels import launch as metal_launch

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "metal_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
GATE = PARITY / "gate_metal_no_pml_fused_electric_pairs.py"

FAMILIES = ("no_pml_fused_electric_pair", "no_pml_conductive_fused_electric_pair")


@pytest.fixture(scope="module", params=FAMILIES)
def product(request):
    return importlib.import_module(f"meep_gpu.metal_kernels.{request.param}")


@pytest.fixture(scope="module")
def matrix():
    if str(PARITY) not in sys.path:
        sys.path.insert(0, str(PARITY))
    module = importlib.import_module("metal_composition_matrix")
    module.prepare_environment()
    return module


def _fixture_for(product, matrix):
    pair = matrix.real_stored_e_no_pml(with_polarization=True)
    if "conductive" in product.FAMILY:
        pair = matrix.conductive(pair)
    return pair


class _Electric:
    field_type = "D"
    component = "Ez"
    is_integrated = True
    _point_ix = numpy.array([1])
    _point_iy = numpy.array([1])
    _point_iz = numpy.array([0])


class _ScaledElectric(_Electric):
    """The conductive corpus rows' own source shape: NOT integrated."""

    is_integrated = False


class _ElectricWithoutIndex:
    field_type = "D"
    component = "Ez"
    is_integrated = True
    _point_ix = None


class _Magnetic:
    field_type = "B"
    component = "Hz"
    is_integrated = True
    _point_ix = numpy.array([1])
    _point_iy = numpy.array([1])
    _point_iz = numpy.array([0])


def reasons_of(product, fields, pml, sources):
    """The predicate's answer WITH a residency declared.

    ``None`` is itself a refusal (coverage._residency_declaration_reasons: two
    sub-steps mirroring the same volume separately would trade stale bytes), so
    the admitting-direction tests here must declare one the way every plan does;
    the refusal-direction tests check named substrings and hold either way.
    """
    from .metal_kernels.device import Residency

    return list(getattr(product, f"{product.FAMILY}_coverage")(
        fields, pml, sources, Residency()).reasons)


# ---------------------------------------------------------------------------
# The declaration — the second repair, and every half of it
# ---------------------------------------------------------------------------

def test_the_product_declares_the_repair_and_which_one(product):
    """Flag, path, plan attribute and predicate threading travel together."""
    assert product.CARRIES_DEPOSIT_REPAIR is True
    assert product.REPAIR_PATHS == (deposit_repair.PLAIN_PATH,)
    plan_class = next(value for name, value in vars(product).items()
                      if inspect.isclass(value) and name.endswith("Plan"))
    assert plan_class.repair_paths == product.REPAIR_PATHS
    text = (PACKAGE_DIR / f"{product.FAMILY}.py").read_text(encoding="utf-8")
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in text
    assert "repair_paths=REPAIR_PATHS" in text


def test_the_declared_path_is_the_one_this_configuration_actually_runs(
        product, matrix):
    """Read from ``deposit_repair``'s side: the engine's own answer about which
    recurrence executes, against what the product says it carries."""
    fields, pml = _fixture_for(product, matrix)
    assert deposit_repair.repair_path_for(pml) == deposit_repair.PLAIN_PATH
    assert deposit_repair.repairable(fields, "D", pml,
                                     paths=product.REPAIR_PATHS) == (True, ())
    # ...and the DEFAULT declaration still refuses it, which is what makes this
    # product's declaration load-bearing rather than decorative.
    covered, why = deposit_repair.repairable(fields, "D", pml)
    assert not covered, why


def test_replaces_is_the_drivers_own_call_order(product):
    """``REPLACES`` names three call sites, in the order ``FdtdDriver.step`` makes
    them, read off the driver body by AST."""
    from . import driver as driver_module

    assert product.REPLACES == ("step_D", "zero_metal_D", "update_E")
    body = ast.parse(textwrap.dedent(
        inspect.getsource(driver_module.FdtdDriver.step)))
    calls = [node for node in ast.walk(body)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id in set(product.REPLACES)]
    order = [node.func.id for node in
             sorted(calls, key=lambda node: (node.lineno, node.col_offset))]
    seen = [name for index, name in enumerate(order) if name not in order[:index]]
    assert seen == list(product.REPLACES), seen


def test_the_curl_table_is_READ_from_the_arm_and_not_from_launch(product, matrix):
    """step_D's no-absorber sources are the B volumes: H is deliberately
    unallocated on this branch and ``launch.SUB_STEPS`` disagrees on purpose."""
    from .triton_kernels import launch as triton_launch

    assert product.CURL_SOURCES == ("Bx", "By", "Bz")
    assert product.CURL_SOURCES != tuple(triton_launch.SUB_STEPS["step_D"]["sources"])
    assert product.CURL_TARGETS == ("Dx", "Dy", "Dz")
    fields, _pml = _fixture_for(product, matrix)
    assert fields.Hx is None and fields.Bx is not None


# ---------------------------------------------------------------------------
# Transcription — the lift is the certified text, and the edits are data
# ---------------------------------------------------------------------------

def test_the_lift_edits_are_data_and_the_seam_line_is_the_first(product):
    old, new, _why = product.LIFT_EDITS[0]
    assert old == "    float source = d_in[idx];"
    assert "value{axis}" in new


def test_the_emitted_source_drops_the_certified_reload_and_names_the_register(
        product):
    """The one certified line that moves is gone from the fused text, and the
    register read stands in its place; the emitters' own anchor assertions have
    already checked every OTHER line against the certified output."""
    emit = getattr(product, f"{product.FAMILY}_source")
    for axis in (0, 1, 2):
        arguments = ((1, 0, 1), axis, 2)
        if "conductive" in product.FAMILY:
            source = emit(*arguments, True, (True, False, True))
        else:
            source = emit(*arguments, (True, False, True))
        assert "float source = d_in[idx];" not in source
        assert f"float source = value{axis};" in source
        assert "e_out[idx] = source * inv_e[idx];" in source


def test_the_signature_is_the_declared_binding_count(product):
    binding = re.compile(r"\[\[buffer\((\d+)\)\]\]")
    emit = getattr(product, f"{product.FAMILY}_source")
    if "conductive" in product.FAMILY:
        source = emit((0, 0, 0), 0, 8, True, (False, False, False))
    else:
        source = emit((0, 0, 0), 0, 8, (False, False, False))
    slots = sorted({int(number) for number in binding.findall(source)})
    assert len(slots) == product.BINDINGS_PER_COMPONENT
    assert slots == list(range(product.BINDINGS_PER_COMPONENT))


# ---------------------------------------------------------------------------
# The predicate, both directions
# ---------------------------------------------------------------------------

def test_an_undeclared_source_list_is_refused(product, matrix):
    fields, pml = _fixture_for(product, matrix)
    found = reasons_of(product, fields, pml, None)
    assert any("was not declared" in reason for reason in found), found[:6]


def test_an_electric_source_with_no_deposit_index_is_refused(product, matrix):
    fields, pml = _fixture_for(product, matrix)
    found = reasons_of(product, fields, pml, (_ElectricWithoutIndex(),))
    assert any("does not publish the index" in reason for reason in found), found[:6]


def test_a_magnetic_source_never_reaches_this_seam(product, matrix):
    fields, pml = _fixture_for(product, matrix)
    found = reasons_of(product, fields, pml, (_Magnetic(),))
    assert not any("is electric" in reason for reason in found), found[:6]


def test_an_integrated_electric_deposit_is_admitted(product, matrix):
    """The flag's whole purpose, in the ADMITTING direction."""
    fields, pml = _fixture_for(product, matrix)
    assert reasons_of(product, fields, pml, (_Electric(),)) == []


def test_an_active_absorber_is_refused_by_both_halves(product, matrix):
    fields, pml = matrix.cart(pml=2)
    found = reasons_of(product, fields, pml, (_Electric(),))
    assert found, "an active layer must be refused"


def test_the_two_curl_arms_PARTITION_the_space_rather_than_overlap(matrix):
    """Exactly one of the two families admits any no-absorber stored-E run: the
    conductive one requires a sigma on a curl target BY NAME and the lossless one
    refuses every sigma BY NAME — the arms' own partition, restated over the welds."""
    plain = importlib.import_module(
        "meep_gpu.metal_kernels.no_pml_fused_electric_pair")
    conductive = importlib.import_module(
        "meep_gpu.metal_kernels.no_pml_conductive_fused_electric_pair")
    for wrap, expected in ((lambda pair: pair, "plain"),
                           (matrix.conductive, "conductive")):
        fields, pml = wrap(matrix.real_stored_e_no_pml(with_polarization=True))
        admits = {
            "plain": not reasons_of(plain, fields, pml, (_Electric(),)),
            "conductive": not reasons_of(conductive, fields, pml, (_Electric(),)),
        }
        assert sum(admits.values()) == 1, admits
        assert admits[expected], admits


# ---------------------------------------------------------------------------
# THE LIFTED REFUSAL — the conductive cell's two corpus rows, now admitted
# ---------------------------------------------------------------------------

class _TableLessScaledElectric:
    """A scaled electric source publishing NO deposit table — the driver's dense
    fallback, and the one shape the lifted clause still refuses."""

    field_type = "D"
    component = "Ez"
    is_integrated = False


def test_a_scaled_TABLE_PUBLISHING_source_on_a_conductive_run_is_admitted(matrix):
    """The lift, pinned in the admitting direction: the corpus rows' own source
    shape — non-integrated, ``_point_ix`` published — no longer trips the
    conductive-rescale clause, because the driver replays the condinv rescale
    sparsely at the published deposit cells."""
    conductive = importlib.import_module(
        "meep_gpu.metal_kernels.no_pml_conductive_fused_electric_pair")
    fields, pml = matrix.conductive(
        matrix.real_stored_e_no_pml(with_polarization=True))
    assert reasons_of(conductive, fields, pml, (_ScaledElectric(),)) == []


def test_a_scaled_TABLE_LESS_source_on_a_conductive_run_is_still_refused(matrix):
    """The surviving clause, pinned by name: a source with no ``_point_ix``
    attribute routes through the driver's whole-volume fallback, which still
    canonicalises -0.0 at cells no deposit closure can name."""
    conductive = importlib.import_module(
        "meep_gpu.metal_kernels.no_pml_conductive_fused_electric_pair")
    fields, pml = matrix.conductive(
        matrix.real_stored_e_no_pml(with_polarization=True))
    found = reasons_of(conductive, fields, pml, (_TableLessScaledElectric(),))
    assert any("publishes NO deposit table" in reason for reason in found), found[:6]


def test_the_clause_is_conditional_on_the_conductivity_and_not_on_the_seam(matrix):
    """The SAME source on a LOSSLESS run is admitted — the discriminating half, and
    the whole reason the plain cell serves its one corpus row
    (examples:material-dispersion.py carries a non-integrated source too)."""
    plain = importlib.import_module(
        "meep_gpu.metal_kernels.no_pml_fused_electric_pair")
    fields, pml = matrix.real_stored_e_no_pml(with_polarization=True)
    assert reasons_of(plain, fields, pml, (_ScaledElectric(),)) == []


def test_the_clause_is_quiet_except_for_the_table_less_fallback(matrix):
    conductive = importlib.import_module(
        "meep_gpu.metal_kernels.no_pml_conductive_fused_electric_pair")
    fields, pml = matrix.conductive(
        matrix.real_stored_e_no_pml(with_polarization=True))
    assert conductive._scaled_conductive_injection_reasons(
        fields, (_Electric(),)) == ()
    assert conductive._scaled_conductive_injection_reasons(
        fields, (_ScaledElectric(),)) == ()
    assert conductive._scaled_conductive_injection_reasons(
        fields, (_TableLessScaledElectric(),)) != ()


def test_the_device_gate_MEASURES_the_lift_rather_than_asserting_it():
    source = GATE.read_text(encoding="utf-8")
    assert "def leg_lifted_refusal(" in source
    assert "whole_volume_control" in source
    assert "_inject_with_whole_volume_rescale" in source
    assert "pm_zero_lattice" in source
    # The armed control MUST diverge and the lifted walk MUST be identical — both
    # scored, neither assumed.
    assert "control_diverged" in source
    assert "tableless_source_still_refused_by_name" in source


def test_the_gates_driver_transcription_is_the_LIVE_sparse_injection():
    """The gate's ``_inject_like_the_driver`` must transcribe the driver that
    ships — the sparse per-deposit-cell replay with the dense fallback — not the
    retired whole-volume passes, which survive only as the armed control."""
    source = GATE.read_text(encoding="utf-8")
    live = source.split("def _inject_like_the_driver", 1)[1]
    live = live.split("def _inject_with_whole_volume_rescale", 1)[0]
    assert "_deposit_index" in live
    assert "sparse" in live and "dense" in live
    assert 'hasattr(source, "_point_ix")' in live


def test_the_corpus_rows_sources_really_are_non_integrated():
    """The lift serves two NAMED corpus rows through the sparse replay, so the
    fact it rests on — both declare ``mp.Source`` with ``is_integrated`` left at
    MEEP's False default, publishing deposit tables like every in-tree source —
    is pinned against the corpus text when the corpus is present, and skipped
    (never silently passed) when it is not."""
    import os

    root = os.environ.get("MEEP_GPU_CORPUS_ROOT")
    if not root:
        # A sanctioned resource skip, never a bare one: under the no-silent-skip
        # policy (conftest.py) a bare skip is a FAILURE, so a suite run without the
        # corpus checked out went red here on 2026-09-03 instead of reporting the
        # absent artifact by name.
        from conftest import requires_resource_skip  # noqa: PLC0415
        requires_resource_skip("meep_corpus",
                               "no corpus checked out (MEEP_GPU_CORPUS_ROOT unset)")
    for relative in ("examples/absorber-1d.py", "tests/test_absorber_1d.py"):
        path = pathlib.Path(root) / relative
        if not path.is_file():
            pytest.skip(f"{relative} not in this corpus checkout")
        text = path.read_text(encoding="utf-8")
        assert "is_integrated" not in text, (
            f"{relative} now sets is_integrated; the lifted clause's corpus "
            f"arithmetic must be re-measured")


# ---------------------------------------------------------------------------
# Wiring — the absorb rows, the installer threading, and the composed bracket
# ---------------------------------------------------------------------------

def test_the_absorb_rows_name_the_two_measured_arms():
    assert metal_launch.FUSED_PAIR_ARMS["no_pml_fused_electric_pair"] == (
        "no-PML curl", "no-PML stored E")
    assert metal_launch.FUSED_PAIR_ARMS[
        "no_pml_conductive_fused_electric_pair"] == (
        "conductive no-PML curl", "no-PML stored E")
    assert metal_launch.FUSED_PAIR_SEAMS["step_D"] == ("update_E", "D")


def test_the_arms_the_pairs_absorb_really_do_win_their_slots(matrix):
    """The rows above, MEASURED through ``plan_step`` with ``fuse=False``."""
    from .metal_kernels import device

    for wrap, curl_label in ((lambda pair: pair, "no-PML curl"),
                             (matrix.conductive, "conductive no-PML curl")):
        fields, pml = wrap(matrix.real_stored_e_no_pml(with_polarization=True))
        plan = metal_launch.plan_step(fields, pml, residency=device.Residency(),
                                      sources=(), fuse=False)
        assert plan.selected.get("step_D") == curl_label, plan.selected
        assert plan.selected.get("update_E") == "no-PML stored E", plan.selected


def test_the_installer_can_be_told_which_repair_a_product_carries():
    """``launch._install_fused_pair`` takes the declaration and the seam loop reads
    it off the BUILT PLAN — so the day either family is routed differently it
    cannot silently get the split-field default and be refused on every row."""
    signature = inspect.signature(metal_launch._install_fused_pair)
    assert signature.parameters["repair_paths"].default == (
        deposit_repair.SPLIT_FIELD_PATH,)
    tree = ast.parse((PACKAGE_DIR / "launch.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == "LeadingRepairPlan"]
    assert len(calls) == 1 and len(calls[0].args) == 6, ast.dump(calls[0])
    text = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    assert 'getattr(pair, "repair_paths"' in text


@pytest.mark.parametrize("family_name,conductive", [
    ("no_pml_fused_electric_pair", False),
    ("no_pml_conductive_fused_electric_pair", True),
])
def test_the_composed_bracket_takes_the_PLAIN_path(matrix, family_name, conductive):
    """WHICH path the composed plan takes, asked of the shipped composer.

    ``plan_step(..., fuse=True)`` with a real in-seam electric source must land the
    two repair plans in the two slots AND the leading plan must carry THIS
    product's declaration — the plain path, not the split-field default the
    installer would fall back to for a plan with no declaration.
    """
    from .metal_kernels import device
    from .sources import ContinuousEnvelope, VolumeSource

    module = importlib.import_module(f"meep_gpu.metal_kernels.{family_name}")
    pair = matrix.real_stored_e_no_pml(with_polarization=True)
    fields, pml = matrix.conductive(pair) if conductive else pair
    source = VolumeSource(grid=fields.grid, component="Ez", center=(0, 0, 0),
                          size=(0, 0, 0),
                          envelope=ContinuousEnvelope(frequency=1.0,
                                                      is_integrated=conductive))
    plan = metal_launch.plan_step(fields, pml, residency=device.Residency(),
                                  sources=(source,), fuse=True)
    reasons = plan.reasons.get(f"fused_pair_{family_name}", ())
    assert not reasons, reasons
    leading = plan.plans["step_D"]
    trailing = plan.plans["update_E"]
    assert isinstance(leading, deposit_repair.LeadingRepairPlan)
    assert isinstance(trailing, deposit_repair.TrailingRepairPlan)
    assert tuple(leading.repair_paths) == (deposit_repair.PLAIN_PATH,)
    assert leading.pair == "D"
    for slot in ("step_D", "update_E"):
        assert (metal_launch.declaring_plan(plan.plans[slot]).replaces_sub_steps
                == module.REPLACES)


def test_the_scaled_conductive_configuration_now_COMPOSES_the_bracket(matrix):
    """The lift, composed: the corpus rows' own source shape — non-integrated on a
    conductive run — lands the two repair plans in the two slots where it used to
    fall back with the priced refusal."""
    from .metal_kernels import device
    from .sources import ContinuousEnvelope, VolumeSource

    fields, pml = matrix.conductive(
        matrix.real_stored_e_no_pml(with_polarization=True))
    source = VolumeSource(grid=fields.grid, component="Ez", center=(0, 0, 0),
                          size=(0, 0, 0),
                          envelope=ContinuousEnvelope(frequency=1.0,
                                                      is_integrated=False))
    plan = metal_launch.plan_step(fields, pml, residency=device.Residency(),
                                  sources=(source,), fuse=True)
    reasons = plan.reasons.get("fused_pair_no_pml_conductive_fused_electric_pair",
                               ())
    assert not reasons, reasons
    leading = plan.plans["step_D"]
    assert isinstance(leading, deposit_repair.LeadingRepairPlan)
    assert isinstance(plan.plans["update_E"], deposit_repair.TrailingRepairPlan)
    assert tuple(leading.repair_paths) == (deposit_repair.PLAIN_PATH,)


def test_the_TABLE_LESS_configuration_falls_back_to_the_separate_arms(matrix):
    """The surviving refusal composed: a scaled source with no deposit table
    leaves the seam on the two certified arms, with the reason named."""
    from .metal_kernels import device

    fields, pml = matrix.conductive(
        matrix.real_stored_e_no_pml(with_polarization=True))
    plan = metal_launch.plan_step(fields, pml, residency=device.Residency(),
                                  sources=(_TableLessScaledElectric(),), fuse=True)
    reasons = plan.reasons.get("fused_pair_no_pml_conductive_fused_electric_pair",
                               ())
    assert any("publishes NO deposit table" in reason for reason in reasons), reasons
    assert plan.selected.get("step_D") == "conductive no-PML curl"
    assert plan.selected.get("update_E") == "no-PML stored E"


# ---------------------------------------------------------------------------
# Registration and records
# ---------------------------------------------------------------------------

def test_the_products_register_unwired_on_step_D(product):
    from .metal_kernels import arms

    spec = next(spec for spec in arms.registered("step_D")
                if spec.family == product.FAMILY)
    assert spec.wired is False
    assert spec.is_weld
    assert spec.replaces == product.REPLACES


def test_the_registry_names_both_modules():
    from .metal_kernels import registry

    for name in FAMILIES:
        assert name in registry.FAMILY_MODULES


def test_the_gate_exists_and_names_both_products():
    assert GATE.exists()
    source = GATE.read_text(encoding="utf-8")
    for name in FAMILIES:
        assert name in source
    assert "run_current_measurement" in source


def test_the_board_prices_both_products():
    text = (PARITY / "build_fusion_matrix.py").read_text(encoding="utf-8")
    for name in FAMILIES:
        assert name in text, name


def test_the_weld_ledger_carries_the_gate():
    import json

    ledger = json.loads((PACKAGE_DIR / "fingerprints.json").read_text(
        encoding="utf-8"))
    entry = ledger.get("metal_no_pml_fused_electric_pairs_device_gate")
    assert entry is not None, "the shared device gate has no weld entry"
    pins = set(entry["source_sha256"])
    for name in FAMILIES:
        assert f"meep_gpu/metal_kernels/{name}.py" in pins
    assert "meep_gpu/metal_kernels/launch.py" in pins
    assert "parity/meep_gpu/gate_metal_no_pml_fused_electric_pairs.py" in pins
