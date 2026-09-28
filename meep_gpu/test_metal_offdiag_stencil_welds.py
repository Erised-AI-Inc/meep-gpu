"""The host half of the four Metal OFF-DIAGONAL STENCIL welds' certification.

THE PRODUCTS close the last structural bucket on the Metal board: the four cells
scored ``STRUCTURALLY UNFUSABLE ON ANY BACKEND — the off-diagonal constitutive arm
is a STENCIL over the curl arm's own in-place D output``. They are the first
products on either backend to answer that verdict, and they answer it by not
writing D in place: the curl half writes launch-local SCRATCH, the constitutive
half re-derives every foreign tap from PRE-LAUNCH state through the same inline
``step_cell``, and the launcher rotates the buffers after the launch returns.

The bytes are the gate's (``parity/meep_gpu/gate_metal_offdiag_stencil_welds.py``,
on this Mac's MPS). What is here is everything answerable without a device run and
everything that would be a silent widening if it drifted: the declarations, the
binding counts against the EMITTED text, the pack cross-checks, the forced
CARRIES flag with the clause that forces it MEASURED rather than cited, the
predicate in both directions, the absorb rows, the registration shape, and the
two properties the weld's whole design rests on — that the lift is a LIFT, and
that no displacement read survives the redirect.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys

import numpy as np
import pytest

from . import deposit_repair
from .metal_kernels import launch as metal_launch
from .metal_kernels import offdiag_weld_common as weld

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "metal_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
GATE = PARITY / "gate_metal_offdiag_stencil_welds.py"
CLOSED_FORM = (PARITY / "results"
               / "metal_scratch_weld_closed_form_2026-09-01T2" / "probe.json")

FAMILIES = ("offdiag_fused_electric_pair",
            "folded_offdiag_fused_electric_pair",
            "complex_no_pml_offdiag_fused_electric_pair",
            "folded_complex_offdiag_fused_electric_pair")

#: family -> (board cell, rows in the cell, rows CLEARING the source seam). The
#: third number is the demand a fused product can ever reach on this cell, and
#: the gap between it and the second is exactly the electric-source refusal the
#: forced CARRIES flag names. Cited from the 2026-09-01 board's
#: ``gaps_ranked_by_demand`` and pinned here so a docstring and a gate cannot
#: drift apart.
CELLS = {
    "offdiag_fused_electric_pair": ("D_to_E (PML, offdiag)", 16, 8),
    "folded_offdiag_fused_electric_pair":
        ("D_to_E (folded, folded offdiag)", 19, 9),
    "complex_no_pml_offdiag_fused_electric_pair":
        ("D_to_E (complex no-PML curl, complex no-PML off-diagonal)", 2, 1),
    "folded_complex_offdiag_fused_electric_pair":
        ("D_to_E (folded complex, complex folded off-diagonal PML E)", 3, 1),
}

#: family -> the two arm labels its predicate is literally built out of.
ABSORB_ROWS = {
    "offdiag_fused_electric_pair": ("PML", "offdiag"),
    "folded_offdiag_fused_electric_pair": ("folded", "folded offdiag"),
    "complex_no_pml_offdiag_fused_electric_pair":
        ("complex no-PML curl", "complex no-PML off-diagonal"),
    "folded_complex_offdiag_fused_electric_pair":
        ("folded complex", "complex folded off-diagonal PML E"),
}

FOLDED_FAMILIES = ("folded_offdiag_fused_electric_pair",
                   "folded_complex_offdiag_fused_electric_pair")

_BINDING = re.compile(r"\[\[buffer\((\d+)\)\]\]")
_POINTER = re.compile(r"device\s+(?:const\s+)?[\w:]+\s*\*\s*\w+\s*\[\[buffer")


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


def _emit(product):
    """One representative shipped source per family."""
    name = product.FAMILY
    if name == "offdiag_fused_electric_pair":
        return product.offdiag_fused_electric_pair_source(
            (1, 0, 0, 0, 0, 0), (1, 1, 1), (1, 1, 1))
    if name == "folded_offdiag_fused_electric_pair":
        return product.folded_offdiag_fused_electric_pair_source(
            (1, 0, 0, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 1, 0), {1: 1}, {1: 1})
    if name == "complex_no_pml_offdiag_fused_electric_pair":
        return product.complex_no_pml_offdiag_fused_electric_pair_source(
            (1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0), "FMA_V1")
    return product.folded_complex_offdiag_fused_electric_pair_source(
        (0, 0, 1, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0),
        {1: 1}, {1: 1}, "FMA_V1")


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_the_slot_is_the_seams_first_and_replaces_names_the_fold(product):
    assert product.SLOT == "step_D"
    assert product.REPLACES[0] == "step_D"
    assert product.REPLACES[-1] == "update_E"
    if product.FAMILY in FOLDED_FAMILIES:
        # A FOLDED SEAM RUNS BOTH GHOST FILLS between the halves, so a folded
        # product that named three passes would be claiming to replace a driver
        # pass it does not carry.
        assert product.REPLACES == ("step_D", "fill_D", "zero_metal_D",
                                    "fill_folded_far_ghosts_D", "update_E")
    else:
        assert product.REPLACES == ("step_D", "zero_metal_D", "update_E")


def test_the_carries_flag_is_false_and_the_clause_that_forces_it_is_measured(
        product, matrix):
    """CARRIES_DEPOSIT_REPAIR cannot be True here, and this MEASURES why.

    The campaign rule is that the flag is a per-cell measurement rather than a
    choice. On these four cells the measurement is not "does the seam hold a
    deposit" — several of their rows do — but "can the repair carry one", and
    ``deposit_repair.repairable`` answers NO for an off-diagonal chi1inv by name.
    The control is the same fixture with the off-diagonal rows absent, which must
    be repairable: without it a clause that refused everything would pass.
    """
    assert product.CARRIES_DEPOSIT_REPAIR is False

    fields, pml = matrix.cart(rows={"Ex": ("Ey",)})
    ok, reasons = deposit_repair.repairable(fields, "D", pml)
    assert not ok, "an off-diagonal chi1inv must not be repairable"
    assert any("diagonal" in reason for reason in reasons), reasons

    plain, plain_pml = matrix.cart()
    control_ok, control_reasons = deposit_repair.repairable(plain, "D", plain_pml)
    assert control_ok, (
        f"the control must be repairable, or the clause above is vacuous: "
        f"{control_reasons}")


def test_the_flag_reaches_the_seam_clause_by_name(product):
    text = (PACKAGE_DIR / f"{product.FAMILY}.py").read_text(encoding="utf-8")
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in text
    assert "deposit_repair.repairable REFUSES an off-diagonal chi1inv" in text
    # No family declares a repair path: none of them carries a repair at all, and
    # a REPAIR_PATHS declaration here would be a claim about machinery that this
    # seam refuses rather than uses.
    assert "REPAIR_PATHS" not in text


def test_the_cell_demand_the_gate_records_is_this_suites(product):
    """The gate's corpus table and this suite must name the same numbers."""
    cell, rows, clearing = CELLS[product.FAMILY]
    text = GATE.read_text(encoding="utf-8")
    assert f'"cell": "{cell}"' in text, cell
    assert f'"cell_rows": {rows}' in text
    assert f'"seam_instances_reachable": {clearing}' in text
    assert clearing <= rows


def test_the_absorb_rows_name_the_two_measured_arms(product):
    assert (metal_launch.FUSED_PAIR_ARMS[product.FAMILY]
            == ABSORB_ROWS[product.FAMILY])


def test_the_products_register_unwired_on_their_slot(product):
    from .metal_kernels import arms

    spec = next(spec for spec in arms.registered(product.SLOT)
                if spec.family == product.FAMILY)
    assert spec.wired is False
    assert spec.is_weld
    assert spec.replaces == product.REPLACES


def test_the_registry_names_every_one_of_the_four():
    from .metal_kernels import registry

    for family in FAMILIES:
        assert family in registry.FAMILY_MODULES


# ---------------------------------------------------------------------------
# The binding counts, against the emitted text
# ---------------------------------------------------------------------------

def test_the_shipped_signature_counts_are_the_emitted_ones(product):
    source = _emit(product)
    slots = sorted({int(number) for number in _BINDING.findall(source)})
    pointers = len(_POINTER.findall(source))
    assert len(slots) == product.PACKED_BINDINGS, (len(slots), pointers)
    assert slots == list(range(len(slots)))
    assert pointers == product.PACKED_POINTERS
    assert product.PACKED_BINDINGS == product.PACKED_POINTERS + 1
    from .metal_kernels.device import MAX_BUFFER_BINDINGS

    assert product.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS


def test_the_pack_savings_are_the_packs_own_arithmetic(product):
    """Packing N vectors into one buffer saves exactly N - 1 pointers."""
    if not hasattr(product, "UNPACKED_POINTERS"):
        # The complex no-PML family packs NOTHING: it has no per-axis coefficient
        # vector to pack and fits nine under the ceiling without one. That is a
        # measured per-family decision, not an omission.
        assert product.FAMILY == "complex_no_pml_offdiag_fused_electric_pair"
        return
    from .metal_kernels.offdiag_fused_electric_pair import (
        PACKED_VECTORS, PACKED_VOLUMES)

    saved = product.UNPACKED_POINTERS - product.PACKED_POINTERS
    assert saved == (len(PACKED_VECTORS) - 1) + (len(PACKED_VOLUMES) - 1)
    assert product.CPACK_ONLY_BINDINGS == (
        product.UNPACKED_POINTERS - (len(PACKED_VECTORS) - 1) + 1)
    from .metal_kernels.device import MAX_BUFFER_BINDINGS

    assert product.CPACK_ONLY_BINDINGS > MAX_BUFFER_BINDINGS, (
        "the one-pack shape must be OVER the ceiling, or the second pack is a "
        "preference rather than the necessity this family claims")


def test_the_pack_prologue_recreates_every_member_under_its_own_name(product):
    from .metal_kernels.offdiag_fused_electric_pair import (
        PACKED_VECTORS, PACKED_VOLUMES)

    source = _emit(product)
    if not hasattr(product, "UNPACKED_POINTERS"):
        # THE NO-PACK FAMILY IS ASSERTED, NOT SKIPPED: it must bind every member
        # SEPARATELY, which is the positive form of "this family packs nothing".
        assert "cpack" not in source and "mpack" not in source
        for label in PACKED_VOLUMES:
            assert f"device const float*  {label}" in source, label
        return
    for label in PACKED_VECTORS:
        assert f"device const float* {label} = cpack + prm.off_{label};" in source
    for label in PACKED_VOLUMES:
        assert f"device const float* {label} = mpack + prm.off_{label};" in source


def test_the_params_record_matches_the_kernel_struct(product):
    """Every ``Params`` member the kernel declares has a host field, and back."""
    source = _emit(product)
    body = source.split("struct Params {", 1)[1].split("};", 1)[0]
    declared = [name.rstrip(";")
                for line in body.splitlines()
                for name in line.replace(";", " ; ").split()
                if name not in ("uint", "int", "float", "float2", ";")]
    record = product.params_record_dtype()
    assert sorted(declared) == sorted(record.names), (
        sorted(set(declared) ^ set(record.names)))
    assert record.itemsize == product.PARAMS_ITEMSIZE
    assert record.itemsize % 4 == 0


# ---------------------------------------------------------------------------
# The two properties the whole design rests on
# ---------------------------------------------------------------------------

def test_no_displacement_read_survives_the_redirect(product):
    """After the substitution, ``gN[`` must not appear in the kernel body.

    In the fused signature those three pointers are the MAGNETIC field. One
    surviving read takes H for D, which is a smooth, plausible, entirely wrong
    answer rather than a crash — the single silent failure this construction can
    have, and the builder raises on it rather than emitting one.
    """
    source = _emit(product)
    entry = source.split("kernel void", 1)[1]
    for target in range(3):
        assert f"g{target}[" not in entry, target


def test_the_lift_is_a_lift(product):
    """``step_cell``'s body IS the certified curl emitter's own tail.

    Character for character, below the decode anchor and with the stores
    rewritten — which is what makes "the fused arithmetic is the certified
    arithmetic" a property a reader can check rather than a claim.
    """
    source = _emit(product)
    helper = source.split("static inline step_cell_result step_cell(", 1)[1]
    helper = helper.split("\n}\n", 1)[0]
    # Every certified curl line below the anchor survives, modulo the four-space
    # re-indent the function body takes and the stores the lift rewrote.
    assert "float curl0 = " in helper or "float2 curl0 = " in helper
    assert "out.v0 = v0;" in helper
    for name in ("d0s", "n0s", "dfin0"):
        assert name not in helper, (
            f"{name} is the KERNEL's, not the certified curl's; the helper must "
            f"read only pre-launch state")


def test_the_scratch_weld_plan_refuses_an_aliased_pair():
    """The plan's own guard: read and write resolving to one tensor is REFUSED.

    Not a comment — ``_resolve`` raises. This is the guard the design's "nothing
    written is ever read" rests on, and the gate arms it on device too.
    """
    class _Residency:
        device = "mps"

        def tensor_for_host(self, host):
            return "one-and-only-tensor"

    class _Fields:
        Dx = object()

    plan = weld.ScratchWeldPairPlan(
        "probe", _Residency(), _Fields(), {"Dx": object()}, {"off": lambda *a: None},
        ("Dx",), (), ("Dx",), (2, 2, 2), (0, 0, 0), (0, 0, 0),
        (1, 0, 0, 0, 0, 0), ("step_D",))
    with pytest.raises(RuntimeError, match="ONE tensor"):
        plan.run()


def test_the_rotation_order_must_name_the_twins():
    class _Residency:
        device = "mps"

    with pytest.raises(ValueError, match="rotation order"):
        weld.ScratchWeldPairPlan(
            "probe", _Residency(), object(), {"Dx": object()}, {},
            ("Dy",), (), ("Dx",), (2, 2, 2), (0, 0, 0), (0, 0, 0),
            (1, 0, 0, 0, 0, 0), ("step_D",))


# ---------------------------------------------------------------------------
# The closed form's parity spelling — the audit finding, pinned
# ---------------------------------------------------------------------------

_SPELLING_VALUES = (0.0, -0.0, 1.0, -1.0, 3.4e-38, -3.4e-38, 1e-45, -1e-45,
                    7.5, -7.5)


def _words(value):
    return np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=np.uint32)


@pytest.mark.parametrize("weight", (1, -1))
def test_the_complex_parity_spelling_is_the_array_paths(weight):
    """``c_mul((w, +0), z)`` IS ``w * z``; a copy / bare negation is NOT.

    ``stepping._write_mirror_ghost`` spells a parity ``phase * plane`` — a FULL
    complex multiply — and the emitted complex form expands that multiply's own
    arithmetic. A plain copy at +1 and a bare ``-z`` at -1 are the REAL families'
    (correct, measured) spelling and differ from it in TEN of a hundred
    engineered signed-zero pairs, which is why the complex families do not elide
    the +1 product. The gate reaches one of those ten on a real fixture.
    """
    module_differing = 0
    bare_differing = 0
    for left in _SPELLING_VALUES:
        for right in _SPELLING_VALUES:
            value = np.array([np.complex64(complex(left, right))],
                             dtype=np.complex64)
            reference = np.complex64(weight) * value
            w = np.float32(weight)
            a, b = np.float32(left), np.float32(right)
            module = np.array([np.complex64(complex(
                np.float32(w * a - np.float32(0.0) * b),
                np.float32(w * b + np.float32(0.0) * a)))], dtype=np.complex64)
            bare = value.copy() if weight == 1 else -value
            module_differing += int((_words(module) != _words(reference)).any())
            bare_differing += int((_words(bare) != _words(reference)).any())
    assert module_differing == 0
    assert bare_differing == 10, bare_differing


def test_the_real_parity_spelling_is_a_copy_and_a_negation():
    """On float32 the real families' spelling IS the array path's, 0 of 10."""
    for weight in (1, -1):
        differing = 0
        for value in _SPELLING_VALUES:
            array = np.array([np.float32(value)], dtype=np.float32)
            reference = np.float32(weight) * array
            spelled = array.copy() if weight == 1 else -array
            differing += int((_words(spelled) != _words(reference)).any())
        assert differing == 0, weight


def test_the_emitted_parity_lines_are_one_per_axis(product):
    """A doubly-unowned corner carries NESTED parities, not one product.

    ``_fill_symmetry_ghost_cells`` writes plane after plane in X, Y, Z order, so a
    corner unowned on two planes carries ``p_y * (p_x * raw)``. On complex storage
    the two spellings are measurably different (39 of 400 engineered combinations
    in the closed-form probe's word-level leg), so the per-axis emission is
    load-bearing rather than stylistic.
    """
    if product.FAMILY not in FOLDED_FAMILIES:
        # AN UNFOLDED FAMILY EMITS NO PARITY LINE AT ALL, and that is asserted
        # rather than skipped: a parity in an unfolded kernel would be a redirect
        # to a ghost row the grid does not have.
        source = _emit(product)
        assert "near_x" not in source and "near_y" not in source
        assert "was_far" not in source
        return
    if product.FAMILY == "folded_offdiag_fused_electric_pair":
        source = product.folded_offdiag_fused_electric_pair_source(
            (1, 0, 0, 0, 0, 1), (3, 3, 0), (0, 0, 0), (1, 0, 0),
            {0: -1, 1: 1}, {0: -1, 1: 1})
        assert source.count("near_x ? -value : value") >= 1
    else:
        source = product.folded_complex_offdiag_fused_electric_pair_source(
            (1, 0, 0, 0, 0, 1), (3, 3, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0),
            {0: -1, 1: 1}, {0: -1, 1: 1}, "FMA_V1")
        assert source.count("c_mul(float2(-1.0f, 0.0f), value)") >= 1
        assert source.count("c_mul(float2(1.0f, 0.0f), value)") >= 1
    # ONE LINE PER AXIS: a corner emits two guarded applications, never one
    # combined product.
    body = source.split("static inline", 1)[1]
    assert "near_x" in body and "near_y" in body


# ---------------------------------------------------------------------------
# The predicate, in both directions
# ---------------------------------------------------------------------------

def _fixture(matrix, family):
    if family == "offdiag_fused_electric_pair":
        return matrix.cart(rows={"Ex": ("Ey",)})
    if family == "folded_offdiag_fused_electric_pair":
        return matrix.folded(rows={"Ex": ("Ey",)})
    if family == "complex_no_pml_offdiag_fused_electric_pair":
        return matrix.complex_lossless_offdiag_no_pml()
    return matrix.folded(complex_storage=True, rows={"Ey": ["Ez"]})


def _coverage(product, fields, pml, sources):
    name = product.FAMILY
    function = getattr(product, f"metal_{name}_coverage")
    if name.startswith("complex_no_pml") or name.startswith("folded_complex"):
        return function(fields, pml, sources, None, None)
    return function(fields, pml, sources, None)


def test_an_undeclared_source_set_is_a_refusal(product, matrix):
    """Ignorance is never an empty set."""
    fields, pml = _fixture(matrix, product.FAMILY)
    verdict = _coverage(product, fields, pml, None)
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


def test_the_fold_is_refused_by_name_in_both_directions(product, matrix):
    """A folded family refuses an unfolded grid and the reverse, BY NAME.

    The two products would otherwise overlap on the same cell, and the arm
    table's disjointness sweep is right to complain about that.
    """
    folded_fields, folded_pml = matrix.folded(rows={"Ex": ("Ey",)})
    plain_fields, plain_pml = matrix.cart(rows={"Ex": ("Ey",)})
    if product.FAMILY == "offdiag_fused_electric_pair":
        verdict = _coverage(product, folded_fields, folded_pml, ())
        assert not verdict.covered
        assert any("is folded" in reason for reason in verdict.reasons)
    if product.FAMILY == "folded_offdiag_fused_electric_pair":
        verdict = _coverage(product, plain_fields, plain_pml, ())
        assert not verdict.covered
        assert any("no axis is folded" in reason for reason in verdict.reasons)


def test_a_polarization_is_refused_by_name(product, matrix):
    fields, pml = _fixture(matrix, product.FAMILY)
    fields, pml = matrix.dispersive((fields, pml))
    verdict = _coverage(product, fields, pml, ())
    assert not verdict.covered
    assert any("susceptibility" in reason or "polarization" in reason.lower()
               for reason in verdict.reasons), verdict.reasons


# ---------------------------------------------------------------------------
# The closed form's evidence artifact
# ---------------------------------------------------------------------------

def test_the_closed_form_artifact_is_the_one_the_modules_cite():
    """Every module cites the RE-CUT artifact, not the first cut it supersedes."""
    import json

    assert CLOSED_FORM.exists(), CLOSED_FORM
    record = json.loads(CLOSED_FORM.read_text())
    assert record["probe"] == "metal_scratch_weld_closed_form"
    assert "supersedes" in record
    assert record["parity_spellings"]["complex_weight-1"]["module"] == 0
    assert record["parity_spellings"]["complex_weight-1"]["bare_sign"] == 10
    for case in record["cases"]:
        assert not case["step1_differing"], case["case"]
        assert not case["step2_differing"], case["case"]
    for family in FAMILIES:
        text = (PACKAGE_DIR / f"{family}.py").read_text(encoding="utf-8")
        if "metal_scratch_weld_closed_form" in text:
            assert "2026-09-01T2" in text, family
